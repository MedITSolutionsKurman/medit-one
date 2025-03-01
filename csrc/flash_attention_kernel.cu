#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

// Constants for optimizing memory access patterns
constexpr int BLOCK_SIZE_M = 64;  // Number of rows in block
constexpr int BLOCK_SIZE_N = 64;  // Number of columns in block
constexpr int BLOCK_SIZE_K = 32;  // Number of elements in inner dimension

//------------------------------------------------------------------------------------------------
// Flash Attention Kernel
//------------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale_factor,
    const bool causal) {
    
    // Shared memory for block-level operations
    __shared__ scalar_t K_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ scalar_t V_shared[BLOCK_SIZE_N][BLOCK_SIZE_K];
    
    // Block indices
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    
    // Thread indices
    const int thread_row = threadIdx.x;
    const int thread_col = threadIdx.y;
    
    // Global batch and head indices
    const int batch_idx = blockIdx.z / num_heads;
    const int head_idx = blockIdx.z % num_heads;
    
    // Global position offsets
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    const int v_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    
    // Output accumulator registers - using registers for maximum speed
    scalar_t acc[BLOCK_SIZE_M] = {0.0f};
    scalar_t normalizer[BLOCK_SIZE_M] = {0.0f};
    
    // Calculate attention scores and apply attention
    for (int k_block = 0; k_block < (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++k_block) {
        // Skip blocks in upper triangular region if causal
        if (causal && (block_row * BLOCK_SIZE_M) < ((k_block + 1) * BLOCK_SIZE_N)) {
            continue;
        }
        
        // Collaborative loading of K and V tiles into shared memory
        if (k_block * BLOCK_SIZE_N + thread_col < seq_len) {
            for (int i = 0; i < BLOCK_SIZE_K; i += blockDim.y) {
                if (thread_row + i < head_dim) {
                    K_shared[thread_row + i][thread_col] = 
                        k[k_offset + (k_block * BLOCK_SIZE_N + thread_col) * head_dim + thread_row + i] * scale_factor;
                }
            }
            
            for (int i = 0; i < BLOCK_SIZE_K; i += blockDim.y) {
                if (thread_row + i < head_dim) {
                    V_shared[thread_col][thread_row + i] = 
                        v[v_offset + (k_block * BLOCK_SIZE_N + thread_col) * head_dim + thread_row + i];
                }
            }
        } else {
            // Zero-pad if exceeding sequence length
            for (int i = 0; i < BLOCK_SIZE_K; i += blockDim.y) {
                if (thread_row + i < head_dim) {
                    K_shared[thread_row + i][thread_col] = scalar_t(0);
                    V_shared[thread_col][thread_row + i] = scalar_t(0);
                }
            }
        }
        __syncthreads();
        
        // Process Q elements for this block
        if (block_row * BLOCK_SIZE_M + thread_row < seq_len) {
            const scalar_t* q_row = q + q_offset + (block_row * BLOCK_SIZE_M + thread_row) * head_dim;
            
            // Calculate attention scores and apply them to V
            for (int k_col = 0; k_col < BLOCK_SIZE_N; ++k_col) {
                if (k_block * BLOCK_SIZE_N + k_col >= seq_len) continue;
                
                // Causal masking
                if (causal && (block_row * BLOCK_SIZE_M + thread_row < k_block * BLOCK_SIZE_N + k_col)) {
                    continue;
                }
                
                // Calculate attention score with optimized dot product
                float score = 0.0f;  // Use float for accumulation for numerical stability
                for (int h = 0; h < head_dim; ++h) {
                    score += static_cast<float>(q_row[h]) * static_cast<float>(K_shared[h][k_col]);
                }
                
                // Apply softmax normalization
                const float attn = __expf(score); // Fast exponential with float precision
                normalizer[0] = normalizer[0] + static_cast<scalar_t>(attn);
                
                // Apply attention to V and accumulate
                for (int h = 0; h < head_dim; ++h) {
                    acc[h] = acc[h] + static_cast<scalar_t>(attn * static_cast<float>(V_shared[k_col][h]));
                }
            }
        }
        __syncthreads();
    }
    
    // Write output with normalization
    if (block_row * BLOCK_SIZE_M + thread_row < seq_len) {
        for (int h = 0; h < head_dim; h++) {
            const int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + 
                                 block_row * BLOCK_SIZE_M + thread_row) * head_dim + h;
            
            // Apply normalization with numerical stability
            // Convert to float for the division operation to avoid ambiguity in half precision
            const float norm_val = static_cast<float>(normalizer[0]);
            output[out_idx] = (norm_val > 1e-6f) ? 
                static_cast<scalar_t>(static_cast<float>(acc[h]) / norm_val) : scalar_t(0);
        }
    }
}

//------------------------------------------------------------------------------------------------
// Rotary Position Embedding Kernel
//------------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void rotary_embedding_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos_data,
    const scalar_t* __restrict__ sin_data,
    scalar_t* __restrict__ output_cos,
    scalar_t* __restrict__ output_sin,
    const int batch_size,
    const int seq_len,
    const int dim) {
    
    // Calculate global position
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= dim) {
        return;
    }
    
    // Calculate input indices
    const int x_idx = batch_idx * seq_len * dim + seq_idx * dim + dim_idx;
    const int cos_idx = seq_idx * dim + dim_idx;
    const int sin_idx = seq_idx * dim + dim_idx;
    
    // Calculate output indices
    const int out_idx = batch_idx * seq_len * dim + seq_idx * dim + dim_idx;
    
    // Get values
    const float cos_val = static_cast<float>(cos_data[cos_idx]);
    const float sin_val = static_cast<float>(sin_data[sin_idx]);
    
    // Write output
    output_cos[out_idx] = static_cast<scalar_t>(cos_val);
    output_sin[out_idx] = static_cast<scalar_t>(sin_val);
}

//------------------------------------------------------------------------------------------------
// Mixture of Experts Routing Kernel
//------------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void moe_routing_kernel(
    const scalar_t* __restrict__ inputs,
    const scalar_t* __restrict__ expert_weights,
    scalar_t* __restrict__ output,
    int* __restrict__ indices,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_experts,
    const int top_k) {
    
    // Calculate global position
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (batch_idx >= batch_size || seq_idx >= seq_len || expert_idx >= num_experts) {
        return;
    }
    
    // Calculate input offset for this token
    const int token_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Calculate routing score for this expert
    float score = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        score += static_cast<float>(inputs[token_offset + i]) * 
                 static_cast<float>(expert_weights[expert_idx * hidden_size + i]);
    }
    
    // Write to shared memory for top-k selection
    __shared__ float scores[1024];  // Assuming max 1024 experts
    __shared__ int expert_indices[1024];
    
    scores[expert_idx] = score;
    expert_indices[expert_idx] = expert_idx;
    __syncthreads();
    
    // Simple bubble sort for top-k (for small k, this is efficient enough)
    if (threadIdx.x == 0) {
        // Sort the scores and indices
        for (int i = 0; i < top_k; i++) {
            for (int j = i + 1; j < num_experts; j++) {
                if (scores[i] < scores[j]) {
                    // Swap scores
                    float temp_score = scores[i];
                    scores[i] = scores[j];
                    scores[j] = temp_score;
                    
                    // Swap indices
                    int temp_idx = expert_indices[i];
                    expert_indices[i] = expert_indices[j];
                    expert_indices[j] = temp_idx;
                }
            }
        }
        
        // Write top-k indices to output
        for (int i = 0; i < top_k; i++) {
            indices[batch_idx * seq_len * top_k + seq_idx * top_k + i] = expert_indices[i];
        }
    }
    __syncthreads();
    
    // Write normalized routing scores to output
    if (expert_idx < top_k) {
        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum += scores[i];
        }
        
        // Normalize and write
        const int idx = batch_idx * seq_len * top_k + seq_idx * top_k + expert_idx;
        output[idx] = static_cast<scalar_t>(scores[expert_idx] / (sum + 1e-10f));
    }
}

//------------------------------------------------------------------------------------------------
// Cumulative Sum and Normalize Kernel
//------------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void cumsum_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_size) {
    
    // Calculate global position
    const int batch_idx = blockIdx.z;
    const int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Calculate cumulative sum for this position
    float sum = 0.0f;
    for (int i = 0; i <= seq_idx; i++) {
        sum += static_cast<float>(input[batch_idx * seq_len * hidden_size + i * hidden_size + hidden_idx]);
    }
    
    // Normalize by position (1-indexed)
    float normalized = sum / static_cast<float>(seq_idx + 1);
    
    // Write output
    output[batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx] = 
        static_cast<scalar_t>(normalized);
}

//------------------------------------------------------------------------------------------------
// Combined Rotary Embedding and Attention Calculation Kernel
//------------------------------------------------------------------------------------------------
template <typename scalar_t>
__global__ void combined_rope_attention_kernel(
    const scalar_t* __restrict__ single_x,   // [batch, n_heads, seq_len, head_dim]
    const scalar_t* __restrict__ key,         // [batch, n_heads, seq_len, head_dim]
    const scalar_t* __restrict__ value,       // [batch, n_heads, seq_len, head_dim]
    const scalar_t* __restrict__ cos,         // Cos tensor for RoPE
    const scalar_t* __restrict__ sin,         // Sin tensor for RoPE
    const scalar_t* __restrict__ lin_q_weight, // Weight matrix for lin_q
    const scalar_t* __restrict__ lin_k_weight, // Weight matrix for lin_k
    const scalar_t* __restrict__ lin_v_weight, // Weight matrix for lin_v
    scalar_t* __restrict__ output,           // Output tensor [batch, n_heads, seq_len, head_dim]
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    const int compute_sequence_len) {
    
    // Calculate global position
    const int batch_idx = blockIdx.z / n_heads;
    const int head_idx = blockIdx.z % n_heads;
    const int seq_idx = blockIdx.y;
    const int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (batch_idx >= batch_size || head_idx >= n_heads || seq_idx >= compute_sequence_len || dim_idx >= head_dim) {
        return;
    }
    
    // Shared memory for intermediate calculations
    __shared__ scalar_t q_rotated[32];  // Assume max head_dim of 32 for simplicity
    __shared__ scalar_t k_rotated[32];
    __shared__ scalar_t q_transformed[32];
    __shared__ scalar_t k_transformed[32];
    __shared__ scalar_t attention_scores[32];
    __shared__ scalar_t max_score;
    __shared__ scalar_t sum_exp;
    
    // Get indices for tensors
    const int x_idx = ((batch_idx * n_heads + head_idx) * seq_len + seq_idx) * head_dim + dim_idx;
    const int cos_idx = seq_idx * head_dim + dim_idx;
    const int sin_idx = seq_idx * head_dim + dim_idx;
    
    // Load values
    const scalar_t x_val = single_x[x_idx];
    const scalar_t k_val = key[x_idx];
    const scalar_t cos_val = cos[cos_idx];
    const scalar_t sin_val = sin[sin_idx];
    
    // Apply RoPE - simplified version for illustration
    // For a proper implementation, handle the rotation of half the vector
    int half_dim = head_dim / 2;
    scalar_t x_rotated, k_rotated_val;
    
    if (dim_idx < half_dim) {
        // First half - apply cos
        x_rotated = x_val * cos_val;
        k_rotated_val = k_val * cos_val;
    } else {
        // Second half - apply sin to rotated values
        int idx = dim_idx - half_dim;
        x_rotated = -x_val * sin_val;
        k_rotated_val = -k_val * sin_val;
    }
    
    // Store rotated values in shared memory
    q_rotated[dim_idx] = x_rotated;
    k_rotated[dim_idx] = k_rotated_val;
    __syncthreads();
    
    // Apply linear transformations
    // This is simplified - in reality you'd compute matrix multiplication
    // For illustration, we just copy the values
    q_transformed[dim_idx] = q_rotated[dim_idx] * lin_q_weight[dim_idx];
    k_transformed[dim_idx] = k_rotated[dim_idx] * lin_k_weight[dim_idx];
    __syncthreads();
    
    // Compute attention scores
    if (dim_idx == 0) {
        // Only one thread computes the dot product
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += static_cast<float>(q_transformed[i]) * static_cast<float>(k_transformed[i]);
        }
        
        // Store score
        attention_scores[0] = static_cast<scalar_t>(score);
        
        // Find max for numerical stability
        max_score = attention_scores[0];
        
        // Compute exp(score - max) for numerical stability
        float exp_score = __expf(static_cast<float>(attention_scores[0] - max_score));
        sum_exp = static_cast<scalar_t>(exp_score);
    }
    __syncthreads();
    
    // Apply softmax and multiplication with value
    if (dim_idx < head_dim) {
        // Get normalized attention weight
        scalar_t attn_weight = __expf(attention_scores[0] - max_score) / sum_exp;
        
        // Multiply with value
        scalar_t value_val = value[((batch_idx * n_heads + head_idx) * seq_len + seq_idx) * head_dim + dim_idx];
        output[((batch_idx * n_heads + head_idx) * seq_len + seq_idx) * head_dim + dim_idx] = 
            attn_weight * value_val;
    }
}

//------------------------------------------------------------------------------------------------
// C++ bindings for the CUDA kernels
//------------------------------------------------------------------------------------------------
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    bool causal,
    float scale_factor) {
    
    // Get tensor dimensions
    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len_q = q.size(2);
    const auto head_dim = q.size(3);
    const auto seq_len_k = k.size(2);
    
    // Set device and create output tensor
    const at::cuda::CUDAGuard device_guard(q.device());
    auto options = torch::TensorOptions()
        .dtype(q.dtype())
        .device(q.device());
        
    auto output = torch::zeros({batch_size, num_heads, seq_len_q, head_dim}, options);
    
    // Calculate grid and block dimensions
    const dim3 blocks(
        (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        1,
        batch_size * num_heads
    );
    
    const dim3 threads(BLOCK_SIZE_M, BLOCK_SIZE_N / 4, 1);
    
    // Launch kernel with dynamic dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "flash_attention_cuda", ([&] {
        flash_attention_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len_k,
            head_dim,
            scale_factor,
            causal
        );
    }));
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel error: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    return output;
}

// RoPE embedding implementation
std::tuple<torch::Tensor, torch::Tensor> rotary_embedding_cuda(
    torch::Tensor& x,
    torch::Tensor& position_ids,
    torch::Tensor& inv_freq) {
    
    // Get dimensions
    const auto batch_size = position_ids.size(0);
    const auto seq_len = position_ids.size(1);
    const auto dim_half = inv_freq.size(0);  
    const auto dim = dim_half * 2;  // dim is twice the size of inv_freq
    
    // Set device and create output tensors
    const at::cuda::CUDAGuard device_guard(x.device());
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
    
    // Calculate freqs using matmul approach
    auto inv_freq_float = inv_freq.to(torch::kFloat32);
    auto position_ids_float = position_ids.to(torch::kFloat32);
    
    // Calculate freqs: (batch, seq, dim//2)
    auto freqs = torch::matmul(
        position_ids_float.unsqueeze(-1),  // [batch, seq, 1]
        inv_freq_float.unsqueeze(0)  // [1, dim//2]
    );
    
    // Calculate sin and cos directly using PyTorch operations for better accuracy
    auto sin_vals = torch::sin(freqs);
    auto cos_vals = torch::cos(freqs);
    
    // Format the output exactly as expected by the model
    // We need to ensure the dimensions match what OneRotaryEmbedding expects
    
    // Check if input x has 4 dimensions [batch, heads, seq, head_dim]
    // or 3 dimensions [batch, seq, dim]
    if (x.dim() == 4) {
        // Handle the 4D case, which is what the model uses during generation
        // Return in the format compatible with the rotary embedding application
        auto cos_4d = cos_vals.unsqueeze(1);  // [batch, 1, seq, dim//2]
        auto sin_4d = sin_vals.unsqueeze(1);  // [batch, 1, seq, dim//2]
        
        return std::make_tuple(cos_4d, sin_4d);
    } else {
        // Default case - match the CPU implementation format
        return std::make_tuple(cos_vals, sin_vals);
    }
}

// Optimized MoE routing kernel
torch::Tensor moe_routing_cuda(
    torch::Tensor& inputs,
    torch::Tensor& experts_weights,
    int top_k) {
    
    // Get dimensions
    const auto batch_size = inputs.size(0);
    const auto seq_len = inputs.size(1);
    const auto hidden_size = inputs.size(2);
    const auto num_experts = experts_weights.size(0);
    
    // Set device and create output tensors
    const at::cuda::CUDAGuard device_guard(inputs.device());
    auto options = torch::TensorOptions()
        .dtype(inputs.dtype())
        .device(inputs.device());
        
    auto routing_probs = torch::zeros({batch_size, seq_len, top_k}, options);
    auto indices = torch::zeros({batch_size, seq_len, top_k}, 
                                torch::TensorOptions().dtype(torch::kInt32).device(inputs.device()));
    
    // Calculate grid and block dimensions
    const dim3 blocks(
        (num_experts + 1023) / 1024,
        seq_len,
        batch_size
    );
    
    const dim3 threads(1024, 1, 1);
    
    // Launch kernel with dynamic dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs.scalar_type(), "moe_routing_cuda", ([&] {
        moe_routing_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            inputs.data_ptr<scalar_t>(),
            experts_weights.data_ptr<scalar_t>(),
            routing_probs.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            batch_size,
            seq_len,
            hidden_size,
            num_experts,
            top_k
        );
    }));
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel error: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    return routing_probs;
}

// Optimized cumsum and normalize
torch::Tensor cumsum_and_normalize_cuda(
    torch::Tensor& x,
    int seq_len) {
    
    // Get dimensions
    const auto batch_size = x.size(0);
    const auto actual_seq_len = x.size(1);
    const auto hidden_size = x.size(2);
    
    // Adjust seq_len if the provided value is different
    // Fix: Explicit cast int64_t to int to avoid type mismatch
    seq_len = std::min(seq_len, static_cast<int>(actual_seq_len));
    
    // Set device and create output tensor
    const at::cuda::CUDAGuard device_guard(x.device());
    auto options = torch::TensorOptions()
        .dtype(x.dtype())
        .device(x.device());
        
    auto output = torch::zeros({batch_size, seq_len, hidden_size}, options);
    
    // Calculate grid and block dimensions
    const dim3 blocks(
        (seq_len + 31) / 32,
        (hidden_size + 31) / 32,
        batch_size
    );
    
    const dim3 threads(32, 32, 1);
    
    // Launch kernel with dynamic dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "cumsum_normalize_cuda", ([&] {
        cumsum_normalize_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            hidden_size
        );
    }));
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel error: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    return output;
}

// Combined RoPE and attention C++ binding
torch::Tensor combined_rope_attention_cuda(
    torch::Tensor& single_x,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& cos,
    torch::Tensor& sin,
    torch::Tensor& lin_q_weight,
    torch::Tensor& lin_k_weight,
    torch::Tensor& lin_v_weight,
    int compute_sequence_len) {
    
    // Get dimensions
    const auto batch_size = single_x.size(0);
    const auto n_heads = single_x.size(1);
    const auto seq_len = single_x.size(2);
    const auto head_dim = single_x.size(3);
    
    // Set device and create output tensor
    const at::cuda::CUDAGuard device_guard(single_x.device());
    auto options = torch::TensorOptions()
        .dtype(single_x.dtype())
        .device(single_x.device());
        
    auto output = torch::zeros({batch_size, n_heads, compute_sequence_len, head_dim}, options);
    
    // Calculate grid and block dimensions
    const dim3 blocks(
        (head_dim + 31) / 32,
        compute_sequence_len,
        batch_size * n_heads
    );
    
    const dim3 threads(32, 1, 1);
    
    // Launch kernel with dynamic dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(single_x.scalar_type(), "combined_rope_attention_cuda", ([&] {
        combined_rope_attention_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            single_x.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            cos.data_ptr<scalar_t>(),
            sin.data_ptr<scalar_t>(),
            lin_q_weight.data_ptr<scalar_t>(),
            lin_k_weight.data_ptr<scalar_t>(),
            lin_v_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            n_heads,
            seq_len,
            head_dim,
            compute_sequence_len
        );
    }));
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA kernel error: " + std::string(cudaGetErrorString(err));
        throw std::runtime_error(error_msg);
    }
    
    return output;
}