#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations of CUDA kernels
torch::Tensor flash_attention_forward_cuda(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    bool causal,
    float scale_factor);

torch::Tensor moe_routing_cuda(
    torch::Tensor& inputs,
    torch::Tensor& experts_weights,
    int top_k);

std::tuple<torch::Tensor, torch::Tensor> rotary_embedding_cuda(
    torch::Tensor& x,
    torch::Tensor& position_ids,
    torch::Tensor& inv_freq);

torch::Tensor cumsum_and_normalize_cuda(
    torch::Tensor& x,
    int seq_len);

torch::Tensor combined_rope_attention_cuda(
    torch::Tensor& single_x,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& cos,
    torch::Tensor& sin,
    torch::Tensor& lin_q_weight,
    torch::Tensor& lin_k_weight,
    torch::Tensor& lin_v_weight,
    int compute_sequence_len);

// C++ interface to call CUDA implementations
torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal = true,
    float scale_factor = 1.0) {
    
    // Input validation
    TORCH_CHECK(q.dim() == 4, "Expected 4D tensor for q, got ", q.dim(), "D");
    TORCH_CHECK(k.dim() == 4, "Expected 4D tensor for k, got ", k.dim(), "D");
    TORCH_CHECK(v.dim() == 4, "Expected 4D tensor for v, got ", v.dim(), "D");
    
    // Check device
    TORCH_CHECK(q.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(k.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "Input must be a CUDA tensor");
    
    // Check dimensions
    TORCH_CHECK(q.size(0) == k.size(0), "Batch size mismatch");
    TORCH_CHECK(q.size(1) == k.size(1), "Number of heads mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "Sequence length mismatch between k and v");
    
    return flash_attention_forward_cuda(q, k, v, causal, scale_factor);
}

torch::Tensor moe_routing(
    torch::Tensor inputs,
    torch::Tensor experts_weights,
    int top_k) {
    
    TORCH_CHECK(inputs.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(experts_weights.device().is_cuda(), "Expert weights must be a CUDA tensor");
    
    return moe_routing_cuda(inputs, experts_weights, top_k);
}

std::tuple<torch::Tensor, torch::Tensor> rotary_embedding(
    torch::Tensor x,
    torch::Tensor position_ids,
    torch::Tensor inv_freq) {
    
    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(position_ids.device().is_cuda(), "Position IDs must be a CUDA tensor");
    TORCH_CHECK(inv_freq.device().is_cuda(), "inv_freq must be a CUDA tensor");
    
    return rotary_embedding_cuda(x, position_ids, inv_freq);
}

torch::Tensor cumsum_and_normalize(
    torch::Tensor x,
    int seq_len) {
    
    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    
    return cumsum_and_normalize_cuda(x, seq_len);
}

torch::Tensor combined_rope_attention(
    torch::Tensor single_x,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cos,
    torch::Tensor sin, 
    torch::Tensor lin_q_weight,
    torch::Tensor lin_k_weight,
    torch::Tensor lin_v_weight,
    int compute_sequence_len) {
    
    // Input validation
    TORCH_CHECK(single_x.dim() == 4, "Expected 4D tensor for single_x, got ", single_x.dim(), "D");
    TORCH_CHECK(key.dim() == 4, "Expected 4D tensor for key, got ", key.dim(), "D");
    TORCH_CHECK(value.dim() == 4, "Expected 4D tensor for value, got ", value.dim(), "D");
    
    // Check device
    TORCH_CHECK(single_x.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(key.device().is_cuda(), "Key must be a CUDA tensor");
    TORCH_CHECK(value.device().is_cuda(), "Value must be a CUDA tensor");
    TORCH_CHECK(cos.device().is_cuda(), "Cos must be a CUDA tensor");
    TORCH_CHECK(sin.device().is_cuda(), "Sin must be a CUDA tensor");
    
    return combined_rope_attention_cuda(
        single_x, key, value, cos, sin, 
        lin_q_weight, lin_k_weight, lin_v_weight, 
        compute_sequence_len
    );
}

// Module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention_forward, "Flash Attention optimized for One architecture");
    m.def("moe_routing", &moe_routing, "Optimized Mixture of Experts routing");
    m.def("rotary_embedding", &rotary_embedding, "Optimized RoPE embedding");
    m.def("cumsum_and_normalize", &cumsum_and_normalize, "Optimized cumsum and normalize operation");
    m.def("combined_rope_attention", &combined_rope_attention, "Combined RoPE embedding and attention in one step");
}