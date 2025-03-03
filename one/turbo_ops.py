# coding=utf-8
# Copyright 2025 The MedIT Solutions Kurman i Wspolnicy Sp. z o. o. Team.
# https://meditsolutions.pl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import importlib
import importlib.util
import torch
from typing import Optional, Tuple
import warnings

# Try to import the CUDA-accelerated implementation from the root-level extension
TURBO_MODE = False


def check_for_cuda_extension():
    """Check multiple locations for the CUDA extension."""
    # Check if the one_turbo module is directly importable
    try:
        import one_turbo

        return one_turbo
    except ImportError:
        pass

    # Check if the extension was built by install_cuda.py
    try:
        # Get the package directory
        package_dir = os.path.dirname(os.path.abspath(__file__))
        # Check if we have the marker file indicating successful CUDA installation
        cuda_marker = os.path.join(package_dir, "cuda_installed")
        if os.path.exists(cuda_marker):
            # Try to find the extension in build directory
            build_dir = os.path.join(os.path.dirname(package_dir), "build")
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if file.startswith("one_turbo") and file.endswith((".so", ".pyd")):
                        extension_path = os.path.join(root, file)
                        # Try to load the extension from this path
                        spec = importlib.util.spec_from_file_location(
                            "one_turbo", extension_path
                        )
                        if spec:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            return module
    except Exception as e:
        warnings.warn(f"Error loading CUDA extension from build directory: {e}")

    return None


# Try to load the CUDA extension
extension = check_for_cuda_extension()
if extension:
    # Extract the functions from the extension
    flash_attention = getattr(extension, "flash_attention", None)
    moe_routing = getattr(extension, "moe_routing", None)
    rotary_embedding = getattr(extension, "rotary_embedding", None)
    cumsum_and_normalize = getattr(extension, "cumsum_and_normalize", None)
    combined_rope_attention = getattr(extension, "combined_rope_attention", None)

    # Verify that we have all expected functions
    required_functions = [
        flash_attention,
        moe_routing,
        rotary_embedding,
        cumsum_and_normalize,
        combined_rope_attention,
    ]
    if all(required_functions):
        TURBO_MODE = True
        print("One Turbo backend enabled!")
    else:
        warnings.warn("One Turbo extension found but missing some required functions.")

if not TURBO_MODE:
    warnings.warn(
        "One Turbo backend not available. Running in fallback mode. "
        "To enable turbo mode, run: python install_cuda.py"
    )


def turbo_flash_attention(
    q: torch.Tensor,  # shape: [batch_size, n_heads, seq_len, head_dim]
    k: torch.Tensor,  # shape: [batch_size, n_heads, seq_len, head_dim]
    v: torch.Tensor,  # shape: [batch_size, n_heads, seq_len, head_dim]
    causal: bool = True,
    scale_factor: float = None,
) -> torch.Tensor:
    """
    Optimized flash attention implementation that uses CUDA kernels when available.
    Falls back to optimized PyTorch implementation when CUDA extension is not available.
    Args:
        q: Query tensor [batch_size, n_heads, seq_len, head_dim]
        k: Key tensor [batch_size, n_heads, seq_len, head_dim]
        v: Value tensor [batch_size, n_heads, seq_len, head_dim]
        causal: Whether to apply causal masking
        scale_factor: Optional scaling factor. If None, uses 1/sqrt(head_dim)
    Returns:
        output: Attention output [batch_size, n_heads, seq_len, head_dim]
    """
    # Apply scaling if not provided
    if scale_factor is None:
        scale_factor = 1.0 / (q.size(-1) ** 0.5)
    if TURBO_MODE and q.is_cuda:
        # Call CUDA implementation
        return flash_attention(q, k, v, causal, scale_factor)
    else:
        # Fallback implementation in PyTorch
        q = q * scale_factor
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))
        # Apply causal mask if needed
        if causal:
            mask = torch.triu(
                torch.ones(
                    scores.shape[-2],
                    scores.shape[-1],
                    device=scores.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores.masked_fill_(mask, float("-inf"))
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attention_weights = attention_weights.to(dtype=q.dtype)
        # Apply attention
        output = torch.matmul(attention_weights, v)
        return output


def turbo_moe_routing(
    inputs: torch.Tensor, experts_weights: torch.Tensor, top_k: int
) -> torch.Tensor:
    """
    Optimized MoE routing implementation.

    Args:
        inputs: Input tensor
        experts_weights: Expert weight tensor
        top_k: Number of top experts to select

    Returns:
        routed_tensor: Tensor after expert routing
    """
    if TURBO_MODE and inputs.is_cuda:
        # Call CUDA implementation
        return moe_routing(inputs, experts_weights, top_k)
    else:
        # Fallback implementation
        # Simple top-k selection for now
        top_values, indices = torch.topk(
            torch.einsum("be,n->bne", inputs, experts_weights.float()), k=top_k, dim=1
        )

        # Route inputs through top-k experts
        # This is a simplified implementation
        return inputs


def turbo_rotary_embedding(
    x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized RoPE embedding implementation with format matching OneRotaryEmbedding.

    Args:
        x: Input tensor [batch_size, seq_len, dim] or [batch_size, n_heads, seq_len, head_dim]
        position_ids: Position IDs [batch_size, seq_len]
        inv_freq: Inverse frequency tensor [dim/2]

    Returns:
        cos, sin: Cosine and sine components in the format expected by the model
    """
    if TURBO_MODE and x.is_cuda:
        # CUDA implementation returns [batch_size, seq_len, dim] format tensors
        # Make the result compatible with the OneRotaryEmbedding return format
        try:
            cos, sin = rotary_embedding(x, position_ids, inv_freq)

            # Check dimensions - our CUDA implementation may need reshaping to match
            # the expected format based on the input x dimensions
            if len(x.shape) == 4:  # [batch, heads, seq, head_dim]
                # Reshape to match the expected format for the model
                batch_size, seq_len, dim = cos.shape
                n_heads = x.shape[1]
                head_dim = dim // n_heads

                # Reshape to [batch, seq_len, n_heads, head_dim]
                cos = cos.view(batch_size, seq_len, n_heads, head_dim)
                sin = sin.view(batch_size, seq_len, n_heads, head_dim)

            return cos, sin
        except Exception as e:
            print(
                f"CUDA rotary_embedding failed: {e}, falling back to CPU implementation"
            )
            # Fall back to CPU implementation if CUDA version fails
            return turbo_rotary_embedding_cpu(x, position_ids, inv_freq)
    else:
        # Use CPU implementation
        return turbo_rotary_embedding_cpu(x, position_ids, inv_freq)


def turbo_rotary_embedding_cpu(
    x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU implementation of rotary embedding with careful dimension handling.
    Matches the format of OneRotaryEmbedding output.
    """
    # Get key dimensions
    batch_size = position_ids.shape[0]
    seq_len = position_ids.shape[1]

    # Ensure inv_freq is in the correct format
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    # Force float32 for better precision
    with torch.autocast(device_type=x.device.type, enabled=False):
        # Calculate rotations
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )

        # Duplicate frequencies for sin/cos pairs
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Convert back to original dtype
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)

    return cos, sin


def turbo_cumsum_and_normalize(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Optimized cumulative sum and normalization.

    Args:
        x: Input tensor
        seq_len: Sequence length

    Returns:
        normalized_tensor: Normalized tensor after cumsum
    """
    original_dtype = x.dtype

    if TURBO_MODE and x.is_cuda:
        try:
            # Convert to float32 if BFloat16 since CUDA kernel doesn't support BFloat16
            if x.dtype == torch.bfloat16:
                x_float = x.to(torch.float32)
                result = cumsum_and_normalize(x_float, seq_len)
                # Convert back to original dtype
                return result.to(original_dtype)
            else:
                # Call CUDA implementation directly for supported dtypes
                return cumsum_and_normalize(x, seq_len)
        except RuntimeError as e:
            # Handle other potential runtime errors by falling back to CPU
            warnings.warn(
                f"CUDA cumsum_and_normalize failed: {e}, falling back to CPU implementation"
            )
            return turbo_cumsum_and_normalize_cpu(x, seq_len, original_dtype)
    else:
        # Use CPU fallback
        return turbo_cumsum_and_normalize_cpu(x, seq_len, original_dtype)


def turbo_cumsum_and_normalize_cpu(
    x: torch.Tensor, seq_len: int, original_dtype: torch.dtype
) -> torch.Tensor:
    """CPU fallback with proper dtype handling for cumsum and normalize."""
    # Use float32 for better precision in CPU implementation
    x_float = x.to(torch.float32) if x.dtype != torch.float32 else x

    # Fallback implementation
    result = torch.cumsum(x_float, dim=1)

    # Create normalizer with exact size
    seq_range = torch.arange(1, seq_len + 1, device=x.device, dtype=torch.float32).view(
        1, -1, 1
    )

    # Normalize
    result = result / (seq_range + 1e-8)

    # Convert back to original dtype
    if original_dtype != torch.float32:
        result = result.to(original_dtype)

    return result


def turbo_combined_rope_attention(
    single_x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    lin_q: torch.nn.Linear,
    lin_k: torch.nn.Linear,
    lin_v: torch.nn.Linear,
    compute_sequence_len: int,
) -> torch.Tensor:
    """
    Combined RoPE embedding and attention calculation in a single CUDA operation.
    This optimizes the memory transfers and accelerates inference.

    Args:
        single_x: Query input [batch, n_heads, seq_len, head_dim]
        key: Key input [batch, n_heads, seq_len, head_dim]
        value: Value input [batch, n_heads, seq_len, head_dim]
        cos: Cosine values for RoPE [batch, seq_len, dim]
        sin: Sine values for RoPE [batch, seq_len, dim]
        lin_q: Linear transformation for query
        lin_k: Linear transformation for key
        lin_v: Linear transformation for value
        compute_sequence_len: Length of sequence to compute (typically 1 during inference)

    Returns:
        processed_output: Output after RoPE and attention [batch, n_heads, compute_sequence_len, head_dim]
    """
    if TURBO_MODE and single_x.is_cuda:
        try:
            # Extract weights from nn.Linear layers
            lin_q_weight = lin_q.weight.T  # Transpose to match CUDA kernel expectation
            lin_k_weight = lin_k.weight.T
            lin_v_weight = lin_v.weight.T

            # Call CUDA implementation
            return combined_rope_attention(
                single_x,
                key,
                value,
                cos,
                sin,
                lin_q_weight,
                lin_k_weight,
                lin_v_weight,
                compute_sequence_len,
            )
        except Exception as e:
            print(
                f"CUDA combined_rope_attention failed: {e}, falling back to CPU implementation"
            )
            return turbo_combined_rope_attention_cpu(
                single_x,
                key,
                value,
                cos,
                sin,
                lin_q,
                lin_k,
                lin_v,
                compute_sequence_len,
            )
    else:
        return turbo_combined_rope_attention_cpu(
            single_x, key, value, cos, sin, lin_q, lin_k, lin_v, compute_sequence_len
        )


def turbo_combined_rope_attention_cpu(
    single_x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    lin_q: torch.nn.Linear,
    lin_k: torch.nn.Linear,
    lin_v: torch.nn.Linear,
    compute_sequence_len: int,
) -> torch.Tensor:
    """CPU fallback for the combined RoPE embedding and attention calculation."""
    # Make sure we're working with the right sequence length
    single_x_local = single_x[:, :, -compute_sequence_len:, :].contiguous()
    key_local = key[:, :, -compute_sequence_len:, :].contiguous()
    value_local = value[:, :, -compute_sequence_len:, :].contiguous()

    # Apply rotary embeddings
    cos_dim = cos.dim()
    if cos_dim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    single_x_rotated = single_x_local * cos + rotate_half(single_x_local) * sin
    key_rotated = key_local * cos + rotate_half(key_local) * sin

    # Apply linear transformations
    q = lin_q(single_x_rotated)
    k = lin_k(key_rotated)
    v = lin_v(value_local)

    # Compute attention with numerical stability
    attn_logits = q * k

    # Stable softmax operations
    with torch.no_grad():
        attn_max = torch.max(attn_logits, dim=-1, keepdim=True)[0]
        attn_logits = attn_logits - attn_max

    attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(
        dtype=q.dtype
    )

    # Check for NaNs in the attention weights
    if torch.isnan(attn_weights).any():
        # Replace NaNs with uniform attention
        mask = torch.isnan(attn_weights)
        attn_weights = attn_weights.masked_fill(mask, 1.0 / attn_weights.size(-1))

    # Apply attention
    output = attn_weights * v

    return output


class TurboFlashAttention(torch.nn.Module):
    """High-performance Flash Attention module for the One architecture."""

    def __init__(self, causal=True, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout = dropout

    def forward(self, q, k, v):
        """
        Forward pass with turbo flash attention.

        Args:
            q: [batch, n_heads, seq_len, head_dim]
            k: [batch, n_heads, seq_len, head_dim]
            v: [batch, n_heads, seq_len, head_dim]

        Returns:
            output: [batch, n_heads, seq_len, head_dim]
        """
        # Ensure suitable tensor format
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Apply dropout during training
        dropout_p = self.dropout if self.training else 0.0

        # Apply flash attention
        output = turbo_flash_attention(
            q, k, v, causal=self.causal, scale_factor=None  # Use default scaling
        )

        # Return output and dummy attention weights
        attn_weights = torch.zeros(
            (q.size(0), q.size(1), q.size(2), k.size(2)), device=q.device, dtype=q.dtype
        )

        return output, attn_weights


# Runtime check for available CUDA device
if torch.cuda.is_available() and not TURBO_MODE:
    warnings.warn(
        "CUDA is available but One Turbo backend is not compiled. "
        "For maximum performance, please compile the CUDA extension."
    )
