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
import torch
import math


def rotate_half(x):
    """Optimized implementation of rotate_half"""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_flash(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary embeddings to q and k for Flash Attention"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlashAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for Flash Attention to avoid materializing the full attention matrix.
    This implementation is specifically tailored for the One2 architecture with improved numerical stability.
    """

    @staticmethod
    def forward(ctx, q, k, v, dropout_p=0.0, causal=False):
        """
        Forward pass of Flash Attention with improved numerical stability.
        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            v: Value tensor of shape [batch, heads, seq_len, head_dim]
            dropout_p: Dropout probability
            causal: Whether to use causal masking
        """
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v)
        ctx.dropout_p = dropout_p
        ctx.causal = causal

        batch_size, n_heads, seq_len, _ = q.shape

        # Scale q for numerical stability
        scale = 1.0 / math.sqrt(q.size(-1))
        q = q * scale

        # Set block sizes based on hardware - adjust these for your specific GPU
        block_size_m = 64
        block_size_n = 64

        # Initialize output
        o = torch.zeros_like(q)

        # Initialize softmax normalization factor
        l = torch.zeros(
            (batch_size, n_heads, seq_len, 1), device=q.device, dtype=q.dtype
        )

        # Process blocks directly using the block sizes
        for m_start in range(0, seq_len, block_size_m):
            m_end = min(m_start + block_size_m, seq_len)

            for n_start in range(0, seq_len, block_size_n):
                n_end = min(n_start + block_size_n, seq_len)

                # Skip if using causal mask and this block is above the diagonal
                if causal and m_start > n_end:
                    continue

                # Compute attention scores for this block
                block_k = k[:, :, n_start:n_end]
                block_q = q[:, :, m_start:m_end]

                # Compute attention scores [batch, n_heads, block_m, block_n]
                block_attn = torch.matmul(block_q, block_k.transpose(-1, -2))

                # Apply causal mask if needed
                if causal:
                    m_indices = torch.arange(m_start, m_end, device=q.device)[:, None]
                    n_indices = torch.arange(n_start, n_end, device=q.device)[None, :]
                    causal_mask = m_indices >= n_indices
                    block_attn = block_attn.masked_fill(
                        ~causal_mask.view(1, 1, m_end - m_start, n_end - n_start),
                        float("-inf"),
                    )

                # Stable softmax computations - use float32 for better precision
                with torch.autocast(enabled=False, device_type=q.device.type):
                    block_attn = block_attn.float()
                    block_attn_max = torch.max(block_attn, dim=-1, keepdim=True)[
                        0
                    ].detach()
                    block_attn_exp = torch.exp(block_attn - block_attn_max)

                    # Add small epsilon to prevent division by zero later
                    block_attn_exp = block_attn_exp + 1e-10

                # Convert back to original dtype
                block_attn_exp = block_attn_exp.to(dtype=q.dtype)

                # Apply dropout if training
                if dropout_p > 0.0 and m_start == 0:  # Apply only once per block row
                    dropout_mask = torch.rand_like(block_attn_exp) > dropout_p
                    block_attn_exp = (
                        block_attn_exp * dropout_mask / (1.0 - dropout_p + 1e-10)
                    )

                # Update softmax normalization
                block_l = torch.sum(block_attn_exp, dim=-1, keepdim=True)
                l[:, :, m_start:m_end] += block_l

                # Update output
                block_v = v[:, :, n_start:n_end]
                o[:, :, m_start:m_end] += torch.matmul(block_attn_exp, block_v)

        # Normalize output safely
        o = o / l.clamp(min=1e-10)

        # Check for NaNs or Infs in output and replace them
        o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)

        return o

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with numerical stability improvements.
        """
        q, k, v = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        causal = ctx.causal

        # Replace any NaN values in the gradients
        grad_output = torch.nan_to_num(grad_output, nan=0.0, posinf=0.0, neginf=0.0)

        # Simple implementation that's numerically stable and correctly handles dimensions
        with torch.autocast(enabled=False, device_type=grad_output.device.type):
            # Cast to float32 for better numerical stability
            q = q.float()
            k = k.float()
            v = v.float()
            grad_output = grad_output.float()

            # Get dimensions
            _, _, seq_len, head_dim = q.shape

            # Scale q for numerical stability
            scale = 1.0 / math.sqrt(head_dim)
            q = q * scale

            # Compute attention scores [batch, heads, seq, seq]
            attn = torch.matmul(q, k.transpose(-1, -2))

            # Apply causal mask if needed
            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device), diagonal=1
                ).bool()
                attn = attn.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Stable softmax computation
            attn_max = torch.max(attn, dim=-1, keepdim=True)[0].detach()
            attn = attn - attn_max
            attn_weights = torch.exp(attn)
            attn_sum = torch.sum(attn_weights, dim=-1, keepdim=True) + 1e-10
            attn_weights = attn_weights / attn_sum

            # Apply dropout if needed
            if dropout_p > 0.0:
                attn_dropout = torch.nn.functional.dropout(attn_weights, p=dropout_p)
            else:
                attn_dropout = attn_weights

            # Compute gradients - manually calculate without autograd.grad
            # grad_v: [batch, heads, seq_len, head_dim]
            grad_v = torch.matmul(attn_dropout.transpose(-2, -1), grad_output)

            # grad_attn_weights: [batch, heads, seq_len, seq_len]
            grad_attn_weights = torch.matmul(grad_output, v.transpose(-2, -1))

            # Apply softmax gradient formula: P * (G - sum(P*G))
            # where P is softmax probabilities and G is gradient
            grad_sum = torch.sum(grad_attn_weights * attn_weights, dim=-1, keepdim=True)
            grad_attn = attn_weights * (grad_attn_weights - grad_sum)

            # grad_k: [batch, heads, seq_len, head_dim]
            grad_k = torch.matmul(grad_attn.transpose(-2, -1), q)

            # grad_q: [batch, heads, seq_len, head_dim]
            grad_q = torch.matmul(grad_attn, k) * scale

            # Convert back to original dtype
            grad_q = grad_q.to(q.dtype)
            grad_k = grad_k.to(k.dtype)
            grad_v = grad_v.to(v.dtype)

            # Final NaN check
            grad_q = torch.nan_to_num(grad_q, nan=0.0, posinf=0.0, neginf=0.0)
            grad_k = torch.nan_to_num(grad_k, nan=0.0, posinf=0.0, neginf=0.0)
            grad_v = torch.nan_to_num(grad_v, nan=0.0, posinf=0.0, neginf=0.0)

        return grad_q, grad_k, grad_v, None, None


def flash_attention(q, k, v, dropout_p=0.0, causal=False):
    """
    Interface function for Flash Attention with improved error handling and numerical stability
    """
    try:
        # Try to use FlashAttention
        result = FlashAttentionFunction.apply(q, k, v, dropout_p, causal)

        # Check for NaNs or Infs in result and fall back if needed
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise RuntimeError("NaN detected in FlashAttention output")

        return result

    except Exception as e:
        # Fallback to standard attention with numerical stability improvements
        print(f"Flash attention failed, falling back to standard attention: {e}")

        # Use float32 for better numerical stability
        with torch.autocast(enabled=False, device_type=q.device.type):
            q_float = q.float()
            k_float = k.float()
            v_float = v.float()

            # Apply proper scaling
            scale = 1.0 / math.sqrt(q_float.size(-1))
            q_float = q_float * scale

            # Compute attention scores
            attn = torch.matmul(q_float, k_float.transpose(-1, -2))

            if causal:
                seq_len = q.size(2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device), diagonal=1
                ).bool()
                attn = attn.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # Stable softmax computation
            attn_max = torch.max(attn, dim=-1, keepdim=True)[0]
            attn = attn - attn_max  # Subtract max for stability
            attn = torch.exp(attn)
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + 1e-10  # Add epsilon
            attn = attn / attn_sum

            if dropout_p > 0.0:
                attn = torch.nn.functional.dropout(attn, p=dropout_p)

            result = torch.matmul(attn, v_float)

            # Convert back to original dtype
            result = result.to(dtype=q.dtype)

            # Replace any NaNs in the result
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            return result, attn


class OneFlashAttention(torch.nn.Module):
    """Flash Attention module specifically designed for One2 architecture"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.lstm_dropout if hasattr(config, "lstm_dropout") else 0.0
        self.causal = True  # Set to True for causal language modeling

    def forward(self, q, k, v):
        """
        Forward pass with flash attention and improved numerical stability.

        Args:
            q: [batch, n_heads, seq_len, head_dim]
            k: [batch, n_heads, seq_len, head_dim]
            v: [batch, n_heads, seq_len, head_dim]

        Returns:
            output: [batch, n_heads, seq_len, head_dim]
        """
        # Clip extreme values to prevent numerical instability
        q = torch.clamp(q, min=-1e4, max=1e4)
        k = torch.clamp(k, min=-1e4, max=1e4)

        result, attn_weights = flash_attention(
            q, k, v, self.dropout if self.training else 0.0, self.causal
        )

        # Final safety check
        if torch.isnan(result).any():
            # Emergency fallback - uniform attention
            print(
                "WARNING: NaN detected in attention result, using uniform attention instead"
            )
            uniform_weights = torch.ones_like(q) / q.size(-1)
            result = torch.matmul(uniform_weights, v)

        return result, attn_weights
