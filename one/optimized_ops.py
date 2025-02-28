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
import torch.nn.functional as F
from einops import rearrange


def optimized_cumsum_and_normalize(x, s):
    """Optimized version of cumsum operation followed by normalization"""
    b, seq_len, d = x.shape
    device, dtype = x.device, x.dtype

    # Handle tensor sizes properly to avoid resizing warnings
    if s <= seq_len:
        # Only use the last s tokens, dimensions match exactly
        result = torch.zeros((b, s, d), device=device, dtype=dtype)
        source = x[:, -s:, :]
        torch.cumsum(source, dim=1, out=result)
    else:
        # When s > seq_len, we need to pad with zeros
        result = torch.zeros((b, s, d), device=device, dtype=dtype)
        # Only cumsum the available tokens
        if seq_len > 0:
            source_sum = torch.cumsum(x, dim=1)
            result[:, :seq_len, :] = source_sum

    # Create normalizer with exact size
    seq_range = torch.arange(1, s + 1, device=device, dtype=dtype).view(1, -1, 1)
    return result / seq_range


def fused_rope_attention(q, k, v, cos, sin):
    """Fused implementation of RoPE attention for better performance"""
    # Apply rotary position embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # Compute attention scores with improved numerical stability
    attn_weights = torch.softmax((q_embed * k_embed), dim=-1, dtype=torch.float32)

    # Apply attention and return
    return attn_weights * v


def rotate_half(x):
    """Optimized implementation of rotate_half"""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def flash_attention_wrapper(q, k, v, use_flash, flash_module=None):
    """Wrapper to choose between standard and flash attention"""
    if use_flash and flash_module is not None:
        return flash_module(q, k, v)
    else:
        # Standard attention
        attn_weights = torch.softmax((q * k), dim=-1, dtype=torch.float32).to(
            dtype=q.dtype
        )
        return attn_weights * v


# Calculate memory savings from using Flash Attention
def calculate_flash_attention_memory_savings(batch_size, seq_len, num_heads, head_dim):
    """Calculate the memory savings from using Flash Attention"""
    # Memory for standard attention
    attn_matrix_size = (
        batch_size * num_heads * seq_len * seq_len * 4
    )  # 4 bytes per float

    # Memory for flash attention - no explicit attention matrix
    flash_memory = batch_size * num_heads * seq_len * head_dim * 4  # Only store Q, K, V

    # Memory savings in bytes
    memory_savings = attn_matrix_size - flash_memory

    return memory_savings / (1024 * 1024)  # Return in MB
