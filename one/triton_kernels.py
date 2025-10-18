# coding=utf-8
"""Triton kernels used by MedIT One optimized operators."""
from __future__ import annotations

import warnings
from typing import Tuple

import torch

try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False


def _require_triton(device: torch.device) -> None:
    """Ensure Triton is available on the current device."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed – falling back to PyTorch kernels")
    if device.type != "cuda":
        raise RuntimeError("Triton kernels require CUDA tensors")


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


if TRITON_AVAILABLE:  # pragma: no branch - only defined when Triton exists

    @triton.jit
    def _apply_rope_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        out_q_ptr,
        out_k_ptr,
        stride_q_batch,
        stride_q_head,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_seq,
        stride_k_dim,
        stride_cos_batch,
        stride_cos_head,
        stride_cos_seq,
        stride_cos_dim,
        stride_sin_batch,
        stride_sin_head,
        stride_sin_seq,
        stride_sin_dim,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        seq_idx = pid % seq_len
        head_idx = (pid // seq_len) % num_heads
        batch_idx = pid // (seq_len * num_heads)

        half_dim = head_dim // 2

        dim_offsets = tl.arange(0, BLOCK_SIZE)
        mask = dim_offsets < half_dim

        q_base = q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head + seq_idx * stride_q_seq
        k_base = k_ptr + batch_idx * stride_k_batch + head_idx * stride_k_head + seq_idx * stride_k_seq
        cos_base = cos_ptr + batch_idx * stride_cos_batch + head_idx * stride_cos_head + seq_idx * stride_cos_seq
        sin_base = sin_ptr + batch_idx * stride_sin_batch + head_idx * stride_sin_head + seq_idx * stride_sin_seq

        q_first = tl.load(q_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        q_second = tl.load(q_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        k_first = tl.load(k_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        k_second = tl.load(k_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

        cos_first = tl.load(cos_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        cos_second = tl.load(cos_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        sin_first = tl.load(sin_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        sin_second = tl.load(sin_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

        q_rot_first = -q_second
        q_rot_second = q_first
        k_rot_first = -k_second
        k_rot_second = k_first

        q_embed_first = q_first * cos_first + q_rot_first * sin_first
        q_embed_second = q_second * cos_second + q_rot_second * sin_second
        k_embed_first = k_first * cos_first + k_rot_first * sin_first
        k_embed_second = k_second * cos_second + k_rot_second * sin_second

        out_q_first = q_embed_first
        out_q_second = q_embed_second
        out_k_first = k_embed_first
        out_k_second = k_embed_second

        out_q_base = out_q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head + seq_idx * stride_q_seq
        out_k_base = out_k_ptr + batch_idx * stride_k_batch + head_idx * stride_k_head + seq_idx * stride_k_seq

        tl.store(out_q_base + dim_offsets, out_q_first, mask=mask)
        tl.store(out_q_base + dim_offsets + half_dim, out_q_second, mask=mask)
        tl.store(out_k_base + dim_offsets, out_k_first, mask=mask)
        tl.store(out_k_base + dim_offsets + half_dim, out_k_second, mask=mask)

    @triton.jit
    def _fused_rope_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_q_batch,
        stride_q_head,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_head,
        stride_k_seq,
        stride_k_dim,
        stride_v_batch,
        stride_v_head,
        stride_v_seq,
        stride_v_dim,
        stride_cos_batch,
        stride_cos_head,
        stride_cos_seq,
        stride_cos_dim,
        stride_sin_batch,
        stride_sin_head,
        stride_sin_seq,
        stride_sin_dim,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        seq_idx = pid % seq_len
        head_idx = (pid // seq_len) % num_heads
        batch_idx = pid // (seq_len * num_heads)

        half_dim = head_dim // 2

        dim_offsets = tl.arange(0, BLOCK_SIZE)
        mask = dim_offsets < half_dim

        q_base = q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head + seq_idx * stride_q_seq
        k_base = k_ptr + batch_idx * stride_k_batch + head_idx * stride_k_head + seq_idx * stride_k_seq
        v_base = v_ptr + batch_idx * stride_v_batch + head_idx * stride_v_head + seq_idx * stride_v_seq
        cos_base = cos_ptr + batch_idx * stride_cos_batch + head_idx * stride_cos_head + seq_idx * stride_cos_seq
        sin_base = sin_ptr + batch_idx * stride_sin_batch + head_idx * stride_sin_head + seq_idx * stride_sin_seq

        q_first = tl.load(q_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        q_second = tl.load(q_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        k_first = tl.load(k_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        k_second = tl.load(k_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        v_first = tl.load(v_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        v_second = tl.load(v_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

        cos_first = tl.load(cos_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        cos_second = tl.load(cos_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        sin_first = tl.load(sin_base + dim_offsets, mask=mask, other=0.0).to(tl.float32)
        sin_second = tl.load(sin_base + dim_offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

        q_rot_first = -q_second
        q_rot_second = q_first
        k_rot_first = -k_second
        k_rot_second = k_first

        q_embed_first = q_first * cos_first + q_rot_first * sin_first
        q_embed_second = q_second * cos_second + q_rot_second * sin_second
        k_embed_first = k_first * cos_first + k_rot_first * sin_first
        k_embed_second = k_second * cos_second + k_rot_second * sin_second

        attn_first = q_embed_first * k_embed_first
        attn_second = q_embed_second * k_embed_second

        attn_first = tl.where(mask, attn_first, float("-inf"))
        attn_second = tl.where(mask, attn_second, float("-inf"))

        max_first = tl.max(attn_first, axis=0)
        max_second = tl.max(attn_second, axis=0)
        max_val = tl.maximum(max_first, max_second)

        exp_first = tl.exp(attn_first - max_val)
        exp_second = tl.exp(attn_second - max_val)

        exp_first = tl.where(mask, exp_first, 0.0)
        exp_second = tl.where(mask, exp_second, 0.0)

        denom = tl.sum(exp_first, axis=0) + tl.sum(exp_second, axis=0) + 1e-6

        softmax_first = exp_first / denom
        softmax_second = exp_second / denom

        out_first = softmax_first * v_first
        out_second = softmax_second * v_second

        out_base = out_ptr + batch_idx * stride_v_batch + head_idx * stride_v_head + seq_idx * stride_v_seq

        tl.store(out_base + dim_offsets, out_first, mask=mask)
        tl.store(out_base + dim_offsets + half_dim, out_second, mask=mask)


def apply_rotary_pos_emb_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using Triton kernels."""
    _require_triton(q.device)

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be 4D tensors [batch, head, seq, dim]")
    if q.shape != k.shape:
        raise ValueError("q and k must have matching shapes")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for rotary embeddings")

    cos_aligned = cos
    sin_aligned = sin
    if cos_aligned.shape != q.shape:
        cos_aligned = cos_aligned.expand_as(q).contiguous()
    if sin_aligned.shape != q.shape:
        sin_aligned = sin_aligned.expand_as(q).contiguous()

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    half_dim = head_dim // 2
    block_size = min(128, _next_power_of_2(half_dim))

    grid = (batch_size * num_heads * seq_len,)

    _apply_rope_kernel[grid](
        q,
        k,
        cos_aligned,
        sin_aligned,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        cos_aligned.stride(0),
        cos_aligned.stride(1),
        cos_aligned.stride(2),
        cos_aligned.stride(3),
        sin_aligned.stride(0),
        sin_aligned.stride(1),
        sin_aligned.stride(2),
        sin_aligned.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return out_q, out_k


def fused_rope_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Fused rotary attention implemented in Triton."""
    _require_triton(q.device)

    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k, and v must share the same shape")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for rotary attention")

    cos_aligned = cos
    sin_aligned = sin
    if cos_aligned.shape != q.shape:
        cos_aligned = cos_aligned.expand_as(q).contiguous()
    if sin_aligned.shape != q.shape:
        sin_aligned = sin_aligned.expand_as(q).contiguous()

    out = torch.empty_like(v)

    half_dim = head_dim // 2
    block_size = min(128, _next_power_of_2(half_dim))

    grid = (batch_size * num_heads * seq_len,)

    _fused_rope_attention_kernel[grid](
        q,
        k,
        v,
        cos_aligned,
        sin_aligned,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        cos_aligned.stride(0),
        cos_aligned.stride(1),
        cos_aligned.stride(2),
        cos_aligned.stride(3),
        sin_aligned.stride(0),
        sin_aligned.stride(1),
        sin_aligned.stride(2),
        sin_aligned.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return out


def warn_triton_unavailable() -> None:
    if not TRITON_AVAILABLE:
        warnings.warn(
            "Triton is not installed; MedIT One will use PyTorch implementations instead.",
            stacklevel=2,
        )
