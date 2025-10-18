import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from one.flash_attention import apply_rotary_pos_emb_flash, rotate_half
from one.optimized_ops import fused_rope_attention


def _reference_apply_rotary(q, k, cos, sin):
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q_ref = (q * cos_u) + (rotate_half(q) * sin_u)
    k_ref = (k * cos_u) + (rotate_half(k) * sin_u)
    return q_ref, k_ref


def _reference_fused_rope_attention(q, k, v, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    attn = torch.softmax(q_embed * k_embed, dim=-1, dtype=torch.float32).to(q.dtype)
    return attn * v


@pytest.mark.parametrize("dtype", [torch.float32])
def test_apply_rotary_pos_emb_flash_matches_reference_cpu(dtype):
    device = torch.device("cpu")
    batch, heads, seq_len, head_dim = 2, 2, 4, 8
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    cos = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)

    q_ref, k_ref = _reference_apply_rotary(q, k, cos, sin)
    q_out, k_out = apply_rotary_pos_emb_flash(q, k, cos, sin)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_rope_attention_matches_reference_cpu(dtype):
    device = torch.device("cpu")
    batch, heads, seq_len, head_dim = 2, 2, 4, 8
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cos = torch.randn_like(q)
    sin = torch.randn_like(q)

    expected = _reference_fused_rope_attention(q, k, v, cos, sin)
    actual = fused_rope_attention(q, k, v, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_rotary_pos_emb_flash_cuda_matches_reference(dtype):
    device = torch.device("cuda")
    batch, heads, seq_len, head_dim = 2, 2, 4, 16
    try:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    except RuntimeError as exc:
        pytest.skip(f"dtype {dtype} not supported on this device: {exc}")
    k = torch.randn_like(q)
    cos = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)

    q_ref, k_ref = _reference_apply_rotary(q, k, cos, sin)
    q_out, k_out = apply_rotary_pos_emb_flash(q, k, cos, sin)

    atol = 5e-3 if dtype == torch.float16 else 1e-4
    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rope_attention_cuda_matches_reference(dtype):
    device = torch.device("cuda")
    batch, heads, seq_len, head_dim = 2, 2, 4, 16
    try:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    except RuntimeError as exc:
        pytest.skip(f"dtype {dtype} not supported on this device: {exc}")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cos = torch.randn_like(q)
    sin = torch.randn_like(q)

    expected = _reference_fused_rope_attention(q, k, v, cos, sin)
    actual = fused_rope_attention(q, k, v, cos, sin)

    atol = 5e-3 if dtype == torch.float16 else 1e-4
    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
