"""Tests for MultiHeadAttention and CausalSelfAttention."""

import pytest
import torch

from nova.models.attention import MultiHeadAttention, CausalSelfAttention


def test_multi_head_attention_shapes():
    B, T, C = 2, 8, 32
    n_heads = 4
    mha = MultiHeadAttention(d_model=C, n_heads=n_heads, causal=False)
    x = torch.randn(B, T, C)

    out, weights = mha(x, return_attention_weights=True)
    assert out.shape == torch.Size([B, T, C])
    assert weights.shape == torch.Size([B, n_heads, T, T])

    # Weights along last dimension should sum to 1.0
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)


def test_causal_masking_prevents_future_leakage():
    """Mathematical guarantee: token at position i cannot be influenced by token at position j > i."""
    B, T, C = 1, 4, 16
    c_attn = CausalSelfAttention(d_model=C, n_heads=2)
    c_attn.eval()

    # Create original input
    x1 = torch.randn(B, T, C)

    # Create modified input where only the last token (t=3) is altered
    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(B, C)

    with torch.no_grad():
        out1, _ = c_attn(x1)
        out2, _ = c_attn(x2)

    # For positions 0, 1, 2, the outputs MUST be identical because of the causal mask
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)

    # For position 3 (the altered token), outputs will differ
    assert not torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-3)
