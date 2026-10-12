"""Attention mechanisms implemented from scratch in PyTorch for Nova."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.
    
    Splits the embedding dimension d_model into n_heads subspaces of dimension d_k = d_model / n_heads.
    Computes Scaled Dot-Product Attention in parallel across all heads:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + Mask) V
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Multi-Head Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean or float mask of shape (batch_size, seq_len) or (seq_len, seq_len)
            return_attention_weights: If True, returns attention distribution tensor
            
        Returns:
            Tuple of (output tensor of shape (batch_size, seq_len, d_model), optional attention weights)
        """
        B, T, C = x.shape

        # 1. Project input to Q, K, V
        # Shape: (B, T, C) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product: scores = (Q @ K^T) / sqrt(d_k)
        # Shape: (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 3. Apply causal mask if requested
        if self.causal:
            # Lower triangular mask: 1s where allowed, 0s where forbidden
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 4. Apply additional padding mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Shape (B, T) -> (B, 1, 1, T)
                pad_mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(~pad_mask, float("-inf"))
            elif mask.dim() == 4:
                scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Softmax to obtain attention probability weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 6. Weight values by attention probabilities: weights @ V
        # Shape: (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        context = torch.matmul(attn_weights_dropped, v)

        # 7. Concatenate heads and project out
        # Shape: (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, C)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(context)

        if return_attention_weights:
            return output, attn_weights
        return output, None


class CausalSelfAttention(MultiHeadAttention):
    """Causal Self-Attention convenience subclass configured with causal=True."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__(d_model=d_model, n_heads=n_heads, dropout=dropout, causal=True)
