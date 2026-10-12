"""Transformer Block implementing Pre-LayerNorm architecture with Residual Connections."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Two-layer Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = Linear2(Dropout(GELU(Linear1(x))))
        return self.dropout(self.linear2(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with Pre-LayerNorm and residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            causal=causal,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN attention with residual connection: x = x + Attention(LN1(x))
        norm_x = self.ln1(x)
        attn_out, weights = self.attn(
            norm_x,
            mask=mask,
            return_attention_weights=return_attention_weights,
        )
        x = x + self.dropout(attn_out)

        # Pre-LN FFN with residual connection: x = x + FFN(LN2(x))
        x = x + self.ffn(self.ln2(x))

        return x, weights
