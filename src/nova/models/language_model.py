"""Decoder-only Transformer Language Model for Nova."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock


class DecoderLanguageModel(nn.Module):
    """Educational Decoder-only Transformer Language Model.
    
    Trained with next-token prediction objective using causal self-attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # 1. Token Embeddings and Positional Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # 2. Stack of Causal Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                causal=True,
            )
            for _ in range(n_layers)
        ])

        # 3. Final Layer Normalization
        self.ln_final = nn.LayerNorm(d_model)

        # 4. Output projection to vocabulary logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns the number of trainable parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through decoder-only Transformer.
        
        Args:
            idx: Tensor of token indices, shape (batch_size, seq_len)
            targets: Optional tensor of target token indices, shape (batch_size, seq_len)
            
        Returns:
            Tuple of (logits of shape (B, T, vocab_size), optional cross-entropy loss)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum sequence length {self.max_seq_len}"

        # Position indices: [0, 1, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Embeddings: token + position
        tok_emb = self.token_embeddings(idx)  # (B, T, d_model)
        pos_emb = self.position_embeddings(positions)  # (T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to compute Cross Entropy over all non-padding tokens
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 25,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generates new tokens given a prompt tensor.
        
        Args:
            idx: Initial token IDs of shape (1, T) or (B, T)
            max_new_tokens: Maximum number of tokens to predict and append
            temperature: Softmax temperature scaling (1.0 = standard, <1.0 = sharper/greedy)
            top_k: Optional top-k filtering
            eos_token_id: Early stopping token ID
            
        Returns:
            Tensor of generated token IDs including original prompt
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]

            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0.0 or temperature == 0:
                # Greedy selection
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        return idx
