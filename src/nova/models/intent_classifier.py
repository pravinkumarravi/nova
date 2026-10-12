"""Neural Intent Classifier and Baseline Matcher for Nova."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock


class NeuralIntentClassifier(nn.Module):
    """Bidirectional Transformer Encoder for Intent Classification.
    
    Encodes the input question sequence using multi-head self-attention and
    pools representations to predict the intent label with a confidence score.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Bidirectional Transformer blocks (causal=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                causal=False,
            )
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

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

    def forward(
        self,
        idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for intent classification.
        
        Args:
            idx: Tensor of token IDs (B, T)
            labels: Optional ground-truth integer labels (B,)
            
        Returns:
            Tuple of (logits of shape (B, num_classes), optional cross_entropy loss)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(positions)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Padding mask: True where valid token, False where padding
        pad_mask = idx != self.pad_token_id  # (B, T)

        for block in self.blocks:
            x, _ = block(x, mask=pad_mask)

        x = self.ln_final(x)

        # Mean pooling over non-padding tokens
        mask_expanded = pad_mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_embeddings = (x * mask_expanded).sum(dim=1)  # (B, d_model)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        pooled = sum_embeddings / sum_mask

        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    def predict(
        self,
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns predicted class indices and confidence scores (via softmax)."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(idx)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)
        return predicted_idx, confidence


class RuleBasedBaselineClassifier:
    """Deterministic pattern-matching baseline for diagnostic comparison.
    
    Used to benchmark neural classifier accuracy and identify cases where
    simple keyword or regex rules fail compared to learned representations.
    """

    KEYWORD_RULES = {
        "ASK_CURRENT_TIME": ["time", "clock", "hour", "minute"],
        "ASK_CURRENT_DATE": ["date", "day", "today", "weekday", "month", "year"],
        "ASK_NAME": ["name", "call you", "called"],
        "ASK_IDENTITY": ["who are you", "who're you", "who r u", "who r you", "what are you", "introduce", "describe yourself"],
        "ASK_HUMAN": ["human", "person", "living human", "flesh and blood"],
        "ASK_CAPABILITIES": ["what can you do", "capabilities", "features", "skills", "help me with"],
        "ASK_MODEL_TYPE": ["architecture", "transformer", "neural", "parameters", "model are you", "how were you built"],
        "ASK_SUPPORTED_TOPICS": ["topics", "what questions", "what can i ask", "supported"],
        "ASK_OFFLINE_STATUS": ["offline", "internet", "cloud", "locally", "local machine"],
        "GREETING": ["hello", "hi", "hey", "morning", "afternoon", "evening", "greetings", "howdy", "how are you", "how r u", "how do you do"],
        "THANKS": ["thank", "thanks", "appreciate", "thx"],
        "GOODBYE": ["bye", "goodbye", "farewell", "see you"],
    }

    def predict(self, text: str) -> Tuple[str, float]:
        norm = text.lower().strip()
        for intent, patterns in self.KEYWORD_RULES.items():
            for pat in patterns:
                if pat in norm:
                    return intent, 0.95
        return "UNKNOWN", 0.50
