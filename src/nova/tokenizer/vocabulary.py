"""Vocabulary management for Nova's educational tokenizer."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class Vocabulary:
    """Manages the mapping between tokens and their integer IDs.
    
    Includes reserved special tokens (<PAD>, <UNK>, <BOS>, <EOS>) and tracks token frequency.
    """

    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_counts: Dict[str, int] = Counter()

        # Add reserved special tokens in deterministic order
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self._add_special(tok)

    def _add_special(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.token_counts[token] = 0
            return idx
        return self.token_to_id[token]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def add_token(self, token: str, count: int = 1) -> int:
        """Adds a token or increments its count."""
        if token in self.token_to_id:
            self.token_counts[token] += count
            return self.token_to_id[token]

        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        self.token_counts[token] = count
        return idx

    def build_from_tokens(
        self,
        tokenized_corpus: Iterable[Sequence[str]],
        min_freq: int = 1,
    ) -> None:
        """Builds vocabulary from an iterable of token lists, respecting minimum frequency."""
        counts = Counter()
        for tokens in tokenized_corpus:
            counts.update(tokens)

        # Sort tokens by frequency (descending), then alphabetically for determinism
        sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        for tok, freq in sorted_tokens:
            if freq >= min_freq and tok not in self.token_to_id:
                self.add_token(tok, count=freq)

    def token_to_index(self, token: str) -> int:
        """Returns token ID, falling back to unk_id if unseen."""
        return self.token_to_id.get(token, self.unk_id)

    def index_to_token(self, idx: int) -> str:
        """Returns token string for a given index."""
        return self.id_to_token.get(idx, self.unk_token)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def save(self, filepath: str | Path) -> None:
        """Saves vocabulary structure and counts to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "token_to_id": self.token_to_id,
            "token_counts": dict(self.token_counts),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> "Vocabulary":
        """Loads vocabulary from saved JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
        )
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(idx): tok for tok, idx in vocab.token_to_id.items()}
        vocab.token_counts = Counter(data.get("token_counts", {}))
        return vocab

    def inspect(self, top_n: int = 15) -> Dict[str, Any]:
        """Returns diagnostic inspection metadata."""
        return {
            "size": len(self),
            "special_tokens": {
                "pad": (self.pad_token, self.pad_id),
                "unk": (self.unk_token, self.unk_id),
                "bos": (self.bos_token, self.bos_id),
                "eos": (self.eos_token, self.eos_id),
            },
            "top_tokens": sorted(
                [(tok, self.token_counts.get(tok, 0)) for tok in self.token_to_id if tok not in [
                    self.pad_token, self.unk_token, self.bos_token, self.eos_token
                ]],
                key=lambda x: -x[1],
            )[:top_n],
        }
