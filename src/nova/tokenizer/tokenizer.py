"""Educational word-level tokenizer for Nova."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .vocabulary import Vocabulary


class SimpleTokenizer:
    """Educational rule-based word and punctuation tokenizer.
    
    Splits text on whitespace and punctuation boundaries, normalizes case,
    and encodes words into integer token IDs using a learned Vocabulary.
    """

    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        vocabulary: Optional[Vocabulary] = None,
        lowercase: bool = True,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
    ) -> None:
        self.lowercase = lowercase
        self.vocabulary = vocabulary or Vocabulary(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    @property
    def pad_token_id(self) -> int:
        return self.vocabulary.pad_id

    @property
    def unk_token_id(self) -> int:
        return self.vocabulary.unk_id

    @property
    def bos_token_id(self) -> int:
        return self.vocabulary.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.vocabulary.eos_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def normalize(self, text: str) -> str:
        """Normalizes input string by trimming and optional lowercasing."""
        if not text:
            return ""
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits normalized text into tokens preserving punctuation."""
        norm = self.normalize(text)
        if not norm:
            return []
        return self.TOKEN_PATTERN.findall(norm)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encodes text to a list of integer token IDs."""
        tokens = self.tokenize(text)
        token_ids = [self.vocabulary.token_to_index(tok) for tok in tokens]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decodes a sequence of token IDs back into readable English text."""
        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        tokens: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.vocabulary.index_to_token(tid)
            tokens.append(tok)

        if not tokens:
            return ""

        # Reconstruct natural spacing around punctuation
        reconstructed: List[str] = []
        for tok in tokens:
            if tok in {".", ",", "!", "?", ":", ";", "'"} and reconstructed:
                reconstructed[-1] = f"{reconstructed[-1]}{tok}"
            else:
                reconstructed.append(tok)

        return " ".join(reconstructed)

    def train_from_corpus(
        self,
        corpus: Sequence[str],
        min_freq: int = 1,
    ) -> None:
        """Builds the internal vocabulary from a corpus of texts."""
        tokenized_corpus = [self.tokenize(text) for text in corpus]
        self.vocabulary.build_from_tokens(tokenized_corpus, min_freq=min_freq)

    def save(self, directory: str | Path) -> None:
        """Saves tokenizer configuration and vocabulary files."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        config_data = {
            "lowercase": self.lowercase,
            "vocab_size": len(self.vocabulary),
            "pad_token": self.vocabulary.pad_token,
            "unk_token": self.vocabulary.unk_token,
            "bos_token": self.vocabulary.bos_token,
            "eos_token": self.vocabulary.eos_token,
        }
        with open(dirpath / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        self.vocabulary.save(dirpath / "vocab.json")

    @classmethod
    def load(cls, directory: str | Path) -> "SimpleTokenizer":
        """Loads a SimpleTokenizer from saved directory."""
        dirpath = Path(directory)
        config_path = dirpath / "tokenizer_config.json"
        vocab_path = dirpath / "vocab.json"

        if not config_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(f"Missing tokenizer artifacts in {dirpath}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        vocab = Vocabulary.load(vocab_path)
        tokenizer = cls(
            vocabulary=vocab,
            lowercase=cfg.get("lowercase", True),
            pad_token=cfg.get("pad_token", "<PAD>"),
            unk_token=cfg.get("unk_token", "<UNK>"),
            bos_token=cfg.get("bos_token", "<BOS>"),
            eos_token=cfg.get("eos_token", "<EOS>"),
        )
        return tokenizer
