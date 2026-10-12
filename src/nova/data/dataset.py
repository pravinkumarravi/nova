"""PyTorch Dataset implementations for Nova language modeling and intent classification."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """Dataset for next-token causal language modeling.
    
    Transforms each text string into input token IDs and target token IDs shifted by one token.
    For input [BOS, w1, w2, w3], the target is [w1, w2, w3, EOS].
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 64,
    ) -> None:
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[str] = []

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.examples.append(item["text"])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.examples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Build full sequence [BOS] + tokens + [EOS]
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        full_seq = [bos_id] + tokens + [eos_id]

        # Truncate to max_seq_len + 1 so x and y have max_seq_len
        if len(full_seq) > self.max_seq_len + 1:
            full_seq = full_seq[: self.max_seq_len + 1]

        # x: from 0 to len-1, y: from 1 to len
        x_ids = full_seq[:-1]
        y_ids = full_seq[1:]

        # Pad to max_seq_len
        seq_len = len(x_ids)
        if seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - seq_len
            x_ids = x_ids + [pad_id] * padding_len
            y_ids = y_ids + [pad_id] * padding_len

        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_ids, dtype=torch.long),
        )


class IntentDataset(Dataset):
    """Dataset for neural intent classification.
    
    Transforms text input into padded token IDs and integer intent labels.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Any,
        label_to_idx: Dict[str, int],
        max_seq_len: int = 32,
    ) -> None:
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, int]] = []

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item["text"]
                    intent_str = item["intent"]
                    if intent_str not in self.label_to_idx:
                        raise ValueError(f"Unknown intent: {intent_str}")
                    self.samples.append((text, self.label_to_idx[intent_str]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label = self.samples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        pad_id = self.tokenizer.pad_token_id
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        else:
            tokens = tokens + [pad_id] * (self.max_seq_len - len(tokens))

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
