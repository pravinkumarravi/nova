"""Tests for PyTorch Dataset implementations."""

from pathlib import Path
import pytest
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.data.dataset import LanguageModelDataset, IntentDataset
from nova.data.generate_data import INTENTS


def test_language_model_dataset():
    tok = SimpleTokenizer()
    tok.train_from_corpus(["User: Hello!\nNova: Hello!"])
    data_path = Path("data/generated/language_model_train.jsonl")

    ds = LanguageModelDataset(data_path, tok, max_seq_len=16)
    assert len(ds) > 0

    x, y = ds[0]
    assert x.shape == torch.Size([16])
    assert y.shape == torch.Size([16])
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # Shift-by-one verification: first token of x is BOS
    assert x[0].item() == tok.bos_token_id


def test_intent_dataset():
    tok = SimpleTokenizer()
    tok.train_from_corpus(["What is your name?", "What time is it?"])
    data_path = Path("data/generated/intent_train.jsonl")

    label_to_idx = {name: i for i, name in enumerate(INTENTS)}
    ds = IntentDataset(data_path, tok, label_to_idx, max_seq_len=20)
    assert len(ds) > 0

    x, y = ds[0]
    assert x.shape == torch.Size([20])
    assert isinstance(y.item(), int)
    assert 0 <= y.item() < len(INTENTS)
