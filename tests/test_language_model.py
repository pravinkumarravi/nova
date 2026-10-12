"""Tests for DecoderLanguageModel architecture and training pipeline."""

import pytest
import torch

from nova.models.language_model import DecoderLanguageModel


def test_language_model_forward():
    vocab_size = 50
    d_model = 32
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=d_model, n_layers=2, n_heads=4, max_seq_len=16)

    idx = torch.randint(0, vocab_size, (2, 8))
    targets = torch.randint(0, vocab_size, (2, 8))

    logits, loss = model(idx, targets=targets)

    assert logits.shape == torch.Size([2, 8, vocab_size])
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0


def test_language_model_backward_and_optimization():
    vocab_size = 30
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    idx = torch.randint(0, vocab_size, (1, 6))
    targets = torch.randint(0, vocab_size, (1, 6))

    # Store initial weights
    initial_weight = model.head.weight.clone()

    optimizer.zero_grad()
    _, loss = model(idx, targets=targets)
    loss.backward()

    # Check gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    optimizer.step()

    # Verify weights changed
    assert not torch.equal(model.head.weight, initial_weight)


def test_language_model_overfit_tiny_batch():
    """Educational verification: Model must easily overfit 1 sample."""
    vocab_size = 20
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4, max_seq_len=8, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    initial_loss = None
    final_loss = None

    for step in range(40):
        optimizer.zero_grad()
        _, loss = model(idx, targets=targets)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss
    assert final_loss < 0.2


def test_language_model_autoregressive_generation():
    vocab_size = 25
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2, max_seq_len=16)

    prompt = torch.tensor([[2, 5]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=4, temperature=1.0)

    assert generated.shape == torch.Size([1, 6])
    assert torch.equal(generated[:, :2], prompt)
