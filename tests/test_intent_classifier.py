"""Tests for NeuralIntentClassifier and Baseline Matcher."""

import pytest
import torch

from nova.models.intent_classifier import NeuralIntentClassifier, RuleBasedBaselineClassifier


def test_intent_classifier_forward_and_loss():
    vocab_size = 40
    num_classes = 12
    model = NeuralIntentClassifier(vocab_size=vocab_size, num_classes=num_classes, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)

    idx = torch.randint(0, vocab_size, (2, 10))
    labels = torch.tensor([1, 5], dtype=torch.long)

    logits, loss = model(idx, labels=labels)
    assert logits.shape == torch.Size([2, num_classes])
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0


def test_intent_classifier_predict():
    vocab_size = 40
    num_classes = 12
    model = NeuralIntentClassifier(vocab_size=vocab_size, num_classes=num_classes, d_model=32, n_layers=1, n_heads=2, max_seq_len=8)

    idx = torch.randint(0, vocab_size, (2, 8))
    pred_idx, conf = model.predict(idx)

    assert pred_idx.shape == torch.Size([2])
    assert conf.shape == torch.Size([2])
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_intent_classifier_overfit():
    vocab_size = 20
    num_classes = 3
    model = NeuralIntentClassifier(vocab_size=vocab_size, num_classes=num_classes, d_model=32, n_layers=1, n_heads=2, max_seq_len=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    idx = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([0, 2], dtype=torch.long)

    for _ in range(25):
        optimizer.zero_grad()
        _, loss = model(idx, labels=labels)
        loss.backward()
        optimizer.step()

    pred, _ = model.predict(idx)
    assert torch.equal(pred, labels)


def test_baseline_classifier():
    baseline = RuleBasedBaselineClassifier()
    intent, conf = baseline.predict("what time is it")
    assert intent == "ASK_CURRENT_TIME"

    intent, conf = baseline.predict("what is your name")
    assert intent == "ASK_NAME"

    intent, conf = baseline.predict("how do airplanes fly")
    assert intent == "UNKNOWN"
