"""Tests for AgentRouter and confidence management."""

from datetime import datetime
import pytest
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.intent_classifier import NeuralIntentClassifier
from nova.agent.router import AgentRouter
from nova.agent.confidence import evaluate_confidence
from nova.data.generate_data import INTENTS
from nova.tools.date_time import get_kolkata_timezone


def mock_clock() -> datetime:
    tz = get_kolkata_timezone()
    return datetime(2026, 10, 12, 10, 30, 0, tzinfo=tz)


def test_confidence_evaluation():
    # Above threshold
    eff, passed, meta = evaluate_confidence("GREETING", 0.85, threshold=0.65)
    assert passed is True
    assert eff == "GREETING"

    # Below threshold
    eff, passed, meta = evaluate_confidence("GREETING", 0.40, threshold=0.65)
    assert passed is False
    assert eff == "UNKNOWN"


def test_router_pipeline_mock_run():
    tok = SimpleTokenizer()
    tok.train_from_corpus(["what time is it", "who are you"])
    label_to_idx = {name: i for i, name in enumerate(INTENTS)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    classifier = NeuralIntentClassifier(
        vocab_size=tok.vocab_size,
        num_classes=len(INTENTS),
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_seq_len=16,
    )

    router = AgentRouter(
        classifier=classifier,
        tokenizer=tok,
        idx_to_label=idx_to_label,
        confidence_threshold=0.50,
    )

    trace = router.process("what time is it", now_fn=mock_clock)
    assert "original_input" in trace
    assert "normalized_input" in trace
    assert "tokens" in trace
    assert "predicted_intent" in trace
    assert "final_response" in trace
