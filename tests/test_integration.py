"""End-to-end integration tests for Nova agent pipeline."""

from datetime import datetime
import pytest
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.intent_classifier import NeuralIntentClassifier
from nova.agent.router import AgentRouter
from nova.agent.response_generator import ResponseGenerator
from nova.tools.registry import default_registry
from nova.tools.date_time import get_kolkata_timezone
from nova.data.generate_data import INTENTS


def mock_clock() -> datetime:
    tz = get_kolkata_timezone()
    return datetime(2026, 10, 12, 10, 30, 0, tzinfo=tz)


@pytest.fixture
def agent():
    tok = SimpleTokenizer()
    examples = [
        ("what is your name", "ASK_NAME"),
        ("who are you", "ASK_IDENTITY"),
        ("what can you do", "ASK_CAPABILITIES"),
        ("what time is it", "ASK_CURRENT_TIME"),
        ("what is today's date", "ASK_CURRENT_DATE"),
        ("what is the capital of france", "UNKNOWN"),
        ("hello", "GREETING"),
        ("thank you", "THANKS"),
        ("goodbye", "GOODBYE"),
    ]
    tok.train_from_corpus([t for t, _ in examples])

    label_to_idx = {name: i for i, name in enumerate(INTENTS)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    # Train a tiny classifier to map these exact inputs accurately
    classifier = NeuralIntentClassifier(
        vocab_size=tok.vocab_size,
        num_classes=len(INTENTS),
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=16,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    batch_x = []
    batch_y = []
    for text, it in examples:
        ids = tok.encode(text, add_special_tokens=True)
        padded = ids + [tok.pad_token_id] * (16 - len(ids))
        batch_x.append(torch.tensor(padded, dtype=torch.long))
        batch_y.append(label_to_idx[it])

    x_tensor = torch.stack(batch_x)
    y_tensor = torch.tensor(batch_y, dtype=torch.long)

    classifier.train()
    for _ in range(60):
        optimizer.zero_grad()
        _, loss = classifier(x_tensor, labels=y_tensor)
        loss.backward()
        optimizer.step()

    classifier.eval()
    router = AgentRouter(
        classifier=classifier,
        tokenizer=tok,
        idx_to_label=idx_to_label,
        confidence_threshold=0.50,
        response_generator=ResponseGenerator("config/identity.yaml"),
        tool_registry=default_registry,
    )
    return router


def test_agent_identity(agent):
    trace = agent.process("What is your name?")
    assert trace["effective_intent"] == "ASK_NAME"
    assert "Nova" in trace["final_response"]


def test_agent_capabilities(agent):
    trace = agent.process("What can you do?")
    assert trace["effective_intent"] == "ASK_CAPABILITIES"
    assert "capabilities" in trace["final_response"].lower() or "identity" in trace["final_response"].lower()


def test_agent_time_tool(agent):
    trace = agent.process("What time is it?", now_fn=mock_clock)
    assert trace["effective_intent"] == "ASK_CURRENT_TIME"
    assert trace["selected_action"] == "invoke_tool:get_current_time"
    assert "10:30 AM IST" in trace["final_response"]


def test_agent_date_tool(agent):
    trace = agent.process("What is today's date?", now_fn=mock_clock)
    assert trace["effective_intent"] == "ASK_CURRENT_DATE"
    assert trace["selected_action"] == "invoke_tool:get_current_date"
    assert "Monday, 12 October 2026" in trace["final_response"]


def test_agent_unsupported_unknown(agent):
    trace = agent.process("What is the capital of France?")
    assert trace["effective_intent"] == "UNKNOWN"
    assert "general knowledge" in trace["final_response"]
