"""Model artifact loading utilities for Nova."""

import json
from pathlib import Path
from typing import Tuple
import torch

from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.models.language_model import DecoderLanguageModel
from nova.models.intent_classifier import NeuralIntentClassifier
from nova.agent.router import AgentRouter
from nova.training.checkpoint import load_checkpoint


def load_language_model(
    artifact_dir: str | Path = "artifacts/language_model",
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
) -> Tuple[DecoderLanguageModel, SimpleTokenizer]:
    """Loads trained DecoderLanguageModel and its associated tokenizer."""
    dirpath = Path(artifact_dir)
    ckpt_path = dirpath / checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the LM first.")

    tokenizer = SimpleTokenizer.load(dirpath)
    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt["config"]

    model = DecoderLanguageModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.1),
        pad_token_id=cfg.get("pad_token_id", tokenizer.pad_token_id),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer


def load_agent_router(
    artifact_dir: str | Path = "artifacts/intent_classifier",
    checkpoint_name: str = "best_model.pt",
    confidence_threshold: float = 0.65,
    device: str = "cpu",
) -> AgentRouter:
    """Loads trained NeuralIntentClassifier, tokenizer, labels, and returns configured AgentRouter."""
    dirpath = Path(artifact_dir)
    ckpt_path = dirpath / checkpoint_name
    labels_path = dirpath / "labels.json"

    if not ckpt_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing required artifacts in {dirpath}. Train the intent classifier first."
        )

    tokenizer = SimpleTokenizer.load(dirpath)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    label_to_idx = label_data["label_to_idx"]
    idx_to_label = {int(idx): name for name, idx in label_to_idx.items()}

    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt["config"]

    classifier = NeuralIntentClassifier(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.1),
        pad_token_id=cfg.get("pad_token_id", tokenizer.pad_token_id),
    ).to(device)

    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()

    router = AgentRouter(
        classifier=classifier,
        tokenizer=tokenizer,
        idx_to_label=idx_to_label,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    return router
