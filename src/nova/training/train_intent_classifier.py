"""Explicit training and evaluation pipeline for Nova Neural Intent Classifier."""

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from nova.config import NovaConfig, load_config
from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.data.dataset import IntentDataset
from nova.data.generate_data import INTENTS
from nova.models.intent_classifier import NeuralIntentClassifier
from nova.training.checkpoint import save_checkpoint
from nova.training.metrics import compute_classification_metrics, TrainingLogger


def train_intent_classifier(
    config: Optional[NovaConfig] = None,
    epochs_override: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains the NeuralIntentClassifier from scratch on the synthetic intent corpus."""
    cfg = config or load_config()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    intent_cfg = cfg.intent_classifier
    epochs = epochs_override or intent_cfg.epochs
    artifact_dir = Path(cfg.paths["intent_artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg.paths["logs_dir"])

    train_path = Path(cfg.paths["intent_train_data"])
    val_path = Path(cfg.paths["intent_val_data"])
    test_path = Path(cfg.paths["intent_test_data"])

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found at {p}. Run generate-data first.")

    # 1. Label to Index Mapping
    label_to_idx = {name: idx for idx, name in enumerate(INTENTS)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}

    with open(artifact_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "intents": INTENTS}, f, indent=2)

    # 2. Build Tokenizer on Training split ONLY
    train_texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_texts.append(json.loads(line)["text"])

    tokenizer = SimpleTokenizer(
        lowercase=cfg.tokenizer.lowercase,
        pad_token=cfg.tokenizer.pad_token,
        unk_token=cfg.tokenizer.unk_token,
        bos_token=cfg.tokenizer.bos_token,
        eos_token=cfg.tokenizer.eos_token,
    )
    tokenizer.train_from_corpus(train_texts, min_freq=cfg.tokenizer.min_freq)
    tokenizer.save(artifact_dir)
    print(f"Intent Tokenizer saved to {artifact_dir}. Vocab size: {tokenizer.vocab_size}")

    # 3. Create Datasets and DataLoaders
    train_dataset = IntentDataset(train_path, tokenizer, label_to_idx, max_seq_len=intent_cfg.max_seq_len)
    val_dataset = IntentDataset(val_path, tokenizer, label_to_idx, max_seq_len=intent_cfg.max_seq_len)
    test_dataset = IntentDataset(test_path, tokenizer, label_to_idx, max_seq_len=intent_cfg.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=intent_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=intent_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=intent_cfg.batch_size, shuffle=False)

    # 4. Initialize Neural Intent Classifier
    model = NeuralIntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(INTENTS),
        d_model=intent_cfg.d_model,
        n_layers=intent_cfg.n_layers,
        n_heads=intent_cfg.n_heads,
        d_ff=intent_cfg.d_ff,
        max_seq_len=intent_cfg.max_seq_len,
        dropout=intent_cfg.dropout,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=intent_cfg.learning_rate,
        weight_decay=0.01,
    )

    logger = TrainingLogger(logs_dir, "intent_classifier")
    best_val_f1 = -1.0
    start_time = time.time()

    print(f"Training NeuralIntentClassifier for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0.0
        steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, loss = model(x, labels=y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), intent_cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_train_loss = total_loss / max(1, steps)

        # Validation evaluation
        model.eval()
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(y.cpu().tolist())

        val_metrics = compute_classification_metrics(val_targets, val_preds, INTENTS)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        epoch_duration = time.time() - epoch_start

        logger.log_epoch({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "duration_sec": round(epoch_duration, 2),
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val Macro-F1: {val_f1:.4f} | "
                f"Time: {epoch_duration:.2f}s"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                save_path=artifact_dir / "best_model.pt",
                model=model,
                config={
                    "vocab_size": tokenizer.vocab_size,
                    "num_classes": len(INTENTS),
                    "d_model": intent_cfg.d_model,
                    "n_layers": intent_cfg.n_layers,
                    "n_heads": intent_cfg.n_heads,
                    "d_ff": intent_cfg.d_ff,
                    "max_seq_len": intent_cfg.max_seq_len,
                    "dropout": intent_cfg.dropout,
                    "pad_token_id": tokenizer.pad_token_id,
                },
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_accuracy": val_acc, "val_macro_f1": val_f1},
                extra_metadata={"label_to_idx": label_to_idx},
            )

    # Final evaluation on held-out test dataset
    model.eval()
    test_preds: List[int] = []
    test_targets: List[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            test_preds.extend(preds.cpu().tolist())
            test_targets.extend(y.cpu().tolist())

    test_metrics = compute_classification_metrics(test_targets, test_preds, INTENTS)

    with open(artifact_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    json_log, csv_log = logger.save()
    total_time = time.time() - start_time

    print(f"\nIntent Classifier training complete in {total_time:.2f}s.")
    print(f"Test Set Accuracy: {test_metrics['accuracy'] * 100:.2f}% | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    return {
        "status": "COMPLETED",
        "best_val_f1": round(best_val_f1, 4),
        "test_metrics": test_metrics,
        "checkpoint_path": str(artifact_dir / "best_model.pt"),
        "evaluation_report": str(artifact_dir / "evaluation_report.json"),
        "history_json": str(json_log),
        "history_csv": str(csv_log),
    }


if __name__ == "__main__":
    train_intent_classifier()
