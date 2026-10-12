"""Explicit training pipeline for Nova Decoder-Only Language Model."""

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, random_split

from nova.config import NovaConfig, load_config
from nova.tokenizer.tokenizer import SimpleTokenizer
from nova.data.dataset import LanguageModelDataset
from nova.models.language_model import DecoderLanguageModel
from nova.training.checkpoint import save_checkpoint
from nova.training.metrics import TrainingLogger


def train_language_model(
    config: Optional[NovaConfig] = None,
    epochs_override: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains the DecoderLanguageModel from scratch using explicit PyTorch loops."""
    cfg = config or load_config()

    # 1. Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    lm_cfg = cfg.language_model
    epochs = epochs_override or lm_cfg.epochs
    artifact_dir = Path(cfg.paths["lm_artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg.paths["logs_dir"])

    lm_data_path = Path(cfg.paths["lm_train_data"])
    if not lm_data_path.exists():
        raise FileNotFoundError(f"Training data not found at {lm_data_path}. Run generate-data first.")

    # 2. Build / Train Tokenizer exclusively on training data
    print("Building vocabulary from training dialogues...")
    dialogue_texts = []
    with open(lm_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                dialogue_texts.append(item["text"])

    tokenizer = SimpleTokenizer(
        lowercase=cfg.tokenizer.lowercase,
        pad_token=cfg.tokenizer.pad_token,
        unk_token=cfg.tokenizer.unk_token,
        bos_token=cfg.tokenizer.bos_token,
        eos_token=cfg.tokenizer.eos_token,
    )
    tokenizer.train_from_corpus(dialogue_texts, min_freq=cfg.tokenizer.min_freq)
    tokenizer.save(artifact_dir)
    print(f"Tokenizer saved to {artifact_dir}. Vocabulary size: {tokenizer.vocab_size}")

    # 3. Create Dataset and DataLoader
    full_dataset = LanguageModelDataset(
        file_path=lm_data_path,
        tokenizer=tokenizer,
        max_seq_len=lm_cfg.max_seq_len,
    )

    # 90/10 train/validation split
    n_val = max(1, int(len(full_dataset) * 0.10))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=lm_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=lm_cfg.batch_size, shuffle=False)

    # 4. Instantiate Model
    model = DecoderLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=lm_cfg.d_model,
        n_layers=lm_cfg.n_layers,
        n_heads=lm_cfg.n_heads,
        d_ff=lm_cfg.d_ff,
        max_seq_len=lm_cfg.max_seq_len,
        dropout=lm_cfg.dropout,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    print(f"Initialized DecoderLanguageModel: {model.get_num_params():,} trainable parameters.")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lm_cfg.learning_rate,
        weight_decay=0.01,
    )

    logger = TrainingLogger(logs_dir, "language_model")

    best_val_loss = float("inf")
    start_time = time.time()

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, loss = model(x, targets=y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), lm_cfg.grad_clip)

            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / max(1, train_steps)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, val_loss = model(x, targets=y)
                total_val_loss += val_loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        epoch_duration = time.time() - epoch_start

        logger.log_epoch({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "learning_rate": lm_cfg.learning_rate,
            "duration_sec": round(epoch_duration, 2),
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Time: {epoch_duration:.2f}s"
            )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                save_path=artifact_dir / "best_model.pt",
                model=model,
                config={
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": lm_cfg.d_model,
                    "n_layers": lm_cfg.n_layers,
                    "n_heads": lm_cfg.n_heads,
                    "d_ff": lm_cfg.d_ff,
                    "max_seq_len": lm_cfg.max_seq_len,
                    "dropout": lm_cfg.dropout,
                    "pad_token_id": tokenizer.pad_token_id,
                },
                optimizer=optimizer,
                epoch=epoch,
                metrics={"best_val_loss": best_val_loss},
            )

    # Save final model
    save_checkpoint(
        save_path=artifact_dir / "last_model.pt",
        model=model,
        config={
            "vocab_size": tokenizer.vocab_size,
            "d_model": lm_cfg.d_model,
            "n_layers": lm_cfg.n_layers,
            "n_heads": lm_cfg.n_heads,
            "d_ff": lm_cfg.d_ff,
            "max_seq_len": lm_cfg.max_seq_len,
            "dropout": lm_cfg.dropout,
            "pad_token_id": tokenizer.pad_token_id,
        },
        optimizer=optimizer,
        epoch=epochs,
        metrics={"final_train_loss": avg_train_loss, "final_val_loss": avg_val_loss},
    )

    json_log, csv_log = logger.save()
    total_time = time.time() - start_time

    summary = {
        "status": "COMPLETED",
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": round(avg_train_loss, 4),
        "total_time_seconds": round(total_time, 2),
        "checkpoint_path": str(artifact_dir / "best_model.pt"),
        "history_json": str(json_log),
        "history_csv": str(csv_log),
    }

    print(f"\nLanguage Model training completed in {total_time:.2f}s. Best val loss: {best_val_loss:.4f}")
    return summary


if __name__ == "__main__":
    train_language_model()
