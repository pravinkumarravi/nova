"""Checkpoint management for Nova models."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    save_path: str | Path,
    model: torch.nn.Module,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves complete model checkpoint with state dict, hyperparams, and metadata."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics or {},
        "extra_metadata": extra_metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint and verifies essential keys exist."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # Use weights_only=False to allow dictionary payloads
    try:
        data = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {path}: {e}")

    required_keys = ["model_state_dict", "config"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Corrupted checkpoint: missing keys {missing} in {path}")

    return data
