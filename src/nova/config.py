"""Configuration loader and settings dataclasses for Nova."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class TokenizerConfig:
    pad_token: str = "<PAD>"
    unk_token: "<UNK>" = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    min_freq: int = 1
    lowercase: bool = True


@dataclass
class LMConfig:
    vocab_size: Optional[int] = None
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 64
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 35
    grad_clip: float = 1.0


@dataclass
class IntentConfig:
    vocab_size: Optional[int] = None
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 32
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 30
    grad_clip: float = 1.0
    confidence_threshold: float = 0.65


@dataclass
class AgentConfig:
    identity_config: str = "config/identity.yaml"
    confidence_threshold: float = 0.65
    timezone: str = "Asia/Kolkata"


@dataclass
class NovaConfig:
    seed: int = 42
    paths: Dict[str, str] = field(default_factory=lambda: {
        "data_dir": "data/generated",
        "artifacts_dir": "artifacts",
        "logs_dir": "logs",
        "lm_train_data": "data/generated/language_model_train.jsonl",
        "intent_train_data": "data/generated/intent_train.jsonl",
        "intent_val_data": "data/generated/intent_validation.jsonl",
        "intent_test_data": "data/generated/intent_test.jsonl",
        "manifest_path": "data/generated/dataset_manifest.json",
        "lm_artifact_dir": "artifacts/language_model",
        "intent_artifact_dir": "artifacts/intent_classifier",
    })
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    language_model: LMConfig = field(default_factory=LMConfig)
    intent_classifier: IntentConfig = field(default_factory=IntentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Safely loads a YAML file and returns dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path = "config/default.yaml") -> NovaConfig:
    """Loads NovaConfig from YAML file with fallback to defaults."""
    path = Path(config_path)
    if not path.exists():
        return NovaConfig()

    raw = load_yaml(path)
    paths_dict = raw.get("paths", {})
    tok_dict = raw.get("tokenizer", {})
    lm_dict = raw.get("language_model", {})
    intent_dict = raw.get("intent_classifier", {})
    agent_dict = raw.get("agent", {})

    cfg = NovaConfig(
        seed=raw.get("seed", 42),
        paths=paths_dict,
        tokenizer=TokenizerConfig(**tok_dict),
        language_model=LMConfig(**lm_dict),
        intent_classifier=IntentConfig(**intent_dict),
        agent=AgentConfig(**agent_dict),
    )
    return cfg


def load_identity(identity_path: str | Path = "config/identity.yaml") -> Dict[str, Any]:
    """Loads identity and canned response configurations."""
    path = Path(identity_path)
    if not path.exists():
        raise FileNotFoundError(f"Identity file not found at {path}")
    return load_yaml(path)
