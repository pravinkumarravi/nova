"""Audit tool for Nova datasets.

Checks split sizes, distribution across intents, vocabulary coverage,
duplicate counts within splits, and data leakage across splits.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {filepath}:{line_num}: {e}")
    return records


def audit_datasets(
    data_dir: str = "data/generated",
) -> Dict[str, Any]:
    """Runs a comprehensive audit on all generated datasets."""
    base = Path(data_dir)
    manifest_file = base / "dataset_manifest.json"
    lm_file = base / "language_model_train.jsonl"
    intent_train_file = base / "intent_train.jsonl"
    intent_val_file = base / "intent_validation.jsonl"
    intent_test_file = base / "intent_test.jsonl"

    files_exist = {
        "manifest": manifest_file.exists(),
        "lm_train": lm_file.exists(),
        "intent_train": intent_train_file.exists(),
        "intent_val": intent_val_file.exists(),
        "intent_test": intent_test_file.exists(),
    }

    if not all(files_exist.values()):
        return {
            "status": "FAILED",
            "error": f"Missing required dataset files: {[k for k, v in files_exist.items() if not v]}",
            "files_exist": files_exist,
        }

    lm_train = load_jsonl(lm_file)
    train_intent = load_jsonl(intent_train_file)
    val_intent = load_jsonl(intent_val_file)
    test_intent = load_jsonl(intent_test_file)

    # Check text field presence
    for split_name, dataset in [("train", train_intent), ("val", val_intent), ("test", test_intent)]:
        for i, item in enumerate(dataset):
            if "text" not in item or "intent" not in item:
                raise ValueError(f"Record {i} in {split_name} missing 'text' or 'intent' field.")

    # Deduplication and leakage checks
    train_texts = [item["text"].strip().lower() for item in train_intent]
    val_texts = [item["text"].strip().lower() for item in val_intent]
    test_texts = [item["text"].strip().lower() for item in test_intent]

    train_unique = set(train_texts)
    val_unique = set(val_texts)
    test_unique = set(test_texts)

    train_val_leakage = train_unique.intersection(val_unique)
    train_test_leakage = train_unique.intersection(test_unique)
    val_test_leakage = val_unique.intersection(test_unique)

    # Token frequency across datasets
    all_tokens: List[str] = []
    for item in train_intent + val_intent + test_intent:
        all_tokens.extend(item["text"].lower().split())
    token_freq = Counter(all_tokens)

    # Intent counts per split
    split_counts: Dict[str, Dict[str, int]] = {}
    for name, data in [("train", train_intent), ("validation", val_intent), ("test", test_intent)]:
        cnt: Dict[str, int] = {}
        for row in data:
            it = row["intent"]
            cnt[it] = cnt.get(it, 0) + 1
        split_counts[name] = cnt

    report: Dict[str, Any] = {
        "status": "PASSED" if not (train_val_leakage or train_test_leakage or val_test_leakage) else "LEAKAGE_DETECTED",
        "counts": {
            "language_model_train": len(lm_train),
            "intent_train": len(train_intent),
            "intent_validation": len(val_intent),
            "intent_test": len(test_intent),
        },
        "duplicates_within_splits": {
            "intent_train": len(train_texts) - len(train_unique),
            "intent_validation": len(val_texts) - len(val_unique),
            "intent_test": len(test_texts) - len(test_unique),
        },
        "data_leakage": {
            "train_val_overlap": len(train_val_leakage),
            "train_test_overlap": len(train_test_leakage),
            "val_test_overlap": len(val_test_leakage),
        },
        "intent_distribution": split_counts,
        "vocabulary_stats": {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(token_freq),
            "top_10_most_common": token_freq.most_common(10),
        },
    }

    return report


if __name__ == "__main__":
    report = audit_datasets()
    print("Dataset Audit Report:")
    print(json.dumps(report, indent=2))
