"""Evaluation metrics and training history logging for Nova."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
) -> Dict[str, Any]:
    """Computes accuracy, macro precision, recall, F1, per-class metrics, and confusion matrix."""
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    n_classes = len(class_names)
    total = len(y_true)

    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    # Confusion matrix: rows = true label, cols = predicted label
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    correct = 0

    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
            if t == p:
                correct += 1

    accuracy = correct / total

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(n_classes):
        name = class_names[c]
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes) if r != c)
        fn = sum(cm[c][col] for col in range(n_classes) if col != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class_metrics[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    macro_precision = sum(precisions) / n_classes if n_classes > 0 else 0.0
    macro_recall = sum(recalls) / n_classes if n_classes > 0 else 0.0
    macro_f1 = sum(f1s) / n_classes if n_classes > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


class TrainingLogger:
    """Logs training metrics across epochs and saves them to JSON and CSV."""

    def __init__(self, log_dir: str | Path, run_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch_data: Dict[str, Any]) -> None:
        self.history.append(epoch_data)

    def save(self) -> Tuple[Path, Path]:
        json_path = self.log_dir / f"{self.run_name}_history.json"
        csv_path = self.log_dir / f"{self.run_name}_history.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self.history:
            keys = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.history:
                    # Flatten complex fields for CSV if any
                    clean_row = {
                        k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in row.items()
                    }
                    writer.writerow(clean_row)

        return json_path, csv_path
