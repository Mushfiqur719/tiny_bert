"""
src/evaluation/metrics.py — Evaluation metrics for Bangla clickbait detection.

Computes:
  - Accuracy
  - Precision (macro, binary/clickbait, not-clickbait)
  - Recall (macro, binary/clickbait, not-clickbait)
  - F1 (macro, binary/clickbait, not-clickbait)
  - Confusion Matrix [[TN, FP], [FN, TP]]

Usage:
    from src.evaluation.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_pred, y_true)
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_pred: Sequence[int] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute binary and macro classification metrics.

    Args:
        y_pred: Predicted class indices (0 or 1).
        y_true: Ground truth labels (0 or 1).

    Returns:
        Dictionary containing:
          - accuracy
          - f1_macro, precision_macro, recall_macro
          - f1_clickbait, precision_clickbait, recall_clickbait (class 1)
          - f1_not_clickbait, precision_not_clickbait, recall_not_clickbait (class 0)
          - confusion_matrix: 2x2 list [[TN, FP], [FN, TP]]
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro averages
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class scores (labels [0, 1])
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_not_clickbait": round(float(f1_per_class[0]), 4),
        "precision_not_clickbait": round(float(prec_per_class[0]), 4),
        "recall_not_clickbait": round(float(rec_per_class[0]), 4),
        "f1_clickbait": round(float(f1_per_class[1]), 4),
        "precision_clickbait": round(float(prec_per_class[1]), 4),
        "recall_clickbait": round(float(rec_per_class[1]), 4),
        "confusion_matrix": cm,
    }


def compute_metrics_from_eval_pred(eval_pred) -> dict[str, float]:
    """Adapter for HuggingFace Trainer compute_metrics callback.

    Args:
        eval_pred: EvalPrediction object or (predictions, label_ids) tuple.

    Returns:
        Flat dict of metric names to float values.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        # Model may return (logits, hidden_states, ...)
        predictions = predictions[0]

    y_pred = np.argmax(predictions, axis=1)
    metrics = compute_classification_metrics(y_pred=y_pred, y_true=labels)

    # Return scalar metrics (omit nested confusion_matrix for Trainer logs)
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
