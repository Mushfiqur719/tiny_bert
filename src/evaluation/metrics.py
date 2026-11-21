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


def find_optimal_threshold(
    logits: np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    low: float = 0.20,
    high: float = 0.80,
    step: float = 0.01,
) -> tuple[float, float]:
    """Find the probability threshold for class 1 that maximizes Macro-F1 on validation logits.

    Args:
        logits: Raw model logits of shape [N, 2].
        y_true: Ground-truth binary labels of shape [N].
        low: Minimum candidate threshold.
        high: Maximum candidate threshold.
        step: Grid step size.

    Returns:
        Tuple of (best_threshold, best_macro_f1).
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true)

    # Numerically stable softmax for class 1 probability
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs_pos = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

    best_thresh = 0.50
    best_f1 = float(f1_score(y_true, (probs_pos >= 0.50).astype(int), average="macro", zero_division=0))

    thresholds = np.arange(low, high + step / 2.0, step)
    for t in thresholds:
        preds = (probs_pos >= t).astype(int)
        score = float(f1_score(y_true, preds, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thresh = float(round(t, 4))

    return best_thresh, round(best_f1, 4)


def predict_with_threshold(logits: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    """Convert [N, 2] logits into binary predictions using a probability threshold for class 1."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs_pos = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
    return (probs_pos >= threshold).astype(int)
