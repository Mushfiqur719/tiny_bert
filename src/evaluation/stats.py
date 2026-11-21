"""
src/evaluation/stats.py — Statistical helpers for robust model comparison.

  - bootstrap_ci:        percentile bootstrap CI for a single model's metric.
  - paired_bootstrap:    paired bootstrap CI for the metric difference between two models
                         evaluated on the SAME test items.
  - mean_std:            mean / sample std across seeds.
  - holm_correction:     Holm-Bonferroni adjusted p-values for multiple comparisons.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y_true, y_pred), float(lo), float(hi)


def paired_bootstrap(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of metric(A) - metric(B) on identical test items.

    Returns dict with delta, ci_low, ci_high and a two-sided bootstrap p-value
    (fraction of resamples where the sign of the difference flips, doubled).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    delta = metric(y_true, pred_a) - metric(y_true, pred_b)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return {
        "delta": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * p_one)),
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running_max = max(running_max, val)
        adjusted[i] = min(1.0, running_max)
    return adjusted.tolist()
