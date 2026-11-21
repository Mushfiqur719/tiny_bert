"""tests/test_metric_consistency.py — Verifies mathematical consistency of stored metrics.

Recomputes accuracy, per-class precision/recall/F1, and macro averages
directly from the recorded confusion matrix in every metrics.json file.
Ensures zero drift, reporting mismatch, or inconsistent numbers in results.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def recompute_from_confusion_matrix(cm: list[list[int]]) -> dict[str, float]:
    """Recompute all classification metrics directly from 2x2 confusion matrix."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Class 0: not-clickbait
    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0.0

    # Class 1: clickbait
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

    # Macro averages
    macro_p = (p0 + p1) / 2.0
    macro_r = (r0 + r1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_not_clickbait": p0,
        "recall_not_clickbait": r0,
        "f1_not_clickbait": f1_0,
        "precision_clickbait": p1,
        "recall_clickbait": r1,
        "f1_clickbait": f1_1,
    }


def find_metric_files() -> list[Path]:
    """Discover all JSON metric files with confusion matrices in experiments/."""
    base_dir = Path("experiments")
    if not base_dir.exists():
        return []

    files = []
    for p in base_dir.rglob("*.json"):
        if "checkpoint" in str(p) or "wandb" in str(p):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and ("confusion_matrix" in data or "calibrated" in data):
                files.append(p)
        except Exception:
            continue
    return files


@pytest.mark.parametrize("metric_path", find_metric_files(), ids=lambda p: str(p))
def test_metric_mathematical_consistency(metric_path: Path):
    """Verify that stored metrics match recomputed values from the confusion matrix."""
    data = json.loads(metric_path.read_text(encoding="utf-8"))

    # Check uncalibrated / base metrics if present
    if "confusion_matrix" in data and isinstance(data["confusion_matrix"], list):
        cm = data["confusion_matrix"]
        if len(cm) == 2 and len(cm[0]) == 2:
            recomputed = recompute_from_confusion_matrix(cm)
            for k, val in recomputed.items():
                if k in data:
                    assert abs(data[k] - val) < 1e-3, (
                        f"Mismatch in {metric_path} for key '{k}': "
                        f"stored={data[k]}, recomputed={val}"
                    )

    # Check calibrated metrics if present
    if "calibrated" in data and isinstance(data["calibrated"], dict):
        cal = data["calibrated"]
        if "confusion_matrix" in cal and isinstance(cal["confusion_matrix"], list):
            cm = cal["confusion_matrix"]
            if len(cm) == 2 and len(cm[0]) == 2:
                recomputed = recompute_from_confusion_matrix(cm)
                for k, val in recomputed.items():
                    if k in cal:
                        assert abs(cal[k] - val) < 1e-3, (
                            f"Mismatch in {metric_path} (calibrated) for key '{k}': "
                            f"stored={cal[k]}, recomputed={val}"
                        )
