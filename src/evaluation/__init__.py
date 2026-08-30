# src/evaluation/__init__.py
"""Evaluation sub-package: metrics, significance testing, and cross-domain eval."""

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
)

__all__ = [
    "compute_classification_metrics",
    "compute_metrics_from_eval_pred",
]
