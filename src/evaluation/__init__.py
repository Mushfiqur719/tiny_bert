# src/evaluation/__init__.py
"""Evaluation sub-package: metrics, ablation analysis, significance testing, and cross-domain eval."""

from src.evaluation.ablation import (
    compile_ablation_dataframe,
    load_run_metrics,
    plot_f1_comparison,
    run_ablation_analysis,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
)

__all__ = [
    "compute_classification_metrics",
    "compute_metrics_from_eval_pred",
    "load_run_metrics",
    "compile_ablation_dataframe",
    "plot_f1_comparison",
    "run_ablation_analysis",
]
