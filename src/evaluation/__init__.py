# src/evaluation/__init__.py
"""Evaluation sub-package: metrics, ablation analysis, McNemar significance testing, and cross-domain eval."""

from src.evaluation.ablation import (
    compile_ablation_dataframe,
    load_run_metrics,
    plot_f1_comparison,
    run_ablation_analysis,
)
from src.evaluation.mcnemar_test import (
    build_contingency_table,
    get_model_test_predictions,
    run_full_mcnemar_suite,
    run_pairwise_mcnemar_test,
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
    "build_contingency_table",
    "run_pairwise_mcnemar_test",
    "get_model_test_predictions",
    "run_full_mcnemar_suite",
]
