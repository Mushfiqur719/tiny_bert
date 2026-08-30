# src/explainability/__init__.py
"""Explainability sub-package: SHAP, LIME, and cross-method agreement analysis."""

from src.explainability.agreement import (
    aggregate_agreement_report,
    compute_explanation_agreement,
    compute_token_jaccard,
)
from src.explainability.lime_runner import LimeExplainerRunner
from src.explainability.shap_runner import ShapExplainerRunner
from src.explainability.visualizer import (
    format_side_by_side_markdown,
    highlight_tokens_html,
)

__all__ = [
    "ShapExplainerRunner",
    "LimeExplainerRunner",
    "compute_token_jaccard",
    "compute_explanation_agreement",
    "aggregate_agreement_report",
    "highlight_tokens_html",
    "format_side_by_side_markdown",
]
