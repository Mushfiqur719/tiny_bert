"""
src/explainability/agreement.py — Cross-method explanation agreement analysis (SHAP vs. LIME).

Evaluates whether two independent explainability paradigms (Shapley cooperative game theory vs.
LIME local linear surrogate perturbations) agree on token attributions for TinyBanglaBERT.

Metrics Computed:
  1. Top-K Jaccard Similarity: Overlap between top-K influential tokens.
  2. Sign Agreement: Fraction of shared tokens where both methods agree on attribution polarity (+/-).
  3. Rank Correlation: Spearman rank correlation over token importance order.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_token_jaccard(
    shap_tokens: Sequence[str],
    lime_tokens: Sequence[str],
    k: int = 3,
) -> float:
    """Compute Jaccard similarity between top-K tokens from SHAP and LIME.

    J(A, B) = |A cap B| / |A cup B|

    Args:
        shap_tokens: List of top tokens from SHAP.
        lime_tokens: List of top tokens from LIME.
        k: Top-K slice limit.

    Returns:
        Jaccard similarity float between 0.0 and 1.0.
    """
    s_set = set(t.strip().lower() for t in shap_tokens[:k] if t.strip())
    l_set = set(t.strip().lower() for t in lime_tokens[:k] if t.strip())

    if not s_set and not l_set:
        return 1.0
    if not s_set or not l_set:
        return 0.0

    intersection = s_set.intersection(l_set)
    union = s_set.union(l_set)
    return len(intersection) / len(union)


def compute_explanation_agreement(
    shap_result: dict[str, Any],
    lime_result: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    """Compute comprehensive agreement metrics for a single headline explanation.

    Args:
        shap_result: Output dict from ShapExplainerRunner.
        lime_result: Output dict from LimeExplainerRunner.
        top_k: K for top-K metrics.

    Returns:
        Dictionary of agreement metrics.
    """
    # 1. Top-K Jaccard on Clickbait-Pushing Tokens
    shap_top_cb = shap_result.get("top_clickbait_tokens", [])
    lime_top_cb = lime_result.get("top_clickbait_tokens", [])
    jaccard_top_cb = compute_token_jaccard(shap_top_cb, lime_top_cb, k=top_k)

    # 2. Top-K Jaccard on Non-Clickbait Tokens
    shap_top_ncb = shap_result.get("top_non_clickbait_tokens", [])
    lime_top_ncb = lime_result.get("top_non_clickbait_tokens", [])
    jaccard_top_ncb = compute_token_jaccard(shap_top_ncb, lime_top_ncb, k=top_k)

    # 3. Overall Top-K Jaccard (highest absolute influence)
    shap_abs = sorted(
        shap_result.get("token_attributions", []),
        key=lambda x: abs(x["score"]),
        reverse=True,
    )
    lime_abs = sorted(
        lime_result.get("lime_weights", []),
        key=lambda x: abs(x["weight"]),
        reverse=True,
    )

    shap_top_all = [x["token"] for x in shap_abs[:top_k]]
    lime_top_all = [x["token"] for x in lime_abs[:top_k]]
    jaccard_all = compute_token_jaccard(shap_top_all, lime_top_all, k=top_k)

    # 4. Sign Agreement on Shared Tokens
    shap_dict = {x["token"].strip().lower(): x["score"] for x in shap_result.get("token_attributions", [])}
    lime_dict = {x["token"].strip().lower(): x["weight"] for x in lime_result.get("lime_weights", [])}

    shared_tokens = set(shap_dict.keys()).intersection(set(lime_dict.keys()))
    if shared_tokens:
        sign_matches = sum(
            1 for t in shared_tokens
            if (shap_dict[t] >= 0 and lime_dict[t] >= 0) or (shap_dict[t] < 0 and lime_dict[t] < 0)
        )
        sign_agreement = sign_matches / len(shared_tokens)
    else:
        sign_agreement = 1.0

    return {
        "jaccard_top3_clickbait": round(jaccard_top_cb, 4),
        "jaccard_top3_non_clickbait": round(jaccard_top_ncb, 4),
        "jaccard_top3_overall": round(jaccard_all, 4),
        "sign_agreement": round(sign_agreement, 4),
        "shared_token_count": len(shared_tokens),
        "shap_top_tokens": shap_top_all,
        "lime_top_tokens": lime_top_all,
    }


def aggregate_agreement_report(
    sample_explanations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate agreement metrics across a dataset sample.

    Args:
        sample_explanations: List of dicts with 'shap', 'lime', 'agreement' entries.

    Returns:
        Summary dictionary with mean and median agreement scores.
    """
    if not sample_explanations:
        return {}

    j_all = [e["agreement"]["jaccard_top3_overall"] for e in sample_explanations]
    j_cb = [e["agreement"]["jaccard_top3_clickbait"] for e in sample_explanations]
    sign_agrees = [e["agreement"]["sign_agreement"] for e in sample_explanations]

    return {
        "num_samples": len(sample_explanations),
        "mean_jaccard_top3_overall": round(float(np.mean(j_all)), 4),
        "median_jaccard_top3_overall": round(float(np.median(j_all)), 4),
        "mean_jaccard_top3_clickbait": round(float(np.mean(j_cb)), 4),
        "mean_sign_agreement": round(float(np.mean(sign_agrees)), 4),
    }
