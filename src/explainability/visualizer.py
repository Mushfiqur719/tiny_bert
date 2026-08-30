"""
src/explainability/visualizer.py — HTML & Markdown visual rendering for SHAP and LIME token attributions.

Color Scheme:
  - Red / Coral (rgba(231, 76, 60, alpha)): Positive attribution pushing model toward 'clickbait'.
  - Blue / Teal (rgba(52, 152, 219, alpha)): Negative attribution pushing model toward 'not-clickbait'.
"""

from __future__ import annotations

import html
from typing import Any, Sequence


def highlight_tokens_html(
    token_score_pairs: Sequence[tuple[str, float]],
    max_score: float | None = None,
) -> str:
    """Generate HTML text with color-intensity highlights for token attributions.

    Args:
        token_score_pairs: List of (token_str, score_float).
        max_score: Optional maximum absolute score for normalization.

    Returns:
        HTML snippet string.
    """
    if not token_score_pairs:
        return ""

    if max_score is None or max_score <= 0:
        max_score = max(abs(s) for _, s in token_score_pairs) or 1.0

    html_parts = []
    for token, score in token_score_pairs:
        escaped_tok = html.escape(token)
        norm_intensity = min(1.0, abs(score) / max_score)
        alpha = 0.15 + (0.70 * norm_intensity)  # Range 0.15 to 0.85

        if score > 0:
            # Positive -> Clickbait (Red/Coral)
            color_style = f"background-color: rgba(231, 76, 60, {alpha:.2f}); color: #1a1a1a; font-weight: bold;"
            title = f"Clickbait attribution: +{score:.4f}"
        elif score < 0:
            # Negative -> Not-Clickbait (Blue/Azure)
            color_style = f"background-color: rgba(52, 152, 219, {alpha:.2f}); color: #1a1a1a;"
            title = f"Not-Clickbait attribution: {score:.4f}"
        else:
            color_style = "background-color: transparent;"
            title = "Neutral"

        span = (
            f'<span style="{color_style} padding: 2px 4px; margin: 1px; border-radius: 3px;" '
            f'title="{title}">{escaped_tok}</span>'
        )
        html_parts.append(span)

    return " ".join(html_parts)


def format_side_by_side_markdown(
    headline_idx: int,
    headline: str,
    true_label: int,
    pred_label: int,
    shap_res: dict[str, Any],
    lime_res: dict[str, Any],
    agreement: dict[str, Any],
) -> str:
    """Format a single headline explanation as a detailed markdown report section."""
    true_name = "clickbait" if true_label == 1 else "not-clickbait"
    pred_name = "clickbait" if pred_label == 1 else "not-clickbait"
    is_correct = "✅ Correct" if true_label == pred_label else "❌ Misclassified"

    # SHAP highlight
    shap_pairs = [(x["token"], x["score"]) for x in shap_res.get("token_attributions", [])]
    shap_html = highlight_tokens_html(shap_pairs)

    # LIME highlight
    lime_pairs = [(x["token"], x["weight"]) for x in lime_res.get("lime_weights", [])]
    lime_html = highlight_tokens_html(lime_pairs)

    # Top-3 summary badges
    shap_top_str = ", ".join(f"`{t}`" for t in shap_res.get("top_clickbait_tokens", [])[:3]) or "None"
    lime_top_str = ", ".join(f"`{t}`" for t in lime_res.get("top_clickbait_tokens", [])[:3]) or "None"

    lines = [
        f"### Headline #{headline_idx}: {is_correct}",
        f"**Headline Text:** *{headline}*",
        "",
        f"- **Ground Truth:** `{true_name}` (Label: {true_label})",
        f"- **TinyBanglaBERT Prediction:** `{pred_name}` (Clickbait Prob: `{shap_res['clickbait_prob']:.2%}`)",
        f"- **Method Agreement (Top-3 Jaccard):** **`{agreement['jaccard_top3_overall']:.2f}`** | **Sign Agreement:** `{agreement['sign_agreement']:.0%}`",
        "",
        "#### 1. SHAP Token Attributions (Shapley Values)",
        f"> {shap_html}",
        f"*Top Clickbait Tokens (SHAP):* {shap_top_str}",
        "",
        "#### 2. LIME Local Linear Perturbations",
        f"> {lime_html}",
        f"*Top Clickbait Tokens (LIME):* {lime_top_str}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)
