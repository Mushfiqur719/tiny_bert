"""
src/explainability/shap_runner.py — SHAP (SHapley Additive exPlanations) for TinyBanglaBERT.

Provides per-token attribution scores using Shapley values computed via SHAP's Partition/Text explainer.
Identifies which tokens push the model's prediction toward 'clickbait' (positive Shapley value)
versus 'not-clickbait' (negative Shapley value).

Usage:
    from src.explainability.shap_runner import ShapExplainerRunner
    runner = ShapExplainerRunner(model, tokenizer)
    explanation = runner.explain_headline("চমকপ্রদ খবর দেখুন ভিডিওতে এখনই!")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import shap
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ShapExplainerRunner:
    """SHAP Explainer wrapper for Bangla transformer text classification models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | None = None,
        max_seq_len: int = 128,
    ) -> None:
        """
        Args:
            model: PyTorch / HuggingFace Sequence Classification Model.
            tokenizer: Tokenizer for text encoding.
            device: CPU or CUDA device (defaults to CPU for consistent explainer execution).
            max_seq_len: Maximum token sequence length.
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Define predict_proba function for SHAP
        self.predict_fn = self._build_predict_fn()

        # Build SHAP Explainer with Regex word/whitespace masker for Bangla text
        self.masker = shap.maskers.Text(r"\s+")
        self.explainer = shap.Explainer(
            self.predict_fn,
            masker=self.masker,
            output_names=["not-clickbait", "clickbait"],
        )

    def _build_predict_fn(self) -> Callable[[Sequence[str]], np.ndarray]:
        """Build prediction probability function for string arrays."""
        def predict_proba(texts: Sequence[str]) -> np.ndarray:
            text_list = list(texts)
            if not text_list:
                return np.zeros((0, 2))

            inputs = self.tokenizer(
                text_list,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            return probs

        return predict_proba

    def explain_headline(self, headline: str) -> dict[str, Any]:
        """Compute token-level SHAP values for a single Bangla headline.

        Args:
            headline: Input Bangla text string.

        Returns:
            Dictionary containing:
              - 'headline': Input text
              - 'pred_label': Predicted class (0 or 1)
              - 'pred_class_name': 'not-clickbait' or 'clickbait'
              - 'pred_proba': Class probabilities [p0, p1]
              - 'tokens': List of segmented words/tokens
              - 'shap_values_clickbait': Shapley values for class 1 (clickbait)
              - 'top_clickbait_tokens': Top tokens contributing toward clickbait
              - 'top_non_clickbait_tokens': Top tokens contributing toward non-clickbait
        """
        probs = self.predict_fn([headline])[0]
        pred_label = int(np.argmax(probs))
        pred_class_name = "clickbait" if pred_label == 1 else "not-clickbait"

        # Compute SHAP explanation object
        shap_values_obj = self.explainer([headline])

        # Extract tokens and Shapley values for clickbait class (index 1)
        raw_tokens = list(shap_values_obj.data[0])
        # Values array has shape (n_tokens, n_classes) or (n_tokens,)
        values_arr = shap_values_obj.values[0]
        if values_arr.ndim == 2:
            clickbait_shap = values_arr[:, 1].tolist()
        else:
            clickbait_shap = values_arr.tolist()

        # Clean tokens and filter empty whitespace
        tokens_cleaned = []
        scores_cleaned = []
        for tok, sc in zip(raw_tokens, clickbait_shap):
            t_str = str(tok).strip()
            if t_str:
                tokens_cleaned.append(t_str)
                scores_cleaned.append(float(sc))

        token_score_pairs = list(zip(tokens_cleaned, scores_cleaned))

        # Top positive (push toward clickbait) and top negative (push toward non-clickbait)
        sorted_by_score = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)
        top_clickbait = [t for t, s in sorted_by_score if s > 0][:5]
        top_non_clickbait = [t for t, s in sorted_by_score if s < 0][-5:]

        return {
            "headline": headline,
            "pred_label": pred_label,
            "pred_class_name": pred_class_name,
            "pred_proba": [round(float(p), 4) for p in probs],
            "clickbait_prob": round(float(probs[1]), 4),
            "tokens": tokens_cleaned,
            "shap_values_clickbait": [round(s, 4) for s in scores_cleaned],
            "token_attributions": [
                {"token": t, "score": round(s, 4)} for t, s in token_score_pairs
            ],
            "top_clickbait_tokens": top_clickbait,
            "top_non_clickbait_tokens": top_non_clickbait,
        }
