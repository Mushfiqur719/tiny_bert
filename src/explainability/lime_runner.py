"""
src/explainability/lime_runner.py — LIME (Local Interpretable Model-agnostic Explanations) for TinyBanglaBERT.

Uses local perturbation and linear surrogate models to compute feature importance weights for tokens
in Bangla headlines, predicting whether a word contributes to 'clickbait' or 'not-clickbait'.

Usage:
    from src.explainability.lime_runner import LimeExplainerRunner
    runner = LimeExplainerRunner(model, tokenizer)
    explanation = runner.explain_headline("চমকপ্রদ খবর দেখুন ভিডিওতে এখনই!")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LimeExplainerRunner:
    """LIME Explainer wrapper for Bangla transformer text classification models."""

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
            device: CPU or CUDA device.
            max_seq_len: Maximum token sequence length.
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.predict_fn = self._build_predict_fn()

        # Initialize LIME with regex whitespace splitter for Bangla
        self.explainer = LimeTextExplainer(
            class_names=["not-clickbait", "clickbait"],
            split_expression=r"\s+",
            bow=False,
            random_state=42,
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
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            return probs

        return predict_proba

    def explain_headline(
        self,
        headline: str,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> dict[str, Any]:
        """Compute token-level LIME feature weights for a single Bangla headline.

        Args:
            headline: Input Bangla headline string.
            num_features: Maximum number of features to explain.
            num_samples: Number of perturbation samples (default: 500).

        Returns:
            Dictionary containing:
              - 'headline': Input text
              - 'pred_label': Predicted class (0 or 1)
              - 'pred_class_name': 'not-clickbait' or 'clickbait'
              - 'pred_proba': Class probabilities [p0, p1]
              - 'lime_weights': List of (word, weight) tuples for clickbait class
              - 'top_clickbait_tokens': Tokens with highest positive LIME weights
              - 'top_non_clickbait_tokens': Tokens with negative LIME weights
        """
        probs = self.predict_fn([headline])[0]
        pred_label = int(np.argmax(probs))
        pred_class_name = "clickbait" if pred_label == 1 else "not-clickbait"

        # Generate LIME explanation for class 1 (clickbait)
        exp = self.explainer.explain_instance(
            headline,
            self.predict_fn,
            labels=(0, 1),
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extract weights for clickbait class (index 1)
        raw_weights = exp.as_list(label=1)
        # Sort weights descending
        sorted_weights = sorted(raw_weights, key=lambda x: x[1], reverse=True)

        top_clickbait = [word for word, wt in sorted_weights if wt > 0][:5]
        top_non_clickbait = [word for word, wt in sorted(raw_weights, key=lambda x: x[1]) if wt < 0][:5]

        return {
            "headline": headline,
            "pred_label": pred_label,
            "pred_class_name": pred_class_name,
            "pred_proba": [round(float(p), 4) for p in probs],
            "clickbait_prob": round(float(probs[1]), 4),
            "lime_weights": [
                {"token": word, "weight": round(float(wt), 4)} for word, wt in sorted_weights
            ],
            "top_clickbait_tokens": top_clickbait,
            "top_non_clickbait_tokens": top_non_clickbait,
        }
