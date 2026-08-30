"""
tests/test_explainability.py — Unit tests for SHAP, LIME, and cross-method agreement analysis.

Run with:  pytest tests/test_explainability.py -v
"""

from __future__ import annotations

import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.explainability.agreement import (
    compute_token_jaccard,
    compute_explanation_agreement,
    aggregate_agreement_report,
)
from src.explainability.shap_runner import ShapExplainerRunner
from src.explainability.lime_runner import LimeExplainerRunner
from src.explainability.visualizer import highlight_tokens_html


@pytest.fixture(scope="module")
def student_model_and_tokenizer():
    path = "experiments/tinybanglabert_final"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer


class TestExplainability:

    def test_jaccard_similarity_math(self):
        a = ["ভিডিও", "ভাইরাল", "দেখুন"]
        b = ["ভাইরাল", "দেখুন", "খবর"]
        # Intersection = {"ভাইরাল", "দেখুন"} (2), Union = {"ভিডিও", "ভাইরাল", "দেখুন", "খবর"} (4)
        j = compute_token_jaccard(a, b, k=3)
        assert j == pytest.approx(2 / 4, abs=1e-3)

        # Disjoint
        assert compute_token_jaccard(["এক"], ["দুই"], k=3) == 0.0
        # Identical
        assert compute_token_jaccard(["এক", "দুই"], ["এক", "দুই"], k=3) == 1.0

    def test_highlight_html_generation(self):
        pairs = [("চমকপ্রদ", 0.8), ("খবর", -0.4), ("নরমাল", 0.0)]
        out_html = highlight_tokens_html(pairs)
        assert "background-color" in out_html
        assert "চমকপ্রদ" in out_html
        assert "খবর" in out_html

    def test_shap_explanation_runner(self, student_model_and_tokenizer):
        model, tokenizer = student_model_and_tokenizer
        runner = ShapExplainerRunner(model, tokenizer)

        headline = "চমকপ্রদ খবর দেখুন ভিডিওতে এখনই!"
        res = runner.explain_headline(headline)

        assert "pred_label" in res
        assert "clickbait_prob" in res
        assert "tokens" in res
        assert "token_attributions" in res
        assert len(res["tokens"]) > 0
        assert len(res["token_attributions"]) > 0

    def test_lime_explanation_runner(self, student_model_and_tokenizer):
        model, tokenizer = student_model_and_tokenizer
        runner = LimeExplainerRunner(model, tokenizer)

        headline = "বাংলাদেশ জাতীয় ফুটবল দলের ঐতিহাসিক জয়"
        res = runner.explain_headline(headline, num_features=5, num_samples=100)

        assert "pred_label" in res
        assert "lime_weights" in res
        assert len(res["lime_weights"]) > 0

    def test_agreement_computation(self, student_model_and_tokenizer):
        model, tokenizer = student_model_and_tokenizer
        shap_runner = ShapExplainerRunner(model, tokenizer)
        lime_runner = LimeExplainerRunner(model, tokenizer)

        headline = "ভাইরাল ভিডিওতে দেখুন সেই চাঞ্চল্যকর ঘটনা!"
        s_res = shap_runner.explain_headline(headline)
        l_res = lime_runner.explain_headline(headline, num_features=5, num_samples=100)

        agr = compute_explanation_agreement(s_res, l_res, top_k=3)
        assert "jaccard_top3_overall" in agr
        assert "sign_agreement" in agr
        assert 0.0 <= agr["jaccard_top3_overall"] <= 1.0
        assert 0.0 <= agr["sign_agreement"] <= 1.0
