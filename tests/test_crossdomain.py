"""
tests/test_crossdomain.py — Unit tests for cross-domain dataset loading and evaluation.

Run with:  pytest tests/test_crossdomain.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_crossdomain import (
    align_label_schema,
    detect_columns,
    load_and_clean_crossdomain_dataset,
)
from src.evaluation.crossdomain_eval import evaluate_model_on_dataset


class TestCrossDomain:

    def test_detect_columns_various_names(self):
        df1 = pd.DataFrame({"title": ["খবর ১", "খবর ২"], "is_clickbait": [1, 0]})
        h1, l1 = detect_columns(df1)
        assert h1 == "title"
        assert l1 == "is_clickbait"

        df2 = pd.DataFrame({"news_headline": ["খবর ১"], "category": ["clickbait"]})
        h2, l2 = detect_columns(df2)
        assert h2 == "news_headline"
        assert l2 == "category"

    def test_align_label_schema_strings(self):
        s = pd.Series(["clickbait", "not-clickbait", "bait", "news", "misleading", "genuine"])
        aligned = align_label_schema(s)
        assert aligned.tolist() == [1, 0, 1, 0, 1, 0]

    def test_align_label_schema_integers(self):
        s = pd.Series([1, 0, 1, 0])
        aligned = align_label_schema(s)
        assert aligned.tolist() == [1, 0, 1, 0]

    def test_load_and_clean_crossdomain_csv(self, tmp_path):
        sample_csv = tmp_path / "crossdomain_sample.csv"
        df = pd.DataFrame({
            "title": [
                "চমকপ্রদ খবর দেখুন ভিডিওতে এখনই!",
                "বাংলাদেশ জাতীয় ক্রিকেট দলের জয়",
                "জা’নিনা কার সন্তান আমার পে’টে…",
            ],
            "label": ["clickbait", "not_clickbait", "clickbait"],
        })
        df.to_csv(sample_csv, index=False)

        cleaned_df = load_and_clean_crossdomain_dataset(sample_csv)
        assert len(cleaned_df) == 3
        assert list(cleaned_df.columns) == ["headline", "label"]
        assert cleaned_df["label"].tolist() == [1, 0, 1]
        assert "headline" in cleaned_df.columns

    def test_evaluate_model_on_toy_dataset(self, tmp_path):
        toy_df = pd.DataFrame({
            "headline": [
                "চমকপ্রদ খবর দেখুন এখনই!",
                "অর্থনৈতিক সমীক্ষা ও বাজেট রিপোর্ট",
            ],
            "label": [1, 0],
        })
        metrics = evaluate_model_on_dataset("experiments/tinybanglabert_final", toy_df)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert len(metrics["predictions"]) == 2
