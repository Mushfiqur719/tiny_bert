"""
tests/test_significance.py — Unit tests for McNemar's statistical significance testing.

Run with:  pytest tests/test_significance.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.mcnemar_test import (
    build_contingency_table,
    run_pairwise_mcnemar_test,
)


class TestSignificance:

    def test_contingency_table_math(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_a =    [1, 1, 0, 1, 1, 0, 0, 0]  # A correct: 0, 1, 2, 4, 5, 7 (6 correct), wrong: 3, 6 (2 wrong)
        y_b =    [1, 0, 0, 0, 1, 1, 0, 0]  # B correct: 0, 2, 3, 4, 7 (5 correct), wrong: 1, 5, 6 (3 wrong)

        table, breakdown = build_contingency_table(y_true, y_a, y_b)
        assert table.shape == (2, 2)
        assert breakdown["n_total"] == 8
        assert (
            breakdown["both_correct_n00"]
            + breakdown["a_only_correct_n01"]
            + breakdown["b_only_correct_n10"]
            + breakdown["both_wrong_n11"]
        ) == 8

    def test_identical_models_yield_p_value_one(self):
        y_true = [1, 0, 1, 0, 1, 0]
        y_a = [1, 0, 1, 1, 0, 0]
        y_b = [1, 0, 1, 1, 0, 0]

        res = run_pairwise_mcnemar_test(y_true, y_a, y_b, "ModelA", "ModelB", exact=True)
        assert res["counts"]["discordant_pairs"] == 0
        assert res["p_value"] == 1.0
        assert not res["is_significant_alpha_05"]

    def test_highly_divergent_models_yield_significant_p_value(self):
        # 100 samples where Model A is always correct and Model B is always wrong
        y_true = [1] * 50 + [0] * 50
        y_a = [1] * 50 + [0] * 50
        y_b = [0] * 50 + [1] * 50

        res = run_pairwise_mcnemar_test(y_true, y_a, y_b, "SuperiorModel", "InferiorModel", exact=True)
        assert res["counts"]["a_only_correct_n01"] == 100
        assert res["counts"]["b_only_correct_n10"] == 0
        assert res["p_value"] < 1e-10
        assert res["is_significant_alpha_05"]
        assert res["is_significant_alpha_01"]
        assert "statistically significant" in res["interpretation"]
