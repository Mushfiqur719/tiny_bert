"""
tests/test_metrics.py — Unit tests for evaluation metrics.

Run with:  pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
)


class TestMetrics:

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        m = compute_classification_metrics(y_pred, y_true)

        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["f1_clickbait"] == 1.0
        assert m["f1_not_clickbait"] == 1.0
        assert m["confusion_matrix"] == [[3, 0], [0, 3]]

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = compute_classification_metrics(y_pred, y_true)

        assert m["accuracy"] == 0.0
        assert m["f1_macro"] == 0.0
        assert m["confusion_matrix"] == [[0, 2], [2, 0]]

    def test_imbalanced_toy_case(self):
        # 4 non-clickbait (0), 2 clickbait (1)
        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1]  # 1 false positive on class 0

        m = compute_classification_metrics(y_pred, y_true)
        # TN=3, FP=1, FN=0, TP=2
        assert m["confusion_matrix"] == [[3, 1], [0, 2]]
        assert m["accuracy"] == pytest.approx(5 / 6, abs=1e-3)
        assert m["recall_clickbait"] == 1.0
        assert m["precision_clickbait"] == pytest.approx(2 / 3, abs=1e-3)

    def test_all_zero_predictions_edge_case(self):
        y_true = [0, 1, 1]
        y_pred = [0, 0, 0]
        m = compute_classification_metrics(y_pred, y_true)

        assert m["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
        assert m["f1_clickbait"] == 0.0
        assert m["precision_clickbait"] == 0.0
        assert m["recall_clickbait"] == 0.0

    def test_compute_metrics_from_eval_pred_adapter(self):
        # 3 samples, 2 classes logits
        logits = np.array([
            [2.5, 0.1],   # argmax 0
            [0.2, 1.8],   # argmax 1
            [1.0, 1.1],   # argmax 1
        ])
        labels = np.array([0, 1, 0])

        res = compute_metrics_from_eval_pred((logits, labels))
        assert "accuracy" in res
        assert "f1_macro" in res
        assert "confusion_matrix" not in res  # Scalar dict for Trainer
        assert res["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
