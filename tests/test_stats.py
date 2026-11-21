"""tests/test_stats.py — Unit tests for bootstrap / Holm helpers (no model or network needed)."""

from __future__ import annotations

import numpy as np

from src.evaluation.stats import bootstrap_ci, holm_correction, mean_std, paired_bootstrap


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    pred = np.where(rng.random(400) < 0.8, y, 1 - y)
    est, lo, hi = bootstrap_ci(y, pred, n_boot=300)
    assert lo <= est <= hi
    assert 0.0 < hi - lo < 0.2


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = np.where(rng.random(600) < 0.9, y, 1 - y)
    bad = np.where(rng.random(600) < 0.65, y, 1 - y)
    r = paired_bootstrap(y, good, bad, n_boot=300)
    assert r["delta"] > 0.15
    assert r["ci_low"] > 0
    assert r["p_value"] < 0.05


def test_paired_bootstrap_identical_models_not_significant():
    y = np.array([0, 1] * 200)
    pred = y.copy()
    pred[::7] = 1 - pred[::7]
    r = paired_bootstrap(y, pred, pred, n_boot=200)
    assert r["delta"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0


def test_holm_correction_monotone_and_bounded():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj[0] == 0.03  # 3 * 0.01
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[2] >= adj[0]


def test_mean_std_single_value():
    assert mean_std([0.5]) == (0.5, 0.0)


def test_selective_prediction_and_cascade():
    from src.evaluation.selective import (
        evaluate_edge_cloud_cascade,
        evaluate_selective_prediction,
        fit_temperature_on_val,
    )

    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    # High confidence when correct, low confidence when wrong
    logits = np.zeros((200, 2), dtype=np.float64)
    for i, label in enumerate(y):
        if i < 160:
            logits[i, label] = 3.0
            logits[i, 1 - label] = -1.0
        else:
            logits[i, 1 - label] = 0.2
            logits[i, label] = 0.0

    t_cal = fit_temperature_on_val(logits, y)
    assert 0.2 <= t_cal <= 5.0

    sel_df, summary = evaluate_selective_prediction(logits, y, temperature=t_cal)
    assert not sel_df.empty
    acc_100 = sel_df.loc[sel_df["Coverage (%)"] == 100.0, "Selective Accuracy (%)"].iloc[0]
    acc_50 = sel_df.loc[sel_df["Coverage (%)"] == 50.0, "Selective Accuracy (%)"].iloc[0]
    assert acc_50 >= acc_100
    assert summary["aurc"] >= 0.0

    cas_df = evaluate_edge_cloud_cascade(logits, logits * 1.5, y, student_temp=t_cal)
    assert len(cas_df) == 8


def test_rdrop_kl_non_negative_and_zero_when_identical():
    import torch
    from src.training.train_teacher import compute_rdrop_kl

    a = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    b = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    kl_same = compute_rdrop_kl(a, a.detach())
    kl_diff = compute_rdrop_kl(a, b)
    assert torch.isclose(kl_same, torch.tensor(0.0), atol=1e-6)
    assert kl_diff.item() > 0.0

