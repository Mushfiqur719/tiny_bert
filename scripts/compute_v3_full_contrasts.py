"""scripts/compute_v3_full_contrasts.py — Compute direct KD-vs-CE and Init-vs-Rand paired bootstrap tests on saved test logits.

Reads saved test_logits.npy (no re-inference, no extra test access) and computes
direct planned contrasts + Holm correction for both BanglaBait and BaitBuster-Bangla.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.final_test import holm_correction, paired_bootstrap_test
from scripts.train_v3_phase_e import SEEDS, get_dataset_paths
from src.data.split import apply_splits


def run_contrasts(dataset: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    data_dir, exp_dir, splits_path = get_dataset_paths(dataset)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    csv_path = data_dir / "labeled_clean_v3.csv"
    clean_df = pd.read_csv(csv_path)
    os.environ["TBB_ALLOW_TEST"] = "1"
    test_df = apply_splits(clean_df, splits, split="test")
    y_test = test_df["label"].to_numpy()

    def load_preds(model_name: str) -> np.ndarray:
        arr = []
        for s in SEEDS:
            logits = np.load(exp_dir / model_name / f"seed{s}" / "test_logits.npy")
            arr.append(np.argmax(logits, axis=1))
        return np.stack(arr, axis=0)

    rows = []
    for a, b in pairs:
        res = paired_bootstrap_test(load_preds(a), load_preds(b), y_test, n_resamples=10000)
        rows.append({"dataset": dataset, "model_a": a, "model_b": b, **res})

    p_adj = holm_correction([r["p_value"] for r in rows])
    for r, pa in zip(rows, p_adj):
        r["p_value"] = round(r["p_value"], 5)
        r["p_holm"] = round(pa, 5)

    df = pd.DataFrame(rows)
    out_csv = exp_dir / "direct_test_contrasts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n=== Direct Test Contrasts ({dataset}) ===")
    print(df.to_markdown(index=False))
    return df


if __name__ == "__main__":
    run_contrasts(
        "banglabait",
        [
            ("F_strided_kd_100", "F_strided_ce_100"),
            ("F_rand_kd_100", "F_rand_ce_100"),
            ("F_strided_kd_100", "F_rand_kd_100"),
            ("F_strided_ce_100", "F_rand_ce_100"),
            ("F_strided_kd_100", "teacher_base"),
            ("F_strided_kd_100", "baseline_small_12l_ce"),
        ],
    )
    run_contrasts(
        "baitbuster",
        [
            ("F_strided_kd_100_uw", "F_strided_ce_100_uw"),
            ("F_rand_kd_100_uw", "F_rand_ce_100_uw"),
            ("F_strided_kd_100_uw", "F_rand_kd_100_uw"),
            ("F_strided_ce_100_uw", "F_rand_ce_100_uw"),
            ("F_strided_kd_100_uw", "teacher_base_uw"),
            ("F_strided_kd_100_uw", "baseline_small_12l_ce_uw"),
        ],
    )
