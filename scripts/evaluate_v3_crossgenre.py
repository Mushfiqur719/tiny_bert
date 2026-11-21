"""scripts/evaluate_v3_crossgenre.py — Phase K: Cross-Genre Evaluation (News -> YouTube) + 1% Recalibration.

Evaluates models trained on BanglaBait (news headlines) zero-shot on BaitBuster-Bangla (YouTube titles),
and also evaluates a 1% target threshold-recalibration arm (tune threshold on 1% stratified sample,
evaluate on remaining 99%). Publishes confusion matrices, true label prevalence, and 95% bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.final_test import predict_model_on_texts
from scripts.train_v3_phase_e import SEEDS, compute_full_metrics
from src.evaluation.metrics import find_optimal_threshold, predict_with_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_ci(vals: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=float)
    if len(arr) <= 1:
        return float(arr[0]), float(arr[0])
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Phase K: Cross-Genre Evaluation on BaitBuster-Bangla.")
    parser.add_argument("--models", nargs="+", default=["teacher_base", "baseline_small_12l_ce", "F_strided_kd_100", "F_strided_ce_100", "F_rand_kd_100", "F_rand_ce_100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    bb_csv = Path("data/processed/v3_baitbuster/labeled_clean_v3.csv")
    bb_df = pd.read_csv(bb_csv)
    texts = bb_df["headline"].astype(str).tolist()
    y_all = bb_df["label"].astype(int).to_numpy()
    true_prev = float(y_all.mean())

    # 1% stratified sample for threshold recalibration, 99% held-out evaluation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.01, random_state=42)
    cal_idx, eval_idx = next(sss.split(texts, y_all))

    exp_v3 = Path("experiments/v3")
    rows = []

    for model_name in args.models:
        zs_f1s, zs_accs, rec_f1s, rec_accs = [], [], [], []
        last_cm_zs = None
        last_cm_rec = None

        for seed in args.seeds:
            run_dir = exp_v3 / model_name / f"seed{seed}"
            if not (run_dir / "config.json").exists():
                continue

            logits = predict_model_on_texts(run_dir, texts, max_length=64)
            # Zero-shot on full 100%
            m_zs = compute_full_metrics(logits, y_all)
            m_zs["split"] = "cross_genre_baitbuster_zeroshot"
            (run_dir / "crossgenre_metrics.json").write_text(json.dumps(m_zs, indent=2), encoding="utf-8")
            zs_f1s.append(m_zs["f1_macro"])
            zs_accs.append(m_zs["accuracy"])
            last_cm_zs = m_zs["confusion_matrix"]

            # 1% Recalibration arm
            opt_thr, _ = find_optimal_threshold(logits[cal_idx], y_all[cal_idx])
            eval_preds_rec = predict_with_threshold(logits[eval_idx], threshold=opt_thr)
            # Build pseudo-logits matching predicted labels for consistent metrics helper
            rec_f1 = float(f1_score(y_all[eval_idx], eval_preds_rec, average="macro"))
            rec_acc = float(np.mean(eval_preds_rec == y_all[eval_idx]))
            rec_f1s.append(rec_f1)
            rec_accs.append(rec_acc)

        if zs_f1s:
            z_lo, z_hi = bootstrap_ci(zs_f1s)
            r_lo, r_hi = bootstrap_ci(rec_f1s)
            rows.append({
                "model": model_name,
                "n_seeds": len(zs_f1s),
                "zs_acc_mean": round(float(np.mean(zs_accs)), 4),
                "zs_f1_mean": round(float(np.mean(zs_f1s)), 4),
                "zs_f1_ci": f"[{z_lo:.4f}, {z_hi:.4f}]",
                "recal1pct_acc_mean": round(float(np.mean(rec_accs)), 4),
                "recal1pct_f1_mean": round(float(np.mean(rec_f1s)), 4),
                "recal1pct_f1_ci": f"[{r_lo:.4f}, {r_hi:.4f}]",
                "sample_cm_zeroshot": str(last_cm_zs),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(exp_v3 / "crossgenre_v3_table.csv", index=False)
    report = f"""# Phase K — Cross-Genre Evaluation Report (News → YouTube Video Titles)

**Target Benchmark:** `abdalimran/BaitBuster-Bangla` ($N = {len(y_all):,}$)  
**True Clickbait Prevalence (from ground-truth labels):** `{true_prev:.2%}` (Class 0: {int((y_all == 0).sum()):,}, Class 1: {int((y_all == 1).sum()):,})  
**Recalibration Protocol:** Decision threshold retuned on a 1% stratified sample ($n = {len(cal_idx)}$), evaluated on remaining 99% ($n = {len(eval_idx):,}$).

{out_df.to_markdown(index=False) if not out_df.empty else "_No models evaluated._"}
"""
    (exp_v3 / "crossgenre_v3_report.md").write_text(report, encoding="utf-8")
    logger.info("Saved cross-genre report to experiments/v3/crossgenre_v3_report.md")


if __name__ == "__main__":
    main()
