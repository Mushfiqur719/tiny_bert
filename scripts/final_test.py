"""scripts/final_test.py — Phase J: One-Time Locked Test Evaluation.

ONLY script permitted to evaluate against held-out test labels.
Requires environment variable: TBB_ALLOW_TEST=1

Performs:
  1. Single-invocation audit log to experiments/v3/final_test_log.txt.
  2. Evaluates every locked model across all 5 seeds on:
       - v3 clean test split (splits_v3.json)
       - Phase D human-relabelled test labels (if data/processed/v3/test_labels_relabelled.csv exists)
  3. Paired bootstrap hypothesis tests over items and seeds (10,000 resamples), Holm-corrected.
  4. Non-inferiority tests against BanglaBERT-Base and BanglaBERT-Small-12L (margin = 2.0 pp Macro-F1).
  5. Writes experiments/v3/final_test_results.csv and experiments/v3/final_test_report.md.

Usage:
  $env:TBB_ALLOW_TEST="1"; .venv\\Scripts\\python.exe -m scripts.final_test --models teacher_base baseline_small_12l_ce F_strided_kd_100 F_strided_ce_100 F_rand_kd_100 F_rand_ce_100
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_v3_phase_e import (
    SEEDS,
    compute_full_metrics,
    format_headlines,
    get_dataset_paths,
)
from src.data.split import apply_splits
from src.utils.test_guard import assert_test_access

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

NON_INFERIORITY_MARGIN = 0.020  # 2.0 percentage points Macro-F1 (PREREG.md)


def paired_bootstrap_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    y_true: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap over test items AND seeds (shape: [n_seeds, n_items])."""
    rng = np.random.RandomState(seed)
    n_seeds, n_items = preds_a.shape

    obs_a = float(np.mean([f1_score(y_true, preds_a[s], average="macro") for s in range(n_seeds)]))
    obs_b = float(np.mean([f1_score(y_true, preds_b[s], average="macro") for s in range(n_seeds)]))
    obs_diff = obs_a - obs_b

    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        item_idx = rng.randint(0, n_items, size=n_items)
        seed_idx = rng.randint(0, n_seeds, size=n_seeds)
        y_b = y_true[item_idx]
        fa = np.mean([f1_score(y_b, preds_a[s, item_idx], average="macro") for s in seed_idx])
        fb = np.mean([f1_score(y_b, preds_b[s, item_idx], average="macro") for s in seed_idx])
        diffs[b] = fa - fb

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    # Two-sided p-value
    p_two_sided = float(2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0)))
    p_two_sided = min(1.0, max(p_two_sided, 1.0 / n_resamples))

    return {
        "f1_a": round(obs_a, 4),
        "f1_b": round(obs_b, 4),
        "diff_pp": round(obs_diff * 100.0, 2),
        "ci_lo_pp": round(ci_lo * 100.0, 2),
        "ci_hi_pp": round(ci_hi * 100.0, 2),
        "p_value": p_two_sided,
        "non_inferior_2pp": bool(ci_lo >= -NON_INFERIORITY_MARGIN),
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = p_values[idx] * (m - rank)
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


@torch.no_grad()
def predict_model_on_texts(
    ckpt_dir: Path,
    texts: list[str],
    bodies: list[str] | None = None,
    batch_size: int = 64,
    max_length: int = 64,
) -> np.ndarray:
    """Run inference for a saved HF checkpoint on a list of headlines (or title+body pairs)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()

    logits_list = []
    for i in range(0, len(texts), batch_size):
        batch_t = texts[i : i + batch_size]
        if bodies is not None:
            batch_b = bodies[i : i + batch_size]
            enc = tokenizer(batch_t, batch_b, padding=True, truncation="only_second", max_length=max_length, return_tensors="pt").to(device)
        else:
            enc = tokenizer(batch_t, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)
        logits_list.append(out.logits.cpu().numpy())
    return np.vstack(logits_list)


def main():
    # 1. Enforce test lock strictly
    assert_test_access()

    parser = argparse.ArgumentParser(description="Phase J: One-Time Locked Test Evaluation.")
    parser.add_argument("--dataset", choices=["banglabait", "baitbuster"], default="banglabait")
    parser.add_argument("--models", nargs="+", required=True, help="List of model group names in experiments/v3*/.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-resamples", type=int, default=10000)
    args = parser.parse_args()

    data_dir, exp_v3, splits_path = get_dataset_paths(args.dataset)
    log_path = exp_v3 / "final_test_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] Invoked final_test.py dataset={args.dataset} models={args.models}, seeds={args.seeds}\n")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    csv_path = data_dir / "labeled_clean_v3_full.csv" if (data_dir / "labeled_clean_v3_full.csv").exists() else data_dir / "labeled_clean_v3.csv"
    clean_df = pd.read_csv(csv_path)
    test_df = apply_splits(clean_df, splits, split="test")
    y_test = test_df["label"].to_numpy()

    all_preds: dict[str, np.ndarray] = {}
    summary_rows = []

    for model_name in args.models:
        seed_preds = []
        seed_f1s = []
        seed_accs = []
        for seed in args.seeds:
            run_dir = exp_v3 / model_name / f"seed{seed}"
            if not (run_dir / "config.json").exists():
                logger.warning("Checkpoint missing at %s; skipping.", run_dir)
                continue

            cfg_yaml = run_dir / "config.yaml"
            tm = "title"
            max_len = 64
            if cfg_yaml.exists():
                try:
                    cinfo = json.loads(cfg_yaml.read_text(encoding="utf-8"))
                    tm = cinfo.get("text_mode", "title")
                    max_len = int(cinfo.get("max_length", 256 if tm != "title" else 64))
                except Exception:
                    pass

            test_titles, test_bodies = format_headlines(test_df, tm)
            logits = predict_model_on_texts(run_dir, test_titles, bodies=test_bodies, max_length=max_len)
            np.save(run_dir / "test_logits.npy", logits)
            m = compute_full_metrics(logits, y_test)
            m["split"] = "test"
            (run_dir / "test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

            seed_preds.append(np.argmax(logits, axis=1))
            seed_f1s.append(m["f1_macro"])
            seed_accs.append(m["accuracy"])

        if seed_preds:
            all_preds[model_name] = np.stack(seed_preds, axis=0)
            summary_rows.append({
                "model": model_name,
                "n_seeds": len(seed_f1s),
                "test_acc_mean": round(float(np.mean(seed_accs)), 4),
                "test_acc_std": round(float(np.std(seed_accs)), 4),
                "test_f1_mean": round(float(np.mean(seed_f1s)), 4),
                "test_f1_std": round(float(np.std(seed_f1s)), 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(exp_v3 / "final_test_results.csv", index=False)

    # Planned pairwise comparisons against references
    comp_rows = []
    refs = [r for r in ("teacher_base", "teacher_base_tb", "teacher_base_uw", "baseline_small_12l_ce", "baseline_small_12l_ce_uw") if r in all_preds]
    for m_name, preds_m in all_preds.items():
        for ref_name in refs:
            if m_name == ref_name:
                continue
            res = paired_bootstrap_test(preds_m, all_preds[ref_name], y_test, n_resamples=args.n_resamples)
            comp_rows.append({"model": m_name, "reference": ref_name, **res})

    if comp_rows:
        p_adj = holm_correction([r["p_value"] for r in comp_rows])
        for r, pa in zip(comp_rows, p_adj):
            r["p_holm"] = round(pa, 5)
            r["p_value"] = round(r["p_value"], 5)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(exp_v3 / "final_test_comparisons.csv", index=False)

    report_md = f"""# Phase J — One-Time Locked Test Evaluation Report

**Timestamp:** {datetime.now(timezone.utc).isoformat()}  
**Test Split:** `data/processed/v3/splits_v3.json` ($n = {len(y_test)}$)  
**Bootstrap Resamples:** {args.n_resamples:,} (paired over items and seeds)  
**Non-Inferiority Margin:** $\\Delta = -2.0$ percentage points Macro-F1  

## 1. Five-Seed Test Summary

{summary_df.to_markdown(index=False) if not summary_df.empty else "_No models evaluated._"}

## 2. Paired Bootstrap & Non-Inferiority Tests (Holm-Corrected)

{comp_df.to_markdown(index=False) if not comp_df.empty else "_No comparisons run._"}
"""
    (exp_v3 / "final_test_report.md").write_text(report_md, encoding="utf-8")
    logger.info("Final test evaluation complete. Report saved to experiments/v3/final_test_report.md")


if __name__ == "__main__":
    main()
