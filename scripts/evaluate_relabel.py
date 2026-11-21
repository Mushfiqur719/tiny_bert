"""scripts/evaluate_relabel.py — Compute Cohen's kappa and noise rates after Phase D human annotation.

Reads:
  - data/processed/v3/relabel_batch.csv (with human-filled annotator_1_label, annotator_2_label, resolved_label)
  - data/processed/v3/relabel_key.csv
Writes:
  - experiments/v3/relabel_report.md
  - data/processed/v3/test_labels_relabelled.csv
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_relabel() -> None:
    batch_path = Path("data/processed/v3/relabel_batch.csv")
    key_path = Path("data/processed/v3/relabel_key.csv")
    out_report = Path("experiments/v3/relabel_report.md")
    out_test_labels = Path("data/processed/v3/test_labels_relabelled.csv")

    if not batch_path.exists() or not key_path.exists():
        raise FileNotFoundError("Missing relabel_batch.csv or relabel_key.csv. Run scripts.prepare_relabel_batch first.")

    batch_df = pd.read_csv(batch_path)
    key_df = pd.read_csv(key_path)

    for col in ("annotator_1_label", "annotator_2_label", "resolved_label"):
        if col not in batch_df.columns or batch_df[col].isna().all():
            raise ValueError(
                f"Column '{col}' in {batch_path} is empty. Human annotation (Phase D) must be completed first."
            )

    df = batch_df.merge(key_df, on="item_id", how="inner")
    a1 = df["annotator_1_label"].astype(int).to_numpy()
    a2 = df["annotator_2_label"].astype(int).to_numpy()
    res = df["resolved_label"].astype(int).to_numpy()
    orig = df["orig_label"].astype(int).to_numpy()

    # 1. Inter-annotator agreement
    kappa = float(cohen_kappa_score(a1, a2))
    raw_agree = float(np.mean(a1 == a2))

    # 2. Noise rates by subset
    flagged_mask = df["orig_split"] == "train_flagged"
    random_mask = df["orig_split"] == "train_random"
    test_mask = df["orig_split"] == "test"

    def _error_stats(mask: pd.Series) -> tuple[int, int, float]:
        n = int(mask.sum())
        n_err = int((res[mask] != orig[mask]).sum())
        return n_err, n, (n_err / n if n > 0 else 0.0)

    f_err, f_n, f_rate = _error_stats(flagged_mask)
    r_err, r_n, r_rate = _error_stats(random_mask)
    t_err, t_n, t_rate = _error_stats(test_mask)

    # Fisher's exact test comparing flagged vs random noise rate
    _, p_val = fisher_exact([[f_err, f_n - f_err], [r_err, r_n - r_err]])

    # 3. Export human-verified test labels
    test_relabelled = df.loc[test_mask, ["orig_idx", "item_id", "orig_label", "resolved_label"]].copy()
    test_relabelled = test_relabelled.sort_values("orig_idx").reset_index(drop=True)
    test_relabelled.to_csv(out_test_labels, index=False)
    logger.info("Saved human-verified test labels (%d rows) to %s", len(test_relabelled), out_test_labels)

    # 4. Write report
    report_md = f"""# Phase D — Human Relabeling & Label Noise Report

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Total Annotated Items:** {len(df):,}

---

## 1. Inter-Annotator Agreement
- **Cohen's Kappa ($\\kappa$):** `{kappa:.4f}`
- **Raw Percent Agreement:** `{raw_agree:.2%}` ({int((a1 == a2).sum()):,} / {len(df):,})

---

## 2. Empirical Label Error Rates (Original vs. Resolved Human Label)

| Subset | Mislabeled Items | Total Items | Error Rate |
|---|---|---|---|
| **Teacher-Flagged Train ($p > 0.9$ disagreement)** | {f_err} | {f_n} | **{f_rate:.2%}** |
| **Random Unflagged Train** | {r_err} | {r_n} | **{r_rate:.2%}** |
| **Full v3 Test Set** | {t_err} | {t_n} | **{t_rate:.2%}** |

- **Fisher's Exact Test (Flagged vs. Random):** $p = {p_val:.4e}$

---

## 3. Reporting Compliance
- All manuscript claims regarding BanglaBait label noise must cite **only** the verified human error rates above.
- Human-verified test labels are stored at `data/processed/v3/test_labels_relabelled.csv` and remain locked until Phase J (`scripts/final_test.py`).
"""
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report_md, encoding="utf-8")
    logger.info("Saved relabel report to %s", out_report)


if __name__ == "__main__":
    evaluate_relabel()
