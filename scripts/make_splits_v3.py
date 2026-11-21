"""scripts/make_splits_v3.py — Generate leak-free MinHash deduplicated grouped stratified splits.

Produces:
  1. data/processed/v3/splits_v3.json
  2. data/processed/v3/conflicts.csv (label-conflicting near-duplicate groups)
  3. data/processed/v3/train_frac_{10,25,50,100}.json (strictly nested stratified subsets)
  4. experiments/v3/leakage_report.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.clean import clean_dataframe
from src.data.dedup import compute_near_dup_groups, norm_text
from src.data.load import load_raw
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_v3_splits():
    out_dir = Path("data/processed/v3")
    exp_dir = Path("experiments/v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/teacher_baseline.yaml")
    labeled_df, unlabeled_df = load_raw(cfg)
    labeled_df = clean_dataframe(labeled_df, headline_col="headline", min_char_length=5)
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    logger.info("Loaded %d cleaned labeled headlines.", len(labeled_df))

    # 1. Compute MinHash near-duplicate groups
    headlines = labeled_df["headline"].tolist()
    groups = compute_near_dup_groups(headlines, threshold=0.9, num_perm=128)
    labeled_df["group_id"] = groups

    # 2. Identify and remove label-conflicting groups
    group_label_counts = labeled_df.groupby("group_id")["label"].nunique()
    conflicting_group_ids = set(group_label_counts[group_label_counts > 1].index)

    conflicts_df = labeled_df[labeled_df["group_id"].isin(conflicting_group_ids)].copy()
    conflicts_path = out_dir / "conflicts.csv"
    conflicts_df.to_csv(conflicts_path, index=False)
    logger.info("Saved %d conflicting rows across %d groups to %s", len(conflicts_df), len(conflicting_group_ids), conflicts_path)

    clean_df = labeled_df[~labeled_df["group_id"].isin(conflicting_group_ids)].copy().reset_index(drop=True)
    logger.info("Retained %d clean rows after dropping conflicts.", len(clean_df))

    # 3. StratifiedGroupKFold split (fold 0 = test, fold 1 = val, folds 2..9 = train)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    splits_gen = sgkf.split(clean_df, clean_df["label"], clean_df["group_id"])

    test_idx = None
    val_idx = None
    train_idx_list = []

    for fold_num, (_, fold_indices) in enumerate(splits_gen):
        if fold_num == 0:
            test_idx = fold_indices.tolist()
        elif fold_num == 1:
            val_idx = fold_indices.tolist()
        else:
            train_idx_list.extend(fold_indices.tolist())

    assert test_idx is not None and val_idx is not None
    train_idx = train_idx_list

    # 4. Verify zero cross-split group leakage
    train_groups = set(clean_df.loc[train_idx, "group_id"])
    val_groups = set(clean_df.loc[val_idx, "group_id"])
    test_groups = set(clean_df.loc[test_idx, "group_id"])

    assert len(train_groups & val_groups) == 0, "Train and Val share near-duplicate groups!"
    assert len(train_groups & test_groups) == 0, "Train and Test share near-duplicate groups!"
    assert len(val_groups & test_groups) == 0, "Val and Test share near-duplicate groups!"

    logger.info("Verification passed: ZERO cross-split group leakage!")

    # 5. Save splits_v3.json
    splits_v3 = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "dedup_threshold": 0.9,
            "total_clean_labeled": len(clean_df),
            "total_conflicts_dropped": len(conflicts_df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "labeled": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "unlabeled": {
            "all": list(range(len(unlabeled_df))),
        },
    }

    splits_path = out_dir / "splits_v3.json"
    splits_path.write_text(json.dumps(splits_v3, indent=2), encoding="utf-8")
    logger.info("Saved v3 splits to %s", splits_path)

    # 6. Save clean_df as reference CSV for v3
    clean_df_path = out_dir / "labeled_clean_v3.csv"
    clean_df.to_csv(clean_df_path, index=False)
    logger.info("Saved v3 labeled clean DataFrame to %s", clean_df_path)

    # 7. Create strictly nested stratified subsets of train: 10%, 25%, 50%, 100%
    train_df = clean_df.loc[train_idx].copy().reset_index()  # preserves original clean_df index in column 'index'
    train_labels = train_df["label"].tolist()

    # 100%
    train_100_idx = train_df["index"].tolist()

    # 50%
    sub_50_df, _ = train_test_split(train_df, train_size=0.50, stratify=train_labels, random_state=42)
    train_50_idx = sorted(sub_50_df["index"].tolist())

    # 25% (nested within 50%)
    sub_25_df, _ = train_test_split(sub_50_df, train_size=0.50, stratify=sub_50_df["label"], random_state=42)
    train_25_idx = sorted(sub_25_df["index"].tolist())

    # 10% (nested within 25% by taking 0.40 of 25%)
    sub_10_df, _ = train_test_split(sub_25_df, train_size=0.40, stratify=sub_25_df["label"], random_state=42)
    train_10_idx = sorted(sub_10_df["index"].tolist())

    # Verify nesting: 10% ⊂ 25% ⊂ 50% ⊂ 100%
    assert set(train_10_idx).issubset(set(train_25_idx)), "10% is not nested inside 25%!"
    assert set(train_25_idx).issubset(set(train_50_idx)), "25% is not nested inside 50%!"
    assert set(train_50_idx).issubset(set(train_100_idx)), "50% is not nested inside 100%!"

    for frac, idxs in [(10, train_10_idx), (25, train_25_idx), (50, train_50_idx), (100, train_100_idx)]:
        subset_path = out_dir / f"train_frac_{frac}.json"
        subset_data = {
            "fraction": frac,
            "count": len(idxs),
            "indices": idxs,
            "class_distribution": clean_df.loc[idxs, "label"].value_counts().to_dict(),
        }
        subset_path.write_text(json.dumps(subset_data, indent=2), encoding="utf-8")
        logger.info("Saved nested subset (%d%%, %d samples) to %s", frac, len(idxs), subset_path)

    # 8. Write experiments/v3/leakage_report.md
    train_prev = clean_df.loc[train_idx, "label"].mean()
    val_prev = clean_df.loc[val_idx, "label"].mean()
    test_prev = clean_df.loc[test_idx, "label"].mean()

    report_md = f"""# Leakage & Data Audit Report (v3)

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Deduplication Method:** MinHash LSH (Threshold $J \\ge 0.9$, 128 permutations, character 3-shingles)  
**Partitioning:** 10-Fold Stratified Group K-Fold (Fold 0 = Test, Fold 1 = Val, Folds 2–9 = Train)  

---

## 1. Split Counts & Class Balance

| Split | Total Samples | Not-Clickbait (0) | Clickbait (1) | Clickbait Prevalence (%) |
|---|---|---|---|---|
| **Train (80%)** | {len(train_idx):,} | {(clean_df.loc[train_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_idx, 'label'] == 1).sum():,} | {train_prev:.2%} |
| **Validation (10%)** | {len(val_idx):,} | {(clean_df.loc[val_idx, 'label'] == 0).sum():,} | {(clean_df.loc[val_idx, 'label'] == 1).sum():,} | {val_prev:.2%} |
| **Test (10%)** | {len(test_idx):,} | {(clean_df.loc[test_idx, 'label'] == 0).sum():,} | {(clean_df.loc[test_idx, 'label'] == 1).sum():,} | {test_prev:.2%} |
| **Total Clean** | {len(clean_df):,} | {(clean_df['label'] == 0).sum():,} | {(clean_df['label'] == 1).sum():,} | {clean_df['label'].mean():.2%} |

---

## 2. Leakage Verification & Group Disjointness

- **Label Conflicting Rows Dropped:** {len(conflicts_df):,} rows ({len(conflicting_group_ids)} near-duplicate groups) saved to `data/processed/v3/conflicts.csv`.
- **Train ∩ Val Overlap:** **0 groups** (0 samples)
- **Train ∩ Test Overlap:** **0 groups** (0 samples)
- **Val ∩ Test Overlap:** **0 groups** (0 samples)
- **Cross-Split Near-Duplicate Leakage:** **Strictly 0.00%**

---

## 3. Nested Training Subsets

| Fraction | Sample Count | Class 0 (Not-Clickbait) | Class 1 (Clickbait) | Strict Subset Verification |
|---|---|---|---|---|
| **10%** | {len(train_10_idx):,} | {(clean_df.loc[train_10_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_10_idx, 'label'] == 1).sum():,} | 10% ⊂ 25% (Verified) |
| **25%** | {len(train_25_idx):,} | {(clean_df.loc[train_25_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_25_idx, 'label'] == 1).sum():,} | 25% ⊂ 50% (Verified) |
| **50%** | {len(train_50_idx):,} | {(clean_df.loc[train_50_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_50_idx, 'label'] == 1).sum():,} | 50% ⊂ 100% (Verified) |
| **100%** | {len(train_100_idx):,} | {(clean_df.loc[train_100_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_100_idx, 'label'] == 1).sum():,} | Full Clean Train Split |
"""
    leakage_report_path = exp_dir / "leakage_report.md"
    leakage_report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved leakage report to %s", leakage_report_path)


if __name__ == "__main__":
    generate_v3_splits()
