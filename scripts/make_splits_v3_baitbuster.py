"""scripts/make_splits_v3_baitbuster.py — Leak-free MinHash deduplicated v3 splits for BaitBuster-Bangla.

Produces:
  1. data/processed/v3_baitbuster/splits_v3.json
  2. data/processed/v3_baitbuster/labeled_clean_v3.csv
  3. data/processed/v3_baitbuster/conflicts.csv
  4. data/processed/v3_baitbuster/train_frac_{10,25,50,100}.json (strictly nested)
  5. experiments/v3_baitbuster/leakage_report.md
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.dedup import compute_near_dup_groups
from src.data.load_crossdomain import load_and_clean_crossdomain_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_baitbuster_v3_splits() -> None:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    out_dir = Path("data/processed/v3_baitbuster")
    exp_dir = Path("experiments/v3_baitbuster")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_crossdomain_dataset("abdalimran/BaitBuster-Bangla")
    df = df[["headline", "label"]].dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    logger.info("Loaded %d cleaned BaitBuster-Bangla gold rows.", len(df))

    # 1. MinHash LSH near-duplicate clustering
    groups = compute_near_dup_groups(df["headline"].tolist(), threshold=0.9, num_perm=128)
    df["group_id"] = groups

    # 2. Drop label-conflicting groups
    group_label_counts = df.groupby("group_id")["label"].nunique()
    conflicting_ids = set(group_label_counts[group_label_counts > 1].index)
    conflicts_df = df[df["group_id"].isin(conflicting_ids)].copy()
    conflicts_df.to_csv(out_dir / "conflicts.csv", index=False)

    clean_df = df[~df["group_id"].isin(conflicting_ids)].copy().reset_index(drop=True)
    logger.info("Retained %d clean BaitBuster rows after dropping %d conflicts.", len(clean_df), len(conflicts_df))

    # 3. 10-Fold StratifiedGroupKFold (fold 0 = test, fold 1 = val, folds 2..9 = train)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    test_idx, val_idx, train_idx = None, None, []
    for fold_num, (_, fold_indices) in enumerate(sgkf.split(clean_df, clean_df["label"], clean_df["group_id"])):
        if fold_num == 0:
            test_idx = fold_indices.tolist()
        elif fold_num == 1:
            val_idx = fold_indices.tolist()
        else:
            train_idx.extend(fold_indices.tolist())

    assert test_idx is not None and val_idx is not None

    # 4. Verify zero cross-split leakage
    g_tr = set(clean_df.loc[train_idx, "group_id"])
    g_va = set(clean_df.loc[val_idx, "group_id"])
    g_te = set(clean_df.loc[test_idx, "group_id"])
    assert len(g_tr & g_va) == 0 and len(g_tr & g_te) == 0 and len(g_va & g_te) == 0
    logger.info("Verification passed: ZERO cross-split group leakage on BaitBuster-Bangla!")

    splits_v3 = {
        "metadata": {
            "dataset": "abdalimran/BaitBuster-Bangla",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "dedup_threshold": 0.9,
            "total_clean_labeled": len(clean_df),
            "total_conflicts_dropped": len(conflicts_df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "labeled": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    (out_dir / "splits_v3.json").write_text(json.dumps(splits_v3, indent=2), encoding="utf-8")
    clean_df.to_csv(out_dir / "labeled_clean_v3.csv", index=False)

    # 5. Strictly nested stratified training subsets: 10%, 25%, 50%, 100%
    train_df = clean_df.loc[train_idx].copy().reset_index()
    train_100_idx = sorted(train_df["index"].tolist())
    sub_50, _ = train_test_split(train_df, train_size=0.50, stratify=train_df["label"], random_state=42)
    train_50_idx = sorted(sub_50["index"].tolist())
    sub_25, _ = train_test_split(sub_50, train_size=0.50, stratify=sub_50["label"], random_state=42)
    train_25_idx = sorted(sub_25["index"].tolist())
    sub_10, _ = train_test_split(sub_25, train_size=0.40, stratify=sub_25["label"], random_state=42)
    train_10_idx = sorted(sub_10["index"].tolist())

    assert set(train_10_idx).issubset(set(train_25_idx))
    assert set(train_25_idx).issubset(set(train_50_idx))
    assert set(train_50_idx).issubset(set(train_100_idx))

    for frac, idxs in [(10, train_10_idx), (25, train_25_idx), (50, train_50_idx), (100, train_100_idx)]:
        (out_dir / f"train_frac_{frac}.json").write_text(
            json.dumps({
                "fraction": frac,
                "count": len(idxs),
                "indices": idxs,
                "class_distribution": clean_df.loc[idxs, "label"].value_counts().to_dict(),
            }, indent=2),
            encoding="utf-8",
        )

    # 6. Leakage report
    report = f"""# Leakage & Data Audit Report — BaitBuster-Bangla (v3)

**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Source:** `abdalimran/BaitBuster-Bangla` (10,000 human-annotated YouTube video titles)  
**Deduplication:** MinHash LSH ($J \\ge 0.9$, 128 permutations)  

| Split | Total | Not-Clickbait (0) | Clickbait (1) | Prevalence |
|---|---|---|---|---|
| **Train (80%)** | {len(train_idx):,} | {(clean_df.loc[train_idx, 'label'] == 0).sum():,} | {(clean_df.loc[train_idx, 'label'] == 1).sum():,} | {clean_df.loc[train_idx, 'label'].mean():.2%} |
| **Validation (10%)** | {len(val_idx):,} | {(clean_df.loc[val_idx, 'label'] == 0).sum():,} | {(clean_df.loc[val_idx, 'label'] == 1).sum():,} | {clean_df.loc[val_idx, 'label'].mean():.2%} |
| **Test (10%)** | {len(test_idx):,} | {(clean_df.loc[test_idx, 'label'] == 0).sum():,} | {(clean_df.loc[test_idx, 'label'] == 1).sum():,} | {clean_df.loc[test_idx, 'label'].mean():.2%} |
| **Total Clean** | {len(clean_df):,} | {(clean_df['label'] == 0).sum():,} | {(clean_df['label'] == 1).sum():,} | {clean_df['label'].mean():.2%} |

- **Conflicts Dropped:** {len(conflicts_df)} rows ({len(conflicting_ids)} groups)
- **Cross-Split Leakage:** **0 groups (0.00%)**
"""
    (exp_dir / "leakage_report.md").write_text(report, encoding="utf-8")
    logger.info("Saved BaitBuster-Bangla v3 splits and leakage report.")


if __name__ == "__main__":
    generate_baitbuster_v3_splits()
