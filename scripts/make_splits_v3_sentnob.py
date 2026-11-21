"""scripts/make_splits_v3_sentnob.py — Phase H: Leak-free MinHash deduplicated v3 splits for SentNoB.

Dataset: SentNoB (Islam et al., EMNLP 2021) — 3-class Sentiment Analysis on Noisy Bangla Texts.
HuggingFace ID: khondoker/SentNoB (Labels: 0=Neutral, 1=Positive, 2=Negative).

Produces:
  1. data/processed/v3_sentnob/splits_v3.json
  2. data/processed/v3_sentnob/labeled_clean_v3.csv
  3. data/processed/v3_sentnob/conflicts.csv
  4. data/processed/v3_sentnob/train_frac_{10,25,50,100}.json (strictly nested)
  5. experiments/v3_sentnob/leakage_report.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.clean import clean_dataframe
from src.data.dedup import compute_near_dup_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_sentnob_splits() -> None:
    out_dir = Path("data/processed/v3_sentnob")
    exp_dir = Path("experiments/v3_sentnob")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("khondoker/SentNoB")
    dfs = [ds[s].to_pandas() for s in ds.keys()]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    text_col = "Data" if "Data" in df.columns else [c for c in df.columns if c.lower() in ("text", "data", "comment", "sentence")][0]
    label_col = "Label" if "Label" in df.columns else [c for c in df.columns if c.lower() in ("label", "sentiment", "target")][0]
    df = df.rename(columns={text_col: "headline", label_col: "label"})[["headline", "label"]].dropna().reset_index(drop=True)

    # Map string labels if needed, otherwise integer 0, 1, 2
    if df["label"].dtype == object:
        str_map = {"neutral": 0, "positive": 1, "negative": 2, "0": 0, "1": 1, "2": 2}
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(str_map)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)

    df = clean_dataframe(df, headline_col="headline", min_char_length=3)
    df = df[["headline", "label"]].reset_index(drop=True)
    logger.info("Loaded %d cleaned SentNoB rows. Class counts: %s", len(df), df["label"].value_counts().to_dict())

    # MinHash LSH near-duplicate clustering
    groups = compute_near_dup_groups(df["headline"].tolist(), threshold=0.9, num_perm=128)
    df["group_id"] = groups

    # Drop label-conflicting groups
    group_label_counts = df.groupby("group_id")["label"].nunique()
    conflicting_ids = set(group_label_counts[group_label_counts > 1].index)
    conflicts_df = df[df["group_id"].isin(conflicting_ids)].copy()
    conflicts_df.to_csv(out_dir / "conflicts.csv", index=False)

    clean_df = df[~df["group_id"].isin(conflicting_ids)].copy().reset_index(drop=True)
    logger.info("Retained %d clean SentNoB rows after dropping %d conflicts.", len(clean_df), len(conflicts_df))

    # 10-Fold StratifiedGroupKFold (fold 0 = test, fold 1 = val, folds 2..9 = train)
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
    g_tr, g_va, g_te = set(clean_df.loc[train_idx, "group_id"]), set(clean_df.loc[val_idx, "group_id"]), set(clean_df.loc[test_idx, "group_id"])
    assert len(g_tr & g_va) == 0 and len(g_tr & g_te) == 0 and len(g_va & g_te) == 0
    logger.info("ZERO cross-split group leakage verified on SentNoB!")

    splits = {
        "metadata": {
            "dataset": "khondoker/SentNoB",
            "num_labels": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "total_clean_labeled": len(clean_df),
            "total_conflicts_dropped": len(conflicts_df),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "labeled": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    (out_dir / "splits_v3.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    clean_df.to_csv(out_dir / "labeled_clean_v3.csv", index=False)

    # Strictly nested subsets: 10% ⊂ 25% ⊂ 50% ⊂ 100%
    train_df = clean_df.loc[train_idx].copy().reset_index()
    train_100 = sorted(train_df["index"].tolist())
    sub_50, _ = train_test_split(train_df, train_size=0.50, stratify=train_df["label"], random_state=42)
    train_50 = sorted(sub_50["index"].tolist())
    sub_25, _ = train_test_split(sub_50, train_size=0.50, stratify=sub_50["label"], random_state=42)
    train_25 = sorted(sub_25["index"].tolist())
    sub_10, _ = train_test_split(sub_25, train_size=0.40, stratify=sub_25["label"], random_state=42)
    train_10 = sorted(sub_10["index"].tolist())

    assert set(train_10).issubset(set(train_25)) and set(train_25).issubset(set(train_50)) and set(train_50).issubset(set(train_100))

    for frac, idxs in [(10, train_10), (25, train_25), (50, train_50), (100, train_100)]:
        (out_dir / f"train_frac_{frac}.json").write_text(
            json.dumps({
                "fraction": frac,
                "count": len(idxs),
                "indices": idxs,
                "class_distribution": clean_df.loc[idxs, "label"].value_counts().to_dict(),
            }, indent=2),
            encoding="utf-8",
        )

    c_tr = clean_df.loc[train_idx, "label"].value_counts().sort_index().to_dict()
    c_va = clean_df.loc[val_idx, "label"].value_counts().sort_index().to_dict()
    c_te = clean_df.loc[test_idx, "label"].value_counts().sort_index().to_dict()
    report = f"""# Leakage & Data Audit Report — SentNoB (Phase H)

**Source:** `khondoker/SentNoB` (3-class Sentiment Analysis on Noisy Bangla Texts, Islam et al., EMNLP 2021)  
**Deduplication:** MinHash LSH ($J \\ge 0.9$, 128 permutations)  

| Split | Total | Neutral (0) | Positive (1) | Negative (2) |
|---|---:|---:|---:|---:|
| **Train (80%)** | {len(train_idx):,} | {c_tr.get(0,0):,} | {c_tr.get(1,0):,} | {c_tr.get(2,0):,} |
| **Val (10%)** | {len(val_idx):,} | {c_va.get(0,0):,} | {c_va.get(1,0):,} | {c_va.get(2,0):,} |
| **Test (10%)** | {len(test_idx):,} | {c_te.get(0,0):,} | {c_te.get(1,0):,} | {c_te.get(2,0):,} |

- **Conflicts Dropped:** {len(conflicts_df)} rows ({len(conflicting_ids)} groups)
- **Cross-Split Leakage:** **0 groups (0.00%)**
"""
    (exp_dir / "leakage_report.md").write_text(report, encoding="utf-8")
    logger.info("Saved SentNoB v3 splits and leakage report to %s", out_dir)


if __name__ == "__main__":
    generate_sentnob_splits()
