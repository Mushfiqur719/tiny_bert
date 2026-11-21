"""scripts/make_splits_v3_baitbuster_channel.py — Channel-disjoint v3 splits for BaitBuster-Bangla.

Groups strictly by YouTube `channel_id` (54 unique channels) using StratifiedGroupKFold
so ZERO channels span train, validation, or test splits — testing whether models learn
genuine clickbait linguistics rather than channel style.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.clean import clean_dataframe
from src.data.load_crossdomain import align_label_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_channel_disjoint_splits() -> None:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    out_dir = Path("data/processed/v3_baitbuster_ch")
    exp_dir = Path("experiments/v3_baitbuster_ch")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("abdalimran/BaitBuster-Bangla")
    df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    # Filter 10,000 human-annotated gold benchmark rows
    valid_labels = ["Clickbait", "Not Clickbait", "clickbait", "not clickbait"]
    df = df[df["human_labeled"].notna() & df["human_labeled"].isin(valid_labels)].copy().reset_index(drop=True)
    df["headline"] = df["title"].astype(str)
    df["label"] = align_label_schema(df["human_labeled"]).astype(int)
    df["group_id"] = df["channel_id"].astype(str)

    df = clean_dataframe(df, headline_col="headline", min_char_length=5)
    df = df[["headline", "label", "group_id", "channel_name"]].reset_index(drop=True)
    logger.info("Loaded %d gold BaitBuster rows across %d unique channels.", len(df), df["group_id"].nunique())

    # 5-fold StratifiedGroupKFold by channel_id (54 channels -> ~11 channels per fold: 1 test, 1 val, 3 train -> 60/20/20)
    # Or 10-fold (~5-6 channels per fold: 1 test, 1 val, 8 train -> 80/10/10)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    test_idx, val_idx, train_idx = None, None, []
    for fold_num, (_, fold_indices) in enumerate(sgkf.split(df, df["label"], df["group_id"])):
        if fold_num == 0:
            test_idx = fold_indices.tolist()
        elif fold_num == 1:
            val_idx = fold_indices.tolist()
        else:
            train_idx.extend(fold_indices.tolist())

    assert test_idx is not None and val_idx is not None

    c_tr = set(df.loc[train_idx, "group_id"])
    c_va = set(df.loc[val_idx, "group_id"])
    c_te = set(df.loc[test_idx, "group_id"])
    assert len(c_tr & c_va) == 0 and len(c_tr & c_te) == 0 and len(c_va & c_te) == 0
    logger.info(
        "ZERO channel overlap verified! Channels -> Train: %d, Val: %d, Test: %d",
        len(c_tr), len(c_va), len(c_te),
    )

    splits = {
        "metadata": {
            "dataset": "abdalimran/BaitBuster-Bangla (channel-disjoint)",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": 42,
            "train_channels": len(c_tr),
            "val_channels": len(c_va),
            "test_channels": len(c_te),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "labeled": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    (out_dir / "splits_v3.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "labeled_clean_v3.csv", index=False)

    # Nested training subsets
    train_df = df.loc[train_idx].copy().reset_index()
    train_100_idx = sorted(train_df["index"].tolist())
    sub_50, _ = train_test_split(train_df, train_size=0.50, stratify=train_df["label"], random_state=42)
    train_50_idx = sorted(sub_50["index"].tolist())
    sub_25, _ = train_test_split(sub_50, train_size=0.50, stratify=sub_50["label"], random_state=42)
    train_25_idx = sorted(sub_25["index"].tolist())
    sub_10, _ = train_test_split(sub_25, train_size=0.40, stratify=sub_25["label"], random_state=42)
    train_10_idx = sorted(sub_10["index"].tolist())

    for frac, idxs in [(10, train_10_idx), (25, train_25_idx), (50, train_50_idx), (100, train_100_idx)]:
        (out_dir / f"train_frac_{frac}.json").write_text(
            json.dumps({"fraction": frac, "count": len(idxs), "indices": idxs}, indent=2),
            encoding="utf-8",
        )

    report = f"""# Channel-Disjoint Split Audit — BaitBuster-Bangla

| Split | Rows | Unique Channels | Clickbait Prevalence |
|---|---:|---:|---:|
| **Train** | {len(train_idx):,} | {len(c_tr)} | {df.loc[train_idx, 'label'].mean():.2%} |
| **Val** | {len(val_idx):,} | {len(c_va)} | {df.loc[val_idx, 'label'].mean():.2%} |
| **Test** | {len(test_idx):,} | {len(c_te)} | {df.loc[test_idx, 'label'].mean():.2%} |

- **Cross-Split Channel Overlap:** **0 channels (0.00%)**
"""
    (exp_dir / "leakage_report.md").write_text(report, encoding="utf-8")
    logger.info("Saved channel-disjoint BaitBuster splits to %s", out_dir)


if __name__ == "__main__":
    generate_channel_disjoint_splits()
