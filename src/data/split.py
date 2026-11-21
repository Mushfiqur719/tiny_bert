"""
src/data/split.py — Fixed stratified 80/10/10 train/val/test split.

The split is performed **once**, saved as integer row-indices (into the
cleaned, labeled DataFrame) to ``data/processed/splits.json``, and
reloaded by every downstream script.  The test set is never exposed
during training or hyperparameter tuning.

Format of ``splits.json``
--------------------------
{
  "metadata": {
    "created_at": "...",
    "seed": 42,
    "total_labeled": 15406,
    "total_unlabeled": 65405,
    "train_ratio": 0.8,
    "val_ratio":   0.1,
    "test_ratio":  0.1
  },
  "labeled": {
    "train":      [0, 5, 12, ...],   # row indices into cleaned labeled_df
    "val":        [3, 9, ...],
    "test":       [1, 7, ...]
  },
  "unlabeled": {
    "all":        [0, 1, 2, ...]     # row indices into unlabeled_df
  }
}

Usage
-----
    from src.data.split import create_splits, load_splits, apply_splits

    # First time:
    create_splits(labeled_df, unlabeled_df, cfg, seed=42)

    # Every subsequent time:
    splits = load_splits(cfg)
    train_df, val_df, test_df = apply_splits(labeled_df, splits)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.test_guard import assert_test_access, TestSetLocked

logger = logging.getLogger(__name__)


def create_splits(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    cfg,
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> dict:
    """Create and save the fixed stratified split.

    Args:
        labeled_df: Cleaned labeled DataFrame (columns: headline, label).
        unlabeled_df: Cleaned unlabeled DataFrame.
        cfg: OmegaConf config (reads data.splits_file, data.{train,val,test}_ratio).
        seed: Random seed (falls back to cfg.seed if None).
        overwrite: If False and splits_file already exists, raises FileExistsError.

    Returns:
        The splits dict (same format as stored in JSON).

    Raises:
        FileExistsError: If splits_file exists and overwrite=False.
    """
    splits_path = Path(cfg.data.splits_file)
    if splits_path.exists() and not overwrite:
        raise FileExistsError(
            f"Splits file already exists: {splits_path}. "
            "Pass overwrite=True to regenerate."
        )

    seed = int(seed if seed is not None else cfg.seed)
    train_r = float(cfg.data.train_ratio)
    val_r   = float(cfg.data.val_ratio)
    test_r  = float(cfg.data.test_ratio)
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_r + val_r + test_r}"

    indices = labeled_df.index.tolist()
    labels  = labeled_df["label"].tolist()

    # First split: train vs (val + test), stratified by label
    val_test_ratio = val_r + test_r
    train_idx, valtest_idx, _, valtest_labels = train_test_split(
        indices,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test, stratified by label.
    # Fallback to non-stratified if the pool is too small for stratification
    # (sklearn requires >= 2 samples per class in both resulting sets).
    relative_test_ratio = test_r / val_test_ratio
    _min_class_count = min(
        sum(1 for l in valtest_labels if l == c) for c in set(valtest_labels)
    )
    _can_stratify = _min_class_count >= 2
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=relative_test_ratio,
        stratify=valtest_labels if _can_stratify else None,
        random_state=seed,
    )
    if not _can_stratify:
        logger.warning(
            "Val+test pool too small for stratification (min class count=%d). "
            "Using random split. This only affects tiny test fixtures — "
            "real BanglaBait data will always stratify correctly.",
            _min_class_count,
        )

    splits = {
        "metadata": {
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "seed":             seed,
            "total_labeled":    len(labeled_df),
            "total_unlabeled":  len(unlabeled_df),
            "train_ratio":      train_r,
            "val_ratio":        val_r,
            "test_ratio":       test_r,
            "train_size":       len(train_idx),
            "val_size":         len(val_idx),
            "test_size":        len(test_idx),
        },
        "labeled": {
            "train": sorted(train_idx),
            "val":   sorted(val_idx),
            "test":  sorted(test_idx),
        },
        "unlabeled": {
            "all": unlabeled_df.index.tolist(),
        },
    }

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    _log_split_stats(labeled_df, splits)
    logger.info("Splits saved to %s", splits_path)
    return splits


def load_splits(cfg) -> dict:
    """Load an existing splits.json file.

    Args:
        cfg: OmegaConf config (reads data.splits_file).

    Returns:
        Splits dict.

    Raises:
        FileNotFoundError: If splits_file does not exist.
    """
    splits_path = Path(cfg.data.splits_file)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}. "
            "Run create_splits() first or run: python -m scripts.prepare_data"
        )
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    logger.info(
        "Loaded splits from %s (seed=%s, created=%s)",
        splits_path,
        splits["metadata"].get("seed"),
        splits["metadata"].get("created_at"),
    )
    return splits


def apply_splits(
    labeled_df: pd.DataFrame,
    splits: dict,
    split: Optional[str] = None,
) -> Union[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Slice *labeled_df* using the indices in *splits*.

    Args:
        labeled_df: The cleaned labeled DataFrame (must be indexed 0..N-1).
        splits: Dict as returned by ``load_splits`` / ``create_splits``.
        split: Optional split selector ('train', 'val', or 'test').
               If 'test', assert_test_access() is strictly enforced.

    Returns:
        If split is provided, returns that single DataFrame.
        If split is None, returns (train_df, val_df, test_df).
    """
    if split == "train":
        return labeled_df.loc[splits["labeled"]["train"]].reset_index(drop=True)
    elif split == "val":
        return labeled_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    elif split == "test":
        assert_test_access()
        return labeled_df.loc[splits["labeled"]["test"]].reset_index(drop=True)

    train_df = labeled_df.loc[splits["labeled"]["train"]].reset_index(drop=True)
    val_df   = labeled_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    test_df  = labeled_df.loc[splits["labeled"]["test"]].reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ── Helpers ────────────────────────────────────────────────────────────────

def _log_split_stats(labeled_df: pd.DataFrame, splits: dict) -> None:
    for split_name in ("train", "val", "test"):
        idx = splits["labeled"][split_name]
        sub = labeled_df.loc[idx]
        counts = sub["label"].value_counts().sort_index()
        total = len(sub)
        parts = [f"  {split_name} (n={total}):"]
        for cls, cnt in counts.items():
            parts.append(f"    label={cls}: {cnt} ({cnt/total:.1%})")
        logger.info("\n".join(parts))
