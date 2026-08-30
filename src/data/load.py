"""
src/data/load.py — Raw data loading for the BanglaBait dataset.

Pulls the dataset from the HuggingFace Hub (``Motahar/banglabait``),
concatenates *all* splits into a single DataFrame, then separates
labeled from unlabeled rows.

Design decisions
----------------
* We pool every HF split and re-split ourselves so we fully own the
  train/val/test boundary (see ``src/data/split.py``).
* The ``headline`` column maps to the ``title`` field in the HF dataset.
* Label encoding: 0 = not-clickbait, 1 = clickbait (matches original
  int labels in the dataset).
* Unlabeled rows (label == -1 in the source) are separated out and
  returned so the caller can optionally use them for semi-supervised
  training.  Their indices are stored in ``splits.json`` later.

Usage
-----
    from src.data.load import load_raw

    labeled_df, unlabeled_df = load_raw(cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# HF hub id — can be overridden via config
DEFAULT_HF_DATASET_ID = "Motahar/banglabait"

# The HF dataset uses -1 for unlabeled rows
_UNLABELED_SENTINEL = -1


def load_raw(
    cfg,  # OmegaConf DictConfig
    *,
    hf_dataset_id: Optional[str] = None,
    local_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BanglaBait and return (labeled_df, unlabeled_df).

    Both DataFrames have columns: ``[headline, label]``.
    ``labeled_df`` has label ∈ {0, 1}.
    ``unlabeled_df`` has label == -1 (kept as-is for semi-supervised use).

    Args:
        cfg: OmegaConf config (reads ``data.raw_dir`` for caching).
        hf_dataset_id: Override the HF hub id (default: Motahar/banglabait).
        local_path: If provided, load from a local CSV / parquet instead of HF.
        cache_dir: HF datasets cache directory (defaults to HF default).

    Returns:
        Tuple of (labeled_df, unlabeled_df).

    Raises:
        AssertionError: If labeled_df has nulls or fewer than 2 classes.
    """
    dataset_id = hf_dataset_id or DEFAULT_HF_DATASET_ID
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if local_path is not None:
        df = _load_local(Path(local_path))
    else:
        df = _load_from_hub(dataset_id, cache_dir=cache_dir)

    # Normalise to [headline, label] schema
    df = _normalise_schema(df, cfg)

    # Split labeled vs unlabeled
    labeled_df = df[df["label"] != _UNLABELED_SENTINEL].copy().reset_index(drop=True)
    unlabeled_df = df[df["label"] == _UNLABELED_SENTINEL].copy().reset_index(drop=True)

    logger.info(
        "Raw load complete: %d labeled rows, %d unlabeled rows",
        len(labeled_df),
        len(unlabeled_df),
    )

    # ── Integrity checks ──────────────────────────────────────────────────
    assert labeled_df["headline"].notna().all(), "Null headlines found in labeled data!"
    assert labeled_df["label"].notna().all(), "Null labels found in labeled data!"
    assert labeled_df["headline"].str.strip().ne("").all(), \
        "Empty (whitespace-only) headlines found in labeled data!"

    # Log class balance
    _log_class_balance(labeled_df, split_name="labeled")

    # Non-degenerate check: neither class < 5% of total
    value_counts = labeled_df["label"].value_counts(normalize=True)
    for cls, frac in value_counts.items():
        assert frac >= 0.05, (
            f"Class {cls} is degenerate: only {frac:.1%} of labeled data. "
            "Check your data source."
        )

    return labeled_df, unlabeled_df


# ── Private helpers ────────────────────────────────────────────────────────

def _load_from_hub(dataset_id: str, cache_dir=None) -> pd.DataFrame:
    """Download all HF splits and concatenate into one DataFrame."""
    from datasets import load_dataset  # local import — optional dep at import time

    logger.info("Loading dataset '%s' from HuggingFace Hub...", dataset_id)
    ds = load_dataset(dataset_id, cache_dir=cache_dir)

    dfs = []
    for split_name, split_ds in ds.items():
        split_df = split_ds.to_pandas()
        split_df["_hf_split"] = split_name   # track origin for debugging
        dfs.append(split_df)
        logger.info("  HF split '%s': %d rows", split_name, len(split_df))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Total rows after concat: %d", len(combined))
    return combined


def _load_local(path: Path) -> pd.DataFrame:
    """Load from a local file (CSV or Parquet)."""
    logger.info("Loading dataset from local path: %s", path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported local file format: {suffix}. Use .csv or .parquet.")


def _normalise_schema(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Map HF column names → [headline, label] and cast label to int."""
    text_col = cfg.data.get("text_col", "headline")
    label_col = cfg.data.get("label_col", "label")

    # BanglaBait uses 'title' for the headline column on HF Hub
    if text_col not in df.columns and "title" in df.columns:
        logger.debug("Mapping 'title' → 'headline'")
        df = df.rename(columns={"title": "headline"})
    elif text_col != "headline":
        df = df.rename(columns={text_col: "headline"})

    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. Available: {list(df.columns)}"
        )
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # Cast to int (HF sometimes loads as float due to NaNs in unlabeled rows)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)

    return df[["headline", "label"]].copy()


def _log_class_balance(df: pd.DataFrame, split_name: str = "") -> None:
    """Log class counts and percentages."""
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    lines = [f"\nClass balance ({split_name}, n={total}):"]
    label_names = {0: "not-clickbait", 1: "clickbait", -1: "unlabeled"}
    for cls, cnt in counts.items():
        name = label_names.get(int(cls), str(cls))
        lines.append(f"  label={cls} ({name}): {cnt:,}  ({cnt / total:.1%})")
    logger.info("\n".join(lines))
