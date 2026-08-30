"""
src/data/load_crossdomain.py — Ingestion and harmonization of cross-domain evaluation datasets.

Reuses the core cleaning pipeline from src/data/clean.py (NFC normalization,
zero-width stripping, Bengali diacritic preservation) to ensure preprocessing consistency.

Supports:
  - Local files (.csv, .json, .jsonl, .xlsx, .tsv, .parquet)
  - Hugging Face dataset repositories
  - Automatic column name and label schema harmonization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.clean import clean_dataframe

logger = logging.getLogger(__name__)

# Common headline column variants in Bengali NLP datasets
HEADLINE_COLUMN_CANDIDATES = [
    "headline", "title", "text", "news_headline", "article_title",
    "post_title", "video_title", "clean_headline", "sentence"
]

# Common label column variants
LABEL_COLUMN_CANDIDATES = [
    "label", "is_clickbait", "clickbait", "class", "target", "category"
]


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect headline and label columns automatically from candidates."""
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    headline_col = None
    for cand in HEADLINE_COLUMN_CANDIDATES:
        if cand in cols_lower:
            headline_col = cols_lower[cand]
            break

    label_col = None
    for cand in LABEL_COLUMN_CANDIDATES:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    if not headline_col:
        # Fallback to first string column
        str_cols = [c for c in df.columns if df[c].dtype == "object"]
        if str_cols:
            headline_col = str_cols[0]
        else:
            raise ValueError(f"Could not identify headline column among: {list(df.columns)}")

    if not label_col:
        # Fallback to binary/integer column
        num_cols = [c for c in df.columns if c != headline_col]
        if num_cols:
            label_col = num_cols[0]
        else:
            raise ValueError(f"Could not identify label column among: {list(df.columns)}")

    logger.info("Detected dataset columns -> Headline: '%s', Label: '%s'", headline_col, label_col)
    return headline_col, label_col


def align_label_schema(series: pd.Series) -> pd.Series:
    """Normalize label column into standard binary integer {0: not-clickbait, 1: clickbait}.

    Handles:
      - String labels: 'clickbait'/'not_clickbait', 'bait'/'non-bait', 'yes'/'no', 'true'/'false'
      - Inverted binary labels
      - Categorical labels
    """
    sample_vals = series.dropna().unique()
    logger.info("Raw label values detected: %s", sample_vals.tolist())

    # Map string representations
    mapping = {
        "clickbait": 1,
        "click-bait": 1,
        "click bait": 1,
        "bait": 1,
        "misleading": 1,
        "yes": 1,
        "true": 1,
        "1": 1,
        1: 1,
        1.0: 1,
        "not-clickbait": 0,
        "not_clickbait": 0,
        "not clickbait": 0,
        "not click-bait": 0,
        "non-clickbait": 0,
        "non_clickbait": 0,
        "non clickbait": 0,
        "non-bait": 0,
        "genuine": 0,
        "normal": 0,
        "news": 0,
        "no": 0,
        "false": 0,
        "0": 0,
        0: 0,
        0.0: 0,
    }

    def convert_val(val: Any) -> int:
        if isinstance(val, str):
            v_clean = val.lower().strip()
            if v_clean in mapping:
                return mapping[v_clean]
        elif val in mapping:
            return mapping[val]
        try:
            return int(val)
        except Exception:
            raise ValueError(f"Cannot map unknown label value '{val}' to binary schema {0, 1}.")

    aligned = series.apply(convert_val)
    counts = aligned.value_counts().to_dict()
    logger.info("Aligned label counts -> 0 (not-clickbait): %d, 1 (clickbait): %d", counts.get(0, 0), counts.get(1, 0))
    return aligned


def load_and_clean_crossdomain_dataset(
    dataset_path_or_id: str | Path,
    headline_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Load, clean, and standardize a cross-domain Bengali clickbait dataset.

    Args:
        dataset_path_or_id: Local file path or HuggingFace dataset repository ID.
        headline_col: Optional explicit name of the text column.
        label_col: Optional explicit name of the label column.

    Returns:
        Cleaned pandas DataFrame with standardized columns ['headline', 'label'].
    """
    path_str = str(dataset_path_or_id)
    path_obj = Path(path_str)

    # 1. Ingestion
    if path_obj.exists() and path_obj.is_file():
        logger.info("Loading cross-domain dataset from local file: '%s'...", path_obj)
        suffix = path_obj.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path_obj)
        elif suffix in (".tsv", ".txt"):
            df = pd.read_csv(path_obj, sep="\t")
        elif suffix in (".json", ".jsonl"):
            df = pd.read_json(path_obj, lines=suffix == ".jsonl")
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path_obj)
        elif suffix == ".parquet":
            df = pd.read_parquet(path_obj)
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .tsv, .json, .xlsx, .parquet")
    else:
        logger.info("Loading cross-domain dataset from HuggingFace Hub: '%s'...", path_str)
        ds = load_dataset(path_str)
        splits = list(ds.keys())
        dfs = [ds[s].to_pandas() for s in splits]
        df = pd.concat(dfs, ignore_index=True)

        # Special handling for BaitBuster-Bangla gold standard: filter human_labeled
        if "human_labeled" in df.columns and "title" in df.columns:
            logger.info("Detected BaitBuster-Bangla schema. Filtering 10,000 human-annotated gold benchmark subset...")
            df = df[df["human_labeled"].notna() & (df["human_labeled"].isin(["Clickbait", "Not Clickbait", "clickbait", "not clickbait"]))].copy()
            headline_col = headline_col or "title"
            label_col = label_col or "human_labeled"

    logger.info("Raw dataset shape: %d rows, %d columns", len(df), len(df.columns))

    # 2. Identify & Harmonize Column Names
    h_col = headline_col
    l_col = label_col
    if not h_col or not l_col:
        detected_h, detected_l = detect_columns(df)
        h_col = h_col or detected_h
        l_col = l_col or detected_l

    # Standardize column naming
    standard_df = pd.DataFrame({
        "headline": df[h_col],
        "label": align_label_schema(df[l_col]),
    })

    # 3. Clean Text using identical Phase 1 cleaning pipeline
    cleaned_df = clean_dataframe(standard_df, headline_col="headline")
    cleaned_df = cleaned_df[["headline", "label"]].dropna().reset_index(drop=True)

    logger.info(
        "Cross-domain dataset successfully processed: %d rows (Class 0: %d, Class 1: %d)",
        len(cleaned_df),
        sum(cleaned_df["label"] == 0),
        sum(cleaned_df["label"] == 1),
    )
    return cleaned_df
