"""scripts/attach_v3_bodies.py — Attach cleaned article body text ('content') to v3 splits.

Guarantees exact 1:1 row-index alignment with data/processed/v3/splits_v3.json
and data/processed/v3/labeled_clean_v3.csv.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.data.clean import clean_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def clean_body_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # plenty for max_length=256 tokens


def attach_bodies() -> None:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    v3_dir = Path("data/processed/v3")
    old_clean_path = v3_dir / "labeled_clean_v3.csv"
    conflicts_path = v3_dir / "conflicts.csv"
    out_path = v3_dir / "labeled_clean_v3_full.csv"

    assert old_clean_path.exists(), "Run scripts.make_splits_v3 first!"
    old_df = pd.read_csv(old_clean_path)

    ds = load_dataset("Motahar/banglabait")
    dfs = [split_ds.to_pandas() for split_ds in ds.values()]
    raw_df = pd.concat(dfs, ignore_index=True)

    # Match src/data/load.py schema normalization
    raw_df = raw_df.rename(columns={"title": "headline"})
    raw_df["label"] = pd.to_numeric(raw_df["label"], errors="coerce").fillna(-1).astype(int)
    raw_df = raw_df[raw_df["label"] != -1].copy().reset_index(drop=True)

    # Clean headline identically to make_splits_v3.py
    raw_df = clean_dataframe(raw_df, headline_col="headline", min_char_length=5)
    raw_df = raw_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    # Clean content and categories columns
    raw_df["content"] = raw_df["content"].fillna("").apply(clean_body_text)
    raw_df["has_body"] = (raw_df["content"].str.len() > 20).astype(int)
    raw_df["categories"] = raw_df["categories"].fillna("General").astype(str).str.strip()

    matched_contents = []
    matched_has_body = []
    matched_cats = []
    raw_iter_map: dict[tuple[str, int], list[tuple[str, int, str]]] = {}
    for h, l, c, hb, cat in zip(
        raw_df["headline"], raw_df["label"], raw_df["content"], raw_df["has_body"], raw_df["categories"]
    ):
        raw_iter_map.setdefault((h, int(l)), []).append((c, int(hb), cat))

    for h, l in zip(old_df["headline"], old_df["label"]):
        key = (h, int(l))
        assert key in raw_iter_map and len(raw_iter_map[key]) > 0, f"Row mismatch for headline: {h[:40]}"
        c, hb, cat = raw_iter_map[key].pop(0)
        matched_contents.append(c)
        matched_has_body.append(hb)
        matched_cats.append(cat)

    full_df = old_df.copy()
    full_df["content"] = matched_contents
    full_df["has_body"] = matched_has_body
    full_df["categories"] = matched_cats

    # Strict 1:1 alignment verification
    assert len(full_df) == len(old_df)
    assert (full_df["headline"] == old_df["headline"]).all()
    assert (full_df["label"] == old_df["label"]).all()

    full_df.to_csv(out_path, index=False)
    logger.info(
        "Saved %d rows with article bodies (%.1f%% non-empty) to %s",
        len(full_df),
        100.0 * full_df["has_body"].mean(),
        out_path,
    )


if __name__ == "__main__":
    attach_bodies()
