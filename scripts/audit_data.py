"""
scripts/audit_data.py — Phase A data-validity audit.

Checks:
  1. Exact + near-duplicate headlines across train/val/test (BanglaBait).
  2. Exact + near-duplicate overlap between BanglaBait (all splits) and BaitBuster-Bangla.
  3. Tokenizer compatibility: BanglaBERT vs BanglaBERT-Small (vocab identity).
  4. Label definition sanity: class balance in both datasets.

Usage:
    python -m scripts.audit_data --config configs/base.yaml
    python -m scripts.audit_data --no-crossdomain      # skip BaitBuster download
Output: experiments/audit/data_audit.md and data_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.load_crossdomain import load_and_clean_crossdomain_dataset
from src.data.split import apply_splits, load_splits
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    """Normalization for duplicate detection.

    Keeps letters (L), digits (N) AND combining marks (M: Bengali vowel signs, virama,
    chandrabindu). str.isalnum() would wrongly drop those marks and make different
    Bangla words collide. Punctuation/whitespace/symbols are removed.
    """
    s = unicodedata.normalize("NFC", str(text)).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in ("L", "N", "M"))


def _shingles(text: str, n: int = 4) -> set[str]:
    s = _norm(text)
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _near_duplicate_pairs(a: list[str], b: list[str], threshold: float = 0.9) -> list[tuple[int, int, float]]:
    """Return (i, j, jaccard) for char-4gram Jaccard >= threshold. Uses an inverted index."""
    b_sh = [_shingles(t) for t in b]
    index: dict[str, list[int]] = {}
    for j, sh in enumerate(b_sh):
        for g in sh:
            index.setdefault(g, []).append(j)
    pairs = []
    for i, t in enumerate(a):
        sh = _shingles(t)
        counts: dict[int, int] = {}
        for g in sh:
            for j in index.get(g, ()):
                counts[j] = counts.get(j, 0) + 1
        for j, inter in counts.items():
            union = len(sh) + len(b_sh[j]) - inter
            jac = inter / union if union else 0.0
            if jac >= threshold:
                pairs.append((i, j, round(jac, 3)))
    return pairs


def audit_splits(train_df, val_df, test_df) -> tuple[dict, dict]:
    """Return (summary, leaky_indices). leaky_indices lists positions (within val/test)
    whose headline duplicates / near-duplicates a TRAIN headline."""
    res = {}
    leaky: dict[str, list[int]] = {}
    tr, va, te = (d["headline"].tolist() for d in (train_df, val_df, test_df))
    tr_labels = train_df["label"].tolist()

    for name, a, b, b_df in (
        ("train_vs_val", tr, va, val_df),
        ("train_vs_test", tr, te, test_df),
        ("val_vs_test", va, te, test_df),
    ):
        a_norm = {_norm(t) for t in a}
        exact_idx = {i for i, t in enumerate(b) if _norm(t) in a_norm}
        near = _near_duplicate_pairs(b, a, 0.9)
        near_idx = {i for i, _, _ in near}
        all_idx = sorted(exact_idx | near_idx)
        res[name] = {
            "exact_duplicates_in_second": len(exact_idx),
            "near_duplicate_items_in_second": len(near_idx),
            "union_items": len(all_idx),
            "pct_of_second": round(100 * len(all_idx) / max(len(b), 1), 2),
        }
        if name == "train_vs_val":
            leaky["val"] = all_idx
        elif name == "train_vs_test":
            leaky["test"] = all_idx
            # Do duplicated test items carry the same label as their train twin?
            train_map: dict[str, set[int]] = {}
            for t, l in zip(tr, tr_labels):
                train_map.setdefault(_norm(t), set()).add(int(l))
            agree = [int(int(b_df["label"].iloc[i]) in train_map[_norm(b[i])]) for i in exact_idx]
            res[name]["exact_dup_label_agreement"] = round(float(np.mean(agree)), 3) if agree else None
        logger.info("%s: %s", name, res[name])

    # Within-train duplicates and label conflicts
    norm_to_labels: dict[str, set[int]] = {}
    for t, l in zip(train_df["headline"], train_df["label"]):
        norm_to_labels.setdefault(_norm(t), set()).add(int(l))
    res["train_internal_duplicates"] = int(len(train_df) - len(norm_to_labels))
    res["train_label_conflicts"] = int(sum(1 for v in norm_to_labels.values() if len(v) > 1))
    return res, leaky


def audit_crossdomain(labeled_df, cross_df) -> dict:
    base = labeled_df["headline"].tolist()
    cross = cross_df["headline"].tolist()
    base_norm = {_norm(t) for t in base}
    exact_mask = np.array([_norm(t) in base_norm for t in cross])
    near = _near_duplicate_pairs(cross, base, 0.9)
    near_idx = sorted({i for i, _, _ in near})
    overlap_idx = set(np.where(exact_mask)[0].tolist()) | set(near_idx)

    res = {
        "n_crossdomain": len(cross),
        "exact_overlap": int(exact_mask.sum()),
        "near_overlap_items": len(near_idx),
        "total_overlapping_items": len(overlap_idx),
        "overlap_pct": round(100 * len(overlap_idx) / max(len(cross), 1), 2),
        "crossdomain_clickbait_pct": round(100 * float((cross_df["label"] == 1).mean()), 2),
        "banglabait_clickbait_pct": round(100 * float((labeled_df["label"] == 1).mean()), 2),
    }
    # Label agreement on overlapping items (are the label definitions consistent?)
    if near:
        base_lab = labeled_df["label"].tolist()
        agree = [int(cross_df["label"].iloc[i] == base_lab[j]) for i, j, _ in near]
        res["label_agreement_on_overlap"] = round(float(np.mean(agree)), 3)
    # Save non-overlapping clean subset indices for a leakage-free evaluation
    res["_clean_indices"] = [i for i in range(len(cross)) if i not in overlap_idx]
    logger.info("Cross-domain overlap: %s", {k: v for k, v in res.items() if not k.startswith("_")})
    return res


def audit_tokenizers(teacher_name: str, small_name: str) -> dict:
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(teacher_name)
    s = AutoTokenizer.from_pretrained(small_name)
    tv, sv = t.get_vocab(), s.get_vocab()
    same_size = len(tv) == len(sv)
    same_map = tv == sv
    samples = ["জানেন কি কীভাবে দ্রুত কোটিপতি হবেন!", "মন্ত্রী জানালেন সরকারের নতুন উদ্যোগ", "আপনি বিশ্বাস করবেন না এই ভিডিও"]
    same_ids = all(t(x)["input_ids"] == s(x)["input_ids"] for x in samples)
    res = {
        "teacher_vocab_size": len(tv),
        "small_vocab_size": len(sv),
        "same_vocab_size": same_size,
        "identical_token_to_id_map": same_map,
        "identical_ids_on_samples": same_ids,
        "verdict": "OK: tokenizers are interchangeable" if (same_map and same_ids)
        else "MISMATCH: banglabert_small must use its OWN tokenizer (embeddings are misaligned otherwise)",
    }
    logger.info("Tokenizer audit: %s", res["verdict"])
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--crossdomain-id", default="abdalimran/BaitBuster-Bangla")
    p.add_argument("--no-crossdomain", action="store_true")
    p.add_argument("--no-tokenizer", action="store_true")
    p.add_argument("--output-dir", default="experiments/audit")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labeled_df, _ = load_raw(cfg, hf_dataset_id=cfg.data.hf_dataset_id)
    labeled_df = clean_dataframe(labeled_df, headline_col=cfg.data.text_col, min_char_length=int(cfg.data.min_char_length))
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)
    train_df, val_df, test_df = apply_splits(labeled_df, load_splits(cfg))

    split_res, leaky = audit_splits(train_df, val_df, test_df)
    report: dict = {"splits": split_res}
    (out / "leaky_indices.json").write_text(json.dumps(leaky), encoding="utf-8")

    if not args.no_crossdomain:
        cross_df = load_and_clean_crossdomain_dataset(args.crossdomain_id)
        cd = audit_crossdomain(labeled_df, cross_df)
        clean_idx = cd.pop("_clean_indices")
        cross_df.iloc[clean_idx].reset_index(drop=True).to_csv(out / "crossdomain_deduplicated.csv", index=False)
        report["crossdomain"] = cd
        report["crossdomain"]["deduplicated_size"] = len(clean_idx)

    if not args.no_tokenizer:
        report["tokenizer"] = audit_tokenizers(cfg.model.teacher_model_name, "csebuetnlp/banglabert_small")

    (out / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Data Audit Report", "", "```json", json.dumps(report, indent=2), "```"]
    (out / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    logger.info("Audit written to %s", out)


if __name__ == "__main__":
    main()
