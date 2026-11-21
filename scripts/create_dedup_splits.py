"""
scripts/create_dedup_splits.py — Create a leak-free deduplicated 80/10/10 split
(`data/processed/splits_dedup.json`) while keeping `data/processed/splits.json` intact,
and evaluate existing trained checkpoints on both the original test split (N=1,506)
and the leak-free test subset (N=1,351).

What this script does:
  1. Loads BanglaBait using the exact same cleaning pipeline (`clean_dataframe`).
  2. Groups rows by normalized headline (`_norm`) and merges any near-duplicate clusters
     (char 4-gram Jaccard >= 0.90) using connected components:
       - If a cluster has conflicting labels across duplicate rows, resolves by majority
         vote (or drops exact 50/50 ties) and keeps a single canonical representative row.
  3. Performs a stratified 80/10/10 train/val/test split on the unique canonical rows
     and saves indices (into `labeled_df`) to `data/processed/splits_dedup.json`.
  4. Verifies 0 exact and 0 near-duplicate overlap across the new train/val/test splits.
  5. Evaluates any existing checkpoints (`teacher`, `student_small_pretrained_distilled`,
     `student_teacher_init_distilled`) on the original test set (N=1,506) vs. the
     leak-free test subset (dropping the 155 train-leaked items) so we can immediately
     see the exact leakage impact.

Usage:
    .venv\\Scripts\\python.exe -m scripts.create_dedup_splits
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.audit_data import _near_duplicate_pairs, _norm, audit_splits
from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.evaluation.metrics import compute_classification_metrics
from src.utils import load_config, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def deduplicate_indices(labeled_df) -> tuple[list[int], dict]:
    """Return canonical row indices into labeled_df after removing exact & near duplicates
    and resolving label conflicts."""
    headlines = labeled_df["headline"].tolist()
    labels = labeled_df["label"].tolist()
    n = len(headlines)

    # Step 1: Group by exact normalized text
    norm_groups: dict[str, list[int]] = {}
    for idx, text in enumerate(headlines):
        key = _norm(text)
        if not key:
            continue
        norm_groups.setdefault(key, []).append(idx)

    canonical_indices: list[int] = []
    dropped_ties = 0
    resolved_conflicts = 0

    for key, member_indices in norm_groups.items():
        member_labels = [int(labels[i]) for i in member_indices]
        counts = Counter(member_labels)
        if len(counts) > 1:
            most_common = counts.most_common(2)
            if most_common[0][1] == most_common[1][1]:
                # Exact tie between label 0 and label 1 -> ambiguous, drop all
                dropped_ties += len(member_indices)
                continue
            resolved_conflicts += 1
            winning_label = most_common[0][0]
            rep_idx = next(i for i in member_indices if int(labels[i]) == winning_label)
        else:
            rep_idx = member_indices[0]
        canonical_indices.append(rep_idx)

    canonical_indices.sort()

    # Step 2: Check near-duplicates (char 4-gram Jaccard >= 0.90) among canonical headlines
    can_texts = [headlines[i] for i in canonical_indices]
    near_pairs = _near_duplicate_pairs(can_texts, can_texts, threshold=0.90)

    parent = list(range(len(canonical_indices)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, j, _ in near_pairs:
        if i != j:
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for pos, orig_idx in enumerate(canonical_indices):
        root = find(pos)
        clusters.setdefault(root, []).append(orig_idx)

    final_indices: list[int] = []
    near_dups_removed = 0
    for root, cluster_orig_indices in clusters.items():
        if len(cluster_orig_indices) == 1:
            final_indices.append(cluster_orig_indices[0])
        else:
            near_dups_removed += len(cluster_orig_indices) - 1
            c_labels = [int(labels[i]) for i in cluster_orig_indices]
            counts = Counter(c_labels)
            if len(counts) > 1:
                most_common = counts.most_common(2)
                if most_common[0][1] == most_common[1][1]:
                    dropped_ties += len(cluster_orig_indices)
                    continue
                winning_label = most_common[0][0]
                rep_idx = next(i for i in cluster_orig_indices if int(labels[i]) == winning_label)
            else:
                rep_idx = cluster_orig_indices[0]
            final_indices.append(rep_idx)

    final_indices.sort()
    stats = {
        "raw_cleaned_rows": n,
        "unique_exact_norm_groups": len(norm_groups),
        "exact_duplicates_removed": n - len(canonical_indices) - dropped_ties,
        "near_duplicates_removed": near_dups_removed,
        "label_conflicts_resolved": resolved_conflicts,
        "ambiguous_ties_dropped": dropped_ties,
        "final_deduplicated_rows": len(final_indices),
    }
    logger.info("Deduplication summary: %s", stats)
    return final_indices, stats


def build_dedup_splits(labeled_df, cfg, output_path: Path, seed: int = 42) -> dict:
    """Create stratified 80/10/10 splits over deduplicated canonical indices."""
    dedup_indices, dedup_stats = deduplicate_indices(labeled_df)
    dedup_labels = [int(labeled_df["label"].iloc[i]) for i in dedup_indices]

    train_r = float(cfg.data.train_ratio)
    val_r = float(cfg.data.val_ratio)
    test_r = float(cfg.data.test_ratio)
    val_test_ratio = val_r + test_r

    train_idx, valtest_idx, _, valtest_labels = train_test_split(
        dedup_indices,
        dedup_labels,
        test_size=val_test_ratio,
        stratify=dedup_labels,
        random_state=seed,
    )

    relative_test_ratio = test_r / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=seed,
    )

    splits = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "deduplicated": True,
            "dedup_stats": dedup_stats,
            "total_labeled": len(dedup_indices),
            "train_ratio": train_r,
            "val_ratio": val_r,
            "test_ratio": test_r,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        },
        "labeled": {
            "train": sorted(train_idx),
            "val": sorted(val_idx),
            "test": sorted(test_idx),
        },
        "unlabeled": {"all": []},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    logger.info("Saved deduplicated splits to %s", output_path)
    return splits


@torch.no_grad()
def evaluate_existing_models_on_leakfree_test(test_df, leaky_test_positions: list[int]) -> list[dict]:
    """Evaluate existing checkpoints on original test set (N=1,506) vs leak-free subset (N=1,351)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leaky_set = set(leaky_test_positions)
    clean_mask = np.array([i not in leaky_set for i in range(len(test_df))], dtype=bool)
    leaky_mask = ~clean_mask

    headlines = test_df["headline"].tolist()
    y_true = test_df["label"].to_numpy()

    candidates = [
        ("Teacher (BanglaBERT)", Path("experiments/teacher")),
        ("Student: Teacher-Init + KD", Path("experiments/student_teacher_init_distilled")),
        ("Student: BanglaBERT-Small 4L + KD", Path("experiments/student_small_pretrained_distilled")),
    ]

    rows = []
    for name, ckpt_dir in candidates:
        if not (ckpt_dir / "config.json").exists():
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(device)
        model.eval()

        all_preds = []
        batch_size = 64
        for start in range(0, len(headlines), batch_size):
            batch_texts = headlines[start : start + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)

        y_pred = np.concatenate(all_preds, axis=0)
        full_m = compute_classification_metrics(y_pred=y_pred, y_true=y_true)
        clean_m = compute_classification_metrics(y_pred=y_pred[clean_mask], y_true=y_true[clean_mask])
        leaky_m = compute_classification_metrics(y_pred=y_pred[leaky_mask], y_true=y_true[leaky_mask])

        row = {
            "model": name,
            "full_n": int(len(y_true)),
            "full_acc": full_m["accuracy"],
            "full_f1": full_m["f1_macro"],
            "clean_n": int(clean_mask.sum()),
            "clean_acc": clean_m["accuracy"],
            "clean_f1": clean_m["f1_macro"],
            "leaked_n": int(leaky_mask.sum()),
            "leaked_acc": leaky_m["accuracy"],
            "leaked_f1": leaky_m["f1_macro"],
            "f1_inflation_from_leakage": round(full_m["f1_macro"] - clean_m["f1_macro"], 4),
        }
        rows.append(row)
        logger.info(
            "%-34s | Full F1: %.4f (Acc %.4f) | Leak-Free F1: %.4f (Acc %.4f) | Leaked-155 Acc: %.4f | Inflation: %+.4f",
            name,
            row["full_f1"],
            row["full_acc"],
            row["clean_f1"],
            row["clean_acc"],
            row["leaked_acc"],
            row["f1_inflation_from_leakage"],
        )

    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--output-splits", default="data/processed/splits_dedup.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(args.seed)

    labeled_df, _ = load_raw(cfg, hf_dataset_id=cfg.data.hf_dataset_id)
    labeled_df = clean_dataframe(
        labeled_df,
        headline_col=cfg.data.text_col,
        min_char_length=int(cfg.data.min_char_length),
    )
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    # 1. Measure leakage impact on existing test split
    orig_splits = load_splits(cfg)
    orig_train_df, orig_val_df, orig_test_df = apply_splits(labeled_df, orig_splits)
    _, leaky = audit_splits(orig_train_df, orig_val_df, orig_test_df)
    leakage_eval = evaluate_existing_models_on_leakfree_test(orig_test_df, leaky["test"])

    # 2. Build and verify deduplicated splits
    out_splits_path = Path(args.output_splits)
    dedup_splits = build_dedup_splits(labeled_df, cfg, out_splits_path, seed=args.seed)
    d_train, d_val, d_test = apply_splits(labeled_df, dedup_splits)
    dedup_audit, _ = audit_splits(d_train, d_val, d_test)

    summary = {
        "leakage_impact_on_existing_models": leakage_eval,
        "dedup_splits_metadata": dedup_splits["metadata"],
        "dedup_splits_audit": dedup_audit,
    }
    out_report = Path("experiments/audit/dedup_impact_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
