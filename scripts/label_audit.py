"""
scripts/label_audit.py — Stage 1: Out-of-Fold Label Noise Audit on Training Split ONLY.

Uses the 5-fold Out-of-Fold (OOF) teacher logits (`experiments/teacher/train_oof_soft_labels.pt`)
to detect mislabeled/noisy training examples in `train_df` without ever touching `val_df` or `test_df`.

Supports:
  - `cleanlab.filter.find_label_issues` (if `cleanlab` is installed), plus
  - Confident-Joint / Self-Confidence OOF filtering fallback (identifies training examples where
    the out-of-fold teacher predicts the opposite class with probability >= `prune_threshold`).

Outputs:
  1. `data/processed/train_label_issues.csv` — Ranked table of flagged noisy training rows.
  2. `experiments/audit/label_audit_report.md` & `label_audit_summary.json` — Audit statistics & top examples.
  3. `data/processed/splits_clean_train.json` — Identical to `splits.json` except flagged noisy indices
     are pruned from `labeled.train` (`val` and `test` are 100% untouched).
  4. `data/processed/splits_dedup_clean_train.json` — Same for `splits_dedup.json` if present.

Usage:
    .venv\\Scripts\\python.exe -m scripts.label_audit --prune-threshold 0.88
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


def detect_oof_label_issues(
    labels: np.ndarray,
    oof_probs: np.ndarray,
    prune_threshold: float = 0.88,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Identify noisy training labels using OOF probabilities.

    Args:
        labels: Ground-truth integer training labels [N].
        oof_probs: Out-of-fold predicted probabilities [N, 2].
        prune_threshold: Minimum OOF probability assigned to the opposite class to flag as label error.

    Returns:
        Tuple of (issue_ boolean mask [N], self_confidence [N], method_used).
    """
    n = len(labels)
    self_conf = oof_probs[np.arange(n), labels]
    pred_cls = np.argmax(oof_probs, axis=1)
    pred_conf = np.max(oof_probs, axis=1)

    method = f"oof_confident_threshold>={prune_threshold:.2f}"
    try:
        from cleanlab.filter import find_label_issues  # type: ignore

        cl_mask = find_label_issues(
            labels=labels,
            pred_probs=oof_probs,
            filter_by="prune_by_noise_rate",
        )
        # Intersect with high-confidence disagreement so we don't over-prune borderline hard examples
        high_conf_mask = (pred_cls != labels) & (pred_conf >= prune_threshold)
        issue_mask = np.asarray(cl_mask, dtype=bool) & high_conf_mask
        method = f"cleanlab_intersect_conf>={prune_threshold:.2f}"
    except Exception:
        issue_mask = (pred_cls != labels) & (pred_conf >= prune_threshold)

    return issue_mask, self_conf, method


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Audit training split for label noise using 5-fold OOF logits.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--oof-path", default="experiments/teacher/train_oof_soft_labels.pt")
    p.add_argument(
        "--prune-threshold",
        type=float,
        default=0.88,
        help="OOF opposite-class probability threshold to prune noisy training examples (default: 0.88).",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    labeled_df, _ = load_raw(cfg, hf_dataset_id=cfg.data.hf_dataset_id)
    labeled_df = clean_dataframe(
        labeled_df,
        headline_col=cfg.data.text_col,
        min_char_length=int(cfg.data.min_char_length),
    )
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    splits = load_splits(cfg)
    train_df, _, _ = apply_splits(labeled_df, splits)
    train_orig_indices = splits["labeled"]["train"]

    oof_file = Path(args.oof_path)
    if not oof_file.exists():
        raise FileNotFoundError(
            f"OOF soft labels not found at '{oof_file}'. Run scripts.generate_oof_soft_labels first."
        )

    oof_data = torch.load(oof_file, map_location="cpu")
    raw_oof_logits = oof_data.get("oof_raw_logits", oof_data["logits"]).numpy()
    oof_probs = softmax_np(raw_oof_logits)
    labels = train_df[cfg.data.label_col].to_numpy(dtype=int)
    headlines = train_df[cfg.data.text_col].tolist()

    issue_mask, self_conf, method = detect_oof_label_issues(
        labels=labels,
        oof_probs=oof_probs,
        prune_threshold=args.prune_threshold,
    )

    flagged_positions = np.where(issue_mask)[0]
    # Sort by lowest self-confidence (most likely mislabeled first)
    flagged_positions = flagged_positions[np.argsort(self_conf[flagged_positions])]

    rows = []
    flagged_orig_indices = set()
    for pos in flagged_positions:
        orig_idx = int(train_orig_indices[pos])
        flagged_orig_indices.add(orig_idx)
        rows.append(
            {
                "train_pos": int(pos),
                "orig_index": orig_idx,
                "headline": headlines[pos],
                "given_label": int(labels[pos]),
                "oof_pred_label": int(np.argmax(oof_probs[pos])),
                "oof_prob_clickbait": round(float(oof_probs[pos, 1]), 4),
                "self_confidence": round(float(self_conf[pos]), 4),
            }
        )

    issues_df = pd.DataFrame(rows)
    out_csv = Path("data/processed/train_label_issues.csv")
    issues_df.to_csv(out_csv, index=False, encoding="utf-8")

    # 1. Create splits_clean_train.json (Val and Test strictly untouched!)
    clean_train_indices = [idx for idx in train_orig_indices if idx not in flagged_orig_indices]
    clean_splits = json.loads(json.dumps(splits))
    clean_splits["labeled"]["train"] = clean_train_indices
    clean_splits["metadata"]["train_size"] = len(clean_train_indices)
    clean_splits["metadata"]["pruned_noisy_train_rows"] = len(flagged_orig_indices)
    clean_splits["metadata"]["prune_method"] = method

    clean_splits_path = Path("data/processed/splits_clean_train.json")
    clean_splits_path.write_text(json.dumps(clean_splits, indent=2), encoding="utf-8")

    # 2. Also create splits_dedup_clean_train.json if splits_dedup.json exists
    dedup_path = Path("data/processed/splits_dedup.json")
    dedup_pruned_count = 0
    if dedup_path.exists():
        dedup_splits = json.loads(dedup_path.read_text(encoding="utf-8"))
        d_train = dedup_splits["labeled"]["train"]
        d_clean_train = [idx for idx in d_train if idx not in flagged_orig_indices]
        dedup_pruned_count = len(d_train) - len(d_clean_train)
        dedup_splits["labeled"]["train"] = d_clean_train
        dedup_splits["metadata"]["train_size"] = len(d_clean_train)
        dedup_splits["metadata"]["pruned_noisy_train_rows"] = dedup_pruned_count
        Path("data/processed/splits_dedup_clean_train.json").write_text(
            json.dumps(dedup_splits, indent=2), encoding="utf-8"
        )

    # 3. Summary & Markdown report
    given_0_to_1 = int(sum(1 for r in rows if r["given_label"] == 0))
    given_1_to_0 = int(sum(1 for r in rows if r["given_label"] == 1))
    summary = {
        "method": method,
        "prune_threshold": args.prune_threshold,
        "total_train_rows": len(labels),
        "flagged_noisy_rows": len(rows),
        "flagged_pct": round(100.0 * len(rows) / len(labels), 2),
        "given_not_cb_predicted_cb": given_0_to_1,
        "given_cb_predicted_not_cb": given_1_to_0,
        "clean_train_rows_remaining": len(clean_train_indices),
        "dedup_train_pruned_rows": dedup_pruned_count,
    }

    audit_dir = Path("experiments/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "label_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Training Set Out-of-Fold Label Noise Audit Report",
        "",
        f"- **Detection Method:** `{method}`",
        f"- **Total Training Headlines Audited:** `{len(labels):,}` *(Validation and Test splits strictly untouched)*",
        f"- **Flagged High-Confidence Label Errors:** **`{len(rows):,}` (`{summary['flagged_pct']:.2f}%`)**",
        f"  - Given `0 (not-clickbait)` → OOF Teacher `1 (clickbait)`: `{given_0_to_1}`",
        f"  - Given `1 (clickbait)` → OOF Teacher `0 (not-clickbait)`: `{given_1_to_0}`",
        f"- **Clean Training Set Size (`splits_clean_train.json`):** **`{len(clean_train_indices):,}`**",
        "",
        "## Top 20 Most Confident Mislabeled Training Examples",
        "",
        "| Rank | Orig Index | Headline | Given Label | OOF Pred | OOF Clickbait Prob | Self-Confidence |",
        "|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    for rank, r in enumerate(rows[:20], 1):
        md_lines.append(
            f"| {rank} | {r['orig_index']} | {r['headline'][:55]} | `{r['given_label']}` | **`{r['oof_pred_label']}`** | `{r['oof_prob_clickbait']:.4f}` | `{r['self_confidence']:.4f}` |"
        )

    (audit_dir / "label_audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Label audit complete: %s", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
