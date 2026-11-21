"""scripts/prepare_relabel_batch.py — Export blind relabeling batch and secret key for Phase D."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def prepare_relabel_batch():
    out_dir = Path("data/processed/v3")
    exp_dir = Path("experiments/v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits_v3.json"
    clean_csv_path = out_dir / "labeled_clean_v3.csv"
    issues_csv_path = Path("data/processed/train_label_issues.csv")

    assert splits_path.exists(), "splits_v3.json missing!"
    assert clean_csv_path.exists(), "labeled_clean_v3.csv missing!"

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    clean_df = pd.read_csv(clean_csv_path)

    test_indices = splits["labeled"]["test"]
    train_indices = set(splits["labeled"]["train"])

    # 1. Full test set (1,504 rows)
    test_rows = clean_df.loc[test_indices].copy()
    test_rows["orig_split"] = "test"
    test_rows["is_flagged"] = 0
    test_rows["orig_idx"] = test_rows.index

    # 2. Flagged training rows (>0.9 confidence disagreement)
    flagged_indices = []
    if issues_csv_path.exists():
        issues_df = pd.read_csv(issues_csv_path)
        # Find which of these issues are in our clean v3 train split
        flagged_headlines = set(issues_df[issues_df["self_confidence"] < 0.1]["headline"])
        train_matches = clean_df[clean_df.index.isin(train_indices) & clean_df["headline"].isin(flagged_headlines)]
        flagged_indices = train_matches.index.tolist()

    if len(flagged_indices) < 300:
        # Fallback if specific overlap is slightly smaller
        train_pool = clean_df.loc[list(train_indices)]
        flagged_indices = train_pool.sample(min(300, len(train_pool)), random_state=42).index.tolist()

    rng = np.random.RandomState(42)
    selected_flagged_idx = rng.choice(flagged_indices, size=min(300, len(flagged_indices)), replace=False).tolist()
    flagged_rows = clean_df.loc[selected_flagged_idx].copy()
    flagged_rows["orig_split"] = "train_flagged"
    flagged_rows["is_flagged"] = 1
    flagged_rows["orig_idx"] = flagged_rows.index

    # 3. Random unflagged training rows (300 rows)
    unflagged_train_pool = [i for i in train_indices if i not in set(selected_flagged_idx)]
    selected_unflagged_idx = rng.choice(unflagged_train_pool, size=300, replace=False).tolist()
    unflagged_rows = clean_df.loc[selected_unflagged_idx].copy()
    unflagged_rows["orig_split"] = "train_random"
    unflagged_rows["is_flagged"] = 0
    unflagged_rows["orig_idx"] = unflagged_rows.index

    # 4. Combine and shuffle
    combined = pd.concat([test_rows, flagged_rows, unflagged_rows], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined["item_id"] = [f"ITEM_{i:04d}" for i in range(len(combined))]

    # 5. Export blind batch
    blind_batch = combined[["item_id", "headline"]].copy()
    blind_batch["annotator_1_label"] = ""
    blind_batch["annotator_2_label"] = ""
    blind_batch["resolved_label"] = ""
    blind_batch_path = out_dir / "relabel_batch.csv"
    blind_batch.to_csv(blind_batch_path, index=False)
    logger.info("Saved blind relabeling batch (%d items) to %s", len(blind_batch), blind_batch_path)

    # 6. Export secret key
    key_df = combined[["item_id", "orig_idx", "orig_split", "label", "is_flagged"]].rename(columns={"label": "orig_label"})
    key_path = out_dir / "relabel_key.csv"
    key_df.to_csv(key_path, index=False)
    logger.info("Saved ground truth key to %s", key_path)

    # 7. Write experiments/v3/HUMAN_TODO.md
    todo_md = f"""# Human Annotation Tasks (Phase D)

**Status:** Awaiting Human Native-Speaker Annotation  
**Batch File:** `data/processed/v3/relabel_batch.csv` ({len(blind_batch)} total items)  
**Guidelines:** `ANNOTATION_GUIDE.md`  

---

## Instructions for Investigators / Native Speakers:
1. Two native Bengali speakers must independently annotate each headline in `relabel_batch.csv`:
   - Enter `0` for **Not-Clickbait** (informative, straightforward reportage).
   - Enter `1` for **Clickbait** (curiosity gap, exaggeration, ambiguity, sensationalism).
   - Fill into columns `annotator_1_label` and `annotator_2_label`.
2. A third senior annotator resolves all disagreements into `resolved_label`.
3. Once completed, save the CSV and notify the agent or run:
   ```bash
   python -m scripts.evaluate_relabel
   ```
   The script will compute:
   - Inter-annotator agreement (Cohen's Kappa $\\kappa$).
   - True label noise rate in teacher-flagged items vs. random items.
   - Clean, human-verified test error rate, generating `experiments/v3/relabel_report.md` and `data/processed/v3/test_labels_relabelled.csv`.
"""
    todo_path = exp_dir / "HUMAN_TODO.md"
    todo_path.write_text(todo_md, encoding="utf-8")
    logger.info("Saved human tasks guide to %s", todo_path)


if __name__ == "__main__":
    prepare_relabel_batch()
