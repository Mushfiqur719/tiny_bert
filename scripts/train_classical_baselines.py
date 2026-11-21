"""scripts/train_classical_baselines.py — Classical ML baselines (TF-IDF + Logistic Regression).

Trains across the 5 pre-registered seeds [13, 42, 87, 1234, 2026] on data/processed/v3/splits_v3.json.
Evaluates strictly on validation split (respecting the Test Set Lock).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.dedup import norm_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [13, 42, 87, 1234, 2026]


def recompute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0.0

    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0.0

    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round((p0 + p1) / 2.0, 4),
        "recall_macro": round((r0 + r1) / 2.0, 4),
        "f1_macro": round((f1_0 + f1_1) / 2.0, 4),
        "f1_not_clickbait": round(f1_0, 4),
        "precision_not_clickbait": round(p0, 4),
        "recall_not_clickbait": round(r0, 4),
        "f1_clickbait": round(f1_1, 4),
        "precision_clickbait": round(p1, 4),
        "recall_clickbait": round(r1, 4),
        "confusion_matrix": cm,
    }


def train_tfidf_baselines():
    splits_path = Path("data/processed/v3/splits_v3.json")
    clean_csv_path = Path("data/processed/v3/labeled_clean_v3.csv")
    assert splits_path.exists() and clean_csv_path.exists()

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    clean_df = pd.read_csv(clean_csv_path)

    train_idx = splits["labeled"]["train"]
    val_idx = splits["labeled"]["val"]

    train_df = clean_df.loc[train_idx].reset_index(drop=True)
    val_df = clean_df.loc[val_idx].reset_index(drop=True)

    X_train_raw = train_df["headline"].apply(norm_text).tolist()
    y_train = train_df["label"].to_numpy()

    X_val_raw = val_df["headline"].apply(norm_text).tolist()
    y_val = val_df["label"].to_numpy()

    # Pre-tune C on validation split with seed 42
    best_c = 1.0
    best_val_f1 = -1.0
    for c_candidate in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_tr = vec.fit_transform(X_train_raw)
        X_va = vec.transform(X_val_raw)
        clf = LogisticRegression(C=c_candidate, max_iter=1000, random_state=42, class_weight="balanced")
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_va)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_c = c_candidate

    logger.info("Selected optimal C=%.2f (Val Macro-F1: %.4f)", best_c, best_val_f1)

    # Train across 5 seeds
    results = []
    for seed in SEEDS:
        run_dir = Path(f"experiments/v3/baseline_tfidf/seed{seed}")
        run_dir.mkdir(parents=True, exist_ok=True)

        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_tr = vec.fit_transform(X_train_raw)
        X_va = vec.transform(X_val_raw)

        clf = LogisticRegression(C=best_c, max_iter=1000, random_state=seed, class_weight="balanced")
        clf.fit(X_tr, y_train)

        val_probs = clf.predict_proba(X_va)
        val_preds = clf.predict(X_va)

        metrics = recompute_metrics(y_val, val_preds)
        metrics["val_f1_macro"] = metrics["f1_macro"]
        metrics["split"] = "val"

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.save(run_dir / "val_logits.npy", val_probs)
        np.save(run_dir / "val_labels.npy", y_val)

        config_info = {
            "model": "TF-IDF + LogisticRegression",
            "C": best_c,
            "max_features": 10000,
            "ngram_range": [1, 2],
            "seed": seed,
        }
        (run_dir / "config.yaml").write_text(json.dumps(config_info, indent=2), encoding="utf-8")
        logger.info("Seed %d | Val Acc: %.4f | Val Macro-F1: %.4f", seed, metrics["accuracy"], metrics["f1_macro"])
        results.append(metrics["f1_macro"])

    mean_f1 = np.mean(results)
    std_f1 = np.std(results)
    logger.info("TF-IDF Baseline across 5 seeds: Mean Val Macro-F1 = %.4f +/- %.4f", mean_f1, std_f1)


if __name__ == "__main__":
    train_tfidf_baselines()
