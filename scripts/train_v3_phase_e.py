"""scripts/train_v3_phase_e.py — Phase E: Neural Baselines, Teachers & 5-Fold OOF Soft Labels.

Strictly follows v3 Global Rules:
  - Uses data/processed/v3/splits_v3.json and data/processed/v3/labeled_clean_v3.csv.
  - NEVER reads test labels (validation-only evaluation; checkpoints saved for Phase J).
  - Logs to experiments/v3/<run_name>/seed<k>/{config.yaml, metrics.json, val_logits.npy, val_labels.npy, git_sha.txt}.
  - Supports: bilstm, mdistilbert, mminilm, small_12l, teacher_base, teacher_oof, or all.

Usage:
  .venv\\Scripts\\python.exe -m scripts.train_v3_phase_e --models teacher_oof teacher_base small_12l
  .venv\\Scripts\\python.exe -m scripts.train_v3_phase_e --models bilstm mdistilbert mminilm
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.evaluation.metrics import compute_metrics_from_eval_pred
from src.training.dataset import BanglaHeadlineDataset
from src.training.train_teacher import WeightedTrainer, calculate_balanced_class_weights
from src.utils.seeding import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [13, 42, 87, 1234, 2026]
SPLITS_PATH = Path("data/processed/v3/splits_v3.json")
CLEAN_CSV_PATH = Path("data/processed/v3/labeled_clean_v3.csv")
CLEAN_FULL_CSV_PATH = Path("data/processed/v3/labeled_clean_v3_full.csv")


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return round(float(ece), 4)


def compute_full_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Compute consistent classification metrics + confusion matrix + ECE."""
    preds = np.argmax(logits, axis=1)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
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

    # Softmax for ECE
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)

    return {
        "split": "val",
        "accuracy": round(acc, 4),
        "precision_macro": round((p0 + p1) / 2.0, 4),
        "recall_macro": round((r0 + r1) / 2.0, 4),
        "f1_macro": round((f1_0 + f1_1) / 2.0, 4),
        "val_f1_macro": round((f1_0 + f1_1) / 2.0, 4),
        "f1_not_clickbait": round(f1_0, 4),
        "precision_not_clickbait": round(p0, 4),
        "recall_not_clickbait": round(r0, 4),
        "f1_clickbait": round(f1_1, 4),
        "precision_clickbait": round(p1, 4),
        "recall_clickbait": round(r1, 4),
        "ece": compute_ece(probs, labels),
        "confusion_matrix": cm,
        "true_class_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
    }


class WeightedCETrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_dataset_paths(dataset: str = "banglabait") -> tuple[Path, Path, Path]:
    """Return (data_dir, exp_dir, splits_path) for a given dataset."""
    if dataset == "baitbuster":
        d = Path("data/processed/v3_baitbuster")
        e = Path("experiments/v3_baitbuster")
    else:
        d = Path("data/processed/v3")
        e = Path("experiments/v3")
    return d, e, d / "splits_v3.json"


def format_headlines(df: pd.DataFrame, text_mode: str) -> tuple[list[str], list[str] | None]:
    """Extract headlines and optional bodies for title, title_body, or cat_title_body."""
    titles = df["headline"].astype(str).tolist()
    if text_mode == "cat_title_body" and "categories" in df.columns:
        cats = df["categories"].fillna("General").astype(str).tolist()
        titles = [f"[{c}] {t}" for c, t in zip(cats, titles)]
    bodies = df["content"].fillna("").astype(str).tolist() if text_mode in ("title_body", "cat_title_body") and "content" in df.columns else None
    return titles, bodies


def load_v3_train_val(text_mode: str = "title", dataset: str = "banglabait") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load strictly train and val DataFrames from v3 splits (never touches test)."""
    data_dir, _, splits_path = get_dataset_paths(dataset)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if text_mode in ("title_body", "cat_title_body") and (data_dir / "labeled_clean_v3_full.csv").exists():
        csv_path = data_dir / "labeled_clean_v3_full.csv"
    else:
        csv_path = data_dir / "labeled_clean_v3.csv"
    clean_df = pd.read_csv(csv_path)
    if "content" in clean_df.columns:
        clean_df["content"] = clean_df["content"].fillna("").astype(str)
    train_df = clean_df.loc[splits["labeled"]["train"]].reset_index(drop=True)
    val_df = clean_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    return train_df, val_df, splits


def train_hf_model(
    run_name: str,
    model_id: str,
    seeds: list[int],
    lr: float = 3e-5,
    epochs: int = 8,
    batch_size: int = 32,
    max_length: int = 64,
    text_mode: str = "title",
    dataset: str = "banglabait",
    unweighted: bool = False,
    label_smoothing: float = 0.0,
    rdrop_alpha: float = 0.0,
    llrd_factor: float = 1.0,
    skip_existing: bool = True,
) -> None:
    """Train a HuggingFace sequence classifier across seeds on v3 train/val."""
    _, exp_dir, _ = get_dataset_paths(dataset)
    train_df, val_df, _ = load_v3_train_val(text_mode=text_mode, dataset=dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    class_weights = None if unweighted else calculate_balanced_class_weights(train_df["label"].tolist())

    tr_titles, tr_bodies = format_headlines(train_df, text_mode)
    va_titles, va_bodies = format_headlines(val_df, text_mode)
    train_ds = BanglaHeadlineDataset(tr_titles, train_df["label"].tolist(), tokenizer, max_seq_len=max_length, bodies=tr_bodies)
    val_ds = BanglaHeadlineDataset(va_titles, val_df["label"].tolist(), tokenizer, max_seq_len=max_length, bodies=va_bodies)
    val_labels = val_df["label"].to_numpy()
    git_sha = get_git_sha()

    for seed in seeds:
        run_dir = exp_dir / run_name / f"seed{seed}"
        if skip_existing and (run_dir / "metrics.json").exists():
            logger.info("[SKIP] %s/seed%d already exists.", run_name, seed)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        logger.info("=== Training %s (%s | %s) | Seed %d ===", run_name, model_id, dataset, seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, id2label={0: "not-clickbait", 1: "clickbait"}, label2id={"not-clickbait": 0, "clickbait": 1}
        )
        n_params = sum(p.numel() for p in model.parameters())

        ckpt_dir = run_dir / "hf_checkpoints"
        total_steps = max(1, (len(train_ds) // batch_size + 1) * epochs)
        warmup_steps = int(total_steps * 0.1)
        args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            seed=seed,
            report_to="none",
            logging_steps=100,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            llrd_factor=llrd_factor,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_from_eval_pred,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()

        # Predict on validation set ONLY
        val_pred_out = trainer.predict(val_ds)
        val_logits = val_pred_out.predictions
        metrics = compute_full_metrics(val_logits, val_labels)
        metrics["params"] = n_params

        # Save artifacts
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
        np.save(run_dir / "val_logits.npy", val_logits)
        np.save(run_dir / "val_labels.npy", val_labels)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "git_sha.txt").write_text(git_sha, encoding="utf-8")
        (run_dir / "config.yaml").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "model_id": model_id,
                    "text_mode": text_mode,
                    "unweighted": unweighted,
                    "label_smoothing": label_smoothing,
                    "rdrop_alpha": rdrop_alpha,
                    "llrd_factor": llrd_factor,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "seed": seed,
                    "params": n_params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        import shutil
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        logger.info("Finished %s/seed%d | Val Macro-F1: %.4f | Val Acc: %.4f", run_name, seed, metrics["f1_macro"], metrics["accuracy"])


# ── BiLSTM Baseline ──────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        out, (hn, _) = self.lstm(emb)
        cat = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(self.dropout(cat))


def train_bilstm(seeds: list[int], epochs: int = 12, batch_size: int = 64, lr: float = 1e-3, max_length: int = 64, skip_existing: bool = True) -> None:
    """Train a lightweight PyTorch BiLSTM baseline using BanglaBERT WordPiece vocab."""
    train_df, val_df, _ = load_v3_train_val()
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = calculate_balanced_class_weights(train_df["label"].tolist()).to(device)

    train_enc = tokenizer(train_df["headline"].tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_df["headline"].tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    y_train = torch.tensor(train_df["label"].tolist(), dtype=torch.long)
    y_val = val_df["label"].to_numpy()
    git_sha = get_git_sha()

    class TensorDS(Dataset):
        def __init__(self, ids, labels):
            self.ids, self.labels = ids, labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return self.ids[i], self.labels[i]

    train_loader = DataLoader(TensorDS(train_enc["input_ids"], y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDS(val_enc["input_ids"], torch.tensor(y_val, dtype=torch.long)), batch_size=128, shuffle=False)

    for seed in seeds:
        run_dir = Path(f"experiments/v3/baseline_bilstm/seed{seed}")
        if skip_existing and (run_dir / "metrics.json").exists():
            logger.info("[SKIP] baseline_bilstm/seed%d already exists.", seed)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        model = BiLSTMClassifier(vocab_size=tokenizer.vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = -1.0
        best_logits = None
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            logits_list = []
            with torch.no_grad():
                for x_b, _ in val_loader:
                    logits_list.append(model(x_b.to(device)).cpu().numpy())
            val_logits = np.vstack(logits_list)
            m = compute_full_metrics(val_logits, y_val)
            if m["f1_macro"] > best_f1:
                best_f1 = m["f1_macro"]
                best_logits = val_logits
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        metrics = compute_full_metrics(best_logits, y_val)
        metrics["params"] = sum(p.numel() for p in model.parameters())
        torch.save(best_state, run_dir / "bilstm_best.pt")
        np.save(run_dir / "val_logits.npy", best_logits)
        np.save(run_dir / "val_labels.npy", y_val)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "git_sha.txt").write_text(git_sha, encoding="utf-8")
        (run_dir / "config.yaml").write_text(json.dumps({"model": "BiLSTM-2L", "seed": seed, "params": metrics["params"]}, indent=2), encoding="utf-8")
        logger.info("Finished baseline_bilstm/seed%d | Val Macro-F1: %.4f | Val Acc: %.4f", seed, metrics["f1_macro"], metrics["accuracy"])


# ── 5-Fold OOF Teacher Soft Labels on v3 Train Split ─────────────────────────

def generate_v3_oof_teacher(
    n_folds: int = 5,
    lr: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 32,
    max_length: int = 64,
    text_mode: str = "title",
    dataset: str = "banglabait",
    unweighted: bool = False,
    label_smoothing: float = 0.0,
    rdrop_alpha: float = 0.0,
    llrd_factor: float = 1.0,
) -> None:
    """Generate leak-free 5-fold StratifiedGroupKFold Out-of-Fold teacher logits on v3 train split."""
    data_dir, exp_dir, splits_path = get_dataset_paths(dataset)
    suffix = "_ctb" if text_mode == "cat_title_body" else ("_tb" if text_mode == "title_body" else "")
    if unweighted:
        suffix += "_uw"
    out_pt = data_dir / f"teacher_oof_logits{suffix}.pt"
    if out_pt.exists():
        logger.info("[SKIP] %s already exists.", out_pt)
        return

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    csv_path = data_dir / "labeled_clean_v3_full.csv" if (text_mode in ("title_body", "cat_title_body") and (data_dir / "labeled_clean_v3_full.csv").exists()) else data_dir / "labeled_clean_v3.csv"
    clean_df = pd.read_csv(csv_path)
    if "content" in clean_df.columns:
        clean_df["content"] = clean_df["content"].fillna("").astype(str)
    train_indices = splits["labeled"]["train"]
    train_df = clean_df.loc[train_indices].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    oof_logits = np.zeros((len(train_df), 2), dtype=np.float32)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr_pos, va_pos) in enumerate(sgkf.split(train_df, train_df["label"], train_df["group_id"])):
        logger.info("=== OOF Teacher (%s | %s) Fold %d / %d ===", dataset, text_mode, fold + 1, n_folds)
        seed_everything(42 + fold)
        f_tr = train_df.iloc[tr_pos].reset_index(drop=True)
        f_va = train_df.iloc[va_pos].reset_index(drop=True)

        tr_titles, tr_bodies = format_headlines(f_tr, text_mode)
        va_titles, va_bodies = format_headlines(f_va, text_mode)
        tr_ds = BanglaHeadlineDataset(tr_titles, f_tr["label"].tolist(), tokenizer, max_seq_len=max_length, bodies=tr_bodies)
        va_ds = BanglaHeadlineDataset(va_titles, f_va["label"].tolist(), tokenizer, max_seq_len=max_length, bodies=va_bodies)
        cw = None if unweighted else calculate_balanced_class_weights(f_tr["label"].tolist())

        model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)
        ckpt_dir = exp_dir / f"teacher_oof{suffix}" / f"fold_{fold}"
        total_steps = max(1, (len(tr_ds) // batch_size + 1) * epochs)
        warmup_steps = int(total_steps * 0.1)
        args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            seed=42 + fold,
            report_to="none",
        )
        trainer = WeightedTrainer(
            class_weights=cw,
            label_smoothing=label_smoothing,
            rdrop_alpha=rdrop_alpha,
            llrd_factor=llrd_factor,
            model=model,
            args=args,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            compute_metrics=compute_metrics_from_eval_pred,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        fold_preds = trainer.predict(va_ds).predictions
        oof_logits[va_pos] = fold_preds

        import shutil
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    torch.save(
        {
            "logits": torch.tensor(oof_logits, dtype=torch.float32),
            "headline_indices": train_indices,
        },
        out_pt,
    )
    oof_metrics = compute_full_metrics(oof_logits, train_df["label"].to_numpy())
    oof_metrics["split"] = "train_oof"
    oof_dir = exp_dir / f"teacher_oof{suffix}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    (oof_dir / "metrics.json").write_text(json.dumps(oof_metrics, indent=2), encoding="utf-8")
    logger.info("Saved 5-fold OOF teacher logits to %s | OOF Macro-F1: %.4f | OOF Acc: %.4f", out_pt, oof_metrics["f1_macro"], oof_metrics["accuracy"])


def main():
    parser = argparse.ArgumentParser(description="Phase E: Train v3 baselines, teachers, and OOF soft labels.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "bilstm", "mdistilbert", "mminilm", "small_12l", "teacher_base", "teacher_oof"],
    )
    parser.add_argument("--dataset", choices=["banglabait", "baitbuster"], default="banglabait")
    parser.add_argument("--text-mode", choices=["title", "title_body", "cat_title_body"], default="title")
    parser.add_argument("--unweighted", action="store_true", help="Use unweighted CE instead of inverse-frequency weights.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--llrd", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    selected = args.models
    if "all" in selected:
        selected = ["bilstm", "teacher_oof", "teacher_base", "small_12l", "mdistilbert", "mminilm"]

    ds = args.dataset
    tm = "title" if ds == "baitbuster" else args.text_mode
    suffix = "_ctb" if tm == "cat_title_body" else ("_tb" if tm == "title_body" else "")
    if args.unweighted:
        suffix += "_uw"
    max_len = 256 if tm in ("title_body", "cat_title_body") else 64
    base_bs = 16 if max_len == 256 else 32
    small_bs = 32
    skip = not args.no_skip

    common_kw = dict(
        text_mode=tm,
        dataset=ds,
        unweighted=args.unweighted,
        label_smoothing=args.label_smoothing,
        rdrop_alpha=args.rdrop_alpha,
        llrd_factor=args.llrd,
        skip_existing=skip,
    )

    for m in selected:
        if m == "bilstm":
            train_bilstm(seeds=args.seeds, skip_existing=skip)
        elif m == "teacher_oof":
            generate_v3_oof_teacher(
                batch_size=base_bs,
                max_length=max_len,
                text_mode=tm,
                dataset=ds,
                unweighted=args.unweighted,
                label_smoothing=args.label_smoothing,
                rdrop_alpha=args.rdrop_alpha,
                llrd_factor=args.llrd,
            )
        elif m == "teacher_base":
            train_hf_model(f"teacher_base{suffix}", "csebuetnlp/banglabert", seeds=args.seeds, lr=2e-5, epochs=6, batch_size=base_bs, max_length=max_len, **common_kw)
        elif m == "small_12l":
            train_hf_model(f"baseline_small_12l_ce{suffix}", "csebuetnlp/banglabert_small", seeds=args.seeds, lr=5e-5, epochs=8, batch_size=small_bs, max_length=max_len, **common_kw)
        elif m == "mdistilbert":
            train_hf_model(f"baseline_mdistilbert{suffix}", "distilbert-base-multilingual-cased", seeds=args.seeds, lr=3e-5, epochs=6, batch_size=base_bs, max_length=max_len, **common_kw)
        elif m == "mminilm":
            train_hf_model(f"baseline_mminilm{suffix}", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", seeds=args.seeds, lr=3e-5, epochs=6, batch_size=base_bs, max_length=max_len, **common_kw)


if __name__ == "__main__":
    main()
