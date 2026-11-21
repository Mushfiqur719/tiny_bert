"""
scripts/generate_oof_soft_labels.py — Phase B2: 5-Fold Out-of-Fold (OOF) Teacher Soft Labels.

Why Out-of-Fold Soft Labels?
  When a 110M Teacher is fine-tuned on 100% of `train_df` and then predicts logits on that same
  `train_df`, it has memorized the training headlines (near one-hot probabilities), stripping away
  the soft "dark knowledge" needed for Knowledge Distillation.
  With K-fold OOF (K=5), each 20% fold of `train_df` receives logits predicted by a Teacher trained
  strictly on the other 80% of `train_df`.

Outputs:
  - Backs up existing `experiments/teacher/train_soft_labels.pt` to `train_soft_labels_infold.pt`
  - Saves out-of-fold soft labels to `experiments/teacher/train_oof_soft_labels.pt`
  - Replaces `experiments/teacher/train_soft_labels.pt` with the OOF soft labels (or a 70/30 OOF+full blend)

Usage:
    .venv\\Scripts\\python.exe -m scripts.generate_oof_soft_labels --config configs/teacher_baseline.yaml --n-splits 5 --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import EarlyStoppingCallback, TrainingArguments

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.evaluation.metrics import compute_classification_metrics, compute_metrics_from_eval_pred
from src.models.teacher import get_teacher_tokenizer, load_teacher_model
from src.training.dataset import BanglaHeadlineDataset
from src.training.train_teacher import WeightedTrainer, calculate_balanced_class_weights
from src.utils import load_config, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate K-fold Out-of-Fold Teacher soft labels.")
    p.add_argument("--config", default="configs/teacher_baseline.yaml")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3, help="Epochs per fold (3 epochs avoids overfitting).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--blend-infold",
        type=float,
        default=0.25,
        help="Weight of full-teacher logits blended with OOF logits (0.0 = pure OOF, 0.25 = 75%% OOF + 25%% full).",
    )
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
    splits = load_splits(cfg)
    train_df, _, _ = apply_splits(labeled_df, splits)

    headlines = train_df[cfg.data.text_col].tolist()
    labels = train_df[cfg.data.label_col].to_numpy(dtype=np.int64)
    n_train = len(headlines)

    teacher_model_name = str(cfg.model.teacher_model_name)
    tokenizer = get_teacher_tokenizer(teacher_model_name)
    max_seq_len = int(cfg.model.max_seq_len)

    oof_logits = np.zeros((n_train, int(cfg.data.num_labels)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    batch_size = int(cfg.training.batch_size)
    out_base = Path("experiments/teacher/oof_folds")
    out_base.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(headlines, labels), start=1):
        logger.info("=" * 65)
        logger.info("OOF Teacher Fold %d/%d (Train: %d, Held-Out Fold: %d)", fold_idx, args.n_splits, len(tr_idx), len(va_idx))
        logger.info("=" * 65)

        fold_tr_texts = [headlines[i] for i in tr_idx]
        fold_tr_labels = [int(labels[i]) for i in tr_idx]
        fold_va_texts = [headlines[i] for i in va_idx]
        fold_va_labels = [int(labels[i]) for i in va_idx]

        fold_tr_ds = BanglaHeadlineDataset(fold_tr_texts, fold_tr_labels, tokenizer, max_seq_len=max_seq_len)
        fold_va_ds = BanglaHeadlineDataset(fold_va_texts, fold_va_labels, tokenizer, max_seq_len=max_seq_len)

        class_weights = calculate_balanced_class_weights(fold_tr_labels)
        model = load_teacher_model(
            model_name_or_path=teacher_model_name,
            num_labels=int(cfg.data.num_labels),
            dropout=float(cfg.model.dropout),
        )

        total_steps = int((len(fold_tr_ds) // batch_size + 1) * args.epochs)
        warmup_steps = int(total_steps * float(cfg.training.warmup_ratio))
        fold_dir = out_base / f"fold_{fold_idx}"

        training_args = TrainingArguments(
            output_dir=str(fold_dir),
            num_train_epochs=float(args.epochs),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=float(cfg.training.lr),
            lr_scheduler_type=str(getattr(cfg.training, "lr_scheduler_type", "cosine")),
            weight_decay=float(cfg.training.weight_decay),
            warmup_steps=warmup_steps,
            max_grad_norm=float(cfg.training.gradient_clip),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=use_fp16,
            dataloader_num_workers=0,
            save_total_limit=1,
            logging_strategy="epoch",
            seed=args.seed + fold_idx,
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=fold_tr_ds,
            eval_dataset=fold_va_ds,
            compute_metrics=compute_metrics_from_eval_pred,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights,
            label_smoothing=float(getattr(cfg.training, "label_smoothing", 0.0)),
        )
        trainer.train()

        pred_out = trainer.predict(fold_va_ds)
        fold_logits = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) else pred_out.predictions
        oof_logits[va_idx] = fold_logits

        fold_m = compute_classification_metrics(np.argmax(fold_logits, axis=1), fold_va_labels)
        logger.info("Fold %d Held-Out OOF Macro-F1: %.4f | Acc: %.4f", fold_idx, fold_m["f1_macro"], fold_m["accuracy"])

        # Free disk space for fold checkpoints
        shutil.rmtree(fold_dir, ignore_errors=True)

    oof_m = compute_classification_metrics(np.argmax(oof_logits, axis=1), labels)
    logger.info("=" * 65)
    logger.info("FULL 5-FOLD OOF TRAIN SPLIT METRICS -> Macro-F1: %.4f | Acc: %.4f", oof_m["f1_macro"], oof_m["accuracy"])
    logger.info("=" * 65)

    canonical_path = Path("experiments/teacher/train_soft_labels.pt")
    infold_backup = Path("experiments/teacher/train_soft_labels_infold.pt")
    oof_only_path = Path("experiments/teacher/train_oof_soft_labels.pt")

    if canonical_path.exists() and not infold_backup.exists():
        shutil.copy2(canonical_path, infold_backup)
        logger.info("Backed up in-fold teacher soft labels to %s", infold_backup)

    oof_tensor = torch.tensor(oof_logits, dtype=torch.float32)
    if args.blend_infold > 0.0 and infold_backup.exists():
        infold_data = torch.load(infold_backup, map_location="cpu")
        infold_tensor = infold_data["logits"].float()
        w = float(args.blend_infold)
        final_logits = (1.0 - w) * oof_tensor + w * infold_tensor
        logger.info("Blended %.0f%% OOF logits + %.0f%% full-teacher logits.", (1.0 - w) * 100, w * 100)
    else:
        final_logits = oof_tensor

    payload = {
        "logits": final_logits,
        "oof_raw_logits": oof_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "headline_indices": splits["labeled"]["train"],
        "teacher_model_name": teacher_model_name,
        "n_splits": args.n_splits,
        "blend_infold": args.blend_infold,
        "seed": args.seed,
    }
    torch.save(payload, oof_only_path)
    torch.save(payload, canonical_path)
    logger.info("Saved OOF soft labels to %s and updated %s", oof_only_path, canonical_path)


if __name__ == "__main__":
    main()
