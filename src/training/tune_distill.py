"""
src/training/tune_distill.py — Hyperparameter Grid Search & Optuna for Distillation (T, alpha).

Objective:
  Find optimal (Temperature T, Alpha alpha) for TinyBanglaBERT distillation based
  STRICTLY on validation set Macro-F1. The test set is NEVER touched during search.

Grid Search Space:
  - Temperature T in {1.0, 2.0, 4.0, 8.0, 16.0}
  - Alpha alpha in {0.1, 0.3, 0.5, 0.7, 0.9}

Outputs:
  - `experiments/tuning/kd_grid_search_report.md` (Markdown table for thesis appendix)
  - `experiments/tuning/kd_grid_results.json` (Structured results)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
)

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.evaluation.metrics import compute_metrics_from_eval_pred
from src.models.distillation import (
    DistillationDataset,
    KnowledgeDistillationLoss,
    load_teacher_soft_labels,
)
from src.models.student import (
    create_student_config,
    init_from_teacher,
    init_random,
)
from src.models.teacher import get_teacher_tokenizer
from src.training.dataset import BanglaHeadlineDataset
from src.training.train_distill import DistillationTrainer
from src.training.train_teacher import calculate_balanced_class_weights
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

DEFAULT_T_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def evaluate_kd_hyperparameter_pair(
    cfg: DictConfig,
    temperature: float,
    alpha: float,
    train_ds: DistillationDataset,
    val_ds: BanglaHeadlineDataset,
    class_weights: torch.Tensor,
    init_type: str = "teacher_init",
    epochs: int = 3,
    seed: int = 42,
    output_dir: str | Path = "experiments/tuning/checkpoints",
) -> dict[str, Any]:
    """Train a student on a specific (T, alpha) pair and evaluate on the VALIDATION split.

    Args:
        cfg: Base configuration.
        temperature: Temperature T.
        alpha: Loss balance alpha.
        train_ds: Training distillation dataset with soft labels.
        val_ds: Validation dataset.
        class_weights: Class weights tensor.
        init_type: 'teacher_init' or 'random'.
        epochs: Number of tuning epochs (default: 3).
        seed: Random seed.
        output_dir: Checkpoint directory.

    Returns:
        Dictionary of validation metrics for this (T, alpha) combination.
    """
    seed_everything(seed)
    tokenizer_name = str(cfg.model.teacher_model_name)
    tokenizer = get_teacher_tokenizer(tokenizer_name)

    student_cfg = create_student_config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        num_labels=int(cfg.data.num_labels),
        dropout=float(cfg.model.dropout),
    )

    if init_type in ("teacher_init", "teacher"):
        teacher_model = AutoModelForSequenceClassification.from_pretrained("experiments/teacher")
        model = init_from_teacher(teacher_model, student_cfg, strategy="projection")
    else:
        model = init_random(student_cfg)

    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        class_weights=class_weights,
    )

    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    batch_size = int(cfg.training.batch_size)
    total_steps = int((len(train_ds) // batch_size + 1) * epochs)
    warmup_steps = int(total_steps * float(cfg.training.warmup_ratio))

    ckpt_dir = Path(output_dir) / f"run_T{temperature}_a{alpha}"
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=float(cfg.training.lr),
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
        seed=seed,
        report_to="none",
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_from_eval_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        kd_loss_fn=kd_loss_fn,
    )

    trainer.train()

    # Evaluate on VALIDATION set (Test set is NEVER touched here)
    eval_metrics = trainer.evaluate()
    val_macro_f1 = float(eval_metrics.get("eval_f1_macro", 0.0))
    val_acc = float(eval_metrics.get("eval_accuracy", 0.0))
    val_loss = float(eval_metrics.get("eval_loss", 0.0))
    val_clickbait_f1 = float(eval_metrics.get("eval_f1_clickbait", 0.0))

    result = {
        "temperature": temperature,
        "alpha": alpha,
        "val_f1_macro": round(val_macro_f1, 4),
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_f1_clickbait": round(val_clickbait_f1, 4),
    }
    logger.info(
        "T=%.1f, alpha=%.2f -> Val Macro-F1: %.4f | Val Acc: %.4f | Val Loss: %.4f",
        temperature,
        alpha,
        val_macro_f1,
        val_acc,
        val_loss,
    )
    return result


def run_grid_search_kd(
    cfg: DictConfig,
    temperatures: Sequence[float] = DEFAULT_T_GRID,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    init_type: str = "teacher_init",
    epochs_per_trial: int = 3,
    seed: int = 42,
    output_dir: str | Path = "experiments/tuning",
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Execute complete grid search over (T, alpha) space on the Validation split.

    Args:
        cfg: Base configuration.
        temperatures: List of temperature values.
        alphas: List of alpha values.
        init_type: Initialization strategy for the tuning runs.
        epochs_per_trial: Training epochs per combination.
        seed: Random seed.
        output_dir: Target output directory for reports and results.

    Returns:
        Tuple of (best_params_dict, all_results_list, markdown_table_string).
    """
    seed_everything(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    hf_id = getattr(cfg.data, "hf_dataset_id", "Motahar/banglabait")
    labeled_df, _ = load_raw(cfg, hf_dataset_id=hf_id)
    labeled_df = clean_dataframe(labeled_df, headline_col=cfg.data.text_col)
    splits = load_splits(cfg)
    train_df, val_df, _ = apply_splits(labeled_df, splits)

    class_weights = calculate_balanced_class_weights(train_df[cfg.data.label_col].tolist())
    soft_data = load_teacher_soft_labels("experiments/teacher/train_soft_labels.pt")
    tokenizer = get_teacher_tokenizer(str(cfg.model.teacher_model_name))
    max_seq_len = int(cfg.model.max_seq_len)

    train_ds = DistillationDataset(
        headlines=train_df[cfg.data.text_col].tolist(),
        labels=train_df[cfg.data.label_col].tolist(),
        teacher_logits=soft_data["logits"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    val_ds = BanglaHeadlineDataset(
        headlines=val_df[cfg.data.text_col].tolist(),
        labels=val_df[cfg.data.label_col].tolist(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    total_combinations = len(temperatures) * len(alphas)
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Hyperparameter Grid Search")
    logger.info(
        "Search Space: %d combinations (T in %s, alpha in %s)",
        total_combinations,
        temperatures,
        alphas,
    )
    logger.info("Objective: Maximize Validation Macro-F1 (Test set NEVER touched)")
    logger.info("=" * 60)

    results: list[dict[str, Any]] = []
    trial_num = 1

    for T in temperatures:
        for a in alphas:
            logger.info("--- Trial %d/%d: T=%.1f, alpha=%.2f ---", trial_num, total_combinations, T, a)
            res = evaluate_kd_hyperparameter_pair(
                cfg=cfg,
                temperature=T,
                alpha=a,
                train_ds=train_ds,
                val_ds=val_ds,
                class_weights=class_weights,
                init_type=init_type,
                epochs=epochs_per_trial,
                seed=seed,
                output_dir=out_dir / "checkpoints",
            )
            results.append(res)
            trial_num += 1

    # Sort results by validation Macro-F1 (descending)
    results_sorted = sorted(results, key=lambda x: x["val_f1_macro"], reverse=True)
    best_combo = results_sorted[0]

    # Generate Markdown Table Report for Thesis Appendix
    report_lines = [
        "# Knowledge Distillation (T, alpha) Hyperparameter Search Report",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Optimization Metric: Validation Set Macro-F1 (Held-out Test set untouched)*",
        f"*Initialization Strategy: {init_type}*",
        "",
        "## Best Hyperparameter Setting",
        f"- **Optimal Temperature ($T^*$):** `{best_combo['temperature']}`",
        f"- **Optimal Alpha ($\\alpha^*$):** `{best_combo['alpha']}`",
        f"- **Best Validation Macro-F1:** **`{best_combo['val_f1_macro']:.4f}`**",
        f"- **Validation Accuracy:** `{best_combo['val_accuracy']:.4f}`",
        "",
        "## Full Grid Search Results (Validation Split)",
        "",
        "| Rank | Temperature ($T$) | Alpha ($\\alpha$) | Val Macro-F1 | Val Accuracy | Val Clickbait F1 | Val Loss |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, r in enumerate(results_sorted, 1):
        bold = "**" if rank == 1 else ""
        report_lines.append(
            f"| {rank} | {r['temperature']} | {r['alpha']} | {bold}{r['val_f1_macro']:.4f}{bold} | {r['val_accuracy']:.4f} | {r['val_f1_clickbait']:.4f} | {r['val_loss']:.4f} |"
        )

    # 2D Grid Pivot Table
    pivot_df = pd.DataFrame(results).pivot(index="temperature", columns="alpha", values="val_f1_macro")
    report_lines += [
        "",
        "## Validation Macro-F1 Response Surface Matrix ($T$ vs $\\alpha$)",
        "",
        pivot_df.to_markdown(),
        "",
    ]

    report_md = "\n".join(report_lines)

    # Save to disk
    report_path = out_dir / "kd_grid_search_report.md"
    results_json_path = out_dir / "kd_grid_results.json"
    report_path.write_text(report_md, encoding="utf-8")
    results_json_path.write_text(json.dumps({"best_params": best_combo, "grid_results": results}, indent=2), encoding="utf-8")

    logger.info("Grid search completed! Report written to: %s", report_path)
    return best_combo, results, report_md
