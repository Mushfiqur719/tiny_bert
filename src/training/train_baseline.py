"""
src/training/train_baseline.py — Supervised-only fine-tuning for TinyBanglaBERT Student.

Runs either:
  1. `student_random_baseline`: Student with Random Initialization + Cross-Entropy on labels.
  2. `student_teacher_init_baseline`: Student with Teacher-Projected Init + Cross-Entropy on labels.

Protocol:
  - Same data splits and train budget (epochs, batch size, learning rate, warmup) as Teacher.
  - Early stopping on validation split Macro-F1.
  - Test set evaluated ONCE at the very end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
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
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
)
from src.models.student import (
    create_student_config,
    init_from_teacher,
    init_random,
    save_student_checkpoint,
)
from src.models.teacher import get_teacher_tokenizer
from src.training.dataset import prepare_datasets
from src.training.train_teacher import (
    MetricLoggingCallback,
    WeightedTrainer,
    calculate_balanced_class_weights,
)
from src.utils.logging_utils import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def train_student_baseline(
    cfg: DictConfig,
    init_type: str = "random",
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Train TinyBanglaBERT student with supervised cross-entropy only.

    Args:
        cfg: Configuration object.
        init_type: 'random' or 'teacher_init'.
        seed: Seed override.

    Returns:
        Dictionary of final test metrics.
    """
    effective_seed = int(seed if seed is not None else cfg.seed)
    seed_everything(effective_seed)

    run_name = str(cfg.training.get("run_name", f"student_{init_type}_baseline"))
    exp = ExperimentLogger(run_name=run_name, cfg=cfg, seed=effective_seed)

    logger.info("=" * 60)
    logger.info("Student Baseline Fine-Tuning (%s Init)", init_type.upper())
    logger.info("Run Name: %s | Seed: %d", run_name, effective_seed)
    logger.info("=" * 60)

    # 1. Load data
    hf_id = getattr(cfg.data, "hf_dataset_id", "Motahar/banglabait")
    labeled_df, _ = load_raw(cfg, hf_dataset_id=hf_id)
    labeled_df = clean_dataframe(
        labeled_df,
        headline_col=cfg.data.text_col,
        min_char_length=int(cfg.data.min_char_length),
    )
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    splits = load_splits(cfg)
    train_df, val_df, test_df = apply_splits(labeled_df, splits)

    # 2. Class weights
    class_weights = calculate_balanced_class_weights(train_df[cfg.data.label_col].tolist())

    # 3. Tokenizer & Datasets
    tokenizer_name = str(cfg.model.teacher_model_name)
    tokenizer = get_teacher_tokenizer(tokenizer_name)
    max_seq_len = int(cfg.model.max_seq_len)

    train_ds, val_ds, test_ds = prepare_datasets(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_col=cfg.data.text_col,
        label_col=cfg.data.label_col,
    )

    # 4. Student Model Initialization
    student_cfg = create_student_config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        num_labels=int(cfg.data.num_labels),
        dropout=float(cfg.model.dropout),
    )

    if init_type == "random":
        model = init_random(student_cfg)
    elif init_type in ("teacher_init", "teacher"):
        teacher_path = "experiments/teacher"
        logger.info("Loading fine-tuned Teacher model from '%s' for initialization...", teacher_path)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        model = init_from_teacher(teacher_model, student_cfg, strategy="projection")
    else:
        raise ValueError(f"Unknown init_type: '{init_type}'. Use 'random' or 'teacher_init'.")

    # 5. Training Arguments
    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    batch_size = int(cfg.training.batch_size)
    epochs = float(cfg.training.epochs)
    total_steps = int((len(train_ds) // batch_size + (1 if len(train_ds) % batch_size != 0 else 0)) * epochs)
    warmup_steps = int(total_steps * float(cfg.training.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(exp.checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        warmup_steps=warmup_steps,
        max_grad_norm=float(cfg.training.gradient_clip),
        eval_strategy=str(cfg.training.eval_strategy),
        save_strategy=str(cfg.training.save_strategy),
        load_best_model_at_end=bool(cfg.training.load_best_model_at_end),
        metric_for_best_model="f1_macro",
        greater_is_better=bool(cfg.training.greater_is_better),
        fp16=use_fp16,
        dataloader_num_workers=int(cfg.training.dataloader_num_workers),
        save_total_limit=2,
        logging_strategy="epoch",
        seed=effective_seed,
        report_to="none",
    )

    patience = int(getattr(cfg.training, "early_stopping_patience", 3))
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=patience),
        MetricLoggingCallback(exp_logger=exp),
    ]

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_from_eval_pred,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    # 6. Train
    logger.info("Training Student Baseline on %d samples...", len(train_ds))
    trainer.train()

    # 7. Save Checkpoint
    best_ckpt_dir = exp.checkpoint_path("best_val_f1")
    trainer.save_model(str(best_ckpt_dir))
    tokenizer.save_pretrained(str(best_ckpt_dir))
    save_student_checkpoint(trainer.model, tokenizer, exp.run_dir)

    # 8. Test Set Evaluation (EXACTLY ONCE)
    logger.info("Evaluating Student on held-out test split (%d samples)...", len(test_ds))
    test_prediction_output = trainer.predict(test_ds)
    test_logits = test_prediction_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_preds = np.argmax(test_logits, axis=1)
    test_labels = test_prediction_output.label_ids

    test_metrics = compute_classification_metrics(y_pred=test_preds, y_true=test_labels)
    logger.info("=" * 60)
    logger.info("STUDENT BASELINE (%s) FINAL TEST SET METRICS:", init_type.upper())
    logger.info("Accuracy:             %.4f", test_metrics["accuracy"])
    logger.info("F1 (Macro):           %.4f", test_metrics["f1_macro"])
    logger.info("Precision (Macro):    %.4f", test_metrics["precision_macro"])
    logger.info("Recall (Macro):       %.4f", test_metrics["recall_macro"])
    logger.info("F1 (Clickbait):       %.4f", test_metrics["f1_clickbait"])
    logger.info("F1 (Not-Clickbait):   %.4f", test_metrics["f1_not_clickbait"])
    logger.info("Confusion Matrix:     %s", test_metrics["confusion_matrix"])
    logger.info("=" * 60)

    scalar_test_metrics = {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
    exp.log_metrics({"split": "test", **scalar_test_metrics})

    test_metrics_path = exp.run_dir / "metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (exp.run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    exp.finish()
    return test_metrics
