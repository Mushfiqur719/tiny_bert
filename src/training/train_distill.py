"""
src/training/train_distill.py — Knowledge Distillation training pipeline for TinyBanglaBERT Student.

Runs either:
  1. `student_random_distilled`: Student with Random Initialization + Distillation from Teacher.
  2. `student_teacher_init_distilled`: Student with Teacher-Projected Init + Distillation from Teacher.

Distillation Loss:
  L = alpha * CE(student_logits, true_labels) + (1 - alpha) * (T^2) * KL(student/T, teacher/T)

Precomputed Soft Targets:
  Uses precomputed Teacher logits from `experiments/teacher/train_soft_labels.pt`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
)
from src.models.distillation import (
    DistillationDataset,
    KnowledgeDistillationLoss,
    load_teacher_soft_labels,
)
from src.models.student import (
    create_student_config,
    init_from_teacher,
    init_random,
    save_student_checkpoint,
)
from src.models.teacher import get_teacher_tokenizer
from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
from src.training.train_teacher import (
    MetricLoggingCallback,
    calculate_balanced_class_weights,
)
from src.utils.logging_utils import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Custom Trainer implementing Hinton Knowledge Distillation Loss with precomputed teacher logits."""

    def __init__(
        self,
        kd_loss_fn: KnowledgeDistillationLoss,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kd_loss_fn = kd_loss_fn

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute Knowledge Distillation loss for training, or standard CE for evaluation."""
        labels = inputs.get("labels")
        teacher_logits = inputs.get("teacher_logits")

        # Forward pass on student
        student_inputs = {k: v for k, v in inputs.items() if k != "teacher_logits"}
        outputs = model(**student_inputs)
        student_logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]

        if teacher_logits is not None and model.training:
            # Training mode: Apply Knowledge Distillation Loss
            loss, _ = self.kd_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
            )
        else:
            # Eval mode: Standard Cross-Entropy on ground-truth labels
            if labels is not None:
                weights = self.kd_loss_fn.class_weights.to(student_logits.device) if self.kd_loss_fn.class_weights is not None else None
                loss_fct = nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(student_logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def train_student_distill(
    cfg: DictConfig,
    init_type: str = "random",
    temperature: Optional[float] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
    soft_labels_path: str = "experiments/teacher/train_soft_labels.pt",
) -> dict[str, Any]:
    """Execute Knowledge Distillation training for TinyBanglaBERT Student.

    Args:
        cfg: Configuration object.
        init_type: 'random' or 'teacher_init'.
        temperature: Distillation temperature T override.
        alpha: Distillation hard/soft weight alpha override.
        seed: Seed override.
        soft_labels_path: Path to pre-computed teacher soft labels.

    Returns:
        Dictionary of final test metrics.
    """
    effective_seed = int(seed if seed is not None else cfg.seed)
    seed_everything(effective_seed)

    T = float(temperature if temperature is not None else getattr(cfg.distillation, "temperature", 4.0))
    a = float(alpha if alpha is not None else getattr(cfg.distillation, "alpha", 0.7))

    run_name = str(cfg.training.get("run_name", f"student_{init_type}_distilled_T{T}_a{a}"))
    exp = ExperimentLogger(run_name=run_name, cfg=cfg, seed=effective_seed)

    logger.info("=" * 60)
    logger.info("Student Knowledge Distillation (%s Init)", init_type.upper())
    logger.info("Temperature: %.2f | Alpha: %.2f | Run Name: %s | Seed: %d", T, a, run_name, effective_seed)
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

    # 2. Class weights & Soft labels
    class_weights = calculate_balanced_class_weights(train_df[cfg.data.label_col].tolist())
    soft_data = load_teacher_soft_labels(soft_labels_path)
    teacher_train_logits = soft_data["logits"]

    # 3. Tokenizer & Datasets
    tokenizer_name = str(cfg.model.teacher_model_name)
    tokenizer = get_teacher_tokenizer(tokenizer_name)
    max_seq_len = int(cfg.model.max_seq_len)

    train_ds = DistillationDataset(
        headlines=train_df[cfg.data.text_col].tolist(),
        labels=train_df[cfg.data.label_col].tolist(),
        teacher_logits=teacher_train_logits,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    val_ds = BanglaHeadlineDataset(
        headlines=val_df[cfg.data.text_col].tolist(),
        labels=val_df[cfg.data.label_col].tolist(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    test_ds = BanglaHeadlineDataset(
        headlines=test_df[cfg.data.text_col].tolist(),
        labels=test_df[cfg.data.label_col].tolist(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
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

    # 5. KD Loss Module
    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=T,
        alpha=a,
        class_weights=class_weights,
    )

    # 6. Training Arguments
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

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_from_eval_pred,
        callbacks=callbacks,
        kd_loss_fn=kd_loss_fn,
    )

    # 7. Train
    logger.info("Training Student Distillation on %d samples (T=%.2f, alpha=%.2f)...", len(train_ds), T, a)
    trainer.train()

    # 8. Save Checkpoint
    best_ckpt_dir = exp.checkpoint_path("best_val_f1")
    trainer.save_model(str(best_ckpt_dir))
    tokenizer.save_pretrained(str(best_ckpt_dir))
    save_student_checkpoint(trainer.model, tokenizer, exp.run_dir)

    # 9. Test Set Evaluation (EXACTLY ONCE)
    logger.info("Evaluating Distilled Student on held-out test split (%d samples)...", len(test_ds))
    test_prediction_output = trainer.predict(test_ds)
    test_logits = test_prediction_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_preds = np.argmax(test_logits, axis=1)
    test_labels = test_prediction_output.label_ids

    test_metrics = compute_classification_metrics(y_pred=test_preds, y_true=test_labels)
    logger.info("=" * 60)
    logger.info("STUDENT DISTILLED (%s) FINAL TEST SET METRICS:", init_type.upper())
    logger.info("Temperature:          %.2f | Alpha: %.2f", T, a)
    logger.info("Accuracy:             %.4f", test_metrics["accuracy"])
    logger.info("F1 (Macro):           %.4f", test_metrics["f1_macro"])
    logger.info("Precision (Macro):    %.4f", test_metrics["precision_macro"])
    logger.info("Recall (Macro):       %.4f", test_metrics["recall_macro"])
    logger.info("F1 (Clickbait):       %.4f", test_metrics["f1_clickbait"])
    logger.info("F1 (Not-Clickbait):   %.4f", test_metrics["f1_not_clickbait"])
    logger.info("Confusion Matrix:     %s", test_metrics["confusion_matrix"])
    logger.info("=" * 60)

    scalar_test_metrics = {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}
    exp.log_metrics({"split": "test", "temperature": T, "alpha": a, **scalar_test_metrics})

    test_metrics_path = exp.run_dir / "metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (exp.run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    exp.finish()
    return test_metrics
