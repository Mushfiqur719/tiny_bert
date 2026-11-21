"""
src/training/train_teacher.py — Fine-tuning pipeline for Teacher (BanglaBERT) model.

Workflow:
  1. Load and clean BanglaBait data according to fixed splits (data/processed/splits.json).
  2. Compute balanced class weights to address the ~65/35 class imbalance.
  3. Fine-tune BanglaBERT (csebuetnlp/banglabert) with HuggingFace Trainer.
  4. Use validation split for early stopping and checkpoint selection on macro-F1.
  5. Freeze teacher model parameters (requires_grad = False).
  6. Save teacher model + tokenizer to experiments/teacher/ and checkpoint dirs.
  7. Evaluate ONCE on the held-out test split and save final metrics.
  8. Pre-compute and save soft predictions (logits) on the full train split for Phase 4 distillation.

Usage:
    from src.training.train_teacher import train_teacher
    test_metrics = train_teacher(cfg, seed=42)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_metrics_from_eval_pred,
    find_optimal_threshold,
    predict_with_threshold,
)
from src.models.teacher import (
    freeze_model,
    get_teacher_tokenizer,
    load_teacher_model,
    save_teacher_checkpoint,
)
from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
from src.utils.logging_utils import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def compute_rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Compute symmetric KL divergence between two dropout forward passes (R-Drop)."""
    log_p1 = torch.nn.functional.log_softmax(logits1, dim=-1)
    log_p2 = torch.nn.functional.log_softmax(logits2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = torch.nn.functional.kl_div(log_p2, p1, reduction="batchmean")
    kl_21 = torch.nn.functional.kl_div(log_p1, p2, reduction="batchmean")
    return 0.5 * (kl_12 + kl_21)


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_factor: float = 0.90,
) -> list[dict[str, Any]]:
    """Construct optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD)."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight")
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 12)
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        if "embeddings" in name:
            depth = 0
        elif "encoder.layer." in name:
            parts = name.split("encoder.layer.")[1].split(".")
            layer_idx = int(parts[0])
            depth = layer_idx + 1
        else:
            # Classifier head gets full base_lr
            depth = num_layers + 1

        lr = round(base_lr * (llrd_factor ** (num_layers + 1 - depth)), 10)
        groups.setdefault((lr, wd), []).append(param)

    return [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]


class WeightedTrainer(Trainer):
    """Custom HuggingFace Trainer supporting class-weighted CE, label smoothing, R-Drop, and LLRD."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        rdrop_alpha: float = 0.0,
        llrd_factor: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.rdrop_alpha = float(rdrop_alpha)
        self.llrd_factor = float(llrd_factor)

    def create_optimizer(self):
        if self.optimizer is None and 0.0 < self.llrd_factor < 1.0:
            param_groups = build_llrd_param_groups(
                self.model,
                base_lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                llrd_factor=self.llrd_factor,
            )
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute weighted cross-entropy loss (with optional label smoothing and R-Drop during training)."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]

        if labels is not None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            smoothing = self.label_smoothing if model.training else 0.0
            loss_fct = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            if model.training and self.rdrop_alpha > 0.0:
                outputs2 = model(**inputs)
                logits2 = outputs2.get("logits") if isinstance(outputs2, dict) else outputs2[0]
                loss2 = loss_fct(logits2.view(-1, self.model.config.num_labels), labels.view(-1))
                kl_loss = compute_rdrop_kl(logits, logits2)
                loss = 0.5 * (loss + loss2) + self.rdrop_alpha * kl_loss
        else:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class MetricLoggingCallback(TrainerCallback):
    """Callback to log evaluation metrics to ExperimentLogger at every epoch."""

    def __init__(self, exp_logger: ExperimentLogger) -> None:
        self.exp_logger = exp_logger

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if metrics:
            epoch = int(metrics.get("epoch", state.epoch or 0))
            cleaned_metrics = {
                "epoch": epoch,
                "step": state.global_step,
                "split": "val",
                "val_loss": round(float(metrics.get("eval_loss", 0.0)), 4),
                "val_accuracy": round(float(metrics.get("eval_accuracy", 0.0)), 4),
                "val_f1_macro": round(float(metrics.get("eval_f1_macro", 0.0)), 4),
                "val_f1_clickbait": round(float(metrics.get("eval_f1_clickbait", 0.0)), 4),
                "val_f1_not_clickbait": round(float(metrics.get("eval_f1_not_clickbait", 0.0)), 4),
            }
            self.exp_logger.log_metrics(cleaned_metrics)


def calculate_balanced_class_weights(train_labels: Sequence[int]) -> torch.Tensor:
    """Calculate balanced class weights: N / (n_classes * class_count).

    Args:
        train_labels: List of ground-truth labels in train split.

    Returns:
        1D Tensor with weights for classes [0, 1].
    """
    labels = np.asarray(train_labels)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    class_counts = np.bincount(labels, minlength=n_classes)

    weights = n_samples / (n_classes * class_counts.astype(float))
    logger.info(
        "Calculated class weights (Class 0: %d, Class 1: %d) -> Weights: [0: %.4f, 1: %.4f]",
        class_counts[0],
        class_counts[1],
        weights[0],
        weights[1],
    )
    return torch.tensor(weights, dtype=torch.float)


def train_teacher(cfg: DictConfig, seed: Optional[int] = None) -> dict[str, Any]:
    """Execute Teacher (BanglaBERT) fine-tuning pipeline.

    Args:
        cfg: Configuration object.
        seed: Random seed override.

    Returns:
        Dictionary of final test metrics.
    """
    effective_seed = int(seed if seed is not None else cfg.seed)
    seed_everything(effective_seed)

    run_name = str(cfg.training.run_name)
    exp = ExperimentLogger(run_name=run_name, cfg=cfg, seed=effective_seed)

    logger.info("=" * 60)
    logger.info("Teacher Model (BanglaBERT) Fine-Tuning Pipeline")
    logger.info("Run Name: %s | Seed: %d", run_name, effective_seed)
    logger.info("=" * 60)

    # 1. Load and clean data
    hf_id = getattr(cfg.data, "hf_dataset_id", "Motahar/banglabait")
    labeled_df, _ = load_raw(cfg, hf_dataset_id=hf_id)
    labeled_df = clean_dataframe(
        labeled_df,
        headline_col=cfg.data.text_col,
        min_char_length=int(cfg.data.min_char_length),
    )
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)

    # 2. Apply fixed splits
    splits = load_splits(cfg)
    train_df, val_df, test_df = apply_splits(labeled_df, splits)

    # 3. Class weights
    class_weights = calculate_balanced_class_weights(train_df[cfg.data.label_col].tolist())

    # 4. Tokenizer & Datasets
    teacher_model_name = str(cfg.model.teacher_model_name)
    tokenizer = get_teacher_tokenizer(teacher_model_name)
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

    # 5. Model
    model = load_teacher_model(
        model_name_or_path=teacher_model_name,
        num_labels=int(cfg.data.num_labels),
        dropout=float(cfg.model.dropout),
    )

    # 6. Training Arguments
    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    batch_size = int(cfg.training.batch_size)
    grad_accum = int(getattr(cfg.training, "gradient_accumulation_steps", 1))
    if bool(getattr(cfg.training, "gradient_checkpointing", False)) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    epochs = float(cfg.training.epochs)
    eff_bs = batch_size * grad_accum
    total_steps = int((len(train_ds) // eff_bs + (1 if len(train_ds) % eff_bs != 0 else 0)) * epochs)
    warmup_ratio = float(cfg.training.warmup_ratio)
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler_type = str(getattr(cfg.training, "lr_scheduler_type", "cosine"))
    label_smoothing = float(getattr(cfg.training, "label_smoothing", 0.0))
    rdrop_alpha = float(getattr(cfg.training, "rdrop_alpha", 0.0))
    llrd_factor = float(getattr(cfg.training, "llrd_factor", 1.0))

    training_args = TrainingArguments(
        output_dir=str(exp.checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg.training.lr),
        lr_scheduler_type=lr_scheduler_type,
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
        label_smoothing=label_smoothing,
        rdrop_alpha=rdrop_alpha,
        llrd_factor=llrd_factor,
    )

    # 7. Train
    logger.info("Starting Teacher fine-tuning on %d samples (Val: %d)...", len(train_ds), len(val_ds))
    train_result = trainer.train()
    logger.info("Teacher training completed. Global step: %d", train_result.global_step)

    # 8. Save best model checkpoint
    best_ckpt_dir = exp.checkpoint_path("best_val_f1")
    trainer.save_model(str(best_ckpt_dir))
    tokenizer.save_pretrained(str(best_ckpt_dir))

    # Save to canonical experiments/teacher/ location only when run_name == "teacher"
    canonical_teacher_dir = Path(cfg.training.output_dir) / "teacher"
    canonical_teacher_dir.mkdir(parents=True, exist_ok=True)
    save_student_or_teacher_dir = exp.run_dir
    trainer.save_model(str(save_student_or_teacher_dir))
    tokenizer.save_pretrained(str(save_student_or_teacher_dir))
    if run_name == "teacher":
        trainer.save_model(str(canonical_teacher_dir))
        tokenizer.save_pretrained(str(canonical_teacher_dir))

    # 9. Freeze teacher model parameters
    freeze_model(trainer.model)

    # 10. Tune Decision Threshold on Validation Split (Test set remains untouched)
    val_prediction_output = trainer.predict(val_ds)
    val_logits = val_prediction_output.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    optimal_threshold, best_val_f1_cal = find_optimal_threshold(
        logits=val_logits,
        y_true=val_prediction_output.label_ids,
    )
    logger.info(
        "Validation threshold calibration: optimal_threshold=%.2f (Val Macro-F1: %.4f)",
        optimal_threshold,
        best_val_f1_cal,
    )

    # 11. Held-out Test Set Evaluation (EXACTLY ONCE)
    logger.info("Evaluating Teacher model ONCE on held-out test split (%d samples)...", len(test_ds))
    test_prediction_output = trainer.predict(test_ds)
    test_logits = test_prediction_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_preds = np.argmax(test_logits, axis=1)
    test_preds_cal = predict_with_threshold(test_logits, threshold=optimal_threshold)
    test_labels = test_prediction_output.label_ids

    np.save(exp.run_dir / "test_logits.npy", test_logits)
    np.save(exp.run_dir / "test_labels.npy", test_labels)
    np.save(exp.run_dir / "val_logits.npy", val_logits)
    np.save(exp.run_dir / "val_labels.npy", val_prediction_output.label_ids)

    test_metrics = compute_classification_metrics(y_pred=test_preds, y_true=test_labels)
    test_metrics_cal = compute_classification_metrics(y_pred=test_preds_cal, y_true=test_labels)
    test_metrics["optimal_threshold"] = optimal_threshold
    test_metrics["calibrated"] = test_metrics_cal

    logger.info("=" * 60)
    logger.info("TEACHER FINAL TEST SET METRICS (Threshold=0.50):")
    logger.info("Accuracy:             %.4f", test_metrics["accuracy"])
    logger.info("F1 (Macro):           %.4f", test_metrics["f1_macro"])
    logger.info("Precision (Macro):    %.4f", test_metrics["precision_macro"])
    logger.info("Recall (Macro):       %.4f", test_metrics["recall_macro"])
    logger.info("F1 (Clickbait):       %.4f", test_metrics["f1_clickbait"])
    logger.info("F1 (Not-Clickbait):   %.4f", test_metrics["f1_not_clickbait"])
    logger.info("Confusion Matrix:     %s", test_metrics["confusion_matrix"])
    logger.info("-" * 60)
    logger.info("CALIBRATED TEST METRICS (Val-Tuned Threshold=%.2f):", optimal_threshold)
    logger.info("Accuracy (Cal):       %.4f", test_metrics_cal["accuracy"])
    logger.info("F1 Macro (Cal):       %.4f", test_metrics_cal["f1_macro"])
    logger.info("F1 Clickbait (Cal):   %.4f", test_metrics_cal["f1_clickbait"])
    logger.info("Confusion Matrix:     %s", test_metrics_cal["confusion_matrix"])
    logger.info("=" * 60)

    # Log to ExperimentLogger and save metrics.json
    scalar_test_metrics = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
    exp.log_metrics({"split": "test", **scalar_test_metrics})

    test_metrics_path = exp.run_dir / "metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (exp.run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    if run_name == "teacher":
        (canonical_teacher_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    logger.info("Test metrics saved to %s", test_metrics_path)

    # 11. Pre-compute and save Teacher soft labels (logits) on full train split
    logger.info("Pre-computing Teacher soft labels on full train split (%d samples)...", len(train_ds))
    train_prediction_output = trainer.predict(train_ds)
    raw_train_logits = train_prediction_output.predictions
    if isinstance(raw_train_logits, tuple):
        raw_train_logits = raw_train_logits[0]

    train_soft_labels_tensor = torch.tensor(raw_train_logits, dtype=torch.float32)
    soft_labels_dict = {
        "logits": train_soft_labels_tensor,
        "labels": torch.tensor(train_df[cfg.data.label_col].tolist(), dtype=torch.long),
        "headline_indices": splits["labeled"]["train"],
        "teacher_model_name": teacher_model_name,
        "seed": effective_seed,
    }

    run_soft_path = exp.run_dir / "train_soft_labels.pt"
    torch.save(soft_labels_dict, run_soft_path)
    if run_name == "teacher":
        soft_labels_path = canonical_teacher_dir / "train_soft_labels.pt"
        torch.save(soft_labels_dict, soft_labels_path)
    logger.info(
        "Teacher train soft labels saved to %s (tensor shape: %s)",
        run_soft_path,
        tuple(train_soft_labels_tensor.shape),
    )

    exp.finish()
    return test_metrics
