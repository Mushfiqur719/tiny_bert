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
    find_optimal_threshold,
    predict_with_threshold,
)
from src.models.distillation import (
    DistillationDataset,
    IntermediateAlignmentModule,
    KnowledgeDistillationLoss,
    load_teacher_soft_labels,
)
from src.models.student import (
    _select_teacher_layers,
    create_student_config,
    init_from_small_pretrained,
    init_from_teacher,
    init_random,
    save_student_checkpoint,
)
from src.models.teacher import get_teacher_tokenizer
from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
from src.training.train_teacher import (
    MetricLoggingCallback,
    calculate_balanced_class_weights,
    compute_rdrop_kl,
)
from src.utils.logging_utils import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Custom Trainer implementing Hinton KD Loss + optional Intermediate Hidden & Attention KD."""

    def __init__(
        self,
        kd_loss_fn: KnowledgeDistillationLoss,
        teacher_model: Optional[nn.Module] = None,
        inter_align_module: Optional[IntermediateAlignmentModule] = None,
        rdrop_alpha: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kd_loss_fn = kd_loss_fn
        self.teacher_model = teacher_model
        self.inter_align_module = inter_align_module
        self.rdrop_alpha = float(rdrop_alpha)
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.inter_align_module is not None:
            device = next(self.model.parameters()).device
            self.inter_align_module.to(device)
            existing_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            extra_params = [p for p in self.inter_align_module.parameters() if id(p) not in existing_params]
            if extra_params:
                optimizer.add_param_group(
                    {"params": extra_params, "lr": self.args.learning_rate, "weight_decay": 0.0}
                )
        return optimizer

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

        use_inter = (
            model.training
            and self.teacher_model is not None
            and self.inter_align_module is not None
        )
        need_attns = use_inter and self.inter_align_module.attention_loss_weight > 0.0
        need_hidden = use_inter and self.inter_align_module.hidden_loss_weight > 0.0

        # Forward pass on student
        student_inputs = {k: v for k, v in inputs.items() if k != "teacher_logits"}
        if use_inter:
            student_inputs["output_hidden_states"] = need_hidden
            student_inputs["output_attentions"] = need_attns

        outputs = model(**student_inputs)
        student_logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]

        if model.training:
            if teacher_logits is None:
                raise ValueError(
                    "teacher_logits is missing from the training batch! "
                    "Ensure TrainingArguments(remove_unused_columns=False) is set."
                )
            # Training mode: Apply Knowledge Distillation Loss
            loss, _ = self.kd_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
            )
            if self.rdrop_alpha > 0.0:
                outputs2 = model(**student_inputs)
                student_logits2 = outputs2.get("logits") if isinstance(outputs2, dict) else outputs2[0]
                loss2, _ = self.kd_loss_fn(
                    student_logits=student_logits2,
                    teacher_logits=teacher_logits,
                    labels=labels,
                )
                rdrop_kl = compute_rdrop_kl(student_logits, student_logits2)
                loss = 0.5 * (loss + loss2) + self.rdrop_alpha * rdrop_kl
            if use_inter:
                dev = student_logits.device
                if next(self.teacher_model.parameters()).device != dev:
                    self.teacher_model.to(dev)
                if next(self.inter_align_module.parameters()).device != dev:
                    self.inter_align_module.to(dev)
                with torch.no_grad():
                    t_inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        "output_hidden_states": need_hidden,
                        "output_attentions": need_attns,
                    }
                    if "token_type_ids" in inputs:
                        t_inputs["token_type_ids"] = inputs["token_type_ids"]
                    t_out = self.teacher_model(**t_inputs)

                inter_loss, _ = self.inter_align_module(
                    student_hidden_states=getattr(outputs, "hidden_states", None),
                    teacher_hidden_states=getattr(t_out, "hidden_states", None),
                    student_attentions=getattr(outputs, "attentions", None),
                    teacher_attentions=getattr(t_out, "attentions", None),
                    attention_mask=inputs.get("attention_mask"),
                )
                loss = loss + inter_loss
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

    # 2. Class weights & Soft labels (with automatic index alignment for clean/dedup train splits)
    class_weights = calculate_balanced_class_weights(train_df[cfg.data.label_col].tolist())
    soft_data = load_teacher_soft_labels(soft_labels_path)
    teacher_train_logits = soft_data["logits"]
    current_train_indices = splits["labeled"]["train"]
    saved_train_indices = soft_data.get("headline_indices")
    if len(teacher_train_logits) != len(train_df) and saved_train_indices is not None:
        idx_to_pos = {int(orig_idx): pos for pos, orig_idx in enumerate(saved_train_indices)}
        aligned_positions = [idx_to_pos[int(i)] for i in current_train_indices]
        teacher_train_logits = teacher_train_logits[aligned_positions]
        logger.info(
            "Aligned teacher soft labels to filtered train split (%d -> %d samples).",
            len(saved_train_indices),
            len(teacher_train_logits),
        )

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
    num_hidden_layers = int(getattr(cfg.model, "num_hidden_layers", 4))
    hidden_size = int(getattr(cfg.model, "hidden_size", 256))
    num_attention_heads = int(getattr(cfg.model, "num_attention_heads", 4))
    intermediate_size = int(getattr(cfg.model, "intermediate_size", 1024))
    layer_selection = str(getattr(cfg.model, "layer_selection", "strided"))

    student_cfg = create_student_config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_labels=int(cfg.data.num_labels),
        dropout=float(cfg.model.dropout),
    )

    if init_type == "random":
        model = init_random(student_cfg)
    elif init_type in ("teacher_init", "teacher"):
        teacher_path = "experiments/teacher"
        logger.info("Loading fine-tuned Teacher model from '%s' for initialization...", teacher_path)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        model = init_from_teacher(
            teacher_model,
            student_cfg,
            strategy="projection",
            layer_selection=layer_selection,
        )
    elif init_type == "small_pretrained":
        pretrained_small = str(getattr(cfg.model, "pretrained_small_name", "csebuetnlp/banglabert_small"))
        model = init_from_small_pretrained(
            pretrained_name=pretrained_small,
            num_hidden_layers=num_hidden_layers,
            num_labels=int(cfg.data.num_labels),
            dropout=float(cfg.model.dropout),
            layer_selection=layer_selection,
        )
    else:
        raise ValueError(
            f"Unknown init_type: '{init_type}'. Use 'random', 'teacher_init', or 'small_pretrained'."
        )

    # 5. KD Loss Module & Optional Intermediate-Layer Alignment
    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=T,
        alpha=a,
        class_weights=class_weights,
    )

    use_inter_kd = bool(getattr(cfg.distillation, "intermediate_layers", False))
    hidden_loss_weight = float(getattr(cfg.distillation, "hidden_loss_weight", 0.0))
    attention_loss_weight = float(getattr(cfg.distillation, "attention_loss_weight", 0.0))

    online_teacher = None
    inter_align_module = None
    if use_inter_kd and (hidden_loss_weight > 0.0 or attention_loss_weight > 0.0):
        teacher_path = "experiments/teacher"
        logger.info(
            "Intermediate-layer KD enabled (hidden_w=%.2f, attn_w=%.2f). Loading frozen teacher from '%s'...",
            hidden_loss_weight,
            attention_loss_weight,
            teacher_path,
        )
        if attention_loss_weight > 0.0:
            model.config._attn_implementation = "eager"
            online_teacher = AutoModelForSequenceClassification.from_pretrained(
                teacher_path, attn_implementation="eager"
            )
        else:
            online_teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        t_layers = _select_teacher_layers(
            num_teacher_layers=online_teacher.config.num_hidden_layers,
            num_student_layers=model.config.num_hidden_layers,
            layer_selection=layer_selection,
        )
        inter_align_module = IntermediateAlignmentModule(
            student_hidden_size=model.config.hidden_size,
            teacher_hidden_size=online_teacher.config.hidden_size,
            teacher_layer_indices=t_layers,
            hidden_loss_weight=hidden_loss_weight,
            attention_loss_weight=attention_loss_weight,
        )

    # 6. Training Arguments
    use_fp16 = bool(cfg.training.fp16) and torch.cuda.is_available()
    batch_size = int(cfg.training.batch_size)
    epochs = float(cfg.training.epochs)
    total_steps = int((len(train_ds) // batch_size + (1 if len(train_ds) % batch_size != 0 else 0)) * epochs)
    warmup_steps = int(total_steps * float(cfg.training.warmup_ratio))
    lr_scheduler_type = str(getattr(cfg.training, "lr_scheduler_type", "cosine"))

    training_args = TrainingArguments(
        output_dir=str(exp.checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
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
        remove_unused_columns=False,
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

    rdrop_alpha = float(getattr(cfg.training, "rdrop_alpha", 0.0))

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_from_eval_pred,
        callbacks=callbacks,
        kd_loss_fn=kd_loss_fn,
        teacher_model=online_teacher,
        inter_align_module=inter_align_module,
        rdrop_alpha=rdrop_alpha,
    )

    # 7. Train
    logger.info("Training Student Distillation on %d samples (T=%.2f, alpha=%.2f)...", len(train_ds), T, a)
    trainer.train()

    # 8. Save Checkpoint
    best_ckpt_dir = exp.checkpoint_path("best_val_f1")
    trainer.save_model(str(best_ckpt_dir))
    tokenizer.save_pretrained(str(best_ckpt_dir))
    save_student_checkpoint(trainer.model, tokenizer, exp.run_dir)

    # 9. Tune Decision Threshold on Validation Split (Test set remains untouched)
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

    # 10. Test Set Evaluation (EXACTLY ONCE)
    logger.info("Evaluating Distilled Student on held-out test split (%d samples)...", len(test_ds))
    test_prediction_output = trainer.predict(test_ds)
    test_logits = test_prediction_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_preds = np.argmax(test_logits, axis=1)
    test_preds_cal = predict_with_threshold(test_logits, threshold=optimal_threshold)
    test_labels = test_prediction_output.label_ids

    # Persist logits so bootstrap CIs / McNemar can be computed later without re-running
    np.save(exp.run_dir / "test_logits.npy", test_logits)
    np.save(exp.run_dir / "test_labels.npy", test_labels)
    np.save(exp.run_dir / "val_logits.npy", val_logits)
    np.save(exp.run_dir / "val_labels.npy", val_prediction_output.label_ids)

    test_metrics = compute_classification_metrics(y_pred=test_preds, y_true=test_labels)
    test_metrics_cal = compute_classification_metrics(y_pred=test_preds_cal, y_true=test_labels)
    test_metrics["optimal_threshold"] = optimal_threshold
    test_metrics["calibrated"] = test_metrics_cal

    logger.info("=" * 60)
    logger.info("STUDENT DISTILLED (%s) FINAL TEST SET METRICS (Threshold=0.50):", init_type.upper())
    logger.info("Temperature:          %.2f | Alpha: %.2f", T, a)
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

    scalar_test_metrics = {
        k: v for k, v in test_metrics.items() if isinstance(v, (int, float))
    }
    exp.log_metrics({"split": "test", "temperature": T, "alpha": a, **scalar_test_metrics})

    test_metrics_path = exp.run_dir / "metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (exp.run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    exp.finish()
    return test_metrics
