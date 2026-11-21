"""scripts/run_v3_grid.py — Phase F: Controlled Factorial Grid & Depth Study on Clickbait.

Core Research Question:
  In low-resource fine-tuning, does initialisation matter more than the distillation objective?

Design (v3 Rules 10, 11, 12):
  - Identical 4-layer, 7.42M parameter architecture across all core arms:
      * init:  rand | strided ([2,5,8,11]) | first4 ([0,1,2,3]) | last4 ([8,9,10,11])
      * obj:   ce | kd (T=4.0, alpha=0.5) | kd_hid (T=4.0, alpha=0.5 + hidden alignment)
      * frac:  10 | 25 | 50 | 100 (% of v3 train split, strictly nested)
      * seeds: [13, 42, 87, 1234, 2026]
  - Fixed hyperparameters across ALL arms: lr=5e-5, wd=0.01, warmup=0.1, bs=32, max_len=64, epochs=10 (patience 3).
  - Depth study at 100% data: depths {4, 6, 12} x {ce, kd} x 5 seeds.
  - Strictly validation-only (NEVER reads test labels).
  - Resumable: skips runs where metrics.json already exists.

Usage:
  .venv\\Scripts\\python.exe -m scripts.run_v3_grid --inits rand strided --objs ce kd --fracs 10 25 50 100
  .venv\\Scripts\\python.exe -m scripts.run_v3_grid --depth-study
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from scripts.train_v3_phase_e import (
    CLEAN_CSV_PATH,
    CLEAN_FULL_CSV_PATH,
    SEEDS,
    SPLITS_PATH,
    WeightedCETrainer,
    compute_full_metrics,
    format_headlines,
    get_dataset_paths,
    get_git_sha,
)
from src.evaluation.metrics import compute_metrics_from_eval_pred
from src.models.distillation import (
    DistillationDataset,
    IntermediateAlignmentModule,
    KnowledgeDistillationLoss,
)
from src.models.student import (
    _select_teacher_layers,
    count_parameters,
    init_from_small_pretrained,
    init_small_random,
)
from src.training.dataset import BanglaHeadlineDataset
from src.training.train_teacher import calculate_balanced_class_weights
from src.utils.seeding import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Fixed Rule-11 Hyperparameters across ALL arms
FIXED_LR = 5e-5
FIXED_WD = 0.01
FIXED_WARMUP = 0.1
FIXED_BS = 32
FIXED_MAX_LEN = 64
FIXED_EPOCHS = 10
FIXED_T = 4.0
FIXED_ALPHA = 0.5
FIXED_BETA_HIDDEN = 0.5
OOF_SOFT_LABELS_PATH = Path("data/processed/v3/teacher_oof_logits.pt")


class ValCurveCallback(TrainerCallback):
    """Records validation loss and Macro-F1 per epoch for Phase G mechanism analysis."""

    def __init__(self):
        self.history: list[dict[str, float]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.history.append({
                "epoch": round(float(state.epoch or 0.0), 2),
                "eval_loss": round(float(metrics.get("eval_loss", 0.0)), 4),
                "eval_f1_macro": round(float(metrics.get("eval_f1_macro", 0.0)), 4),
            })


class V3DistillTrainer(Trainer):
    """Unified KD / KD+Hidden trainer with strict teacher_logits guard."""

    def __init__(
        self,
        kd_loss_fn: KnowledgeDistillationLoss,
        teacher_model: Optional[nn.Module] = None,
        align_module: Optional[IntermediateAlignmentModule] = None,
        teacher_layer_map: Optional[list[int]] = None,
        beta_hidden: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kd_loss_fn = kd_loss_fn
        self.teacher_model = teacher_model
        self.align_module = align_module
        self.teacher_layer_map = teacher_layer_map
        self.beta_hidden = beta_hidden
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def create_optimizer(self):
        opt = super().create_optimizer()
        if self.align_module is not None and self.beta_hidden > 0.0:
            opt.add_param_group({
                "params": list(self.align_module.parameters()),
                "weight_decay": self.args.weight_decay,
            })
        return opt

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits", None)

        use_hidden = (
            model.training
            and self.teacher_model is not None
            and self.align_module is not None
            and self.beta_hidden > 0.0
        )

        outputs = model(**inputs, output_hidden_states=use_hidden)
        student_logits = outputs.logits

        if teacher_logits is None:
            if model.training:
                raise RuntimeError("teacher_logits missing during KD training! Check remove_unused_columns=False.")
            loss = nn.CrossEntropyLoss()(student_logits, labels)
            return (loss, outputs) if return_outputs else loss

        loss, _ = self.kd_loss_fn(student_logits, teacher_logits.to(student_logits.device), labels)

        if use_hidden:
            device = student_logits.device
            self.teacher_model.to(device)
            self.align_module.to(device)
            with torch.no_grad():
                t_out = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )
            hid_loss, _ = self.align_module(
                outputs.hidden_states,
                t_out.hidden_states,
                None,
                None,
                inputs["attention_mask"],
            )
            loss = loss + hid_loss

        return (loss, outputs) if return_outputs else loss


def build_student_model(init_name: str, num_layers: int = 4) -> nn.Module:
    """Instantiate parameter-matched compact student architecture."""
    if init_name == "rand":
        return init_small_random(pretrained_name="csebuetnlp/banglabert_small", num_hidden_layers=num_layers)
    elif init_name in ("strided", "first4", "last4"):
        return init_from_small_pretrained(
            pretrained_name="csebuetnlp/banglabert_small",
            num_hidden_layers=num_layers,
            layer_selection=init_name,
        )
    elif init_name == "full":
        return init_from_small_pretrained(
            pretrained_name="csebuetnlp/banglabert_small",
            num_hidden_layers=num_layers,
            layer_selection="first",
        )
    else:
        raise ValueError(f"Unknown init arm: {init_name}")


def load_teacher_val_preds(text_mode: str = "title", dataset: str = "banglabait", suffix: str = "") -> Optional[np.ndarray]:
    """Load teacher_base seed42 validation predictions for student-teacher agreement metric."""
    _, exp_dir, _ = get_dataset_paths(dataset)
    p = exp_dir / f"teacher_base{suffix}" / "seed42" / "val_logits.npy"
    if p.exists():
        return np.argmax(np.load(p), axis=1)
    return None


def run_single_cell(
    run_name: str,
    init_name: str,
    obj_name: str,
    frac: int,
    num_layers: int,
    seed: int,
    clean_df: pd.DataFrame,
    splits: dict,
    tokenizer,
    oof_data: Optional[dict],
    teacher_model_for_hid: Optional[nn.Module],
    teacher_val_preds: Optional[np.ndarray],
    git_sha: str,
    text_mode: str = "title",
    dataset: str = "banglabait",
    unweighted: bool = False,
    skip_existing: bool = True,
) -> None:
    data_dir, exp_dir, _ = get_dataset_paths(dataset)
    run_dir = exp_dir / run_name / f"seed{seed}"
    if skip_existing and (run_dir / "metrics.json").exists():
        logger.info("[SKIP] %s/seed%d already completed.", run_name, seed)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    max_len = 256 if text_mode in ("title_body", "cat_title_body") else FIXED_MAX_LEN
    batch_size = 16 if (max_len == 256 and (obj_name == "kd_hid" or num_layers == 12)) else FIXED_BS

    # Load nested subset indices
    frac_path = data_dir / f"train_frac_{frac}.json"
    frac_indices = json.loads(frac_path.read_text(encoding="utf-8"))["indices"]

    train_df = clean_df.loc[frac_indices].reset_index(drop=True)
    val_df = clean_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    val_labels = val_df["label"].to_numpy()

    tr_titles, tr_bodies = format_headlines(train_df, text_mode)
    va_titles, va_bodies = format_headlines(val_df, text_mode)

    class_weights = None if unweighted else calculate_balanced_class_weights(train_df["label"].tolist())
    val_ds = BanglaHeadlineDataset(va_titles, val_df["label"].tolist(), tokenizer, max_seq_len=max_len, bodies=va_bodies)

    model = build_student_model(init_name, num_layers=num_layers)
    n_params = count_parameters(model)["total_params"]

    curve_cb = ValCurveCallback()
    ckpt_dir = run_dir / "hf_checkpoints"

    total_steps = max(1, (len(train_df) // batch_size + 1) * FIXED_EPOCHS)
    warmup_steps = int(total_steps * FIXED_WARMUP)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=FIXED_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=FIXED_LR,
        lr_scheduler_type="cosine",
        weight_decay=FIXED_WD,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,  # CRITICAL: keep teacher_logits!
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        logging_steps=100,
    )

    if obj_name == "ce":
        train_ds = BanglaHeadlineDataset(tr_titles, train_df["label"].tolist(), tokenizer, max_seq_len=max_len, bodies=tr_bodies)
        from src.training.train_teacher import WeightedTrainer
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_from_eval_pred,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), curve_cb],
        )
    else:
        assert oof_data is not None, "OOF teacher soft labels required for KD! Run Phase E teacher_oof first."
        saved_indices = oof_data["headline_indices"]
        idx_to_pos = {int(orig_i): pos for pos, orig_i in enumerate(saved_indices)}
        aligned_pos = [idx_to_pos[int(i)] for i in frac_indices]
        aligned_logits = oof_data["logits"][aligned_pos]

        train_ds = DistillationDataset(
            headlines=tr_titles,
            labels=train_df["label"].tolist(),
            teacher_logits=aligned_logits,
            tokenizer=tokenizer,
            max_seq_len=max_len,
            bodies=tr_bodies,
        )
        kd_loss_fn = KnowledgeDistillationLoss(temperature=FIXED_T, alpha=FIXED_ALPHA, class_weights=class_weights)

        align_mod = None
        t_layer_map = None
        beta_h = 0.0
        if obj_name == "kd_hid":
            assert teacher_model_for_hid is not None, "Need trained teacher_base/seed42 checkpoint for kd_hid!"
            t_layer_map = _select_teacher_layers(12, num_layers, layer_selection="strided")
            align_mod = IntermediateAlignmentModule(
                student_hidden_size=model.config.hidden_size,
                teacher_hidden_size=teacher_model_for_hid.config.hidden_size,
                teacher_layer_indices=t_layer_map,
                hidden_loss_weight=FIXED_BETA_HIDDEN,
                attention_loss_weight=0.0,
            )
            beta_h = FIXED_BETA_HIDDEN

        trainer = V3DistillTrainer(
            kd_loss_fn=kd_loss_fn,
            teacher_model=teacher_model_for_hid if obj_name == "kd_hid" else None,
            align_module=align_mod,
            teacher_layer_map=t_layer_map,
            beta_hidden=beta_h,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_from_eval_pred,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), curve_cb],
        )

    logger.info("=== Running %s | Seed %d | Params: %d ===", run_name, seed, n_params)
    trainer.train()

    # Predict on validation ONLY (Test Set Locked)
    val_logits = trainer.predict(val_ds).predictions
    metrics = compute_full_metrics(val_logits, val_labels)
    metrics["params"] = n_params
    metrics["init"] = init_name
    metrics["objective"] = obj_name
    metrics["frac"] = frac
    metrics["num_layers"] = num_layers
    metrics["seed"] = seed

    # Mechanism metrics (Phase G)
    if curve_cb.history:
        best_ep_entry = max(curve_cb.history, key=lambda x: x["eval_f1_macro"])
        metrics["epochs_to_best"] = best_ep_entry["epoch"]
    else:
        metrics["epochs_to_best"] = 0

    if teacher_val_preds is not None:
        student_val_preds = np.argmax(val_logits, axis=1)
        metrics["teacher_agreement"] = round(float(np.mean(student_val_preds == teacher_val_preds)), 4)

    # Save checkpoint only for 100% frac runs (to save disk space while enabling Phase J/K/L on final models)
    if frac == 100:
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

    np.save(run_dir / "val_logits.npy", val_logits)
    np.save(run_dir / "val_labels.npy", val_labels)
    (run_dir / "val_loss_curve.json").write_text(json.dumps(curve_cb.history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "git_sha.txt").write_text(git_sha, encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "text_mode": text_mode,
                "init": init_name,
                "objective": obj_name,
                "frac": frac,
                "num_layers": num_layers,
                "params": n_params,
                "lr": FIXED_LR,
                "T": FIXED_T if obj_name != "ce" else None,
                "alpha": FIXED_ALPHA if obj_name != "ce" else None,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    logger.info(
        "Done %s/seed%d | Val F1: %.4f | ECE: %.4f | EpBest: %s",
        run_name,
        seed,
        metrics["f1_macro"],
        metrics["ece"],
        metrics["epochs_to_best"],
    )


def main():
    parser = argparse.ArgumentParser(description="Phase F: Core Factorial Grid & Depth Study.")
    parser.add_argument("--dataset", choices=["banglabait", "baitbuster"], default="banglabait")
    parser.add_argument("--inits", nargs="+", default=["rand", "strided", "first4", "last4"])
    parser.add_argument("--objs", nargs="+", default=["ce", "kd", "kd_hid"])
    parser.add_argument("--fracs", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--text-mode", choices=["title", "title_body", "cat_title_body"], default="title")
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--depth-study", action="store_true", help="Run depth study (4L, 6L, 12L x {ce, kd} at 100%% data).")
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    ds = args.dataset
    data_dir, exp_dir, splits_path = get_dataset_paths(ds)
    tm = "title" if ds == "baitbuster" else args.text_mode
    suffix = "_ctb" if tm == "cat_title_body" else ("_tb" if tm == "title_body" else "")
    if args.unweighted:
        suffix += "_uw"

    csv_path = data_dir / "labeled_clean_v3_full.csv" if (tm in ("title_body", "cat_title_body") and (data_dir / "labeled_clean_v3_full.csv").exists()) else data_dir / "labeled_clean_v3.csv"
    oof_path = data_dir / f"teacher_oof_logits{suffix}.pt"

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    clean_df = pd.read_csv(csv_path)
    if "content" in clean_df.columns:
        clean_df["content"] = clean_df["content"].fillna("").astype(str)
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    git_sha = get_git_sha()
    teacher_val_preds = load_teacher_val_preds(text_mode=tm, dataset=ds, suffix=suffix)

    oof_data = torch.load(oof_path, weights_only=False) if oof_path.exists() else None

    teacher_model_for_hid = None
    if "kd_hid" in args.objs and not args.depth_study:
        t_ckpt = exp_dir / f"teacher_base{suffix}" / "seed42"
        if t_ckpt.exists():
            teacher_model_for_hid = AutoModelForSequenceClassification.from_pretrained(t_ckpt)
        else:
            logger.warning("%s not found; kd_hid requires Phase E teacher_base first.", t_ckpt)

    skip = not args.no_skip

    if args.depth_study:
        for depth in [4, 6, 12]:
            init_mode = "full" if depth == 12 else "strided"
            for obj in ["ce", "kd"]:
                run_name = f"D_{depth}L_{obj}_100{suffix}"
                for seed in args.seeds:
                    run_single_cell(
                        run_name=run_name,
                        init_name=init_mode,
                        obj_name=obj,
                        frac=100,
                        num_layers=depth,
                        seed=seed,
                        clean_df=clean_df,
                        splits=splits,
                        tokenizer=tokenizer,
                        oof_data=oof_data,
                        teacher_model_for_hid=None,
                        teacher_val_preds=teacher_val_preds,
                        git_sha=git_sha,
                        text_mode=tm,
                        dataset=ds,
                        unweighted=args.unweighted,
                        skip_existing=skip,
                    )
    else:
        for frac in args.fracs:
            for init_name in args.inits:
                for obj_name in args.objs:
                    run_name = f"F_{init_name}_{obj_name}_{frac}{suffix}"
                    for seed in args.seeds:
                        run_single_cell(
                            run_name=run_name,
                            init_name=init_name,
                            obj_name=obj_name,
                            frac=frac,
                            num_layers=4,
                            seed=seed,
                            clean_df=clean_df,
                            splits=splits,
                            tokenizer=tokenizer,
                            oof_data=oof_data,
                            teacher_model_for_hid=teacher_model_for_hid,
                            teacher_val_preds=teacher_val_preds,
                            git_sha=git_sha,
                            text_mode=tm,
                            dataset=ds,
                            unweighted=args.unweighted,
                            skip_existing=skip,
                        )


if __name__ == "__main__":
    main()
