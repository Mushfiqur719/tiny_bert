"""scripts/run_v3_sentnob.py — Phase H: Complete Generalisation Pipeline on SentNoB (3-Class Sentiment).

Executes the exact v3 controlled protocol on SentNoB (khondoker/SentNoB, 3 classes: Neutral/Positive/Negative):
  --stage phase_e    : 5-fold OOF BanglaBERT-Base teacher + 5-seed teacher_base + 5-seed Small-12L + TF-IDF
  --stage phase_f    : 2x2x4x5 controlled 7.42M 4L factorial grid (rand/strided x ce/kd x 10/25/50/100% x 5 seeds)
  --stage phase_g    : Centered full-interaction mixed-effects model + direct H1 Wald contrast
  --stage phase_j    : One-time locked test evaluation (requires TBB_ALLOW_TEST=1, 10k paired bootstrap + Holm)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from scripts.final_test import holm_correction, paired_bootstrap_test, predict_model_on_texts
from scripts.run_v3_grid import V3DistillTrainer, ValCurveCallback
from scripts.train_v3_phase_e import SEEDS, compute_ece, get_git_sha
from src.data.dedup import norm_text
from src.data.split import apply_splits
from src.models.distillation import DistillationDataset, KnowledgeDistillationLoss
from src.models.student import count_parameters, init_from_small_pretrained, init_small_random
from src.training.dataset import BanglaHeadlineDataset
from src.utils.seeding import seed_everything
from src.utils.test_guard import assert_test_access

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed/v3_sentnob")
EXP_DIR = Path("experiments/v3_sentnob")
NUM_LABELS = 3
MAX_LEN = 64


def compute_multiclass_metrics(logits: np.ndarray, labels: np.ndarray, split_name: str = "val") -> dict[str, Any]:
    preds = np.argmax(logits, axis=1)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist()
    acc = float(accuracy_score(labels, preds))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    return {
        "split": split_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "val_f1_macro": round(float(f1_m), 4),
        "ece": compute_ece(probs, labels),
        "confusion_matrix": cm,
        "true_class_counts": {str(c): int((labels == c).sum()) for c in (0, 1, 2)},
    }


def eval_pred_3class(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def run_phase_e(seeds: list[int]) -> None:
    splits = json.loads((DATA_DIR / "splits_v3.json").read_text(encoding="utf-8"))
    clean_df = pd.read_csv(DATA_DIR / "labeled_clean_v3.csv")
    train_indices = splits["labeled"]["train"]
    train_df = clean_df.loc[train_indices].reset_index(drop=True)
    val_df = clean_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    y_val = val_df["label"].to_numpy()
    git_sha = get_git_sha()

    # 1. TF-IDF + LogReg baseline (5 seeds)
    X_tr_raw = train_df["headline"].apply(norm_text).tolist()
    X_va_raw = val_df["headline"].apply(norm_text).tolist()
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_tr_raw)
    X_va = vec.transform(X_va_raw)
    for seed in seeds:
        rdir = EXP_DIR / "baseline_tfidf" / f"seed{seed}"
        if (rdir / "metrics.json").exists():
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        clf = LogisticRegression(C=2.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr, train_df["label"].to_numpy())
        probs = clf.predict_proba(X_va)
        m = compute_multiclass_metrics(probs, y_val)
        (rdir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    logger.info("SentNoB TF-IDF baseline done.")

    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

    # 2. 5-Fold OOF Teacher logits
    oof_pt = DATA_DIR / "teacher_oof_logits.pt"
    if not oof_pt.exists():
        oof_logits = np.zeros((len(train_df), NUM_LABELS), dtype=np.float32)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_p, va_p) in enumerate(sgkf.split(train_df, train_df["label"], train_df["group_id"])):
            logger.info("=== SentNoB OOF Teacher Fold %d / 5 ===", fold + 1)
            seed_everything(42 + fold)
            f_tr, f_va = train_df.iloc[tr_p], train_df.iloc[va_p]
            tr_ds = BanglaHeadlineDataset(f_tr["headline"].tolist(), f_tr["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)
            va_ds = BanglaHeadlineDataset(f_va["headline"].tolist(), f_va["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)
            model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=NUM_LABELS)
            ckpt_dir = EXP_DIR / "teacher_oof" / f"fold_{fold}"
            total_steps = max(1, (len(tr_ds) // 32 + 1) * 5)
            args = TrainingArguments(
                output_dir=str(ckpt_dir), num_train_epochs=5, per_device_train_batch_size=32,
                per_device_eval_batch_size=64, learning_rate=2e-5, lr_scheduler_type="cosine",
                weight_decay=0.01, warmup_steps=int(total_steps * 0.1), eval_strategy="epoch",
                save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1_macro",
                greater_is_better=True, save_total_limit=1, fp16=torch.cuda.is_available(),
                seed=42 + fold, report_to="none",
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                compute_metrics=eval_pred_3class, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
            trainer.train()
            oof_logits[va_p] = trainer.predict(va_ds).predictions
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        torch.save({"logits": torch.tensor(oof_logits, dtype=torch.float32), "headline_indices": train_indices}, oof_pt)
        m_oof = compute_multiclass_metrics(oof_logits, train_df["label"].to_numpy(), split_name="train_oof")
        (EXP_DIR / "teacher_oof").mkdir(parents=True, exist_ok=True)
        (EXP_DIR / "teacher_oof" / "metrics.json").write_text(json.dumps(m_oof, indent=2), encoding="utf-8")
        logger.info("Saved SentNoB OOF teacher logits | OOF Macro-F1: %.4f", m_oof["f1_macro"])

    # 3. 5-Seed Teacher Base & Small-12L
    tr_ds = BanglaHeadlineDataset(train_df["headline"].tolist(), train_df["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)
    va_ds = BanglaHeadlineDataset(val_df["headline"].tolist(), val_df["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)

    for run_name, model_id, lr, epochs in [
        ("teacher_base", "csebuetnlp/banglabert", 2e-5, 6),
        ("baseline_small_12l_ce", "csebuetnlp/banglabert_small", 5e-5, 8),
    ]:
        for seed in seeds:
            rdir = EXP_DIR / run_name / f"seed{seed}"
            if (rdir / "metrics.json").exists():
                continue
            rdir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed)
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS)
            ckpt_dir = rdir / "hf_checkpoints"
            total_steps = max(1, (len(tr_ds) // 32 + 1) * epochs)
            args = TrainingArguments(
                output_dir=str(ckpt_dir), num_train_epochs=epochs, per_device_train_batch_size=32,
                per_device_eval_batch_size=64, learning_rate=lr, lr_scheduler_type="cosine",
                weight_decay=0.01, warmup_steps=int(total_steps * 0.1), eval_strategy="epoch",
                save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1_macro",
                greater_is_better=True, save_total_limit=1, fp16=torch.cuda.is_available(),
                seed=seed, report_to="none",
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                compute_metrics=eval_pred_3class, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )
            trainer.train()
            val_logits = trainer.predict(va_ds).predictions
            m = compute_multiclass_metrics(val_logits, y_val)
            m["params"] = count_parameters(model)["total_params"]
            model.save_pretrained(rdir)
            tokenizer.save_pretrained(rdir)
            np.save(rdir / "val_logits.npy", val_logits)
            np.save(rdir / "val_labels.npy", y_val)
            (rdir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            (rdir / "git_sha.txt").write_text(git_sha, encoding="utf-8")
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info("SentNoB %s/seed%d | Val F1: %.4f | Acc: %.4f", run_name, seed, m["f1_macro"], m["accuracy"])


def run_phase_f(seeds: list[int], fracs: list[int], inits: list[str], objs: list[str]) -> None:
    splits = json.loads((DATA_DIR / "splits_v3.json").read_text(encoding="utf-8"))
    clean_df = pd.read_csv(DATA_DIR / "labeled_clean_v3.csv")
    val_df = clean_df.loc[splits["labeled"]["val"]].reset_index(drop=True)
    y_val = val_df["label"].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    va_ds = BanglaHeadlineDataset(val_df["headline"].tolist(), val_df["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)
    oof_data = torch.load(DATA_DIR / "teacher_oof_logits.pt", weights_only=False)
    idx_to_pos = {int(orig_i): pos for pos, orig_i in enumerate(oof_data["headline_indices"])}
    t_val_p = EXP_DIR / "teacher_base" / "seed42" / "val_logits.npy"
    t_val_preds = np.argmax(np.load(t_val_p), axis=1) if t_val_p.exists() else None
    git_sha = get_git_sha()

    for frac in fracs:
        frac_indices = json.loads((DATA_DIR / f"train_frac_{frac}.json").read_text(encoding="utf-8"))["indices"]
        tr_df = clean_df.loc[frac_indices].reset_index(drop=True)
        aligned_logits = oof_data["logits"][[idx_to_pos[int(i)] for i in frac_indices]]

        for init_name in inits:
            for obj_name in objs:
                run_name = f"F_{init_name}_{obj_name}_{frac}"
                for seed in seeds:
                    rdir = EXP_DIR / run_name / f"seed{seed}"
                    if (rdir / "metrics.json").exists():
                        continue
                    rdir.mkdir(parents=True, exist_ok=True)
                    seed_everything(seed)

                    if init_name == "rand":
                        model = init_small_random(num_hidden_layers=4, num_labels=NUM_LABELS)
                    else:
                        model = init_from_small_pretrained(num_hidden_layers=4, num_labels=NUM_LABELS, layer_selection=init_name)

                    ckpt_dir = rdir / "hf_checkpoints"
                    total_steps = max(1, (len(tr_df) // 32 + 1) * 10)
                    args = TrainingArguments(
                        output_dir=str(ckpt_dir), num_train_epochs=10, per_device_train_batch_size=32,
                        per_device_eval_batch_size=64, learning_rate=5e-5, lr_scheduler_type="cosine",
                        weight_decay=0.01, warmup_steps=int(total_steps * 0.1), eval_strategy="epoch",
                        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1_macro",
                        greater_is_better=True, save_total_limit=1, remove_unused_columns=False,
                        fp16=torch.cuda.is_available(), seed=seed, report_to="none",
                    )
                    curve_cb = ValCurveCallback()
                    if obj_name == "ce":
                        tr_ds = BanglaHeadlineDataset(tr_df["headline"].tolist(), tr_df["label"].tolist(), tokenizer, max_seq_len=MAX_LEN)
                        trainer = Trainer(
                            model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                            compute_metrics=eval_pred_3class, callbacks=[EarlyStoppingCallback(early_stopping_patience=3), curve_cb],
                        )
                    else:
                        tr_ds = DistillationDataset(tr_df["headline"].tolist(), tr_df["label"].tolist(), aligned_logits, tokenizer, max_seq_len=MAX_LEN)
                        kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.5, class_weights=None)
                        trainer = V3DistillTrainer(
                            kd_loss_fn=kd_loss, model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                            compute_metrics=eval_pred_3class, callbacks=[EarlyStoppingCallback(early_stopping_patience=3), curve_cb],
                        )

                    trainer.train()
                    val_logits = trainer.predict(va_ds).predictions
                    m = compute_multiclass_metrics(val_logits, y_val)
                    m.update({
                        "params": count_parameters(model)["total_params"],
                        "init": init_name,
                        "objective": obj_name,
                        "frac": frac,
                        "seed": seed,
                    })
                    if t_val_preds is not None:
                        m["teacher_agreement"] = round(float(np.mean(np.argmax(val_logits, axis=1) == t_val_preds)), 4)

                    if frac == 100:
                        model.save_pretrained(rdir)
                        tokenizer.save_pretrained(rdir)
                    np.save(rdir / "val_logits.npy", val_logits)
                    np.save(rdir / "val_labels.npy", y_val)
                    (rdir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
                    (rdir / "git_sha.txt").write_text(git_sha, encoding="utf-8")
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                    logger.info("SentNoB %s/seed%d | Val F1: %.4f | ECE: %.4f", run_name, seed, m["f1_macro"], m["ece"])


def run_phase_g() -> None:
    rows = []
    for run_dir in sorted(EXP_DIR.glob("F_*")):
        if not run_dir.is_dir():
            continue
        for mf in sorted(run_dir.glob("seed*/metrics.json")):
            d = json.loads(mf.read_text(encoding="utf-8"))
            rows.append({
                "run_name": run_dir.name, "init": d["init"], "objective": d["objective"],
                "frac": int(d["frac"]), "log_frac": float(np.log(int(d["frac"]))),
                "seed": int(d["seed"]), "f1_macro": float(d["f1_macro"]),
                "accuracy": float(d["accuracy"]), "ece": float(d["ece"]),
                "teacher_agreement": d.get("teacher_agreement", np.nan),
            })
    df = pd.DataFrame(rows)
    df.to_csv(EXP_DIR / "grid_master.csv", index=False)
    df["log_frac_c"] = df["log_frac"] - df["log_frac"].mean()
    df["init"] = pd.Categorical(df["init"], categories=["rand", "strided"])
    df["objective"] = pd.Categorical(df["objective"], categories=["ce", "kd"])
    mdf = smf.mixedlm("f1_macro ~ C(init) * C(objective) * log_frac_c", df, groups=df["seed"]).fit(reml=False)

    fe_names = list(mdf.fe_params.index)
    ik, ok = "C(init)[T.strided]", "C(objective)[T.kd]"
    r_full = np.zeros((1, len(mdf.params)))
    r_full[0, fe_names.index(ik)] = 1.0
    r_full[0, fe_names.index(ok)] = -1.0
    wald = mdf.wald_test(r_full, scalar=True)
    diff_val = float(mdf.fe_params[ik] - mdf.fe_params[ok])
    cov = mdf.cov_params().loc[[ik, ok], [ik, ok]].to_numpy()
    se_diff = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]))
    h1_line = f"Direct H1 Wald Contrast (beta_strided - beta_kd): {diff_val*100:+.2f} pp (SE={se_diff*100:.2f} pp, chi2={float(wald.statistic):.2f}, p={float(wald.pvalue):.4e})"
    summary = df.groupby(["init", "objective", "frac"]).agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"), ece_mean=("ece", "mean"), agree_mean=("teacher_agreement", "mean")).reset_index()
    report = f"# SentNoB Phase G — Mechanism Report\n\n{summary.to_markdown(index=False, floatfmt='.4f')}\n\n{h1_line}\n\n```\n{mdf.summary()}\n```\n"
    (EXP_DIR / "mechanism_report.md").write_text(report, encoding="utf-8")
    logger.info("SentNoB Phase G complete: %s", h1_line)


def run_phase_j(seeds: list[int]) -> None:
    assert_test_access()
    splits = json.loads((DATA_DIR / "splits_v3.json").read_text(encoding="utf-8"))
    clean_df = pd.read_csv(DATA_DIR / "labeled_clean_v3.csv")
    test_df = apply_splits(clean_df, splits, split="test")
    texts = test_df["headline"].astype(str).tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    models = ["teacher_base", "baseline_small_12l_ce", "F_strided_kd_100", "F_strided_ce_100", "F_rand_kd_100", "F_rand_ce_100"]
    all_preds = {}
    s_rows = []
    for m_name in models:
        sp, sf, sa = [], [], []
        for s in seeds:
            rdir = EXP_DIR / m_name / f"seed{s}"
            logits = predict_model_on_texts(rdir, texts, max_length=MAX_LEN)
            np.save(rdir / "test_logits.npy", logits)
            m = compute_multiclass_metrics(logits, y_test, split_name="test")
            (rdir / "test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            sp.append(np.argmax(logits, axis=1))
            sf.append(m["f1_macro"])
            sa.append(m["accuracy"])
        all_preds[m_name] = np.stack(sp, axis=0)
        s_rows.append({"model": m_name, "n_seeds": len(sf), "test_acc_mean": round(float(np.mean(sa)), 4), "test_acc_std": round(float(np.std(sa)), 4), "test_f1_mean": round(float(np.mean(sf)), 4), "test_f1_std": round(float(np.std(sf)), 4)})

    pairs = [
        ("F_strided_kd_100", "F_rand_kd_100"),
        ("F_strided_ce_100", "F_rand_ce_100"),
        ("F_strided_kd_100", "F_strided_ce_100"),
        ("F_rand_kd_100", "F_rand_ce_100"),
        ("F_strided_kd_100", "baseline_small_12l_ce"),
        ("F_strided_kd_100", "teacher_base"),
    ]
    c_rows = []
    for a, b in pairs:
        res = paired_bootstrap_test(all_preds[a], all_preds[b], y_test, n_resamples=10000)
        c_rows.append({"model_a": a, "model_b": b, **res})
    p_adj = holm_correction([r["p_value"] for r in c_rows])
    for r, pa in zip(c_rows, p_adj):
        r["p_value"], r["p_holm"] = round(r["p_value"], 5), round(pa, 5)

    sdf, cdf = pd.DataFrame(s_rows), pd.DataFrame(c_rows)
    sdf.to_csv(EXP_DIR / "final_test_results.csv", index=False)
    cdf.to_csv(EXP_DIR / "direct_test_contrasts.csv", index=False)
    report = f"# SentNoB Phase J — Locked Test Report ($n={len(y_test)}$)\n\n{sdf.to_markdown(index=False)}\n\n## Direct Paired Bootstrap Contrasts\n\n{cdf.to_markdown(index=False)}\n"
    (EXP_DIR / "final_test_report.md").write_text(report, encoding="utf-8")
    logger.info("SentNoB Phase J complete!")


def main():
    parser = argparse.ArgumentParser(description="Phase H: SentNoB 3-Class Sentiment Generalisation Pipeline.")
    parser.add_argument("--stage", choices=["phase_e", "phase_f", "phase_g", "phase_j", "all_val"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if args.stage == "phase_e":
        run_phase_e(args.seeds)
    elif args.stage == "phase_f":
        run_phase_f(args.seeds, [10, 25, 50, 100], ["rand", "strided"], ["ce", "kd"])
    elif args.stage == "phase_g":
        run_phase_g()
    elif args.stage == "phase_j":
        run_phase_j(args.seeds)
    elif args.stage == "all_val":
        run_phase_e(args.seeds)
        run_phase_f(args.seeds, [10, 25, 50, 100], ["rand", "strided"], ["ce", "kd"])
        run_phase_g()


if __name__ == "__main__":
    main()
