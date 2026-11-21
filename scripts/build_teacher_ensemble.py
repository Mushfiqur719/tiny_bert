"""
scripts/build_teacher_ensemble.py — Stage 2: Build Calibrated Multi-Teacher Ensemble & Soft Labels.

Combines all available Teacher runs:
  - `experiments/teacher` (BanglaBERT-Base)
  - `experiments/teacher_rdrop` (BanglaBERT-Base + R-Drop + LLRD)
  - `experiments/teacher_large` (BanglaBERT-Large 335M + R-Drop + LLRD)
plus the 5-fold Out-of-Fold soft labels (`experiments/teacher/train_oof_soft_labels.pt`).

Steps:
  1. Fits post-hoc temperature scaling T_cal on `val_logits.npy` for each available Teacher.
  2. Searches over convex combination weights on `val_df` to maximize Validation Macro-F1.
  3. Evaluates the Calibrated Teacher Ensemble on `test_df` (saving `experiments/teacher_ensemble/metrics.json`,
     `val_logits.npy`, and `test_logits.npy`).
  4. Combines the teachers' training logits (plus OOF logits) into calibrated ensemble training logits
     and writes `experiments/teacher/train_ensemble_soft_labels.pt` and updates `experiments/teacher/train_soft_labels.pt`.

Usage:
    .venv\\Scripts\\python.exe -m scripts.build_teacher_ensemble
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.evaluation.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    predict_with_threshold,
)
from src.evaluation.selective import ensure_cached_logits, fit_temperature_on_val

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_TEACHERS = [
    ("teacher", Path("experiments/teacher")),
    ("teacher_rdrop", Path("experiments/teacher_rdrop")),
    ("teacher_large", Path("experiments/teacher_large")),
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build calibrated teacher ensemble & soft labels.")
    p.add_argument("--oof-weight", type=float, default=0.25, help="Weight of 5-fold OOF logits in train soft labels.")
    args = p.parse_args(argv)

    available = []
    for name, d in CANDIDATE_TEACHERS:
        if (d / "config.json").exists():
            infold_pt = d / "train_soft_labels_infold.pt"
            soft_pt = infold_pt if infold_pt.exists() else (d / "train_soft_labels.pt")
            if soft_pt.exists():
                ensure_cached_logits(d)
                available.append((name, d, soft_pt))

    if not available:
        raise FileNotFoundError("No teacher directories with val_logits.npy and train_soft_labels.pt found!")

    logger.info("Found %d available teachers: %s", len(available), [a[0] for a in available])

    val_labels = np.load(available[0][1] / "val_labels.npy")
    test_labels = np.load(available[0][1] / "test_labels.npy")

    val_scaled_list = []
    test_scaled_list = []
    train_scaled_list = []
    teacher_info = []
    first_soft_data = torch.load(available[0][2], map_location="cpu")

    for name, d, soft_pt in available:
        v_log = np.load(d / "val_logits.npy")
        t_log = np.load(d / "test_logits.npy")
        t_cal = fit_temperature_on_val(v_log, val_labels)

        v_m = compute_classification_metrics(np.argmax(v_log, axis=1), val_labels)
        t_m = compute_classification_metrics(np.argmax(t_log, axis=1), test_labels)
        logger.info(
            "Teacher '%-14s' | T_cal=%.3f | Val F1: %.4f (Acc %.4f) | Test F1: %.4f (Acc %.4f)",
            name,
            t_cal,
            v_m["f1_macro"],
            v_m["accuracy"],
            t_m["f1_macro"],
            t_m["accuracy"],
        )

        # Standardize logit scale by dividing by T_cal
        val_scaled_list.append(v_log / t_cal)
        test_scaled_list.append(t_log / t_cal)

        s_data = torch.load(soft_pt, map_location="cpu")
        tr_log = s_data["logits"].float().numpy() / t_cal
        train_scaled_list.append(tr_log)

        teacher_info.append(
            {
                "name": name,
                "temperature_cal": t_cal,
                "val_f1_macro": v_m["f1_macro"],
                "test_f1_macro": t_m["f1_macro"],
                "test_accuracy": t_m["accuracy"],
            }
        )

    # Search optimal convex weights on Validation split
    k = len(available)
    if k == 1:
        best_weights = [1.0]
        best_val_f1 = teacher_info[0]["val_f1_macro"]
    else:
        grid = np.linspace(0.0, 1.0, 11)
        best_weights = [1.0 / k] * k
        best_val_f1 = -1.0
        for combo in itertools.product(grid, repeat=k):
            if sum(combo) <= 1e-6:
                continue
            w = [c / sum(combo) for c in combo]
            ens_val = sum(wi * vl for wi, vl in zip(w, val_scaled_list))
            preds = np.argmax(ens_val, axis=1)
            f1 = compute_classification_metrics(preds, val_labels)["f1_macro"]
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_weights = [round(float(x), 4) for x in w]

    logger.info("Optimal Validation Ensemble Weights: %s -> Val Macro-F1: %.4f", dict(zip([a[0] for a in available], best_weights)), best_val_f1)

    ens_val_logits = sum(wi * vl for wi, vl in zip(best_weights, val_scaled_list))
    ens_test_logits = sum(wi * tl for wi, tl in zip(best_weights, test_scaled_list))
    ens_train_logits = sum(wi * trl for wi, trl in zip(best_weights, train_scaled_list))

    # Blend with 5-fold OOF soft logits if present
    oof_path = Path("experiments/teacher/train_oof_soft_labels.pt")
    if oof_path.exists() and args.oof_weight > 0.0:
        oof_data = torch.load(oof_path, map_location="cpu")
        oof_raw = oof_data.get("oof_raw_logits", oof_data["logits"]).float().numpy()
        if len(oof_raw) == len(ens_train_logits):
            w_oof = float(args.oof_weight)
            ens_train_logits = (1.0 - w_oof) * ens_train_logits + w_oof * oof_raw
            logger.info("Blended %.0f%% Ensemble Teacher logits + %.0f%% 5-Fold OOF logits.", (1 - w_oof) * 100, w_oof * 100)

    opt_thresh, val_cal_f1 = find_optimal_threshold(ens_val_logits, val_labels)
    test_m = compute_classification_metrics(np.argmax(ens_test_logits, axis=1), test_labels)
    test_m_cal = compute_classification_metrics(predict_with_threshold(ens_test_logits, opt_thresh), test_labels)
    test_m["optimal_threshold"] = opt_thresh
    test_m["calibrated"] = test_m_cal
    test_m["ensemble_weights"] = dict(zip([a[0] for a in available], best_weights))
    test_m["member_teachers"] = teacher_info

    ens_dir = Path("experiments/teacher_ensemble")
    ens_dir.mkdir(parents=True, exist_ok=True)
    np.save(ens_dir / "val_logits.npy", ens_val_logits)
    np.save(ens_dir / "val_labels.npy", val_labels)
    np.save(ens_dir / "test_logits.npy", ens_test_logits)
    np.save(ens_dir / "test_labels.npy", test_labels)
    (ens_dir / "metrics.json").write_text(json.dumps(test_m, indent=2), encoding="utf-8")

    # Save ensemble soft labels
    payload = {
        "logits": torch.tensor(ens_train_logits, dtype=torch.float32),
        "labels": first_soft_data["labels"],
        "headline_indices": first_soft_data["headline_indices"],
        "teacher_model_name": "TeacherEnsemble(" + ",".join(a[0] for a in available) + ")",
        "ensemble_weights": test_m["ensemble_weights"],
        "seed": 42,
    }
    ens_soft_path = Path("experiments/teacher/train_ensemble_soft_labels.pt")
    torch.save(payload, ens_soft_path)
    torch.save(payload, Path("experiments/teacher/train_soft_labels.pt"))

    logger.info("=" * 65)
    logger.info("TEACHER ENSEMBLE FINAL TEST METRICS -> Acc: %.4f | Macro-F1: %.4f (Cal Acc: %.4f | Cal F1: %.4f)", test_m["accuracy"], test_m["f1_macro"], test_m_cal["accuracy"], test_m_cal["f1_macro"])
    logger.info("Saved ensemble soft labels to %s and experiments/teacher/train_soft_labels.pt", ens_soft_path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
