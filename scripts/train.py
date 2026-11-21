"""
scripts/train.py — Unified CLI entry point for training Teacher and Student models.

Supports:
  1. Teacher BanglaBERT fine-tuning:
     python -m scripts.train --config configs/teacher_baseline.yaml --seed 42

  2. Student 1 (Random Init + Supervised Baseline):
     python -m scripts.train --config configs/student_random_baseline.yaml --seed 42

  3. Student 2 (Random Init + Knowledge Distillation):
     python -m scripts.train --config configs/student_random_distilled.yaml --seed 42

  4. Student 3 (Teacher Init + Supervised Baseline):
     python -m scripts.train --config configs/student_teacher_init_baseline.yaml --seed 42

  5. Student 4 (Teacher Init + Knowledge Distillation):
     python -m scripts.train --config configs/student_teacher_init_distilled.yaml --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_baseline import train_student_baseline
from src.training.train_distill import train_student_distill
from src.training.train_teacher import train_teacher
from src.utils import load_config, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Teacher or Student model.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument(
        "--init",
        type=str,
        default=None,
        choices=["random", "teacher_init", "small_pretrained"],
        help="Override student initialization strategy (random, teacher_init, or small_pretrained).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override distillation temperature T.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override distillation alpha weight.",
    )
    p.add_argument(
        "--soft-labels",
        type=str,
        default="experiments/teacher/train_soft_labels.pt",
        help="Path to precomputed teacher soft labels (.pt).",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, e.g. training.lr=3e-5",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.override or [])

    # CLI seed overrides config seed
    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)

    config_str = str(args.config).lower()
    run_name = str(cfg.training.get("run_name", "")).lower()
    model_name = str(cfg.model.get("student_model_name", "")).lower()

    # Determine model type and training mode
    is_distillation = (
        "distill" in config_str
        or "distill" in run_name
        or "distillation" in cfg
        and cfg.distillation.get("alpha") is not None
        and "baseline" not in config_str
    )

    is_student = (
        "student" in config_str
        or "student" in run_name
        or "tiny" in model_name
        or "student_random" in run_name
        or "student_teacher_init" in run_name
    )

    init_type = args.init or cfg.model.get("init_type", "random")

    if not is_student and ("teacher" in config_str or "teacher" in run_name):
        logger.info("Executing Teacher model fine-tuning (BanglaBERT)...")
        metrics = train_teacher(cfg=cfg, seed=seed)
    elif is_distillation:
        logger.info("Executing Student Knowledge Distillation training (%s Init)...", init_type)
        metrics = train_student_distill(
            cfg=cfg,
            init_type=init_type,
            temperature=args.temperature,
            alpha=args.alpha,
            seed=seed,
            soft_labels_path=args.soft_labels,
        )
    else:
        logger.info("Executing Student Supervised Baseline training (%s Init)...", init_type)
        metrics = train_student_baseline(
            cfg=cfg,
            init_type=init_type,
            seed=seed,
        )

    logger.info("Training script execution finished successfully.")
    return metrics


if __name__ == "__main__":
    main()
