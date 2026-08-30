"""
scripts/train.py — CLI entry point for training (teacher fine-tuning or student distillation).

Usage:
    python -m scripts.train --config configs/teacher_baseline.yaml --seed 42
    python -m scripts.train --config configs/student_distillation.yaml --override training.lr=3e-5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_teacher import train_teacher
from src.utils import load_config, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train teacher or student model.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, e.g. training.lr=2e-5",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.override or [])

    # CLI seed overrides config seed
    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)

    config_path_str = str(args.config).lower()
    run_name = str(cfg.training.run_name).lower()

    # Route based on config type
    if "student" in config_path_str or "distill" in run_name:
        raise NotImplementedError(
            "Student distillation training logic is scheduled for Phase 4."
        )
    else:
        # Default to Teacher training
        logger.info("Executing Teacher model fine-tuning with config '%s'...", args.config)
        metrics = train_teacher(cfg=cfg, seed=seed)
        logger.info("Teacher training run completed successfully.")
        return metrics


if __name__ == "__main__":
    main()
