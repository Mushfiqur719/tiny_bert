"""
scripts/tune_kd.py — CLI entry point for Distillation (T, alpha) Hyperparameter Grid Search.

Usage:
    python -m scripts.tune_kd --config configs/student_teacher_init_distilled.yaml
    python -m scripts.tune_kd --temperatures 1.0 2.0 4.0 8.0 --alphas 0.1 0.3 0.5 0.7 0.9 --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.tune_distill import run_grid_search_kd
from src.utils import load_config, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Knowledge Distillation (T, alpha) Grid Search.")
    p.add_argument("--config", default="configs/student_teacher_init_distilled.yaml", help="Path to base YAML config.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        help="List of temperature values T to evaluate.",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of alpha values to evaluate.",
    )
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs per trial.")
    p.add_argument("--init", default="teacher_init", choices=["random", "teacher_init"], help="Student initialization type.")
    p.add_argument("--output-dir", default="experiments/tuning", help="Output directory for reports.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(args.seed)

    best_combo, results, report_md = run_grid_search_kd(
        cfg=cfg,
        temperatures=args.temperatures,
        alphas=args.alphas,
        init_type=args.init,
        epochs_per_trial=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print(report_md)
    print("=" * 60)
    logger.info("Grid search successfully completed.")


if __name__ == "__main__":
    main()
