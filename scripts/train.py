"""
scripts/train.py — CLI entry point for training (teacher or student).

Usage:
    python -m scripts.train --config configs/teacher_baseline.yaml --seed 42
    python -m scripts.train --config configs/student_distillation.yaml --override training.lr=3e-5

This is a *thin* wrapper — all logic lives in src/.
Phase 4 will wire in the actual Trainer.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Ensure project root is on path when invoked as a script.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import seed_everything, load_config, ExperimentLogger

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

    # CLI seed overrides config seed.
    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)

    exp = ExperimentLogger(run_name=cfg.training.run_name, cfg=cfg, seed=seed)
    logger.info("Training run '%s' initialised (seed=%d).", cfg.training.run_name, seed)
    logger.info("Config: %s", args.config)

    # ── Phase 4: wire in src.training.trainer here ──────────────────────
    raise NotImplementedError(
        "Training logic not yet implemented — coming in Phase 4. "
        "Run this script after the training module is added."
    )


if __name__ == "__main__":
    main()
