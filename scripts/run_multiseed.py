"""
scripts/run_multiseed.py — Run one student config across several seeds.

Each seed trains into its own run directory  experiments/<run_name>_s<seed>/ .
It prints (does not hide) failures and continues with the next seed.

Usage:
    python -m scripts.run_multiseed --config configs/student_small_pretrained_distilled.yaml
    python -m scripts.run_multiseed --config configs/student_small_pretrained_distilled.yaml \
        --seeds 13 42 87 --temperature 4.0 --alpha 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import main as train_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [13, 42, 87, 123, 2024]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--init", default=None, choices=["random", "teacher_init", "small_pretrained"])
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--tag", default="", help="Suffix added to run_name, e.g. 'T4a01'.")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    from src.utils import load_config

    base_run_name = str(load_config(args.config, overrides=args.override or []).training.run_name)
    if args.tag:
        base_run_name = f"{base_run_name}_{args.tag}"

    failed = []
    for seed in args.seeds:
        run_name = f"{base_run_name}_s{seed}"
        logger.info("=" * 70)
        logger.info("MULTISEED: %s (seed=%d)", run_name, seed)
        logger.info("=" * 70)
        cli = ["--config", args.config, "--seed", str(seed)]
        if args.init:
            cli += ["--init", args.init]
        if args.temperature is not None:
            cli += ["--temperature", str(args.temperature)]
        if args.alpha is not None:
            cli += ["--alpha", str(args.alpha)]
        cli += ["--override", f"training.run_name={run_name}", *(args.override or [])]
        try:
            train_main(cli)
        except Exception:
            failed.append(seed)
            logger.error("Seed %d FAILED:\n%s", seed, traceback.format_exc())

    if failed:
        logger.error("Failed seeds: %s", failed)
    else:
        logger.info("All seeds completed for '%s'.", base_run_name)


if __name__ == "__main__":
    main()
