"""
scripts/prepare_data.py — CLI entry point for the full data pipeline.

Runs in order:
  1. Load raw data from HuggingFace Hub (or local path)
  2. Clean headlines
  3. Create fixed stratified splits (skips if splits.json already exists)
  4. Generate data_report.md

Usage:
    python -m scripts.prepare_data --config configs/base.yaml --seed 42
    python -m scripts.prepare_data --config configs/base.yaml --overwrite-splits

Output:
    data/processed/splits.json
    data/processed/data_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.utils import seed_everything, load_config
from src.data.load import load_raw
from src.data.clean import clean_dataframe
from src.data.split import create_splits, load_splits
from src.data.report import generate_report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the BanglaBait data pipeline.")
    p.add_argument("--config", default="configs/base.yaml", help="YAML config path.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    p.add_argument(
        "--overwrite-splits",
        action="store_true",
        help="Regenerate splits.json even if it already exists.",
    )
    p.add_argument(
        "--local-path",
        default=None,
        help="Load from a local CSV/parquet instead of HF Hub.",
    )
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.override or [])
    seed = int(args.seed if args.seed is not None else cfg.seed)
    seed_everything(seed)

    logger.info("=" * 60)
    logger.info("BanglaBait Data Pipeline")
    logger.info("Config: %s | Seed: %d", args.config, seed)
    logger.info("=" * 60)

    # 1. Load
    hf_id = getattr(cfg.data, "hf_dataset_id", None)
    labeled_df, unlabeled_df = load_raw(
        cfg,
        hf_dataset_id=hf_id,
        local_path=args.local_path,
    )

    # 2. Clean
    logger.info("Cleaning labeled headlines...")
    labeled_df = clean_dataframe(labeled_df)
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore")
    labeled_df = labeled_df.reset_index(drop=True)

    logger.info("Cleaning unlabeled headlines...")
    unlabeled_df = clean_dataframe(unlabeled_df)
    unlabeled_df = unlabeled_df.drop(columns=["_was_cleaned"], errors="ignore")
    unlabeled_df = unlabeled_df.reset_index(drop=True)

    # 3. Splits
    splits_path = Path(cfg.data.splits_file)
    if splits_path.exists() and not args.overwrite_splits:
        logger.info(
            "splits.json already exists at %s — loading existing splits. "
            "Pass --overwrite-splits to regenerate.",
            splits_path,
        )
        splits = load_splits(cfg)
    else:
        logger.info("Creating fixed stratified splits (seed=%d)...", seed)
        splits = create_splits(
            labeled_df, unlabeled_df, cfg, seed=seed, overwrite=args.overwrite_splits
        )

    # 4. Report
    logger.info("Generating data report...")
    report_md = generate_report(labeled_df, splits, cfg, seed=seed)
    print("\n" + "=" * 60)
    # Write to stdout with UTF-8 encoding (Windows console may be cp1252)
    import sys
    sys.stdout.buffer.write(report_md.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    print("=" * 60)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
