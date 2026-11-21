"""
scripts/evaluate.py — Unified CLI entry point for Ablation Aggregation, McNemar Statistical
Significance Testing (with Holm-Bonferroni correction), and Cross-Domain Evaluation.

Usage:
    # Run ablation table + select best student -> experiments/tinybanglabert_final + McNemar tests:
    .venv\\Scripts\\python.exe -m scripts.evaluate

    # Also run zero-shot cross-domain evaluation on BaitBuster-Bangla (10,000 headlines):
    .venv\\Scripts\\python.exe -m scripts.evaluate --crossdomain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ablation import run_ablation_analysis
from src.evaluation.crossdomain_eval import run_crossdomain_evaluation
from src.evaluation.mcnemar_test import run_full_mcnemar_suite
from src.utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Ablation, McNemar Significance, and Cross-Domain Evaluation.")
    p.add_argument("--output-dir", default="experiments", help="Base experiments directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--crossdomain", action="store_true", help="Also run BaitBuster-Bangla cross-domain evaluation.")
    p.add_argument(
        "--crossdomain-dataset",
        default="abdalimran/BaitBuster-Bangla",
        help="HuggingFace dataset ID or local path for cross-domain evaluation.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    logger.info("Step 1: Running Ablation Aggregation & Selecting Best 4L Student...")
    ablation_df, best_id = run_ablation_analysis(output_dir=args.output_dir)
    print("\n" + "=" * 70)
    print(f"BEST 4-LAYER STUDENT SELECTED: {best_id}")
    print(ablation_df.drop(columns=["_run_id", "_run_dir"], errors="ignore").to_markdown(index=False))
    print("=" * 70 + "\n")

    logger.info("Step 2: Running Pairwise Exact Binomial McNemar Tests (with Holm correction)...")
    mcnemar_df, _ = run_full_mcnemar_suite(output_dir=args.output_dir, seed=args.seed)
    print("\n" + "=" * 70)
    print(mcnemar_df.to_markdown(index=False))
    print("=" * 70 + "\n")

    if args.crossdomain:
        logger.info("Step 3: Running Zero-Shot Cross-Domain Generalization Evaluation...")
        cd_df, _ = run_crossdomain_evaluation(
            dataset_path_or_id=args.crossdomain_dataset,
            teacher_dir=Path(args.output_dir) / "teacher",
            student_dir=Path(args.output_dir) / "tinybanglabert_final",
            output_dir=args.output_dir,
            n_explain_samples=5,
        )
        print("\n" + "=" * 70)
        print(cd_df.to_markdown(index=False))
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
