"""
scripts/benchmark.py — CLI entry point for efficiency, latency, and ONNX benchmarking.

Usage:
    python -m scripts.benchmark --n-iterations 200
    python -m scripts.benchmark --teacher-dir experiments/teacher --student-dir experiments/tinybanglabert_final
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.efficiency.benchmark import run_full_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TinyBanglaBERT efficiency and latency.")
    p.add_argument("--teacher-dir", default="experiments/teacher", help="Teacher checkpoint directory.")
    p.add_argument("--student-dir", default="experiments/tinybanglabert_final", help="Student checkpoint directory.")
    p.add_argument("--onnx-dir", default="experiments/onnx", help="Directory for exported ONNX models.")
    p.add_argument("--n-iterations", type=int, default=200, help="Number of benchmark iterations (default: 200).")
    p.add_argument("--output-dir", default="experiments", help="Output directory for reports.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    df, report_md = run_full_benchmark(
        teacher_dir=args.teacher_dir,
        student_dir=args.student_dir,
        onnx_dir=args.onnx_dir,
        n_iterations=args.n_iterations,
        output_dir=args.output_dir,
    )

    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.write(report_md.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
