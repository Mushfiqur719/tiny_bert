"""
scripts/benchmark.py — CLI entry point for efficiency benchmarking (latency, size, ONNX).
Phase 7 will wire in src/efficiency/.
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import seed_everything, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark model efficiency (latency/size/ONNX).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override or [])
    seed = args.seed if args.seed is not None else int(cfg.seed)
    seed_everything(seed)
    raise NotImplementedError("Efficiency benchmarking logic coming in Phase 7.")

if __name__ == "__main__":
    main()
