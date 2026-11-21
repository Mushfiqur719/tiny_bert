"""scripts/benchmark_v3_efficiency.py — Phase L: Unified ONNX FP32/INT8 & Cascade RTT Benchmark.

Benchmarks every model (Teacher Base 110.6M, Small-12L 13.7M, 4L Student 7.42M) in:
  - ONNX FP32 and Dynamic INT8
  - Thread counts: 1 and 4
  - Protocol: batch=1, max_length=64, 200 warm-up + 1,000 timed runs (p50 and p95 in ms)
  - Cascade latency simulation with 50 ms and 150 ms network RTT for escalated items
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx_and_int8(ckpt_dir: Path, out_dir: Path, max_length: int = 64) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / "model_int8.onnx"

    # Re-export if missing or if a tiny dynamo external-data stub was left behind
    if not fp32_path.exists() or fp32_path.stat().st_size < 5_000_000:
        for f in out_dir.glob("model_fp32.onnx*"):
            f.unlink(missing_ok=True)
        int8_path.unlink(missing_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).cpu().eval()
        dummy = tokenizer("বাংলাদেশ ক্রিকেট দল", padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )
        logger.info("Exported FP32 ONNX: %s (%.2f MB)", fp32_path, fp32_path.stat().st_size / 1e6)

    if not int8_path.exists():
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        logger.info("Exported INT8 ONNX: %s (%.2f MB)", int8_path, int8_path.stat().st_size / 1e6)

    return fp32_path, int8_path


def bench_onnx(onnx_path: Path, n_threads: int, max_length: int = 64, warmup: int = 200, runs: int = 1000) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": np.ones((1, max_length), dtype=np.int64),
        "attention_mask": np.ones((1, max_length), dtype=np.int64),
    }

    for _ in range(warmup):
        sess.run(None, feed)

    latencies_ms = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "disk_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase L: Unified ONNX Efficiency & Cascade RTT Benchmark.")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    exp_v3 = Path("experiments/v3")
    onnx_root = exp_v3 / "onnx_exports"

    # Candidate checkpoints (preferring v3 seed42 checkpoints)
    candidates = [
        ("BanglaBERT-Base (110.6M)", Path("experiments/v3/teacher_base/seed42")),
        ("BanglaBERT-Small-12L (13.7M)", Path("experiments/v3/baseline_small_12l_ce/seed42")),
        ("TinyBanglaBERT-4L (7.42M)", Path("experiments/v3/F_strided_kd_100/seed42")),
    ]

    rows = []
    cpu_name = platform.processor() or platform.machine()
    os_name = f"{platform.system()} {platform.release()}"
    ort_ver = ort.__version__

    for label, ckpt in candidates:
        if not (ckpt / "config.json").exists():
            logger.warning("Skipping %s (checkpoint not found at %s)", label, ckpt)
            continue
        safe_name = label.split("(")[0].strip().lower().replace(" ", "_").replace("-", "_")
        fp32_p, int8_p = export_to_onnx_and_int8(ckpt, onnx_root / safe_name)

        for prec, path in [("FP32", fp32_p), ("INT8", int8_p)]:
            for threads in (1, 4):
                logger.info("Benchmarking %s | %s | threads=%d ...", label, prec, threads)
                res = bench_onnx(path, n_threads=threads, warmup=args.warmup, runs=args.runs)
                rows.append({
                    "model": label,
                    "precision": prec,
                    "threads": threads,
                    **res,
                    "cpu": cpu_name,
                    "os": os_name,
                    "ort_version": ort_ver,
                })

    df = pd.DataFrame(rows)
    df.to_csv(exp_v3 / "efficiency.csv", index=False)

    # Cascade simulated RTT latency table (assuming 4L INT8 4-thread edge + Base FP32 4-thread cloud)
    cascade_rows = []
    s_row = df[(df["model"].str.contains("4L")) & (df["precision"] == "INT8") & (df["threads"] == 4)]
    t_row = df[(df["model"].str.contains("Base")) & (df["precision"] == "FP32") & (df["threads"] == 4)]
    if not s_row.empty and not t_row.empty:
        s_lat = float(s_row.iloc[0]["p50_ms"])
        t_lat = float(t_row.iloc[0]["p50_ms"])
        for esc_rate in (0.0, 0.10, 0.25, 0.50, 1.0):
            for rtt_ms in (0.0, 50.0, 150.0):
                # At 100% escalation (cloud-only baseline), edge inference is bypassed
                avg_lat = (t_lat + rtt_ms) if esc_rate == 1.0 else (s_lat + esc_rate * (t_lat + rtt_ms))
                cascade_rows.append({
                    "escalation_rate": "100% (Cloud-Only)" if esc_rate == 1.0 else f"{esc_rate:.0%}",
                    "network_rtt_ms": rtt_ms,
                    "edge_p50_ms": 0.0 if esc_rate == 1.0 else s_lat,
                    "cloud_p50_ms": t_lat,
                    "effective_avg_ms": round(avg_lat, 2),
                })

    cdf = pd.DataFrame(cascade_rows)
    cdf.to_csv(exp_v3 / "cascade_rtt_table.csv", index=False)

    report = f"""# Phase L — Unified ONNX Efficiency & Cascade RTT Report

**Hardware:** `{cpu_name}` | **OS:** `{os_name}` | **ONNX Runtime:** `{ort_ver}`  
**Protocol:** Batch = 1, `max_length = 64`, {args.warmup} warm-up + {args.runs} timed runs.

## 1. Model × Precision × Thread Benchmark

{df.to_markdown(index=False) if not df.empty else "_No models benchmarked._"}

## 2. Edge–Cloud Cascade Effective Latency (with 50 ms & 150 ms Network RTT)

{cdf.to_markdown(index=False) if not cdf.empty else "_Cascade table unavailable._"}
"""
    (exp_v3 / "efficiency_v3_report.md").write_text(report, encoding="utf-8")
    logger.info("Saved efficiency report to experiments/v3/efficiency_v3_report.md")


if __name__ == "__main__":
    main()
