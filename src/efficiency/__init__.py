# src/efficiency/__init__.py
"""Efficiency sub-package: latency benchmarking, peak memory tracking, and ONNX export."""

from src.efficiency.benchmark import (
    get_checkpoint_disk_size_mb,
    measure_onnx_latency_and_memory,
    measure_pytorch_latency_and_memory,
    run_full_benchmark,
)
from src.efficiency.export_onnx import (
    export_to_onnx,
    quantize_onnx_dynamic,
    verify_onnx_parity,
)

__all__ = [
    "export_to_onnx",
    "quantize_onnx_dynamic",
    "verify_onnx_parity",
    "measure_pytorch_latency_and_memory",
    "measure_onnx_latency_and_memory",
    "get_checkpoint_disk_size_mb",
    "run_full_benchmark",
]
