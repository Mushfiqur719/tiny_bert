"""
tests/test_efficiency.py — Unit tests for efficiency benchmarking and ONNX export.

Run with:  pytest tests/test_efficiency.py -v
"""

from __future__ import annotations

from pathlib import Path
import torch
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.efficiency.export_onnx import (
    export_to_onnx,
    verify_onnx_parity,
    quantize_onnx_dynamic,
)
from src.efficiency.benchmark import (
    measure_pytorch_latency_and_memory,
    measure_onnx_latency_and_memory,
    get_checkpoint_disk_size_mb,
)


@pytest.fixture(scope="module")
def student_model_and_tokenizer():
    path = "experiments/tinybanglabert_final"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer


class TestEfficiency:

    def test_export_onnx_and_verify_parity(self, student_model_and_tokenizer, tmp_path):
        model, tokenizer = student_model_and_tokenizer
        onnx_file = tmp_path / "test_student.onnx"

        exported_path = export_to_onnx(model, tokenizer, onnx_file, max_seq_len=64, verify=True)
        assert exported_path.exists()
        assert exported_path.stat().st_size > 10 * 1024 * 1024  # > 10MB (full ~44MB model)

    def test_dynamic_quantization(self, student_model_and_tokenizer, tmp_path):
        model, tokenizer = student_model_and_tokenizer
        onnx_file = tmp_path / "test_student.onnx"
        quant_file = tmp_path / "test_student_int8.onnx"

        export_to_onnx(model, tokenizer, onnx_file, max_seq_len=64, verify=False)
        quant_path = quantize_onnx_dynamic(onnx_file, quant_file)

        assert quant_path.exists()
        assert quant_path.stat().st_size < onnx_file.stat().st_size

    def test_pytorch_latency_measurement(self, student_model_and_tokenizer):
        model, tokenizer = student_model_and_tokenizer
        res = measure_pytorch_latency_and_memory(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            n_iterations=5,
            warmup=2,
            max_seq_len=32,
        )

        assert "mean_latency_ms" in res
        assert "p95_latency_ms" in res
        assert "throughput_fps" in res
        assert "peak_memory_mb" in res
        assert res["mean_latency_ms"] > 0.0

    def test_onnx_latency_measurement(self, student_model_and_tokenizer, tmp_path):
        model, tokenizer = student_model_and_tokenizer
        onnx_file = tmp_path / "test_student.onnx"
        export_to_onnx(model, tokenizer, onnx_file, max_seq_len=32, verify=False)

        res = measure_onnx_latency_and_memory(
            onnx_path=onnx_file,
            tokenizer=tokenizer,
            n_iterations=5,
            warmup=2,
            max_seq_len=32,
        )

        assert "mean_latency_ms" in res
        assert "p95_latency_ms" in res
        assert "throughput_fps" in res
        assert res["mean_latency_ms"] > 0.0

    def test_get_checkpoint_disk_size(self):
        size_mb = get_checkpoint_disk_size_mb("experiments/tinybanglabert_final")
        assert size_mb > 10.0  # ~44MB
