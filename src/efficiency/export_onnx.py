"""
src/efficiency/export_onnx.py — Export TinyBanglaBERT and Teacher to ONNX and INT8 Quantization.

Provides:
  1. `export_to_onnx`: Exports a PyTorch Transformer sequence classification model to ONNX
     with dynamic batching and sequence length.
  2. `verify_onnx_parity`: Compares PyTorch vs. ONNX Runtime inference logits on sample inputs.
  3. `quantize_onnx_dynamic`: Performs dynamic INT8 quantization via ONNX Runtime for edge deployment.

Usage:
    from src.efficiency.export_onnx import export_to_onnx, quantize_onnx_dynamic
    onnx_path = export_to_onnx(model, tokenizer, "experiments/onnx/tinybanglabert.onnx")
    int8_path = quantize_onnx_dynamic(onnx_path, "experiments/onnx/tinybanglabert_int8.onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str | Path,
    max_seq_len: int = 128,
    opset_version: int = 14,
    verify: bool = True,
) -> Path:
    """Export a PyTorch sequence classification model to ONNX format.

    Args:
        model: Trained PyTorch / HuggingFace model.
        tokenizer: Associated tokenizer.
        output_path: Destination path for `.onnx` file.
        max_seq_len: Dummy input sequence length for export trace.
        opset_version: ONNX operator set version (default: 14).
        verify: If True, validates ONNX graph and verifies numerical parity against PyTorch.

    Returns:
        Path to exported ONNX model.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()

    # Create dummy inputs
    sample_text = "বাংলাদেশ জাতীয় ক্রিকেট দলের ঐতিহাসিক টেস্ট ম্যাচ জয়"
    inputs = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    input_names = ["input_ids", "attention_mask"]
    dummy_args = (inputs["input_ids"], inputs["attention_mask"])

    if "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        dummy_args = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }
    if "token_type_ids" in inputs:
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    logger.info("Exporting PyTorch model to ONNX: '%s' (opset=%d)...", out_file, opset_version)

    torch.onnx.export(
        model_cpu,
        dummy_args,
        str(out_file),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info("ONNX export complete: '%s' (Size: %.2f MB)", out_file, size_mb)

    if verify:
        import onnx
        onnx_model = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX graph validation passed.")

        # Check numerical parity
        verify_onnx_parity(model_cpu, str(out_file), tokenizer, sample_text)

    return out_file


def verify_onnx_parity(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    test_text: str = "আজকের প্রধান খবর বাংলাদেশ ও অর্থনীতি",
    atol: float = 1e-4,
) -> None:
    """Verify numerical parity between PyTorch model and ONNX runtime session.

    Args:
        pytorch_model: PyTorch model in eval mode.
        onnx_path: Path to exported ONNX model.
        tokenizer: Tokenizer.
        test_text: Sample sentence for verification.
        atol: Absolute tolerance for difference in logits.
    """
    import onnxruntime as ort

    inputs = tokenizer(test_text, padding="max_length", max_length=64, return_tensors="pt")

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_outputs = pytorch_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pt_logits = pt_outputs.logits.numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    if "token_type_ids" in inputs and "token_type_ids" in [inp.name for inp in session.get_inputs()]:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].numpy()

    ort_outputs = session.run(None, ort_inputs)
    onnx_logits = ort_outputs[0]

    max_diff = np.max(np.abs(pt_logits - onnx_logits))
    logger.info("PyTorch vs ONNX Parity Max Absolute Difference: %.6f", max_diff)

    assert np.allclose(pt_logits, onnx_logits, atol=atol), (
        f"ONNX vs PyTorch output mismatch! Max diff: {max_diff:.6f} > {atol}"
    )
    logger.info("Numerical parity verified successfully (tolerance <= %s).", atol)


def quantize_onnx_dynamic(
    input_onnx_path: str | Path,
    output_onnx_path: str | Path,
) -> Path:
    """Perform dynamic INT8 quantization on an ONNX model.

    Args:
        input_onnx_path: Path to standard FP32 `.onnx` model.
        output_onnx_path: Path to save quantized INT8 `.onnx` model.

    Returns:
        Path to quantized ONNX model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    in_file = Path(input_onnx_path)
    out_file = Path(output_onnx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Performing dynamic INT8 quantization on '%s'...", in_file)

    quantize_dynamic(
        model_input=str(in_file),
        model_output=str(out_file),
        weight_type=QuantType.QUInt8,
    )

    size_orig_mb = in_file.stat().st_size / (1024 * 1024)
    size_quant_mb = out_file.stat().st_size / (1024 * 1024)
    compression = size_orig_mb / size_quant_mb

    logger.info(
        "INT8 Quantization complete: '%s' (%.2f MB -> %.2f MB, %.2fx smaller)",
        out_file,
        size_orig_mb,
        size_quant_mb,
        compression,
    )
    return out_file
