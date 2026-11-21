"""scripts/verify_v3_int8_5seed.py — Evaluate 5-seed ONNX Dynamic INT8 held-out test accuracy & Macro-F1.

Eliminates the single-seed INT8 limitation by exporting and evaluating all 5 seeds
of TinyBanglaBERT-4L (F_strided_kd_100) and BanglaBERT-Small-12L in ONNX INT8
on the locked BanglaBait held-out test split (TBB_ALLOW_TEST=1).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_v3_phase_e import SEEDS, compute_full_metrics, get_dataset_paths
from src.data.split import apply_splits
from src.utils.test_guard import assert_test_access

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def eval_int8_seed(ckpt_dir: Path, texts: list[str], y_test: np.ndarray, out_dir: Path, max_length: int = 64) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_p = out_dir / "model_fp32.onnx"
    int8_p = out_dir / "model_int8.onnx"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if not fp32_p.exists() or fp32_p.stat().st_size < 5_000_000:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).cpu().eval()
        dummy = tokenizer("বাংলাদেশ", padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(fp32_p),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )

    if not int8_p.exists():
        quantize_dynamic(str(fp32_p), str(int8_p), weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])

    sess = ort.InferenceSession(str(int8_p), providers=["CPUExecutionProvider"])
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    logits_list = []
    bs = 64
    for i in range(0, len(texts), bs):
        feed = {
            "input_ids": enc["input_ids"][i : i + bs].astype(np.int64),
            "attention_mask": enc["attention_mask"][i : i + bs].astype(np.int64),
        }
        logits_list.append(sess.run(None, feed)[0])
    logits = np.vstack(logits_list)
    m = compute_full_metrics(logits, y_test)
    m["split"] = "test_int8"
    return m


def main():
    os.environ["TBB_ALLOW_TEST"] = "1"
    assert_test_access()

    data_dir, exp_dir, splits_path = get_dataset_paths("banglabait")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    clean_df = pd.read_csv(data_dir / "labeled_clean_v3.csv")
    test_df = apply_splits(clean_df, splits, split="test")
    texts = test_df["headline"].astype(str).tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    rows = []
    for model_name in ("F_strided_kd_100", "baseline_small_12l_ce"):
        accs, f1s, fp32_f1s = [], [], []
        for s in SEEDS:
            ckpt = exp_dir / model_name / f"seed{s}"
            fp32_m = json.loads((ckpt / "test_metrics.json").read_text(encoding="utf-8"))
            fp32_f1s.append(fp32_m["f1_macro"])
            m_int8 = eval_int8_seed(ckpt, texts, y_test, exp_dir / "onnx_5seed" / model_name / f"seed{s}")
            accs.append(m_int8["accuracy"])
            f1s.append(m_int8["f1_macro"])
            logger.info("%s/seed%d | FP32 F1=%.4f | INT8 F1=%.4f | INT8 Acc=%.4f", model_name, s, fp32_m["f1_macro"], m_int8["f1_macro"], m_int8["accuracy"])

        rows.append({
            "model": model_name,
            "n_seeds": len(SEEDS),
            "fp32_f1_mean": round(float(np.mean(fp32_f1s)), 4),
            "int8_acc_mean": round(float(np.mean(accs)), 4),
            "int8_acc_std": round(float(np.std(accs)), 4),
            "int8_f1_mean": round(float(np.mean(f1s)), 4),
            "int8_f1_std": round(float(np.std(f1s)), 4),
            "delta_f1_pp": round((float(np.mean(f1s)) - float(np.mean(fp32_f1s))) * 100.0, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(exp_dir / "int8_5seed_verification.csv", index=False)
    print("\n=== 5-Seed ONNX Dynamic INT8 Held-Out Test Verification ===")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
