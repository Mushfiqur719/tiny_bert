"""
src/evaluation/selective.py — Stage 4: Selective Prediction (Risk–Coverage / AURC) &
Two-Tier Edge–Cloud Deferral Cascade.

Computes:
  1. Temperature scaling T_cal fit strictly on validation logits (minimizing validation NLL).
  2. Accuracy and Macro-F1 across coverage levels C in {100%, 90%, 85%, 80%, 75%, 70%, 60%, 50%}
     when abstaining on the lowest-confidence (max_c p(y=c|x)) predictions.
  3. Area Under the Risk–Coverage Curve (AURC) and the coverage level achieving >= 90%, 92%, 95% accuracy.
  4. Two-Tier Edge–Cloud Deferral Cascade:
     - Tier 1 (Edge: TinyBanglaBERT-4L INT8, 4.00 ms) classifies the top C% most confident headlines locally.
     - Tier 2 (Cloud: Teacher BanglaBERT, 12.60 ms GPU / 123.36 ms CPU) classifies only the deferred (100 - C)% headlines.
     - Reports combined system Accuracy, Macro-F1, % Cloud Calls Saved, and Amortized Latency (ms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.clean import clean_dataframe
from src.data.load import load_raw
from src.data.split import apply_splits, load_splits
from src.utils import load_config


@torch.no_grad()
def ensure_cached_logits(model_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached val/test logits & labels from model_dir, or compute & save them if missing."""
    d = Path(model_dir)
    val_lg_path = d / "val_logits.npy"
    val_lb_path = d / "val_labels.npy"
    test_lg_path = d / "test_logits.npy"
    test_lb_path = d / "test_labels.npy"

    if val_lg_path.exists() and val_lb_path.exists() and test_lg_path.exists() and test_lb_path.exists():
        return (
            np.load(val_lg_path),
            np.load(val_lb_path),
            np.load(test_lg_path),
            np.load(test_lb_path),
        )

    cfg = load_config("configs/base.yaml")
    labeled_df, _ = load_raw(cfg, hf_dataset_id=cfg.data.hf_dataset_id)
    labeled_df = clean_dataframe(
        labeled_df,
        headline_col=cfg.data.text_col,
        min_char_length=int(cfg.data.min_char_length),
    )
    labeled_df = labeled_df.drop(columns=["_was_cleaned"], errors="ignore").reset_index(drop=True)
    _, val_df, test_df = apply_splits(labeled_df, load_splits(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(d))
    model = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device).eval()

    def _infer(df_split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        texts = df_split[cfg.data.text_col].tolist()
        labels = df_split[cfg.data.label_col].to_numpy(dtype=np.int64)
        chunks = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i : i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            chunks.append(model(**enc).logits.cpu().numpy())
        return np.concatenate(chunks, axis=0), labels

    val_logits, val_labels = _infer(val_df)
    test_logits, test_labels = _infer(test_df)

    np.save(val_lg_path, val_logits)
    np.save(val_lb_path, val_labels)
    np.save(test_lg_path, test_logits)
    np.save(test_lb_path, test_labels)
    return val_logits, val_labels, test_logits, test_labels


def softmax_scaled(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-4)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


def fit_temperature_on_val(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit post-hoc temperature scalar T > 0 minimizing Negative Log-Likelihood on validation split."""
    y = np.asarray(val_labels, dtype=int)
    n = len(y)

    def nll(t: float) -> float:
        probs = softmax_scaled(val_logits, temperature=t)
        p_true = np.clip(probs[np.arange(n), y], 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return round(float(res.x), 4) if res.success else 1.0


def compute_aurc(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration."""
    order = np.argsort(-confidences)
    errors = (y_pred[order] != y_true[order]).astype(np.float64)
    n = len(errors)
    cum_errors = np.cumsum(errors)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_errors / ks
    coverages = ks / n
    return float(np.trapezoid(risks, coverages))


def evaluate_selective_prediction(
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    temperature: float = 1.0,
    coverages: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute selective accuracy & Macro-F1 across coverage levels."""
    if coverages is None:
        coverages = [1.00, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30]

    y_true = np.asarray(test_labels, dtype=int)
    probs = softmax_scaled(test_logits, temperature=temperature)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    order = np.argsort(-conf)
    n = len(y_true)

    rows = []
    for cov in coverages:
        k = max(10, int(round(cov * n)))
        idx = order[:k]
        acc = float(accuracy_score(y_true[idx], y_pred[idx]))
        f1 = float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        cb_f1 = float(f1_score(y_true[idx], y_pred[idx], pos_label=1, average="binary", zero_division=0))
        min_conf = float(conf[idx[-1]])
        rows.append(
            {
                "Coverage (%)": round(cov * 100.0, 1),
                "Retained Samples": k,
                "Confidence Threshold": round(min_conf, 4),
                "Selective Accuracy (%)": round(acc * 100.0, 2),
                "Selective Macro-F1": round(f1, 4),
                "Selective Clickbait-F1": round(cb_f1, 4),
                "Error Risk (%)": round((1.0 - acc) * 100.0, 2),
            }
        )

    # Find highest coverage achieving >= 90%, >= 92%, >= 95% accuracy
    targets = {90.0: 0.0, 92.0: 0.0, 95.0: 0.0}
    correct_sorted = (y_pred[order] == y_true[order]).astype(np.float64)
    running_acc = np.cumsum(correct_sorted) / np.arange(1, n + 1) * 100.0
    for target_acc in targets:
        valid_ks = np.where((running_acc >= target_acc) & (np.arange(1, n + 1) >= 25))[0]
        if len(valid_ks) > 0:
            max_k = int(valid_ks[-1]) + 1
            targets[target_acc] = round(100.0 * max_k / n, 2)

    aurc = compute_aurc(y_true, y_pred, conf)
    summary = {
        "temperature_cal": temperature,
        "aurc": round(aurc, 5),
        "max_coverage_at_90pct_acc": targets[90.0],
        "max_coverage_at_92pct_acc": targets[92.0],
        "max_coverage_at_95pct_acc": targets[95.0],
        "peak_selective_acc_pct": round(float(np.max(running_acc[24:])), 2),
    }
    return pd.DataFrame(rows), summary


def evaluate_edge_cloud_cascade(
    student_test_logits: np.ndarray,
    teacher_test_logits: np.ndarray,
    test_labels: np.ndarray,
    student_temp: float = 1.0,
    edge_latency_ms: float = 4.00,
    cloud_gpu_latency_ms: float = 12.60,
    cloud_cpu_latency_ms: float = 123.36,
    edge_coverages: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate a 2-tier Edge (TinyBanglaBERT-4L INT8) -> Cloud (Teacher) deferral system."""
    if edge_coverages is None:
        edge_coverages = [1.00, 0.90, 0.80, 0.75, 0.70, 0.60, 0.50, 0.00]

    y_true = np.asarray(test_labels, dtype=int)
    s_probs = softmax_scaled(student_test_logits, temperature=student_temp)
    s_pred = np.argmax(s_probs, axis=1)
    s_conf = np.max(s_probs, axis=1)
    t_pred = np.argmax(teacher_test_logits, axis=1)

    order = np.argsort(-s_conf)
    n = len(y_true)

    rows = []
    for edge_cov in edge_coverages:
        k_edge = int(round(edge_cov * n))
        edge_idx = set(order[:k_edge].tolist())
        hybrid_pred = np.array(
            [s_pred[i] if i in edge_idx else t_pred[i] for i in range(n)],
            dtype=int,
        )
        acc = float(accuracy_score(y_true, hybrid_pred))
        f1 = float(f1_score(y_true, hybrid_pred, average="macro", zero_division=0))
        cb_f1 = float(f1_score(y_true, hybrid_pred, pos_label=1, average="binary", zero_division=0))

        defer_rate = 1.0 - edge_cov
        # Edge always runs (4.00 ms); deferred items also incur cloud latency
        amort_gpu_ms = edge_latency_ms + defer_rate * cloud_gpu_latency_ms if edge_cov > 0 else cloud_gpu_latency_ms
        amort_cpu_ms = edge_latency_ms + defer_rate * cloud_cpu_latency_ms if edge_cov > 0 else cloud_cpu_latency_ms

        rows.append(
            {
                "Edge Coverage (%)": round(edge_cov * 100.0, 1),
                "Cloud Deferral (%)": round(defer_rate * 100.0, 1),
                "Cloud Calls Saved (%)": round(edge_cov * 100.0, 1),
                "System Accuracy (%)": round(acc * 100.0, 2),
                "System Macro-F1": round(f1, 4),
                "System Clickbait-F1": round(cb_f1, 4),
                "Amortized Latency (Edge+GPU ms)": round(amort_gpu_ms, 2),
                "Amortized Latency (Edge+CPU ms)": round(amort_cpu_ms, 2),
            }
        )
    return pd.DataFrame(rows)


def plot_selective_and_cascade(
    sel_df: pd.DataFrame,
    cas_df: pd.DataFrame,
    output_path: str | Path = "experiments/figures/fig7_selective_accuracy_coverage.png",
) -> Path:
    """Generate 300 DPI two-panel plot of Selective Accuracy vs Coverage and Edge-Cloud Cascade."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    # Panel A: Selective Accuracy & Macro-F1 vs Coverage
    covs = sel_df["Coverage (%)"].tolist()
    accs = sel_df["Selective Accuracy (%)"].tolist()
    f1s = [x * 100.0 for x in sel_df["Selective Macro-F1"].tolist()]

    ax1.plot(covs, accs, marker="o", linewidth=2.4, color="#1d3557", label="Selective Accuracy (%)")
    ax1.plot(covs, f1s, marker="s", linewidth=2.2, linestyle="--", color="#2a9d8f", label="Selective Macro-F1 (%)")
    ax1.axhline(90.0, color="#e63946", linestyle=":", linewidth=1.2, label="90% Accuracy Target")
    ax1.invert_xaxis()
    ax1.set_xlabel("Coverage (% of Headlines Classified Locally)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Performance (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) TinyBanglaBERT-4L Selective Accuracy vs. Coverage", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # Panel B: Two-Tier Edge-Cloud Cascade Accuracy vs Amortized CPU Latency
    e_covs = cas_df["Edge Coverage (%)"].tolist()
    sys_accs = cas_df["System Accuracy (%)"].tolist()
    lats = cas_df["Amortized Latency (Edge+CPU ms)"].tolist()

    ax2.plot(e_covs, sys_accs, marker="D", linewidth=2.4, color="#e07a5f", label="Cascade System Accuracy (%)")
    ax2.set_xlabel("Edge Offload Ratio (% Handled by 4ms Edge Student)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Overall System Accuracy (%)", color="#e07a5f", fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#e07a5f")

    ax2_r = ax2.twinx()
    ax2_r.plot(e_covs, lats, marker="^", linewidth=2.2, linestyle="--", color="#2b5c8f", label="Amortized CPU Latency (ms)")
    ax2_r.set_ylabel("Amortized CPU Latency (ms/headline)", color="#2b5c8f", fontsize=10.5, fontweight="bold")
    ax2_r.tick_params(axis="y", labelcolor="#2b5c8f")
    ax2_r.grid(False)
    ax2.set_title("(b) Two-Tier Edge–Cloud Deferral Cascade", fontsize=11, fontweight="bold")

    fig.suptitle("Figure 7: Selective Prediction & Hybrid Edge–Cloud Deferral Performance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
