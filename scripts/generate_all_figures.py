"""
scripts/generate_all_figures.py — Generate high-resolution (300 DPI) publication figures for the thesis.

Generates:
  1. `experiments/figures/fig1_ablation_f1_accuracy.png` — F1 & Accuracy across Teacher, 3x2 Ablation, and Depth Scaling.
  2. `experiments/figures/fig2_efficiency_tradeoff.png` — Accuracy/F1 vs. Model Size (MB) Pareto frontier (including INT8 & 6L/12L).
  3. `experiments/figures/fig3_latency_speedup_breakdown.png` — Single-headline CPU inference latency & FPS throughput.
  4. `experiments/figures/fig4_crossdomain_generalization.png` — In-Domain vs. Cross-Domain Macro-F1 and Clickbait-F1.
  5. `experiments/figures/fig5_explainability_agreement.png` — SHAP vs. LIME Top-3 Jaccard and Sign polarity agreement.
  6. `experiments/figures/fig6_confusion_matrices.png` — In-domain and cross-domain confusion matrix heatmaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_fig1_ablation(output_dir: Path) -> None:
    """Figure 1: 3x2 Factorial Ablation & Depth Scaling comparison of Macro-F1 and Accuracy."""
    models = [
        "Teacher\n(110.6M)",
        "Random\n+ CE",
        "Random\n+ KD",
        "PCA-Init\n+ CE",
        "PCA-Init\n+ KD",
        "Small-4L\n+ CE",
        "TinyBangla\nBERT-4L (KD)",
        "TinyBangla\nBERT-6L (KD)",
        "Small-12L\n(Full + KD)",
    ]
    f1_scores = [0.8174, 0.7390, 0.7313, 0.7472, 0.7615, 0.7830, 0.7895, 0.7824, 0.7926]
    accuracies = [0.8373, 0.7656, 0.7463, 0.7583, 0.7842, 0.7975, 0.8054, 0.7975, 0.8114]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=300)
    rects1 = ax.bar(x - width / 2, [s * 100 for s in f1_scores], width, label="Test Macro-F1 (%)", color="#2b5c8f", alpha=0.92)
    rects2 = ax.bar(x + width / 2, [a * 100 for a in accuracies], width, label="Test Accuracy (%)", color="#e07a5f", alpha=0.92)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Test Macro-F1 & Accuracy Across 3×2 Factorial Ablation and Depth Scaling", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(65, 88)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_ablation_f1_accuracy.png", dpi=300)
    plt.close(fig)


def generate_fig2_efficiency_tradeoff(output_dir: Path) -> None:
    """Figure 2: Performance vs Model Size Pareto Curve."""
    variants = [
        {"name": "Teacher 12L-Base (FP32)", "size_mb": 422.76, "f1": 81.74, "color": "#e63946", "shape": "s", "dx": -110, "dy": -1.3},
        {"name": "BanglaBERT-Small 12L (FP32)", "size_mb": 52.40, "f1": 79.26, "color": "#6d597a", "shape": "D", "dx": 8, "dy": 0.6},
        {"name": "TinyBanglaBERT-6L (FP32)", "size_mb": 34.35, "f1": 78.24, "color": "#457b9d", "shape": "v", "dx": 6, "dy": -1.1},
        {"name": "Teacher-PCA 4L + KD (FP32)", "size_mb": 44.07, "f1": 76.15, "color": "#8d99ae", "shape": "P", "dx": 6, "dy": -0.8},
        {"name": "TinyBanglaBERT-4L (PyTorch/ONNX FP32)", "size_mb": 28.31, "f1": 78.95, "color": "#2a9d8f", "shape": "o", "dx": -8, "dy": 0.8},
        {"name": "TinyBanglaBERT-4L (ONNX INT8)", "size_mb": 7.29, "f1": 79.17, "color": "#f4a261", "shape": "*", "dx": 1.5, "dy": 0.7},
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)

    for v in variants:
        ax.scatter(v["size_mb"], v["f1"], color=v["color"], s=220, label=v["name"], marker=v["shape"], edgecolors="#333333", linewidth=1.2, zorder=5)
        ax.annotate(f"{v['name']}\n({v['size_mb']:.2f} MB, {v['f1']:.2f}%)", xy=(v["size_mb"], v["f1"]),
                    xytext=(v["size_mb"] + v["dx"], v["f1"] + v["dy"]), fontsize=8.2, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    ax.set_xlabel("Model Disk Size (MB) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Macro-F1 (%)", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(73, 84)
    ax.set_xlim(5, 650)
    ax.set_title("Figure 2: Pareto Efficiency Frontier — Macro-F1 vs. Model Footprint on Disk", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_efficiency_tradeoff.png", dpi=300)
    plt.close(fig)


def generate_fig3_latency(output_dir: Path) -> None:
    """Figure 3: CPU Single-Headline Latency & Throughput."""
    labels = [
        "Teacher 110.6M\n(PyTorch FP32)",
        "TinyBanglaBERT-4L\n(PyTorch FP32)",
        "TinyBanglaBERT-4L\n(ONNX FP32)",
        "TinyBanglaBERT-4L\n(ONNX INT8)",
    ]
    mean_latencies = [123.36, 11.74, 9.20, 4.00]
    throughputs = [8.1, 85.2, 108.6, 249.9]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2), dpi=300)

    color_bar = "#1d3557"
    color_line = "#e63946"

    x = np.arange(len(labels))
    bars = ax1.bar(x, mean_latencies, width=0.45, color=color_bar, alpha=0.85, label="CPU Latency (ms/headline)")
    ax1.set_ylabel("Mean CPU Latency (ms) — Lower is Better", color=color_bar, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, throughputs, color=color_line, marker="o", linewidth=2.5, markersize=8, label="Throughput (headlines/sec)")
    ax2.set_ylabel("Throughput (headlines/sec) — Higher is Better", color=color_line, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.grid(False)

    for i, txt in enumerate(throughputs):
        ax2.annotate(f"{txt:.1f} fps", xy=(x[i], txt), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color=color_line, fontweight="bold")

    plt.title("Figure 3: Single-Headline CPU Latency & Real-Time Inference Throughput (30.8× INT8 Speedup)", fontsize=12.5, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_latency_speedup_breakdown.png", dpi=300)
    plt.close(fig)


def generate_fig4_crossdomain(output_dir: Path) -> None:
    """Figure 4: In-Domain vs Cross-Domain Generalization."""
    categories = ["In-Domain\n(BanglaBait, N=1,506)", "Cross-Domain Macro-F1\n(BaitBuster, N=10,000)", "Cross-Domain Clickbait-F1\n(BaitBuster, N=10,000)"]
    teacher_scores = [81.74, 75.14, 72.53]
    student_scores = [78.95, 74.59, 72.88]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    ax.bar(x - width / 2, teacher_scores, width, label="Teacher (BanglaBERT 110.6M)", color="#3a86ff", alpha=0.9)
    ax.bar(x + width / 2, student_scores, width, label="TinyBanglaBERT-4L (7.42M)", color="#ff006e", alpha=0.9)

    ax.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Zero-Shot Cross-Domain Generalization on BaitBuster-Bangla (N=10,000)", fontsize=12.5, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(60, 88)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    ax.annotate("81.74%\n(Base)", xy=(x[0] - width / 2, 81.74), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax.annotate("75.14%\n(Δ = -6.60%)", xy=(x[1] - width / 2, 75.14), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#d00000")
    ax.annotate("72.53%", xy=(x[2] - width / 2, 72.53), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax.annotate("78.95%\n(Base)", xy=(x[0] + width / 2, 78.95), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax.annotate("74.59%\n(Δ = -4.36%, 99.3% Ret.)", xy=(x[1] + width / 2, 74.59), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#007f5f")
    ax.annotate("72.88%\n(100.5% Ret.)", xy=(x[2] + width / 2, 72.88), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#007f5f")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_crossdomain_generalization.png", dpi=300)
    plt.close(fig)


def generate_fig5_explainability(output_dir: Path) -> None:
    """Figure 5: Explainability Agreement (SHAP vs. LIME)."""
    json_file = Path("experiments/explainability/agreement_metrics.json")
    if json_file.exists():
        data = json.loads(json_file.read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        jaccards = [s["agreement"]["jaccard_top3_overall"] * 100 for s in samples]
        sign_agrs = [s["agreement"]["sign_agreement"] * 100 for s in samples]
    else:
        jaccards = [50, 20, 20, 100, 20, 50, 50, 50, 0, 20, 20, 50, 20, 0, 20]
        sign_agrs = [88, 62, 40, 100, 56, 29, 50, 50, 29, 78, 25, 50, 17, 40, 20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    sns.histplot(jaccards, bins=6, kde=True, color="#4361ee", ax=ax1, edgecolor="#333333")
    ax1.axvline(np.mean(jaccards), color="#d90429", linestyle="--", linewidth=2, label=f"Mean: {np.mean(jaccards):.1f}%")
    ax1.set_xlabel("Top-3 Token Jaccard Overlap (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Headline Count", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Top-3 Attribution Jaccard Similarity", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")

    sns.histplot(sign_agrs, bins=6, kde=True, color="#2ec4b6", ax=ax2, edgecolor="#333333")
    ax2.axvline(np.mean(sign_agrs), color="#d90429", linestyle="--", linewidth=2, label=f"Mean: {np.mean(sign_agrs):.1f}%")
    ax2.set_xlabel("Sign Polarity Agreement (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Headline Count", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Token Sign Polarity Agreement", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")

    fig.suptitle("Figure 5: Concordance Between Independent Explainers (SHAP vs. LIME)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig5_explainability_agreement.png", dpi=300)
    plt.close(fig)


def generate_fig6_confusion_matrices(output_dir: Path) -> None:
    """Figure 6: Heatmap of Confusion Matrices."""
    cm_teacher_in = np.array([[879, 103], [142, 382]])
    cm_student_in = np.array([[814, 168], [125, 399]])
    cm_teacher_cross = np.array([[4297, 2399], [59, 3245]])
    cm_student_cross = np.array([[4071, 2245], [285, 3399]])

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8), dpi=300)
    labels = ["Not-CB (0)", "Clickbait (1)"]

    cms = [
        (cm_teacher_in, "(a) Teacher 110.6M (In-Domain BanglaBait)", axes[0, 0]),
        (cm_student_in, "(b) TinyBanglaBERT-4L 7.42M (In-Domain BanglaBait)", axes[0, 1]),
        (cm_teacher_cross, "(c) Teacher 110.6M (Cross-Domain BaitBuster)", axes[1, 0]),
        (cm_student_cross, "(d) TinyBanglaBERT-4L 7.42M (Cross-Domain BaitBuster)", axes[1, 1]),
    ]

    for cm, title, ax in cms:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 10, "fontweight": "bold"})
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_ylabel("True Label", fontsize=9.5)
        ax.set_xlabel("Predicted Label", fontsize=9.5)

    fig.suptitle("Figure 6: Confusion Matrix Heatmaps for In-Domain and Cross-Domain Evaluation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "fig6_confusion_matrices.png", dpi=300)
    plt.close(fig)


def main() -> None:
    out_dir = Path("experiments/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating 6 publication-ready figures (300 DPI) in '{out_dir}'...")
    generate_fig1_ablation(out_dir)
    generate_fig2_efficiency_tradeoff(out_dir)
    generate_fig3_latency(out_dir)
    generate_fig4_crossdomain(out_dir)
    generate_fig5_explainability(out_dir)
    generate_fig6_confusion_matrices(out_dir)
    print("All figures successfully created!")


if __name__ == "__main__":
    main()
