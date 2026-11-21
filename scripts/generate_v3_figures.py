"""scripts/generate_v3_figures.py — Generate 300 DPI publication-grade figures for TinyBanglaBERT v3.

Reads strictly from verified v3 CSVs and JSON metric outputs:
  - experiments/v3/grid_master.csv & experiments/v3_baitbuster/grid_master.csv
  - experiments/v3/final_test_comparisons.csv & experiments/v3_baitbuster/final_test_comparisons.csv
  - experiments/v3/crossgenre_v3_table.csv
  - experiments/v3/efficiency.csv & experiments/v3/cascade_rtt_table.csv
  - Held-out test_metrics.json confusion matrices

Outputs 6 figures to experiments/v3/figures/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("experiments/v3/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig1_factorial_interactions() -> None:
    """Fig 1: Dual-dataset H1/H2 learning curves (Init x Objective x Data Fraction)."""
    df_bb = pd.read_csv("experiments/v3/grid_master.csv")
    df_bu = pd.read_csv("experiments/v3_baitbuster/grid_master.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    styles = {
        ("strided", "kd"): dict(label="Strided Init + KD", color="#1b9e77", marker="o", ls="-", lw=2.3),
        ("strided", "ce"): dict(label="Strided Init + CE", color="#2b8cbe", marker="s", ls="--", lw=2.0),
        ("rand", "kd"):    dict(label="Random Init + KD",  color="#d95f02", marker="^", ls="-", lw=2.1),
        ("rand", "ce"):    dict(label="Random Init + CE",  color="#7570b3", marker="D", ls="--", lw=1.9),
    }

    for ax, df, title in [
        (axes[0], df_bb, "(a) BanglaBait (News Headlines, N=15,032)"),
        (axes[1], df_bu, "(b) BaitBuster-Bangla (YouTube Titles, N=10,000)"),
    ]:
        agg = df.groupby(["init", "objective", "frac"])["f1_macro"].agg(["mean", "std"]).reset_index()
        for (init_k, obj_k), st in styles.items():
            sub = agg[(agg["init"] == init_k) & (agg["objective"] == obj_k)].sort_values("frac")
            if sub.empty:
                continue
            ax.errorbar(
                sub["frac"], sub["mean"], yerr=sub["std"].fillna(0),
                capsize=3.5, markersize=6, **st
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Labelled Training Data Fraction (%)")
        ax.set_xticks([10, 25, 50, 100])
        ax.grid(alpha=0.3, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("5-Seed Validation Macro-F1 (Mean ± Std)")
    axes[1].legend(loc="lower right", frameon=True)
    plt.suptitle("Fig. 1: Pre-Trained Initialisation vs. Distillation Objective Across Data Scales (7.42M 4L)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_v3_factorial_interactions.png")
    plt.close()
    logger.info("Generated fig1_v3_factorial_interactions.png")


def fig2_calibration_and_agreement() -> None:
    """Fig 2: (a) Expected Calibration Error (ECE) CE vs KD; (b) Teacher Agreement."""
    df_bb = pd.read_csv("experiments/v3/grid_master.csv")
    df_bu = pd.read_csv("experiments/v3_baitbuster/grid_master.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # Left: 100% data ECE reduction from KD
    labels = ["BBait\nRand", "BBait\nStrided", "BBuster\nRand", "BBuster\nStrided"]
    ce_ece = [
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "rand") & (df_bb["objective"] == "ce")]["ece"].mean(),
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "strided") & (df_bb["objective"] == "ce")]["ece"].mean(),
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "rand") & (df_bu["objective"] == "ce")]["ece"].mean(),
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "strided") & (df_bu["objective"] == "ce")]["ece"].mean(),
    ]
    kd_ece = [
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "rand") & (df_bb["objective"] == "kd")]["ece"].mean(),
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "strided") & (df_bb["objective"] == "kd")]["ece"].mean(),
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "rand") & (df_bu["objective"] == "kd")]["ece"].mean(),
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "strided") & (df_bu["objective"] == "kd")]["ece"].mean(),
    ]
    x = np.arange(len(labels))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, ce_ece, w, label="Cross-Entropy (CE)", color="#6baed6", edgecolor="black", lw=0.6)
    b2 = axes[0].bar(x + w / 2, kd_ece, w, label="Logit Distillation (KD)", color="#1b9e77", edgecolor="black", lw=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Expected Calibration Error (ECE ↓)")
    axes[0].set_title("(a) Probability Calibration at 100% Data", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3, ls=":")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: Teacher Agreement at 100% data
    ce_ag = [
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "rand") & (df_bb["objective"] == "ce")]["teacher_agreement"].mean() * 100,
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "strided") & (df_bb["objective"] == "ce")]["teacher_agreement"].mean() * 100,
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "rand") & (df_bu["objective"] == "ce")]["teacher_agreement"].mean() * 100,
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "strided") & (df_bu["objective"] == "ce")]["teacher_agreement"].mean() * 100,
    ]
    kd_ag = [
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "rand") & (df_bb["objective"] == "kd")]["teacher_agreement"].mean() * 100,
        df_bb[(df_bb["frac"] == 100) & (df_bb["init"] == "strided") & (df_bb["objective"] == "kd")]["teacher_agreement"].mean() * 100,
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "rand") & (df_bu["objective"] == "kd")]["teacher_agreement"].mean() * 100,
        df_bu[(df_bu["frac"] == 100) & (df_bu["init"] == "strided") & (df_bu["objective"] == "kd")]["teacher_agreement"].mean() * 100,
    ]
    b3 = axes[1].bar(x - w / 2, ce_ag, w, label="Cross-Entropy (CE)", color="#6baed6", edgecolor="black", lw=0.6)
    b4 = axes[1].bar(x + w / 2, kd_ag, w, label="Logit Distillation (KD)", color="#1b9e77", edgecolor="black", lw=0.6)
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(75, 103)
    axes[1].set_ylabel("Student–Teacher Agreement (%) ↑")
    axes[1].set_title("(b) Teacher Prediction Fidelity at 100% Data", fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", alpha=0.3, ls=":")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Fig. 2: Mechanism Analysis — Distillation Drives Calibration & Teacher Fidelity", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_v3_calibration_agreement.png")
    plt.close()
    logger.info("Generated fig2_v3_calibration_agreement.png")


def fig3_test_noninferiority_forest() -> None:
    """Fig 3: Forest plot of 10k paired bootstrap Macro-F1 differences with -2.0 pp non-inferiority margin."""
    c_bu = pd.read_csv("experiments/v3_baitbuster/final_test_comparisons.csv")
    c_bb = pd.read_csv("experiments/v3/final_test_comparisons.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # Left: BaitBuster vs Teacher Base (110.6M)
    sub_bu = c_bu[c_bu["reference"] == "teacher_base_uw"].copy()
    name_map_bu = {
        "baseline_small_12l_ce_uw": "Small-12L (13.7M)",
        "F_strided_ce_100_uw": "4L Strided + CE (7.4M)",
        "F_strided_kd_100_uw": "4L Strided + KD (7.4M)",
        "F_rand_ce_100_uw": "4L Rand + CE (7.4M)",
        "F_rand_kd_100_uw": "4L Rand + KD (7.4M)",
    }
    sub_bu["label"] = sub_bu["model"].map(name_map_bu)
    sub_bu = sub_bu.dropna(subset=["label"]).reset_index(drop=True)

    # Right: BanglaBait vs Small-12L (13.7M)
    sub_bb = c_bb[c_bb["reference"] == "baseline_small_12l_ce"].copy()
    name_map_bb = {
        "teacher_base_tb": "Base Title+Body (110.6M)",
        "teacher_base": "Base Title (110.6M)",
        "F_strided_kd_100": "4L Strided + KD (7.4M)",
        "F_strided_ce_100": "4L Strided + CE (7.4M)",
        "F_rand_ce_100": "4L Rand + CE (7.4M)",
        "F_rand_kd_100": "4L Rand + KD (7.4M)",
    }
    sub_bb["label"] = sub_bb["model"].map(name_map_bb)
    sub_bb = sub_bb.dropna(subset=["label"]).reset_index(drop=True)

    for ax, sub, title in [
        (axes[0], sub_bu, "(a) BaitBuster Test vs. 110.6M Teacher Base"),
        (axes[1], sub_bb, "(b) BanglaBait Test vs. 13.7M Small-12L"),
    ]:
        y = np.arange(len(sub))
        for i, row in sub.iterrows():
            color = "#1b9e77" if bool(row["non_inferior_2pp"]) else "#d95f02"
            err_lo = row["diff_pp"] - row["ci_lo_pp"]
            err_hi = row["ci_hi_pp"] - row["diff_pp"]
            ax.errorbar(row["diff_pp"], i, xerr=[[err_lo], [err_hi]], fmt="o", color=color, capsize=4, lw=2.2, ms=7)
            ax.text(row["ci_hi_pp"] + 0.15, i, f"{row['diff_pp']:+.2f} pp", va="center", fontsize=8.5, fontweight="bold", color=color)

        ax.axvline(0.0, color="black", lw=1.0, ls="-", alpha=0.6)
        ax.axvline(-2.0, color="#cb181d", lw=1.6, ls="--", label="Non-Inferiority Margin (-2.0 pp)")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"])
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 Difference vs. Reference (percentage points, 95% CI)")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8.5)
        ax.grid(axis="x", alpha=0.3, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_xlim(-2.6, 1.8)
    axes[1].set_xlim(-6.2, 7.6)
    plt.suptitle("Fig. 3: One-Time Held-Out Test Non-Inferiority Analysis (10,000 Paired Bootstrap Resamples)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_v3_test_noninferiority.png")
    plt.close()
    logger.info("Generated fig3_v3_test_noninferiority.png")


def fig4_crossgenre_recalibration() -> None:
    """Fig 4: Zero-Shot vs 1% Recalibrated Cross-Genre Macro-F1 (News -> YouTube, N=10,000)."""
    df = pd.read_csv("experiments/v3/crossgenre_v3_table.csv")
    name_map = {
        "teacher_base": "Teacher Base\n(110.6M)",
        "baseline_small_12l_ce": "Small-12L\n(13.7M)",
        "F_strided_kd_100": "4L Strided+KD\n(7.42M)",
        "F_strided_ce_100": "4L Strided+CE\n(7.42M)",
        "F_rand_kd_100": "4L Rand+KD\n(7.42M)",
    }
    labels = [name_map.get(m, m) for m in df["model"]]
    x = np.arange(len(df))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    b1 = ax.bar(x - w / 2, df["zs_f1_mean"], w, label="Zero-Shot (News Threshold)", color="#74a9cf", edgecolor="black", lw=0.6)
    b2 = ax.bar(x + w / 2, df["recal1pct_f1_mean"], w, label="1% Target Recalibration (n=100)", color="#1b9e77", edgecolor="black", lw=0.6)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.65, 0.92)
    ax.set_ylabel("5-Seed Cross-Genre Macro-F1 (YouTube Titles)")
    ax.set_title("Fig. 4: Cross-Genre Generalisation (BanglaBait News → BaitBuster YouTube, N=10,000)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_v3_crossgenre_recalibration.png")
    plt.close()
    logger.info("Generated fig4_v3_crossgenre_recalibration.png")


def fig5_efficiency_and_cascade() -> None:
    """Fig 5: (a) Unified ONNX Runtime Latency (FP32 vs INT8, 4T); (b) Cascade Effective Latency with RTT."""
    eff = pd.read_csv("experiments/v3/efficiency.csv")
    cas = pd.read_csv("experiments/v3/cascade_rtt_table.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    # Left: 4-thread p50 latency & disk size
    sub4 = eff[eff["threads"] == 4].copy()
    models_order = ["BanglaBERT-Base (110.6M)", "BanglaBERT-Small-12L (13.7M)", "TinyBanglaBERT-4L (7.42M)"]
    short_labels = ["Base\n(110.6M)", "Small-12L\n(13.7M)", "Tiny-4L\n(7.42M)"]
    fp32_lat = [sub4[(sub4["model"] == m) & (sub4["precision"] == "FP32")]["p50_ms"].iloc[0] for m in models_order]
    int8_lat = [sub4[(sub4["model"] == m) & (sub4["precision"] == "INT8")]["p50_ms"].iloc[0] for m in models_order]

    x = np.arange(len(models_order))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, fp32_lat, w, label="ONNX FP32 (4T)", color="#74a9cf", edgecolor="black", lw=0.6)
    b2 = axes[0].bar(x + w / 2, int8_lat, w, label="ONNX Dynamic INT8 (4T)", color="#1b9e77", edgecolor="black", lw=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.2f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels)
    axes[0].set_ylabel("p50 Single-Headline CPU Latency (ms) ↓")
    axes[0].set_ylim(0, 86)
    axes[0].set_title("(a) Unified ONNX Runtime CPU Latency (Batch=1)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3, ls=":")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: Cascade effective latency under 0, 50, 150 ms RTT
    cas["esc_num"] = cas["escalation_rate"].str.rstrip("%").astype(float)
    for rtt, color, marker in [(0.0, "#2b8cbe", "o"), (50.0, "#fdae6b", "s"), (150.0, "#cb181d", "^")]:
        csub = cas[cas["network_rtt_ms"] == rtt].sort_values("esc_num")
        axes[1].plot(csub["esc_num"], csub["effective_avg_ms"], label=f"Network RTT = {int(rtt)} ms", color=color, marker=marker, lw=2.2, ms=6)
    axes[1].set_xlabel("Cloud Escalation Rate (%)")
    axes[1].set_ylabel("Effective Average Latency (ms) ↓")
    axes[1].set_xticks([0, 10, 25, 50, 100])
    axes[1].set_title("(b) Edge–Cloud Cascade Latency with Mobile RTT", fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3, ls=":")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Fig. 5: Edge Inference Speedup (42.1×) and Real-World Network Cascade Latency", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_v3_efficiency_cascade.png")
    plt.close()
    logger.info("Generated fig5_v3_efficiency_cascade.png")


def fig6_confusion_matrices() -> None:
    """Fig 6: 2x2 Held-Out Test Confusion Matrices (Seed 42) across both benchmarks."""
    panels = [
        ("experiments/v3_baitbuster/teacher_base_uw/seed42/test_metrics.json", "(a) BaitBuster — Base 110.6M (Test)"),
        ("experiments/v3_baitbuster/F_strided_kd_100_uw/seed42/test_metrics.json", "(b) BaitBuster — Tiny-4L 7.42M (Test)"),
        ("experiments/v3/teacher_base_tb/seed42/test_metrics.json", "(c) BanglaBait — Base Title+Body (Test)"),
        ("experiments/v3/F_strided_kd_100/seed42/test_metrics.json", "(d) BanglaBait — Tiny-4L 7.42M (Test)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
    for ax, (path_str, title) in zip(axes.flat, panels):
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
        cm = np.array(data["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Not-CB (0)", "Clickbait (1)"],
            yticklabels=["Not-CB (0)", "Clickbait (1)"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_title(f"{title}\nAcc={data['accuracy']*100:.2f}% | F1={data['f1_macro']:.4f}", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Fig. 6: Held-Out Test Confusion Matrices Across Both Benchmarks (Seed 42)", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_v3_confusion_matrices.png")
    plt.close()
    logger.info("Generated fig6_v3_confusion_matrices.png")


if __name__ == "__main__":
    fig1_factorial_interactions()
    fig2_calibration_and_agreement()
    fig3_test_noninferiority_forest()
    fig4_crossgenre_recalibration()
    fig5_efficiency_and_cascade()
    fig6_confusion_matrices()
    logger.info("All 6 v3 publication figures generated in %s", OUT_DIR)
