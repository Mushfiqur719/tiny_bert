"""scripts/analyze_v3_mechanism.py — Phase G: Mixed-Effects Mechanism Analysis & Interaction Plots.

Reads all Phase F runs in experiments/v3/F_*/seed*/metrics.json:
  1. Compiles experiments/v3/grid_master.csv
  2. Fits linear mixed-effects model:
       f1_macro ~ C(init) * C(objective) * log_frac + (1 | seed)
  3. Plots Macro-F1 vs data fraction (one line per objective, one panel per initialisation)
     to experiments/v3/fig_mechanism_interactions.png
  4. Generates experiments/v3/mechanism_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def collect_grid_dataframe(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(exp_dir.glob("F_*")):
        if not run_dir.is_dir():
            continue
        for mf in sorted(run_dir.glob("seed*/metrics.json")):
            data = json.loads(mf.read_text(encoding="utf-8"))
            rows.append({
                "run_name": run_dir.name,
                "init": data.get("init"),
                "objective": data.get("objective"),
                "frac": int(data.get("frac", 100)),
                "log_frac": float(np.log(int(data.get("frac", 100)))),
                "seed": int(data.get("seed", 42)),
                "f1_macro": float(data.get("f1_macro", 0.0)),
                "accuracy": float(data.get("accuracy", 0.0)),
                "ece": float(data.get("ece", 0.0)),
                "epochs_to_best": float(data.get("epochs_to_best", 0.0)),
                "teacher_agreement": data.get("teacher_agreement", np.nan),
            })
    return pd.DataFrame(rows)


def run_mechanism_analysis(dataset: str = "banglabait") -> None:
    exp_v3 = Path("experiments/v3_baitbuster" if dataset == "baitbuster" else "experiments/v3")
    df = collect_grid_dataframe(exp_v3)
    if df.empty:
        logger.warning("No Phase F runs found in %s/F_*. Run scripts.run_v3_grid first.", exp_v3)
        return

    csv_path = exp_v3 / "grid_master.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d Phase F run rows to %s", len(df), csv_path)

    # 1. Summary table by (init, objective, frac)
    summary = (
        df.groupby(["init", "objective", "frac"])
        .agg(
            n_seeds=("f1_macro", "count"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            ece_mean=("ece", "mean"),
            agree_mean=("teacher_agreement", "mean"),
            ep_best_mean=("epochs_to_best", "mean"),
        )
        .reset_index()
    )

    # 2. Mixed-effects model with centered log_frac, full interactions, and direct H1 Wald contrast
    model_summary_text = ""
    h1_contrast_text = ""
    try:
        df["log_frac_c"] = df["log_frac"] - df["log_frac"].mean()
        init_cats = [c for c in ["rand", "first4", "last4", "strided"] if c in df["init"].unique()]
        obj_cats = [c for c in ["ce", "kd", "kd_hid"] if c in df["objective"].unique()]
        df["init"] = pd.Categorical(df["init"], categories=init_cats)
        df["objective"] = pd.Categorical(df["objective"], categories=obj_cats)
        formula = "f1_macro ~ C(init) * C(objective) * log_frac_c"
        md = smf.mixedlm(formula, df, groups=df["seed"])
        mdf = md.fit(reml=False)
        model_summary_text = str(mdf.summary())

        # Direct H1 Wald contrast: C(init)[T.strided] - C(objective)[T.kd] = 0
        fe_names = list(mdf.fe_params.index)
        init_key = "C(init)[T.strided]"
        obj_key = "C(objective)[T.kd]"
        if init_key in fe_names and obj_key in fe_names:
            r_vec = np.zeros((1, len(fe_names)))
            r_vec[0, fe_names.index(init_key)] = 1.0
            r_vec[0, fe_names.index(obj_key)] = -1.0
            # Pad for random effects variance parameter if needed
            k_full = len(mdf.params)
            if k_full > len(fe_names):
                r_full = np.zeros((1, k_full))
                r_full[0, : len(fe_names)] = r_vec[0]
            else:
                r_full = r_vec
            wald = mdf.wald_test(r_full, scalar=True)
            diff_val = float(mdf.fe_params[init_key] - mdf.fe_params[obj_key])
            cov_fe = mdf.cov_params().loc[[init_key, obj_key], [init_key, obj_key]].to_numpy()
            se_diff = float(np.sqrt(cov_fe[0, 0] + cov_fe[1, 1] - 2.0 * cov_fe[0, 1]))
            ci_lo = (diff_val - 1.96 * se_diff) * 100.0
            ci_hi = (diff_val + 1.96 * se_diff) * 100.0
            h1_contrast_text = (
                f"Direct H1 Wald Contrast (beta_strided - beta_kd at mean log_frac): "
                f"{diff_val*100:+.2f} pp (SE={se_diff*100:.2f} pp, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}] pp, "
                f"chi2={float(wald.statistic):.2f}, p={float(wald.pvalue):.4e})"
            )
            logger.info(h1_contrast_text)
        logger.info("Mixed-effects model converged.")
    except Exception as e:
        model_summary_text = f"Mixed-effects model could not be fit with current subset of cells: {e}"
        logger.warning(model_summary_text)

    # 3. Interaction plot: F1 vs Data Fraction, one panel per init, one line per objective
    inits = [i for i in ["rand", "first4", "last4", "strided"] if i in df["init"].unique()]
    if inits:
        fig, axes = plt.subplots(1, len(inits), figsize=(4.5 * len(inits), 4.2), sharey=True)
        if len(inits) == 1:
            axes = [axes]
        colors = {"ce": "#1f77b4", "kd": "#ff7f0e", "kd_hid": "#2ca02c"}
        markers = {"ce": "o", "kd": "s", "kd_hid": "^"}

        for ax, init_arm in zip(axes, inits):
            sub = summary[summary["init"] == init_arm]
            for obj_arm in ["ce", "kd", "kd_hid"]:
                osub = sub[sub["objective"] == obj_arm].sort_values("frac")
                if osub.empty:
                    continue
                ax.errorbar(
                    osub["frac"],
                    osub["f1_mean"],
                    yerr=osub["f1_std"].fillna(0.0),
                    label=obj_arm.upper(),
                    color=colors.get(obj_arm),
                    marker=markers.get(obj_arm),
                    capsize=3,
                    linewidth=2,
                )
            ax.set_title(f"Init: {init_arm}", fontweight="bold")
            ax.set_xlabel("Training Data Fraction (%)")
            ax.set_xticks([10, 25, 50, 100])
            ax.grid(alpha=0.3)

        axes[0].set_ylabel("Validation Macro-F1 (Mean ± Std)")
        axes[-1].legend(title="Objective", loc="lower right")
        plt.suptitle("Phase G: Initialisation × Distillation Objective × Data Fraction", fontweight="bold")
        plt.tight_layout()
        fig_path = exp_v3 / "fig_mechanism_interactions.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        logger.info("Saved interaction plot to %s", fig_path)

    # 4. Write report
    report_md = f"""# Phase G — Mechanism & Mixed-Effects Analysis Report

**Total Grid Runs Analysed:** {len(df)}

## 1. Cell-Level Summary (Validation Split)

{summary.to_markdown(index=False, floatfmt=".4f")}

---

## 2. Linear Mixed-Effects Model (Centered log_frac, Full Interactions)
Formula: `f1_macro ~ C(init) * C(objective) * log_frac_c + (1 | seed)`

{h1_contrast_text}

```
{model_summary_text}
```
"""
    report_path = exp_v3 / "mechanism_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved mechanism report to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase G: Mixed-effects mechanism analysis.")
    parser.add_argument("--dataset", choices=["banglabait", "baitbuster"], default="banglabait")
    args = parser.parse_args()
    run_mechanism_analysis(dataset=args.dataset)
