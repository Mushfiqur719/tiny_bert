"""
scripts/format_grid_report.py — Compiles grid search checkpoint logs into a markdown report.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    ckpt_dirs = glob.glob("experiments/tuning/checkpoints/run_T*_a*")
    results = []

    for d in ckpt_dirs:
        p = Path(d)
        name = p.name  # run_T1.0_a0.1
        parts = name.replace("run_T", "").split("_a")
        T = float(parts[0])
        alpha = float(parts[1])

        state_files = list(p.glob("**/trainer_state.json"))
        if not state_files:
            continue
        state = json.loads(state_files[0].read_text(encoding="utf-8"))
        eval_logs = [l for l in state["log_history"] if "eval_f1_macro" in l]
        if not eval_logs:
            continue
        best_log = max(eval_logs, key=lambda x: x["eval_f1_macro"])

        results.append({
            "temperature": T,
            "alpha": alpha,
            "val_f1_macro": round(float(best_log["eval_f1_macro"]), 4),
            "val_accuracy": round(float(best_log["eval_accuracy"]), 4),
            "val_f1_clickbait": round(float(best_log.get("eval_f1_clickbait", 0.0)), 4),
            "val_loss": round(float(best_log["eval_loss"]), 4),
        })

    results_sorted = sorted(results, key=lambda x: x["val_f1_macro"], reverse=True)
    best_combo = results_sorted[0]

    report_lines = [
        "# Knowledge Distillation (T, alpha) Hyperparameter Search Report",
        "",
        "*Optimization Metric: Validation Set Macro-F1 (Held-out Test set strictly untouched)*",
        "",
        "## Best Hyperparameter Setting",
        f"- **Optimal Temperature ($T^*$):** `{best_combo['temperature']}`",
        f"- **Optimal Alpha ($\\alpha^*$):** `{best_combo['alpha']}`",
        f"- **Best Validation Macro-F1:** **`{best_combo['val_f1_macro']:.4f}`**",
        f"- **Validation Accuracy:** `{best_combo['val_accuracy']:.4f}`",
        "",
        "## Full Grid Search Results (Validation Split)",
        "",
        "| Rank | Temperature ($T$) | Alpha ($\\alpha$) | Val Macro-F1 | Val Accuracy | Val Clickbait F1 | Val Loss |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, r in enumerate(results_sorted, 1):
        bold = "**" if rank == 1 else ""
        report_lines.append(
            f"| {rank} | {r['temperature']} | {r['alpha']} | {bold}{r['val_f1_macro']:.4f}{bold} | {r['val_accuracy']:.4f} | {r['val_f1_clickbait']:.4f} | {r['val_loss']:.4f} |"
        )

    pivot_df = pd.DataFrame(results).pivot(index="temperature", columns="alpha", values="val_f1_macro")
    report_lines += [
        "",
        "## Validation Macro-F1 Response Matrix ($T$ vs $\\alpha$)",
        "",
        pivot_df.to_markdown(),
        "",
    ]

    report_md = "\n".join(report_lines)
    out_dir = Path("experiments/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kd_grid_search_report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "kd_grid_results.json").write_text(
        json.dumps({"best_params": best_combo, "grid_results": results_sorted}, indent=2),
        encoding="utf-8",
    )

    import sys
    sys.stdout.buffer.write(report_md.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
