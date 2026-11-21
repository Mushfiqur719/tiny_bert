"""
scripts/evaluate_selective.py — CLI entry point for Stage 4 Selective Prediction (Risk–Coverage,
AURC) and Two-Tier Edge–Cloud Deferral Cascade evaluation.

Usage:
    .venv\\Scripts\\python.exe -m scripts.evaluate_selective
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.selective import (
    ensure_cached_logits,
    evaluate_edge_cloud_cascade,
    evaluate_selective_prediction,
    fit_temperature_on_val,
    plot_selective_and_cascade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate Selective Prediction and Edge-Cloud Cascade.")
    p.add_argument("--student-dir", default="experiments/student_small_pretrained_distilled")
    p.add_argument("--teacher-dir", default="experiments/teacher")
    p.add_argument("--output-dir", default="experiments")
    args = p.parse_args(argv)

    s_dir = Path(args.student_dir)
    t_dir = Path(args.teacher_dir)
    out_dir = Path(args.output_dir)

    s_val_logits, s_val_labels, s_test_logits, s_test_labels = ensure_cached_logits(s_dir)
    _, _, t_test_logits, _ = ensure_cached_logits(t_dir)

    # 1. Fit temperature strictly on validation split
    t_cal = fit_temperature_on_val(s_val_logits, s_val_labels)
    logger.info("Fitted Student Validation Temperature T_cal = %.4f", t_cal)

    # 2. Selective prediction curve
    sel_df, sel_summary = evaluate_selective_prediction(
        test_logits=s_test_logits,
        test_labels=s_test_labels,
        temperature=t_cal,
    )
    sel_df.to_csv(out_dir / "selective_prediction_table.csv", index=False)

    # 3. Edge-Cloud cascade table
    cas_df = evaluate_edge_cloud_cascade(
        student_test_logits=s_test_logits,
        teacher_test_logits=t_test_logits,
        test_labels=s_test_labels,
        student_temp=t_cal,
    )
    cas_df.to_csv(out_dir / "edge_cloud_cascade_table.csv", index=False)

    # 4. Plot Fig 7
    fig_path = plot_selective_and_cascade(sel_df, cas_df, out_dir / "figures" / "fig7_selective_accuracy_coverage.png")
    logger.info("Saved selective prediction figure to %s", fig_path)

    # 5. Write Markdown & JSON report
    (out_dir / "selective_metrics.json").write_text(
        json.dumps(
            {
                "summary": sel_summary,
                "selective_table": sel_df.to_dict(orient="records"),
                "cascade_table": cas_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# Selective Prediction (Risk–Coverage / AURC) & Edge–Cloud Deferral Report",
        "",
        f"- **Validation-Fitted Calibration Temperature ($T_{{\\text{{cal}}}}$):** `{t_cal:.4f}`",
        f"- **Area Under Risk–Coverage Curve (AURC):** **`{sel_summary['aurc']:.5f}`** *(lower is better)*",
        f"- **Coverage at $\\ge 90\\%$ Selective Accuracy:** **`{sel_summary['max_coverage_at_90pct_acc']:.1f}%`**",
        f"- **Coverage at $\\ge 92\\%$ Selective Accuracy:** **`{sel_summary['max_coverage_at_92pct_acc']:.1f}%`**",
        f"- **Coverage at $\\ge 95\\%$ Selective Accuracy:** **`{sel_summary['max_coverage_at_95pct_acc']:.1f}%`**",
        f"- **Peak High-Confidence Selective Accuracy:** **`{sel_summary['peak_selective_acc_pct']:.2f}%`**",
        "",
        "## 1. Selective Accuracy & Macro-F1 Across Coverage Thresholds (TinyBanglaBERT-4L)",
        "",
        sel_df.to_markdown(index=False),
        "",
        "## 2. Two-Tier Edge–Cloud Deferral Cascade (`TinyBanglaBERT-4L INT8` 4.00 ms $\\to$ `Teacher`)",
        "",
        cas_df.to_markdown(index=False),
        "",
    ]
    report_md = "\n".join(md_lines)
    (out_dir / "selective_prediction_report.md").write_text(report_md, encoding="utf-8")
    print(report_md)


if __name__ == "__main__":
    main()
