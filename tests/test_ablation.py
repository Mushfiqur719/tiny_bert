"""
tests/test_ablation.py — Unit tests for model ablation comparison and report generation.

Run with:  pytest tests/test_ablation.py -v
"""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from src.evaluation.ablation import (
    load_run_metrics,
    compile_ablation_dataframe,
    plot_f1_comparison,
    run_ablation_analysis,
)


class TestAblation:

    def test_load_teacher_run_metrics(self):
        m = load_run_metrics("experiments/teacher")
        assert "test" in m
        assert "val_f1_macro" in m
        assert m["test"]["accuracy"] == 0.8420
        assert m["test"]["f1_macro"] == 0.8233

    def test_compile_ablation_dataframe_has_all_five_models(self):
        df = compile_ablation_dataframe()
        assert len(df) == 5
        assert "Teacher (BanglaBERT Baseline)" in df["Model / Configuration"].values
        assert "Student: Random Init + Supervised" in df["Model / Configuration"].values
        assert "Student: Teacher Init + Distilled" in df["Model / Configuration"].values
        assert "Test Macro F1" in df.columns
        assert "Confusion Matrix (TN/FP/FN/TP)" in df.columns

    def test_plot_f1_comparison_creates_png(self, tmp_path):
        df = compile_ablation_dataframe()
        out_png = tmp_path / "test_f1_plot.png"
        plot_f1_comparison(df, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000

    def test_run_ablation_analysis_end_to_end(self):
        df, best_id = run_ablation_analysis(output_dir="experiments")
        assert len(df) == 5
        assert best_id in ("student_teacher_init_distilled", "student_teacher_init_baseline")

        # Verify destination files
        assert Path("experiments/ablation_table.csv").exists()
        assert Path("experiments/ablation_summary.md").exists()
        assert Path("experiments/ablation_f1_comparison.png").exists()
        assert Path("experiments/tinybanglabert_final").exists()
        assert Path("experiments/tinybanglabert_final/model.safetensors").exists()
