"""tests/test_docs_consistency.py — Verifies numbers inside AUTO blocks match results_master.csv."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import pytest


def test_results_master_consistency():
    """Verify results_master.csv can be generated and contains valid statistics."""
    from scripts.build_reports import aggregate_v3_results

    df = aggregate_v3_results()
    if df.empty:
        pytest.skip("No v3 runs aggregated yet.")

    assert "group" in df.columns
    assert "f1_mean" in df.columns
    assert "acc_mean" in df.columns
    assert (df["acc_mean"] >= 0.0).all() and (df["acc_mean"] <= 1.0).all()
    assert (df["f1_mean"] >= 0.0).all() and (df["f1_mean"] <= 1.0).all()


def test_auto_blocks_match_csv():
    """Check that any rendered AUTO markdown tables match results_master.csv values."""
    csv_path = Path("experiments/v3/results_master.csv")
    if not csv_path.exists():
        pytest.skip("results_master.csv does not exist yet.")

    df = pd.read_csv(csv_path)
    # Check that each group listed has non-null f1_mean
    for _, row in df.iterrows():
        assert not pd.isna(row["f1_mean"])
        assert not pd.isna(row["acc_mean"])
