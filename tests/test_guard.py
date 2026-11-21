"""tests/test_guard.py — Tests for test set access control and leakage guard."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import apply_splits
from src.utils.test_guard import TestSetLocked, assert_test_access


def test_guard_raises_when_env_var_missing(monkeypatch):
    """Guard must raise TestSetLocked if TBB_ALLOW_TEST is not set to '1'."""
    monkeypatch.delenv("TBB_ALLOW_TEST", raising=False)
    with pytest.raises(TestSetLocked) as exc_info:
        assert_test_access()
    assert "Test labels are locked" in str(exc_info.value)


def test_guard_raises_when_env_var_invalid(monkeypatch):
    """Guard must raise TestSetLocked if TBB_ALLOW_TEST has an invalid value."""
    monkeypatch.setenv("TBB_ALLOW_TEST", "0")
    with pytest.raises(TestSetLocked):
        assert_test_access()


def test_guard_passes_when_env_var_set(monkeypatch):
    """Guard must pass without error when TBB_ALLOW_TEST=1."""
    monkeypatch.setenv("TBB_ALLOW_TEST", "1")
    assert_test_access()  # Should not raise


def test_apply_splits_enforces_guard_on_test(monkeypatch):
    """apply_splits with split='test' must raise TestSetLocked if unauthorized."""
    monkeypatch.delenv("TBB_ALLOW_TEST", raising=False)
    df = pd.DataFrame({"headline": ["H1", "H2", "H3"], "label": [0, 1, 0]})
    splits = {
        "labeled": {
            "train": [0],
            "val": [1],
            "test": [2],
        }
    }
    # Train and val should work freely
    train_df = apply_splits(df, splits, split="train")
    assert len(train_df) == 1
    val_df = apply_splits(df, splits, split="val")
    assert len(val_df) == 1

    # Test split must raise
    with pytest.raises(TestSetLocked):
        apply_splits(df, splits, split="test")

    # When authorized, test split succeeds
    monkeypatch.setenv("TBB_ALLOW_TEST", "1")
    test_df = apply_splits(df, splits, split="test")
    assert len(test_df) == 1
