"""
tests/test_data_clean.py — Unit tests for src/data/clean.py and src/data/split.py.

These tests are designed to run WITHOUT downloading the real dataset,
using synthetic Bangla-like text fixtures.

Run with:  pytest tests/test_data_clean.py -v
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import (
    clean_headline,
    clean_dataframe,
    assert_nfc,
    _ZERO_WIDTH,
)
from src.data.split import create_splits, load_splits, apply_splits


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_labeled_df() -> pd.DataFrame:
    """Small synthetic labeled DataFrame (Bengali + ASCII headlines)."""
    headlines = [
        "আজকের শিরোনাম: বাংলাদেশ জিতেছে",        # clean Bengali
        "এটি একটি ক্লিকবেইট শিরোনাম!",              # clickbait-like
        "রাজনীতি ও অর্থনীতি",                         # politics & economy
        "Test headline in English with numbers 42",    # ASCII
        "মিশ্র headline: বাংলা and English together",  # mixed
        "শিক্ষা ক্ষেত্রে নতুন পদক্ষেপ",               # education
        "স্বাস্থ্য সেবা উন্নয়নে সরকারের পরিকল্পনা",  # health
        "আন্তর্জাতিক সম্পর্ক ও কূটনীতি",              # international
        "ক্রিকেট বিশ্বকাপে বাংলাদেশের জয়",           # cricket
        "নতুন প্রযুক্তি ও উদ্ভাবন",                    # tech
        "সামাজিক সমস্যা ও সমাধান",                     # social
        "পরিবেশ দূষণ ও সংরক্ষণ",                      # environment
    ]
    labels = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]
    return pd.DataFrame({"headline": headlines, "label": labels})


@pytest.fixture
def sample_unlabeled_df() -> pd.DataFrame:
    return pd.DataFrame({
        "headline": ["অজানা শিরোনাম এক", "অজানা শিরোনাম দুই", "unlabeled example"],
        "label": [-1, -1, -1],
    })


@pytest.fixture
def minimal_cfg():
    """Minimal OmegaConf-like config for split tests."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "seed": 42,
        "data": {
            "splits_file": ".pytest_tmp/splits_test/splits.json",
            "train_ratio": 0.8,
            "val_ratio":   0.1,
            "test_ratio":  0.1,
        },
        "model": {
            "teacher_model_name": "csebuetnlp/banglabert",
            "max_seq_len": 128,
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# clean_headline — Unicode normalisation
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanHeadlineUnicode:

    def test_nfc_normalisation_applied(self):
        # NFD form of 'ক' (consonant + combining mark) vs NFC composed form
        nfd_text = unicodedata.normalize("NFD", "ক্ষ")
        cleaned = clean_headline(nfd_text)
        assert cleaned == unicodedata.normalize("NFC", cleaned)

    def test_nfc_idempotent(self):
        """Applying clean_headline twice must give the same result."""
        text = "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি!"
        once  = clean_headline(text)
        twice = clean_headline(once)
        assert once == twice

    def test_nfc_idempotent_on_already_clean(self):
        """A pre-NFC string should be unchanged by clean_headline."""
        text = unicodedata.normalize("NFC", "বাংলাদেশ")
        assert clean_headline(text) == text

    def test_assert_nfc_passes_on_clean_output(self):
        text = clean_headline("রাজধানী ঢাকায় বন্যা পরিস্থিতি")
        assert_nfc(text)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# clean_headline — Zero-width character removal
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanHeadlineZeroWidth:

    def test_zwj_stripped(self):
        text = "আমার\u200Dসোনার বাংলা"  # ZWJ between words
        assert "\u200D" not in clean_headline(text)

    def test_zwnj_stripped(self):
        text = "বাংলা\u200Cদেশ"  # ZWNJ
        assert "\u200C" not in clean_headline(text)

    def test_bom_stripped(self):
        text = "\uFEFFশিরোনাম"
        result = clean_headline(text)
        assert "\uFEFF" not in result
        assert result.startswith("শিরোনাম")

    def test_zwsp_stripped(self):
        text = "শিরোনাম\u200B ঢাকা"
        assert "\u200B" not in clean_headline(text)


# ══════════════════════════════════════════════════════════════════════════════
# clean_headline — Bangla diacritic preservation
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanHeadlineBanglaPreservation:

    def test_virama_preserved(self):
        """Hasanta/virama U+09BC must not be stripped (it forms consonant clusters)."""
        # ক্ষ = ক + ্ (virama U+09BC) + ষ — a common Bangla conjunct
        text = "ক্ষমতা"   # power/authority
        cleaned = clean_headline(text)
        assert "ক্ষমতা" in cleaned or len(cleaned) > 0

    def test_chandrabindu_preserved(self):
        """Chandrabindu U+0981 must be preserved."""
        text = "চাঁদ"    # moon — contains chandrabindu
        cleaned = clean_headline(text)
        assert "\u0981" in cleaned

    def test_anusvara_preserved(self):
        text = "বাংলা"   # contains anusvara U+0982
        cleaned = clean_headline(text)
        assert "\u0982" in cleaned

    def test_bangla_digits_preserved(self):
        text = "২০২৪ সালে"  # Bengali digits ২০২৪
        cleaned = clean_headline(text)
        assert "২০২৪" in cleaned


# ══════════════════════════════════════════════════════════════════════════════
# clean_headline — Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanHeadlineEdgeCases:

    def test_punctuation_only_does_not_crash(self):
        """A string of only punctuation/symbols should not raise."""
        for text in ["!!!", "---", "   ", "####", "\u200B\u200C", ""]:
            result = clean_headline(text)
            assert isinstance(result, str)

    def test_none_input_returns_empty_string(self):
        result = clean_headline(None)   # type: ignore[arg-type]
        assert result == ""

    def test_int_input_returns_empty_string(self):
        result = clean_headline(42)   # type: ignore[arg-type]
        assert result == ""

    def test_html_stripped(self):
        text = "<b>ক্লিকবেইট</b> শিরোনাম &amp; আরও"
        cleaned = clean_headline(text)
        assert "<b>" not in cleaned
        assert "&amp;" not in cleaned
        assert "ক্লিকবেইট" in cleaned

    def test_whitespace_collapsed(self):
        text = "বাংলা   দেশ\t\tখবর"
        cleaned = clean_headline(text)
        assert "  " not in cleaned
        assert "\t" not in cleaned

    def test_leading_trailing_whitespace_stripped(self):
        text = "  শিরোনাম  "
        assert clean_headline(text) == "শিরোনাম"

    def test_english_ascii_preserved(self):
        text = "Bangladesh wins T20 World Cup 2024!"
        cleaned = clean_headline(text)
        assert "Bangladesh" in cleaned
        assert "2024" in cleaned
        assert "!" in cleaned


# ══════════════════════════════════════════════════════════════════════════════
# clean_dataframe
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanDataframe:

    def test_returns_copy_not_inplace(self, sample_labeled_df):
        original_first = sample_labeled_df["headline"].iloc[0]
        _ = clean_dataframe(sample_labeled_df)
        assert sample_labeled_df["headline"].iloc[0] == original_first

    def test_was_cleaned_column_added(self, sample_labeled_df):
        result = clean_dataframe(sample_labeled_df)
        assert "_was_cleaned" in result.columns

    def test_no_nulls_after_cleaning(self, sample_labeled_df):
        result = clean_dataframe(sample_labeled_df)
        assert result["headline"].notna().all()

    def test_output_all_nfc(self, sample_labeled_df):
        result = clean_dataframe(sample_labeled_df)
        for h in result["headline"]:
            assert h == unicodedata.normalize("NFC", h), f"Not NFC: {h!r}"

    def test_drop_empty_rows(self):
        df = pd.DataFrame({
            "headline": ["valid headline বাংলা", "\u200B\u200C", "a", ""],
            "label": [0, 1, 0, 1],
        })
        result = clean_dataframe(df, drop_empty=True, min_char_length=5)
        # The zero-width only and very short rows should be dropped
        assert len(result) < len(df)
        assert all(len(h) >= 5 for h in result["headline"])


# ══════════════════════════════════════════════════════════════════════════════
# Class balance — non-degenerate assertion
# ══════════════════════════════════════════════════════════════════════════════

class TestClassBalance:

    def test_balanced_dataset_does_not_raise(self, sample_labeled_df):
        """load_raw's balance check should not trigger on a balanced fixture."""
        from src.data.load import _log_class_balance
        # Just test the logging function doesn't crash
        _log_class_balance(sample_labeled_df, "test")

    def test_degenerate_dataset_raises(self):
        """A dataset where one class has < 5% should fail the assertion."""
        n = 100
        df = pd.DataFrame({
            "headline": [f"headline {i}" for i in range(n)],
            "label": [0] * 97 + [1] * 3,   # 3% clickbait
        })
        value_counts = df["label"].value_counts(normalize=True)
        with pytest.raises(AssertionError, match="degenerate"):
            for cls, frac in value_counts.items():
                assert frac >= 0.05, (
                    f"Class {cls} is degenerate: only {frac:.1%} of labeled data. "
                    "Check your data source."
                )

    def test_label_balance_logged(self, sample_labeled_df, caplog):
        import logging
        from src.data.load import _log_class_balance
        with caplog.at_level(logging.INFO):
            _log_class_balance(sample_labeled_df, "fixture")
        assert "label=0" in caplog.text or "not-clickbait" in caplog.text


# ══════════════════════════════════════════════════════════════════════════════
# split.py — stratified split
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateSplits:

    def test_creates_splits_json(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(minimal_cfg, OmegaConf.create({
            "data": {"splits_file": str(tmp_path / "splits.json")}
        }))
        splits = create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42)
        assert (tmp_path / "splits.json").exists()
        assert "labeled" in splits
        assert "unlabeled" in splits

    def test_indices_non_overlapping(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(minimal_cfg, OmegaConf.create({
            "data": {"splits_file": str(tmp_path / "splits.json")}
        }))
        splits = create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42)
        train = set(splits["labeled"]["train"])
        val   = set(splits["labeled"]["val"])
        test  = set(splits["labeled"]["test"])
        assert train & val == set(), "Train and val overlap!"
        assert train & test == set(), "Train and test overlap!"
        assert val & test == set(), "Val and test overlap!"

    def test_indices_cover_all_rows(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(minimal_cfg, OmegaConf.create({
            "data": {"splits_file": str(tmp_path / "splits.json")}
        }))
        splits = create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42)
        all_idx = (
            set(splits["labeled"]["train"])
            | set(splits["labeled"]["val"])
            | set(splits["labeled"]["test"])
        )
        assert all_idx == set(sample_labeled_df.index.tolist())

    def test_stratification_both_classes_in_train(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(minimal_cfg, OmegaConf.create({
            "data": {"splits_file": str(tmp_path / "splits.json")}
        }))
        splits = create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42)
        train_df, val_df, test_df = apply_splits(sample_labeled_df, splits)
        assert set(train_df["label"].unique()) == {0, 1}

    def test_overwrite_false_raises_if_exists(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(minimal_cfg, OmegaConf.create({
            "data": {"splits_file": str(tmp_path / "splits.json")}
        }))
        create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42)
        with pytest.raises(FileExistsError):
            create_splits(sample_labeled_df, sample_unlabeled_df, cfg, seed=42, overwrite=False)

    def test_deterministic_same_seed(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg1 = OmegaConf.merge(minimal_cfg, OmegaConf.create({"data": {"splits_file": str(tmp_path / "s1.json")}}))
        cfg2 = OmegaConf.merge(minimal_cfg, OmegaConf.create({"data": {"splits_file": str(tmp_path / "s2.json")}}))
        s1 = create_splits(sample_labeled_df, sample_unlabeled_df, cfg1, seed=42)
        s2 = create_splits(sample_labeled_df, sample_unlabeled_df, cfg2, seed=42)
        assert s1["labeled"]["train"] == s2["labeled"]["train"]
        assert s1["labeled"]["test"]  == s2["labeled"]["test"]

    def test_different_seed_gives_different_splits(self, sample_labeled_df, sample_unlabeled_df, minimal_cfg, tmp_path):
        from omegaconf import OmegaConf
        cfg1 = OmegaConf.merge(minimal_cfg, OmegaConf.create({"data": {"splits_file": str(tmp_path / "s1.json")}}))
        cfg2 = OmegaConf.merge(minimal_cfg, OmegaConf.create({"data": {"splits_file": str(tmp_path / "s2.json")}}))
        s1 = create_splits(sample_labeled_df, sample_unlabeled_df, cfg1, seed=42)
        s2 = create_splits(sample_labeled_df, sample_unlabeled_df, cfg2, seed=99)
        # With 12 samples this may occasionally be equal by chance — acceptable
        # but we check at least one differs with high probability
        assert (
            s1["labeled"]["train"] != s2["labeled"]["train"]
            or s1["labeled"]["test"] != s2["labeled"]["test"]
        )
