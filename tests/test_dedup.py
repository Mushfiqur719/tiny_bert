"""tests/test_dedup.py — Unit tests for MinHash near-duplicate clustering."""

from __future__ import annotations

import pytest
from src.data.dedup import compute_near_dup_groups, norm_text, get_shingles


def test_norm_text_preserves_bengali_characters():
    raw = "  ঢাকা, বাংলাদেশ! ১২৩   "
    norm = norm_text(raw)
    assert norm == "ঢাকা বাংলাদেশ ১২৩"


def test_identical_strings_group_together():
    texts = [
        "সাকিব আল হাসানের দুর্দান্ত শতক",
        "সাকিব আল হাসানের দুর্দান্ত শতক",
        "অন্য একটি সম্পূর্ণ ভিন্ন সংবাদ",
    ]
    groups = compute_near_dup_groups(texts, threshold=0.9)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_punctuation_only_differences_group_together():
    texts = [
        "ব্রেকিং নিউজ: ঢাকায় বড় দুর্ঘটনা!",
        "ব্রেকিং নিউজ ঢাকায় বড় দুর্ঘটনা?",
    ]
    groups = compute_near_dup_groups(texts, threshold=0.9)
    assert groups[0] == groups[1]


def test_different_texts_get_distinct_groups():
    texts = [
        "করোনাভাইরাস সংক্রমণ পরিস্থিতি",
        "শেয়ার বাজারে রেকর্ড দরপতন",
        "বিশ্বকাপ ফুটবল ফাইনাল ম্যাচ",
    ]
    groups = compute_near_dup_groups(texts, threshold=0.8)
    assert len(set(groups)) == 3
