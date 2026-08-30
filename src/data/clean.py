"""
src/data/clean.py — Bangla-safe text cleaning for BanglaBait headlines.

Cleaning philosophy for Bangla (Bengali) text
----------------------------------------------
Bangla uses the Bengali script (Unicode block U+0980–U+09FF).  The script
has several combining characters that are essential for correct rendering:

  * **Hasanta / Virama (U+09BC)** — suppresses the inherent vowel to form
    consonant clusters. MUST NOT be stripped or words become unreadable.
  * **Anusvara (U+0982), Visarga (U+0983)** — nasal and aspiration markers.
  * **Chandrabindu (U+0981)** — nasalisation diacritic.
  * **Nukta (U+09BC)** — modifies consonants for loanword phonemes.

What we *do* strip:
  * Zero-width characters (ZWJ U+200D, ZWNJ U+200C, ZWSP U+200B, BOM U+FEFF)
  * Common invisible formatting characters (U+00AD soft-hyphen, U+FEFF BOM)
  * HTML entities and remnant markup (angle brackets, &amp; etc.)
  * Leading/trailing whitespace and collapsed internal runs of whitespace
  * Punctuation/symbols that are NOT Bangla-specific, using a Unicode-category
    whitelist approach (keep Bengali-script characters + common punctuation
    used in Bangla writing + ASCII alphanumerics for mixed titles)

What we preserve:
  * All Unicode category Ll/Lu/Lo characters (letters in any script)
  * All Bengali punctuation: দাঁড়ি (।), double danda (॥), etc.
  * Standard ASCII punctuation: !, ?, -, ',  " etc. (appear in headlines)
  * Digits (Bengali ০–৯ and ASCII 0–9)

Unicode normalization
---------------------
We apply **NFC** (Canonical Decomposition followed by Canonical Composition).
NFC is the correct form for text storage and retrieval in Bangla because it
produces composed forms (e.g., single code-point representation of characters
with diacritics) which tokenizers like BanglaBERT's SentencePiece expect.

Usage
-----
    from src.data.clean import clean_headline, clean_dataframe

    text = clean_headline("আমার  সোনার\u200c বাংলা")
    df = clean_dataframe(raw_df)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Character sets to strip ────────────────────────────────────────────────

# Zero-width / invisible formatting characters (safe to remove everywhere)
_ZERO_WIDTH = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060\u2061\u2062\u2063]"
)

# HTML entities and simple tags (some HF rows carry residual markup)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(?:[a-z]+|#\d+|#x[0-9a-f]+);", re.IGNORECASE)

# Whitespace normalisation (collapse any run of spaces/tabs/NBSP to single space)
_MULTI_SPACE = re.compile(r"[ \t\u00A0\u202F\u2009\u200A]+")

# Characters that are clearly junk in a headline context:
# control characters except newline (we strip newlines too) + private-use area
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B-\x1F\x7F\uE000-\uF8FF]")

# Allowed Unicode categories (whitelist approach for non-ASCII):
#   L  = letters (all scripts — Bengali Lo, Latin Lu/Ll, etc.)
#   N  = numbers (Nd covers ০–৯ and 0–9; No = other numbers)
#   P  = punctuation (Pc, Pd, Pe, Pf, Pi, Po, Ps)
#   Z  = separators (Zs = space separators; we normalise to ASCII space)
#   Mn = non-spacing marks (diacritics, Bengali matras/virama — KEEP)
#   Mc = spacing combining marks (Bengali dependent vowels — KEEP)
# We strip anything in category C (control), S (symbols, except some),
# but allow common ASCII symbols that appear in headlines.
_ALLOWED_EXTRA_ASCII = frozenset(
    "!?\"''-–—()[]{}/:;@#%&*+<>=\\|~`^_."
    # Note: < > are here but we strip HTML tags first, so residual bare
    # < > in non-tag contexts will survive (rare but valid e.g. comparisons).
)


def clean_headline(text: str) -> str:
    """Clean a single Bangla headline string.

    Steps (in order):
    1. Guard against non-string input.
    2. NFC Unicode normalisation.
    3. Strip HTML tags and entities.
    4. Remove zero-width / invisible characters.
    5. Remove control characters.
    6. Remove unwanted symbol characters (whitelist-based, preserves Bangla).
    7. Collapse whitespace; strip leading/trailing space.

    Args:
        text: Raw headline string.

    Returns:
        Cleaned string.  Empty string if input is null/blank.
    """
    if not isinstance(text, str):
        return ""

    # 1. NFC normalisation — idempotent; safe to apply repeatedly
    text = unicodedata.normalize("NFC", text)

    # 2. Strip HTML
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)

    # 3. Zero-width chars
    text = _ZERO_WIDTH.sub("", text)

    # 4. Control chars
    text = _CONTROL_CHARS.sub("", text)

    # 5. Symbol strip (whitelist approach)
    text = _strip_unwanted_symbols(text)

    # 6. Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text


def _strip_unwanted_symbols(text: str) -> str:
    """Remove symbol characters not in the allowed set.

    Keeps:
      - Any character whose Unicode general category starts with L (letter),
        N (number), M (mark/diacritic), P (punctuation), or Z (separator).
      - ASCII symbols in ``_ALLOWED_EXTRA_ASCII``.
    Removes:
      - Unicode general category S (symbol) characters not in the allowed list.
      - Miscellaneous Unicode symbols, emoji, box-drawing, etc.
    """
    result = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N", "M", "P", "Z"):
            result.append(ch)
        elif ch in _ALLOWED_EXTRA_ASCII:
            result.append(ch)
        elif ch.isspace():
            result.append(" ")
        # else: drop (symbol categories Sc, Sm, So, Sk that aren't whitelisted)
    return "".join(result)


def clean_dataframe(
    df: pd.DataFrame,
    headline_col: str = "headline",
    drop_empty: bool = True,
    min_char_length: int = 5,
) -> pd.DataFrame:
    """Apply ``clean_headline`` to every row of *df* and return cleaned copy.

    Args:
        df: DataFrame with at least a ``headline_col`` column.
        headline_col: Name of the text column.
        drop_empty: If True, drop rows whose headline is empty after cleaning.
        min_char_length: Rows shorter than this (chars) are treated as empty.

    Returns:
        New DataFrame with the headline column cleaned in-place.
        Adds a boolean column ``_was_cleaned`` that is True if the text changed.
    """
    df = df.copy()
    original = df[headline_col].copy()

    df[headline_col] = df[headline_col].apply(clean_headline)
    df["_was_cleaned"] = df[headline_col] != original

    n_changed = df["_was_cleaned"].sum()
    logger.info(
        "Cleaning complete: %d/%d rows modified (%.1f%%)",
        n_changed,
        len(df),
        100 * n_changed / max(len(df), 1),
    )

    if drop_empty:
        short_mask = df[headline_col].str.len() < min_char_length
        n_dropped = short_mask.sum()
        if n_dropped:
            logger.warning(
                "Dropping %d rows with headline shorter than %d chars after cleaning.",
                n_dropped,
                min_char_length,
            )
        df = df[~short_mask].reset_index(drop=True)

    return df


def assert_nfc(text: str) -> None:
    """Assert that *text* is already in NFC form (useful in tests)."""
    assert text == unicodedata.normalize("NFC", text), \
        f"Text is not NFC-normalised: {text!r}"
