"""src/data/dedup.py — MinHash LSH near-duplicate detection and group clustering."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Sequence

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


def norm_text(text: str) -> str:
    """Normalize text with Unicode NFC while retaining Bengali characters, letters, and numbers."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    # Replace non-word and non-Bengali characters with space, preserving combining characters
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch.isalnum() or cat.startswith(("L", "N", "M")):
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    s = "".join(cleaned).lower()
    return re.sub(r"\s+", " ", s).strip()


def get_shingles(text: str, k: int = 3) -> set[str]:
    """Extract character k-shingles from normalized text."""
    s = norm_text(text)
    if len(s) <= k:
        return {s} if s else {""}
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def compute_near_dup_groups(
    texts: Sequence[str],
    threshold: float = 0.9,
    num_perm: int = 128,
) -> list[int]:
    """Cluster texts into near-duplicate connected components using MinHash LSH.

    Args:
        texts: List of input strings.
        threshold: Jaccard similarity threshold for clustering (default: 0.9).
        num_perm: Number of permutations for MinHash.

    Returns:
        List of cluster IDs corresponding to each input text.
    """
    logger.info("Computing MinHash LSH near-duplicate groups for %d texts...", len(texts))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []

    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for shingle in get_shingles(t, k=3):
            m.update(shingle.encode("utf-8"))
        lsh.insert(str(i), m)
        minhashes.append(m)

    # Disjoint set union (Union-Find)
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, m in enumerate(minhashes):
        matches = lsh.query(m)
        for match_idx_str in matches:
            j = int(match_idx_str)
            union(i, j)

    groups = [find(i) for i in range(len(texts))]
    num_unique = len(set(groups))
    logger.info("MinHash clustering complete: %d unique groups from %d texts.", num_unique, len(texts))
    return groups
