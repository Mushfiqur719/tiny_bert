"""
src/data/report.py — Generate data_report.md after the pipeline runs.

Writes ``data/processed/data_report.md`` with:
  - Class balance table
  - Headline character- and token-length distribution (using BanglaBERT tokenizer)
  - 5 example headlines per class (after cleaning)

Usage
-----
    from src.data.report import generate_report
    generate_report(labeled_df, splits, cfg)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_LABEL_NAMES = {0: "not-clickbait", 1: "clickbait"}


def generate_report(
    labeled_df: pd.DataFrame,
    splits: dict,
    cfg,
    tokenizer_name: Optional[str] = None,
    n_examples: int = 5,
    seed: int = 42,
) -> str:
    """Generate a Markdown data report and write it to disk.

    Args:
        labeled_df: Full cleaned + labeled DataFrame.
        splits: Dict from ``create_splits`` / ``load_splits``.
        cfg: OmegaConf config.
        tokenizer_name: HF model name used for token-length stats.
            Defaults to ``cfg.model.teacher_model_name``.
        n_examples: Number of example headlines per class.
        seed: Random seed for sampling examples.

    Returns:
        The Markdown string (also written to disk).
    """
    from src.data.split import apply_splits

    tokenizer_name = tokenizer_name or cfg.model.teacher_model_name
    output_path = Path(cfg.data.processed_dir) / "data_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = apply_splits(labeled_df, splits)

    lines = [
        "# BanglaBait Data Report",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Seed: {splits['metadata']['seed']}*",
        "",
    ]

    # ── 1. Dataset overview ───────────────────────────────────────────────
    lines += [
        "## 1. Dataset Overview",
        "",
        f"| Split   | Rows   |",
        f"|---------|--------|",
        f"| Train   | {len(train_df):,} |",
        f"| Val     | {len(val_df):,} |",
        f"| Test    | {len(test_df):,} |",
        f"| **Total labeled** | **{len(labeled_df):,}** |",
        f"| Unlabeled (pooled) | {splits['metadata']['total_unlabeled']:,} |",
        "",
    ]

    # ── 2. Class balance ─────────────────────────────────────────────────
    lines += ["## 2. Class Balance", ""]
    for split_name, split_df in [("train", train_df), ("val", val_df),
                                  ("test", test_df), ("all labeled", labeled_df)]:
        counts = split_df["label"].value_counts().sort_index()
        total = len(split_df)
        rows = [f"### {split_name.title()} (n={total:,})", ""]
        rows += ["| Label | Class | Count | % |", "|-------|-------|-------|---|"]
        for cls, cnt in counts.items():
            name = _LABEL_NAMES.get(int(cls), str(cls))
            rows.append(f"| {cls} | {name} | {cnt:,} | {cnt / total:.1%} |")
        rows.append("")
        lines += rows

    # ── 3. Headline length distribution ──────────────────────────────────
    lines += ["## 3. Headline Length Distribution", ""]

    # Character lengths
    char_len = labeled_df["headline"].str.len()
    lines += _describe_series(char_len, "Characters (chars)")

    # Token lengths via tokenizer
    lines += _token_length_section(labeled_df, tokenizer_name, cfg)

    # ── 4. Example headlines ─────────────────────────────────────────────
    lines += ["## 4. Example Headlines (after cleaning)", ""]
    for cls in sorted(_LABEL_NAMES.keys()):
        name = _LABEL_NAMES[cls]
        subset = labeled_df[labeled_df["label"] == cls]
        examples = subset["headline"].sample(
            min(n_examples, len(subset)), random_state=seed
        ).tolist()
        lines.append(f"### Class {cls}: {name}")
        lines.append("")
        for i, h in enumerate(examples, 1):
            lines.append(f"{i}. {h}")
        lines.append("")

    report_md = "\n".join(lines)
    output_path.write_text(report_md, encoding="utf-8")
    logger.info("Data report written to %s", output_path)
    return report_md


def _describe_series(s: pd.Series, label: str) -> list[str]:
    desc = s.describe()
    return [
        f"### {label}",
        "",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| min  | {desc['min']:.0f} |",
        f"| p25  | {desc['25%']:.0f} |",
        f"| median | {desc['50%']:.0f} |",
        f"| mean | {desc['mean']:.1f} |",
        f"| p75  | {desc['75%']:.0f} |",
        f"| p95  | {s.quantile(0.95):.0f} |",
        f"| max  | {desc['max']:.0f} |",
        "",
    ]


def _token_length_section(
    df: pd.DataFrame,
    tokenizer_name: str,
    cfg,
) -> list[str]:
    """Compute BPE token counts using the teacher tokenizer."""
    try:
        from transformers import AutoTokenizer
        logger.info("Loading tokenizer '%s' for token-length stats...", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        max_len = int(cfg.model.max_seq_len)

        # Sample up to 2000 rows for speed
        sample = df["headline"].sample(min(2000, len(df)), random_state=42)
        token_lens = sample.apply(
            lambda t: len(tokenizer.encode(t, add_special_tokens=True,
                                           truncation=False))
        )

        pct_truncated = (token_lens > max_len).mean()
        lines = _describe_series(token_lens, f"Tokens ({tokenizer_name.split('/')[-1]})")
        lines.insert(
            -1,
            f"| >max_seq_len ({max_len}) | {pct_truncated:.1%} of sample |",
        )
        return lines
    except Exception as e:
        logger.warning("Could not compute token lengths: %s", e)
        return [f"*Token-length stats skipped: {e}*", ""]
