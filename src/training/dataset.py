"""
src/training/dataset.py — PyTorch Dataset for Bangla clickbait headlines.

Provides:
  - `BanglaHeadlineDataset`: tokenizes headlines and wraps inputs + labels for PyTorch / HuggingFace Trainer.
  - `prepare_datasets`: creates train, validation, and test datasets from DataFrames.

Usage:
    from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
    train_ds, val_ds, test_ds = prepare_datasets(train_df, val_df, test_df, tokenizer, max_seq_len=128)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class BanglaHeadlineDataset(Dataset):
    """PyTorch Dataset for text classification on Bangla headlines."""

    def __init__(
        self,
        headlines: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_seq_len: int = 128,
    ) -> None:
        """
        Args:
            headlines: List of headline strings.
            labels: List of integer class labels (0 or 1), or None for inference.
            tokenizer: HuggingFace Tokenizer.
            max_seq_len: Maximum sequence length for truncation and padding.
        """
        if tokenizer is None:
            raise ValueError("A HuggingFace tokenizer must be provided.")

        self.headlines = list(headlines)
        self.labels = [int(l) for l in labels] if labels is not None else None
        self.max_seq_len = max_seq_len

        # Pre-tokenize all headlines for efficient batching and training
        self.encodings = tokenizer(
            self.headlines,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.headlines)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 128,
    text_col: str = "headline",
    label_col: str = "label",
) -> tuple[BanglaHeadlineDataset, BanglaHeadlineDataset, Optional[BanglaHeadlineDataset]]:
    """Create train, validation, and test datasets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Optional Test DataFrame.
        tokenizer: HuggingFace Tokenizer.
        max_seq_len: Max sequence length (default: 128).
        text_col: Column name containing text.
        label_col: Column name containing labels.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    logger.info("Creating tokenized datasets (max_seq_len=%d)...", max_seq_len)

    train_dataset = BanglaHeadlineDataset(
        headlines=train_df[text_col].tolist(),
        labels=train_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    val_dataset = BanglaHeadlineDataset(
        headlines=val_df[text_col].tolist(),
        labels=val_df[label_col].tolist(),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    test_dataset = None
    if test_df is not None:
        test_dataset = BanglaHeadlineDataset(
            headlines=test_df[text_col].tolist(),
            labels=test_df[label_col].tolist(),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

    logger.info(
        "Prepared datasets: Train=%d, Val=%d, Test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset) if test_dataset is not None else 0,
    )

    return train_dataset, val_dataset, test_dataset
