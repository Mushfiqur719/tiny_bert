"""
src/models/teacher.py — Teacher Model (BanglaBERT) sequence classification wrapper.

Loads and configures BanglaBERT (csebuetnlp/banglabert) for binary clickbait detection,
provides model freezing/unfreezing helpers, and checkpoint management.

Usage:
    from src.models.teacher import load_teacher_model, get_teacher_tokenizer, freeze_model
    model = load_teacher_model("csebuetnlp/banglabert", num_labels=2)
    tokenizer = get_teacher_tokenizer("csebuetnlp/banglabert")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_NAME = "csebuetnlp/banglabert"
ID2LABEL = {0: "not-clickbait", 1: "clickbait"}
LABEL2ID = {"not-clickbait": 0, "clickbait": 1}


def get_teacher_tokenizer(
    model_name_or_path: str = DEFAULT_TEACHER_NAME,
) -> PreTrainedTokenizerBase:
    """Load tokenizer for the teacher model.

    Args:
        model_name_or_path: HuggingFace model hub ID or local checkpoint directory.

    Returns:
        HuggingFace Tokenizer instance.
    """
    logger.info("Loading teacher tokenizer from '%s'...", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def load_teacher_model(
    model_name_or_path: str = DEFAULT_TEACHER_NAME,
    num_labels: int = 2,
    dropout: Optional[float] = 0.1,
) -> PreTrainedModel:
    """Load BanglaBERT sequence classification model.

    Args:
        model_name_or_path: HuggingFace model hub ID or path to trained checkpoint.
        num_labels: Number of classification classes (default: 2).
        dropout: Classifier hidden dropout probability (if supported by config).

    Returns:
        AutoModelForSequenceClassification instance.
    """
    logger.info("Loading teacher model from '%s' (num_labels=%d)...", model_name_or_path, num_labels)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if dropout is not None:
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    return model


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all parameters in a model (requires_grad = False).

    Args:
        model: PyTorch Module or Transformer PreTrainedModel.

    Returns:
        The frozen model.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    logger.info("Model parameters frozen (requires_grad=False).")
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters in a model (requires_grad = True).

    Args:
        model: PyTorch Module or Transformer PreTrainedModel.

    Returns:
        The unfrozen model.
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Model parameters unfrozen (requires_grad=True).")
    return model


def save_teacher_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
) -> Path:
    """Save model weights, configuration, and tokenizer to output directory.

    Args:
        model: Trained teacher model.
        tokenizer: Associated tokenizer.
        output_dir: Destination directory.

    Returns:
        Path to output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved teacher model and tokenizer to '%s'", output_path)
    return output_path
