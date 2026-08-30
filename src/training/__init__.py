# src/training/__init__.py
"""Training sub-package: datasets, trainer wrappers, and training pipelines."""

from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
from src.training.train_teacher import (
    WeightedTrainer,
    calculate_balanced_class_weights,
    train_teacher,
)

__all__ = [
    "BanglaHeadlineDataset",
    "prepare_datasets",
    "WeightedTrainer",
    "calculate_balanced_class_weights",
    "train_teacher",
]
