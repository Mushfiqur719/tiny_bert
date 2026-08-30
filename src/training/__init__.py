# src/training/__init__.py
"""Training sub-package: datasets, trainer wrappers, and training pipelines."""

from src.training.dataset import BanglaHeadlineDataset, prepare_datasets
from src.training.train_baseline import train_student_baseline
from src.training.train_distill import (
    DistillationTrainer,
    train_student_distill,
)
from src.training.train_teacher import (
    MetricLoggingCallback,
    WeightedTrainer,
    calculate_balanced_class_weights,
    train_teacher,
)
from src.training.tune_distill import (
    evaluate_kd_hyperparameter_pair,
    run_grid_search_kd,
)

__all__ = [
    "BanglaHeadlineDataset",
    "prepare_datasets",
    "WeightedTrainer",
    "DistillationTrainer",
    "MetricLoggingCallback",
    "calculate_balanced_class_weights",
    "train_teacher",
    "train_student_baseline",
    "train_student_distill",
    "evaluate_kd_hyperparameter_pair",
    "run_grid_search_kd",
]
