# src/models/__init__.py
"""Model sub-package: teacher, student, and distillation definitions."""

from src.models.distillation import (
    DistillationDataset,
    KnowledgeDistillationLoss,
    load_teacher_soft_labels,
)
from src.models.student import (
    count_parameters,
    create_student_config,
    init_from_teacher,
    init_random,
    save_student_checkpoint,
)
from src.models.teacher import (
    freeze_model,
    get_teacher_tokenizer,
    load_teacher_model,
    save_teacher_checkpoint,
    unfreeze_model,
)

__all__ = [
    "load_teacher_model",
    "get_teacher_tokenizer",
    "freeze_model",
    "unfreeze_model",
    "save_teacher_checkpoint",
    "create_student_config",
    "init_random",
    "init_from_teacher",
    "count_parameters",
    "save_student_checkpoint",
    "KnowledgeDistillationLoss",
    "DistillationDataset",
    "load_teacher_soft_labels",
]
