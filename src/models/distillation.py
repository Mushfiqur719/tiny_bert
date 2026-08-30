"""
src/models/distillation.py — Knowledge Distillation loss and utilities.

Standard Hinton Knowledge Distillation Loss:
    L = alpha * CE(student_logits, true_labels) + (1 - alpha) * (T^2) * KL(student_probs_T, teacher_probs_T)

Where:
    - CE is weighted/unweighted Cross Entropy loss on hard ground-truth labels.
    - KL is Kullback-Leibler divergence between temperature-softened student and teacher probability distributions.
    - T is the distillation temperature (smooths the teacher logit distribution).
    - alpha balances the relative contribution of hard ground-truth vs soft teacher supervision.
    - T^2 scaling factor ensures the gradient magnitudes of the soft loss match the hard cross-entropy loss.

Precomputed Soft Labels:
    Pre-computed teacher logits from `experiments/teacher/train_soft_labels.pt` are loaded and
    aligned with the training dataset to ensure exact, reproducible, and fast student distillation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """Computes Hinton Knowledge Distillation Loss combining CrossEntropy and KL Divergence."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            temperature: Distillation temperature T (T >= 1.0).
            alpha: Weight for hard cross-entropy loss (0.0 <= alpha <= 1.0).
                   Soft KD loss weight is (1.0 - alpha).
            class_weights: Optional 1D tensor of class weights for CrossEntropy loss.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.class_weights = class_weights
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined distillation loss.

        Args:
            student_logits: Logits from student model [batch_size, num_classes].
            teacher_logits: Logits from teacher model [batch_size, num_classes].
            labels: Ground truth class indices [batch_size].

        Returns:
            Tuple of (total_loss_tensor, loss_breakdown_dict).
        """
        T = self.temperature

        # 1. Soft loss: KL divergence on temperature-softened probabilities
        # student input to KLDivLoss must be log-probabilities
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # teacher target to KLDivLoss must be probabilities
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # Scale by T^2 as per standard distillation literature
        soft_loss = self.kl_div(student_log_probs, teacher_probs) * (T * T)

        # 2. Hard loss: CrossEntropy on ground truth labels
        if labels is not None and self.alpha > 0.0:
            weights = self.class_weights.to(student_logits.device) if self.class_weights is not None else None
            hard_loss = F.cross_entropy(student_logits, labels, weight=weights)
            total_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)
        else:
            hard_loss = torch.tensor(0.0, device=student_logits.device)
            total_loss = soft_loss

        breakdown = {
            "total_loss": float(total_loss.item()),
            "hard_loss": float(hard_loss.item()) if labels is not None else 0.0,
            "soft_loss": float(soft_loss.item()),
            "temperature": T,
            "alpha": self.alpha,
        }

        return total_loss, breakdown


class DistillationDataset(Dataset):
    """PyTorch Dataset that returns tokenized inputs, ground-truth labels, and precomputed teacher logits."""

    def __init__(
        self,
        headlines: list[str],
        labels: list[int],
        teacher_logits: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 128,
    ) -> None:
        """
        Args:
            headlines: List of text headlines.
            labels: List of ground-truth integer labels.
            teacher_logits: Pre-computed teacher logits [N, num_classes].
            tokenizer: Tokenizer.
            max_seq_len: Maximum token sequence length.
        """
        if len(headlines) != len(labels) or len(headlines) != len(teacher_logits):
            raise ValueError(
                f"Length mismatch: {len(headlines)} headlines, {len(labels)} labels, {len(teacher_logits)} teacher logits"
            )

        self.headlines = headlines
        self.labels = [int(l) for l in labels]
        self.teacher_logits = teacher_logits.clone().detach().float()
        self.max_seq_len = max_seq_len

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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["teacher_logits"] = self.teacher_logits[idx]
        return item


def load_teacher_soft_labels(
    soft_labels_path: str | Path = "experiments/teacher/train_soft_labels.pt",
) -> dict[str, Any]:
    """Load pre-computed teacher soft labels from disk.

    Args:
        soft_labels_path: Path to `.pt` soft labels file.

    Returns:
        Dictionary with keys 'logits', 'labels', 'headline_indices', 'teacher_model_name'.
    """
    path = Path(soft_labels_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher soft labels not found at '{path}'. "
            "Please run teacher training first to generate 'train_soft_labels.pt'."
        )

    data = torch.load(path, map_location="cpu")
    logger.info(
        "Loaded teacher soft labels from '%s' (logits shape: %s, %d samples)",
        path,
        tuple(data["logits"].shape),
        len(data["labels"]),
    )
    return data
