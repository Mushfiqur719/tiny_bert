"""
tests/test_distillation.py — Unit tests for Knowledge Distillation loss and Dataset.

Run with:  pytest tests/test_distillation.py -v
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from transformers import AutoTokenizer

from src.models.distillation import (
    KnowledgeDistillationLoss,
    DistillationDataset,
    load_teacher_soft_labels,
)


class TestDistillationLoss:

    def test_kd_loss_positive_scalar(self):
        loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 2, requires_grad=True)
        teacher_logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])

        loss, breakdown = loss_fn(student_logits, teacher_logits, labels)

        assert loss.ndim == 0
        assert loss.item() > 0.0
        assert "hard_loss" in breakdown
        assert "soft_loss" in breakdown
        assert breakdown["temperature"] == 4.0
        assert breakdown["alpha"] == 0.5

    def test_alpha_one_matches_cross_entropy(self):
        loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=1.0)

        student_logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
        teacher_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])

        loss, _ = loss_fn(student_logits, teacher_logits, labels)
        expected_ce = F.cross_entropy(student_logits, labels)

        assert torch.isclose(loss, expected_ce, atol=1e-5)

    def test_alpha_zero_pure_kl_with_t_squared_scaling(self):
        T = 3.0
        loss_fn = KnowledgeDistillationLoss(temperature=T, alpha=0.0)

        student_logits = torch.tensor([[1.5, 0.5], [-1.0, 2.0]], requires_grad=True)
        teacher_logits = torch.tensor([[2.0, 0.0], [-0.5, 1.0]])

        loss, breakdown = loss_fn(student_logits, teacher_logits, labels=None)

        s_log_p = F.log_softmax(student_logits / T, dim=-1)
        t_p = F.softmax(teacher_logits / T, dim=-1)
        expected_kl = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T * T)

        assert torch.isclose(loss, expected_kl, atol=1e-5)
        assert breakdown["hard_loss"] == 0.0

    def test_gradient_flow(self):
        loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)

        student_logits = torch.randn(8, 2, requires_grad=True)
        teacher_logits = torch.randn(8, 2)
        labels = torch.randint(0, 2, (8,))

        loss, _ = loss_fn(student_logits, teacher_logits, labels)
        loss.backward()

        assert student_logits.grad is not None
        assert not torch.isnan(student_logits.grad).any()
        assert student_logits.grad.abs().sum().item() > 0.0


class TestDistillationDatasetAndLoader:

    def test_distillation_dataset_shapes(self):
        tokenizer = AutoTokenizer.from_pretrained("experiments/teacher")
        headlines = ["শিরোনাম এক", "শিরোনাম দুই", "শিরোনাম তিন"]
        labels = [0, 1, 0]
        teacher_logits = torch.tensor([[2.1, -1.2], [-0.5, 1.8], [1.4, -0.8]])

        ds = DistillationDataset(
            headlines=headlines,
            labels=labels,
            teacher_logits=teacher_logits,
            tokenizer=tokenizer,
            max_seq_len=32,
        )

        assert len(ds) == 3
        item = ds[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "teacher_logits" in item
        assert item["input_ids"].shape == (32,)
        assert item["labels"].item() == 0
        assert item["teacher_logits"].shape == (2,)

    def test_load_real_teacher_soft_labels(self):
        data = load_teacher_soft_labels("experiments/teacher/train_soft_labels.pt")

        assert "logits" in data
        assert "labels" in data
        assert "headline_indices" in data
        assert data["logits"].shape == (12044, 2)
        assert len(data["labels"]) == 12044
        assert len(data["headline_indices"]) == 12044
