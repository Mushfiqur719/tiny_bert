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
    DistillationDataset,
    IntermediateAlignmentModule,
    KnowledgeDistillationLoss,
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


class TestDistillationTrainerRegression:

    def test_trainer_preserves_teacher_logits_and_uses_kd_loss(self, tmp_path):
        from transformers import TrainingArguments
        from src.models.student import create_student_config, init_random
        from src.training.train_distill import DistillationTrainer

        tokenizer = AutoTokenizer.from_pretrained("experiments/teacher")
        headlines = ["শিরোনাম এক", "শিরোনাম দুই", "শিরোনাম তিন", "শিরোনাম চার"]
        labels = [0, 1, 0, 1]
        teacher_logits_a = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]])
        teacher_logits_b = torch.tensor([[-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]])

        ds_a = DistillationDataset(headlines, labels, teacher_logits_a, tokenizer, max_seq_len=16)
        ds_b = DistillationDataset(headlines, labels, teacher_logits_b, tokenizer, max_seq_len=16)

        cfg = create_student_config(vocab_size=tokenizer.vocab_size, num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64)
        model = init_random(cfg)
        model.train()

        kd_loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.1)
        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            remove_unused_columns=False,
            use_cpu=True,
            report_to="none",
        )
        trainer_a = DistillationTrainer(model=model, args=args, train_dataset=ds_a, kd_loss_fn=kd_loss_fn)
        trainer_b = DistillationTrainer(model=model, args=args, train_dataset=ds_b, kd_loss_fn=kd_loss_fn)

        batch_a = next(iter(trainer_a.get_train_dataloader()))
        batch_b = next(iter(trainer_b.get_train_dataloader()))

        assert "teacher_logits" in batch_a, "Trainer stripped teacher_logits from batch!"
        loss_a = trainer_a.compute_loss(model, batch_a)
        loss_b = trainer_b.compute_loss(model, batch_b)
        assert not torch.isclose(loss_a, loss_b, atol=1e-4), "KD loss ignored teacher_logits!"

    def test_trainer_raises_if_teacher_logits_missing_in_train_mode(self, tmp_path):
        from transformers import TrainingArguments
        from src.models.student import create_student_config, init_random
        from src.training.train_distill import DistillationTrainer

        tokenizer = AutoTokenizer.from_pretrained("experiments/teacher")
        cfg = create_student_config(vocab_size=tokenizer.vocab_size, num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64)
        model = init_random(cfg)
        model.train()

        kd_loss_fn = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)
        args = TrainingArguments(
            output_dir=str(tmp_path),
            remove_unused_columns=False,
            use_cpu=True,
            report_to="none",
        )
        trainer = DistillationTrainer(model=model, args=args, kd_loss_fn=kd_loss_fn)

        inputs = tokenizer(["শিরোনাম এক"], padding="max_length", max_length=16, return_tensors="pt")
        inputs["labels"] = torch.tensor([0])
        with pytest.raises(ValueError, match="teacher_logits"):
            trainer.compute_loss(model, inputs)

    def test_intermediate_alignment_module_forward_and_grad(self):
        align = IntermediateAlignmentModule(
            student_hidden_size=32,
            teacher_hidden_size=64,
            teacher_layer_indices=[2, 5, 8, 11],
            hidden_loss_weight=0.5,
            attention_loss_weight=0.2,
        )
        bsz, seq_len = 3, 10
        s_hidden = tuple(torch.randn(bsz, seq_len, 32, requires_grad=True) for _ in range(5))
        t_hidden = tuple(torch.randn(bsz, seq_len, 64) for _ in range(13))
        s_attn = tuple(torch.softmax(torch.randn(bsz, 4, seq_len, seq_len), dim=-1).requires_grad_(True) for _ in range(4))
        t_attn = tuple(torch.softmax(torch.randn(bsz, 12, seq_len, seq_len), dim=-1) for _ in range(12))
        mask = torch.ones(bsz, seq_len)

        loss, breakdown = align(s_hidden, t_hidden, s_attn, t_attn, attention_mask=mask)
        assert loss.ndim == 0
        assert loss.item() > 0.0
        assert breakdown["hidden_loss"] > 0.0
        assert breakdown["attention_loss"] > 0.0
        loss.backward()
        assert align.projectors[0].weight.grad is not None
