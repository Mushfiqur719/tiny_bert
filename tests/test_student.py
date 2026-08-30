"""
tests/test_student.py — Unit tests for TinyBanglaBERT Student model architecture & initializations.

Run with:  pytest tests/test_student.py -v
"""

from __future__ import annotations

import torch
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.student import (
    create_student_config,
    init_random,
    init_from_teacher,
    count_parameters,
)


@pytest.fixture(scope="module")
def teacher_and_tokenizer():
    """Load the fine-tuned teacher model and tokenizer once for the test module."""
    teacher_path = "experiments/teacher"
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    return teacher, tokenizer


@pytest.fixture
def student_config():
    return create_student_config()


class TestStudentArchitecture:

    def test_student_config_spec(self, student_config):
        assert student_config.num_hidden_layers == 4
        assert student_config.hidden_size == 256
        assert student_config.num_attention_heads == 4
        assert student_config.intermediate_size == 1024
        assert student_config.vocab_size == 32000
        assert student_config.num_labels == 2

    def test_student_random_init_parameter_count(self, student_config):
        student_rand = init_random(student_config)
        param_info = count_parameters(student_rand)

        total_params = param_info["total_params"]
        # Target is ~11.5M (tolerance between 11.0M and 12.0M)
        assert 11_000_000 <= total_params <= 12_000_000, f"Unexpected parameter count: {total_params}"
        assert param_info["trainable_params"] == total_params

    def test_student_random_forward_pass(self, student_config, teacher_and_tokenizer):
        _, tokenizer = teacher_and_tokenizer
        student_rand = init_random(student_config)

        headlines = [
            "বাংলাদেশ জাতীয় ফুটবল দলের ঐতিহাসিক জয়",
            "জানেন কি কীভাবে দ্রুত কোটিপতি হবেন!",
        ]
        inputs = tokenizer(headlines, padding="max_length", max_length=64, return_tensors="pt")

        student_rand.eval()
        with torch.no_grad():
            outputs = student_rand(**inputs)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (2, 2)
        assert not torch.isnan(outputs.logits).any()

    def test_student_teacher_init_projection_strategy(self, student_config, teacher_and_tokenizer):
        teacher, tokenizer = teacher_and_tokenizer
        student_proj = init_from_teacher(teacher, student_config, strategy="projection")
        param_info = count_parameters(student_proj)

        assert 11_000_000 <= param_info["total_params"] <= 12_000_000

        headlines = [
            "নতুন শিক্ষানীতি বাস্তবায়নে সরকারের উদ্যোগ",
            "চমকপ্রদ খবর দেখুন ভিডিওতে এখনই!",
        ]
        inputs = tokenizer(headlines, padding="max_length", max_length=64, return_tensors="pt")

        student_proj.eval()
        with torch.no_grad():
            outputs = student_proj(**inputs)

        assert outputs.logits.shape == (2, 2)
        assert not torch.isnan(outputs.logits).any()

    def test_student_teacher_init_truncation_strategy(self, student_config, teacher_and_tokenizer):
        teacher, tokenizer = teacher_and_tokenizer
        student_trunc = init_from_teacher(teacher, student_config, strategy="truncation")

        headlines = ["টেস্ট শিরোনাম বাংলা"]
        inputs = tokenizer(headlines, padding="max_length", max_length=32, return_tensors="pt")

        student_trunc.eval()
        with torch.no_grad():
            outputs = student_trunc(**inputs)

        assert outputs.logits.shape == (1, 2)
        assert not torch.isnan(outputs.logits).any()

    def test_student_forward_with_labels_computes_loss(self, student_config, teacher_and_tokenizer):
        _, tokenizer = teacher_and_tokenizer
        student_rand = init_random(student_config)

        headlines = ["শিরোনাম এক", "শিরোনাম দুই"]
        labels = torch.tensor([0, 1], dtype=torch.long)
        inputs = tokenizer(headlines, padding="max_length", max_length=32, return_tensors="pt")
        inputs["labels"] = labels

        outputs = student_rand(**inputs)
        assert outputs.loss is not None
        assert outputs.loss.item() > 0.0
