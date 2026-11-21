"""
src/models/student.py — TinyBanglaBERT Student Model Architecture & Initialization.

TinyBanglaBERT Specification:
  - 4 Transformer encoder layers (num_hidden_layers = 4)
  - Hidden size 256 (hidden_size = 256)
  - 4 Attention heads (num_attention_heads = 4, head_dim = 64)
  - Intermediate feed-forward size 1024 (intermediate_size = 1024)
  - Binary sequence classification head (num_labels = 2)
  - Vocabulary: Reuses BanglaBERT's 32,000 WordPiece vocabulary
  - Total Parameters: ~11.55M (vs ~110.6M for Teacher, ~9.6x compression)

Initialization Strategies:
  1. `init_random(config)`: Fresh random initialization following normal distribution.
  2. `init_from_teacher(teacher_model, config, strategy="projection")`:
     - Embeddings: Word & position embeddings projected from 768 to 256 dimensions.
     - Encoder Layers: First 4 layers {0, 1, 2, 3} of the Teacher (earliest, most general
       syntactic/semantic representations) transferred via orthogonal PCA projection
       to preserve maximum variance without arbitrary dimension slicing.
     - Classifier: Dense projection layer and classification head weights transferred.

Usage:
    from src.models.student import create_student_config, init_random, init_from_teacher, count_parameters
    cfg = create_student_config()
    model_rand = init_random(cfg)
    model_teacher_init = init_from_teacher(teacher_model, cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    ElectraConfig,
    ElectraForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

ID2LABEL = {0: "not-clickbait", 1: "clickbait"}
LABEL2ID = {"not-clickbait": 0, "clickbait": 1}


def create_student_config(
    vocab_size: int = 32000,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 512,
    num_labels: int = 2,
    dropout: float = 0.1,
) -> ElectraConfig:
    """Construct the configuration for TinyBanglaBERT.

    Args:
        vocab_size: Size of vocabulary (default: 32000, matching BanglaBERT).
        hidden_size: Transformer hidden state dimensionality (default: 256).
        num_hidden_layers: Number of transformer encoder layers (default: 4).
        num_attention_heads: Number of attention heads (default: 4).
        intermediate_size: Feed-forward dimension (default: 1024).
        max_position_embeddings: Maximum sequence length (default: 512).
        num_labels: Number of classification targets (default: 2).
        dropout: Dropout probability (default: 0.1).

    Returns:
        ElectraConfig configured for TinyBanglaBERT.
    """
    config = ElectraConfig(
        vocab_size=vocab_size,
        embedding_size=hidden_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
        initializer_range=0.02,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return config


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Count model parameters and memory footprint.

    Args:
        model: PyTorch module / Transformer model.

    Returns:
        Dictionary with total_params, trainable_params, non_trainable_params, and size_mb.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / (1024 * 1024)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "size_mb": round(size_mb, 2),
    }


def init_random(config: ElectraConfig) -> ElectraForSequenceClassification:
    """Strategy 2a: Initialize Student model with standard random weights.

    Args:
        config: TinyBanglaBERT ElectraConfig.

    Returns:
        Randomly initialized ElectraForSequenceClassification model.
    """
    logger.info("Initializing TinyBanglaBERT (Random Init)...")
    model = ElectraForSequenceClassification(config)
    param_info = count_parameters(model)
    logger.info(
        "TinyBanglaBERT (Random Init) created — Total Params: %d (%.2f MB)",
        param_info["total_params"],
        param_info["size_mb"],
    )
    return model


def _compute_projection_matrix(
    teacher_weights: torch.Tensor,
    target_dim: int = 256,
) -> torch.Tensor:
    """Compute orthogonal low-rank PCA projection matrix P [D_teacher, D_student].

    Preserves maximum variance directions of teacher representation space.
    """
    with torch.no_grad():
        # SVD on centered/standardized teacher embedding weights
        W = teacher_weights.float()
        _, _, V = torch.pca_lowrank(W, q=target_dim, center=True)
        # V has shape [D_teacher, target_dim]
        return V.to(teacher_weights.dtype)


def _compute_intermediate_projection(
    teacher_dim: int = 3072,
    student_dim: int = 1024,
) -> torch.Tensor:
    """Compute block-averaging orthogonal projection for intermediate layer [3072, 1024]."""
    scale = 3.0 ** 0.5
    P = torch.zeros(teacher_dim, student_dim)
    for i in range(student_dim):
        P[i * 3 : (i + 1) * 3, i] = 1.0 / scale
    return P


def _select_teacher_layers(
    num_teacher_layers: int,
    num_student_layers: int,
    layer_selection: str = "strided",
) -> list[int]:
    """Choose which teacher encoder layer indices to transfer into the student.

    Args:
        num_teacher_layers: Total layers in teacher (e.g., 12).
        num_student_layers: Total layers in student (e.g., 4).
        layer_selection: 'strided' (default, e.g. [2, 5, 8, 11] for 12->4),
                         'first' / 'first4' ([0, 1, 2, 3]), or
                         'last' / 'last4' ([8, 9, 10, 11]).

    Returns:
        List of teacher layer indices of length `num_student_layers`.
    """
    if num_student_layers > num_teacher_layers:
        raise ValueError(
            f"Student layers ({num_student_layers}) cannot exceed teacher layers ({num_teacher_layers})."
        )
    if layer_selection in ("first", "first4"):
        return list(range(num_student_layers))
    elif layer_selection in ("last", "last4"):
        return list(range(num_teacher_layers - num_student_layers, num_teacher_layers))
    elif layer_selection == "strided":
        # Evenly spaced across depth, always including the final semantic layer
        stride = num_teacher_layers / num_student_layers
        return [min(num_teacher_layers - 1, int(round((i + 1) * stride)) - 1) for i in range(num_student_layers)]
    else:
        raise ValueError(
            f"Unknown layer_selection: '{layer_selection}'. Choose 'strided', 'first', or 'last'."
        )


def init_from_teacher(
    teacher_model: PreTrainedModel,
    config: ElectraConfig,
    strategy: str = "projection",
    layer_selection: str = "strided",
) -> ElectraForSequenceClassification:
    """Strategy 2b: Initialize Student model by transferring and projecting Teacher weights.

    Design & Justification:
      1. SVD / Orthogonal PCA Projection (Default): Rather than arbitrarily slicing / truncating
         the first 256 coordinates (which discards ~67% of coordinate variance), we compute
         an orthogonal projection matrix P from the Teacher's word embedding space.
      2. LayerNorm Preservation: LayerNorm weights are element-wise scale parameters (~1.0),
         not vectors in embedding space, so they are initialized to neutral (weight=1, bias=0)
         rather than multiplied by P.
      3. Strided Layer Selection: By default ('strided'), we extract evenly spaced layers
         {2, 5, 8, 11} from the 12-layer Teacher so the Student inherits both low-level Bangla
         syntax and high-level semantic/pragmatic clickbait representations.
      4. Classifier Head: The classification dense and projection layers are mapped into
         the 256-dimensional space to preserve the learned decision boundary.

    Args:
        teacher_model: Fine-tuned Teacher BanglaBERT model.
        config: TinyBanglaBERT ElectraConfig.
        strategy: Transfer strategy — 'projection' (PCA-based) or 'truncation' (slice 0..256).
        layer_selection: 'strided' (default) or 'first'.

    Returns:
        Initialized ElectraForSequenceClassification model with transferred weights.
    """
    layer_indices = _select_teacher_layers(
        num_teacher_layers=teacher_model.config.num_hidden_layers,
        num_student_layers=config.num_hidden_layers,
        layer_selection=layer_selection,
    )
    logger.info(
        "Initializing TinyBanglaBERT from Teacher (Strategy: %s, Layers: %s)...",
        strategy,
        layer_indices,
    )
    student_model = ElectraForSequenceClassification(config)

    t_emb = teacher_model.electra.embeddings
    s_emb = student_model.electra.embeddings

    t_hidden = teacher_model.config.hidden_size  # 768
    s_hidden = config.hidden_size                # 256
    t_inter = teacher_model.config.intermediate_size  # 3072
    s_inter = config.intermediate_size                # 1024

    with torch.no_grad():
        if strategy == "projection":
            # Compute projection matrix P [768, 256]
            P = _compute_projection_matrix(t_emb.word_embeddings.weight.data, target_dim=s_hidden)
            P_inter = _compute_intermediate_projection(teacher_dim=t_inter, student_dim=s_inter).to(P.device)

            # 1. Embeddings Transfer
            # Word embeddings: [32000, 768] @ [768, 256] -> [32000, 256]
            s_emb.word_embeddings.weight.copy_(t_emb.word_embeddings.weight.data @ P)

            # Position embeddings: [512, 768] @ [768, 256] -> [512, 256]
            s_emb.position_embeddings.weight.copy_(t_emb.position_embeddings.weight.data @ P)

            # Token type embeddings: [2, 768] @ [768, 256] -> [2, 256]
            s_emb.token_type_embeddings.weight.copy_(t_emb.token_type_embeddings.weight.data @ P)

            # LayerNorm: neutral init (element-wise scale != embedding space vector)
            nn.init.ones_(s_emb.LayerNorm.weight)
            nn.init.zeros_(s_emb.LayerNorm.bias)

            # 2. Transformer Encoder Layers Transfer
            for s_idx, t_idx in enumerate(layer_indices):
                t_layer = teacher_model.electra.encoder.layer[t_idx]
                s_layer = student_model.electra.encoder.layer[s_idx]

                # Attention Query: [768, 768] -> [256, 256] via P.T @ W @ P
                s_layer.attention.self.query.weight.copy_(P.T @ t_layer.attention.self.query.weight.data @ P)
                s_layer.attention.self.query.bias.copy_(t_layer.attention.self.query.bias.data @ P)

                # Attention Key: [768, 768] -> [256, 256]
                s_layer.attention.self.key.weight.copy_(P.T @ t_layer.attention.self.key.weight.data @ P)
                s_layer.attention.self.key.bias.copy_(t_layer.attention.self.key.bias.data @ P)

                # Attention Value: [768, 768] -> [256, 256]
                s_layer.attention.self.value.weight.copy_(P.T @ t_layer.attention.self.value.weight.data @ P)
                s_layer.attention.self.value.bias.copy_(t_layer.attention.self.value.bias.data @ P)

                # Attention Output Dense: [768, 768] -> [256, 256]
                s_layer.attention.output.dense.weight.copy_(P.T @ t_layer.attention.output.dense.weight.data @ P)
                s_layer.attention.output.dense.bias.copy_(t_layer.attention.output.dense.bias.data @ P)
                nn.init.ones_(s_layer.attention.output.LayerNorm.weight)
                nn.init.zeros_(s_layer.attention.output.LayerNorm.bias)

                # Intermediate Dense: [3072, 768] -> [1024, 256] via P_inter.T @ W @ P
                s_layer.intermediate.dense.weight.copy_(P_inter.T @ t_layer.intermediate.dense.weight.data @ P)
                s_layer.intermediate.dense.bias.copy_(t_layer.intermediate.dense.bias.data @ P_inter)

                # Output Dense: [768, 3072] -> [256, 1024] via P.T @ W @ P_inter
                s_layer.output.dense.weight.copy_(P.T @ t_layer.output.dense.weight.data @ P_inter)
                s_layer.output.dense.bias.copy_(t_layer.output.dense.bias.data @ P)
                nn.init.ones_(s_layer.output.LayerNorm.weight)
                nn.init.zeros_(s_layer.output.LayerNorm.bias)

            # 3. Classifier Head Transfer
            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                # Dense projection: [768, 768] -> [256, 256]
                student_model.classifier.dense.weight.copy_(P.T @ teacher_model.classifier.dense.weight.data @ P)
                student_model.classifier.dense.bias.copy_(teacher_model.classifier.dense.bias.data @ P)

                # Out projection: [2, 768] @ [768, 256] -> [2, 256]
                student_model.classifier.out_proj.weight.copy_(teacher_model.classifier.out_proj.weight.data @ P)
                student_model.classifier.out_proj.bias.copy_(teacher_model.classifier.out_proj.bias.data)

        elif strategy == "truncation":
            # Slice first 0..256 dimensions
            s_emb.word_embeddings.weight.copy_(t_emb.word_embeddings.weight.data[:, :s_hidden])
            s_emb.position_embeddings.weight.copy_(t_emb.position_embeddings.weight.data[:, :s_hidden])
            s_emb.token_type_embeddings.weight.copy_(t_emb.token_type_embeddings.weight.data[:, :s_hidden])
            s_emb.LayerNorm.weight.copy_(t_emb.LayerNorm.weight.data[:s_hidden])
            s_emb.LayerNorm.bias.copy_(t_emb.LayerNorm.bias.data[:s_hidden])

            for s_idx, t_idx in enumerate(layer_indices):
                t_layer = teacher_model.electra.encoder.layer[t_idx]
                s_layer = student_model.electra.encoder.layer[s_idx]

                s_layer.attention.self.query.weight.copy_(t_layer.attention.self.query.weight.data[:s_hidden, :s_hidden])
                s_layer.attention.self.query.bias.copy_(t_layer.attention.self.query.bias.data[:s_hidden])
                s_layer.attention.self.key.weight.copy_(t_layer.attention.self.key.weight.data[:s_hidden, :s_hidden])
                s_layer.attention.self.key.bias.copy_(t_layer.attention.self.key.bias.data[:s_hidden])
                s_layer.attention.self.value.weight.copy_(t_layer.attention.self.value.weight.data[:s_hidden, :s_hidden])
                s_layer.attention.self.value.bias.copy_(t_layer.attention.self.value.bias.data[:s_hidden])

                s_layer.attention.output.dense.weight.copy_(t_layer.attention.output.dense.weight.data[:s_hidden, :s_hidden])
                s_layer.attention.output.dense.bias.copy_(t_layer.attention.output.dense.bias.data[:s_hidden])
                s_layer.attention.output.LayerNorm.weight.copy_(t_layer.attention.output.LayerNorm.weight.data[:s_hidden])
                s_layer.attention.output.LayerNorm.bias.copy_(t_layer.attention.output.LayerNorm.bias.data[:s_hidden])

                s_layer.intermediate.dense.weight.copy_(t_layer.intermediate.dense.weight.data[:s_inter, :s_hidden])
                s_layer.intermediate.dense.bias.copy_(t_layer.intermediate.dense.bias.data[:s_inter])

                s_layer.output.dense.weight.copy_(t_layer.output.dense.weight.data[:s_hidden, :s_inter])
                s_layer.output.dense.bias.copy_(t_layer.output.dense.bias.data[:s_hidden])
                s_layer.output.LayerNorm.weight.copy_(t_layer.output.LayerNorm.weight.data[:s_hidden])
                s_layer.output.LayerNorm.bias.copy_(t_layer.output.LayerNorm.bias.data[:s_hidden])

            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                student_model.classifier.dense.weight.copy_(teacher_model.classifier.dense.weight.data[:s_hidden, :s_hidden])
                student_model.classifier.dense.bias.copy_(teacher_model.classifier.dense.bias.data[:s_hidden])
                student_model.classifier.out_proj.weight.copy_(teacher_model.classifier.out_proj.weight.data[:, :s_hidden])
                student_model.classifier.out_proj.bias.copy_(teacher_model.classifier.out_proj.bias.data)
        else:
            raise ValueError(f"Unknown transfer strategy: '{strategy}'. Choose 'projection' or 'truncation'.")

    param_info = count_parameters(student_model)
    logger.info(
        "TinyBanglaBERT (Teacher Init) created — Total Params: %d (%.2f MB)",
        param_info["total_params"],
        param_info["size_mb"],
    )
    return student_model


def init_from_small_pretrained(
    pretrained_name: str = "csebuetnlp/banglabert_small",
    num_hidden_layers: int = 4,
    num_labels: int = 2,
    dropout: float = 0.1,
    layer_selection: str = "strided",
) -> ElectraForSequenceClassification:
    """Strategy 2c: Initialize a 4-layer compact Student from pre-trained BanglaBERT-Small.

    Preserves exact pre-trained Bengali embeddings (including factorized embedding_size=128
    -> hidden_size=256 projection) and copies strided encoder layers {2, 5, 8, 11} without
    any PCA dimension mismatch.

    Args:
        pretrained_name: HuggingFace model ID for compact pre-trained backbone.
        num_hidden_layers: Number of encoder layers to retain (default: 4).
        num_labels: Number of classification labels (default: 2).
        dropout: Hidden and attention dropout probability.
        layer_selection: 'strided' (default) or 'first'.

    Returns:
        Compact ElectraForSequenceClassification initialized from pre-trained weights.
    """
    logger.info(
        "Initializing %d-layer TinyBanglaBERT from pre-trained '%s'...",
        num_hidden_layers,
        pretrained_name,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    base_cfg = base_model.config
    layer_indices = _select_teacher_layers(
        num_teacher_layers=base_cfg.num_hidden_layers,
        num_student_layers=num_hidden_layers,
        layer_selection=layer_selection,
    )

    student_cfg = ElectraConfig(
        vocab_size=base_cfg.vocab_size,
        embedding_size=getattr(base_cfg, "embedding_size", base_cfg.hidden_size),
        hidden_size=base_cfg.hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=base_cfg.num_attention_heads,
        intermediate_size=base_cfg.intermediate_size,
        hidden_act=base_cfg.hidden_act,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=base_cfg.max_position_embeddings,
        type_vocab_size=base_cfg.type_vocab_size,
        initializer_range=base_cfg.initializer_range,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    student_model = ElectraForSequenceClassification(student_cfg)

    with torch.no_grad():
        # Copy embeddings directly (exact shape match)
        student_model.electra.embeddings.load_state_dict(
            base_model.electra.embeddings.state_dict()
        )
        # Copy factorized embedding projection if present (e.g., 128 -> 256)
        if hasattr(base_model.electra, "embeddings_project") and hasattr(
            student_model.electra, "embeddings_project"
        ):
            student_model.electra.embeddings_project.load_state_dict(
                base_model.electra.embeddings_project.state_dict()
            )
        # Copy selected encoder layers directly
        for s_idx, t_idx in enumerate(layer_indices):
            student_model.electra.encoder.layer[s_idx].load_state_dict(
                base_model.electra.encoder.layer[t_idx].state_dict()
            )

    param_info = count_parameters(student_model)
    logger.info(
        "TinyBanglaBERT (Small Pre-trained Init, layers=%s) created — Total Params: %d (%.2f MB)",
        layer_indices,
        param_info["total_params"],
        param_info["size_mb"],
    )
    return student_model


def init_small_random(
    pretrained_name: str = "csebuetnlp/banglabert_small",
    num_hidden_layers: int = 4,
    num_labels: int = 2,
    dropout: float = 0.1,
) -> ElectraForSequenceClassification:
    """Initialize a parameter-matched compact Student with fresh random weights.

    Matches the exact factorized architecture (embedding_size=128, hidden_size=256,
    vocab_size=32000, ~7.42M params at 4L) of `init_from_small_pretrained` without
    copying any pre-trained weights, enabling strictly controlled initialisation ablations.
    """
    base_cfg = AutoConfig.from_pretrained(pretrained_name)
    student_cfg = ElectraConfig(
        vocab_size=base_cfg.vocab_size,
        embedding_size=getattr(base_cfg, "embedding_size", base_cfg.hidden_size),
        hidden_size=base_cfg.hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=base_cfg.num_attention_heads,
        intermediate_size=base_cfg.intermediate_size,
        hidden_act=base_cfg.hidden_act,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=base_cfg.max_position_embeddings,
        type_vocab_size=base_cfg.type_vocab_size,
        initializer_range=base_cfg.initializer_range,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = ElectraForSequenceClassification(student_cfg)
    param_info = count_parameters(model)
    logger.info(
        "TinyBanglaBERT (Parameter-Matched Random Init, %dL) created — Total Params: %d (%.2f MB)",
        num_hidden_layers,
        param_info["total_params"],
        param_info["size_mb"],
    )
    return model


def save_student_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
) -> Path:
    """Save student model weights, configuration, and tokenizer.

    Args:
        model: Trained student model.
        tokenizer: Associated tokenizer.
        output_dir: Target directory path.

    Returns:
        Path to output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved student model and tokenizer to '%s'", output_path)
    return output_path
