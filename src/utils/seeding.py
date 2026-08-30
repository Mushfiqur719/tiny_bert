"""
seeding.py — Reproducibility helpers.

Sets seeds for Python's `random`, NumPy, PyTorch (CPU + CUDA), and
HuggingFace Transformers so that every run with the same --seed produces
bit-identical results on the same hardware.

Usage:
    from src.utils.seeding import seed_everything
    seed_everything(42)
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all relevant RNG seeds for full reproducibility.

    Args:
        seed: Integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safety

    # Make cuDNN deterministic (slight perf cost, essential for thesis).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # HuggingFace Transformers picks up this env var for its own RNG calls.
    os.environ["TRANSFORMERS_SEED"] = str(seed)

    logger.info("Global seed set to %d", seed)
