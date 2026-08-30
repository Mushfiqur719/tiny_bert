"""
config_loader.py — YAML config loading with OmegaConf.

Every experiment reads a single YAML file.  CLI overrides follow the
OmegaConf dotlist syntax, e.g. ``training.lr=3e-5 model.dropout=0.1``.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config("configs/teacher_baseline.yaml", overrides=["training.lr=2e-5"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load a YAML config file and apply optional CLI overrides.

    Args:
        config_path: Path to the ``.yaml`` config file.
        overrides: Sequence of ``"key=value"`` override strings (OmegaConf
            dotlist syntax).  Pass ``None`` or ``[]`` for no overrides.

    Returns:
        A merged, read-only :class:`omegaconf.DictConfig`.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_cfg: DictConfig = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    logger.info("Loaded config from %s", config_path)
    if overrides:
        logger.info("Applied CLI overrides: %s", overrides)

    return cfg


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf DictConfig to a plain Python dict (for JSON serialisation)."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Serialise a config to YAML at *output_path*.

    Args:
        cfg: Config object to save.
        output_path: Destination ``.yaml`` path (parent dirs are created).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)
    logger.info("Config saved to %s", output_path)
