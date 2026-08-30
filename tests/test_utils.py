"""
tests/test_utils.py — Unit tests for src/utils (seeding + config loading).

Run with:  pytest tests/test_utils.py -v
"""

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.utils.seeding import seed_everything
from src.utils.config_loader import load_config, save_config, cfg_to_dict
from src.utils.logging_utils import ExperimentLogger


# ── Seeding ───────────────────────────────────────────────────────────────

class TestSeedEverything:
    def test_python_random_deterministic(self):
        seed_everything(42)
        a = [random.random() for _ in range(10)]
        seed_everything(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_deterministic(self):
        seed_everything(0)
        a = np.random.rand(5).tolist()
        seed_everything(0)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_torch_deterministic(self):
        seed_everything(7)
        a = torch.rand(4).tolist()
        seed_everything(7)
        b = torch.rand(4).tolist()
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = torch.rand(4).tolist()
        seed_everything(2)
        b = torch.rand(4).tolist()
        assert a != b


# ── Config loader ─────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_load_base_config(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.seed == 42
        assert cfg.data.num_labels == 2

    def test_cli_override(self):
        cfg = load_config("configs/base.yaml", overrides=["training.lr=1e-4"])
        assert abs(cfg.training.lr - 1e-4) < 1e-10

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")

    def test_save_and_reload(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        out = tmp_path / "saved.yaml"
        save_config(cfg, out)
        reloaded = load_config(out)
        assert reloaded.seed == cfg.seed

    def test_cfg_to_dict_is_plain_dict(self):
        cfg = load_config("configs/base.yaml")
        d = cfg_to_dict(cfg)
        assert isinstance(d, dict)
        assert isinstance(d["training"], dict)


# ── ExperimentLogger ──────────────────────────────────────────────────────

class TestExperimentLogger:
    def test_creates_run_directory(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=42, experiments_root=tmp_path)
        assert (tmp_path / "test_run").is_dir()
        logger.finish()

    def test_config_yaml_written(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=42, experiments_root=tmp_path)
        assert (tmp_path / "test_run" / "config.yaml").exists()
        logger.finish()

    def test_run_info_has_seed(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=99, experiments_root=tmp_path)
        run_info = json.loads((tmp_path / "test_run" / "run_info.json").read_text())
        assert run_info["seed"] == 99
        logger.finish()

    def test_log_metrics_creates_files(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=42, experiments_root=tmp_path)
        logger.log_metrics({"epoch": 1, "split": "val", "val_f1_macro": 0.91})
        assert (tmp_path / "test_run" / "metrics.json").exists()
        assert (tmp_path / "test_run" / "metrics.csv").exists()
        logger.finish()

    def test_metrics_json_is_valid(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=42, experiments_root=tmp_path)
        logger.log_metrics({"epoch": 1, "split": "val", "val_f1_macro": 0.91})
        rows = json.loads((tmp_path / "test_run" / "metrics.json").read_text())
        assert len(rows) == 1
        assert rows[0]["val_f1_macro"] == pytest.approx(0.91)
        logger.finish()

    def test_checkpoint_path_created(self, tmp_path):
        cfg = load_config("configs/base.yaml")
        logger = ExperimentLogger("test_run", cfg, seed=42, experiments_root=tmp_path)
        ckpt = logger.checkpoint_path("best_val_f1")
        assert ckpt.is_dir()
        logger.finish()
