"""
logging_utils.py — Structured experiment logging to JSON / CSV.

Creates a per-run directory under ``experiments/<run_name>/`` and writes:
  - ``config.yaml``  — the exact config used
  - ``metrics.json`` — all metric snapshots appended over the run
  - ``metrics.csv``  — same data in tabular form for quick inspection
  - ``run_info.json`` — git commit hash, seed, timestamps, Python/PyTorch versions

Designed so Weights & Biases can be bolted on later: just call
``wandb.init()`` / ``wandb.log()`` alongside these helpers.

Usage:
    from src.utils.logging_utils import ExperimentLogger
    exp = ExperimentLogger(run_name="teacher_baseline", cfg=cfg, seed=42)
    exp.log_metrics({"epoch": 1, "val_f1": 0.923, "split": "val"})
    exp.finish()
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

EXPERIMENTS_ROOT = Path("experiments")


def _git_commit_hash() -> str:
    """Return the short git commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def setup_file_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Attach a FileHandler to the root logger so all log lines land in *log_path*."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)


class ExperimentLogger:
    """Manages all artefacts written to ``experiments/<run_name>/``.

    Args:
        run_name: Unique identifier for this run (e.g. ``"teacher_baseline_lr2e5"``).
        cfg: OmegaConf config object used for this run.
        seed: The global seed used (logged alongside results).
        experiments_root: Override the default ``experiments/`` directory.
    """

    def __init__(
        self,
        run_name: str,
        cfg: DictConfig,
        seed: int = 42,
        experiments_root: Path | str = EXPERIMENTS_ROOT,
    ) -> None:
        self.run_name = run_name
        self.cfg = cfg
        self.seed = seed
        self.run_dir = Path(experiments_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories.
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # File paths.
        self._metrics_json = self.run_dir / "metrics.json"
        self._metrics_csv = self.run_dir / "metrics.csv"
        self._run_info_json = self.run_dir / "run_info.json"
        self._config_yaml = self.run_dir / "config.yaml"

        self._metric_rows: list[dict[str, Any]] = []
        self._csv_writer: csv.DictWriter | None = None
        self._csv_file = None  # kept open for the duration of the run

        self._init_run_info()
        self._save_config()
        setup_file_logging(self.run_dir / "run.log")
        logger.info("ExperimentLogger initialised — run dir: %s", self.run_dir)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _init_run_info(self) -> None:
        info: dict[str, Any] = {
            "run_name": self.run_name,
            "seed": self.seed,
            "git_commit": _git_commit_hash(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
        self._run_info: dict[str, Any] = info
        self._write_json(self._run_info_json, info)

    def _save_config(self) -> None:
        OmegaConf.save(self.cfg, self._config_yaml)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Append a metrics snapshot (one row) to metrics.json and metrics.csv.

        Args:
            metrics: Flat dict of metric names → values.  Include at minimum
                ``"epoch"`` (or ``"step"``) and ``"split"`` (``"train"`` /
                ``"val"`` / ``"test"``) for unambiguous downstream analysis.
        """
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        self._metric_rows.append(row)

        # JSON — rewrite full file each time (simple, correct, small files).
        self._write_json(self._metrics_json, self._metric_rows)

        # CSV — open once, write incrementally.
        if self._csv_writer is None:
            self._csv_file = open(self._metrics_csv, "w", newline="", encoding="utf-8")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()), extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        # Handle new keys appearing mid-run gracefully.
        self._csv_writer.writerow({k: row.get(k, "") for k in self._csv_writer.fieldnames})
        self._csv_file.flush()  # type: ignore[union-attr]

        logger.info("Metrics logged: %s", json.dumps(metrics, default=str))

    def checkpoint_path(self, name: str) -> Path:
        """Return a path inside ``checkpoints/`` for saving a model artefact.

        Args:
            name: Sub-directory name, e.g. ``"best_val_f1"`` or ``"epoch_3"``.

        Returns:
            :class:`pathlib.Path` pointing to ``checkpoints/<name>/`` (created).
        """
        p = self.checkpoints_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def finish(self) -> None:
        """Mark the run as finished and close open file handles."""
        self._run_info["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(self._run_info_json, self._run_info)
        if self._csv_file is not None:
            self._csv_file.close()
        logger.info("Run '%s' finished. Artefacts at: %s", self.run_name, self.run_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
