# TinyBanglaBERT 🔬

> **B.Sc. Thesis** — *A Lightweight Knowledge-Distilled Transformer for Explainable Bangla Clickbait Headline Detection*

---

## Research Question

Can a small, purpose-distilled **Student** transformer (TinyBanglaBERT) match
most of a large **Teacher** transformer (BanglaBERT)'s clickbait-detection accuracy at a fraction of the
parameter count / inference latency, while remaining **explainable** and **generalising** to an unseen
cross-domain dataset?

---

## Methodology Overview

```
Phase 1  ── Repo scaffold + utilities               ← YOU ARE HERE
Phase 2  ── Dataset (BanglaBait): loading, cleaning, fixed splits
Phase 3  ── Teacher (BanglaBERT) fine-tuning + evaluation
Phase 4  ── Student (TinyBanglaBERT): architecture + knowledge distillation
Phase 5  ── Evaluation: metrics, McNemar significance test, cross-domain eval
Phase 6  ── Explainability: SHAP token attributions + LIME perturbation
Phase 7  ── Efficiency benchmarking: param count, size, latency (ONNX), memory
Phase 8  ── Ablation study: temperature T, alpha α, intermediate-layer matching
```

---

## Repo Structure

```
tinybanglabert/
├── configs/                    # YAML configs, one per experiment
│   ├── base.yaml               # shared defaults
│   ├── teacher_baseline.yaml
│   └── student_distillation.yaml
├── data/
│   ├── raw/                    # untouched downloads (gitignored)
│   └── processed/              # cleaned splits (splits.json tracked in git)
├── src/
│   ├── data/                   # Phase 2: loading, cleaning, splitting
│   ├── models/                 # Phase 3-4: teacher.py, student.py, distillation.py
│   ├── training/               # Phase 4: train loops, Trainer wrappers
│   ├── evaluation/             # Phase 5: metrics, McNemar, cross-domain
│   ├── explainability/         # Phase 6: shap_runner.py, lime_runner.py
│   ├── efficiency/             # Phase 7: param count, latency, ONNX export
│   └── utils/                  # ← implemented in Phase 1 ✓
│       ├── seeding.py          #   seed_everything()
│       ├── config_loader.py    #   load_config(), save_config(), cfg_to_dict()
│       └── logging_utils.py    #   ExperimentLogger
├── experiments/                # per-run outputs (weights gitignored, CSVs tracked)
├── notebooks/                  # EDA + result visualisation only
├── tests/
│   ├── test_utils.py           # ← implemented in Phase 1 ✓
│   ├── test_data.py            # Phase 2
│   └── test_metrics.py        # Phase 5
├── scripts/
│   ├── train.py                # thin CLI → src/training
│   ├── evaluate.py             # thin CLI → src/evaluation
│   ├── explain.py              # thin CLI → src/explainability
│   └── benchmark.py           # thin CLI → src/efficiency
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Models

| Model | HuggingFace ID | Role |
|---|---|---|
| **BanglaBERT** (Teacher) | `csebuetnlp/banglabert` | Fine-tuned baseline; generates soft labels for distillation |
| **TinyBanglaBERT** (Student) | `csebuetnlp/banglabert_small` | Distilled model; primary thesis contribution |

---

## Dataset

**BanglaBait** — a Bengali clickbait headline dataset.  
Fixed, stratified 80 / 10 / 10 train / val / test split saved once to `data/processed/splits.json`.  
The test set is **touched exactly once**, at the very end, per model.

---

## Reproducibility

Every run is fully reproducible via:

1. **Seeds** — `seed_everything(seed)` sets Python `random`, NumPy, PyTorch (CPU + CUDA), and the `TRANSFORMERS_SEED` env var.
2. **Config snapshot** — the exact `config.yaml` is copied into `experiments/<run_name>/`.
3. **Git commit** — `run_info.json` records the short commit hash alongside every result.
4. **Fixed splits** — `data/processed/splits.json` stores headline IDs, not regenerated per run.

```bash
# Reproduce any run exactly:
git checkout <commit-hash>
python -m scripts.train --config configs/teacher_baseline.yaml --seed 42
```

---

## Quick Start

```bash
# 1. Clone and create a virtual environment
git clone <repo-url>
cd tinybanglabert
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                       # Linux/Mac

# 2. Install dependencies
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# 3. Run utils unit tests (no GPU / data needed)
pytest tests/test_utils.py -v

# 4. Train teacher (Phase 3+)
python -m scripts.train --config configs/teacher_baseline.yaml --seed 42

# 5. Distil student (Phase 4+)
python -m scripts.train --config configs/student_distillation.yaml --seed 42

# 6. Evaluate (Phase 5+)
python -m scripts.evaluate --config configs/teacher_baseline.yaml \
    --checkpoint experiments/teacher_baseline/checkpoints/best_val_f1/

# 7. Explain (Phase 6+)
python -m scripts.explain --config configs/teacher_baseline.yaml \
    --checkpoint experiments/teacher_baseline/checkpoints/best_val_f1/ \
    --method both

# 8. Benchmark (Phase 7+)
python -m scripts.benchmark --config configs/teacher_baseline.yaml \
    --checkpoint experiments/teacher_baseline/checkpoints/best_val_f1/
```

---

## Config System

All hyperparameters live in YAML. Use CLI overrides for one-off changes:

```bash
python -m scripts.train --config configs/student_distillation.yaml \
    --override distillation.temperature=6.0 distillation.alpha=0.5
```

Key config fields (see [`configs/base.yaml`](configs/base.yaml)):

| Section | Key fields |
|---|---|
| `data` | `dataset_name`, `train_ratio`, `val_ratio`, `test_ratio` |
| `model` | `teacher_model_name`, `student_model_name`, `max_seq_len`, `dropout` |
| `training` | `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_ratio`, `fp16` |
| `distillation` | `temperature`, `alpha`, `intermediate_layers`, `hidden_loss_weight` |
| `optuna` | `n_trials`, `timeout_sec`, `study_name` |
| `explainability` | `shap.n_samples`, `lime.n_samples`, `lime.num_features` |
| `efficiency` | `onnx_opset`, `latency_bench_iters`, `batch_sizes` |

---

## Experiment Tracking

Outputs are written to `experiments/<run_name>/`:

```
experiments/teacher_baseline/
  config.yaml        ← exact config (git-tracked)
  run_info.json      ← git hash, seed, hardware (git-tracked)
  metrics.csv        ← one row per epoch (git-tracked)
  metrics.json       ← same data, JSON (gitignored for large runs)
  run.log            ← full Python log (gitignored)
  checkpoints/
    best_val_f1/     ← model weights + tokenizer (gitignored)
```

> Weights & Biases can be added later — replace `exp.log_metrics(...)` calls with `wandb.log(...)` in `src/utils/logging_utils.py`.

---

## Tech Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch ≥ 2.2 |
| Transformer models | HuggingFace `transformers` ≥ 4.40 |
| Dataset handling | HuggingFace `datasets` ≥ 2.19 |
| Metrics | HuggingFace `evaluate` ≥ 0.4 |
| Hyperparameter search | Optuna ≥ 3.6 |
| Explainability | SHAP ≥ 0.45, LIME ≥ 0.2 |
| Statistical testing | statsmodels (McNemar) |
| ONNX export + latency | `onnx`, `onnxruntime` |
| Config management | OmegaConf ≥ 2.3 |
| Testing | pytest ≥ 8.2 |

---

## Citation

```bibtex
@thesis{rahman2026tinybanglabert,
  title  = {A Lightweight Knowledge-Distilled Transformer for Explainable
             Bangla Clickbait Headline Detection},
  author = {Mushfiqur Rahman},
  year   = {2026},
  school = {<Your University>},
  type   = {Bachelor's Thesis}
}
```

---

## License

MIT
