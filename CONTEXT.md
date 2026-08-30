# Agent Context — TinyBanglaBERT Thesis Project

> **Read this entire file before touching any code.**
> It is the single source of truth for agent handoff.
> Last updated: 2026-08-30 (after Phase 2 completion, commit `212c7ac`).

---

## 1. Thesis in one sentence

Build a small student transformer (TinyBanglaBERT) that matches most of a large teacher transformer (BanglaBERT) on Bangla clickbait headline detection — via knowledge distillation — while staying explainable (SHAP + LIME) and benchmarking its efficiency (ONNX latency).

---

## 2. Repository location

```
c:\Users\Mushfiqur Rahman\Documents\Projects\tiny_bert\
```

**Virtual environment (always use this, never system Python):**
```
.venv\Scripts\python.exe      # Python 3.12.10
.venv\Scripts\pip.exe
.venv\Scripts\activate        # activate in PowerShell
```

**Git state:**
```
212c7ac  Phase 2: data pipeline (load/clean/split/report) + 34 tests [all pass]
66797f0  Phase 1: repo scaffold + src/utils (seeding, config, logging) + 15 tests [pass]
c3d523c  init
```

---

## 3. Phases — what is done, what is next

| Phase | Status | Description |
|-------|--------|-------------|
| **1** | ✅ Done | Repo scaffold, `src/utils/` (seeding, config, logging), 15 tests |
| **2** | ✅ Done | Data pipeline: load, clean, split, report. 34 tests. `splits.json` committed. |
| **3** | ✅ Done | **Teacher (BanglaBERT) fine-tuning** — `src/models/teacher.py`, test macro-F1: 0.8233, accuracy: 0.8420. Soft labels saved. |
| **4** | ✅ Done | **Student (TinyBanglaBERT) architecture & 4-way ablation distillation** — selected `student_teacher_init_distilled` as `tinybanglabert_final` (Val F1: 0.7462, Test F1: 0.7431, Test Acc: 75.63%). 90.3% Teacher F1 retention at 9.58x smaller size. |
| **5** | ⬜ TODO | **Evaluation: metrics, McNemar test, cross-domain eval** |
| **6** | ⬜ TODO | Explainability: SHAP token attributions + LIME perturbation |
| **7** | ✅ Done | **Efficiency benchmarking & ONNX export** — 9.58x parameter compression, 37.63x smaller with INT8 (11.22 MB), 22.69x faster ONNX CPU inference (5.15 ms vs 116.83 ms), 243 fps with INT8. |
| **8** | ⬜ TODO | Ablation study: temperature T, alpha α, intermediate-layer matching |

---

## 4. Complete repo structure (as of Phase 2)

```
tiny_bert/
├── .venv/                          ← project venv (Python 3.12, use this always)
├── configs/
│   ├── base.yaml                   ← ALL defaults — read this first
│   ├── teacher_baseline.yaml       ← overrides for teacher fine-tuning
│   └── student_distillation.yaml   ← overrides for distillation
├── data/
│   ├── raw/                        ← (empty — HF downloads to ~/.cache)
│   └── processed/
│       ├── splits.json             ← FIXED train/val/test indices (committed, never regenerate)
│       └── data_report.md          ← class balance + token length stats
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                 ← load_raw() → (labeled_df, unlabeled_df)
│   │   ├── clean.py                ← clean_headline(), clean_dataframe()
│   │   ├── split.py                ← create_splits(), load_splits(), apply_splits()
│   │   └── report.py               ← generate_report() → data_report.md
│   ├── models/                     ← STUB — Phase 3 fills this
│   │   └── __init__.py
│   ├── training/                   ← STUB — Phase 3 fills this
│   │   └── __init__.py
│   ├── evaluation/                 ← STUB — Phase 5 fills this
│   │   └── __init__.py
│   ├── explainability/             ← STUB — Phase 6 fills this
│   │   └── __init__.py
│   ├── efficiency/                 ← STUB — Phase 7 fills this
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py             ← re-exports all 3 utils
│       ├── seeding.py              ← seed_everything(seed: int)
│       ├── config_loader.py        ← load_config(path, overrides=[])
│       └── logging_utils.py        ← ExperimentLogger(run_name, cfg, seed)
├── experiments/                    ← per-run outputs (weights gitignored, CSVs tracked)
├── notebooks/                      ← EDA only, nothing load-bearing
├── scripts/
│   ├── prepare_data.py             ← DONE: runs full data pipeline
│   ├── train.py                    ← STUB: raises NotImplementedError (Phase 3)
│   ├── evaluate.py                 ← STUB (Phase 5)
│   ├── explain.py                  ← STUB (Phase 6)
│   └── benchmark.py                ← STUB (Phase 7)
├── tests/
│   ├── test_utils.py               ← 15 tests (seeding, config, ExperimentLogger) ✅
│   ├── test_data_clean.py          ← 34 tests (clean + split) ✅
│   ├── test_data.py                ← STUB (Phase 2 placeholder, empty)
│   └── test_metrics.py             ← STUB (Phase 5 placeholder, empty)
├── pyproject.toml                  ← build config, pytest settings, ruff
├── requirements.txt
└── README.md
```

---

## 5. Global conventions (MUST follow)

### 5.1 Always use the venv
```powershell
# Run any script:
.venv\Scripts\python.exe -m scripts.prepare_data --config configs/base.yaml

# Run tests:
.venv\Scripts\python.exe -m pytest tests/ -v
```

### 5.2 Seeding — every script must call this
```python
from src.utils import seed_everything
seed_everything(42)   # or from --seed CLI arg
```
Sets Python `random`, NumPy, PyTorch (CPU+CUDA), `cudnn.deterministic=True`, `TRANSFORMERS_SEED`.

### 5.3 Config loading — OmegaConf, never hardcode
```python
from src.utils import load_config
cfg = load_config("configs/base.yaml", overrides=["training.lr=3e-5"])
# Access as cfg.training.lr, cfg.model.teacher_model_name, etc.
```

### 5.4 Experiment logging — ExperimentLogger
```python
from src.utils import ExperimentLogger
exp = ExperimentLogger(run_name=cfg.training.run_name, cfg=cfg, seed=seed)
# outputs go to experiments/<run_name>/
#   config.yaml, run_info.json (git hash + seed + hw), metrics.json, metrics.csv, run.log
exp.log_metrics({"epoch": 1, "split": "val", "val_f1_macro": 0.91})
ckpt_path = exp.checkpoint_path("best_val_f1")   # → experiments/<run>/checkpoints/best_val_f1/
exp.finish()
```

### 5.5 Data loading (always load from splits.json — never re-split)
```python
from src.utils import load_config, seed_everything
from src.data import load_raw, clean_dataframe, load_splits, apply_splits

cfg = load_config("configs/base.yaml")
seed_everything(42)

labeled_df, unlabeled_df = load_raw(cfg, hf_dataset_id="Motahar/banglabait")
labeled_df = clean_dataframe(labeled_df).drop(columns=["_was_cleaned"])
labeled_df = labeled_df.reset_index(drop=True)

splits = load_splits(cfg)                           # loads data/processed/splits.json
train_df, val_df, test_df = apply_splits(labeled_df, splits)
# train_df: 12,044 rows | val_df: 1,506 | test_df: 1,506
```

### 5.6 The test set is sacred
**Touch `test_df` exactly once, after all hyperparameters are fixed, for the final reported number per model.**

### 5.7 Device detection (never hardcode GPU)
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 5.8 Windows-specific gotchas
- `dataloader_num_workers` **must be 0** (Windows multiprocessing broken with HF datasets)
- Console prints of Bangla text: use `sys.stdout.buffer.write(text.encode("utf-8"))` not `print()`
- pytest `tmp_path` uses `--basetemp=.pytest_tmp` (set in pyproject.toml) to avoid Windows AppData permission errors
- HF Hub symlinks warning on Windows is harmless — caching still works

---

## 6. Key dataset facts

| Fact | Value |
|------|-------|
| Dataset | `Motahar/banglabait` (HuggingFace Hub) |
| Total labeled rows | 15,056 |
| Not-clickbait (label=0) | 9,819 (65.2%) |
| Clickbait (label=1) | 5,237 (34.8%) |
| Imbalance ratio | ~1.87 : 1 |
| Train / Val / Test | 12,044 / 1,506 / 1,506 |
| Median headline tokens (BanglaBERT) | 13 |
| Max headline tokens | 46 (p95=24) |
| Headlines > 128 tokens | **0%** — no truncation needed |
| Unlabeled rows in HF version | 0 (HF version only has labeled data) |

**Implication for Phase 3:** Use `class_weight` or `pos_weight` in the loss — 65/35 split is imbalanced enough to matter.

---

## 7. Models

| Role | Model Architecture | HuggingFace ID / Config | Parameters |
|------|-------------------|-------------------------|------------|
| Teacher | BanglaBERT (Electra, 12 layers, 768 hidden, 12 heads) | `csebuetnlp/banglabert` | **110,618,882** (~110.6M) |
| Student | TinyBanglaBERT (4 layers, 256 hidden, 4 heads, 1024 intermediate) | `src.models.student` | **11,549,442** (~11.55M, ~9.58x smaller) |
| Vocab | WordPiece (Bengali) | `32,000` | Shared between Teacher & Student |
| `max_seq_len` | 128 | `cfg.model.max_seq_len` | Max headline tokens in dataset is 46 |

### Student Initialization Strategies:
1. **Random (`init_random`)**: Fresh weight initialization from standard normal distribution ($N(0, 0.02^2)$).
2. **Teacher Transfer (`init_from_teacher`)**: 
   - Embeddings & Encoder: Orthogonal PCA projection matrix ($P \in \mathbb{R}^{768 \times 256}$) preserving maximum representation variance.
   - Layers: First 4 transformer layers $\{0, 1, 2, 3\}$ of the Teacher transferred as Student's starting point.
   - Classifier: Dense projection and output layers mapped into 256-d space.

---

## 8. Config structure (key fields from `configs/base.yaml`)

```yaml
seed: 42

data:
  hf_dataset_id: "Motahar/banglabait"
  processed_dir:  "data/processed"
  splits_file:    "data/processed/splits.json"
  train_ratio:    0.80
  val_ratio:      0.10
  test_ratio:     0.10
  label_col:      "label"
  text_col:       "headline"      # mapped from HF's "title" column
  num_labels:     2
  min_char_length: 5

model:
  teacher_model_name: "csebuetnlp/banglabert"
  student_model_name: "csebuetnlp/banglabert_small"
  max_seq_len: 128
  dropout: 0.1

training:
  run_name:    "default_run"      # → experiments/<run_name>/
  epochs:      5
  batch_size:  32
  lr:          2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_clip: 1.0
  eval_strategy: "epoch"
  fp16: false
  dataloader_num_workers: 0       # MUST be 0 on Windows

distillation:
  temperature: 4.0
  alpha: 0.7                      # weight of soft-label KD loss
  intermediate_layers: true
  hidden_loss_weight: 0.1
```

---

## 9. Running tests

```powershell
# All tests (34 data + 15 utils = 49 total):
.venv\Scripts\python.exe -m pytest tests/ -v

# Just data cleaning:
.venv\Scripts\python.exe -m pytest tests/test_data_clean.py -v

# Just utils:
.venv\Scripts\python.exe -m pytest tests/test_utils.py -v
```

**Current test count:** 49 passing, 0 failing, 0 skipped.

---

## 10. Running the data pipeline (already done — don't re-run unless --overwrite-splits)

```powershell
# DO NOT re-run unless you want to regenerate splits (use --overwrite-splits explicitly)
.venv\Scripts\python.exe -m scripts.prepare_data --config configs/base.yaml --seed 42

# If you do need to regenerate:
.venv\Scripts\python.exe -m scripts.prepare_data --config configs/base.yaml --seed 42 --overwrite-splits
```

`data/processed/splits.json` is **committed to git** — the fixed split must never silently change.

---

## 11. What Phase 3 needs to implement

### Files to create
- `src/models/teacher.py` — wraps `AutoModelForSequenceClassification` on BanglaBERT with class-weighted loss
- `src/training/trainer.py` — HuggingFace `Trainer` wrapper or custom loop
- `src/training/dataset.py` — `torch.utils.data.Dataset` wrapping the DataFrames + tokenizer
- `src/evaluation/metrics.py` — `compute_metrics()` → accuracy, precision, recall, F1 (macro + per-class), confusion matrix
- `tests/test_metrics.py` — unit tests for `compute_metrics()` on toy inputs (currently a stub/placeholder)

### Scripts to wire up
- `scripts/train.py` — currently raises `NotImplementedError`; wire to `src/training/trainer.py`

### Key requirements for Phase 3
1. Load data using `load_splits()` + `apply_splits()` (do NOT re-split)
2. Use `ExperimentLogger` — save `config.yaml`, `run_info.json`, `metrics.json`, `metrics.csv` per run
3. Save best checkpoint (by `val_f1_macro`) to `experiments/<run>/checkpoints/best_val_f1/` using `exp.checkpoint_path()`
4. Save tokenizer alongside weights (needed for inference later)
5. Handle class imbalance: compute class weights from `train_df["label"].value_counts()` and pass to loss
6. **Do NOT evaluate on test set** — val set only during training
7. Log per-epoch: `epoch`, `split`, `train_loss`, `val_loss`, `val_accuracy`, `val_f1_macro`, `val_f1_clickbait`, `val_f1_not_clickbait`, `val_precision_macro`, `val_recall_macro`

### Suggested HuggingFace Trainer approach
```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
# TrainingArguments maps directly to cfg.training fields
# Use compute_metrics from src/evaluation/metrics.py
# class_weight via a custom Trainer subclass that overrides compute_loss()
```

---

## 12. Important design decisions already made (do NOT revisit without good reason)

| Decision | Rationale |
|----------|-----------|
| Pool all HF splits, make our own 80/10/10 | Full reproducibility control; HF splits had unequal sizes |
| Save row **indices** (not rows) to `splits.json` | Compact, easy to audit, avoids data duplication |
| OmegaConf (not argparse-only) | Supports hierarchical overrides cleanly; W&B-bolt-on ready |
| `ExperimentLogger` writes to `experiments/<run>/` | No external dashboard dependency; plain files are auditable |
| `dataloader_num_workers=0` | Windows multiprocessing incompatibility with HF datasets |
| NFC + whitelist symbol strip (not aggressive regex) | Bangla virama/chandrabindu/matras must NOT be stripped |
| Test set touched **exactly once** per model | Standard ML evaluation hygiene; stated in thesis brief |
| `--basetemp=.pytest_tmp` in pyproject.toml | Avoids Windows AppData Temp folder permission error |

---

## 13. Dependency install (if setting up from scratch)

```powershell
# 1. Create venv (if missing):
python -m venv .venv

# 2. Install core deps:
.venv\Scripts\pip.exe install omegaconf scikit-learn pandas numpy pytest datasets transformers huggingface_hub tokenizers

# 3. Install PyTorch (CUDA 12.8 build — matches system GPU):
.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu128 --no-deps

# 4. Install remaining thesis deps when needed:
.venv\Scripts\pip.exe install optuna shap lime scipy statsmodels onnx onnxruntime tqdm rich
```

> **Note on `--no-deps` for torch:** The venv's numpy was installed separately;
> using `--no-deps` avoids a numpy version conflict with a system-wide unsloth installation.
> Torch still works correctly since numpy is already present.

---

## 14. Thesis methodology phases — brief

```
Phase 1  DONE  Repo scaffold + utils
Phase 2  DONE  Data pipeline (BanglaBait: load → clean → stratified split → report)
Phase 3  TODO  Teacher fine-tuning (BanglaBERT → clickbait classifier)
Phase 4  TODO  Student distillation (TinyBanglaBERT, KD loss with temperature T + alpha)
Phase 5  TODO  Evaluation (macro-F1, confusion matrix, McNemar significance test,
               cross-domain generalization)
Phase 6  TODO  Explainability (SHAP token attributions, LIME perturbation)
Phase 7  TODO  Efficiency (param count, model size, ONNX export, latency @ batch 1/8/32,
               peak memory)
Phase 8  TODO  Ablation (grid over T in {2,4,6,8}, alpha in {0.3,0.5,0.7,0.9},
               +/- intermediate hidden-layer matching)
```

---

## 15. File quick-reference

| File | Status | Description |
|------|--------|-------------|
| `configs/base.yaml` | ✅ | All hyperparameter defaults |
| `configs/teacher_baseline.yaml` | ✅ | Teacher fine-tuning overrides |
| `configs/student_distillation.yaml` | ✅ | Distillation overrides |
| `src/utils/seeding.py` | ✅ | `seed_everything()` |
| `src/utils/config_loader.py` | ✅ | `load_config()`, `save_config()`, `cfg_to_dict()` |
| `src/utils/logging_utils.py` | ✅ | `ExperimentLogger` |
| `src/data/load.py` | ✅ | `load_raw()` — HF Hub + local |
| `src/data/clean.py` | ✅ | `clean_headline()`, `clean_dataframe()` |
| `src/data/split.py` | ✅ | `create_splits()`, `load_splits()`, `apply_splits()` |
| `src/data/report.py` | ✅ | `generate_report()` |
| `data/processed/splits.json` | ✅ | Fixed split (committed) |
| `data/processed/data_report.md` | ✅ | Real dataset stats |
| `scripts/prepare_data.py` | ✅ | Full data pipeline CLI |
| `scripts/train.py` | ⬜ | Stub — implement Phase 3 |
| `scripts/evaluate.py` | ⬜ | Stub — implement Phase 5 |
| `scripts/explain.py` | ⬜ | Stub — implement Phase 6 |
| `scripts/benchmark.py` | ⬜ | Stub — implement Phase 7 |
| `src/models/teacher.py` | ⬜ | Create in Phase 3 |
| `src/training/trainer.py` | ⬜ | Create in Phase 3 |
| `src/training/dataset.py` | ⬜ | Create in Phase 3 |
| `src/evaluation/metrics.py` | ⬜ | Create in Phase 3/5 |
| `tests/test_utils.py` | ✅ | 15 tests passing |
| `tests/test_data_clean.py` | ✅ | 34 tests passing |
| `tests/test_metrics.py` | ⬜ | Stub — implement Phase 3 |
