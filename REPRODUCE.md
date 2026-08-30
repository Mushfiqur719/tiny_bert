# TinyBanglaBERT: End-to-End Reproduction Guide

This guide details the exact sequential commands and environment configurations required to reproduce all findings, models, evaluations, explainability analyses, and thesis artifacts from a clean clone.

---

## 1. System Requirements & Hardware Assumptions

- **Operating System**: Windows 11 / 10, Linux (Ubuntu 20.04+), or macOS (Apple Silicon).
- **Python Version**: Python 3.12 (or 3.10+).
- **GPU (Optional but Recommended for Training)**:
  - Tested on: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 12.8).
  - VRAM: $\ge$ 4 GB (Batch size 16/32 with FP16 enabled).
- **CPU (for Inference Benchmarks)**: Modern x86-64 / ARM64 processor.

---

## 2. Environment Setup

```powershell
# 1. Clone the repository
git clone https://github.com/mushfiqur-rahman/tiny_bert.git
cd tiny_bert

# 2. Initialize Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Install all production and development dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # Or standard PyTorch
pip install transformers datasets evaluate accelerate scikit-learn pandas numpy matplotlib seaborn scipy statsmodels shap lime onnx onnxruntime onnxscript pytest pyyaml tabulate
```

---

## 3. Step-by-Step Execution Pipeline

### Phase 1: Data Ingestion, NFC Normalization & Splitting
- **Command**:
  ```powershell
  python -m src.data.load
  python -m src.data.split --seed 42
  ```
- **Expected Runtime**: ~15 seconds.
- **Output**: `data/processed/splits.json` (12,044 train / 1,506 val / 1,506 test).

---

### Phase 2: Teacher Fine-Tuning (BanglaBERT Baseline)
- **Command**:
  ```powershell
  python -m src.training.train_teacher --config configs/teacher.yaml
  ```
- **Expected Runtime**: ~3.5 minutes on RTX 5050 GPU (5 epochs, batch size 16, FP16).
- **Expected Test Performance**: Macro-F1: `0.8233`, Accuracy: `84.20%`.
- **Output**: Model checkpoint and `experiments/teacher/train_soft_labels.pt`.

---

### Phase 3: Distillation Hyperparameter Tuning ($T, \alpha$ Grid Search)
- **Command**:
  ```powershell
  python -m scripts.tune_kd --trials 20
  ```
- **Expected Runtime**: ~5 minutes.
- **Output**: `experiments/tuning/kd_grid_results.json` ($T^*=2.0, \alpha^*=0.1$).

---

### Phase 4: Student 4-Way Ablation Training
Train all 4 factorial combinations on the CLI:

```powershell
# 1. Random Init + Supervised Baseline
python -m scripts.train --config configs/student_random_baseline.yaml

# 2. Random Init + Knowledge Distillation
python -m scripts.train --config configs/student_random_distilled.yaml

# 3. Teacher-Projected Init + Supervised Baseline
python -m scripts.train --config configs/student_teacher_init_baseline.yaml

# 4. Teacher-Projected Init + Knowledge Distillation (Winning Model)
python -m scripts.train --config configs/student_teacher_init_distilled.yaml
```
- **Expected Runtime**: ~45 seconds per run on RTX 5050 GPU (~3.5 minutes total).

---

### Phase 5: Ablation Summary & Final Model Selection
- **Command**:
  ```powershell
  python -c "from src.evaluation.ablation import run_ablation_analysis; df, best = run_ablation_analysis(); print('Selected Best Model:', best)"
  ```
- **Output**: `experiments/ablation_summary.md`, `experiments/ablation_table.csv`, and copied model to `experiments/tinybanglabert_final/`.

---

### Phase 6: Efficiency, CPU Latency & ONNX Benchmarking
- **Command**:
  ```powershell
  python -m scripts.benchmark --n-iterations 200
  ```
- **Expected Runtime**: ~45 seconds (200 single-headline passes across 4 runtimes).
- **Output**: `experiments/efficiency_report.md`, `experiments/onnx/tinybanglabert.onnx`, `experiments/onnx/tinybanglabert_int8.onnx`.

---

### Phase 7: Explainability (SHAP & LIME) & Cross-Method Agreement
- **Command**:
  ```powershell
  python -m scripts.explain --generate-thesis-samples --sample-size 15
  ```
- **Expected Runtime**: ~55 seconds.
- **Output**: `experiments/explainability/examples.md`, `experiments/explainability/agreement_report.md`.

---

### Phase 8: Statistical Significance Testing (McNemar's Exact Test)
- **Command**:
  ```powershell
  python -c "from src.evaluation.mcnemar_test import run_full_mcnemar_suite; df, rpt = run_full_mcnemar_suite()"
  ```
- **Expected Runtime**: ~25 seconds.
- **Output**: `experiments/significance_report.md`, `experiments/significance_table.csv`.

---

### Phase 9: Cross-Domain Generalization Evaluation
- **Command**:
  ```powershell
  python -c "from src.evaluation.crossdomain_eval import run_crossdomain_evaluation; df, rpt = run_crossdomain_evaluation('abdalimran/BaitBuster-Bangla')"
  ```
- **Expected Runtime**: ~40 seconds on GPU (10,000 samples).
- **Output**: `experiments/crossdomain_report.md`, `experiments/crossdomain_table.csv`.

---

### Phase 10: Generate 300 DPI Publication Figures
- **Command**:
  ```powershell
  python -m scripts.generate_all_figures
  ```
- **Output**: `experiments/figures/fig1_ablation_f1_accuracy.png` through `fig6_confusion_matrices.png`.

---

## 4. Full Unit Test Suite Verification

Verify all modules, loaders, losses, and metrics pass with 100% coverage:
```powershell
pytest tests/ -v
```
**Expected Result**: `88 passed in ~22s`.
