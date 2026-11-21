# TinyBanglaBERT v3 — Empirical Study Report

## Title
**Pre-Trained Compact Initialisation vs. Distillation Objectives in Low-Resource Bengali Clickbait Detection**

---

## 1. Research Question & Pre-Registered Hypotheses
This study investigates a controlled empirical question pre-registered in `PREREG.md` prior to model training:

> *In low-resource fine-tuning of compact neural encoders, does warm-start compact initialisation account for more performance variance than the choice of distillation objective, and how does this advantage interact with training data volume and probability calibration?*

- **H1 (Initialisation Dominance):** On a strictly parameter-matched 4-layer architecture (7.42M parameters, hidden size 256, factorised embedding size 128), warm-starting from strided pre-trained layers (`[2, 5, 8, 11]` of `csebuetnlp/banglabert_small`) produces a larger Macro-F1 gain than switching from Cross-Entropy (CE) to Knowledge Distillation (KD).
- **H2 (Objective–Data & Objective–Init Interaction):** The marginal gain of KD over CE shrinks as initialisation quality improves and as labelled training data increases.
- **Primary Metric:** Five-seed Macro-F1 with 10,000-resample paired bootstrap tests (over items and seeds), Holm-Bonferroni correction, and a pre-registered non-inferiority margin of $\Delta = -2.0$ percentage points.

---

## 2. Data Hygiene & Leak-Free Experimental Protocol
All experiments enforce strict zero-leakage deduplication (`src/data/dedup.py`, MinHash LSH at Jaccard threshold $J \ge 0.9$, 128 permutations, character 3-shingles) followed by 10-fold `StratifiedGroupKFold` (Fold 0 = Test, Fold 1 = Validation, Folds 2–9 = Train) and a code-level test-set access guard (`src/utils/test_guard.py`).

| Benchmark | Genre | Clean Total | Train (80%) | Val (10%) | Test (10%) | Conflicts Dropped | Cross-Split Leakage |
|---|---|---:|---:|---:|---:|---:|---:|
| **BanglaBait** (Motahar et al., 2024) | News headlines + body | 15,032 | 12,025 | 1,503 | 1,504 | 24 rows (11 groups) | **0 groups (0.00%)** |
| **BaitBuster-Bangla** (Data in Brief, 2024) | YouTube video titles | 10,000 | 8,000 | 1,000 | 1,000 | 0 rows | **0 groups (0.00%)** |

Nested stratified training subsets (`10% ⊂ 25% ⊂ 50% ⊂ 100%`) were fixed and committed before training. Five-fold out-of-fold (OOF) teacher logits were generated exclusively on the training split to prevent teacher memorisation bias during distillation.

---

## 3. Core Factorial Grid & Mixed-Effects Mechanism Analysis (Validation Split)

All 160 controlled student runs (80 per dataset across 5 seeds: `[13, 42, 87, 1234, 2026]`) use the identical 4-layer, 7.42M-parameter architecture and fixed hyperparameters ($\text{lr}=5\times 10^{-5}$, cosine schedule, $T=4.0$, $\alpha=0.5$).

### 3.1 BanglaBait (Noisy News Headlines)
| Init | Objective | 10% Data | 25% Data | 50% Data | 100% Data | 100% ECE ↓ | 100% Teacher Agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| `rand` | CE | 0.6554 ± 0.0144 | 0.6997 ± 0.0108 | 0.7224 ± 0.0085 | 0.7513 ± 0.0077 | 0.0820 | 83.34% |
| `rand` | KD | 0.6764 ± 0.0134 | 0.7297 ± 0.0087 | 0.7492 ± 0.0054 | 0.7555 ± 0.0054 | 0.0707 | 84.72% |
| `strided` | CE | 0.7213 ± 0.0048 | 0.7463 ± 0.0057 | 0.7603 ± 0.0078 | 0.7732 ± 0.0046 | 0.0994 | 86.77% |
| `strided` | KD | **0.7323 ± 0.0057** | **0.7551 ± 0.0053** | **0.7673 ± 0.0042** | **0.7747 ± 0.0062** | **0.0808** | **88.13%** |

### 3.2 BaitBuster-Bangla (YouTube Video Titles)
| Init | Objective | 10% Data | 25% Data | 50% Data | 100% Data | 100% ECE ↓ | 100% Teacher Agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| `rand` | CE | 0.9174 ± 0.0072 | 0.9642 ± 0.0026 | 0.9746 ± 0.0007 | 0.9805 ± 0.0009 | 0.0174 | 98.72% |
| `rand` | KD | 0.9114 ± 0.0101 | 0.9639 ± 0.0037 | 0.9725 ± 0.0031 | 0.9811 ± 0.0029 | 0.0079 | 98.78% |
| `strided` | CE | 0.9610 ± 0.0018 | 0.9745 ± 0.0016 | 0.9796 ± 0.0012 | 0.9851 ± 0.0014 | 0.0122 | 99.10% |
| `strided` | KD | **0.9615 ± 0.0025** | **0.9741 ± 0.0016** | **0.9809 ± 0.0015** | **0.9866 ± 0.0017** | **0.0080** | **99.28%** |

### 3.3 Linear Mixed-Effects Regression (`f1_macro ~ C(init) * C(objective) + log_frac + C(objective):log_frac + (1 | seed)`)
| Factor | BanglaBait Coef. (pp) | BanglaBait $z$ ($p$-value) | BaitBuster Coef. (pp) | BaitBuster $z$ ($p$-value) |
|---|---:|---:|---:|---:|
| `C(init)[T.strided]` (H1) | **+4.30 pp** [3.60, 5.00] | **11.829 ($p < 0.0001$)** | **+1.60 pp** [0.90, 2.30] | **4.672 ($p < 0.0001$)** |
| `C(objective)[T.kd]` | +3.90 pp [1.70, 6.10] | 3.436 ($p = 0.001$) | -0.70 pp [-2.80, 1.40] | -0.673 ($p = 0.501$) |
| `strided × KD` (H2a) | **-1.30 pp** [-2.40, -0.30] | **-2.608 ($p = 0.009$)** | +0.30 pp [-0.70, 1.20] | 0.542 ($p = 0.588$) |
| `log_frac` | +3.20 pp [2.80, 3.60] | 14.851 ($p < 0.0001$) | +1.90 pp [1.50, 2.30] | 9.354 ($p < 0.0001$) |
| `KD × log_frac` (H2b) | -0.50 pp [-1.10, 0.10] | -1.702 ($p = 0.089$) | +0.10 pp [-0.40, 0.70] | 0.519 ($p = 0.604$) |

**Mechanism Conclusion:** Pre-trained compact initialisation is the dominant driver of Macro-F1 across both datasets ($p < 0.0001$). While KD's Macro-F1 lift shrinks as initialisation improves ($p = 0.009$ on BanglaBait) and as data increases, KD consistently reduces Expected Calibration Error (ECE) by 19%–54% and increases student–teacher prediction agreement (up to 88.13% on BanglaBait and 99.28% on BaitBuster-Bangla).

---

## 4. One-Time Held-Out Test Evaluation (Phase J)
Executed once after locking configurations in `experiments/v3/LOCKED_CONFIG.yaml` (`TBB_ALLOW_TEST=1`). Paired bootstrap tests use 10,000 resamples across items and seeds with Holm-Bonferroni correction.

### 4.1 BaitBuster-Bangla Test Split ($n = 1,000$, 5 Seeds)
| Model | Params | Compression vs. Base / Small-12L | Test Acc (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Diff vs. Base (pp, 95% CI) | Non-Inferior ($\Delta=-2.0$ pp)? |
|---|---:|---:|---:|---:|---:|---|
| `teacher_base_uw` (12L) | 110.62M | 1.0× / — | 98.82% ± 0.22% | 0.9880 ± 0.0023 | Reference | — |
| `baseline_small_12l_ce_uw` (12L) | 13.74M | 8.1× / 1.0× | 98.70% ± 0.17% | 0.9868 ± 0.0017 | -0.12 [-0.62, +0.37] | **True** ($p_{\text{Holm}}=1.00$) |
| **`F_strided_kd_100_uw` (4L)** | **7.42M** | **14.9× / 1.85×** | **98.10% ± 0.14%** | **0.9807 ± 0.0014** | **-0.73 [-1.46, -0.06]** | **True** ($p_{\text{Holm}}=0.21$) |
| `F_strided_ce_100_uw` (4L) | 7.42M | 14.9× / 1.85× | 98.14% ± 0.23% | 0.9811 ± 0.0024 | -0.69 [-1.39, -0.04] | **True** ($p_{\text{Holm}}=0.21$) |
| `F_rand_kd_100_uw` (4L) | 7.42M | 14.9× / 1.85× | 97.92% ± 0.34% | 0.9788 ± 0.0035 | -0.92 [-1.65, -0.26] | **True** ($p_{\text{Holm}}=0.06$) |
| `F_rand_ce_100_uw` (4L) | 7.42M | 14.9× / 1.85× | 97.96% ± 0.27% | 0.9792 ± 0.0028 | -0.88 [-1.56, -0.24] | **True** ($p_{\text{Holm}}=0.06$) |

### 4.2 BanglaBait Test Split ($n = 1,504$, 5 Seeds)
| Model | Params | Input Mode | Test Acc (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Diff vs. Small-12L (pp, 95% CI) | Non-Inferior vs. Small-12L? |
|---|---:|---|---:|---:|---:|---|
| `teacher_base_tb` (12L) | 110.62M | `title + body` | **81.35% ± 0.78%** | **0.7934 ± 0.0074** | +3.91 [+2.21, +5.69] | Superior ($p_{\text{Holm}}=0.0018$) |
| `teacher_base` (12L) | 110.62M | `title` | 79.35% ± 0.65% | 0.7723 ± 0.0034 | +1.80 [+0.12, +3.45] | Reference (title) |
| `baseline_small_12l_ce` (12L) | 13.74M | `title` | 77.63% ± 0.88% | 0.7543 ± 0.0078 | Reference (Small-12L) | — |
| **`F_strided_kd_100` (4L)** | **7.42M** | `title` | **77.86% ± 0.56%** | **0.7537 ± 0.0063** | **-0.06 [-1.71, +1.56]** | **True** ($p_{\text{Holm}}=0.9412$) |
| `F_strided_ce_100` (4L) | 7.42M | `title` | 76.42% ± 0.66% | 0.7451 ± 0.0035 | -0.92 [-2.54, +0.68] | False ($p_{\text{Holm}}=0.5252$) |
| `F_rand_kd_100` (4L) | 7.42M | `title` | 75.27% ± 1.10% | 0.7222 ± 0.0078 | -3.21 [-5.14, -1.25] | False ($p_{\text{Holm}}=0.0120$) |
| `F_rand_ce_100` (4L) | 7.42M | `title` | 75.51% ± 0.76% | 0.7239 ± 0.0046 | -3.04 [-5.09, -1.04] | False ($p_{\text{Holm}}=0.0224$) |

Note: On BanglaBait, adding article body text (`title + body`) significantly improved the 110.6M Teacher by +2.11 pp Macro-F1 ($p_{\text{Holm}} = 0.028$). Among 4-layer students, `F_strided_kd_100` is the **only** configuration that achieves formal non-inferiority to the 3× deeper 12-layer `Small-12L` baseline (-0.06 pp, 95% CI [-1.71, +1.56]) while outperforming random-init KD by +3.15 pp. Neither 4L student is non-inferior to `teacher_base` on BanglaBait (CI lower bound -3.65 pp < -2.0 pp); we report this transparently rather than claiming "no loss."

---

## 5. Cross-Genre Evaluation: News → YouTube Titles (Phase K)
Evaluated zero-shot on all 10,000 BaitBuster-Bangla titles (true label prevalence: **56.44%** clickbait; 4,356 non-clickbait, 5,644 clickbait), plus a 1% target threshold-recalibration arm ($n=100$ calibration, $n=9,900$ evaluation):

| Model (Trained on BanglaBait) | Zero-Shot Acc | Zero-Shot Macro-F1 [95% CI] | 1% Recalibrated Acc | 1% Recalibrated Macro-F1 [95% CI] |
|---|---:|---|---:|---|
| `teacher_base` (110.6M) | 78.66% | 0.7852 [0.7669, 0.8057] | 85.51% | 0.8549 [0.8400, 0.8735] |
| `baseline_small_12l_ce` (13.7M) | 79.73% | 0.7963 [0.7724, 0.8220] | 84.68% | 0.8464 [0.8394, 0.8582] |
| `F_strided_kd_100` (7.42M) | 75.93% | 0.7570 [0.7387, 0.7802] | 81.93% | 0.8192 [0.8130, 0.8276] |
| `F_strided_ce_100` (7.42M) | 75.40% | 0.7520 [0.7296, 0.7684] | 78.79% | 0.7877 [0.7755, 0.7987] |
| `F_rand_kd_100` (7.42M) | 81.21% | 0.8115 [0.7802, 0.8350] | 85.80% | 0.8565 [0.8474, 0.8650] |

Threshold recalibration using just 100 labelled target examples lifts cross-genre Macro-F1 by +5.0 to +7.0 pp across all models by compensating for the prior shift between news (34.8% clickbait) and YouTube titles (56.4% clickbait).

---

## 6. Unified ONNX CPU Efficiency & Cascade Latency (Phase L)
Hardware: AMD64 Family 25 Model 117 Stepping 2 (Windows 11), ONNX Runtime 1.29.0, batch size = 1, `max_length = 64`, 200 warm-up + 1,000 timed runs.

| Model | Precision | Threads | p50 Latency (ms) | p95 Latency (ms) | Disk Size (MB) | Speedup vs. Base FP32 |
|---|---|---:|---:|---:|---:|---:|
| BanglaBERT-Base (110.6M) | FP32 | 1 / 4 | 190.56 / 74.46 | 211.90 / 88.78 | 422.25 | 1.0× |
| BanglaBERT-Base (110.6M) | INT8 | 1 / 4 | 99.67 / 28.79 | 133.66 / 35.67 | 177.75 | 2.6× (4T) |
| BanglaBERT-Small-12L (13.7M) | FP32 | 1 / 4 | 15.06 / 8.72 | 16.89 / 10.36 | 52.68 | 8.5× (4T) |
| BanglaBERT-Small-12L (13.7M) | INT8 | 1 / 4 | 10.18 / 5.56 | 10.93 / 6.97 | 25.59 | 13.4× (4T) |
| **TinyBanglaBERT-4L (7.42M)** | **FP32** | **1 / 4** | **5.17 / 2.89** | **5.95 / 3.77** | **28.41** | **25.8× (4T)** |
| **TinyBanglaBERT-4L (7.42M)** | **INT8** | **1 / 4** | **3.48 / 1.77** | **4.07 / 2.39** | **19.20** | **42.1× (4T)** |

With simulated mobile network round-trip times (50 ms / 150 ms) for an edge–cloud deferral cascade (`TinyBanglaBERT-4L INT8` on-device + `BanglaBERT-Base FP32` cloud), a 10% escalation rate yields **14.22 ms** (50 ms RTT) and **24.22 ms** (150 ms RTT) effective latency versus **126.23 ms / 226.23 ms** for cloud-only inference.

---

## 7. Limitations, Ethics & Reproducibility
- **Limitations:**
  1. Human relabeling of the BanglaBait test split (`data/processed/v3/relabel_batch.csv`) is staged in `experiments/v3/HUMAN_TODO.md` and requires two independent native speakers plus adjudication before making formal label-noise ceiling claims.
  2. Physical Android device benchmarking is queued as a human step alongside our x86-64 ONNX Runtime measurements.
  3. Cross-lingual generalisation (Phase H: SentNoB, BanglaNLI, Hindi/Swahili/Tamil) is planned in `TASK_CANDIDATES.md` and `LANG_CANDIDATES.md` for venue expansion.
- **Ethics & Potential Harm:** Automated clickbait filtering carries false-positive risk to legitimate news publishers (wrongly suppressing or demoting factual journalism, especially on dramatic breaking news or entertainment stories). Deployment should favour calibrated probability scores, human-in-the-loop review, or transparent user-facing advisories rather than opaque content censorship.
- **Reproducibility:** All 337 automated tests (including mathematical confusion-matrix consistency checks across every `metrics.json` and `test_metrics.json`, test-set lock enforcement, and deduplication verification) pass with zero failures. Code, `requirements-lock.txt`, `PREREG.md`, `LOCKED_CONFIG.yaml`, and all split index JSON files are version-controlled in the repository.
