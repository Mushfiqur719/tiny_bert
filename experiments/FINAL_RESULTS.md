# TinyBanglaBERT v3 — Final Results Summary

This document mirrors the verified v3 empirical results across all completed phases (**A–C, E–G, I–L**). Prior drafts are archived in `experiments/v1_archive/`. See [IEEE_PAPER_REPORT.md](IEEE_PAPER_REPORT.md) and [ieee_manuscript.tex](ieee_manuscript.tex) for the full manuscript and figures.

---

## 1. Held-Out Test Performance (5 Seeds, 10k Paired Bootstrap, Holm-Corrected)

| Dataset / Model | Params | Test Acc (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Non-Inferior ($\Delta = -2.0$ pp) |
|---|---:|---:|---:|---|
| *BanglaBait* — Teacher (`title+body`) | 110.62M | 81.35% ± 0.78% | 0.7934 ± 0.0074 | Superior (+2.11 pp vs Title, $p_{\text{Holm}}=0.028$) |
| *BanglaBait* — Teacher (`title`) | 110.62M | 79.35% ± 0.65% | 0.7723 ± 0.0034 | Reference (Title) |
| *BanglaBait* — Small-12L (`title`) | 13.74M | 77.63% ± 0.88% | 0.7543 ± 0.0078 | Reference (Small-12L) |
| ***BanglaBait* — 4L Strided + KD** | **7.42M** | **77.86% ± 0.56%** | **0.7537 ± 0.0063** | **True vs. Small-12L (-0.06 pp) / False vs. Base (-1.86 pp)** |
| *BanglaBait* — 4L Strided + CE | 7.42M | 76.42% ± 0.66% | 0.7451 ± 0.0035 | False (Both) |
| *BanglaBait* — 4L Rand + CE | 7.42M | 75.51% ± 0.76% | 0.7239 ± 0.0046 | False (Both) |
| *BanglaBait* — 4L Rand + KD | 7.42M | 75.27% ± 1.10% | 0.7222 ± 0.0078 | False (Both) |
| *BaitBuster* — Teacher Base (12L) | 110.62M | 98.82% ± 0.22% | 0.9880 ± 0.0023 | Reference (1.18% error) |
| *BaitBuster* — Small-12L (CE) | 13.74M | 98.70% ± 0.17% | 0.9868 ± 0.0017 | True (vs. Base) |
| ***BaitBuster* — 4L Strided + KD** | **7.42M** | **98.10% ± 0.14%** | **0.9807 ± 0.0014** | **True (Both; 1.90% error, near ceiling)** |
| *BaitBuster* — 4L Strided + CE | 7.42M | 98.14% ± 0.23% | 0.9811 ± 0.0024 | True (Both) |
| *BaitBuster* — 4L Rand + CE | 7.42M | 97.96% ± 0.27% | 0.9792 ± 0.0028 | True (Both) |
| *BaitBuster* — 4L Rand + KD | 7.42M | 97.92% ± 0.34% | 0.9788 ± 0.0035 | True (Both) |

*Direct BanglaBait Test Contrasts:* `Strided+KD` vs. `Rand+KD` = **+3.14 pp** ($p_{\text{Holm}} = 0.0012$); `Strided+CE` vs. `Rand+CE` = +2.12 pp ($p_{\text{Holm}} = 0.090$); `Strided+KD` vs. `Strided+CE` = +0.86 pp ($p_{\text{Holm}} = 0.587$); `Rand+KD` vs. `Rand+CE` = -0.17 pp ($p_{\text{Holm}} = 1.00$).  
*BaitBuster Caveat:* Auditing the 10k gold subset by `channel_id` reveals 13 channels where 94.7% of non-clickbait rows come from a single channel (*Independent Television*), explaining the ~98% ceiling.

---

## 2. Centered Full-Interaction Mixed-Effects Findings (160 Controlled 5-Seed Runs)
- **H1 Confirmed via Direct Wald Contrast ($\beta_{\text{strided}} - \beta_{\text{kd}}$):** **+2.26 pp** ($\chi^2 = 64.37, p = 1.03 \times 10^{-15}$) on BanglaBait; **+1.78 pp** ($\chi^2 = 53.56, p = 2.51 \times 10^{-13}$) on BaitBuster-Bangla.
- **H2 Partially Supported:** `strided × kd` is significant on BanglaBait (**-1.30 pp**, $p = 0.001$) but not on BaitBuster ($p = 0.448$); `kd × log_frac_c` is marginal on BanglaBait ($p = 0.052$) and null on BaitBuster ($p = 0.395$).
- **Calibration:** KD lowers Expected Calibration Error (ECE) by **13.8%–54.6%** across all four 100%-data cells.

---

## 3. Cross-Genre News → YouTube ($N = 10,000$, True Prevalence = 56.44%)
- Threshold recalibration on 1% ($n = 100$) target YouTube titles lifts Macro-F1 by +5 to +7 pp: `teacher_base` `0.7852 → 0.8549`, `Small-12L` `0.7963 → 0.8464`, `F_strided_kd_100` `0.7570 → 0.8192`, `F_strided_ce_100` `0.7520 → 0.7877`, and **`F_rand_kd_100` `0.8115 → 0.8565`** (best cross-genre).

---

## 4. Unified ONNX CPU Efficiency (Batch 1, Max Length 64, ONNX Runtime 1.29.0)
- **TinyBanglaBERT-4L (4 threads):** **1.77 ms p50** (INT8, 19.20 MB) and **2.89 ms p50** (FP32, 28.41 MB) — **16.3× like-for-like INT8 speedup** (vs. 28.79 ms Base INT8), **25.8× like-for-like FP32 speedup** (vs. 74.46 ms Base FP32; 42.1× cross-precision), and **3.1×** over Small-12L INT8 (5.56 ms).
- **Cascade Latency (10% escalation):** **14.22 ms** (50 ms RTT) / **24.22 ms** (150 ms RTT) vs. **124.46 ms / 224.46 ms** cloud-only Base.
