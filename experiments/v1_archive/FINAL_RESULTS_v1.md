# TinyBanglaBERT: Comprehensive Thesis Results & Empirical Evaluation

**Authors / Thesis Project:** TinyBanglaBERT — Lightweight Knowledge-Distilled Transformer for Real-Time Bengali Clickbait Detection  
**Evaluation Benchmark:** BanglaBait In-Domain Test Split ($N = 1,506$, plus Leak-Free Deduplicated Subset $N = 1,351$) & BaitBuster-Bangla Cross-Domain Gold Benchmark ($N = 10,000$, verified `0.0%` overlap)  
**Target Deployment:** Edge Clients (Browser Extensions / Mobile Devices via ONNX Runtime & Dynamic INT8 Quantization)

---

## 📌 Abstract Headline Numbers & Executive Summary

> ### 🏆 Key Findings for Thesis Abstract:
> - **High In-Domain Performance Retention:** **TinyBanglaBERT-4L** (`7.42M` parameters) achieves **`80.54%` Test Accuracy** (`96.19%` of Teacher's `83.73%`) and **`0.7895` Test Macro-F1** (**`96.59%` of the 110.6M Teacher's `0.8174`**) in FP32, and **`80.81%` Test Accuracy / `0.7917` Macro-F1 (`96.86%` Teacher retention)** after dynamic INT8 quantization.
> - **Leak-Free Audit Verification:** On the strictly deduplicated leak-free test subset ($N = 1,351$, removing 155 train-overlapping headlines), TinyBanglaBERT-4L achieves **`80.09%` Accuracy** and **`0.7768` Macro-F1**, retaining **`96.64%`** of the Teacher's leak-free Macro-F1 (`0.8038`).
> - **Extreme Parameter & Footprint Compression:** Reduces parameter count by **`14.91x`** (`7.42M` vs. `110.62M`, **93.29% parameter reduction**), compressing disk size from `422.76 MB` down to **`28.31 MB` (FP32, 14.93x smaller)** and **`7.29 MB` (Dynamic INT8, 57.99x smaller)**.
> - **Real-Time Edge Speedup:** Achieves a **`13.40x` CPU speedup** in ONNX FP32 (`9.20 ms` vs. `123.36 ms`) and **`30.83x` CPU speedup in ONNX INT8** (**`4.00 ms` per headline / `249.9 headlines/sec`**), comfortably satisfying the sub-50ms browser extension rendering budget with **zero quantization accuracy loss (`+0.22%` F1 gain)**.
> - **Statistical Significance (Holm-Corrected McNemar):** Knowledge distillation significantly improves the PCA-initialized 4-layer student (`+2.59%` Acc, `+1.43%` F1, raw $p = 0.0094$, Holm-adjusted $p = 0.0376$). Crucially, 3x layer compression from 12-layer `BanglaBERT-Small-12L + KD` (`13.73M`, `0.7926` F1) to 4-layer `TinyBanglaBERT-4L` (`7.42M`, `0.7895` F1) shows **no statistically significant difference in error rates** ($p = 0.6592$, $b=96, c=89$).
> - **Cross-Domain Generalization ($N = 10,000$):** On `BaitBuster-Bangla`, TinyBanglaBERT-4L suffers a **34% smaller domain shift drop** than the Teacher (`-4.36%` vs. `-6.60%` Macro-F1), retaining **`99.27%` of Teacher Macro-F1** (`0.7459` vs. `0.7514`) and **beating the 110.6M Teacher on Clickbait F1** (`0.7288` vs. `0.7253`, **`100.48%` retention**).

---

## 1. In-Domain Benchmark: Teacher vs. TinyBanglaBERT Variants

Evaluation conducted on the stratified test split of the BanglaBait dataset ($N = 1,506$ headlines: $982$ not-clickbait, $524$ clickbait).

| Model Architecture | Layers / Hidden | Parameter Count | Disk Size | Accuracy | Macro F1 | Macro Precision | Macro Recall | Clickbait F1 (Class 1) | Non-CB F1 (Class 0) | Confusion Matrix `[TN, FP, FN, TP]` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (BanglaBERT-Base)** | 12L / 768H | 110,618,882 | 422.76 MB | **83.73%** *(84.00% cal)* | **0.8174** *(0.8220 cal)* | 0.8243 | 0.8121 | **0.7572** | **0.8777** | `[879, 103, 142, 382]` |
| **BanglaBERT-Small-12L + KD** | 12L / 256H | 13,731,586 | 52.40 MB | **81.14%** *(81.34% cal)* | **0.7926** *(0.7969 cal)* | 0.7921 | 0.7931 | 0.7300 | 0.8551 | `[838, 144, 140, 384]` |
| **TinyBanglaBERT-4L (FP32)** | 4L / 256H | **7,420,418** | **28.31 MB** | **80.54%** | **0.7895** | 0.7853 | 0.7952 | **0.7314** | 0.8475 | `[814, 168, 125, 399]` |
| **TinyBanglaBERT-4L (INT8)** ⭐ | 4L / 256H | **7,420,418** | **7.29 MB** | **80.81%** | **0.7917** | 0.7885 | 0.7953 | **0.7331** | **0.8503** | — |

```
                                  KNOWLEDGE RETENTION BREAKDOWN (INT8 / FP32)
                      ┌─────────────────────────────────────────────────────────────────┐
      Macro-F1 Score  │ Teacher: 0.8174  ────────►  TinyBanglaBERT-4L: 0.7917 (0.7895)  │ (96.86% / 96.59% Retained)
      Accuracy        │ Teacher: 83.73%  ────────►  TinyBanglaBERT-4L: 80.81% (80.54%)  │ (96.51% / 96.19% Retained)
      Parameter Count │ Teacher: 110.6M  ────────►  TinyBanglaBERT-4L: 7.42M            │ (14.91x Smaller)
      Disk Footprint  │ Teacher: 422.8M  ────────►  TinyBanglaBERT-4L: 7.29MB (28.3MB)  │ (57.99x / 14.93x Smaller)
                      └─────────────────────────────────────────────────────────────────┘
```

---

## 2. $3 \times 2$ Factorial Ablation & Depth-Scaling Study

| Model Configuration | Initialization Strategy | Training Objective | Parameters | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Macro Prec | Test Macro Rec | Test Clickbait F1 | Test Non-CB F1 | Confusion Matrix `[TN, FP, FN, TP]` |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher Baseline** | Pre-trained (12L Base) | Supervised CE | 110.62M | 0.8078 | 83.73% | 0.8174 | 0.8243 | 0.8121 | 0.7572 | 0.8777 | `[879, 103, 142, 382]` |
| **1. Random + Supervised** | Random Gaussian (4L) | Supervised CE | 11.55M | 0.7396 | 76.56% | 0.7390 | 0.7418 | 0.7366 | 0.6556 | 0.8223 | `[817, 165, 188, 336]` |
| **2. Random + Distilled** | Random Gaussian (4L) | KD ($T=4, \alpha=0.1$) | 11.55M | 0.7448 | 74.63% | 0.7313 | 0.7272 | 0.7432 | 0.6678 | 0.7948 | `[740, 242, 140, 384]` |
| **3. Teacher-PCA + Supervised** | PCA Projection (4L) | Supervised CE | 11.55M | 0.7668 | 75.83% | 0.7472 | 0.7440 | 0.7653 | 0.6941 | 0.8002 | `[729, 253, 111, 413]` |
| **4. Teacher-PCA + Distilled** | PCA Projection (4L) | KD ($T=4, \alpha=0.1$) | 11.55M | 0.7679 | **78.42%** | **0.7615** | 0.7623 | 0.7606 | 0.6878 | 0.8351 | `[823, 159, 166, 358]` |
| **5. Small-Pretrained 4L + CE** | Strided Pre-trained (4L) | Supervised CE | **7.42M** | 0.7934 | 79.75% | 0.7830 | 0.7777 | 0.7922 | 0.7269 | 0.8391 | `[795, 187, 118, 406]` |
| **6. TinyBanglaBERT-4L (KD)** ⭐ | Strided Pre-trained (4L) | KD ($T=4, \alpha=0.5$) | **7.42M** | **0.8000** | **80.54%** | **0.7895** | **0.7853** | **0.7952** | **0.7314** | **0.8475** | `[814, 168, 125, 399]` |
| **7. TinyBanglaBERT-6L (KD)** | Strided Pre-trained (6L) | KD ($T=4, \alpha=0.5$) | 9.00M | 0.7984 | 79.75% | 0.7824 | 0.7774 | 0.7904 | 0.7250 | 0.8397 | `[799, 183, 122, 402]` |
| **8. BanglaBERT-Small-12L (KD)** | Full Pre-trained (12L) | KD ($T=4, \alpha=0.5$) | 13.73M | 0.7976 | **81.14%** | **0.7926** | **0.7921** | 0.7931 | 0.7300 | **0.8551** | `[838, 144, 140, 384]` |

---

## 3. Efficiency, Latency & Edge Quantization Benchmarks

Single-headline forward pass latency measured over $N = 200$ iterations ($B=1$) on CPU and GPU (NVIDIA RTX 5050), paired with full test-set post-quantization accuracy verification ($N = 1,506$).

| Model Variant | Runtime / Format | Parameter Count | Disk Footprint | CPU Latency (Mean / P95) | CPU Throughput | GPU Latency | Storage Reduction | CPU Speedup | Test Accuracy / Macro-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (BanglaBERT)** | PyTorch FP32 | 110,618,882 | 422.76 MB | 123.36 ms / 152.85 ms | 8.1 fps | 12.60 ms | 1.00x *(Baseline)* | 1.00x *(Baseline)* | **83.73% / 0.8174** |
| **TinyBanglaBERT-4L** | PyTorch FP32 | **7,420,418** | 28.31 MB | 11.74 ms / 15.64 ms | 85.2 fps | **5.50 ms** | **14.93x smaller** | **10.51x faster** | **80.54% / 0.7895** |
| **TinyBanglaBERT-4L (ONNX)** | ONNX Runtime FP32 | **7,420,418** | 28.40 MB | 9.20 ms / 15.12 ms | 108.6 fps | N/A | **14.89x smaller** | **13.40x faster** 🚀 | **80.54% / 0.7895** |
| **TinyBanglaBERT-4L (INT8)** | ONNX Runtime INT8 | **7,420,418 *(8-bit)*** | **7.29 MB** | **4.00 ms / 5.14 ms** | **249.9 fps** | N/A | **57.99x smaller** 🔥 | **30.83x faster** 🚀 | **80.81% / 0.7917** ⭐ |

---

## 4. Statistical Significance Analysis (Holm-Corrected Exact McNemar Test)

| Pairwise Comparison (Model A vs. Model B) | Both Correct ($n_{00}$) | Model A Only ($b$) | Model B Only ($c$) | Both Wrong ($n_{11}$) | Discordant Pairs ($b + c$) | Raw McNemar $p$ | Holm-Adjusted $p$ | Significance ($\alpha=0.05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher vs. TinyBanglaBERT-4L** | 1140 | 121 | 73 | 172 | 194 | `p < 0.001` | **`p = 0.0035`** | **Statistically Significant** |
| **BanglaBERT-Small-12L + KD vs. TinyBanglaBERT-4L** | 1124 | 96 | 89 | 197 | 185 | `p = 0.6592` | **`p = 0.6592`** | **Not Significant ($12\text{L} \approx 4\text{L}$)** |
| **Teacher-PCA + KD vs. Teacher-PCA + Supervised** | 1054 | 127 | 88 | 237 | 215 | `p = 0.0094` | **`p = 0.0376`** | **Statistically Significant** |
| **TinyBanglaBERT-4L vs. Teacher-PCA + Supervised** | 1072 | 141 | 70 | 223 | 211 | `p < 0.001` | **`p < 0.001`** | **Statistically Significant** |
| **TinyBanglaBERT-4L vs. Random + Distilled** | 1048 | 165 | 76 | 217 | 241 | `p < 0.001` | **`p < 0.001`** | **Statistically Significant** |
| **TinyBanglaBERT-4L vs. Random + Supervised** | 1058 | 155 | 95 | 198 | 250 | `p < 0.001` | **`p = 0.0011`** | **Statistically Significant** |
| **TinyBanglaBERT-4L vs. Small-Pretrained 4L + CE** | 1169 | 44 | 32 | 261 | 76 | `p = 0.2067` | **`p = 0.4135`** | Not Significant |

---

## 5. Zero-Shot Cross-Domain Generalization Benchmark (`BaitBuster-Bangla`, $N = 10,000$)

| Evaluated Model | Evaluation Split | Accuracy | Macro F1 | Macro Precision | Macro Recall | Clickbait F1 | Non-CB F1 | Generalization Delta ($\Delta$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (110.6M)** | In-Domain (BanglaBait) | **0.8373** | **0.8174** | 0.8243 | 0.8121 | **0.7572** | **0.8777** | Baseline |
| **Teacher (110.6M)** | Cross-Domain (BaitBuster) | **0.7542** | **0.7514** | 0.7807 | 0.8119 | 0.7253 | **0.7776** | **`-6.60%` F1 / `-8.31%` Acc** |
| **TinyBanglaBERT-4L (7.42M)** | In-Domain (BanglaBait) | 0.8054 | 0.7895 | 0.7853 | 0.7952 | 0.7314 | 0.8475 | Baseline |
| **TinyBanglaBERT-4L (7.42M)** ⭐ | Cross-Domain (BaitBuster) | 0.7470 | 0.7459 | 0.7684 | 0.7836 | **0.7288** | 0.7629 | **`-4.36%` F1 / `-5.84%` Acc** |

---

## 6. Explainability & Cross-Method Concordance (SHAP vs. LIME)

- **Mean Top-3 Feature Jaccard Overlap:** **`34.67%`** (Median: `20.00%`, Clickbait Token Jaccard: **`32.22%`**) across $N = 15$ balanced TP/TN/FP/FN test headlines.
- **Mean Sign Polarity Agreement:** **`45.42%`** across all 4 quadrants (reaching **`83%–90%`** on confident True Positives and True Negatives, while highlighting conflicting cues on False Positive/Negative boundary headlines).

---

## 7. Selective Prediction (Risk–Coverage / AURC), Edge–Cloud Cascade & Label Noise Audit

### A. Selective Prediction & Two-Tier Edge–Cloud Deferral Cascade ($T_{\text{cal}} = 1.7375, \text{AURC} = 0.10268$)

| Edge Coverage (%) | Retained Samples | Confidence Threshold | Selective Edge Acc (%) | Selective Edge Macro-F1 | Cloud Deferral (%) | Cascade System Acc (%) | Cascade System Macro-F1 | Amortized Latency (Edge+GPU) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100.0%** | 1,506 | 0.5005 | 80.54% | 0.7895 | 0.0% | 80.54% | 0.7895 | **4.00 ms** |
| **90.0%** | 1,355 | 0.6526 | **83.69%** | **0.8199** | 10.0% | 82.27% | 0.8054 | **5.26 ms** |
| **80.0%** | 1,205 | 0.7517 | **85.98%** | **0.8441** | 20.0% | **83.20%** | **0.8145** | **6.52 ms** |
| **70.0%** | 1,054 | 0.8058 | **88.33%** | **0.8674** | 30.0% | 83.00% | 0.8115 | 7.78 ms |
| **60.0%** | 904 | 0.8376 | **90.38%** | **0.8905** | 40.0% | **83.60%** | **0.8177** | 9.04 ms |
| **50.0%** | 753 | 0.8597 | **90.84%** | **0.8966** | 50.0% | 83.20% | 0.8131 | 10.30 ms |
| **30.0%** | 452 | 0.8774 | **92.92%** | **0.9283** | — | — | — | — |

### B. Out-of-Fold Training Label Noise Audit ([`label_audit_report.md`](file:///c:/Users/Mushfiqur%20Rahman/Documents/Projects/tiny_bert/experiments/audit/label_audit_report.md))
- Auditing the 12,044 training headlines via 5-fold out-of-fold Teacher probabilities identified **`771` high-confidence label errors (`6.40%` of `train_df`, $\ge 88\%$ opposite-class confidence)**, explaining the empirical $\sim 84\%$ accuracy ceiling on raw BanglaBait annotations.
- Retraining TinyBanglaBERT-4L on the pruned clean training split (`splits_clean_train.json`, $N = 11,273$) achieved **`80.21%` Test Accuracy (`0.7835` Macro-F1)** with natural threshold calibration ($\tau^* = 0.50$).

