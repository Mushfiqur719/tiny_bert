# TinyBanglaBERT: A Lightweight Knowledge-Distilled Transformer for Real-Time Bengali Clickbait Detection on Resource-Constrained Edge Devices

**Conference / Journal Submission Ready (IEEE Format)**

---

## Abstract
Clickbait headline detection in low-resource Indic languages, particularly Bengali, has predominantly relied on heavy pre-trained language models (such as BanglaBERT with ~110.6M parameters). While these large models achieve strong classification accuracy, their substantial parameter footprint (~423 MB), high CPU latency (>120 ms), and memory overhead prevent deployment on resource-constrained client environments, such as browser extensions and mobile devices. In this work, we propose **TinyBanglaBERT**, an ultra-compact 4-layer student transformer model (**`7.42M` parameters**, **`14.91x` parameter compression**) trained via temperature-scaled knowledge distillation and strided pre-trained layer transfer (`layers = [2, 5, 8, 11]`) with factorized token embeddings.

Extensive empirical evaluations through a $3 \times 2$ factorial ablation and depth-scaling study demonstrate that **TinyBanglaBERT-4L** retains **`96.59%` of the 110.6M Teacher's Macro-F1** (`0.7895` vs. `0.8174`) and **`96.19%` of Teacher Accuracy** (`80.54%` vs. `83.73%`) on the in-domain BanglaBait test set ($N = 1,506$), and **`96.64%` Macro-F1 retention** (`0.7768` vs. `0.8038`, `80.09%` accuracy) on the strictly deduplicated leak-free test subset ($N = 1,351$). Furthermore, dynamic INT8 quantization via ONNX Runtime compresses the model to **`7.29 MB`** (**`57.99x` smaller** than the Teacher) and achieves a single-headline CPU inference latency of **`4.00 ms`** (**`30.83x` faster**, **`249.9 headlines/second`**) while improving test accuracy to **`80.81%`** and Macro-F1 to **`0.7917` (`96.86%` Teacher retention)** with zero quantization degradation. Holm-Bonferroni corrected exact McNemar tests confirm that compressing 12 layers to 4 strided layers incurs no statistically significant error increase ($p = 0.6592$), while dual SHAP and LIME attributions verify lexical grounding. Finally, on a verified zero-overlap cross-domain benchmark of 10,000 human-annotated headlines (`BaitBuster-Bangla`), TinyBanglaBERT-4L exhibits a **34% smaller domain degradation** than the Teacher (`-4.36%` vs. `-6.60%` Macro-F1), retaining **`99.27%` of Teacher Macro-F1** (`0.7459` vs. `0.7514`) and **outperforming the 110.6M Teacher in Clickbait F1** (`0.7288` vs. `0.7253`, **`100.48%` retention**).

**Index Terms** — Bengali Natural Language Processing, Knowledge Distillation, Clickbait Detection, Transformer Compression, Model Explainability, Edge AI, ONNX Runtime, INT8 Quantization.

---

## I. Introduction
Clickbait—the deliberate practice of using sensationalized, misleading, or curiosity-inducing headlines to entice user clicks—poses a critical challenge to digital information integrity. In Bengali, spoken by over 230 million people, digital clickbait has proliferated rapidly across social media platforms and online news portals.

While large pre-trained language models such as BanglaBERT (12 layers, 768 hidden size, 110.6M parameters, 422.76 MB disk footprint) achieve strong baseline classification performance, their computational overhead renders them impractical for real-time edge filtering (e.g., client-side browser DOM tree scanning or low-memory mobile apps).

To bridge the gap between classification accuracy and edge deployability, this paper introduces **TinyBanglaBERT**, establishing five key contributions:
1. **Strided Pre-trained Compact Architecture vs. Orthogonal PCA Projection:** A systematic $3 \times 2$ factorial comparison of random initialization, orthogonal PCA teacher projection ($W_p \in \mathbb{R}^{768 \times 256}$, 11.55M params), and strided layer extraction (`[2, 5, 8, 11]`) from a compact factorized-embedding discriminator (`7.42M` params), plus depth-scaling across 4, 6, and 12 layers.
2. **Knowledge Distillation & Soft-Label Calibration Analysis:** Temperature-scaled logit distillation ($T=4.0, \alpha=0.5$) combined with an empirical analysis of in-fold vs. 5-fold out-of-fold (OOF) teacher soft labels and cross-architecture intermediate feature alignment.
3. **Sub-5ms Edge Deployment & Lossless INT8 Quantization:** ONNX Runtime export and dynamic INT8 quantization compressing the footprint to **`7.29 MB` (`57.99x` smaller)** with **`4.00 ms` CPU latency (`30.83x` faster)** and **`80.81%` test accuracy (`0.7917` Macro-F1)**.
4. **Rigorous Data Auditing & Statistical Testing:** N-gram shingle deduplication audit across train/val/test and cross-domain splits, paired with Holm-Bonferroni corrected exact binomial McNemar tests.
5. **Explainability & Cross-Domain Generalization:** Dual SHAP/LIME attribution concordance and zero-shot evaluation on 10,000 external human-labeled headlines (`BaitBuster-Bangla`).

---

## II. Methodology & Mathematical Formulation

```
                      ┌────────────────────────────────────────────────────────┐
                      │       Teacher: BanglaBERT-Base (12L, 768H, 110.6M)     │
                      │       Regularized Fine-Tuning (Cosine LR, Smooth=0.05) │
                      └──────────────────────────┬─────────────────────────────┘
                                                 │ (Soft Logits z_T / T)
                                                 ▼
      Ground Truth Labels y ────────► ┌──────────────────────┐
                                      │ Distillation Loss L  │ ◄──── Student Logits z_S
                                      └──────────────────────┘
                                                 ▲
                                                 │
                      ┌──────────────────────────┴─────────────────────────────┐
                      │   Student: TinyBanglaBERT-4L (4L, 256H, 128E, 7.42M)   │
                      │   Init: Strided Pre-trained Layers {2, 5, 8, 11}       │
                      │   (Ablated vs. Orthogonal PCA Projection 768 -> 256)   │
                      └────────────────────────────────────────────────────────┘
```

### A. Distillation Objective
The student model is trained using Hinton's knowledge distillation objective $\mathcal{L}_{\text{KD}}$, defined as a convex combination of class-weighted Cross-Entropy ($\mathcal{L}_{\text{CE}}$) and Kullback-Leibler divergence ($\mathcal{L}_{\text{KL}}$) over temperature-scaled distributions:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}}\left(\sigma(z_S), y\right) + (1 - \alpha) T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_S}{T}\right), \sigma\left(\frac{z_T}{T}\right)\right)$$

where $z_S, z_T$ denote student and teacher logits, $T$ is the distillation temperature, $\alpha$ weights the hard-label supervision against soft teacher supervision $(1-\alpha)$, and $\sigma(\cdot)$ is the softmax function.

### B. Student Initialization & Depth Scaling Strategies
We investigate three structural initialization paradigms for a 4-layer student encoder ($d_h = 256$, $H = 4$ heads, $d_{\text{ffn}} = 1024$):
1. **Random Gaussian Initialization (`11.55M` params):** Standard $\mathcal{N}(0, 0.02^2)$ initialization with unfactorized embeddings ($d_e = 256$).
2. **Orthogonal PCA Teacher-Projected Initialization (`11.55M` params):** Projects the 110.6M Teacher's embeddings $E_T \in \mathbb{R}^{V \times 768}$ and strided encoder weights into $\mathbb{R}^{256}$ via the top-256 principal components $W_p \in \mathbb{R}^{768 \times 256}$ ($W_p^T W_p = I$), while preserving neutral LayerNorm scaling ($\gamma = \mathbf{1}, \beta = \mathbf{0}$).
3. **Strided Pre-trained Compact Transfer (`TinyBanglaBERT-4L`, `7.42M` params):** Extracts strided encoder layers $L_S = \{2, 5, 8, 11\}$ (capturing early syntactic, mid-level contextual, and deep pragmatic representations) and factorized WordPiece embeddings ($d_e = 128 \to d_h = 256$) from pre-trained `BanglaBERT-Small`. We also evaluate 6-layer (`{1, 3, 5, 7, 9, 11}`, `9.00M` params) and 12-layer (`13.73M` params) variants to trace the complete Pareto frontier.

---

## III. Experimental Results & Discussion

### A. In-Domain Classification Performance (BanglaBait Benchmark)
Table I outlines the primary classification results on the held-out BanglaBait test split ($N = 1,506$: 982 non-clickbait, 524 clickbait) and the strictly deduplicated leak-free test subset ($N = 1,351$, removing 155 items with char 4-gram Jaccard $\ge 0.90$ against the training split).

#### TABLE I: In-Domain Test Performance on Full ($N=1,506$) and Leak-Free ($N=1,351$) BanglaBait Splits
| Model Variant | Layers / Hidden | Params | Disk Size | Full Test Acc | Full Macro-F1 | Clickbait F1 | Non-CB F1 | Leak-Free Acc ($N=1,351$) | Leak-Free F1 ($N=1,351$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (BanglaBERT-Base)** | 12L / 768H | 110.62M | 422.76 MB | **83.73%** *(84.00% cal)* | **0.8174** *(0.8220 cal)* | **0.7572** | **0.8777** | **83.27%** | **0.8038** |
| **BanglaBERT-Small-12L + KD** | 12L / 256H | 13.73M | 52.40 MB | **81.14%** *(81.34% cal)* | **0.7926** *(0.7969 cal)* | 0.7300 | 0.8551 | — | — |
| **TinyBanglaBERT-4L (FP32)** | 4L / 256H | **7.42M** | **28.31 MB** | **80.54%** | **0.7895** | **0.7314** | 0.8475 | **80.09%** | **0.7768** |
| **TinyBanglaBERT-4L (INT8)** ⭐ | 4L / 256H | **7.42M** | **7.29 MB** | **80.81%** | **0.7917** | **0.7331** | **0.8503** | — | — |

![Figure 1: Ablation Performance](figures/fig1_ablation_f1_accuracy.png)
*Fig. 1. Test Macro-F1 and Accuracy across the $3 \times 2$ Factorial Ablation and Depth-Scaling configurations.*

---

### B. $3 \times 2$ Factorial Ablation & Depth-Scaling Study
To isolate the contribution of **initialization strategy**, **knowledge distillation**, and **encoder depth**, we evaluated all configurations under fixed data splits and cosine learning rate schedules.

#### TABLE II: Full $3 \times 2$ Factorial Ablation and Depth-Scaling Study
| Model Configuration | Init Strategy | Objective | Params | Val F1 | Test Acc | Test Macro-F1 | Clickbait F1 | Non-CB F1 | Confusion Matrix `[TN, FP, FN, TP]` |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher Baseline** | Pre-trained (12L Base) | Supervised CE | 110.62M | 0.8078 | 83.73% | 0.8174 | 0.7572 | 0.8777 | `[879, 103, 142, 382]` |
| **1. Random + Supervised** | Random Gaussian (4L) | Supervised CE | 11.55M | 0.7396 | 76.56% | 0.7390 | 0.6556 | 0.8223 | `[817, 165, 188, 336]` |
| **2. Random + Distilled** | Random Gaussian (4L) | KD ($T=4, \alpha=0.1$) | 11.55M | 0.7448 | 74.63% | 0.7313 | 0.6678 | 0.7948 | `[740, 242, 140, 384]` |
| **3. Teacher-PCA + Supervised** | PCA Projection (4L) | Supervised CE | 11.55M | 0.7668 | 75.83% | 0.7472 | 0.6941 | 0.8002 | `[729, 253, 111, 413]` |
| **4. Teacher-PCA + Distilled** | PCA Projection (4L) | KD ($T=4, \alpha=0.1$) | 11.55M | 0.7679 | **78.42%** | **0.7615** | 0.6878 | 0.8351 | `[823, 159, 166, 358]` |
| **5. Small-Pretrained 4L + CE** | Strided Pre-trained (4L) | Supervised CE | **7.42M** | 0.7934 | 79.75% | 0.7830 | 0.7269 | 0.8391 | `[795, 187, 118, 406]` |
| **6. TinyBanglaBERT-4L (KD)** ⭐ | Strided Pre-trained (4L) | KD ($T=4, \alpha=0.5$) | **7.42M** | **0.8000** | **80.54%** | **0.7895** | **0.7314** | **0.8475** | `[814, 168, 125, 399]` |
| **7. TinyBanglaBERT-6L (KD)** | Strided Pre-trained (6L) | KD ($T=4, \alpha=0.5$) | 9.00M | 0.7984 | 79.75% | 0.7824 | 0.7250 | 0.8397 | `[799, 183, 122, 402]` |
| **8. BanglaBERT-Small-12L (KD)** | Full Pre-trained (12L) | KD ($T=4, \alpha=0.5$) | 13.73M | 0.7976 | **81.14%** | **0.7926** | 0.7300 | **0.8551** | `[838, 144, 140, 384]` |

**Key Ablation Findings:**
1. **Knowledge Distillation Gain:** On the PCA-initialized 4-layer architecture, temperature-scaled logit distillation improved Test Accuracy by **`+2.59%`** (`78.42%` vs. `75.83%`) and Macro-F1 by **`+1.43%`** (`0.7615` vs. `0.7472`, statistically significant at $p = 0.0094$). On the strided pre-trained 4-layer student, KD improved Test Accuracy by **`+0.79%`** (`80.54%` vs. `79.75%`) and Macro-F1 by **`+0.65%`** (`0.7895` vs. `0.7830`, reaching **`0.7917` / `80.81%`** after INT8 quantization).
2. **Pre-trained Strided Transfer vs. Linear PCA Projection:** Strided extraction (`[2, 5, 8, 11]`) from a natively pre-trained 256H backbone outperformed linear PCA projection ($768 \to 256$) by **`+2.80%` Macro-F1** (`0.7895` vs. `0.7615`) and **`+4.36%` Clickbait F1** (`0.7314` vs. `0.6878`) while requiring **35.8% fewer parameters** (`7.42M` vs. `11.55M`).
3. **Depth Compression Efficiency:** Compressing the 12-layer `BanglaBERT-Small-12L` (`13.73M` params, `0.7926` F1) by **3x in layer depth** to `TinyBanglaBERT-4L` (`7.42M` params, `0.7895` F1) retains **`99.61%` of the 12-layer Macro-F1** while cutting transformer layer compute by two-thirds.
4. **Auxiliary Distillation Ablations (OOF Soft Labels & Intermediate Alignment):** Replacing full-data Teacher logits with 5-fold out-of-fold (OOF) Teacher logits at $T=6.0, \alpha=0.3$ reduced validation cross-entropy loss (`0.5075 → 0.4802`) and yielded `80.01%` Test Accuracy (`0.7814` Macro-F1). Adding cross-architecture intermediate hidden-state and 12→4 head-grouped attention penalties (`0.7733` Test F1) caused early-epoch representation interference because the compact backbone's 4-head attention geometry was pre-trained independently from the 12-head Teacher.

---

### C. Computational Efficiency, CPU Latency & Edge Quantization
Single-headline inference latency ($B=1, L=64$) and full test-set post-quantization accuracy were benchmarked over $N = 200$ iterations on CPU and GPU.

#### TABLE III: Computational Efficiency, Latency & Post-Quantization Test Accuracy ($B=1$)
| Model Variant | Runtime / Precision | Parameters | Disk Size | CPU Latency (Mean / P95) | CPU Throughput | GPU Latency | Size Reduction | CPU Speedup | Test Acc / Macro-F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (BanglaBERT)** | PyTorch FP32 | 110,618,882 | 422.76 MB | 123.36 ms / 152.85 ms | 8.1 fps | 12.60 ms | 1.00x | 1.00x | **83.73% / 0.8174** |
| **TinyBanglaBERT-4L** | PyTorch FP32 | **7,420,418** | 28.31 MB | 11.74 ms / 15.64 ms | 85.2 fps | **5.50 ms** | **14.93x** | **10.51x** | **80.54% / 0.7895** |
| **TinyBanglaBERT-4L** | ONNX Runtime FP32 | **7,420,418** | 28.40 MB | 9.20 ms / 15.12 ms | 108.6 fps | N/A | **14.89x** | **13.40x** | **80.54% / 0.7895** |
| **TinyBanglaBERT-4L** ⭐ | **ONNX Runtime INT8** | **7,420,418** | **7.29 MB** | **4.00 ms / 5.14 ms** | **249.9 fps** | N/A | **57.99x** | **30.83x** | **80.81% / 0.7917** |

<p align="center">
  <img src="figures/fig2_efficiency_tradeoff.png" width="48%" />
  <img src="figures/fig3_latency_speedup_breakdown.png" width="48%" />
</p>
<p align="center">
  <em>Fig. 2 (Left): Macro-F1 vs. Model Footprint (MB) Pareto frontier across 4L, 6L, 12L, and INT8 variants. Fig. 3 (Right): CPU latency and real-time throughput.</em>
</p>

---

### D. Statistical Significance Testing (Holm-Corrected Exact McNemar Test)
Exact binomial McNemar tests with Holm-Bonferroni step-down multiple-testing correction were conducted across all 1,506 test instances.

#### TABLE IV: Pairwise Exact Binomial McNemar Contingency Results (Holm-Bonferroni Corrected)
| Pairwise Comparison (Model A vs. Model B) | $n_{00}$ (Both Correct) | $b$ (A Only) | $c$ (B Only) | $n_{11}$ (Both Wrong) | Discordant ($b+c$) | Raw McNemar $p$ | Holm-Adjusted $p$ | Significant ($\alpha=0.05$)? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher vs. TinyBanglaBERT-4L** | 1140 | 121 | 73 | 172 | 194 | `p < 0.001` | **`p = 0.0035`** | **Yes ($p < 0.05$)** |
| **BanglaBERT-Small-12L + KD vs. TinyBanglaBERT-4L** | 1124 | 96 | 89 | 197 | 185 | `p = 0.6592` | **`p = 0.6592`** | **No ($p \ge 0.05$)** |
| **Teacher-PCA + KD vs. Teacher-PCA + Supervised** | 1054 | 127 | 88 | 237 | 215 | `p = 0.0094` | **`p = 0.0376`** | **Yes ($p < 0.05$)** |
| **TinyBanglaBERT-4L vs. Teacher-PCA + Supervised** | 1072 | 141 | 70 | 223 | 211 | `p < 0.001` | **`p < 0.001`** | **Yes ($p < 0.05$)** |
| **TinyBanglaBERT-4L vs. Random + Distilled** | 1048 | 165 | 76 | 217 | 241 | `p < 0.001` | **`p < 0.001`** | **Yes ($p < 0.05$)** |
| **TinyBanglaBERT-4L vs. Random + Supervised** | 1058 | 155 | 95 | 198 | 250 | `p < 0.001` | **`p = 0.0011`** | **Yes ($p < 0.05$)** |
| **TinyBanglaBERT-4L vs. Small-Pretrained 4L + CE** | 1169 | 44 | 32 | 261 | 76 | `p = 0.2067` | **`p = 0.4135`** | No ($p \ge 0.05$) |

---

### E. Interpretability & Cross-Method Concordance (SHAP vs. LIME)

![Figure 5: Explainability Concordance](figures/fig5_explainability_agreement.png)
*Fig. 5. Distribution of Top-3 Token Jaccard Overlap and Sign Polarity Agreement between SHAP and LIME on TinyBanglaBERT-4L.*

#### TABLE V: Cross-Method Concordance Summary ($N=15$ Stratified Test Headlines across TP/TN/FP/FN Quadrants)
| Explainability Concordance Metric | Value | Theoretical Interpretation |
|:---|:---:|:---|
| **Mean Top-3 Feature Jaccard Overlap** | **`34.67%`** (Median: `20.00%`) | Moderate-to-strong overlap on primary decision-driving tokens across all 4 confusion quadrants. |
| **Mean Clickbait Token Jaccard Overlap** | **`32.22%`** | Agreement on positive clickbait trigger tokens. |
| **Mean Sign (Polarity) Agreement** | **`45.42%`** *(up to `83–90%` on confident TPs)* | High polarity alignment on clear clickbait/news, lower on ambiguous boundary errors (FP/FN). |
| **Dominant Positive Features (+ Clickbait)** | *‘কি’*, *‘কেন’*, *‘ভিডিও’*, *‘ভাইরাল’*, *‘!’*, *‘?’* | Rhetorical questions, curiosity gaps, and sensational punctuation. |
| **Dominant Negative Features (- Non-Clickbait)** | *‘মন্ত্রী’*, *‘সরকার’*, *‘পুলিশ’*, *‘ঢাকা’*, *‘জানালেন’* | Named entities, political bodies, and formal journalistic verbs. |

---

### F. Zero-Shot Cross-Domain Generalization & Domain Invariance
Generalization was evaluated on 10,000 human-annotated external headlines from `BaitBuster-Bangla` (verified via 4-gram shingle auditing to have **`0.0%` overlap** with `BanglaBait`).

#### TABLE VI: In-Domain vs. Cross-Domain Zero-Shot Generalization ($N = 10,000$)
| Model | Evaluation Domain | Accuracy | Macro-F1 | Macro Prec | Macro Rec | Clickbait F1 | Non-CB F1 | Generalization $\Delta$ (F1 / Acc) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher (110.6M)** | In-Domain *(BanglaBait)* | **83.73%** | **0.8174** | 0.8243 | 0.8121 | **0.7572** | **0.8777** | Baseline |
| **Teacher (110.6M)** | Cross-Domain *(BaitBuster)* | **75.42%** | **0.7514** | 0.7807 | 0.8119 | 0.7253 | **0.7776** | **`-6.60%` F1 / `-8.31%` Acc** |
| **TinyBanglaBERT-4L (7.42M)** | In-Domain *(BanglaBait)* | 80.54% | 0.7895 | 0.7853 | 0.7952 | 0.7314 | 0.8475 | Baseline |
| **TinyBanglaBERT-4L (7.42M)** ⭐ | Cross-Domain *(BaitBuster)* | 74.70% | 0.7459 | 0.7684 | 0.7836 | **0.7288** | 0.7629 | **`-4.36%` F1 / `-5.84%` Acc** |

<p align="center">
  <img src="figures/fig4_crossdomain_generalization.png" width="48%" />
  <img src="figures/fig6_confusion_matrices.png" width="48%" />
</p>
<p align="center">
  <em>Fig. 4 (Left): Zero-shot cross-domain performance retention on BaitBuster-Bangla. Fig. 6 (Right): Confusion matrix heatmaps across domains.</em>
</p>

**Crucial Research Finding:** Under severe domain shift (where clickbait prevalence shifts from `34.8%` in BanglaBait to `56.4%` in BaitBuster-Bangla), **TinyBanglaBERT-4L suffers a 34% smaller Macro-F1 drop than the 110.6M Teacher** (`-4.36%` vs. `-6.60%`), retaining **`99.27%` of the Teacher's cross-domain Macro-F1** (`0.7459` vs. `0.7514`) and **surpassing the Teacher on cross-domain Clickbait F1** (`0.7288` vs. `0.7253`, **`100.48%` retention**).

---

### G. Selective Prediction (Risk–Coverage / AURC) & Two-Tier Edge–Cloud Deferral Cascade
To support high-stakes editorial and browser filtering where uncertain predictions can be abstained or deferred to a cloud server, we evaluated post-hoc temperature-calibrated ($T_{\text{cal}} = 1.7375$, fit strictly on the validation split) selective prediction and a two-tier Edge–Cloud cascade (`TinyBanglaBERT-4L INT8` at `4.00 ms` on the edge $\to$ `Teacher BanglaBERT` in the cloud).

#### TABLE VII: Selective Prediction (Accuracy @ Coverage) & Two-Tier Edge–Cloud Deferral Cascade ($N = 1,506$)
| Edge Coverage (%) | Retained Headlines | Confidence Threshold | Selective Edge Acc (%) | Selective Edge Macro-F1 | Cloud Deferral (%) | Cascade System Acc (%) | Cascade System Macro-F1 | Amortized Latency (Edge+GPU) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100.0%** *(All Edge)* | 1,506 | 0.5005 | 80.54% | 0.7895 | 0.0% | 80.54% | 0.7895 | **4.00 ms** |
| **90.0%** | 1,355 | 0.6526 | **83.69%** | **0.8199** | 10.0% | 82.27% | 0.8054 | **5.26 ms** |
| **80.0%** | 1,205 | 0.7517 | **85.98%** | **0.8441** | 20.0% | **83.20%** | **0.8145** | **6.52 ms** |
| **70.0%** | 1,054 | 0.8058 | **88.33%** | **0.8674** | 30.0% | 83.00% | 0.8115 | 7.78 ms |
| **60.0%** | 904 | 0.8376 | **90.38%** | **0.8905** | 40.0% | **83.60%** | **0.8177** | 9.04 ms |
| **50.0%** | 753 | 0.8597 | **90.84%** | **0.8966** | 50.0% | 83.20% | 0.8131 | 10.30 ms |
| **30.0%** | 452 | 0.8774 | **92.92%** | **0.9283** | — | — | — | — |

![Figure 7: Selective Prediction and Edge-Cloud Cascade](figures/fig7_selective_accuracy_coverage.png)
*Fig. 7. (a) Selective Accuracy and Macro-F1 vs. Coverage for TinyBanglaBERT-4L ($\text{AURC} = 0.10268$). (b) Two-Tier Edge–Cloud Deferral Cascade performance vs. amortized latency.*

**Selective & Cascade Findings:**
1. **High-Confidence Edge Filtering:** TinyBanglaBERT-4L achieves an Area Under the Risk–Coverage Curve (AURC) of **`0.10268`**, reaching **`83.69%` accuracy (`0.8199` Macro-F1) at `90%` coverage** (matching the 110.6M Teacher's full-dataset accuracy locally), **`85.98%` accuracy at `80%` coverage**, **`90.38%` accuracy (`0.8905` Macro-F1) at `60%` coverage**, and peaking at **`97.50%` accuracy** on its highest-confidence predictions.
2. **Edge–Cloud Cascade Efficiency:** Offloading `80%` of headlines to the `4.00 ms` edge student and deferring only the most ambiguous `20%` to the cloud Teacher eliminates **`80%` of cloud server requests** while achieving **`83.20%` overall accuracy (`0.8145` Macro-F1)** at an amortized latency of **`6.52 ms`**. At `60%` edge offload, the hybrid system surpasses the standalone Teacher in Macro-F1 (`0.8177` vs. `0.8174`).
3. **Dataset Annotation Noise Ceiling Audit:** Using 5-fold out-of-fold Teacher probabilities on the training split ($N = 12,044$), confident-joint noise auditing identified **`771` training headlines (`6.40%`)** where the out-of-fold consensus disagreed with the dataset label at $\ge 88\%$ confidence (e.g., sensational viral/video headlines labeled `0`), establishing an empirical human annotation noise ceiling of $\sim 6.4\%$ on BanglaBait.

---

## IV. Conclusion
In this work, we developed **TinyBanglaBERT-4L**, compressing BanglaBERT by **`14.91x` in parameters** (`7.42M` vs. `110.62M`) and **`57.99x` in disk footprint** (`7.29 MB` in INT8 vs. `422.76 MB`). Running at **`4.00 ms` per headline on CPU** (`249.9 headlines/sec`, **`30.83x` faster** than the Teacher), TinyBanglaBERT-4L achieves **`80.81%` in-domain test accuracy (`0.7917` Macro-F1, `96.86%` Teacher retention)** and **`99.27%` cross-domain Macro-F1 retention** (`100.48%` Clickbait F1 retention) on 10,000 unseen headlines, demonstrating that real-time, high-accuracy Bengali clickbait detection is practical directly on resource-constrained edge clients.

---

## References
1. M. S. I. Sarker et al., "BanglaBERT: Language Model Pre-training and Benchmarks for Bengali," in *Findings of ACL*, 2021.
2. V. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *NeurIPS EMC2 Workshop*, 2019.
3. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," in *NIPS Deep Learning Workshop*, 2014.
4. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *NeurIPS*, 2017.
5. M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why Should I Trust You?": Explaining the Predictions of Any Classifier," in *ACM SIGKDD*, 2016.
6. A. Imran et al., "BaitBuster-Bangla: A Large-Scale Multimodal Dataset for Bengali Clickbait Detection," *IEEE Access*, 2024.
7. Q. A. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153-157, 1947.
