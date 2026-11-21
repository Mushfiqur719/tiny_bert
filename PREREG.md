# TinyBanglaBERT v3 — Study Pre-Registration (PREREG)

**Date of Registration:** 2026-11-20  
**Repository:** `tiny_bert`  
**Git Tag:** `v3-prereg`  
**Lead Investigator:** Mushfiqur Rahman  

---

## 1. Study Title & Core Scientific Question
**Pre-trained Compact Initialisation vs. Distillation Objectives in Low-Resource Fine-Tuning**

*Research Question:* In resource-constrained fine-tuning of compact language models, does warm-start compact initialisation account for more performance variance than the choice of distillation objective, and how does this advantage interact with dataset availability and model depth?

---

## 2. Hypotheses

### Hypothesis 1 (Initialisation Dominance, H1)
On a strictly parameter-matched compact architecture (4 transformer layers, hidden dimension 256, factorized embedding dimension 128, 7.42M parameters), **initialising student weights from a pre-trained compact backbone accounts for a significantly larger proportion of variance in test Macro-F1 than the choice of objective** (Cross-Entropy vs. Logit Distillation vs. Logit + Intermediate Hidden Alignment).

### Hypothesis 2 (Objective-Data Interaction, H2)
The marginal benefit of knowledge distillation over standard cross-entropy:
1. **Shrinks as initialisation quality improves** (i.e., the gain from distillation is largest when initializing randomly and smallest when initializing from pre-trained strided layers).
2. **Grows as the volume of labelled fine-tuning data decreases** (i.e., relative distillation gain at 10% data fraction is substantially higher than at 100% data fraction).

---

## 3. Pre-Registered Metrics & Acceptance Criteria

### Primary Metric
- **Macro-Averaged F1 Score ($\text{Macro-F1}$)** on the held-out test split, evaluated over 5 distinct random seeds (`[13, 42, 87, 1234, 2026]`).
- Reported as five-seed mean with standard deviation and 95% non-parametric bootstrap confidence interval (10,000 resamples).

### Non-Inferiority Margin
- **$\Delta = 2.0$ percentage points (0.020 Macro-F1):** A compact student is deemed non-inferior to a reference teacher if the lower bound of the 95% confidence interval of $(\text{Macro-F1}_{\text{student}} - \text{Macro-F1}_{\text{teacher}}) \ge -2.0\%$.

### Secondary Metrics
- Binary Clickbait Class F1 and Non-Clickbait Class F1.
- Test Accuracy.
- Expected Calibration Error (ECE, 10 equal-mass bins).
- Single-thread and 4-thread CPU inference latency (p50 and p95 in milliseconds).
- Selective prediction Area Under Risk-Coverage curve (AURC) and accuracy at coverage thresholds $\in \{100\%, 90\%, 75\%, 50\%\}$.

---

## 4. Planned Comparisons & Statistical Corrections

### Controlled Factorial Matrix (Phase F)
All student variants share the identical 4-layer, 7.42M parameter architecture:
1. **Initialisation Factors (4 arms):**
   - `rand`: Random Gaussian initialisation ($\mu=0, \sigma=0.02$).
   - `strided`: Layers $[2, 5, 8, 11]$ extracted from `csebuetnlp/banglabert_small`.
   - `first4`: Layers $[0, 1, 2, 3]$ from `csebuetnlp/banglabert_small`.
   - `last4`: Layers $[8, 9, 10, 11]$ from `csebuetnlp/banglabert_small`.
2. **Objective Factors (3 arms):**
   - `ce`: Supervised Cross-Entropy with balanced class weighting.
   - `kd`: Logit distillation with temperature $T=4.0$ and hard-loss weighting $\alpha=0.5$.
   - `kd_hid`: Logit distillation ($T=4.0, \alpha=0.5$) combined with intermediate hidden representation cosine distance alignment.
3. **Data Fraction Factors (4 arms):**
   - 10%, 25%, 50%, 100% of stratified training data with strictly nested sample indices.
4. **Random Seeds:**
   - 5 seeds: $13, 42, 87, 1234, 2026$.
   - Total Core Runs: $4 \times 3 \times 4 \times 5 = 240$ runs.

### Multiple Comparison Correction
- All planned pairwise comparisons between initialisation strategies and training objectives will be adjusted for family-wise error rate using the **Holm step-down procedure** (Holm-Bonferroni correction) at $\alpha_{\text{family}} = 0.05$.
- Paired bootstrap hypothesis tests will be conducted over paired item predictions across the 5 seeds (10,000 resamples).

### Mixed-Effects Mechanism Model (Phase G)
To quantify effect sizes and test H1 and H2 simultaneously, we fit:
$$\text{Macro-F1} = \beta_0 + \boldsymbol{\beta}_1^\top \mathbf{Init} + \boldsymbol{\beta}_2^\top \mathbf{Obj} + \beta_3 \log(\text{Frac}) + \boldsymbol{\beta}_4^\top (\mathbf{Init} \times \mathbf{Obj}) + \boldsymbol{\beta}_5^\top (\mathbf{Obj} \times \log(\text{Frac})) + u_{\text{seed}} + \varepsilon$$
where $u_{\text{seed}} \sim \mathcal{N}(0, \sigma^2_s)$ is a random intercept for random seed.

---

## 5. Experimental Integrity Rules
1. **Model Selection:** Model selection and hyperparameter choices are made exclusively on the validation split.
2. **Zero Outcome Bias:** Acceptance criteria test pipeline execution, data hygiene, and audit compliance—never numerical result thresholds.
3. **Test-Set Lock:** No test labels are loaded or accessed until Phase J (`TBB_ALLOW_TEST=1`).
4. **Pre-Registration Immutability:** Any post-registration deviations, amendments, or exploratory analyses must be explicitly noted in the final manuscript.
