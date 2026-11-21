# Phase G — Mechanism & Mixed-Effects Analysis Report

**Total Grid Runs Analysed:** 80

## 1. Cell-Level Summary (Validation Split)

| init    | objective   |   frac |   n_seeds |   f1_mean |   f1_std |   ece_mean |   agree_mean |   ep_best_mean |
|:--------|:------------|-------:|----------:|----------:|---------:|-----------:|-------------:|---------------:|
| rand    | ce          |     10 |         5 |    0.9174 |   0.0072 |     0.0338 |       0.9262 |         9.0000 |
| rand    | ce          |     25 |         5 |    0.9642 |   0.0026 |     0.0262 |       0.9712 |         7.2000 |
| rand    | ce          |     50 |         5 |    0.9746 |   0.0007 |     0.0226 |       0.9806 |         7.6000 |
| rand    | ce          |    100 |         5 |    0.9805 |   0.0009 |     0.0174 |       0.9872 |         5.8000 |
| rand    | kd          |     10 |         5 |    0.9114 |   0.0101 |     0.0222 |       0.9196 |         9.2000 |
| rand    | kd          |     25 |         5 |    0.9639 |   0.0037 |     0.0125 |       0.9706 |         9.0000 |
| rand    | kd          |     50 |         5 |    0.9725 |   0.0031 |     0.0070 |       0.9794 |         6.0000 |
| rand    | kd          |    100 |         5 |    0.9811 |   0.0029 |     0.0079 |       0.9878 |         7.0000 |
| strided | ce          |     10 |         5 |    0.9610 |   0.0018 |     0.0739 |       0.9676 |         7.6000 |
| strided | ce          |     25 |         5 |    0.9745 |   0.0016 |     0.0142 |       0.9810 |         7.2000 |
| strided | ce          |     50 |         5 |    0.9796 |   0.0012 |     0.0168 |       0.9860 |         6.2000 |
| strided | ce          |    100 |         5 |    0.9851 |   0.0014 |     0.0122 |       0.9910 |         6.6000 |
| strided | kd          |     10 |         5 |    0.9615 |   0.0025 |     0.0678 |       0.9682 |         8.6000 |
| strided | kd          |     25 |         5 |    0.9741 |   0.0016 |     0.0072 |       0.9806 |         5.6000 |
| strided | kd          |     50 |         5 |    0.9809 |   0.0015 |     0.0049 |       0.9876 |         6.8000 |
| strided | kd          |    100 |         5 |    0.9866 |   0.0017 |     0.0080 |       0.9928 |         6.8000 |

---

## 2. Linear Mixed-Effects Model (Centered log_frac, Full Interactions)
Formula: `f1_macro ~ C(init) * C(objective) * log_frac_c + (1 | seed)`

Direct H1 Wald Contrast (beta_strided - beta_kd at mean log_frac): +1.78 pp (SE=0.24 pp, 95% CI [+1.30, +2.26] pp, chi2=53.56, p=2.5061e-13)

```
                           Mixed Linear Model Regression Results
============================================================================================
Model:                          MixedLM             Dependent Variable:             f1_macro
No. Observations:               80                  Method:                         ML      
No. Groups:                     5                   Scale:                          0.0001  
Min. group size:                16                  Log-Likelihood:                 275.8087
Max. group size:                16                  Converged:                      Yes     
Mean group size:                16.0                                                        
--------------------------------------------------------------------------------------------
                                                 Coef.  Std.Err.    z    P>|z| [0.025 0.975]
--------------------------------------------------------------------------------------------
Intercept                                         0.959    0.002 557.061 0.000  0.956  0.963
C(init)[T.strided]                                0.016    0.002   6.536 0.000  0.011  0.021
C(objective)[T.kd]                               -0.002    0.002  -0.782 0.434 -0.007  0.003
C(init)[T.strided]:C(objective)[T.kd]             0.003    0.003   0.758 0.448 -0.004  0.009
log_frac_c                                        0.027    0.002  13.383 0.000  0.023  0.031
C(init)[T.strided]:log_frac_c                    -0.017    0.003  -5.840 0.000 -0.022 -0.011
C(objective)[T.kd]:log_frac_c                     0.002    0.003   0.851 0.395 -0.003  0.008
C(init)[T.strided]:C(objective)[T.kd]:log_frac_c -0.002    0.004  -0.477 0.633 -0.010  0.006
Group Var                                         0.000                                     
============================================================================================

```
