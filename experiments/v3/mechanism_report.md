# Phase G — Mechanism & Mixed-Effects Analysis Report

**Total Grid Runs Analysed:** 80

## 1. Cell-Level Summary (Validation Split)

| init    | objective   |   frac |   n_seeds |   f1_mean |   f1_std |   ece_mean |   agree_mean |   ep_best_mean |
|:--------|:------------|-------:|----------:|----------:|---------:|-----------:|-------------:|---------------:|
| rand    | ce          |     10 |         5 |    0.6554 |   0.0144 |     0.1121 |       0.7269 |         8.8000 |
| rand    | ce          |     25 |         5 |    0.6997 |   0.0108 |     0.1248 |       0.7729 |         5.2000 |
| rand    | ce          |     50 |         5 |    0.7224 |   0.0085 |     0.0906 |       0.8015 |         4.2000 |
| rand    | ce          |    100 |         5 |    0.7513 |   0.0077 |     0.0820 |       0.8334 |         3.4000 |
| rand    | kd          |     10 |         5 |    0.6764 |   0.0134 |     0.1000 |       0.7670 |         8.2000 |
| rand    | kd          |     25 |         5 |    0.7297 |   0.0087 |     0.1012 |       0.8105 |         7.6000 |
| rand    | kd          |     50 |         5 |    0.7492 |   0.0054 |     0.0954 |       0.8326 |         6.6000 |
| rand    | kd          |    100 |         5 |    0.7555 |   0.0054 |     0.0707 |       0.8472 |         3.2000 |
| strided | ce          |     10 |         5 |    0.7213 |   0.0048 |     0.0778 |       0.7996 |         7.2000 |
| strided | ce          |     25 |         5 |    0.7463 |   0.0057 |     0.0836 |       0.8243 |         5.0000 |
| strided | ce          |     50 |         5 |    0.7603 |   0.0078 |     0.0952 |       0.8496 |         6.2000 |
| strided | ce          |    100 |         5 |    0.7732 |   0.0046 |     0.0994 |       0.8677 |         6.8000 |
| strided | kd          |     10 |         5 |    0.7323 |   0.0057 |     0.0845 |       0.8176 |         8.0000 |
| strided | kd          |     25 |         5 |    0.7551 |   0.0053 |     0.1049 |       0.8522 |         6.6000 |
| strided | kd          |     50 |         5 |    0.7673 |   0.0042 |     0.0935 |       0.8721 |         5.4000 |
| strided | kd          |    100 |         5 |    0.7747 |   0.0062 |     0.0808 |       0.8813 |         4.6000 |

---

## 2. Linear Mixed-Effects Model (Centered log_frac, Full Interactions)
Formula: `f1_macro ~ C(init) * C(objective) * log_frac_c + (1 | seed)`

Direct H1 Wald Contrast (beta_strided - beta_kd at mean log_frac): +2.26 pp (SE=0.28 pp, 95% CI [+1.71, +2.81] pp, chi2=64.37, p=1.0305e-15)

```
                           Mixed Linear Model Regression Results
============================================================================================
Model:                          MixedLM             Dependent Variable:             f1_macro
No. Observations:               80                  Method:                         ML      
No. Groups:                     5                   Scale:                          0.0001  
Min. group size:                16                  Log-Likelihood:                 263.7580
Max. group size:                16                  Converged:                      No      
Mean group size:                16.0                                                        
--------------------------------------------------------------------------------------------
                                                 Coef.  Std.Err.    z    P>|z| [0.025 0.975]
--------------------------------------------------------------------------------------------
Intercept                                         0.707    0.002 347.434 0.000  0.703  0.711
C(init)[T.strided]                                0.043    0.003  15.297 0.000  0.038  0.049
C(objective)[T.kd]                                0.020    0.003   7.274 0.000  0.015  0.026
C(init)[T.strided]:C(objective)[T.kd]            -0.013    0.004  -3.373 0.001 -0.021 -0.006
log_frac_c                                        0.041    0.002  17.548 0.000  0.036  0.046
C(init)[T.strided]:log_frac_c                    -0.019    0.003  -5.612 0.000 -0.025 -0.012
C(objective)[T.kd]:log_frac_c                    -0.006    0.003  -1.941 0.052 -0.013  0.000
C(init)[T.strided]:C(objective)[T.kd]:log_frac_c  0.003    0.005   0.544 0.587 -0.007  0.012
Group Var                                         0.000    0.000                            
============================================================================================

```
