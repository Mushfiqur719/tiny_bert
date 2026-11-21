# Phase J — One-Time Locked Test Evaluation Report

**Timestamp:** 2026-11-21T07:21:28.124550+00:00  
**Test Split:** `data/processed/v3/splits_v3.json` ($n = 1000$)  
**Bootstrap Resamples:** 10,000 (paired over items and seeds)  
**Non-Inferiority Margin:** $\Delta = -2.0$ percentage points Macro-F1  

## 1. Five-Seed Test Summary

| model                    |   n_seeds |   test_acc_mean |   test_acc_std |   test_f1_mean |   test_f1_std |
|:-------------------------|----------:|----------------:|---------------:|---------------:|--------------:|
| teacher_base_uw          |         5 |          0.9882 |         0.0022 |         0.988  |        0.0023 |
| baseline_small_12l_ce_uw |         5 |          0.987  |         0.0017 |         0.9868 |        0.0017 |
| F_strided_kd_100_uw      |         5 |          0.981  |         0.0014 |         0.9807 |        0.0014 |
| F_strided_ce_100_uw      |         5 |          0.9814 |         0.0023 |         0.9811 |        0.0024 |
| F_rand_kd_100_uw         |         5 |          0.9792 |         0.0034 |         0.9788 |        0.0035 |
| F_rand_ce_100_uw         |         5 |          0.9796 |         0.0027 |         0.9792 |        0.0028 |

## 2. Paired Bootstrap & Non-Inferiority Tests (Holm-Corrected)

| model                    | reference                |   f1_a |   f1_b |   diff_pp |   ci_lo_pp |   ci_hi_pp |   p_value | non_inferior_2pp   |   p_holm |
|:-------------------------|:-------------------------|-------:|-------:|----------:|-----------:|-----------:|----------:|:-------------------|---------:|
| teacher_base_uw          | baseline_small_12l_ce_uw | 0.988  | 0.9868 |      0.12 |      -0.37 |       0.62 |    0.6072 | True               |   1      |
| baseline_small_12l_ce_uw | teacher_base_uw          | 0.9868 | 0.988  |     -0.12 |      -0.62 |       0.37 |    0.6072 | True               |   1      |
| F_strided_kd_100_uw      | teacher_base_uw          | 0.9807 | 0.988  |     -0.73 |      -1.46 |      -0.06 |    0.0314 | True               |   0.2128 |
| F_strided_kd_100_uw      | baseline_small_12l_ce_uw | 0.9807 | 0.9868 |     -0.61 |      -1.35 |       0.04 |    0.0648 | True               |   0.2592 |
| F_strided_ce_100_uw      | teacher_base_uw          | 0.9811 | 0.988  |     -0.69 |      -1.39 |      -0.04 |    0.0364 | True               |   0.2128 |
| F_strided_ce_100_uw      | baseline_small_12l_ce_uw | 0.9811 | 0.9868 |     -0.57 |      -1.3  |       0.1  |    0.0936 | True               |   0.2808 |
| F_rand_kd_100_uw         | teacher_base_uw          | 0.9788 | 0.988  |     -0.92 |      -1.65 |      -0.26 |    0.0062 | True               |   0.062  |
| F_rand_kd_100_uw         | baseline_small_12l_ce_uw | 0.9788 | 0.9868 |     -0.8  |      -1.54 |      -0.08 |    0.0266 | True               |   0.2128 |
| F_rand_ce_100_uw         | teacher_base_uw          | 0.9792 | 0.988  |     -0.88 |      -1.56 |      -0.24 |    0.0068 | True               |   0.062  |
| F_rand_ce_100_uw         | baseline_small_12l_ce_uw | 0.9792 | 0.9868 |     -0.75 |      -1.48 |      -0.06 |    0.0302 | True               |   0.2128 |
