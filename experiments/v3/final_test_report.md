# Phase J — One-Time Locked Test Evaluation Report

**Timestamp:** 2026-11-21T08:13:23.853364+00:00  
**Test Split:** `data/processed/v3/splits_v3.json` ($n = 1504$)  
**Bootstrap Resamples:** 10,000 (paired over items and seeds)  
**Non-Inferiority Margin:** $\Delta = -2.0$ percentage points Macro-F1  

## 1. Five-Seed Test Summary

| model                 |   n_seeds |   test_acc_mean |   test_acc_std |   test_f1_mean |   test_f1_std |
|:----------------------|----------:|----------------:|---------------:|---------------:|--------------:|
| teacher_base          |         5 |          0.7935 |         0.0065 |         0.7723 |        0.0034 |
| teacher_base_tb       |         5 |          0.8135 |         0.0078 |         0.7934 |        0.0074 |
| baseline_small_12l_ce |         5 |          0.7763 |         0.0088 |         0.7543 |        0.0078 |
| F_strided_kd_100      |         5 |          0.7786 |         0.0056 |         0.7537 |        0.0063 |
| F_strided_ce_100      |         5 |          0.7642 |         0.0066 |         0.7451 |        0.0035 |
| F_rand_kd_100         |         5 |          0.7527 |         0.011  |         0.7222 |        0.0078 |
| F_rand_ce_100         |         5 |          0.7551 |         0.0076 |         0.7239 |        0.0046 |

## 2. Paired Bootstrap & Non-Inferiority Tests (Holm-Corrected)

| model                 | reference             |   f1_a |   f1_b |   diff_pp |   ci_lo_pp |   ci_hi_pp |   p_value | non_inferior_2pp   |   p_holm |
|:----------------------|:----------------------|-------:|-------:|----------:|-----------:|-----------:|----------:|:-------------------|---------:|
| teacher_base          | teacher_base_tb       | 0.7723 | 0.7934 |     -2.11 |      -3.64 |      -0.65 |    0.004  | False              |   0.028  |
| teacher_base          | baseline_small_12l_ce | 0.7723 | 0.7543 |      1.8  |       0.12 |       3.45 |    0.0368 | True               |   0.178  |
| teacher_base_tb       | teacher_base          | 0.7934 | 0.7723 |      2.11 |       0.65 |       3.64 |    0.004  | True               |   0.028  |
| teacher_base_tb       | baseline_small_12l_ce | 0.7934 | 0.7543 |      3.91 |       2.21 |       5.69 |    0.0001 | True               |   0.0018 |
| baseline_small_12l_ce | teacher_base          | 0.7543 | 0.7723 |     -1.8  |      -3.45 |      -0.12 |    0.0368 | False              |   0.178  |
| baseline_small_12l_ce | teacher_base_tb       | 0.7543 | 0.7934 |     -3.91 |      -5.69 |      -2.21 |    0.0001 | False              |   0.0018 |
| F_strided_kd_100      | teacher_base          | 0.7537 | 0.7723 |     -1.86 |      -3.65 |      -0.13 |    0.0356 | False              |   0.178  |
| F_strided_kd_100      | teacher_base_tb       | 0.7537 | 0.7934 |     -3.97 |      -6.01 |      -2.09 |    0.0001 | False              |   0.0018 |
| F_strided_kd_100      | baseline_small_12l_ce | 0.7537 | 0.7543 |     -0.06 |      -1.71 |       1.56 |    0.9412 | True               |   0.9412 |
| F_strided_ce_100      | teacher_base          | 0.7451 | 0.7723 |     -2.72 |      -4.36 |      -1.04 |    0.0018 | False              |   0.0162 |
| F_strided_ce_100      | teacher_base_tb       | 0.7451 | 0.7934 |     -4.83 |      -6.63 |      -3.02 |    0.0001 | False              |   0.0018 |
| F_strided_ce_100      | baseline_small_12l_ce | 0.7451 | 0.7543 |     -0.92 |      -2.54 |       0.68 |    0.2626 | False              |   0.5252 |
| F_rand_kd_100         | teacher_base          | 0.7222 | 0.7723 |     -5    |      -7.08 |      -3    |    0.0001 | False              |   0.0018 |
| F_rand_kd_100         | teacher_base_tb       | 0.7222 | 0.7934 |     -7.11 |      -9.43 |      -4.95 |    0.0001 | False              |   0.0018 |
| F_rand_kd_100         | baseline_small_12l_ce | 0.7222 | 0.7543 |     -3.21 |      -5.14 |      -1.25 |    0.0012 | False              |   0.012  |
| F_rand_ce_100         | teacher_base          | 0.7239 | 0.7723 |     -4.83 |      -6.73 |      -2.9  |    0.0001 | False              |   0.0018 |
| F_rand_ce_100         | teacher_base_tb       | 0.7239 | 0.7934 |     -6.94 |      -9.07 |      -4.87 |    0.0001 | False              |   0.0018 |
| F_rand_ce_100         | baseline_small_12l_ce | 0.7239 | 0.7543 |     -3.04 |      -5.09 |      -1.04 |    0.0028 | False              |   0.0224 |
