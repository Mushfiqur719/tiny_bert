# Selective Prediction (Risk–Coverage / AURC) & Edge–Cloud Deferral Report

- **Validation-Fitted Calibration Temperature ($T_{\text{cal}}$):** `1.7375`
- **Area Under Risk–Coverage Curve (AURC):** **`0.10268`** *(lower is better)*
- **Coverage at $\ge 90\%$ Selective Accuracy:** **`61.5%`**
- **Coverage at $\ge 92\%$ Selective Accuracy:** **`36.5%`**
- **Coverage at $\ge 95\%$ Selective Accuracy:** **`7.2%`**
- **Peak High-Confidence Selective Accuracy:** **`97.50%`**

## 1. Selective Accuracy & Macro-F1 Across Coverage Thresholds (TinyBanglaBERT-4L)

|   Coverage (%) |   Retained Samples |   Confidence Threshold |   Selective Accuracy (%) |   Selective Macro-F1 |   Selective Clickbait-F1 |   Error Risk (%) |
|---------------:|-------------------:|-----------------------:|-------------------------:|---------------------:|-------------------------:|-----------------:|
|            100 |               1506 |                 0.5005 |                    80.54 |               0.7895 |                   0.7314 |            19.46 |
|             90 |               1355 |                 0.6526 |                    83.69 |               0.8199 |                   0.7646 |            16.31 |
|             85 |               1280 |                 0.7081 |                    84.92 |               0.8321 |                   0.7784 |            15.08 |
|             80 |               1205 |                 0.7517 |                    85.98 |               0.8441 |                   0.7947 |            14.02 |
|             75 |               1130 |                 0.7796 |                    87.26 |               0.8571 |                   0.81   |            12.74 |
|             70 |               1054 |                 0.8058 |                    88.33 |               0.8674 |                   0.8215 |            11.67 |
|             60 |                904 |                 0.8376 |                    90.38 |               0.8905 |                   0.8523 |             9.62 |
|             50 |                753 |                 0.8597 |                    90.84 |               0.8966 |                   0.8617 |             9.16 |
|             40 |                602 |                 0.8701 |                    91.69 |               0.9101 |                   0.8853 |             8.31 |
|             30 |                452 |                 0.8774 |                    92.92 |               0.9283 |                   0.92   |             7.08 |

## 2. Two-Tier Edge–Cloud Deferral Cascade (`TinyBanglaBERT-4L INT8` 4.00 ms $\to$ `Teacher`)

|   Edge Coverage (%) |   Cloud Deferral (%) |   Cloud Calls Saved (%) |   System Accuracy (%) |   System Macro-F1 |   System Clickbait-F1 |   Amortized Latency (Edge+GPU ms) |   Amortized Latency (Edge+CPU ms) |
|--------------------:|---------------------:|------------------------:|----------------------:|------------------:|----------------------:|----------------------------------:|----------------------------------:|
|                 100 |                    0 |                     100 |                 80.54 |            0.7895 |                0.7314 |                              4    |                              4    |
|                  90 |                   10 |                      90 |                 82.27 |            0.8054 |                0.7474 |                              5.26 |                             16.34 |
|                  80 |                   20 |                      80 |                 83.2  |            0.8145 |                0.7574 |                              6.52 |                             28.67 |
|                  75 |                   25 |                      75 |                 83.13 |            0.8133 |                0.7553 |                              7.15 |                             34.84 |
|                  70 |                   30 |                      70 |                 83    |            0.8115 |                0.7524 |                              7.78 |                             41.01 |
|                  60 |                   40 |                      60 |                 83.6  |            0.8177 |                0.76   |                              9.04 |                             53.34 |
|                  50 |                   50 |                      50 |                 83.2  |            0.8131 |                0.7537 |                             10.3  |                             65.68 |
|                   0 |                  100 |                       0 |                 83.73 |            0.8174 |                0.7572 |                             12.6  |                            123.36 |
