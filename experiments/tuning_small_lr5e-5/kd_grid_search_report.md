# Knowledge Distillation (T, alpha) Hyperparameter Search Report

*Generated: 2026-11-19 20:24 UTC*
*Optimization Metric: Validation Set Macro-F1 (Held-out Test set untouched)*
*Initialization Strategy: small_pretrained*

## Best Hyperparameter Setting
- **Optimal Temperature ($T^*$):** `6.0`
- **Optimal Alpha ($\alpha^*$):** `0.3`
- **Best Validation Macro-F1:** **`0.7925`**
- **Validation Accuracy:** `0.8081`

## Full Grid Search Results (Validation Split)

| Rank | Temperature ($T$) | Alpha ($\alpha$) | Val Macro-F1 | Val Accuracy | Val Clickbait F1 | Val Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.0 | 0.3 | **0.7925** | 0.8081 | 0.7356 | 0.5067 |
| 2 | 2.0 | 0.1 | 0.7914 | 0.8094 | 0.7300 | 0.5059 |
| 3 | 4.0 | 0.1 | 0.7901 | 0.8081 | 0.7286 | 0.5165 |
| 4 | 6.0 | 0.1 | 0.7895 | 0.8074 | 0.7280 | 0.5190 |
| 5 | 4.0 | 0.3 | 0.7893 | 0.8048 | 0.7322 | 0.5045 |
| 6 | 2.0 | 0.5 | 0.7877 | 0.8054 | 0.7264 | 0.5140 |
| 7 | 4.0 | 0.5 | 0.7863 | 0.8041 | 0.7246 | 0.5217 |
| 8 | 6.0 | 0.5 | 0.7857 | 0.8035 | 0.7239 | 0.5243 |
| 9 | 2.0 | 0.3 | 0.7855 | 0.8054 | 0.7202 | 0.5125 |

## Validation Macro-F1 Response Surface Matrix ($T$ vs $\alpha$)

|   temperature |    0.1 |    0.3 |    0.5 |
|--------------:|-------:|-------:|-------:|
|             2 | 0.7914 | 0.7855 | 0.7877 |
|             4 | 0.7901 | 0.7893 | 0.7863 |
|             6 | 0.7895 | 0.7925 | 0.7857 |
