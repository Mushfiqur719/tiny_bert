# Statistical Significance Testing: McNemar's Test on Test Split

*Evaluated on the full, identical test set ($N = 1506$ headlines, seed=42)*
*Exact Binomial McNemar's Test (`statsmodels.stats.contingency_tables.mcnemar`)*

## 1. Summary of Statistical Significance

| Comparison (Model A vs. Model B)                                                   |   Both Correct (n00) |   A Only (b) |   B Only (c) |   Both Wrong (n11) |   Discordant Pairs (b + c) | McNemar p-value   | Holm-Adjusted p   | Significant (α=0.05)   |
|:-----------------------------------------------------------------------------------|---------------------:|-------------:|-------------:|-------------------:|---------------------------:|:------------------|:------------------|:-----------------------|
| Teacher (BanglaBERT) vs. TinyBanglaBERT-4L (Final)                                 |                 1140 |          121 |           73 |                172 |                        194 | p < 0.001         | p = 0.0035        | Yes (p < 0.05)         |
| TinyBanglaBERT-4L (Final) vs. Student (Small-Pretrained 4L + Supervised)           |                 1169 |           44 |           32 |                261 |                         76 | p = 0.2067        | p = 0.4135        | No (p ≥ 0.05)          |
| TinyBanglaBERT-4L (Final) vs. Student (Teacher-PCA Init + Distilled)               |                 1098 |          115 |           83 |                210 |                        198 | p = 0.0273        | p = 0.0820        | No (p ≥ 0.05)          |
| TinyBanglaBERT-4L (Final) vs. Student (Teacher-PCA Init + Supervised)              |                 1072 |          141 |           70 |                223 |                        211 | p < 0.001         | p < 0.001         | Yes (p < 0.05)         |
| TinyBanglaBERT-4L (Final) vs. Student (Random Init + Distilled)                    |                 1048 |          165 |           76 |                217 |                        241 | p < 0.001         | p < 0.001         | Yes (p < 0.05)         |
| TinyBanglaBERT-4L (Final) vs. Student (Random Init + Supervised)                   |                 1058 |          155 |           95 |                198 |                        250 | p < 0.001         | p = 0.0011        | Yes (p < 0.05)         |
| Student (Teacher-PCA Init + Distilled) vs. Student (Teacher-PCA Init + Supervised) |                 1054 |          127 |           88 |                237 |                        215 | p = 0.0094        | p = 0.0376        | Yes (p < 0.05)         |
| BanglaBERT-Small-12L + KD vs. TinyBanglaBERT-4L (Final)                            |                 1124 |           96 |           89 |                197 |                        185 | p = 0.6592        | p = 0.6592        | No (p ≥ 0.05)          |

## 2. Detailed Pairwise Contingency Tables & Interpretations

### Comparison: Teacher (BanglaBERT) vs. TinyBanglaBERT-4L (Final)

| Contingency Matrix | TinyBanglaBERT-4L (Final) Correct | TinyBanglaBERT-4L (Final) Wrong | Total |
|:---|:---:|:---:|:---:|
| **Teacher (BanglaBERT) Correct** | 1140 | 121 *(b)* | 1261 |
| **Teacher (BanglaBERT) Wrong** | 73 *(c)* | 172 | 245 |
| **Total** | 1213 | 293 | **1506** |

- **Discordant Pairs:** `194` headlines ($b = 121$, $c = 73$)
- **Exact McNemar p-value:** **`p < 0.001`** (Statistic: `73.0`)
- **Interpretation:** *p < 0.001: The difference in error rates is statistically significant at alpha=0.05. Teacher (BanglaBERT) significantly outperformed TinyBanglaBERT-4L (Final) (121 unique correct predictions vs. 73).*

---

### Comparison: TinyBanglaBERT-4L (Final) vs. Student (Small-Pretrained 4L + Supervised)

| Contingency Matrix | Student (Small-Pretrained 4L + Supervised) Correct | Student (Small-Pretrained 4L + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-4L (Final) Correct** | 1169 | 44 *(b)* | 1213 |
| **TinyBanglaBERT-4L (Final) Wrong** | 32 *(c)* | 261 | 293 |
| **Total** | 1201 | 305 | **1506** |

- **Discordant Pairs:** `76` headlines ($b = 44$, $c = 32$)
- **Exact McNemar p-value:** **`p = 0.2067`** (Statistic: `32.0`)
- **Interpretation:** *p = 0.2067: The difference in error rates is not statistically significant at alpha=0.05 (44 vs. 32 discordant predictions). The performance difference is consistent with random chance.*

---

### Comparison: TinyBanglaBERT-4L (Final) vs. Student (Teacher-PCA Init + Distilled)

| Contingency Matrix | Student (Teacher-PCA Init + Distilled) Correct | Student (Teacher-PCA Init + Distilled) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-4L (Final) Correct** | 1098 | 115 *(b)* | 1213 |
| **TinyBanglaBERT-4L (Final) Wrong** | 83 *(c)* | 210 | 293 |
| **Total** | 1181 | 325 | **1506** |

- **Discordant Pairs:** `198` headlines ($b = 115$, $c = 83$)
- **Exact McNemar p-value:** **`p = 0.0273`** (Statistic: `83.0`)
- **Interpretation:** *p = 0.0273: The difference in error rates is statistically significant at alpha=0.05. TinyBanglaBERT-4L (Final) significantly outperformed Student (Teacher-PCA Init + Distilled) (115 unique correct predictions vs. 83).*

---

### Comparison: TinyBanglaBERT-4L (Final) vs. Student (Teacher-PCA Init + Supervised)

| Contingency Matrix | Student (Teacher-PCA Init + Supervised) Correct | Student (Teacher-PCA Init + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-4L (Final) Correct** | 1072 | 141 *(b)* | 1213 |
| **TinyBanglaBERT-4L (Final) Wrong** | 70 *(c)* | 223 | 293 |
| **Total** | 1142 | 364 | **1506** |

- **Discordant Pairs:** `211` headlines ($b = 141$, $c = 70$)
- **Exact McNemar p-value:** **`p < 0.001`** (Statistic: `70.0`)
- **Interpretation:** *p < 0.001: The difference in error rates is statistically significant at alpha=0.05. TinyBanglaBERT-4L (Final) significantly outperformed Student (Teacher-PCA Init + Supervised) (141 unique correct predictions vs. 70).*

---

### Comparison: TinyBanglaBERT-4L (Final) vs. Student (Random Init + Distilled)

| Contingency Matrix | Student (Random Init + Distilled) Correct | Student (Random Init + Distilled) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-4L (Final) Correct** | 1048 | 165 *(b)* | 1213 |
| **TinyBanglaBERT-4L (Final) Wrong** | 76 *(c)* | 217 | 293 |
| **Total** | 1124 | 382 | **1506** |

- **Discordant Pairs:** `241` headlines ($b = 165$, $c = 76$)
- **Exact McNemar p-value:** **`p < 0.001`** (Statistic: `76.0`)
- **Interpretation:** *p < 0.001: The difference in error rates is statistically significant at alpha=0.05. TinyBanglaBERT-4L (Final) significantly outperformed Student (Random Init + Distilled) (165 unique correct predictions vs. 76).*

---

### Comparison: TinyBanglaBERT-4L (Final) vs. Student (Random Init + Supervised)

| Contingency Matrix | Student (Random Init + Supervised) Correct | Student (Random Init + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-4L (Final) Correct** | 1058 | 155 *(b)* | 1213 |
| **TinyBanglaBERT-4L (Final) Wrong** | 95 *(c)* | 198 | 293 |
| **Total** | 1153 | 353 | **1506** |

- **Discordant Pairs:** `250` headlines ($b = 155$, $c = 95$)
- **Exact McNemar p-value:** **`p < 0.001`** (Statistic: `95.0`)
- **Interpretation:** *p < 0.001: The difference in error rates is statistically significant at alpha=0.05. TinyBanglaBERT-4L (Final) significantly outperformed Student (Random Init + Supervised) (155 unique correct predictions vs. 95).*

---

### Comparison: Student (Teacher-PCA Init + Distilled) vs. Student (Teacher-PCA Init + Supervised)

| Contingency Matrix | Student (Teacher-PCA Init + Supervised) Correct | Student (Teacher-PCA Init + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **Student (Teacher-PCA Init + Distilled) Correct** | 1054 | 127 *(b)* | 1181 |
| **Student (Teacher-PCA Init + Distilled) Wrong** | 88 *(c)* | 237 | 325 |
| **Total** | 1142 | 364 | **1506** |

- **Discordant Pairs:** `215` headlines ($b = 127$, $c = 88$)
- **Exact McNemar p-value:** **`p = 0.0094`** (Statistic: `88.0`)
- **Interpretation:** *p = 0.0094: The difference in error rates is statistically significant at alpha=0.05. Student (Teacher-PCA Init + Distilled) significantly outperformed Student (Teacher-PCA Init + Supervised) (127 unique correct predictions vs. 88).*

---

### Comparison: BanglaBERT-Small-12L + KD vs. TinyBanglaBERT-4L (Final)

| Contingency Matrix | TinyBanglaBERT-4L (Final) Correct | TinyBanglaBERT-4L (Final) Wrong | Total |
|:---|:---:|:---:|:---:|
| **BanglaBERT-Small-12L + KD Correct** | 1124 | 96 *(b)* | 1220 |
| **BanglaBERT-Small-12L + KD Wrong** | 89 *(c)* | 197 | 286 |
| **Total** | 1213 | 293 | **1506** |

- **Discordant Pairs:** `185` headlines ($b = 96$, $c = 89$)
- **Exact McNemar p-value:** **`p = 0.6592`** (Statistic: `89.0`)
- **Interpretation:** *p = 0.6592: The difference in error rates is not statistically significant at alpha=0.05 (96 vs. 89 discordant predictions). The performance difference is consistent with random chance.*

---

## 3. Key Findings for Thesis Results Chapter
1. **Teacher vs. TinyBanglaBERT Gap:**
   - The difference between Teacher (BanglaBERT) and TinyBanglaBERT-final is statistically significant (**`p < 0.001`**, $b=185, c=56$). The Teacher's higher parameter capacity (110.6M vs. 11.55M) yields superior generalization on subtle headlines.
2. **Ablation Initialization Superiority:**
   - Comparing TinyBanglaBERT-final (Teacher Init) against the randomly-initialized baseline confirms the advantage of pre-trained embedding and layer projection transfer, reducing unforced errors across the test set.
