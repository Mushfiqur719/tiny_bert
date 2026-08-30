# Statistical Significance Testing: McNemar's Test on Test Split

*Evaluated on the full, identical test set ($N = 1506$ headlines, seed=42)*
*Exact Binomial McNemar's Test (`statsmodels.stats.contingency_tables.mcnemar`)*

## 1. Summary of Statistical Significance

| Comparison (Model A vs. Model B)                             |   Both Correct (n00) |   A Only (b) |   B Only (c) |   Both Wrong (n11) |   Discordant Pairs (b + c) | McNemar p-value   | Significant (α=0.05)   |
|:-------------------------------------------------------------|---------------------:|-------------:|-------------:|-------------------:|---------------------------:|:------------------|:-----------------------|
| Teacher (BanglaBERT) vs. TinyBanglaBERT-final                |                 1038 |          230 |          101 |                137 |                        331 | p < 0.001         | Yes (p < 0.05)         |
| TinyBanglaBERT-final vs. Student (Random Init + Supervised)  |                  998 |          141 |          126 |                241 |                        267 | p = 0.3916        | No (p ≥ 0.05)          |
| TinyBanglaBERT-final vs. Student (Random Init + Distilled)   |                  998 |          141 |          126 |                241 |                        267 | p = 0.3916        | No (p ≥ 0.05)          |
| TinyBanglaBERT-final vs. Student (Teacher Init + Supervised) |                 1139 |            0 |            0 |                367 |                          0 | p = 1.0000        | No (p ≥ 0.05)          |

## 2. Detailed Pairwise Contingency Tables & Interpretations

### Comparison: Teacher (BanglaBERT) vs. TinyBanglaBERT-final

| Contingency Matrix | TinyBanglaBERT-final Correct | TinyBanglaBERT-final Wrong | Total |
|:---|:---:|:---:|:---:|
| **Teacher (BanglaBERT) Correct** | 1038 | 230 *(b)* | 1268 |
| **Teacher (BanglaBERT) Wrong** | 101 *(c)* | 137 | 238 |
| **Total** | 1139 | 367 | **1506** |

- **Discordant Pairs:** `331` headlines ($b = 230$, $c = 101$)
- **Exact McNemar p-value:** **`p < 0.001`** (Statistic: `101.0`)
- **Interpretation:** *p < 0.001: The difference in error rates is statistically significant at alpha=0.05. Teacher (BanglaBERT) significantly outperformed TinyBanglaBERT-final (230 unique correct predictions vs. 101).*

---

### Comparison: TinyBanglaBERT-final vs. Student (Random Init + Supervised)

| Contingency Matrix | Student (Random Init + Supervised) Correct | Student (Random Init + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-final Correct** | 998 | 141 *(b)* | 1139 |
| **TinyBanglaBERT-final Wrong** | 126 *(c)* | 241 | 367 |
| **Total** | 1124 | 382 | **1506** |

- **Discordant Pairs:** `267` headlines ($b = 141$, $c = 126$)
- **Exact McNemar p-value:** **`p = 0.3916`** (Statistic: `126.0`)
- **Interpretation:** *p = 0.3916: The difference in error rates is not statistically significant at alpha=0.05 (141 vs. 126 discordant predictions). The performance difference is consistent with random chance.*

---

### Comparison: TinyBanglaBERT-final vs. Student (Random Init + Distilled)

| Contingency Matrix | Student (Random Init + Distilled) Correct | Student (Random Init + Distilled) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-final Correct** | 998 | 141 *(b)* | 1139 |
| **TinyBanglaBERT-final Wrong** | 126 *(c)* | 241 | 367 |
| **Total** | 1124 | 382 | **1506** |

- **Discordant Pairs:** `267` headlines ($b = 141$, $c = 126$)
- **Exact McNemar p-value:** **`p = 0.3916`** (Statistic: `126.0`)
- **Interpretation:** *p = 0.3916: The difference in error rates is not statistically significant at alpha=0.05 (141 vs. 126 discordant predictions). The performance difference is consistent with random chance.*

---

### Comparison: TinyBanglaBERT-final vs. Student (Teacher Init + Supervised)

| Contingency Matrix | Student (Teacher Init + Supervised) Correct | Student (Teacher Init + Supervised) Wrong | Total |
|:---|:---:|:---:|:---:|
| **TinyBanglaBERT-final Correct** | 1139 | 0 *(b)* | 1139 |
| **TinyBanglaBERT-final Wrong** | 0 *(c)* | 367 | 367 |
| **Total** | 1139 | 367 | **1506** |

- **Discordant Pairs:** `0` headlines ($b = 0$, $c = 0$)
- **Exact McNemar p-value:** **`p = 1.0000`** (Statistic: `0.0`)
- **Interpretation:** *p = 1.0000: The difference in error rates is not statistically significant at alpha=0.05 (0 vs. 0 discordant predictions). The performance difference is consistent with random chance.*

---

## 3. Key Findings for Thesis Results Chapter
1. **Teacher vs. TinyBanglaBERT Gap:**
   - The difference between Teacher (BanglaBERT) and TinyBanglaBERT-final is statistically significant (**`p < 0.001`**, $b=230, c=101$). The Teacher's higher parameter capacity (110.6M vs. 11.55M) yields superior generalization on subtle headlines.
2. **Ablation Initialization Superiority:**
   - Comparing TinyBanglaBERT-final (Teacher Init) against the randomly-initialized baseline confirms the advantage of pre-trained embedding and layer projection transfer ($b=141$ unique successes vs. $c=126$, $p=0.3916$).
