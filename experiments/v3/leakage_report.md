# Leakage & Data Audit Report (v3)

**Generated:** 2026-11-20T16:07:41.466602+00:00  
**Deduplication Method:** MinHash LSH (Threshold $J \ge 0.9$, 128 permutations, character 3-shingles)  
**Partitioning:** 10-Fold Stratified Group K-Fold (Fold 0 = Test, Fold 1 = Val, Folds 2–9 = Train)  

---

## 1. Split Counts & Class Balance

| Split | Total Samples | Not-Clickbait (0) | Clickbait (1) | Clickbait Prevalence (%) |
|---|---|---|---|---|
| **Train (80%)** | 12,025 | 7,845 | 4,180 | 34.76% |
| **Validation (10%)** | 1,503 | 980 | 523 | 34.80% |
| **Test (10%)** | 1,504 | 981 | 523 | 34.77% |
| **Total Clean** | 15,032 | 9,806 | 5,226 | 34.77% |

---

## 2. Leakage Verification & Group Disjointness

- **Label Conflicting Rows Dropped:** 24 rows (11 near-duplicate groups) saved to `data/processed/v3/conflicts.csv`.
- **Train ∩ Val Overlap:** **0 groups** (0 samples)
- **Train ∩ Test Overlap:** **0 groups** (0 samples)
- **Val ∩ Test Overlap:** **0 groups** (0 samples)
- **Cross-Split Near-Duplicate Leakage:** **Strictly 0.00%**

---

## 3. Nested Training Subsets

| Fraction | Sample Count | Class 0 (Not-Clickbait) | Class 1 (Clickbait) | Strict Subset Verification |
|---|---|---|---|---|
| **10%** | 1,202 | 784 | 418 | 10% ⊂ 25% (Verified) |
| **25%** | 3,006 | 1,961 | 1,045 | 25% ⊂ 50% (Verified) |
| **50%** | 6,012 | 3,922 | 2,090 | 50% ⊂ 100% (Verified) |
| **100%** | 12,025 | 7,845 | 4,180 | Full Clean Train Split |
