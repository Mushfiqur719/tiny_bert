# Leakage & Data Audit Report — BaitBuster-Bangla (v3)

**Generated:** 2026-11-20T21:37:41.917755+00:00  
**Source:** `abdalimran/BaitBuster-Bangla` (10,000 human-annotated YouTube video titles)  
**Deduplication:** MinHash LSH ($J \ge 0.9$, 128 permutations)  

| Split | Total | Not-Clickbait (0) | Clickbait (1) | Prevalence |
|---|---|---|---|---|
| **Train (80%)** | 8,000 | 3,484 | 4,516 | 56.45% |
| **Validation (10%)** | 1,000 | 436 | 564 | 56.40% |
| **Test (10%)** | 1,000 | 436 | 564 | 56.40% |
| **Total Clean** | 10,000 | 4,356 | 5,644 | 56.44% |

- **Conflicts Dropped:** 0 rows (0 groups)
- **Cross-Split Leakage:** **0 groups (0.00%)**
