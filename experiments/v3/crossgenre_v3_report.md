# Phase K — Cross-Genre Evaluation Report (News → YouTube Video Titles)

**Target Benchmark:** `abdalimran/BaitBuster-Bangla` ($N = 10,000$)  
**True Clickbait Prevalence (from ground-truth labels):** `56.44%` (Class 0: 4,356, Class 1: 5,644)  
**Recalibration Protocol:** Decision threshold retuned on a 1% stratified sample ($n = 100$), evaluated on remaining 99% ($n = 9,900$).

| model                 |   n_seeds |   zs_acc_mean |   zs_f1_mean | zs_f1_ci         |   recal1pct_acc_mean |   recal1pct_f1_mean | recal1pct_f1_ci   | sample_cm_zeroshot          |
|:----------------------|----------:|--------------:|-------------:|:-----------------|---------------------:|--------------------:|:------------------|:----------------------------|
| teacher_base          |         5 |        0.7866 |       0.7852 | [0.7669, 0.8057] |               0.8551 |              0.8549 | [0.8400, 0.8735]  | [[4307, 49], [2299, 3345]]  |
| baseline_small_12l_ce |         5 |        0.7973 |       0.7963 | [0.7724, 0.8220] |               0.8468 |              0.8464 | [0.8394, 0.8582]  | [[3950, 406], [1202, 4442]] |
| F_strided_kd_100      |         5 |        0.7593 |       0.757  | [0.7387, 0.7802] |               0.8193 |              0.8192 | [0.8130, 0.8276]  | [[4278, 78], [2523, 3121]]  |
| F_strided_ce_100      |         5 |        0.754  |       0.752  | [0.7296, 0.7684] |               0.7879 |              0.7877 | [0.7755, 0.7987]  | [[4231, 125], [2365, 3279]] |
| F_rand_kd_100         |         5 |        0.8121 |       0.8115 | [0.7802, 0.8350] |               0.858  |              0.8565 | [0.8474, 0.8650]  | [[4086, 270], [1285, 4359]] |
