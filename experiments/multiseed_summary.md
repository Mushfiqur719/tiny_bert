# Multi-seed & Bootstrap Summary (test split, mean ± sample std over seeds)

| Group | #Seeds | Acc | Macro-F1 | 95% Bootstrap CI (F1) | Clickbait-F1 | Acc (cal) | Macro-F1 (cal) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| teacher | 1 | 0.8373 ± 0.0000 | 0.8174 ± 0.0000 | n/a | 0.7572 ± 0.0000 | 0.8400 ± 0.0000 | 0.8220 ± 0.0000 |
| student_small_12l_distilled | 1 | 0.8114 ± 0.0000 | 0.7926 ± 0.0000 | [0.7712, 0.8128] | 0.7300 ± 0.0000 | 0.8134 ± 0.0000 | 0.7969 ± 0.0000 |
| student_small_6l_distilled | 1 | 0.7975 ± 0.0000 | 0.7824 ± 0.0000 | [0.7597, 0.8031] | 0.7250 ± 0.0000 | 0.7961 ± 0.0000 | 0.7813 ± 0.0000 |
| student_small_pretrained_distilled | 3 | 0.8006 ± 0.0058 | 0.7824 ± 0.0072 | [0.7604, 0.8034] | 0.7195 ± 0.0110 | 0.8032 ± 0.0010 | 0.7841 ± 0.0032 |
| student_small_pretrained_baseline | 1 | 0.7975 ± 0.0000 | 0.7830 ± 0.0000 | [0.7610, 0.8035] | 0.7269 ± 0.0000 | 0.7948 ± 0.0000 | 0.7808 ± 0.0000 |
| student_teacher_init_distilled | 1 | 0.7842 ± 0.0000 | 0.7615 ± 0.0000 | n/a | 0.6878 ± 0.0000 | 0.7875 ± 0.0000 | 0.7671 ± 0.0000 |
| student_teacher_init_baseline | 1 | 0.7583 ± 0.0000 | 0.7472 ± 0.0000 | [0.7234, 0.7697] | 0.6941 ± 0.0000 | 0.7663 ± 0.0000 | 0.7528 ± 0.0000 |
| student_random_distilled | 1 | 0.7709 ± 0.0000 | 0.7488 ± 0.0000 | [0.7238, 0.7699] | 0.6742 ± 0.0000 | 0.7722 ± 0.0000 | 0.7489 ± 0.0000 |
| student_random_baseline | 1 | 0.7656 ± 0.0000 | 0.7390 ± 0.0000 | [0.7154, 0.7608] | 0.6556 ± 0.0000 | 0.7636 ± 0.0000 | 0.7330 ± 0.0000 |

## Paired bootstrap: student_small_pretrained_distilled − student_teacher_init_distilled (Macro-F1, seed-matched)

| Seed | ΔF1 | 95% CI | p | p (Holm) |
|:---:|:---:|:---:|:---:|:---:|
| no seed-matched runs with saved logits | | | | |