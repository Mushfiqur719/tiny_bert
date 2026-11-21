# TinyBanglaBERT Improvement Plan (v2) - Architect Review

## 0. Status
Written plan only. No code run, no repo files modified, no models trained. All original artifacts untouched.

## 1. Realistic accuracy targets
- Current: Teacher 83.73% Acc / 0.8174 F1; TinyBanglaBERT-4L 80.54% / 0.7895 (FP32), 80.81% / 0.7917 (INT8).
- Published BanglaBait SOTA ~84.06% Acc, clickbait F1 0.7595 (Springer 2026).
- Defensible target: student 86-89% Acc, 0.84-0.87 macro-F1 (5 seeds, leak-free split).
- 95%+ only as selective accuracy (accuracy @ coverage, AURC), reported with coverage.
- BaitBuster 99% numbers are YouTube auto-labels; not comparable.

## 2. Critical integrity fixes (do first)
| Issue | Evidence | Fix |
|---|---|---|
| Two "final" models | README 11.55M/0.7431/75.63%/5.15ms vs FINAL_RESULTS 7.42M/0.7895/80.54%/4.00ms | Regenerate all docs from CSVs |
| Teacher drift | 0.8233/84.20% vs 0.8174/83.73% | Multi-seed mean+-std |
| KD hparams mismatch | Text T=2,a=0.1; table T=4,a=0.1/0.5 | Log exact config per run |
| SHAP/LIME mismatch | README 82.5% sign / 0.551 Jaccard vs Fig5 48.8% / 32.7%, n=15 | n>=300, add ERASER faithfulness |
| Cross-domain contradiction | Teacher 0.7282 vs 0.7514 | Single run; drop regularizer claim |
| Test set reused | 9+ models + "cal" metrics on test | Calibrate/threshold on val only |
| Train-test leakage | 155 overlapping test headlines | MinHash near-dup removal, splits_v2.json |
| Single seed | #Seeds=1 for most groups | 5 seeds per config |
| Confounded ablation | Winner changed architecture (7.42M, 128E) | Two studies: init x objective; depth scaling |

## 3. Accuracy roadmap (priority order, est. macro-F1 gain)
1. Leak-free re-split (NFC, near-dup Jaccard>=0.9, stratified 80/10/10).
2. Label ceiling + cleanlab audit on train only (+1-2 pp).
3. Headline + first 200 body tokens (+2-4 pp); keep headline-only student.
4. Domain-adaptive MLM pretraining on 65,406 unlabeled articles (+1.5-3 pp).
5. Teacher ensemble: BanglaBERT-large, XLM-R-large, DAPT-BanglaBERT; 5 seeds; LR {1e-5,2e-5,3e-5}; R-Drop/FGM; LLRD (+2-3 pp teacher).
6. Self-training: ensemble pseudo-labels, conf>=0.9, class-balanced (+2-4 pp student).
7. Intermediate KD (hidden + attention, 256->768/1024 projection); Optuna T{1,2,4,8} x a{.3,.5,.7,.9} on val.
8. Back-translation augmentation (BanglaT5), teacher-labeled, KD set only.
9. Temperature scaling + F1-optimal threshold on val, frozen (+0.3-0.8 pp).
10. Selective prediction: accuracy-coverage curve, AURC, edge->teacher deferral.

## 4. Agent phases
A. Consistency: scripts/build_reports.py; test fails if docs != CSVs.
B. Splits v2: src/data/dedup.py + tests + leakage report.
C. Data: src/data/load_full.py; config text_mode: title | title_body.
D. DAPT: scripts/dapt.py -> experiments/dapt_banglabert/.
E. Teachers: configs/teacher_{base,large,xlmr,dapt}.yaml; scripts/train_multiseed.py; save OOF/val/unlabeled logits.
F. Label audit: scripts/label_audit.py (cleanlab).
G. Students: extend src/models/kd_loss.py; 4L/6L/12L-small x 5 seeds + pseudo-labels.
H. Lock: HPO + calibration on val; commit final_config.lock.yaml.
I. Test once: bootstrap 95% CI, McNemar + Holm, seed-paired bootstrap.
J. Cross-domain: BaitBuster human labels, deduped; zero-shot + 1% few-shot.
K. Efficiency: fixed hardware/threads/seqlen/warmup; add onnxruntime-web (WASM).
L. Explainability: 300+ stratified samples, comprehensiveness/sufficiency, small human plausibility study.

## 5. Paper additions
- Baselines: DistilmBERT, multilingual MiniLM, TinyBERT recipe, BanglaBERT-small FT, SS-GAN.
- Error taxonomy (sensational language ~40% of errors in prior work).
- Limitations: label ceiling, YouTube vs news domain, single annotation source.
