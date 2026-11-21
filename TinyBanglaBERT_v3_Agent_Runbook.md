# TinyBanglaBERT v3 — Agent Runbook

> Audience: any coding agent (or human) with shell access to `tiny_bert/`.
> Research question: **in low-resource fine-tuning, does initialisation matter more than the distillation objective?**
> Supersedes the v2 runbook. Where a step says "v2 Phase X", use the text of `TinyBanglaBERT_v2_Agent_Runbook.md`.
> This plan fixes the review issues. It cannot guarantee a score; the venue is decided at the gate after Phase H.
> Rule zero: **never delete or overwrite existing artifacts.** All v3 work goes to `experiments/v3/` and `data/processed/v3/`.

---

## 0. Global rules (read before every phase)

1. Always use the venv: `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (Linux).
2. One git branch per phase: `git checkout -b v3/phase-<letter>`; merge only when the phase's **Acceptance** passes.
3. Every run goes through `ExperimentLogger(run_name, cfg, seed)` and writes
   `experiments/v3/<run_name>/seed<k>/{config.yaml, metrics.json, val_logits.npy, test_logits.npy, preds.csv, git_sha.txt}`.
4. **Test-set lock:** no script except `scripts/final_test.py` (Phase J) can read test labels.
5. Seeds: `[13, 42, 87, 1234, 2026]`. Report mean ± std and a 95% bootstrap CI.
6. Model selection metric: validation macro-F1. Ties go to fewer parameters.
7. Every numeric claim in a document must come from a CSV produced by `scripts/build_reports.py`.
8. If a step fails twice, stop, write `experiments/v3/BLOCKERS.md` (error, command, hypothesis), and move to the next independent phase.
9. Commit message format: `v3-<phase>: <what> [tests: N pass]`.
10. **No outcome targets.** Acceptance criteria check process only (files exist, tests pass, runs logged), never result values. Do not tune until a desired ordering appears.
11. **One factor per comparison.** Alpha, temperature, learning-rate grid, epochs, max length and batch size are fixed across all arms of a comparison.
12. **Equal tuning budget.** Every arm, baselines included, gets 12 random configs x 2 seeds, selected on validation.
13. Steps marked **HUMAN** are not for the agent. Prepare the input files, write `experiments/v3/HUMAN_TODO.md`, and continue with independent phases.
14. Out of the core path: DAPT, XLM-R-large, pseudo-labels, back-translation, teacher ensembles. Run them only as labelled appendix arms if compute remains.

### Phase dependency graph

```
A -> B -> C -+-> E -> F -> G -> H -> I -> J -+-> K
             |                               +-> L
             +-> D (HUMAN, parallel) ------> J   +-> M
```

---

## Phase A — Guardrails

**Purpose:** stop test leakage through code and stop inconsistent numbers reaching the paper.

1. Run v2 Phase A (test guard, `build_reports.py`, docs-consistency test, v1 archive with banner), writing to `experiments/v3/`.
2. Add `tests/test_metric_consistency.py`: for every `metrics.json`, recompute accuracy, per-class precision/recall/F1 and macro averages from the stored confusion matrix; fail if any differs by more than 1e-4.
3. Make `metrics.json` always store the confusion matrix and the true class counts taken from the labels.

**Acceptance:** `pytest -q` passes; the guard raises when test data is read without the env var.

---

## Phase B — Pre-registration and related work

**Purpose:** fix the claim before seeing results.

1. Write `PREREG.md` containing:
   - H1: on a fixed architecture, pretrained initialisation changes macro-F1 more than the choice of training objective.
   - H2: the benefit of distillation over cross-entropy shrinks as initialisation improves and grows as labelled data shrinks.
   - Primary metric: macro-F1. Non-inferiority margin: 2.0 points macro-F1.
   - The full list of planned comparisons and the correction method (Holm).
2. Write `RELATED_WORK.md` covering at least: pretrained compact students (Turc et al. 2019 — verify the reference), DistilBERT, TinyBERT, MiniLM, BanglaBERT, the BanglaBait dataset paper, BaitBuster-Bangla. For each: what it claims and how this work differs.
3. Fix citations in all docs:
   - BanglaBERT: Bhattacharjee et al., Findings of NAACL 2022.
   - BaitBuster-Bangla: Data in Brief (2024).
   - Add the BanglaBait dataset paper.
4. Remove "Submission Ready", emojis and decorative bolding from the paper draft.

**Acceptance:** `PREREG.md` committed and tagged `v3-prereg` before any v3 training run exists.

---

## Phase C — Data

1. Run v2 Phase B (MinHash dedup, grouped stratified splits) and save to `data/processed/v3/splits_v3.json`.
2. Also keep the original BanglaBait split and its leak-free subset as a secondary evaluation set, so numbers stay comparable with published work.
3. Create stratified training subsets at 10%, 25%, 50% and 100% with fixed indices: `data/processed/v3/train_frac_<p>.json`. Smaller subsets must be nested inside larger ones.
4. Write `leakage_report.md`: counts per split, class balance, groups removed.

**Acceptance:** cross-split near-duplicates = 0; all index files committed and never regenerated.

---

## Phase D — Human relabeling (HUMAN, runs in parallel)

**Purpose:** replace the teacher-based "noise ceiling" with real annotation evidence.

1. Agent: train one BanglaBERT-Base with 5-fold out-of-fold predictions on train; flag rows where it disagrees with the label with confidence above 0.9.
2. Agent: export `relabel_batch.csv` with the full test set, 300 flagged training rows and 300 random unflagged training rows; shuffled, no labels, no source column. Keep the key in `relabel_key.csv`.
3. Agent: write `ANNOTATION_GUIDE.md` from the BanglaBait labelling definition.
4. HUMAN: two native speakers label independently; a third resolves disagreements.
5. Agent: compute Cohen's kappa, the label error rate in flagged vs random rows, and the error rate of the test set. Write `relabel_report.md` and `test_labels_relabelled.csv`.

**Acceptance:** `relabel_report.md` exists. All statements about label noise cite only these numbers. Relabelled test labels stay behind the test guard.

---

## Phase E — Baselines and teachers

1. Train, each with the Rule 12 budget and five seeds:
   - TF-IDF + logistic regression
   - fastText
   - BiLSTM
   - multilingual DistilBERT
   - a multilingual MiniLM
   - BanglaBERT-Small-12L, cross-entropy only
   - BanglaBERT-Base (teacher)
2. Zero-shot LLM baseline: commit the prompt to `prompts/zero_shot.txt`; run on validation, and on test only inside Phase J.
3. Save 5-fold out-of-fold teacher logits on train for use as distillation targets.
4. Report each teacher individually. Any ensemble is a separate arm named `ensemble_*`.

**Acceptance:** all runs logged for five seeds; validation table rendered by `build_reports.py`.

---

## Phase F — Core grid on clickbait

All students in the main grid share one architecture: 4 layers, hidden 256, factorised embeddings, 7.42M parameters.

1. Initialisation arms:
   - `rand`: random
   - `strided`: layers `[2,5,8,11]` of BanglaBERT-Small
   - `first4`: layers `[0,1,2,3]`
   - `last4`: layers `[8,9,10,11]`
2. Objective arms:
   - `ce`: cross-entropy
   - `kd`: logit distillation, T and alpha fixed for all arms
   - `kd_hid`: logit + hidden-state distillation
3. Data arms: 10%, 25%, 50%, 100%.
4. Run the full cross: 4 x 3 x 4 x 5 seeds = 240 runs. Run names: `F_<init>_<obj>_<frac>`.
5. Depth study at 100% data: depths 4, 6, 12 x {`ce`, `kd`} x 5 seeds, strided or full init.
6. Appendix only: the PCA-projected student, labelled as not parameter-matched.

**Acceptance:** 240 + 30 runs logged; no run reads test labels; grid CSV built.

---

## Phase G — Mechanism

1. For every Phase F cell record: student-teacher prediction agreement, expected calibration error, validation loss curve, epochs to best validation score.
2. Fit a mixed-effects model: `f1 ~ init * objective * log(frac) + (1 | seed)`. Report effect sizes and intervals for each factor and interaction.
3. Plot F1 against data fraction, one line per objective, one panel per initialisation.

**Acceptance:** `mechanism_report.md` and figures generated from CSVs.

---

## Phase H — Generalisation

**Purpose:** this phase decides the venue.

1. Agent: write `TASK_CANDIDATES.md` listing 2–3 further Bangla tasks (for example sentiment and NLI from the BanglaBERT benchmark — verify availability and licence) with sizes and splits.
2. Agent: write `LANG_CANDIDATES.md` listing low-resource languages that have both a base and a small pretrained encoder plus a labelled classification task.
3. HUMAN: approve the task and language list.
4. For each approved task and language, repeat Phase C (splits, fractions), the Phase E teacher, and the Phase F main grid. A reduced grid is allowed (`rand`, `strided` x `ce`, `kd` x 4 fractions x 5 seeds) if compute is short; record the reduction in `PREREG.md` as an amendment before running.
5. Rerun the Phase G mixed-effects model with task and language as random effects.

**Acceptance:** all runs logged. Results are reported whether or not they support H1 and H2.

---

## Phase I — Lock

1. Run v2 Phase H: choose final configs on validation only, write `LOCKED_CONFIG.yaml`.
2. Tag the commit `v3-locked`. No training or config change after this tag.

**Acceptance:** tag exists; `git diff v3-locked` is empty for `src/` and `configs/` at Phase J.

---

## Phase J — One-time test

1. Run `scripts/final_test.py` once with `TBB_ALLOW_TEST=1` for every locked model and seed.
2. Statistics, replacing the seed-42 McNemar test:
   - Paired bootstrap over test items and seeds, 10,000 resamples, for each planned comparison; Holm-corrected.
   - Non-inferiority test of the student against BanglaBERT-Base and against Small-12L, using the 2.0-point margin from `PREREG.md`.
3. Report on three test sets: v3 split, original-split leak-free subset, and the Phase D relabelled test labels.
4. Selective prediction: report coverage-accuracy curves; any figure above 95% must be labelled as selective accuracy with its coverage.

**Acceptance:** `final_test_log.txt` shows exactly one invocation; all tables auto-generated.

---

## Phase K — Cross-genre evaluation

1. Run v2 Phase J on BaitBuster-Bangla, described as cross-genre (YouTube video titles), not cross-domain news.
2. Store and publish confusion matrices; compute class prevalence from the labels, not from predictions.
3. Add a recalibration arm: retune the decision threshold on a 1% labelled sample of the target set, evaluate on the remaining 99%.
4. Report absolute macro-F1 with 95% intervals for every model. Do not report "relative degradation" or retention above 100% without an interval.

**Acceptance:** `test_metric_consistency.py` passes on all cross-genre metrics.

---

## Phase L — Efficiency

1. Export every model (teacher, Small-12L, all student depths) to ONNX in FP32 and dynamic INT8.
2. Benchmark with one harness: batch 1, fixed max length, 200 warm-up + 1,000 timed runs, threads = 1 and 4. Record CPU model, OS, ONNX Runtime version, p50 and p95.
3. Report INT8 accuracy for every model across five seeds, not one.
4. Browser: ONNX Runtime Web (WASM) numbers per v2 Phase K.
5. HUMAN: run the same harness on one named Android phone.
6. Cascade: recompute latency with simulated 50 ms and 150 ms network round-trips for escalated items; compare cascade F1 with the teacher using the paired bootstrap.

**Acceptance:** `efficiency.csv` contains every model x precision x thread row.

---

## Phase M — Explainability

1. Run v2 Phase L (300 examples, faithfulness metrics).
2. Report SHAP-LIME rank correlation, sign agreement and top-k overlap with intervals.
3. If sign agreement is below 60%, state that the methods disagree; do not claim lexical grounding.

**Acceptance:** `xai_report.md` built from CSVs.

---

## Reporting rules

- Headline numbers are five-seed means with 95% intervals, differences in percentage points.
- State compression against both BanglaBERT-Base and BanglaBERT-Small.
- Never describe a non-significant difference as "no loss"; use the non-inferiority result.
- Report negative results, including the clean-label retrain and any arm that contradicts H1 or H2.
- Required sections: related work, limitations, ethics (false-positive harm to publishers), reproducibility statement.
- Release code, split files, locked config and the relabelled test labels.

## Decision gate (after Phase H)

- Effect replicates across tasks and languages: write the general paper; target NeurIPS or ACL/EMNLP main track.
- Effect holds only for clickbait: write the single-task study; target Findings or LREC-COLING.
- Either way, the relabelled test set can be submitted separately to a datasets track.

## Review issue -> phase map

| Review issue | Fixed in |
|---|---|
| Thin novelty | B, F, G, H |
| Distillation benefit unproven | F, G |
| Misleading significance claim | B, J |
| Confounded ablation | Rule 11, F |
| Single-run noise | Rule 5, J |
| Inconsistent cross-domain table | A, K |
| Overclaimed cross-domain result | K |
| Unfair latency comparison | L |
| Explainability contradicts claim | M |
| Cascade and noise-ceiling claims | D, L |
| Missing baselines | E |
| Wrong or missing citations | B |
