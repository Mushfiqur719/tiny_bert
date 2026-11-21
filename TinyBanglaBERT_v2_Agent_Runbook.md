# TinyBanglaBERT v2 — Agent Runbook

> Audience: any coding agent (or human) with shell access to `tiny_bert/`.
> Goal: leak-free, multi-seed, reproducible results; student target **0.84–0.87 macro-F1 / 86–89% Acc**; 95%+ reported only as selective accuracy.
> Rule zero: **never delete or overwrite existing artifacts.** All v2 work goes to `experiments/v2/` and `data/processed/v2/`.

---

## 0. Global rules (read before every phase)

1. Always use the venv: `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (Linux).
2. One git branch per phase: `git checkout -b v2/phase-<letter>`; merge only when the phase's **Acceptance** passes.
3. Every run goes through `ExperimentLogger(run_name, cfg, seed)` and writes:
   `experiments/v2/<run_name>/seed<k>/{config.yaml, metrics.json, val_logits.npy, test_logits.npy, preds.csv, git_sha.txt}`.
4. **Test-set lock:** no script except `scripts/final_test.py` (Phase I) can read test labels. Enforce it with a guard (Phase A).
5. Seeds: `[13, 42, 87, 1234, 2026]`. Report mean ± std and a 95% bootstrap CI.
6. Model selection metric: **val macro-F1**. Ties go to fewer params.
7. Every numeric claim in a document must come from a CSV produced by `scripts/build_reports.py`.
8. If a step fails twice, stop, write `experiments/v2/BLOCKERS.md` (error, command, hypothesis), and move to the next independent phase.
9. Commit message format: `v2-<phase>: <what> [tests: N pass]`.

### Phase dependency graph
```
A ─┬─> B ──> C ──> D ──> E ──> F ──> G ──> H ──> I ──> J
   │                                       └──> K
   └───────────────────────────────────────────> L (needs I)
```

### Compute budget (RTX 5050 8 GB, approx.)
| Phase | GPU time |
|---|---|
| D (DAPT base) | 3–6 h |
| E (teachers, 4 types × 5 seeds) | 8–14 h (large models: fp16, bs 8, grad-accum 4) |
| G (students, 3 depths × 5 seeds × 2 variants) | 3–5 h |
| Everything else | < 2 h |

If VRAM is insufficient for `-large` models, use gradient checkpointing (`model.gradient_checkpointing_enable()`), or drop XLM-R-large and note it in the Limitations.

---

## Phase A — Consistency & guardrails

**Purpose:** remove contradictory numbers and stop test leakage through code.

Steps:
1. Install extras:
   `pip install cleanlab datasketch optuna onnxruntime evaluate sentencepiece sacremoses`
   then `pip freeze > requirements-lock.txt`.
2. Create `src/utils/test_guard.py`:
```python
import os
class TestSetLocked(RuntimeError): pass
def assert_test_access():
    if os.environ.get("TBB_ALLOW_TEST") != "1":
        raise TestSetLocked("Test labels are locked. Only scripts/final_test.py may set TBB_ALLOW_TEST=1.")
```
   Call `assert_test_access()` inside `apply_splits()` whenever `split == "test"`.
3. Create `scripts/build_reports.py`. It reads `experiments/v2/**/metrics.json`, aggregates by group into `experiments/v2/results_master.csv` (columns: group, n_seeds, params, acc_mean, acc_std, f1_mean, f1_std, f1_ci_lo, f1_ci_hi, cb_f1_mean, latency_ms), and renders Markdown tables into the docs between markers `<!-- AUTO:table_name -->` … `<!-- /AUTO -->`.
4. Create `tests/test_docs_consistency.py`. It parses every number inside AUTO blocks in README.md, FINAL_RESULTS.md and IEEE_PAPER_REPORT.md and asserts it equals the CSV value (tolerance 1e-4).
5. Move the old v1 docs to `experiments/v1_archive/` (copy, don't delete) and add a banner at the top of each: "v1 – superseded, contains known inconsistencies".

**Acceptance:** `pytest -q` passes; `test_guard` raises when test data is accessed without the env var.

---

## Phase B — Leak-free splits v2

1. Create `src/data/dedup.py`:
```python
import re, unicodedata
from datasketch import MinHash, MinHashLSH
def norm(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()
def shingles(t, k=3):
    t = norm(t); return {t[i:i+k] for i in range(max(1, len(t)-k+1))}
def near_dup_groups(texts, thr=0.9, perm=128):
    lsh = MinHashLSH(threshold=thr, num_perm=perm); mh = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=perm)
        for s in shingles(t): m.update(s.encode())
        lsh.insert(str(i), m); mh.append(m)
    parent = list(range(len(texts)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, m in enumerate(mh):
        for j in lsh.query(m):
            a, b = find(i), find(int(j))
            if a != b: parent[a] = b
    return [find(i) for i in range(len(texts))]
```
2. Create `scripts/make_splits_v2.py`:
   - Load the labeled data and compute dedup groups.
   - Drop label-conflicting groups (same group, different labels) to `data/processed/v2/conflicts.csv`.
   - Use `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`: fold 0 = test, fold 1 = val, the rest = train.
   - Save `data/processed/v2/splits_v2.json` and `leakage_report.md`. The report gives counts per split, class balance, the number of groups removed, and cross-split near-duplicates (must be 0).
3. Tests in `tests/test_dedup.py`: identical strings group together; punctuation-only differences group together; no group spans two splits.

**Acceptance:** cross-split near-duplicates = 0; class ratio in each split is within ±1 pp of the overall ratio; `splits_v2.json` is committed. Never regenerate it after this point.

---

## Phase C — Full data (body text + unlabeled)

1. `src/data/load_full.py`: load the original BanglaBait release (GitHub) with `title`, `content`, `label`, plus the unlabeled set (~65k). If the body is missing for some rows, fill it with an empty string and record `has_body`.
2. Add config keys to `configs/base.yaml`:
```yaml
data:
  text_mode: title        # title | title_body
  body_max_tokens: 200
  max_len: 64             # 256 when title_body
  unlabeled_path: data/raw/banglabait_unlabeled.jsonl
```
3. Tokenize `title_body` as a sentence pair: `tokenizer(title, body, truncation="only_second", max_length=256)`.
4. Dedup the unlabeled set against **val + test** with the Phase B MinHash (threshold 0.9). Write `unlabeled_clean.jsonl` and the number removed.

**Acceptance:** the unlabeled set has no near-duplicates of val/test; tests cover both text modes.

---

## Phase D — Domain-adaptive pretraining (DAPT)

1. `scripts/dapt.py`: MLM continuation of `csebuetnlp/banglabert` on unlabeled titles+bodies + **train** titles only.
   - Electra checkpoint: use the generator (`csebuetnlp/banglabert_generator`) for MLM, **or** load the discriminator with `AutoModelForMaskedLM` only if a head exists. Fallback: run DAPT on `xlm-roberta-base`/BanglaBERT-small generator and record which one was used.
   - Hyperparameters: mlm_prob 0.15, lr 5e-5, warmup 6%, 2 epochs, bs 32 (accumulate as needed), max_len 256, fp16.
2. Output: `experiments/v2/dapt_banglabert/`. Log held-out perplexity (1% of the unlabeled set).

**Acceptance:** held-out perplexity drops by ≥10% vs. the starting checkpoint. If it doesn't, record it and continue (Phase E then simply skips the DAPT teacher).

---

## Phase E — Teacher ensemble

Configs `configs/v2/teacher_{base,large,xlmr,dapt}.yaml`:
| Teacher | Checkpoint | lr grid | bs×accum | epochs |
|---|---|---|---|---|
| base | csebuetnlp/banglabert | 1e-5,2e-5,3e-5 | 32×1 | 5 |
| large | csebuetnlp/banglabert_large | 1e-5,2e-5 | 8×4 | 5 |
| xlmr | xlm-roberta-large | 8e-6,1e-5 | 8×4 | 5 |
| dapt | experiments/v2/dapt_banglabert | 1e-5,2e-5,3e-5 | 32×1 | 5 |

Common settings: LLRD 0.9, warmup 10%, weight decay 0.01, label smoothing 0.0, R-Drop α=1.0 (or FGM ε=1.0), early stopping on val macro-F1 (patience 2). Train both `title` and `title_body` modes.

Steps:
1. `scripts/train_multiseed.py --config <cfg> --seeds 13 42 87 1234 2026 --lr-grid ...` picks the best lr by mean val F1 across 3 seeds, then runs all 5 seeds at that lr.
2. **5-fold OOF on train** (best config per teacher, seed 42) gives out-of-fold logits `train_oof_logits.npy`. These are needed for cleanlab and leak-free KD targets.
3. Score the unlabeled set: `unlabeled_logits.npy` per teacher.
4. Ensemble: average **calibrated** probabilities (temperature fit on val) across teachers and seeds. Choose the teacher subset greedily by val F1 (Caruana selection). Save `experiments/v2/teacher_ensemble/{val,train_oof,unlabeled}_probs.npy`.

**Acceptance:** the ensemble's val macro-F1 ≥ the best single teacher's; all logits are saved.

---

## Phase F — Label audit (train only)

1. `scripts/label_audit.py`: `cleanlab.filter.find_label_issues(labels, pred_probs=train_oof_probs, filter_by="prune_by_noise_rate")`.
2. Write `label_issues.csv` (index, title, given, predicted, confidence). Manually spot-check 50 rows and put the precision estimate in `label_audit.md`.
3. Produce two train variants: `train_full` and `train_clean` (flagged rows removed, or relabeled if confidence > 0.95). **Never** touch val/test labels.

**Acceptance:** report lists flagged %, spot-check precision, and val F1 for the base teacher trained on clean vs. full data.

---

## Phase G — Student distillation

### G1. Loss (`src/models/kd_loss.py`)
```python
import torch, torch.nn as nn, torch.nn.functional as F
class KDLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.5, beta_hid=1.0, beta_att=1.0, s_dim=256, t_dim=768, layer_map=None):
        super().__init__(); self.T, self.a, self.bh, self.ba = T, alpha, beta_hid, beta_att
        self.proj = nn.Linear(s_dim, t_dim); self.layer_map = layer_map or {}
    def forward(self, s_logits, labels, t_probs=None, s_hid=None, t_hid=None, s_att=None, t_att=None):
        ce = F.cross_entropy(s_logits, labels)
        loss = (1 - self.a) * ce
        if t_probs is not None:
            t = torch.softmax(torch.log(t_probs.clamp_min(1e-8)) / self.T, -1)
            loss = loss + self.a * self.T**2 * F.kl_div(F.log_softmax(s_logits / self.T, -1), t, reduction="batchmean")
        if s_hid is not None and t_hid is not None and self.bh:
            loss = loss + self.bh * sum(F.mse_loss(self.proj(s_hid[i]), t_hid[j]) for i, j in self.layer_map.items()) / max(1, len(self.layer_map))
        if s_att is not None and t_att is not None and self.ba:
            loss = loss + self.ba * sum(F.kl_div(torch.log(s_att[i].mean(1).clamp_min(1e-8)), t_att[j].mean(1), reduction="batchmean") for i, j in self.layer_map.items()) / max(1, len(self.layer_map))
        return loss
```
Attention is head-averaged, so student and teacher head counts can differ. Layer map for 4L→12L: `{1:3, 2:6, 3:9, 4:12}` (index 0 = embeddings). Intermediate KD needs a **single** teacher with hidden states. Use the best single teacher (usually `dapt` or `large`, with matching `t_dim`); use ensemble probabilities for the logit term.

### G2. Training data for KD
- Labeled: `train_clean` (or `train_full`, whichever won in F). Teacher targets are the **OOF** ensemble probs, not in-sample probs.
- Pseudo-labeled: unlabeled rows with ensemble max-prob ≥ 0.9. Balance classes to the train ratio, cap at 3× labeled size, store as `pseudo_v1.jsonl`. Labels = argmax, soft targets = probs.
- Augmentation (optional, G4): back-translated train titles, soft-labeled by the ensemble.

### G3. Runs (`configs/v2/student_*.yaml`)
| Run | Init | Depth | Data | Loss |
|---|---|---|---|---|
| S0 | BanglaBERT-small strided | 4L | labeled | CE |
| S1 | same | 4L | labeled | logit KD |
| S2 | same | 4L | labeled | logit + hidden + attn KD |
| S3 | same | 4L | labeled + pseudo | S2 loss |
| S4 | same | 6L | labeled + pseudo | S2 loss |
| S5 | full small | 12L | labeled + pseudo | S2 loss |
| S6 | random | 4L | labeled + pseudo | S2 loss (control) |
| S7 | teacher-PCA | 4L | labeled + pseudo | S2 loss (control) |

All runs: 5 seeds, lr 1e-4 (students), 10 epochs, early stopping on val F1, max_len per text mode. Run each in `title`, and the best one also in `title_body`.

### G4. HPO (val only)
`scripts/hpo_kd.py` uses Optuna (TPE, 40 trials, seed 42 only) on S3: T∈{1,2,4,8}, α∈[0.3,0.9], β_hid∈{0,0.5,1}, β_att∈{0,0.5,1}, lr∈[3e-5,3e-4], pseudo-threshold∈{0.85,0.9,0.95}. Rerun the best config with 5 seeds.

**Acceptance:** S3 mean val F1 > S1 > S0 (this is the ablation story); otherwise report honestly. All logits saved.

---

## Phase H — Lock configuration

1. Fit a temperature on val for each final model. Then choose the decision threshold τ that maximizes val macro-F1 (grid 0.2–0.8, step 0.01).
2. Write `configs/v2/final_config.lock.yaml`: for each model, checkpoint path per seed, T_cal, τ, text_mode, git SHA, data hash (sha256 of `splits_v2.json`).
3. Tag the commit `v2-locked`. **Do not change anything after this tag** except reporting code.

**Acceptance:** the lock file exists and the tag is pushed.

---

## Phase I — One-time test evaluation

`scripts/final_test.py` (only this file sets `TBB_ALLOW_TEST=1`):
1. For each locked model and seed: test metrics (Acc, macro-F1, P, R, per-class F1, confusion matrix), with and without calibrated τ.
2. Stats:
   - Bootstrap 95% CI (10,000 resamples) per model.
   - McNemar exact test (seed 42) for planned pairs: Teacher-ens vs S3, S0 vs S1, S1 vs S2, S2 vs S3, S3 vs S4, S3 vs S5, S3 vs S6, S3 vs S7. Holm-correct across this family.
   - Paired seed-matched bootstrap on ΔF1 for S3 vs S1 and S3 vs S5.
3. Selective prediction: sort by max-prob and report accuracy at coverage {100, 90, 80, 70, 60, 50}%, the AURC, and the coverage at which accuracy ≥ 95%. Plot `fig_selective.png`.
4. Deferral system: the student handles confident cases; cases below the threshold go to the teacher ensemble. Report system accuracy and the % deferred.
5. Write `experiments/v2/final_test/test_results.csv`, then run `build_reports.py`.

**Acceptance:** test files are written exactly once (the script refuses to run if `test_results.csv` exists, unless `--force-reason "<text>"` is given, which is logged).

---

## Phase J — Cross-domain (BaitBuster-Bangla)

1. Use only the human-annotated ~10k subset. NFC-normalize and dedup against all BanglaBait splits (expected ~0 overlap; record it).
2. Zero-shot: locked models, title mode, τ from the BanglaBait val set (no target tuning).
3. Few-shot: fine-tune S3 on 1% and 5% of the target (stratified, seed-repeated ×5) and evaluate on the remaining 90% (fixed held-out).
4. Report per-class F1 and CIs. Don't claim "distillation regularizes" unless the CI of Δdrop excludes 0.

---

## Phase K — Efficiency

1. Export: `torch.onnx.export(..., opset_version=17, dynamic_axes={"input_ids":{0:"b",1:"s"}, ...})` then `onnxruntime.quantization.quantize_dynamic(..., weight_type=QInt8)`.
2. Verify parity: |ΔF1| between PyTorch and ONNX FP32 on val ≤ 0.001; report INT8 F1 on val and test.
3. Benchmark protocol: fixed CPU (record model and frequency), `torch.set_num_threads(1)` and 4-thread variants, seq_len 64 (and 256 for title_body), 50 warmup + 500 timed runs. Report mean, p50, p95, and throughput, plus peak RSS via `psutil`.
4. Browser: `onnxruntime-web` WASM (SIMD on), a 200-run Node/Chrome harness in `bench/web/`. Report p50/p95.
5. Output `efficiency_v2.csv`.

---

## Phase L — Explainability

1. Sample 300 test headlines stratified by (true label × correct/incorrect).
2. SHAP (Partition explainer with a text masker, max_evals 500) and LIME (num_samples 1000), both at the word level with the same tokenization.
3. Metrics: Top-3 and Top-5 Jaccard, sign agreement, Kendall τ on shared tokens, plus ERASER comprehensiveness and sufficiency at k ∈ {1,3,5}, with a random-token baseline.
4. Human plausibility: 2 native annotators rate 50 explanations on a 1–5 scale; report Cohen κ.
5. Error taxonomy: manually tag 100 errors (sensational, question-hook, number-bait, named-entity, sarcasm, label-noise).

---

## Final deliverables checklist
- [ ] `experiments/v2/results_master.csv` (single source of truth)
- [ ] `final_config.lock.yaml` + tag `v2-locked`
- [ ] `test_results.csv`, `significance_v2.csv`, `selective_v2.csv`, `crossdomain_v2.csv`, `efficiency_v2.csv`, `xai_v2.csv`
- [ ] Figures regenerated from CSVs (`scripts/make_figures.py`), 300 DPI
- [ ] README / FINAL_RESULTS / IEEE paper regenerated via AUTO blocks; consistency test passes
- [ ] `REPRODUCE.md` updated with v2 commands; `requirements-lock.txt`
- [ ] `BLOCKERS.md` empty or each item resolved/acknowledged in Limitations

## Reporting rules for the paper
- Headline numbers = 5-seed mean ± std with a 95% CI, on the leak-free test set, threshold from val.
- Never present "cal" or best-seed numbers as main results.
- 95%+ accuracy is stated only as "X% accuracy at Y% coverage."
- Keep the v1 numbers only in an appendix titled "Preliminary results (superseded)."
