# Human Annotation Tasks (Phase D)

**Status:** Awaiting Human Native-Speaker Annotation  
**Batch File:** `data/processed/v3/relabel_batch.csv` (2104 total items)  
**Guidelines:** `ANNOTATION_GUIDE.md`  

---

## Instructions for Investigators / Native Speakers:
1. Two native Bengali speakers must independently annotate each headline in `relabel_batch.csv`:
   - Enter `0` for **Not-Clickbait** (informative, straightforward reportage).
   - Enter `1` for **Clickbait** (curiosity gap, exaggeration, ambiguity, sensationalism).
   - Fill into columns `annotator_1_label` and `annotator_2_label`.
2. A third senior annotator resolves all disagreements into `resolved_label`.
3. Once completed, save the CSV and notify the agent or run:
   ```bash
   python -m scripts.evaluate_relabel
   ```
   The script will compute:
   - Inter-annotator agreement (Cohen's Kappa $\kappa$).
   - True label noise rate in teacher-flagged items vs. random items.
   - Clean, human-verified test error rate, generating `experiments/v3/relabel_report.md` and `data/processed/v3/test_labels_relabelled.csv`.
