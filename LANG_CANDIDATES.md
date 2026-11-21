# Candidate Languages for Cross-Lingual Generalisation (Phase H)

This document pre-selects low-resource and medium-resource languages with paired **Base** (12-layer) and **Small** (compact pre-trained) encoders, along with standard public classification benchmarks, to test the cross-lingual validity of H1 and H2.

---

## Candidate Language 1: Hindi (hi)
- **Typology:** Indo-Aryan, SOV, morphologically rich, Devanagari script (closely related to Bengali).
- **Paired Encoders:**
  - **Base:** `ai4bharat/indic-bert` (12L, 768H) or `l3cube-pune/hindi-bert`
  - **Small Compact:** `google/electra-small-discriminator` fine-tuned / `ai4bharat/indic-bert` compact sub-networks.
- **Benchmark Task:** BBC Hindi News Classification / Sentiment Analysis.
- **Justification:** High linguistic affinity with Bengali while possessing mature community benchmarks and verified vocabulary structures.

---

## Candidate Language 2: Swahili (sw)
- **Typology:** Niger-Congo, Bantu, agglutinative, Latin script.
- **Paired Encoders:**
  - **Base:** `castorini/afro-xlmr-base` or `Davlan/afro-xlmr-base`
  - **Small Compact:** `Davlan/afro-xlmr-small` / compact sub-networks.
- **Benchmark Task:** MasakhaNER / Swahili News Classification (Kaggle / Masakhane).
- **Justification:** Distinct language family and agglutinative morphology, testing whether warm-start initialization advantages hold in non-Indo-European low-resource linguistic environments.

---

## Candidate Language 3: Tamil (ta)
- **Typology:** Dravidian, agglutinative, Tamil script.
- **Paired Encoders:**
  - **Base:** `ai4bharat/indic-bert` / `l3cube-pune/tamil-bert`
  - **Small Compact:** IndicBERT sub-networks.
- **Benchmark Task:** Tamil Sentiment Analysis (DravidianLangTech shared task).
- **Justification:** Distinct Dravidian grammatical structure with significant dialectal variation and suffix-stacking morphology.
