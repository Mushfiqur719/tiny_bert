# TinyBanglaBERT Explainability: SHAP vs. LIME Token Attributions

*Generated: 2026-08-30 15:48 UTC*
*Sample Size: 15 diverse test headlines (covering True Positives, True Negatives, False Positives, and False Negatives)*

### Highlighting Legend
- <span style="background-color: rgba(231, 76, 60, 0.6); padding: 2px 5px; font-weight: bold;">Red Highlights</span> : Tokens with **positive attribution** (push model toward **Clickbait**).
- <span style="background-color: rgba(52, 152, 219, 0.6); padding: 2px 5px;">Blue Highlights</span> : Tokens with **negative attribution** (push model toward **Not-Clickbait** / Informative).

---

### Headline #1: ✅ Correct
**Headline Text:** *‘নায়িকারা বুঝলাম, নায়করাও কি বিছানায় শুয়ে কাজ পায়?’*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `92.17%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `88%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.28); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0068">‘নায়িকারা</span> <span style="background-color: rgba(231, 76, 60, 0.58); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0234">বুঝলাম,</span> <span style="background-color: rgba(52, 152, 219, 0.18); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0015">নায়করাও</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0379">কি</span> <span style="background-color: rgba(231, 76, 60, 0.50); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0190">বিছানায়</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0077">শুয়ে</span> <span style="background-color: rgba(231, 76, 60, 0.43); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0149">কাজ</span> <span style="background-color: rgba(231, 76, 60, 0.63); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0262">পায়?’</span>
*Top Clickbait Tokens (SHAP):* `কি`, `পায়?’`, `বুঝলাম,`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1133">পায়?’</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0834">বিছানায়</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0825">কি</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0824">‘নায়িকারা</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0761">বুঝলাম,</span> <span style="background-color: rgba(231, 76, 60, 0.57); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0679">কাজ</span> <span style="background-color: rgba(231, 76, 60, 0.52); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0592">নায়করাও</span> <span style="background-color: rgba(231, 76, 60, 0.31); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0265">শুয়ে</span>
*Top Clickbait Tokens (LIME):* `পায়?’`, `বিছানায়`, `কি`

---

### Headline #2: ✅ Correct
**Headline Text:** *জা’নিনা কার স’ন্তান আমার পে’টে, স্বা’মী নাকি দে’বরের…*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `86.82%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `62%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.50); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0257">জা’নিনা</span> <span style="background-color: rgba(231, 76, 60, 0.37); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0160">কার</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0042">স’ন্তান</span> <span style="background-color: rgba(231, 76, 60, 0.45); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0223">আমার</span> <span style="background-color: rgba(231, 76, 60, 0.36); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0153">পে’টে,</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0106">স্বা’মী</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0514">নাকি</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0050">দে’বরের…</span>
*Top Clickbait Tokens (SHAP):* `নাকি`, `আমার`, `কার`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0731">স’ন্তান</span> <span style="background-color: rgba(231, 76, 60, 0.75); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0624">নাকি</span> <span style="background-color: rgba(231, 76, 60, 0.71); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0581">পে’টে,</span> <span style="background-color: rgba(231, 76, 60, 0.64); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0514">কার</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0483">আমার</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0373">স্বা’মী</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0371">দে’বরের…</span> <span style="background-color: rgba(231, 76, 60, 0.47); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0335">জা’নিনা</span>
*Top Clickbait Tokens (LIME):* `স’ন্তান`, `নাকি`, `পে’টে,`

---

### Headline #3: ✅ Correct
**Headline Text:** *‘তারা দিনে ঘুমাতেন, রাতে উচ্চবিত্ত পরিবারের সন্তানদের বাসায় ডেকে আনতেন’*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `50.72%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `40%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.44); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0531">‘তারা</span> <span style="background-color: rgba(231, 76, 60, 0.43); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0523">দিনে</span> <span style="background-color: rgba(52, 152, 219, 0.30); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0280">ঘুমাতেন,</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0249">রাতে</span> <span style="background-color: rgba(52, 152, 219, 0.35); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0374">উচ্চবিত্ত</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0089">পরিবারের</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0633">সন্তানদের</span> <span style="background-color: rgba(52, 152, 219, 0.48); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0617">বাসায়</span> <span style="background-color: rgba(52, 152, 219, 0.69); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0999">ডেকে</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1291">আনতেন’</span>
*Top Clickbait Tokens (SHAP):* `‘তারা`, `দিনে`, `রাতে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0055">রাতে</span> <span style="background-color: rgba(231, 76, 60, 0.77); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0049">‘তারা</span> <span style="background-color: rgba(231, 76, 60, 0.70); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0043">সন্তানদের</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0041">দিনে</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0041">পরিবারের</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0036">আনতেন’</span> <span style="background-color: rgba(231, 76, 60, 0.60); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0035">উচ্চবিত্ত</span> <span style="background-color: rgba(231, 76, 60, 0.56); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0032">ঘুমাতেন,</span> <span style="background-color: rgba(231, 76, 60, 0.53); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0030">ডেকে</span> <span style="background-color: rgba(231, 76, 60, 0.35); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0016">বাসায়</span>
*Top Clickbait Tokens (LIME):* `রাতে`, `‘তারা`, `সন্তানদের`

---

### Headline #4: ✅ Correct
**Headline Text:** *প্রেম করছেন শাকিব-শ্রাবন্তী…….!*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `89.85%`)
- **Method Agreement (Top-3 Jaccard):** **`1.00`** | **Sign Agreement:** `100%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.27); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0111">প্রেম</span> <span style="background-color: rgba(231, 76, 60, 0.56); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0371">করছেন</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0628">শাকিব-শ্রাবন্তী…….!</span>
*Top Clickbait Tokens (SHAP):* `শাকিব-শ্রাবন্তী…….!`, `করছেন`, `প্রেম`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0555">শাকিব-শ্রাবন্তী…….!</span> <span style="background-color: rgba(231, 76, 60, 0.65); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0399">করছেন</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0366">প্রেম</span>
*Top Clickbait Tokens (LIME):* `শাকিব-শ্রাবন্তী…….!`, `করছেন`, `প্রেম`

---

### Headline #5: ✅ Correct
**Headline Text:** *আমি কী জবাব দেব স্বামীকে, আমি তো তোমার সন্তানকে বাঁ’চাতে পারলাম না*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `87.92%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `56%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.35); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0117">আমি</span> <span style="background-color: rgba(231, 76, 60, 0.80); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0374">কী</span> <span style="background-color: rgba(52, 152, 219, 0.24); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0049">জবাব</span> <span style="background-color: rgba(231, 76, 60, 0.38); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0132">দেব</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0035">স্বামীকে,</span> <span style="background-color: rgba(231, 76, 60, 0.55); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0231">আমি</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0403">তো</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0064">তোমার</span> <span style="background-color: rgba(52, 152, 219, 0.47); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0184">সন্তানকে</span> <span style="background-color: rgba(52, 152, 219, 0.77); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0357">বাঁ’চাতে</span> <span style="background-color: rgba(231, 76, 60, 0.82); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0383">পারলাম</span> <span style="background-color: rgba(52, 152, 219, 0.43); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0161">না</span>
*Top Clickbait Tokens (SHAP):* `তো`, `পারলাম`, `কী`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0313">তো</span> <span style="background-color: rgba(231, 76, 60, 0.76); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0272">তোমার</span> <span style="background-color: rgba(231, 76, 60, 0.73); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0258">আমি</span> <span style="background-color: rgba(231, 76, 60, 0.70); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0248">স্বামীকে,</span> <span style="background-color: rgba(231, 76, 60, 0.68); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0237">না</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0234">আমি</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0230">কী</span> <span style="background-color: rgba(231, 76, 60, 0.64); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0219">পারলাম</span> <span style="background-color: rgba(231, 76, 60, 0.63); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0213">জবাব</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0209">বাঁ’চাতে</span>
*Top Clickbait Tokens (LIME):* `তো`, `তোমার`, `আমি`

---

### Headline #6: ✅ Correct
**Headline Text:** *তানিয়ার প্রতি মাসে আয় ৭০ হাজার টাকা*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `16.95%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `29%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1994">তানিয়ার</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0180">প্রতি</span> <span style="background-color: rgba(52, 152, 219, 0.36); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0612">মাসে</span> <span style="background-color: rgba(52, 152, 219, 0.44); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0826">আয়</span> <span style="background-color: rgba(52, 152, 219, 0.60); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1286">৭০</span> <span style="background-color: rgba(231, 76, 60, 0.21); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0182">হাজার</span> <span style="background-color: rgba(52, 152, 219, 0.66); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1465">টাকা</span>
*Top Clickbait Tokens (SHAP):* `হাজার`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.76); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0007">টাকা</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0005">প্রতি</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0005">মাসে</span> <span style="background-color: rgba(231, 76, 60, 0.50); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0004">হাজার</span> <span style="background-color: transparent; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Neutral">৭০</span> <span style="background-color: transparent; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Neutral">আয়</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0008">তানিয়ার</span>
*Top Clickbait Tokens (LIME):* `টাকা`, `প্রতি`, `মাসে`

---

### Headline #7: ✅ Correct
**Headline Text:** *ব্রাজিল-ভেনেজুয়েলা ম্যাচ দিয়ে পর্দা উঠছে কোপার*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `8.01%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `50%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.4242">ব্রাজিল-ভেনেজুয়েলা</span> <span style="background-color: rgba(52, 152, 219, 0.19); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0239">ম্যাচ</span> <span style="background-color: rgba(231, 76, 60, 0.22); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0426">দিয়ে</span> <span style="background-color: rgba(231, 76, 60, 0.22); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0431">পর্দা</span> <span style="background-color: rgba(231, 76, 60, 0.19); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0220">উঠছে</span> <span style="background-color: rgba(52, 152, 219, 0.76); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3670">কোপার</span>
*Top Clickbait Tokens (SHAP):* `পর্দা`, `দিয়ে`, `উঠছে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(52, 152, 219, 0.23); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0001">ব্রাজিল-ভেনেজুয়েলা</span> <span style="background-color: rgba(52, 152, 219, 0.54); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0005">উঠছে</span> <span style="background-color: rgba(52, 152, 219, 0.54); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0005">দিয়ে</span> <span style="background-color: rgba(52, 152, 219, 0.62); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0006">পর্দা</span> <span style="background-color: rgba(52, 152, 219, 0.62); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0006">ম্যাচ</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0009">কোপার</span>
*Top Clickbait Tokens (LIME):* None

---

### Headline #8: ✅ Correct
**Headline Text:** *ফের বাংলাদেশের ছবিতে রূপসা*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `11.80%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `50%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.41); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1466">ফের</span> <span style="background-color: rgba(52, 152, 219, 0.35); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1117">বাংলাদেশের</span> <span style="background-color: rgba(52, 152, 219, 0.19); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0229">ছবিতে</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3883">রূপসা</span>
*Top Clickbait Tokens (SHAP):* None

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.30); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0003">ছবিতে</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0001">বাংলাদেশের</span> <span style="background-color: rgba(52, 152, 219, 0.40); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0005">ফের</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0014">রূপসা</span>
*Top Clickbait Tokens (LIME):* `ছবিতে`, `বাংলাদেশের`

---

### Headline #9: ✅ Correct
**Headline Text:** *অনুষ্কা শর্মা হলেন দেশের বৌমাঃ করণ জোহর*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `41.49%`)
- **Method Agreement (Top-3 Jaccard):** **`0.00`** | **Sign Agreement:** `29%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.55); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0742">অনুষ্কা</span> <span style="background-color: rgba(52, 152, 219, 0.60); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0832">শর্মা</span> <span style="background-color: rgba(231, 76, 60, 0.24); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0163">হলেন</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0627">দেশের</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0621">বৌমাঃ</span> <span style="background-color: rgba(231, 76, 60, 0.27); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0229">করণ</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1296">জোহর</span>
*Top Clickbait Tokens (SHAP):* `করণ`, `হলেন`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0036">হলেন</span> <span style="background-color: rgba(231, 76, 60, 0.83); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0035">বৌমাঃ</span> <span style="background-color: rgba(231, 76, 60, 0.75); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0031">দেশের</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0026">করণ</span> <span style="background-color: rgba(231, 76, 60, 0.60); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0023">অনুষ্কা</span> <span style="background-color: rgba(231, 76, 60, 0.60); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0023">জোহর</span> <span style="background-color: rgba(231, 76, 60, 0.58); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0022">শর্মা</span>
*Top Clickbait Tokens (LIME):* `হলেন`, `বৌমাঃ`, `দেশের`

---

### Headline #10: ❌ Misclassified
**Headline Text:** *মেধাবী ডাক্তার থেকে যেভাবে ‘জনপ্রিয় নায়িকা’ হলে ‘সাই পল্লবী’*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `87.12%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `78%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.23); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0051">মেধাবী</span> <span style="background-color: rgba(231, 76, 60, 0.22); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0048">ডাক্তার</span> <span style="background-color: rgba(52, 152, 219, 0.23); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0051">থেকে</span> <span style="background-color: rgba(231, 76, 60, 0.79); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0419">যেভাবে</span> <span style="background-color: rgba(231, 76, 60, 0.49); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0225">‘জনপ্রিয়</span> <span style="background-color: rgba(231, 76, 60, 0.64); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0322">নায়িকা’</span> <span style="background-color: rgba(231, 76, 60, 0.35); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0134">হলে</span> <span style="background-color: rgba(231, 76, 60, 0.37); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0147">‘সাই</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0458">পল্লবী’</span>
*Top Clickbait Tokens (SHAP):* `যেভাবে`, `নায়িকা’`, `‘জনপ্রিয়`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0263">‘সাই</span> <span style="background-color: rgba(231, 76, 60, 0.76); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0230">যেভাবে</span> <span style="background-color: rgba(231, 76, 60, 0.74); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0220">হলে</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0196">নায়িকা’</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0193">মেধাবী</span> <span style="background-color: rgba(231, 76, 60, 0.65); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0187">‘জনপ্রিয়</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0175">ডাক্তার</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0167">পল্লবী’</span> <span style="background-color: rgba(231, 76, 60, 0.38); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0085">থেকে</span>
*Top Clickbait Tokens (LIME):* `‘সাই`, `যেভাবে`, `হলে`

---

### Headline #11: ❌ Misclassified
**Headline Text:** *প্রোটিয়াদের বিপক্ষে যে ভুল থেকে বেরিয়ে আসতে হবে*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `81.62%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `25%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.64); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2614">প্রোটিয়াদের</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0342">বিপক্ষে</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.3734">যে</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0602">ভুল</span> <span style="background-color: rgba(52, 152, 219, 0.15); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0010">থেকে</span> <span style="background-color: rgba(52, 152, 219, 0.20); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0274">বেরিয়ে</span> <span style="background-color: rgba(52, 152, 219, 0.26); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0564">আসতে</span> <span style="background-color: rgba(52, 152, 219, 0.20); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0244">হবে</span>
*Top Clickbait Tokens (SHAP):* `যে`, `ভুল`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0199">যে</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0132">হবে</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0131">থেকে</span> <span style="background-color: rgba(231, 76, 60, 0.57); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0120">ভুল</span> <span style="background-color: rgba(231, 76, 60, 0.54); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0111">বেরিয়ে</span> <span style="background-color: rgba(231, 76, 60, 0.42); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0077">আসতে</span> <span style="background-color: rgba(231, 76, 60, 0.38); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0064">প্রোটিয়াদের</span> <span style="background-color: rgba(231, 76, 60, 0.36); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0059">বিপক্ষে</span>
*Top Clickbait Tokens (LIME):* `যে`, `হবে`, `থেকে`

---

### Headline #12: ❌ Misclassified
**Headline Text:** *সব কিছুর পরে শুটিংয়ে ব্যস্ত পরীমনি*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `79.09%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `50%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.53); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0482">সব</span> <span style="background-color: rgba(231, 76, 60, 0.70); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0697">কিছুর</span> <span style="background-color: rgba(231, 76, 60, 0.71); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0714">পরে</span> <span style="background-color: rgba(52, 152, 219, 0.46); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0392">শুটিংয়ে</span> <span style="background-color: rgba(52, 152, 219, 0.61); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0580">ব্যস্ত</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0888">পরীমনি</span>
*Top Clickbait Tokens (SHAP):* `পরে`, `কিছুর`, `সব`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0140">সব</span> <span style="background-color: rgba(231, 76, 60, 0.84); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0139">কিছুর</span> <span style="background-color: rgba(231, 76, 60, 0.69); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0109">পরীমনি</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0103">পরে</span> <span style="background-color: rgba(231, 76, 60, 0.60); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0091">ব্যস্ত</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0072">শুটিংয়ে</span>
*Top Clickbait Tokens (LIME):* `সব`, `কিছুর`, `পরীমনি`

---

### Headline #13: ❌ Misclassified
**Headline Text:** *যেভাবে বিড়ালটি ফেক আইডি চিনতে পারল*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `42.08%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `17%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.35); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0809">যেভাবে</span> <span style="background-color: rgba(52, 152, 219, 0.20); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0192">বিড়ালটি</span> <span style="background-color: rgba(52, 152, 219, 0.23); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0310">ফেক</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2866">আইডি</span> <span style="background-color: rgba(52, 152, 219, 0.40); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1036">চিনতে</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0073">পারল</span>
*Top Clickbait Tokens (SHAP):* `যেভাবে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0071">যেভাবে</span> <span style="background-color: rgba(231, 76, 60, 0.72); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0058">পারল</span> <span style="background-color: rgba(231, 76, 60, 0.65); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0051">ফেক</span> <span style="background-color: rgba(231, 76, 60, 0.65); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0051">বিড়ালটি</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0048">চিনতে</span> <span style="background-color: rgba(231, 76, 60, 0.50); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0036">আইডি</span>
*Top Clickbait Tokens (LIME):* `যেভাবে`, `পারল`, `ফেক`

---

### Headline #14: ❌ Misclassified
**Headline Text:** *একেই বলে গুরুমারা বিদ্যে! ধোনিকে দুবার ফাঁদে ফেলে আউট করেন পন্থ, ফাঁস স্ট্র্যাটেজি*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `30.63%`)
- **Method Agreement (Top-3 Jaccard):** **`0.00`** | **Sign Agreement:** `40%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.26); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0540">একেই</span> <span style="background-color: rgba(231, 76, 60, 0.27); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0618">বলে</span> <span style="background-color: rgba(52, 152, 219, 0.28); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0667">গুরুমারা</span> <span style="background-color: rgba(231, 76, 60, 0.39); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1234">বিদ্যে!</span> <span style="background-color: rgba(52, 152, 219, 0.19); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0202">ধোনিকে</span> <span style="background-color: rgba(52, 152, 219, 0.20); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0280">দুবার</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0361">ফাঁদে</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0375">ফেলে</span> <span style="background-color: rgba(52, 152, 219, 0.34); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0987">আউট</span> <span style="background-color: rgba(231, 76, 60, 0.23); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0397">করেন</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0084">পন্থ,</span> <span style="background-color: rgba(231, 76, 60, 0.15); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0004">ফাঁস</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3570">স্ট্র্যাটেজি</span>
*Top Clickbait Tokens (SHAP):* `বিদ্যে!`, `বলে`, `করেন`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0051">পন্থ,</span> <span style="background-color: rgba(231, 76, 60, 0.73); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0042">করেন</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0037">ফেলে</span> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0037">বিদ্যে!</span> <span style="background-color: rgba(231, 76, 60, 0.64); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0036">আউট</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0034">বলে</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0034">গুরুমারা</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0026">ফাঁস</span> <span style="background-color: rgba(231, 76, 60, 0.48); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0024">ধোনিকে</span> <span style="background-color: rgba(231, 76, 60, 0.44); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0021">দুবার</span>
*Top Clickbait Tokens (LIME):* `পন্থ,`, `করেন`, `ফেলে`

---

### Headline #15: ❌ Misclassified
**Headline Text:** *তৃতীয়বারের মত জুটি বাঁধলেন সিয়াম-পরী*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `23.11%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `20%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.61); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1697">তৃতীয়বারের</span> <span style="background-color: rgba(231, 76, 60, 0.48); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1220">মত</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0609">জুটি</span> <span style="background-color: rgba(52, 152, 219, 0.67); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1916">বাঁধলেন</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2563">সিয়াম-পরী</span>
*Top Clickbait Tokens (SHAP):* `মত`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0035">মত</span> <span style="background-color: rgba(231, 76, 60, 0.53); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0019">জুটি</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0018">সিয়াম-পরী</span> <span style="background-color: rgba(231, 76, 60, 0.49); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0017">বাঁধলেন</span> <span style="background-color: rgba(231, 76, 60, 0.41); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0013">তৃতীয়বারের</span>
*Top Clickbait Tokens (LIME):* `মত`, `জুটি`, `সিয়াম-পরী`

---
