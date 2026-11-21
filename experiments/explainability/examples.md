# TinyBanglaBERT Explainability: SHAP vs. LIME Token Attributions

*Generated: 2026-11-19 22:27 UTC*
*Sample Size: 15 diverse test headlines (covering True Positives, True Negatives, False Positives, and False Negatives)*

### Highlighting Legend
- <span style="background-color: rgba(231, 76, 60, 0.6); padding: 2px 5px; font-weight: bold;">Red Highlights</span> : Tokens with **positive attribution** (push model toward **Clickbait**).
- <span style="background-color: rgba(52, 152, 219, 0.6); padding: 2px 5px;">Blue Highlights</span> : Tokens with **negative attribution** (push model toward **Not-Clickbait** / Informative).

---

### Headline #1: ✅ Correct
**Headline Text:** *এই সেই ভিআইপি নুপুর প্রতিটি প্রেমের জন্য পারিশ্রমিক দিতে হয় ৫ লাখ থেকে ৫০ লাখ টাকা!*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `98.59%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `90%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.66); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0108">এই</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0149">সেই</span> <span style="background-color: rgba(52, 152, 219, 0.75); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0128">ভিআইপি</span> <span style="background-color: rgba(52, 152, 219, 0.66); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0109">নুপুর</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0005">প্রতিটি</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0011">প্রেমের</span> <span style="background-color: rgba(231, 76, 60, 0.15); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0001">জন্য</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0005">পারিশ্রমিক</span> <span style="background-color: rgba(231, 76, 60, 0.18); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0006">দিতে</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0004">হয়</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0004">৫</span> <span style="background-color: rgba(52, 152, 219, 0.19); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0008">লাখ</span> <span style="background-color: rgba(52, 152, 219, 0.15); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0001">থেকে</span> <span style="background-color: transparent; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Neutral">৫০</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0004">লাখ</span> <span style="background-color: rgba(231, 76, 60, 0.16); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0003">টাকা!</span>
*Top Clickbait Tokens (SHAP):* `সেই`, `এই`, `প্রেমের`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.3962">সেই</span> <span style="background-color: rgba(231, 76, 60, 0.46); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1773">এই</span> <span style="background-color: rgba(231, 76, 60, 0.39); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1338">প্রতিটি</span> <span style="background-color: rgba(231, 76, 60, 0.37); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1235">টাকা!</span> <span style="background-color: rgba(231, 76, 60, 0.36); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1165">প্রেমের</span> <span style="background-color: rgba(231, 76, 60, 0.34); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1064">জন্য</span> <span style="background-color: rgba(231, 76, 60, 0.31); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0927">নুপুর</span> <span style="background-color: rgba(231, 76, 60, 0.30); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0823">দিতে</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0816">৫</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0800">হয়</span>
*Top Clickbait Tokens (LIME):* `সেই`, `এই`, `প্রতিটি`

---

### Headline #2: ✅ Correct
**Headline Text:** *আবারো যে অভিযোগে অভিযুক্ত মোহাম্মদ আশরাফুল*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `79.44%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `83%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0138">আবারো</span> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.4395">যে</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0469">অভিযোগে</span> <span style="background-color: rgba(52, 152, 219, 0.25); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0641">অভিযুক্ত</span> <span style="background-color: rgba(52, 152, 219, 0.40); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1598">মোহাম্মদ</span> <span style="background-color: rgba(52, 152, 219, 0.74); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3704">আশরাফুল</span>
*Top Clickbait Tokens (SHAP):* `যে`, `আবারো`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.5272">যে</span> <span style="background-color: rgba(231, 76, 60, 0.79); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.4842">আবারো</span> <span style="background-color: rgba(231, 76, 60, 0.18); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0218">মোহাম্মদ</span> <span style="background-color: rgba(52, 152, 219, 0.18); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0200">অভিযোগে</span> <span style="background-color: rgba(52, 152, 219, 0.18); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0238">আশরাফুল</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0535">অভিযুক্ত</span>
*Top Clickbait Tokens (LIME):* `যে`, `আবারো`, `মোহাম্মদ`

---

### Headline #3: ✅ Correct
**Headline Text:** *দু’য়ের বেশি সন্তান থাকলে সরকারি সুবিধা মিলবে না*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `91.02%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `38%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0572">দু’য়ের</span> <span style="background-color: rgba(52, 152, 219, 0.31); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0130">বেশি</span> <span style="background-color: rgba(52, 152, 219, 0.69); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0439">সন্তান</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0118">থাকলে</span> <span style="background-color: rgba(52, 152, 219, 0.44); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0237">সরকারি</span> <span style="background-color: rgba(52, 152, 219, 0.46); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0255">সুবিধা</span> <span style="background-color: rgba(52, 152, 219, 0.34); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0158">মিলবে</span> <span style="background-color: rgba(52, 152, 219, 0.39); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0195">না</span>
*Top Clickbait Tokens (SHAP):* `দু’য়ের`, `থাকলে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0280">দু’য়ের</span> <span style="background-color: rgba(231, 76, 60, 0.82); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0269">থাকলে</span> <span style="background-color: rgba(231, 76, 60, 0.74); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0237">মিলবে</span> <span style="background-color: rgba(231, 76, 60, 0.70); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0221">না</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0183">সুবিধা</span> <span style="background-color: rgba(231, 76, 60, 0.49); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0136">বেশি</span> <span style="background-color: rgba(231, 76, 60, 0.34); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0076">সরকারি</span> <span style="background-color: rgba(52, 152, 219, 0.83); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0274">সন্তান</span>
*Top Clickbait Tokens (LIME):* `দু’য়ের`, `থাকলে`, `মিলবে`

---

### Headline #4: ✅ Correct
**Headline Text:** *পুরনো বন্ধুর সঙ্গে দেখা, কোলে বসলেন রচনা!*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `95.05%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `0%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.55); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0046">পুরনো</span> <span style="background-color: rgba(52, 152, 219, 0.75); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0069">বন্ধুর</span> <span style="background-color: rgba(52, 152, 219, 0.35); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0023">সঙ্গে</span> <span style="background-color: rgba(52, 152, 219, 0.38); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0027">দেখা,</span> <span style="background-color: rgba(52, 152, 219, 0.55); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0046">কোলে</span> <span style="background-color: rgba(52, 152, 219, 0.38); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0027">বসলেন</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0081">রচনা!</span>
*Top Clickbait Tokens (SHAP):* None

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.4052">রচনা!</span> <span style="background-color: rgba(231, 76, 60, 0.54); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.2247">বসলেন</span> <span style="background-color: rgba(231, 76, 60, 0.48); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1908">দেখা,</span> <span style="background-color: rgba(231, 76, 60, 0.43); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1630">সঙ্গে</span> <span style="background-color: rgba(231, 76, 60, 0.41); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1516">পুরনো</span> <span style="background-color: rgba(231, 76, 60, 0.39); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1380">কোলে</span> <span style="background-color: rgba(231, 76, 60, 0.31); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0902">বন্ধুর</span>
*Top Clickbait Tokens (LIME):* `রচনা!`, `বসলেন`, `দেখা,`

---

### Headline #5: ✅ Correct
**Headline Text:** *স্ত্রীকে সুখী করতে চাইলে ৯ কৌশল আপনার জানা জরুরি*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `94.95%`)
- **Method Agreement (Top-3 Jaccard):** **`0.00`** | **Sign Agreement:** `44%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.77); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0141">স্ত্রীকে</span> <span style="background-color: rgba(52, 152, 219, 0.74); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0134">সুখী</span> <span style="background-color: rgba(231, 76, 60, 0.42); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0061">করতে</span> <span style="background-color: rgba(231, 76, 60, 0.62); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0105">চাইলে</span> <span style="background-color: rgba(52, 152, 219, 0.46); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0069">৯</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0158">কৌশল</span> <span style="background-color: rgba(231, 76, 60, 0.40); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0057">আপনার</span> <span style="background-color: rgba(231, 76, 60, 0.19); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0009">জানা</span> <span style="background-color: rgba(52, 152, 219, 0.42); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0060">জরুরি</span>
*Top Clickbait Tokens (SHAP):* `চাইলে`, `করতে`, `আপনার`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.2793">আপনার</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1845">জরুরি</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1758">চাইলে</span> <span style="background-color: rgba(231, 76, 60, 0.55); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1582">জানা</span> <span style="background-color: rgba(231, 76, 60, 0.54); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1541">কৌশল</span> <span style="background-color: rgba(231, 76, 60, 0.46); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1240">৯</span> <span style="background-color: rgba(231, 76, 60, 0.40); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1012">করতে</span> <span style="background-color: rgba(231, 76, 60, 0.37); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0893">সুখী</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0558">স্ত্রীকে</span>
*Top Clickbait Tokens (LIME):* `আপনার`, `জরুরি`, `চাইলে`

---

### Headline #6: ✅ Correct
**Headline Text:** *ডিভোর্স দেওয়ার পরও একাধিক হোটেলে রাকিবের স’ঙ্গে ছিলেন তামিমা*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `4.84%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `56%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.41); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1652">ডিভোর্স</span> <span style="background-color: rgba(231, 76, 60, 0.28); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0819">দেওয়ার</span> <span style="background-color: rgba(231, 76, 60, 0.28); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0856">পরও</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0140">একাধিক</span> <span style="background-color: rgba(52, 152, 219, 0.27); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0753">হোটেলে</span> <span style="background-color: rgba(52, 152, 219, 0.78); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.4080">রাকিবের</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0691">স’ঙ্গে</span> <span style="background-color: rgba(52, 152, 219, 0.24); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0559">ছিলেন</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.4524">তামিমা</span>
*Top Clickbait Tokens (SHAP):* `পরও`, `দেওয়ার`, `স’ঙ্গে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0085">স’ঙ্গে</span> <span style="background-color: rgba(231, 76, 60, 0.50); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0042">পরও</span> <span style="background-color: rgba(231, 76, 60, 0.39); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0029">একাধিক</span> <span style="background-color: rgba(231, 76, 60, 0.32); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0021">হোটেলে</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0013">দেওয়ার</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0006">ডিভোর্স</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0002">ছিলেন</span> <span style="background-color: rgba(52, 152, 219, 0.35); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0024">রাকিবের</span> <span style="background-color: rgba(52, 152, 219, 0.77); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0075">তামিমা</span>
*Top Clickbait Tokens (LIME):* `স’ঙ্গে`, `পরও`, `একাধিক`

---

### Headline #7: ✅ Correct
**Headline Text:** *নথি ফাঁসে সাংবাদিকদের ইমেইল ও ফোন রেকর্ড জব্দ নয়*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `4.28%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `67%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3462">নথি</span> <span style="background-color: rgba(52, 152, 219, 0.26); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0557">ফাঁসে</span> <span style="background-color: rgba(52, 152, 219, 0.55); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1991">সাংবাদিকদের</span> <span style="background-color: rgba(52, 152, 219, 0.76); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3029">ইমেইল</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0283">ও</span> <span style="background-color: rgba(52, 152, 219, 0.15); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0004">ফোন</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0550">রেকর্ড</span> <span style="background-color: rgba(231, 76, 60, 0.26); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0524">জব্দ</span> <span style="background-color: rgba(52, 152, 219, 0.38); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1145">নয়</span>
*Top Clickbait Tokens (SHAP):* `রেকর্ড`, `জব্দ`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.71); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0033">নথি</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0026">রেকর্ড</span> <span style="background-color: rgba(231, 76, 60, 0.56); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0024">জব্দ</span> <span style="background-color: rgba(231, 76, 60, 0.27); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0007">নয়</span> <span style="background-color: rgba(231, 76, 60, 0.24); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0005">ও</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0010">ফোন</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0010">ইমেইল</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0041">সাংবাদিকদের</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0041">ফাঁসে</span>
*Top Clickbait Tokens (LIME):* `নথি`, `রেকর্ড`, `জব্দ`

---

### Headline #8: ✅ Correct
**Headline Text:** *ভারতের বিশ্বকাপ দলের প্রশংসায় এখন পঞ্চমুখ স্মিথ*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `3.31%`)
- **Method Agreement (Top-3 Jaccard):** **`1.00`** | **Sign Agreement:** `86%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.4131">ভারতের</span> <span style="background-color: rgba(52, 152, 219, 0.68); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.3125">বিশ্বকাপ</span> <span style="background-color: rgba(52, 152, 219, 0.34); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1101">দলের</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0989">প্রশংসায়</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0393">এখন</span> <span style="background-color: rgba(231, 76, 60, 0.22); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0391">পঞ্চমুখ</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0147">স্মিথ</span>
*Top Clickbait Tokens (SHAP):* `পঞ্চমুখ`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(52, 152, 219, 0.26); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0006">এখন</span> <span style="background-color: rgba(52, 152, 219, 0.33); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0010">পঞ্চমুখ</span> <span style="background-color: rgba(52, 152, 219, 0.44); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0016">স্মিথ</span> <span style="background-color: rgba(52, 152, 219, 0.52); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0020">প্রশংসায়</span> <span style="background-color: rgba(52, 152, 219, 0.72); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0031">দলের</span> <span style="background-color: rgba(52, 152, 219, 0.78); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0034">ভারতের</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0038">বিশ্বকাপ</span>
*Top Clickbait Tokens (LIME):* None

---

### Headline #9: ✅ Correct
**Headline Text:** *বোনের বিয়েবার্ষিকী অনুষ্ঠানের ৯২ হাজার টাকা বিল দেন মুনিয়া!*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `3.25%`)
- **Method Agreement (Top-3 Jaccard):** **`0.00`** | **Sign Agreement:** `67%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.76); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2343">বোনের</span> <span style="background-color: rgba(52, 152, 219, 0.84); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2619">বিয়েবার্ষিকী</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1291">অনুষ্ঠানের</span> <span style="background-color: rgba(52, 152, 219, 0.56); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1571">৯২</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0087">হাজার</span> <span style="background-color: rgba(231, 76, 60, 0.23); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0322">টাকা</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0246">বিল</span> <span style="background-color: rgba(231, 76, 60, 0.37); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0828">দেন</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2668">মুনিয়া!</span>
*Top Clickbait Tokens (SHAP):* `দেন`, `টাকা`, `হাজার`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(52, 152, 219, 0.27); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0007">বিয়েবার্ষিকী</span> <span style="background-color: rgba(52, 152, 219, 0.27); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0007">মুনিয়া!</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0010">হাজার</span> <span style="background-color: rgba(52, 152, 219, 0.41); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0015">বিল</span> <span style="background-color: rgba(52, 152, 219, 0.46); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0018">বোনের</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0020">দেন</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0020">অনুষ্ঠানের</span> <span style="background-color: rgba(52, 152, 219, 0.58); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0025">টাকা</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0041">৯২</span>
*Top Clickbait Tokens (LIME):* None

---

### Headline #10: ❌ Misclassified
**Headline Text:** *শিশু লম্বা হচ্ছে না, জেনে নিন কিছু খাবার ও ব্যায়াম*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `55.93%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `10%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.70); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0699">শিশু</span> <span style="background-color: rgba(52, 152, 219, 0.43); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0356">লম্বা</span> <span style="background-color: rgba(52, 152, 219, 0.53); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0476">হচ্ছে</span> <span style="background-color: rgba(52, 152, 219, 0.54); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0491">না,</span> <span style="background-color: rgba(52, 152, 219, 0.18); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0041">জেনে</span> <span style="background-color: rgba(52, 152, 219, 0.27); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0156">নিন</span> <span style="background-color: rgba(52, 152, 219, 0.17); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0024">কিছু</span> <span style="background-color: rgba(52, 152, 219, 0.68); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0668">খাবার</span> <span style="background-color: rgba(52, 152, 219, 0.50); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0439">ও</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0883">ব্যায়াম</span>
*Top Clickbait Tokens (SHAP):* None

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0667">কিছু</span> <span style="background-color: rgba(231, 76, 60, 0.82); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0642">জেনে</span> <span style="background-color: rgba(231, 76, 60, 0.77); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0593">ব্যায়াম</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0496">লম্বা</span> <span style="background-color: rgba(231, 76, 60, 0.63); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0454">না,</span> <span style="background-color: rgba(231, 76, 60, 0.59); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0416">নিন</span> <span style="background-color: rgba(231, 76, 60, 0.51); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0343">ও</span> <span style="background-color: rgba(231, 76, 60, 0.44); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0280">খাবার</span> <span style="background-color: rgba(231, 76, 60, 0.29); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0135">শিশু</span> <span style="background-color: rgba(52, 152, 219, 0.22); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0071">হচ্ছে</span>
*Top Clickbait Tokens (LIME):* `কিছু`, `জেনে`, `ব্যায়াম`

---

### Headline #11: ❌ Misclassified
**Headline Text:** *প্রকাশ্যে এলো নুসরাতের বেবি বাম্পের ছবি*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `56.81%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `0%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.50); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0520">প্রকাশ্যে</span> <span style="background-color: rgba(52, 152, 219, 0.49); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0509">এলো</span> <span style="background-color: rgba(52, 152, 219, 0.50); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0521">নুসরাতের</span> <span style="background-color: rgba(52, 152, 219, 0.52); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0554">বেবি</span> <span style="background-color: rgba(52, 152, 219, 0.83); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1005">বাম্পের</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1036">ছবি</span>
*Top Clickbait Tokens (SHAP):* None

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0108">ছবি</span> <span style="background-color: rgba(231, 76, 60, 0.75); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0092">প্রকাশ্যে</span> <span style="background-color: rgba(231, 76, 60, 0.71); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0086">বাম্পের</span> <span style="background-color: rgba(231, 76, 60, 0.67); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0080">নুসরাতের</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0071">এলো</span> <span style="background-color: rgba(231, 76, 60, 0.58); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0067">বেবি</span>
*Top Clickbait Tokens (LIME):* `ছবি`, `প্রকাশ্যে`, `বাম্পের`

---

### Headline #12: ❌ Misclassified
**Headline Text:** *যে ‘ক্যানসারে’ আক্রান্ত বাংলাদেশের টি–টোয়েন্টি*

- **Ground Truth:** `not-clickbait` (Label: 0)
- **TinyBanglaBERT Prediction:** `clickbait` (Clickbait Prob: `72.30%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `20%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(231, 76, 60, 0.76); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1722">যে</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1961">‘ক্যানসারে’</span> <span style="background-color: rgba(52, 152, 219, 0.41); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0718">আক্রান্ত</span> <span style="background-color: rgba(52, 152, 219, 0.36); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0589">বাংলাদেশের</span> <span style="background-color: rgba(52, 152, 219, 0.52); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1050">টি–টোয়েন্টি</span>
*Top Clickbait Tokens (SHAP):* `যে`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.7107">যে</span> <span style="background-color: rgba(231, 76, 60, 0.28); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1304">টি–টোয়েন্টি</span> <span style="background-color: rgba(231, 76, 60, 0.23); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0826">আক্রান্ত</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0516">বাংলাদেশের</span> <span style="background-color: rgba(231, 76, 60, 0.16); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0128">‘ক্যানসারে’</span>
*Top Clickbait Tokens (LIME):* `যে`, `টি–টোয়েন্টি`, `আক্রান্ত`

---

### Headline #13: ❌ Misclassified
**Headline Text:** *ফেসবুকে পোস্ট করে সাবধান করলেন নায়িকা মাহিয়া মাহি*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `49.33%`)
- **Method Agreement (Top-3 Jaccard):** **`0.20`** | **Sign Agreement:** `25%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1850">ফেসবুকে</span> <span style="background-color: rgba(52, 152, 219, 0.39); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0641">পোস্ট</span> <span style="background-color: rgba(52, 152, 219, 0.30); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0391">করে</span> <span style="background-color: rgba(231, 76, 60, 0.36); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0553">সাবধান</span> <span style="background-color: rgba(52, 152, 219, 0.36); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0545">করলেন</span> <span style="background-color: rgba(52, 152, 219, 0.39); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0643">নায়িকা</span> <span style="background-color: rgba(52, 152, 219, 0.38); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0619">মাহিয়া</span> <span style="background-color: rgba(52, 152, 219, 0.44); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0756">মাহি</span>
*Top Clickbait Tokens (SHAP):* `সাবধান`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0177">সাবধান</span> <span style="background-color: rgba(231, 76, 60, 0.78); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0160">মাহিয়া</span> <span style="background-color: rgba(231, 76, 60, 0.55); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0101">নায়িকা</span> <span style="background-color: rgba(231, 76, 60, 0.48); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0083">মাহি</span> <span style="background-color: rgba(231, 76, 60, 0.40); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0063">করলেন</span> <span style="background-color: rgba(231, 76, 60, 0.36); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0054">করে</span> <span style="background-color: rgba(231, 76, 60, 0.21); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0015">পোস্ট</span> <span style="background-color: rgba(52, 152, 219, 0.28); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0033">ফেসবুকে</span>
*Top Clickbait Tokens (LIME):* `সাবধান`, `মাহিয়া`, `নায়িকা`

---

### Headline #14: ❌ Misclassified
**Headline Text:** *নিবন্ধন না থাকলেও বন্ধ হচ্ছে না যেসব হ্যান্ডসেট*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `7.23%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `71%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.7985">নিবন্ধন</span> <span style="background-color: rgba(231, 76, 60, 0.20); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0535">না</span> <span style="background-color: rgba(231, 76, 60, 0.17); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0235">থাকলেও</span> <span style="background-color: rgba(52, 152, 219, 0.21); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0676">বন্ধ</span> <span style="background-color: rgba(231, 76, 60, 0.15); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0002">হচ্ছে</span> <span style="background-color: rgba(52, 152, 219, 0.18); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0290">না</span> <span style="background-color: rgba(231, 76, 60, 0.24); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1053">যেসব</span> <span style="background-color: rgba(52, 152, 219, 0.32); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.1976">হ্যান্ডসেট</span>
*Top Clickbait Tokens (SHAP):* `যেসব`, `না`, `থাকলেও`

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.1063">যেসব</span> <span style="background-color: rgba(231, 76, 60, 0.61); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0695">না</span> <span style="background-color: rgba(231, 76, 60, 0.54); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0587">না</span> <span style="background-color: rgba(231, 76, 60, 0.45); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0458">হচ্ছে</span> <span style="background-color: rgba(231, 76, 60, 0.44); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0444">থাকলেও</span> <span style="background-color: rgba(231, 76, 60, 0.44); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0443">বন্ধ</span> <span style="background-color: rgba(52, 152, 219, 0.41); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0393">নিবন্ধন</span> <span style="background-color: rgba(52, 152, 219, 0.56); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0629">হ্যান্ডসেট</span>
*Top Clickbait Tokens (LIME):* `যেসব`, `না`, `না`

---

### Headline #15: ❌ Misclassified
**Headline Text:** *১০ কেজি ওজন কমিয়েছি*

- **Ground Truth:** `clickbait` (Label: 1)
- **TinyBanglaBERT Prediction:** `not-clickbait` (Clickbait Prob: `9.70%`)
- **Method Agreement (Top-3 Jaccard):** **`0.50`** | **Sign Agreement:** `25%`

#### 1. SHAP Token Attributions (Shapley Values)
> <span style="background-color: rgba(52, 152, 219, 0.78); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2101">১০</span> <span style="background-color: rgba(52, 152, 219, 0.78); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2106">কেজি</span> <span style="background-color: rgba(52, 152, 219, 0.84); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2313">ওজন</span> <span style="background-color: rgba(52, 152, 219, 0.85); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.2335">কমিয়েছি</span>
*Top Clickbait Tokens (SHAP):* None

#### 2. LIME Local Linear Perturbations
> <span style="background-color: rgba(231, 76, 60, 0.85); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0047">১০</span> <span style="background-color: rgba(231, 76, 60, 0.58); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0029">ওজন</span> <span style="background-color: rgba(231, 76, 60, 0.55); color: #1a1a1a; font-weight: bold; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Clickbait attribution: +0.0027">কমিয়েছি</span> <span style="background-color: rgba(52, 152, 219, 0.24); color: #1a1a1a; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="Not-Clickbait attribution: -0.0006">কেজি</span>
*Top Clickbait Tokens (LIME):* `১০`, `ওজন`, `কমিয়েছি`

---
