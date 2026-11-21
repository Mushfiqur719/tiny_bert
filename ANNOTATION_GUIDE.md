# Annotation Guide: Bengali Headline Clickbait Classification

This guide establishes the standardized annotation criteria for evaluating Bengali news and social media headlines in **TinyBanglaBERT v3**. It follows the foundational definitions from Motahar et al. (BanglaBait) and standard computational journalism guidelines (Chakraborty et al., 2016).

---

## 1. Objective
Annotators must categorize each Bengali headline into one of two mutually exclusive classes:
- **`0` — Not-Clickbait (Informative / Direct)**
- **`1` — Clickbait (Sensationalist / Curiosity-Driven)**

---

## 2. Definitive Criteria for Clickbait (`1`)
A headline must be labeled as **Clickbait (`1`)** if it exhibits one or more of the following linguistic strategies designed to exploit the reader's "curiosity gap" or manipulate attention:

### A. Curiosity Gap & Withholding Crucial Information
The headline intentionally omits the primary subject, actor, event, or resolution, forcing the reader to click to understand what happened.
- *Example:* "অবশেষে মুখ খুললেন শাকিব খান, জানালেন আসল সত্যি!" (Withholds what Shakib actually said).
- *Example:* "যে কারণে ইত্যাদি ছাড়া আর কোথাও দেখা যায় না নাতিকে" (Withholds the underlying reason).
- *Example:* "কারন জানলে চোখের জ্বল আটকাতে পারবেননা!" (Classic forward-referencing emotional teaser).

### B. Sensationalism, Hyperbole & Exaggeration
Uses inflated, shocking, or hyperbolic phrasing that is disproportionate to the news reality.
- *Example:* "নেটদুনিয়ায় তোলপাড়!", "ফেসবুকে মুহূর্তেই ভিডিও ভাইরাল!", "আগুন ধরালেন অভিনেত্রী!"

### C. Direct Addressing & Imperative Phrasing
Directly commands or instructs the reader to view, react, or behave in a certain way.
- *Example:* "দেখুন পুলিশ বাবাকে কিভাবে অনুরোধ করছে শিশু, এখনই দেখুন!"
- *Example:* "১০ টি মুভি ছেলেদের অবশ্যই গোপনে দেখা উচিৎ"

### D. Misleading or Obfuscated Context
Presents quotes or shocking claims out of context to imply scandal, controversy, or intimacy that does not exist in the article body.
- *Example:* "রুমের দরজা বন্ধ করতেই পাল্টে গেলেন কোয়েল!"

---

## 3. Definitive Criteria for Not-Clickbait (`0`)
A headline must be labeled as **Not-Clickbait (`0`)** if it provides an objective, straightforward, and informative summary of the news story:

### A. Clear 5 W's (Who, What, Where, When, Why)
Identifies the key actors and the specific action or outcome directly in the headline.
- *Example:* "বৃষ্টির কারণে বাংলাদেশ-ভারত দ্বিতীয় টি-টোয়েন্টি ম্যাচ পরিত্যক্ত" (Clearly states teams, match, and outcome).
- *Example:* "গত ২৪ ঘণ্টায় দেশে করোনায় আক্রান্ত হয়ে আরও তিনজনের মৃত্যু" (Specific factual reportage with verifiable data).
- *Example:* "শেয়ারবাজারে সূচকের বড় পতন, লেনদেন কমেছে ২০ শতাংশ" (Direct economic summary).

### B. Conventional Journalistic Phrasing
Factual reports, diplomatic summaries, official press statements, and standard crime reportage without sensationalist commentary.

---

## 4. Edge Cases & Boundary Disambiguation
1. **Dramatic Real Events:** If an event is inherently dramatic or tragic (e.g. "বিমান দুর্ঘটনায় নিহত ১২ জনের পরিচয় শনাক্ত"), but the headline reports the facts straightforwardly without teaser language, label as **`0`**.
2. **Tabloid / Entertainment News:** Entertainment news is often about celebrities. If the headline is straightforward (e.g. "নতুন চলচ্চিত্রে চুক্তিবদ্ধ হলেন চঞ্চল চৌধুরী"), label as **`0`**. If it creates false mystery (e.g. "চঞ্চলের নতুন সঙ্গী কে? জানলে চমকে যাবেন!"), label as **`1`**.
3. **Punctuation Exclamations:** Multiple exclamation marks (`!`, `!!`) or ellipses (`...`) combined with vague claims strongly indicate clickbait (`1`).

---

## 5. Workflow
1. Annotators must independently record labels in columns `annotator_1_label` and `annotator_2_label`.
2. Do NOT discuss individual items during the initial annotation pass.
3. For items where `annotator_1_label != annotator_2_label`, a third adjudicator reviews and inputs the final decision into `resolved_label`.
