# Candidate Tasks for Generalisation (Phase H)

This document lists candidate Bengali NLU tasks to test whether the core finding—that compact warm-start initialisation dominates distillation objectives—generalises across different task formulations beyond clickbait classification.

---

## Candidate 1: SentNoB (Bengali Social Media Sentiment Analysis)
- **Task Type:** 3-class Sentiment Classification (Positive, Negative, Neutral).
- **Source:** Islam et al., "SentNoB: A Dataset for Analyzing Sentiment on Noisy Bengali Texts," *EMNLP 2021*.
- **Domain:** User-generated Bengali social media comments (YouTube, Facebook).
- **Dataset Size:** 15,728 annotated comments (Train: 12,575, Val: 1,574, Test: 1,579).
- **HuggingFace Hub ID:** `csebuetnlp/sentnob`
- **Relevance:** Tests whether strided compact initialization preserves sentiment polarity nuance and informal vernacular understanding under noise, providing direct task divergence from journalistic headline clickbait.

---

## Candidate 2: BanglaNLI (Natural Language Inference)
- **Task Type:** 3-class Sentence-Pair Entailment (Entailment, Contradiction, Neutral).
- **Source:** Bhattacharjee et al., "BanglaBERT: A Large-scale Language Model for Bengali and a Comprehensive Benchmark for Bengali NLU," *Findings of NAACL 2022*.
- **Domain:** Bengali translated MultiNLI and native Bengali sentence pairs.
- **Dataset Size:** 13,000+ sentence pairs.
- **HuggingFace Hub ID:** `csebuetnlp/banglanli`
- **Relevance:** NLI requires complex cross-attention and logical inference rather than single-sentence lexical pattern matching. Evaluating initialisation vs distillation here tests whether structural reasoning capabilities are retained by compact sub-networks.

---

## Candidate 3: Bengali Hate Speech & Toxic Content Detection
- **Task Type:** Binary Classification (Toxic vs Non-Toxic).
- **Source:** Romim et al., "Hate Speech Detection in the Bengali Language: A Dataset and Its Evaluation," *Social Network Analysis and Mining*, 2021.
- **Dataset Size:** 10,000+ social media posts.
- **Relevance:** High real-world edge deployment value (on-device content moderation for mobile apps).
