# Related Work: Compact Transformers, Distillation & Bengali NLU

This document reviews the theoretical and empirical literature underpinning **TinyBanglaBERT v3**, situating our central inquiry: *Does pre-trained initialisation matter more than the distillation objective in low-resource fine-tuning?*

---

## 1. Compact Pre-Trained Models vs. Task-Specific Distillation

### Turc et al. (2019) — *Well-Read Students Learn Better: On the Importance of Pre-training Compact Models*
- **Citation:** I. Turc, M.-W. Chang, K. Lee, and K. Toutanova, "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models," *arXiv preprint arXiv:1908.08962*, 2019.
- **Core Findings:** The authors systematically evaluated the interaction between pre-training and distillation across compact BERT variants (2 to 8 layers, 128 to 512 hidden dimension). They demonstrated that **pre-training compact student architectures before task fine-tuning consistently outperforms distilling directly from a large teacher into a randomly initialised student**. Furthermore, pre-trained compact students could be fine-tuned with standard cross-entropy to match or exceed the performance of students distilled from scratch.
- **Relation & Differences:** While Turc et al. established the value of compact pre-training on high-resource English GLUE benchmarks, they primarily evaluated general pre-training followed by distillation. In contrast, our study investigates:
  1. Low-resource morphologically rich Bengali NLP.
  2. Sub-network extraction from existing small models (`strided` vs. `first-k` vs. `last-k` layer selection) without requiring full, costly pre-training of the student from scratch.
  3. A full factorial cross with dataset size (10% to 100%) to determine whether distillation's value re-emerges when labelled data is strictly scarce.

---

## 2. Knowledge Distillation Frameworks for Transformers

### DistilBERT (Sanh et al., 2019)
- **Citation:** V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *NeurIPS EMC^2 Workshop*, 2019.
- **Methodology:** DistilBERT compresses BERT-base by 40% (6 layers instead of 12) while retaining 97% of its language understanding performance. It initialises the student by taking one layer out of two from the teacher and trains during the pre-training phase using a triple loss: soft target cross-entropy ($L_{\text{ce}}$), hard cross-entropy ($L_{\text{mlm}}$), and cosine embedding alignment ($L_{\text{cos}}$).
- **Difference:** DistilBERT's student shares the teacher's exact hidden dimension ($d=768$), which preserves 66M parameters and high memory requirements. In our work, TinyBanglaBERT compresses both depth ($12 \to 4$) and hidden representation width ($768 \to 256$, factorized embedding $128$), achieving an order-of-magnitude parameter reduction ($110.6\text{M} \to 7.42\text{M}$).

### TinyBERT (Jiao et al., 2020)
- **Citation:** X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, and Q. Liu, "TinyBERT: Distilling BERT for Natural Language Understanding," in *Findings of EMNLP*, 2020.
- **Methodology:** Introduces a two-stage distillation framework (general distillation on unlabelled text + task-specific distillation with data augmentation) incorporating layer-to-layer intermediate transformer matching: embedding layer cosine loss, hidden state MSE with learnable linear projection matrices, multi-head attention matrix KL-divergence, and prediction logits.
- **Difference:** TinyBERT relies heavily on heuristic task-specific data augmentation (word replacement using pre-trained GloVe/BERT) which can introduce syntactic distortion in morphologically complex Bengali text. Our study isolate whether intermediate representations ($kd\_hid$) provide measurable gains over pure logit distillation ($kd$) under controlled, unaugmented conditions across varying sample fractions.

### MiniLM (Wang et al., 2020)
- **Citation:** W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," in *NeurIPS*, 2020.
- **Methodology:** Rather than forcing intermediate hidden states to match via learnable projections, MiniLM distills the self-attention module of the teacher's last transformer layer—specifically the scaled dot-product attention distributions and relation structures between attention values.
- **Difference:** MiniLM is trained task-agnostically on massive pre-training corpora. We examine how compact students fine-tune in task-specific regimes under severe computational and annotation budgets.

---

## 3. Bengali Language Models & Benchmarks

### BanglaBERT (Bhattacharjee et al., Findings of NAACL 2022)
- **Citation:** F. Bhattacharjee, T. Hasan, K. Ahmad, K. Shahriar, N. I. Rifat, M. H. Rahman, M. A. Kabir, and R. Shahriyar, "BanglaBERT: A Large-scale Language Model for Bengali and a Comprehensive Benchmark for Bengali NLU," in *Findings of the Association for Computational Linguistics: NAACL 2022*, 2022, pp. 1094–1110.
- **Contribution:** Introduced ELECTRA-based pre-trained models for Bengali (`csebuetnlp/banglabert` Base 110M and `csebuetnlp/banglabert_small` 13.7M) trained on a cleaned 27.5GB Bengali corpus (110M sentences). Established the Bengali Language Understanding Benchmark (BLUB) across sentiment analysis, intent classification, and question answering.
- **Role in our work:** Serves as the authoritative teacher model (`banglabert` Base) and provides the compact architecture backbone (`banglabert_small`) from which our 4-layer, 7.42M parameter students inherit their weights and vocabulary.

### BanglaBait Dataset (Mahtab et al., RANLP 2023)
- **Citation:** M. A. Mahtab, M. Haque, T. Hasan, and F. Sadeque, "BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset," in *Proceedings of Recent Advances in Natural Language Processing (RANLP 2023)*, pp. 748–758, 2023.
- **Dataset Properties:** Comprises 15,056 labeled Bengali news headlines and 65,405 unlabeled headlines collected across Bangladeshi digital news outlets.
- **Role in our work:** Serves as our primary in-domain benchmark. We identify and resolve historical data hygiene issues (cross-split duplicate leakage and label conflicts) by providing a clean, MinHash-deduplicated split (`splits_v3.json`).

### BaitBuster-Bangla (Al Imran et al., Data in Brief 2024)
- **Citation:** A. Al Imran, A. R. Shovon, and M. F. Mridha, "BaitBuster-Bangla: A comprehensive dataset for clickbait detection in Bangla YouTube videos," *Data in Brief*, vol. 53, p. 110239, 2024.
- **Dataset Properties:** Features user-generated Bengali YouTube video titles and metadata, representing informal, vernacular, and sensationalist social media text with high code-mixing.
- **Role in our work:** Serves as the cross-genre evaluation benchmark to assess model robustness when transitioning from professional journalistic news headlines to user-generated video content.
