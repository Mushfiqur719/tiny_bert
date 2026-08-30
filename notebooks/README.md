# notebooks/README.md

Notebooks here are **exploratory only**.  
Nothing load-bearing (data cleaning, training, evaluation, explainability) lives in notebooks — all of that lives in `src/`.

Suggested notebooks:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | EDA on BanglaBait: label distribution, headline length, character/token stats |
| `02_tokenizer_analysis.ipynb` | Compare BanglaBERT vs TinyBanglaBERT tokenisation on sample headlines |
| `03_training_curves.ipynb` | Plot loss / F1 curves from `experiments/*/metrics.csv` |
| `04_confusion_matrices.ipynb` | Visualise per-model confusion matrices |
| `05_shap_analysis.ipynb` | Interactive SHAP beeswarm / force plots |
| `06_lime_analysis.ipynb` | LIME explanation HTML outputs |
| `07_ablation_summary.ipynb` | Side-by-side comparison table of all models (accuracy, F1, latency) |
