# Reproducing EDA (Easy Data Augmentation)

COMP8240 Applications of Data Science, Macquarie University (Session 2, 2026).
Author: Vinuthan Athihalli Thammanna.

This project reproduces **Wei and Zou (2019), "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text
Classification Tasks"**, EMNLP-IJCNLP 2019 ([paper](https://aclanthology.org/D19-1670), [authors' code](https://github.com/jasonwei20/eda_nlp)).

It has three parts:
1. **Replication** of the paper's main experiment on SST-2.
2. **New data:** EDA applied to Emotion (an existing dataset) and to grocery-app reviews (built for this project).
3. **LLM-as-judge** check of whether EDA-edited texts keep their label and read fluently.

## Repository layout

| Folder | Contents |
|---|---|
| `replication/` | `eda_replication.py`, `EDA_replication_colab.ipynb`, and `results/` (raw SST-2 results and plot) |
| `data_collection/` | `grocery_dataset.py` and `Grocery_dataset_colab.ipynb`: scraping, cleaning and sampling the grocery reviews |
| `new_data/` | `eda_new_data.py`, `EDA_new_data_colab.ipynb`, `grocery_reviews_labelled.csv`, and `results_new/` |
| `judge/` | `judge_results.csv` (80 pairs with ratings) and `judge_rubric.txt` (exact method) |

## How to run (Google Colab)
- **Replication:** open `replication/EDA_replication_colab.ipynb` in Colab and choose Runtime > Run all. It clones the authors'
  code and downloads SST-2 and GloVe itself. It took roughly an hour on the default CPU runtime; a GPU runtime is faster.
- **New data:** open `new_data/EDA_new_data_colab.ipynb`, choose a GPU runtime, run all, and upload `new_data/grocery_reviews_labelled.csv`
  when asked.
- **Data collection:** `data_collection/Grocery_dataset_colab.ipynb` scrapes the newest 600 Google Play (AU) reviews for each of
  `com.woolworths` and `com.coles.android.shopmate`. Re-running it returns newer reviews, so use the labelled CSV to reproduce results.
- Package versions were not pinned (Colab defaults at the time). See `requirements.txt` for the packages used.

## Results

**1. Replication on SST-2** (CNN, 5 seeds, mean test accuracy %). EDA did not help here.

| Training sentences | Normal | With EDA | Difference |
|---|---|---|---|
| 500 | 76.8 | 76.3 | -0.5 |
| 2000 | 80.0 | 79.3 | -0.7 |
| 5000 | 80.9 | 80.4 | -0.5 |
| full | 81.2 | 81.3 | +0.1 |

The paper reports gains of about +2.1 points at 500 sentences, but as an average over five datasets; only SST-2 was tested here.
Differences from the original setup: PyTorch instead of Keras, GloVe 6B instead of 840B, the standard SST-2 split
(6,920 / 872 / 1,821), and a dev set for early stopping.

**2. EDA on new data** (mean %, paired t-test p-values are uncorrected)

Emotion (6 classes, 5 seeds):

| Training size | Metric | Normal | With EDA | Difference | p |
|---|---|---|---|---|---|
| 500 | acc | 62.9 | 64.9 | +2.1 | 0.01 |
| 500 | macro_f1 | 42.2 | 45.1 | +2.9 | 0.05 |
| 2000 | acc | 78.2 | 78.3 | +0.1 | 0.84 |
| 2000 | macro_f1 | 67.5 | 67.5 | 0.0 | 0.98 |
| 5000 | acc | 84.6 | 85.5 | +0.9 | 0.06 |
| 5000 | macro_f1 | 77.6 | 79.6 | +2.0 | 0.03 |

Grocery-app reviews (3 classes, 10 stratified splits, about 40 test reviews each):

| Training size | Metric | Normal | With EDA | Difference | p |
|---|---|---|---|---|---|
| all | acc | 69.8 | 70.5 | +0.8 | 0.56 |
| all | macro_f1 | 49.1 | 51.6 | +2.4 | 0.16 |

**3. LLM-as-judge** (80 edited pairs, 20 per operation)

| Operation | Label kept | Mean fluency (1-5) |
|---|---|---|
| random deletion | 95% | 3.35 |
| random insertion | 100% | 2.70 |
| random swap | 95% | 2.05 |
| synonym replacement | 90% | 3.10 |

On the 40 pairs also rated by a human, label agreement was 95% (Cohen's kappa 0.47; kappa is unstable because almost
all pairs were rated "kept"). See `judge/judge_rubric.txt` for the method and limitations.

## Data notes
- **Grocery reviews** are public Google Play (AU) reviews, collected with the unofficial `google-play-scraper` library. No usernames were collected.
- **Withheld reviews:** the experiments in this project used 199 labelled reviews. Five reviews that disclose health or personal
  circumstances were **withheld from this public copy** for privacy (ids 477, 514, 632, 778, 900), so
  `new_data/grocery_reviews_labelled.csv` has 194 reviews (106 negative, 58 positive, 30 mixed).
  Re-running on this file gives slightly different numbers from those above. A few retained reviews name a store location or suburb
  (a business, not a person); no reviews name individuals and no usernames were collected.
- **Columns:** `id, app, stars, date, text, label, label_name`. Label coding: 0 negative, 1 positive, 2 mixed.
- **Labels:** first-pass labels were suggested by an AI assistant that did not see the star ratings, then reviewed by the author, who changed
  5 of 199. They are not independent human annotation. Agreement with star-derived labels (1-2 stars negative, 4-5 positive, 3 stars excluded):
  91.9%, Cohen's kappa 0.84 (149 reviews).
- **Not redistributed here:** the SST-2 data (from [AcademiaSinicaNLPLab/sentiment_dataset](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset))
  and the Emotion dataset (Hugging Face `dair-ai/emotion`, Saravia et al., 2018). The notebooks download them.

## Limitations
- EDA is random; seeds are set, but GPU nondeterminism may cause small differences between runs.
- The grocery test sets are small, and the uncorrected p-values should be read as suggestive only.
- The LLM judge was a single chat session and cannot be re-run exactly.

## AI assistance
An AI assistant was used for code, analysis and drafting, and to suggest first-pass labels and act as the LLM judge. The author is responsible for the content.
