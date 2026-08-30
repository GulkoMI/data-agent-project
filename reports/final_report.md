# Final Data Project Report

## 1. Task and dataset

Binary cross-domain sentiment classification for English Amazon product and Steam game reviews loaded from two Hugging Face datasets.
Rows: **598**. Sources: `{'amazon_polarity': 300, 'steam_reviews': 298}`. Final labels: `{'negative': 316, 'positive': 282}`.

## 2. What each agent did

- DataCollectionAgent loaded and normalized two independent Hugging Face datasets.
- DataQualityAgent diagnosed missing values, duplicates, length outliers, and imbalance; two strategies were compared.
- AnnotationAgent generated sentiment predictions, confidence, a specification, Label Studio tasks, and a review queue.
- ActiveLearningAgent compared entropy, margin, and random on exactly the same split.
- TrainAgent trained TF-IDF + logistic regression and evaluated an untouched source-label holdout.

## 3. Human-in-the-loop

Verified: **True**. Reviewed rows: **30**. Changed labels: **0**.
Auto-vs-human agreement: **1.0**; Cohen's kappa: **1.0**.
Review provenance is stored per row in `reviewer` and `reviewed_at`.

## 4. Metrics

Final holdout accuracy: **0.7667**; macro F1: **0.7664**.
The final model used **166** AL-selected/reviewed rows and an untouched outer holdout of **120** rows.
Per-domain holdout metrics: `{'amazon_polarity': {'rows': 60, 'accuracy': 0.7833333333333333, 'f1_macro': 0.7818181818181819, 'confusion_matrix': [[21, 9], [4, 26]]}, 'steam_reviews': {'rows': 60, 'accuracy': 0.75, 'f1_macro': 0.7499305362600722, 'confusion_matrix': [[23, 7], [8, 22]]}}`.
Annotation agreement with source labels: **0.8879598662207357**.
Detailed stage metrics are saved under `reports/`.

## 5. Retrospective

What worked: stable IDs, isolated source failures, preserved source/auto/human/final labels, and reproducible AL splits.
Limitations: ratings are weak gold labels, domains differ, confidence is not calibrated, and AL reveals source labels only as a clearly marked simulation oracle after querying.
Next: sample more Steam titles from future dataset snapshots, calibrate probabilities, and repeat AL with labels from multiple annotators.
