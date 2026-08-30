"""Final text model training and independent gold-holdout evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from agents.common import ensure_parent, write_json


class TrainAgent:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.random_seed = int(self.config.get("random_seed", 42))
        self.pipeline: Pipeline | None = None

    def _make_pipeline(self) -> Pipeline:
        ngram = tuple(self.config.get("ngram_range", [1, 2]))
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=int(self.config.get("max_features", 15_000)),
                        ngram_range=ngram,
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=self.random_seed,
                    ),
                ),
            ]
        )

    def train_and_evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        test_df: pd.DataFrame | None = None,
        train_label_col: str = "final_label",
        gold_label_col: str = "source_label",
        model_path: str | Path = "models/sentiment_model.joblib",
        metrics_path: str | Path = "reports/model_metrics.json",
    ) -> dict[str, Any]:
        required = {"record_id", "text", train_label_col}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        clean = dataframe.dropna(subset=["text", train_label_col]).copy()
        if test_df is None:
            if gold_label_col not in clean.columns:
                raise ValueError(f"Missing required column: {gold_label_col}")
            clean = clean.dropna(subset=[gold_label_col])
            train, test = train_test_split(
                clean,
                test_size=float(self.config.get("test_size", 0.2)),
                random_state=self.random_seed,
                stratify=clean[gold_label_col],
            )
            evaluation_mode = "internal_split"
        else:
            required_test = {"record_id", "text", gold_label_col}
            missing_test = sorted(required_test - set(test_df.columns))
            if missing_test:
                raise ValueError(f"Test DataFrame missing required columns: {missing_test}")
            train = clean
            test = test_df.dropna(subset=["text", gold_label_col]).copy()
            overlap = set(train["record_id"].astype(str)) & set(test["record_id"].astype(str))
            if overlap:
                raise ValueError(f"Train/test record_id overlap: {sorted(overlap)[:5]}")
            evaluation_mode = "explicit_outer_holdout"
        if train.empty or test.empty:
            raise ValueError("Training and test datasets must both be non-empty")
        if train[train_label_col].nunique() < 2:
            raise ValueError("Training data must contain at least two labels")
        self.pipeline = self._make_pipeline()
        self.pipeline.fit(train["text"].astype(str), train[train_label_col].astype(str))
        predictions = self.pipeline.predict(test["text"].astype(str))
        truth = test[gold_label_col].astype(str)
        by_source: dict[str, Any] = {}
        if "source" in test.columns:
            prediction_series = pd.Series(predictions, index=test.index)
            for source, source_rows in test.groupby("source", dropna=False):
                source_truth = source_rows[gold_label_col].astype(str)
                source_predictions = prediction_series.loc[source_rows.index]
                by_source[str(source)] = {
                    "rows": len(source_rows),
                    "accuracy": float(accuracy_score(source_truth, source_predictions)),
                    "f1_macro": float(f1_score(source_truth, source_predictions, average="macro")),
                    "confusion_matrix": confusion_matrix(
                        source_truth,
                        source_predictions,
                        labels=["negative", "positive"],
                    ).tolist(),
                }
        metrics: dict[str, Any] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_label_col": train_label_col,
            "test_gold_label_col": gold_label_col,
            "evaluation_mode": evaluation_mode,
            "accuracy": float(accuracy_score(truth, predictions)),
            "f1_macro": float(f1_score(truth, predictions, average="macro")),
            "f1_weighted": float(f1_score(truth, predictions, average="weighted")),
            "confusion_matrix": confusion_matrix(
                truth, predictions, labels=["negative", "positive"]
            ).tolist(),
            "classification_report": classification_report(
                truth, predictions, output_dict=True, zero_division=0
            ),
            "by_source": by_source,
            "train_record_ids": train["record_id"].astype(str).tolist(),
            "test_record_ids": test["record_id"].astype(str).tolist(),
        }
        target = ensure_parent(model_path)
        joblib.dump(self.pipeline, target)
        write_json(metrics, metrics_path)
        return metrics


__all__ = ["TrainAgent"]
