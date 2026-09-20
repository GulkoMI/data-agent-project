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
from sklearn.preprocessing import StandardScaler

from agents.common import ensure_parent, write_json
from agents.features import assert_disjoint_groups, feature_matrix


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
        if self.config.get("feature_column"):
            return self.train_media(
                dataframe,
                test_df=test_df,
                train_label_col=train_label_col,
                reference_label_col=gold_label_col,
                model_path=model_path,
                metrics_path=metrics_path,
            )
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

    def train_media(
        self,
        dataframe: pd.DataFrame,
        *,
        test_df: pd.DataFrame | None = None,
        train_label_col: str = "final_label",
        reference_label_col: str = "source_label",
        model_path: str | Path = "models/visual_model.joblib",
        metrics_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fit visual vectors; evaluate only when a separate reference set is supplied.

        Reference labels may originate from a local manifest. Their presence is
        not evidence that a human independently annotated them in this run.
        """
        if train_label_col not in dataframe:
            raise ValueError(f"Missing training labels: {train_label_col}")
        train = dataframe.dropna(subset=[train_label_col]).copy()
        if train.empty or train[train_label_col].nunique() < 2:
            raise ValueError("Visual training requires verified samples from at least two classes")
        column = self.config.get("feature_column", "features")
        labels = list(
            self.config.get("labels", sorted(train[train_label_col].astype(str).unique()))
        )
        if not set(train[train_label_col].astype(str)) <= set(labels):
            raise ValueError("Training labels do not match the task classes")
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=self.random_seed
                    ),
                ),
            ]
        )
        test = None
        if test_df is not None and not test_df.empty:
            assert_disjoint_groups(train, test_df)
            if reference_label_col not in test_df or test_df[reference_label_col].isna().any():
                raise ValueError("Every supplied test row requires a reference label")
            test = test_df
            if not set(test[reference_label_col].astype(str)) <= set(labels):
                raise ValueError("Reference labels do not match the task classes")
        self.pipeline.fit(feature_matrix(train, column), train[train_label_col].astype(str))
        metrics: dict[str, Any] = {
            "train_rows": len(train),
            "test_rows": 0 if test is None else len(test),
            "classes": labels,
            "train_label_col": train_label_col,
            "reference_label_col": reference_label_col if test is not None else None,
            "evaluation_mode": "group_disjoint_reference" if test is not None else "not_evaluated",
            "train_record_ids": train["record_id"].astype(str).tolist(),
            "test_record_ids": [] if test is None else test["record_id"].astype(str).tolist(),
            "features": self.config.get("features", {"column": column}),
        }
        if test is not None:
            truth = test[reference_label_col].astype(str)
            predictions = self.pipeline.predict(feature_matrix(test, column))
            metrics.update(
                accuracy=float(accuracy_score(truth, predictions)),
                f1_macro=float(
                    f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
                ),
                confusion_matrix=confusion_matrix(truth, predictions, labels=labels).tolist(),
                classification_report=classification_report(
                    truth, predictions, labels=labels, output_dict=True, zero_division=0
                ),
            )
        target = ensure_parent(model_path)
        temporary = target.with_suffix(".tmp")
        joblib.dump(self.pipeline, temporary)
        temporary.replace(target)
        write_json(metrics, target.with_suffix(".metadata.json"))
        if metrics_path is not None:
            write_json(metrics, metrics_path)
        return metrics


__all__ = ["TrainAgent"]
