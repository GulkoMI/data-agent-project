"""Active Learning agent with leakage-safe strategy comparisons."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.common import LABELS, write_json
from agents.features import assert_disjoint_groups, feature_matrix


class ActiveLearningAgent:
    """Train a text baseline and query informative examples from an unlabeled pool."""

    def __init__(
        self,
        model: str = "logreg",
        config: dict[str, Any] | None = None,
        *,
        random_seed: int = 42,
    ) -> None:
        if model != "logreg":
            raise ValueError("This reproducible baseline currently supports model='logreg'")
        self.model_name = model
        self.config = config or {}
        self.random_seed = int(self.config.get("random_seed", random_seed))
        self.pipeline: Pipeline | None = None
        self.last_query_scores: np.ndarray | None = None

    def _make_pipeline(self) -> Pipeline:
        if self.config.get("feature_column"):
            return Pipeline(
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
        tfidf = self.config.get("tfidf", {})
        ngram = tuple(tfidf.get("ngram_range", [1, 2]))
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=int(tfidf.get("max_features", 10_000)),
                        ngram_range=ngram,
                        min_df=1,
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

    def fit(self, labeled_df: pd.DataFrame, label_col: str = "label") -> Pipeline:
        self._validate_frame(labeled_df, label_col=label_col)
        if labeled_df[label_col].nunique() < 2:
            raise ValueError("Active learning requires at least two classes in labeled_df")
        self.pipeline = self._make_pipeline()
        self.pipeline.fit(self._inputs(labeled_df), labeled_df[label_col].astype(str))
        return self.pipeline

    def query(
        self,
        pool: pd.DataFrame,
        strategy: str = "entropy",
        batch_size: int = 20,
        *,
        iteration: int = 0,
    ) -> list[int]:
        if self.pipeline is None:
            raise RuntimeError("Call fit() before query()")
        if pool.empty:
            return []
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        size = min(int(batch_size), len(pool))
        strategy = strategy.lower()

        if strategy == "random":
            generator = np.random.default_rng(self.random_seed + int(iteration))
            indices = generator.choice(len(pool), size=size, replace=False)
            self.last_query_scores = np.full(len(pool), np.nan)
            return indices.tolist()

        probabilities = self.pipeline.predict_proba(self._inputs(pool))
        if strategy == "entropy":
            scores = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
            indices = np.argsort(-scores, kind="stable")[:size]
        elif strategy == "margin":
            ordered = np.sort(probabilities, axis=1)
            margins = ordered[:, -1] - ordered[:, -2]
            scores = -margins
            indices = np.argsort(margins, kind="stable")[:size]
        elif strategy in {"least_confidence", "uncertainty"}:
            scores = 1.0 - probabilities.max(axis=1)
            indices = np.argsort(-scores, kind="stable")[:size]
        else:
            raise ValueError("strategy must be entropy, margin, random, or least_confidence")

        self.last_query_scores = scores
        return indices.tolist()

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        *,
        train_label_col: str = "label",
        test_label_col: str | None = None,
    ) -> dict[str, float]:
        """Evaluate a fitted model, or fit on labeled_df first when test_df is supplied."""
        if test_df is not None:
            if self.config.get("feature_column"):
                assert_disjoint_groups(labeled_df, test_df)
            self.fit(labeled_df, label_col=train_label_col)
            evaluation = test_df
            label_col = test_label_col or train_label_col
        else:
            evaluation = labeled_df
            label_col = test_label_col or train_label_col
        if self.pipeline is None:
            raise RuntimeError("Call fit() before evaluate()")
        self._validate_frame(evaluation, label_col=label_col)
        predictions = self.pipeline.predict(self._inputs(evaluation))
        truth = evaluation[label_col].astype(str)
        return {
            "accuracy": float(accuracy_score(truth, predictions)),
            "f1_macro": float(f1_score(truth, predictions, average="macro")),
            "f1_weighted": float(f1_score(truth, predictions, average="weighted")),
        }

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
        *,
        initial_label_col: str = "final_label",
        oracle_label_col: str = "source_label",
        test_label_col: str = "source_label",
    ) -> list[dict[str, Any]]:
        """Run an AL simulation where source_label is a hidden annotation oracle.

        The pool labels are never used for feature fitting or querying. They are revealed only
        after a row has been selected. This is an experiment, not a claim of human annotation.
        """
        current_labeled = labeled_df.copy().reset_index(drop=True)
        current_pool = pool_df.copy().reset_index(drop=True)
        if self.config.get("feature_column"):
            assert_disjoint_groups(current_labeled, test_df)
            assert_disjoint_groups(current_pool, test_df)
            overlap = set(current_labeled["record_id"]) & set(current_pool["record_id"])
            if overlap:
                raise ValueError("Initial labeled data overlaps the AL pool")
        if "al_label" not in current_labeled:
            if initial_label_col not in current_labeled.columns:
                raise ValueError(f"Initial labeled data is missing {initial_label_col!r}")
            current_labeled["al_label"] = current_labeled[initial_label_col]
        history: list[dict[str, Any]] = []

        for iteration in range(int(n_iterations) + 1):
            pool_was_empty = current_pool.empty
            self.fit(current_labeled, label_col="al_label")
            metrics = self.evaluate(
                test_df, train_label_col="al_label", test_label_col=test_label_col
            )
            record: dict[str, Any] = {
                "iteration": iteration,
                "strategy": strategy,
                "n_labeled": len(current_labeled),
                "pool_remaining": len(current_pool),
                **{key: round(value, 6) for key, value in metrics.items()},
                "selected_record_ids": [],
            }

            if iteration < int(n_iterations) and not current_pool.empty:
                selected_positions = self.query(
                    current_pool,
                    strategy=strategy,
                    batch_size=batch_size,
                    iteration=iteration,
                )
                selected = current_pool.iloc[selected_positions].copy()
                selected["al_label"] = selected[oracle_label_col]
                record["selected_record_ids"] = selected["record_id"].astype(str).tolist()
                current_labeled = pd.concat([current_labeled, selected], ignore_index=True)
                current_pool = current_pool.drop(
                    current_pool.index[selected_positions]
                ).reset_index(drop=True)
            history.append(record)
            if iteration >= int(n_iterations) or pool_was_empty:
                break

        return history

    def compare_strategies(
        self,
        dataframe: pd.DataFrame,
        strategies: list[str] | tuple[str, ...] = ("entropy", "margin", "random"),
        *,
        initial_size: int = 50,
        n_iterations: int = 5,
        batch_size: int = 20,
        test_size: float = 0.2,
        initial_label_col: str = "final_label",
        oracle_label_col: str = "source_label",
    ) -> dict[str, Any]:
        """Use exactly the same stratified split, initial set, pool, and test for every strategy."""
        if self.config.get("feature_column"):
            raise ValueError(
                "For media, use run_cycle with explicit source-group-disjoint initial/pool/test "
                "frames, or the resumable media CLI for human labeling"
            )
        if not strategies:
            raise ValueError("At least one active-learning strategy is required")
        if int(initial_size) < 2:
            raise ValueError("initial_size must be at least 2")
        if int(n_iterations) < 0:
            raise ValueError("n_iterations must be non-negative")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        self._validate_frame(dataframe, label_col=oracle_label_col)
        working = dataframe.dropna(subset=["text", oracle_label_col]).copy()
        working = working[working[oracle_label_col].isin(LABELS)].reset_index(drop=True)
        working["_normalized_text"] = (
            working["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        )
        duplicate_text_rows = int(working["_normalized_text"].duplicated().sum())
        working = working.drop_duplicates("_normalized_text", keep="first").reset_index(drop=True)
        train_pool, test = train_test_split(
            working,
            test_size=float(test_size),
            random_state=self.random_seed,
            stratify=working[oracle_label_col],
        )
        train_pool = train_pool.reset_index(drop=True)
        test = test.reset_index(drop=True)

        initial_parts: list[pd.DataFrame] = []
        classes = sorted(train_pool[oracle_label_col].unique())
        per_class = int(initial_size) // len(classes)
        remainder = int(initial_size) - per_class * len(classes)
        for position, label in enumerate(classes):
            class_rows = train_pool[train_pool[oracle_label_col] == label]
            take = per_class + (1 if position < remainder else 0)
            if len(class_rows) < take:
                raise ValueError(f"Not enough '{label}' rows for initial_size={initial_size}")
            initial_parts.append(
                class_rows.sample(n=take, random_state=self.random_seed + position)
            )
        initial = pd.concat(initial_parts).sort_index().reset_index(drop=True)
        initial_ids = set(initial["record_id"].astype(str))
        pool = train_pool[~train_pool["record_id"].astype(str).isin(initial_ids)].reset_index(
            drop=True
        )

        histories: dict[str, list[dict[str, Any]]] = {}
        for strategy in strategies:
            runner = ActiveLearningAgent(
                model=self.model_name,
                config=deepcopy(self.config),
                random_seed=self.random_seed,
            )
            histories[strategy] = runner.run_cycle(
                initial.copy(),
                pool.copy(),
                test.copy(),
                strategy=strategy,
                n_iterations=n_iterations,
                batch_size=batch_size,
                initial_label_col=initial_label_col,
                oracle_label_col=oracle_label_col,
                test_label_col=oracle_label_col,
            )

        split_manifest = {
            "initial_record_ids": initial["record_id"].astype(str).tolist(),
            "pool_record_ids": pool["record_id"].astype(str).tolist(),
            "test_record_ids": test["record_id"].astype(str).tolist(),
            "random_seed": self.random_seed,
            "oracle_label_col": oracle_label_col,
            "initial_label_col": initial_label_col,
            "normalized_text_duplicates_removed": duplicate_text_rows,
            "note": "source_label is revealed only after selection; this is an AL simulation oracle",
        }
        return {"histories": histories, "split_manifest": split_manifest}

    @staticmethod
    def report(
        results: dict[str, Any], output_dir: str | Path = "reports/active_learning"
    ) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        histories = results["histories"]
        write_json(results, output / "al_results.json")

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for strategy, history in histories.items():
            points = pd.DataFrame(history)
            axes[0].plot(points["n_labeled"], points["accuracy"], marker="o", label=strategy)
            axes[1].plot(points["n_labeled"], points["f1_macro"], marker="o", label=strategy)
        axes[0].set(title="Active Learning: accuracy", xlabel="Labeled examples", ylabel="Accuracy")
        axes[1].set(title="Active Learning: macro F1", xlabel="Labeled examples", ylabel="Macro F1")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "learning_curves.png", dpi=160)
        plt.close(figure)

        random_history = histories.get("random", [])
        random_target = random_history[-1]["f1_macro"] if random_history else None
        comparison: dict[str, Any] = {"random_final_f1": random_target, "strategies": {}}
        for strategy, history in histories.items():
            final = history[-1]
            random_final_n = random_history[-1]["n_labeled"] if random_history else None
            if strategy == "random":
                reached_at = random_final_n
                savings = 0 if random_final_n is not None else None
            else:
                reached_at = None
                if random_target is not None:
                    reached_at = next(
                        (
                            point["n_labeled"]
                            for point in history
                            if point["f1_macro"] >= random_target
                        ),
                        None,
                    )
                savings = (
                    None
                    if reached_at is None or random_final_n is None
                    else int(random_final_n - reached_at)
                )
            comparison["strategies"][strategy] = {
                "final_accuracy": final["accuracy"],
                "final_f1_macro": final["f1_macro"],
                "n_to_reach_random_final_f1": reached_at,
                "saved_labels_vs_random": savings,
            }

        lines = [
            "# Active Learning Report",
            "",
            "All strategies use the same initial sample, pool, and untouched stratified test set.",
            "Pool source labels are used only as a simulation oracle after a query selects a row.",
            "",
            "| Strategy | Final accuracy | Final macro F1 | N reaching random final F1 | Saved labels |",
            "|---|---:|---:|---:|---:|",
        ]
        for strategy, values in comparison["strategies"].items():
            lines.append(
                f"| {strategy} | {values['final_accuracy']:.4f} | {values['final_f1_macro']:.4f} | "
                f"{values['n_to_reach_random_final_f1'] or 'not reached'} | "
                f"{values['saved_labels_vs_random'] if values['saved_labels_vs_random'] is not None else 'n/a'} |"
            )
        (output / "al_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(comparison, output / "strategy_comparison.json")
        return comparison

    def _inputs(self, frame: pd.DataFrame) -> Any:
        column = self.config.get("feature_column")
        return feature_matrix(frame, column) if column else frame["text"].astype(str)

    def _validate_frame(self, frame: pd.DataFrame, *, label_col: str) -> None:
        required = {"record_id", self.config.get("feature_column", "text"), label_col}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if frame.empty:
            raise ValueError("DataFrame is empty")
        if frame[label_col].isna().any():
            raise ValueError(f"Missing labels in {label_col}")


__all__ = ["ActiveLearningAgent"]
