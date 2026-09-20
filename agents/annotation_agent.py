"""Automatic sentiment annotation, quality checks, and human-review helpers."""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from agents.common import (
    LABELS,
    normalize_label,
    read_yaml,
    stable_record_id,
    utc_now_iso,
    write_json,
)

DEFAULT_ANNOTATION_CONFIG: dict[str, Any] = {
    "backend": "transformers",
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "local_model_path": None,
    "batch_size": 16,
    "max_length": 512,
    "confidence_threshold": 0.80,
    "review_target": 30,
    "labels": list(LABELS),
    "device": "auto",
    "allow_lexicon_fallback": True,
}

_LOCAL_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
)

_POSITIVE_WORDS = frozenset(
    {
        "amazing",
        "awesome",
        "best",
        "better",
        "brilliant",
        "delightful",
        "easy",
        "enjoy",
        "enjoyable",
        "excellent",
        "fantastic",
        "fast",
        "fun",
        "good",
        "great",
        "happy",
        "impressive",
        "love",
        "loved",
        "perfect",
        "polished",
        "recommend",
        "satisfied",
        "smooth",
        "useful",
        "wonderful",
        "works",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "awful",
        "bad",
        "boring",
        "broke",
        "broken",
        "bug",
        "bugs",
        "crash",
        "crashes",
        "disappointed",
        "frustrating",
        "hate",
        "horrible",
        "poor",
        "refund",
        "returned",
        "slow",
        "terrible",
        "unplayable",
        "useless",
        "waste",
        "worst",
    }
)
_NEGATIONS = frozenset({"not", "never", "no", "hardly", "isn't", "wasn't", "don't"})
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


class _LocalModelConfigurationError(RuntimeError):
    """A configured local model exists but cannot be loaded safely."""


class AnnotationAgent:
    """Annotate English text sentiment and manage a verifiable HITL hand-off.

    A custom ``backend`` can be supplied for tests or local inference. It may be a
    callable or expose ``predict(texts, **kwargs)`` and must return one prediction
    per text. Predictions can be ``{"label": ..., "score": ...}`` mappings or
    ``(label, score)`` pairs.
    """

    @classmethod
    def for_task(
        cls,
        task: Any,
        *,
        config: str | Path | Mapping[str, Any] | None = None,
        backend: Any | None = None,
    ) -> Any:
        """Select the existing text annotator or the external visual hand-off."""
        if task.modality in {"image", "video"}:
            from agents.visual_annotation import VisualAnnotationAgent

            if backend is not None:
                raise ValueError("Visual annotation uses file responses, not a prediction backend")
            return VisualAnnotationAgent(task)
        if task.modality != "text":
            raise ValueError(f"Unsupported annotation modality: {task.modality}")
        return cls(modality="text", config=config, backend=backend)

    def __init__(
        self,
        modality: str = "text",
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        backend: Any | None = None,
    ) -> None:
        if modality != "text":
            raise NotImplementedError("This project implements text annotation only")

        self.modality = modality
        self.project_root = Path.cwd()
        supplied: Mapping[str, Any] = {}
        if isinstance(config, (str, Path)):
            config_path = Path(config).expanduser().resolve()
            supplied = read_yaml(config_path)
            self.project_root = config_path.parent
        elif config is not None:
            supplied = config

        annotation_config = supplied.get("annotation", supplied)
        self.config = {**DEFAULT_ANNOTATION_CONFIG, **dict(annotation_config)}
        self.labels = tuple(str(value).lower() for value in self.config["labels"])
        if set(self.labels) != set(LABELS):
            raise ValueError(f"This binary agent requires labels {LABELS}")

        self.backend = backend
        self.backend_name = (
            getattr(backend, "name", backend.__class__.__name__)
            if backend is not None
            else str(self.config["backend"])
        )
        self._transformer_pipeline: Any | None = None

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a labeled copy without overwriting the original source label."""
        if df.empty:
            raise ValueError("Cannot annotate an empty DataFrame")
        if "text" not in df.columns:
            raise ValueError("Input DataFrame must contain a text column")

        result = self._ensure_record_ids(df)
        if "source_label" in result.columns:
            source_values = result["source_label"]
        elif "label" in result.columns:
            source_values = result["label"]
        else:
            source_values = pd.Series([None] * len(result), index=result.index)
        result["source_label"] = source_values.map(normalize_label)
        self._validate_optional_labels(result["source_label"], "source_label")

        texts = result["text"].fillna("").astype(str).tolist()
        raw_predictions, backend_used = self._predict(texts)
        predictions = [self._coerce_prediction(value) for value in raw_predictions]
        if len(predictions) != len(result):
            raise ValueError(
                f"Annotation backend returned {len(predictions)} predictions for {len(result)} rows"
            )

        result["auto_label"] = [label for label, _ in predictions]
        result["confidence"] = [score for _, score in predictions]
        result["annotation_backend"] = backend_used

        if "human_label" not in result.columns:
            result["human_label"] = pd.Series(pd.NA, index=result.index, dtype="string")
        else:
            result["human_label"] = result["human_label"].map(normalize_label)
            self._validate_optional_labels(result["human_label"], "human_label")
        result["final_label"] = result["human_label"].fillna(result["auto_label"])
        return self.flag_for_review(result, queue_only=False)

    def flag_for_review(
        self,
        df: pd.DataFrame,
        *,
        threshold: float | None = None,
        review_target: int | None = None,
        queue_only: bool = True,
    ) -> pd.DataFrame:
        """Flag low-confidence cases, then fill the queue by uncertainty.

        Selection deliberately ignores ``source_label``. That label is retained
        for offline evaluation, but using it to select HITL records would leak
        the answer to the reviewer.
        """
        required = {"record_id", "auto_label", "confidence"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Cannot build review queue; missing columns: {missing}")

        result = self._ensure_record_ids(df)
        confidence_threshold = float(
            self.config["confidence_threshold"] if threshold is None else threshold
        )
        target = int(self.config["review_target"] if review_target is None else review_target)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        if target < 0:
            raise ValueError("review_target must be non-negative")

        confidence = pd.to_numeric(result["confidence"], errors="coerce")
        if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("confidence must contain finite values between 0 and 1")
        result["confidence"] = confidence.astype(float)

        low_confidence = confidence < confidence_threshold
        result["needs_review"] = low_confidence.astype(bool)
        result["review_reason"] = low_confidence.map({True: "low_confidence", False: ""})

        desired_size = min(target, len(result))
        missing_slots = max(0, desired_size - int(result["needs_review"].sum()))
        if missing_slots:
            candidates = result.loc[~result["needs_review"], ["record_id", "confidence"]]
            selected = (
                candidates.assign(_record_sort=candidates["record_id"].astype(str))
                .sort_values(["confidence", "_record_sort"], kind="stable")
                .head(missing_slots)
                .index
            )
            result.loc[selected, "needs_review"] = True
            result.loc[selected, "review_reason"] = "uncertainty_sample"

        if queue_only:
            return (
                result.loc[result["needs_review"]]
                .sort_values(["confidence", "record_id"], kind="stable")
                .reset_index(drop=True)
            )
        return result

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a Markdown annotation specification using dataset examples."""
        if "text" not in df.columns:
            raise ValueError("Input DataFrame must contain a text column")
        label_column = self._best_available_label_column(df)
        normalized_labels = df[label_column].map(normalize_label)
        self._validate_optional_labels(normalized_labels, label_column)

        examples: dict[str, pd.DataFrame] = {}
        for label in LABELS:
            subset = df.loc[normalized_labels.eq(label)].copy()
            if len(subset) < 3:
                raise ValueError(
                    f"At least 3 real examples for class '{label}' are required; "
                    f"found {len(subset)}"
                )
            sort_column = "record_id" if "record_id" in subset.columns else "text"
            examples[label] = subset.sort_values(sort_column, kind="stable").head(3)

        boundary = self._boundary_examples(df)
        lines = [
            "# Annotation specification",
            "",
            f"- **Task:** `{task}`",
            "- **Modality:** English review text",
            "- **Unit of annotation:** one complete product or game review",
            "- **Output:** exactly one of `negative` or `positive`",
            "",
            "## Class definitions",
            "",
            "### `negative`",
            "",
            (
                "The author's overall judgement is unfavourable: dissatisfaction, a failed "
                "experience, a warning not to buy/play, or criticism that outweighs praise."
            ),
            "",
            "### `positive`",
            "",
            (
                "The author's overall judgement is favourable: satisfaction, recommendation, "
                "enjoyment, or praise that outweighs criticism."
            ),
            "",
            "## Decision rules",
            "",
            "1. Label the author's overall conclusion, not isolated sentiment words.",
            "2. Respect negation, contrast (`but`, `although`, `yet`) and sarcasm when evident.",
            "3. For mixed reviews, use the final recommendation or the dominant judgement.",
            "4. Judge the reviewed item, not delivery/support, unless that drives the conclusion.",
            "5. If evidence is genuinely balanced, keep the auto-label but flag it for review.",
            "",
            "## Real examples from this dataset",
            "",
        ]
        for label in LABELS:
            lines.extend([f"### `{label}` examples", ""])
            for _, row in examples[label].iterrows():
                record_id = row.get("record_id", "unknown")
                lines.append(f"- `{record_id}` — {self._markdown_quote(row['text'])}")
            lines.append("")

        lines.extend(["## Boundary cases from this dataset", ""])
        for _, row in boundary.iterrows():
            record_id = row.get("record_id", "unknown")
            reason = row.get("review_reason", "mixed or uncertain sentiment")
            lines.append(
                f"- `{record_id}` ({reason or 'mixed or uncertain sentiment'}) — "
                f"{self._markdown_quote(row['text'])}"
            )
        lines.extend(
            [
                "",
                "## Human-review procedure",
                "",
                (
                    "Review every queued row in full context. Enter `negative` or `positive`; an "
                    "empty correction explicitly confirms the auto-label. Record the reviewer "
                    "name and preserve `record_id` so corrections can be validated and merged "
                    "safely."
                ),
                "",
            ]
        )

        target = self._resolve_output(output_path, "reports/annotation_spec.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        """Measure label distribution, confidence and agreement with available labels."""
        if "auto_label" not in df_labeled.columns:
            raise ValueError("Labeled DataFrame must contain auto_label")
        auto = df_labeled["auto_label"].map(self._canonical_prediction_label)
        self._validate_optional_labels(auto, "auto_label")
        distribution = auto.value_counts().reindex(LABELS, fill_value=0)

        if "confidence" in df_labeled.columns:
            confidence = pd.to_numeric(df_labeled["confidence"], errors="coerce")
            confidence_mean = None if confidence.dropna().empty else float(confidence.mean())
        else:
            confidence_mean = None

        source_comparison = self._comparison(
            auto,
            self._optional_normalized_column(df_labeled, "source_label", fallback="label"),
        )
        human_comparison = self._comparison(
            auto,
            self._optional_normalized_column(df_labeled, "human_label"),
        )
        primary = source_comparison if source_comparison["n_compared"] else human_comparison

        needs_review = (
            int(df_labeled["needs_review"].fillna(False).astype(bool).sum())
            if "needs_review" in df_labeled.columns
            else 0
        )
        backend_distribution = (
            df_labeled["annotation_backend"].fillna("unknown").astype(str).value_counts()
            if "annotation_backend" in df_labeled.columns
            else pd.Series({"unknown": len(df_labeled)})
        )
        return {
            "kappa": primary["kappa"],
            "agreement": primary["agreement"],
            "label_dist": {key: int(value) for key, value in distribution.items()},
            "label_distribution": {
                key: {
                    "count": int(distribution[key]),
                    "share": float(distribution[key] / len(auto)) if len(auto) else 0.0,
                }
                for key in LABELS
            },
            "confidence_mean": confidence_mean,
            "needs_review": needs_review,
            "backend_distribution": {
                str(key): int(value) for key, value in backend_distribution.items()
            },
            "degraded_fallback_used": any(
                "fallback" in str(key).lower() or str(key).lower() == "lexicon"
                for key in backend_distribution.index
            ),
            "against_source_label": source_comparison,
            "against_human_label": human_comparison,
        }

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Export a root JSON task array accepted by Label Studio imports."""
        required = {"record_id", "text", "auto_label", "confidence"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Cannot export Label Studio tasks; missing columns: {missing}")
        frame = self._ensure_record_ids(df)
        tasks: list[dict[str, Any]] = []
        model_version = str(self.config["model"])
        for _, row in frame.iterrows():
            label = self._canonical_prediction_label(row["auto_label"])
            score = self._coerce_score(row["confidence"])
            record_id = str(row["record_id"])
            data = {
                "record_id": record_id,
                "text": str(row["text"]),
                "source": self._json_scalar(row.get("source")),
            }
            tasks.append(
                {
                    "data": data,
                    "predictions": [
                        {
                            "model_version": model_version,
                            "score": score,
                            "result": [
                                {
                                    "id": f"sentiment-{record_id}",
                                    "from_name": "sentiment",
                                    "to_name": "review_text",
                                    "type": "choices",
                                    "value": {"choices": [label]},
                                }
                            ],
                        }
                    ],
                }
            )

        target = self._resolve_output(output_path, "data/review/labelstudio_import.json")
        write_json(tasks, target)
        return target

    def apply_human_review(
        self,
        df: pd.DataFrame,
        corrections: pd.DataFrame | str | Path,
        *,
        reviewer: str | None = None,
        reviewed_at: str | None = None,
    ) -> pd.DataFrame:
        """Validate and merge explicit human review into the labeled dataset.

        ``corrected_label`` takes precedence over ``human_label``. A blank correction
        explicitly confirms ``auto_label``. A non-empty reviewer is required so an
        automated or simulated pass cannot silently be represented as human review.
        """
        base = self._ensure_record_ids(df)
        if "auto_label" not in base.columns:
            raise ValueError("Labeled DataFrame must contain auto_label")
        if base["record_id"].duplicated().any():
            raise ValueError("Base DataFrame contains duplicate record_id values")

        if isinstance(corrections, (str, Path)):
            review = pd.read_csv(corrections, dtype={"record_id": "string"})
        else:
            review = corrections.copy()
        if "record_id" not in review.columns:
            raise ValueError("Corrections must contain record_id")
        review["record_id"] = review["record_id"].astype("string")
        if review["record_id"].isna().any() or review["record_id"].str.strip().eq("").any():
            raise ValueError("Corrections contain an empty record_id")
        if review["record_id"].duplicated().any():
            duplicates = review.loc[review["record_id"].duplicated(), "record_id"].tolist()
            raise ValueError(f"Corrections contain duplicate record_id values: {duplicates}")

        known_ids = set(base["record_id"].astype(str))
        unknown_ids = sorted(set(review["record_id"].astype(str)) - known_ids)
        if unknown_ids:
            raise ValueError(f"Corrections contain unknown record_id values: {unknown_ids[:5]}")
        if "corrected_label" in review.columns:
            raw_corrections = review["corrected_label"]
            if "human_label" in review.columns:
                raw_corrections = raw_corrections.where(
                    ~raw_corrections.map(self._is_blank), review["human_label"]
                )
        elif "human_label" in review.columns:
            raw_corrections = review["human_label"]
        else:
            raise ValueError("Corrections must contain corrected_label or human_label")

        normalized = raw_corrections.map(self._normalize_review_label)
        invalid_mask = normalized.notna() & ~normalized.isin(LABELS)
        if invalid_mask.any():
            invalid = sorted(set(normalized.loc[invalid_mask].astype(str)))
            raise ValueError(f"Unsupported correction labels: {invalid}")

        global_reviewer = (reviewer or "").strip()
        if "reviewer" in review.columns:
            row_reviewers = review["reviewer"].fillna("").astype(str).str.strip()
            if global_reviewer:
                row_reviewers = row_reviewers.mask(row_reviewers.eq(""), global_reviewer)
        else:
            row_reviewers = pd.Series(global_reviewer, index=review.index, dtype="string")
        if row_reviewers.eq("").any():
            raise ValueError("A non-empty reviewer is required for every correction")

        if "reviewed_at" in review.columns:
            timestamps = review["reviewed_at"].fillna("").astype(str).str.strip()
        else:
            timestamps = pd.Series("", index=review.index, dtype="string")
        fallback_timestamp = reviewed_at or utc_now_iso()
        timestamps = timestamps.mask(timestamps.eq(""), fallback_timestamp)

        if "human_label" not in base.columns:
            base["human_label"] = pd.Series(pd.NA, index=base.index, dtype="string")
        if "reviewer" not in base.columns:
            base["reviewer"] = pd.Series(pd.NA, index=base.index, dtype="string")
        if "reviewed_at" not in base.columns:
            base["reviewed_at"] = pd.Series(pd.NA, index=base.index, dtype="string")
        if "needs_review" not in base.columns:
            base["needs_review"] = False
        if "review_reason" not in base.columns:
            base["review_reason"] = ""

        lookup = base.reset_index().set_index(base["record_id"].astype(str))["index"].to_dict()
        for position, (_, correction_row) in enumerate(review.iterrows()):
            record_id = str(correction_row["record_id"])
            base_index = lookup[record_id]
            label = normalized.iloc[position]
            if label is None or pd.isna(label):
                label = self._canonical_prediction_label(base.at[base_index, "auto_label"])
            base.at[base_index, "human_label"] = label
            base.at[base_index, "reviewer"] = row_reviewers.iloc[position]
            base.at[base_index, "reviewed_at"] = timestamps.iloc[position]
            base.at[base_index, "needs_review"] = False

        base["final_label"] = (
            base["human_label"]
            .map(normalize_label)
            .fillna(base["auto_label"].map(self._canonical_prediction_label))
        )
        base["review_changed"] = base["human_label"].notna() & base["human_label"].map(
            normalize_label
        ).ne(base["auto_label"].map(self._canonical_prediction_label))
        return base

    def _predict(self, texts: list[str]) -> tuple[Sequence[Any], str]:
        if self.backend is not None:
            return self._call_backend(self.backend, texts), str(self.backend_name)
        if str(self.config["backend"]).lower() == "lexicon":
            return self._lexicon_predict(texts), "lexicon"
        try:
            return self._predict_with_transformers(texts), f"transformers:{self.config['model']}"
        except _LocalModelConfigurationError:
            # A partial local snapshot is a configuration error. Falling back here
            # would hide the broken snapshot and could unexpectedly access the network.
            raise
        except Exception as exc:
            if not bool(self.config.get("allow_lexicon_fallback", True)):
                raise RuntimeError(
                    "Transformer annotation failed and fallback is disabled"
                ) from exc
            warnings.warn(
                f"Transformer annotation unavailable ({exc!s}); using deterministic lexicon fallback",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._lexicon_predict(texts), "lexicon_fallback"

    def _call_backend(self, backend: Any, texts: list[str]) -> Sequence[Any]:
        predictor: Callable[..., Sequence[Any]]
        predictor = backend.predict if hasattr(backend, "predict") else backend
        try:
            return predictor(
                texts,
                batch_size=int(self.config["batch_size"]),
                max_length=int(self.config["max_length"]),
            )
        except TypeError:
            return predictor(texts)

    def _predict_with_transformers(self, texts: list[str]) -> Sequence[Any]:
        if self._transformer_pipeline is None:
            model_source, use_local_model = self._resolve_transformer_model()

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            device_setting = str(self.config.get("device", "auto")).lower()
            if device_setting == "auto":
                if torch.backends.mps.is_available():
                    device: Any = torch.device("mps")
                elif torch.cuda.is_available():
                    device = 0
                else:
                    device = -1
            elif device_setting == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS was requested but is not available")
                device = torch.device("mps")
            elif device_setting == "cpu":
                device = -1
            elif device_setting == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is not available")
                device = 0
            else:
                device = device_setting
            if use_local_model:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    local_files_only=True,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    local_files_only=True,
                )
                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
            else:
                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_source,
                    tokenizer=model_source,
                    device=device,
                )
        return self._transformer_pipeline(
            texts,
            batch_size=int(self.config["batch_size"]),
            truncation=True,
            max_length=int(self.config["max_length"]),
        )

    def _resolve_transformer_model(self) -> tuple[str, bool]:
        """Resolve a complete local snapshot, otherwise retain the logical HF ID.

        A missing or empty configured directory means that no local snapshot has
        been supplied yet. Once any model file is present, the directory must be
        complete; silently mixing it with a remote download would make inference
        non-reproducible.
        """
        remote_model = str(self.config["model"])
        configured_path = self.config.get("local_model_path")
        if configured_path is None or not str(configured_path).strip():
            return remote_model, False

        local_path = Path(str(configured_path)).expanduser()
        if not local_path.is_absolute():
            local_path = self.project_root / local_path
        local_path = local_path.resolve()

        if not local_path.exists():
            return remote_model, False
        if not local_path.is_dir():
            raise _LocalModelConfigurationError(
                "Configured local model path must be a directory"
            )

        visible_entries = [entry for entry in local_path.iterdir() if not entry.name.startswith(".")]
        if not visible_entries:
            return remote_model, False

        missing = [
            filename
            for filename in _LOCAL_MODEL_REQUIRED_FILES
            if not (local_path / filename).is_file()
            or (local_path / filename).stat().st_size == 0
        ]
        if missing:
            raise _LocalModelConfigurationError(
                "Configured local model directory is incomplete; missing files: "
                + ", ".join(missing)
            )
        return str(local_path), True

    def _lexicon_predict(self, texts: list[str]) -> list[dict[str, Any]]:
        predictions = []
        for text in texts:
            tokens = _TOKEN_RE.findall(text.lower())
            score = 0
            for index, token in enumerate(tokens):
                polarity = int(token in _POSITIVE_WORDS) - int(token in _NEGATIVE_WORDS)
                if polarity and any(
                    previous in _NEGATIONS for previous in tokens[max(0, index - 3) : index]
                ):
                    polarity *= -1
                score += polarity
            label = "positive" if score >= 0 else "negative"
            # Neutral/weak lexicon matches stay deliberately uncertain for human review.
            confidence = 0.5 if score == 0 else min(0.95, 0.58 + 0.10 * abs(score))
            predictions.append({"label": label, "score": confidence})
        return predictions

    def _ensure_record_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "record_id" not in result.columns:
            result["record_id"] = pd.NA
        if "source" not in result.columns:
            result["source"] = "unknown"
        if "source_id" not in result.columns:
            result["source_id"] = pd.NA
        for ordinal, index in enumerate(result.index):
            current = result.at[index, "record_id"]
            if self._is_blank(current):
                source_id = result.at[index, "source_id"]
                if self._is_blank(source_id):
                    source_id = ordinal
                result.at[index, "record_id"] = stable_record_id(
                    result.at[index, "source"], source_id, result.at[index, "text"]
                )
        result["record_id"] = result["record_id"].astype(str)
        if result["record_id"].duplicated().any():
            duplicate_ids = result.loc[
                result["record_id"].duplicated(keep=False), "record_id"
            ].tolist()
            raise ValueError(f"record_id must be unique; duplicates include {duplicate_ids[:5]}")
        return result

    def _coerce_prediction(self, value: Any) -> tuple[str, float]:
        if isinstance(value, Mapping):
            label = value.get("label")
            score = value.get("score", value.get("confidence"))
        elif (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2
        ):
            label, score = value
        else:
            raise ValueError(f"Unsupported prediction format: {value!r}")
        return self._canonical_prediction_label(label), self._coerce_score(score)

    @staticmethod
    def _canonical_prediction_label(value: Any) -> str:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            raise ValueError("Prediction label cannot be empty")
        raw = str(value).strip().lower()
        mapping = {
            "label_0": "negative",
            "label_1": "positive",
            "neg": "negative",
            "negative": "negative",
            "pos": "positive",
            "positive": "positive",
        }
        label = mapping.get(raw)
        if label is None:
            raise ValueError(f"Unsupported prediction label: {value!r}")
        return label

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence score: {value!r}") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1; received {score}")
        return score

    @staticmethod
    def _validate_optional_labels(values: pd.Series, column: str) -> None:
        invalid = sorted(set(values.dropna().astype(str)) - set(LABELS))
        if invalid:
            raise ValueError(f"Unsupported labels in {column}: {invalid}")

    def _optional_normalized_column(
        self, df: pd.DataFrame, column: str, fallback: str | None = None
    ) -> pd.Series:
        selected = column if column in df.columns else fallback
        if selected is None or selected not in df.columns:
            return pd.Series([None] * len(df), index=df.index, dtype="object")
        values = df[selected].map(normalize_label)
        self._validate_optional_labels(values, selected)
        return values

    @staticmethod
    def _comparison(predicted: pd.Series, reference: pd.Series) -> dict[str, Any]:
        comparable = predicted.notna() & reference.notna()
        left = predicted.loc[comparable].astype(str)
        right = reference.loc[comparable].astype(str)
        n_compared = int(comparable.sum())
        if n_compared == 0:
            return {"n_compared": 0, "agreement": None, "kappa": None}
        agreement = float(left.eq(right).mean())
        left_counts = Counter(left)
        right_counts = Counter(right)
        expected = sum(
            (left_counts[label] / n_compared) * (right_counts[label] / n_compared)
            for label in LABELS
        )
        if math.isclose(expected, 1.0):
            kappa = 1.0 if math.isclose(agreement, 1.0) else 0.0
        else:
            kappa = (agreement - expected) / (1.0 - expected)
        return {
            "n_compared": n_compared,
            "agreement": agreement,
            "kappa": float(kappa),
        }

    @staticmethod
    def _best_available_label_column(df: pd.DataFrame) -> str:
        for column in ("human_label", "source_label", "final_label", "auto_label", "label"):
            if column in df.columns:
                normalized = df[column].map(normalize_label)
                counts = normalized.value_counts()
                if all(int(counts.get(label, 0)) >= 3 for label in LABELS):
                    return column
        raise ValueError("No label column is available for annotation examples")

    @staticmethod
    def _markdown_quote(value: Any, limit: int = 280) -> str:
        text = " ".join(str(value).split()).replace("`", "'")
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return f"“{text}”"

    @staticmethod
    def _boundary_examples(df: pd.DataFrame) -> pd.DataFrame:
        candidates = df.copy()
        if "needs_review" in candidates.columns:
            reviewed = candidates["needs_review"].fillna(False).astype(bool)
            if reviewed.any():
                candidates = candidates.loc[reviewed]
        if "confidence" in candidates.columns:
            candidates = candidates.assign(
                _confidence=pd.to_numeric(candidates["confidence"], errors="coerce").fillna(1.0)
            ).sort_values("_confidence", kind="stable")
        if candidates.empty:
            candidates = df.head(1)
        return candidates.head(5)

    def _resolve_output(self, output_path: str | Path | None, default: str) -> Path:
        if output_path is None:
            return self.project_root / default
        target = Path(output_path).expanduser()
        return target if target.is_absolute() else self.project_root / target

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return None
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            return False
        return isinstance(value, str) and not value.strip()

    @classmethod
    def _normalize_review_label(cls, value: Any) -> str | None:
        if cls._is_blank(value):
            return None
        raw = str(value).strip().lower()
        mapping = {
            "0": "negative",
            "1": "positive",
            "neg": "negative",
            "negative": "negative",
            "pos": "positive",
            "positive": "positive",
        }
        return mapping.get(raw, raw)
