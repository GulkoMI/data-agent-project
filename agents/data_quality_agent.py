"""Detection, repair, and reporting of review-data quality problems.

The agent deliberately treats labels as read-only metadata.  Cleaning may remove
whole records, but it never imputes, normalizes, or overwrites ``label`` or
``source_label`` values.  This distinction matters later in the pipeline, where
source labels are used as an evaluation reference rather than as auto-labels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.switch_backend("Agg")


DEFAULT_STRATEGIES: dict[str, dict[str, str]] = {
    "conservative": {
        "missing": "drop_required",
        "duplicates": "drop",
        "outliers": "clip_iqr",
    },
    "strict": {
        "missing": "drop_required",
        "duplicates": "drop",
        "outliers": "drop_iqr",
    },
}


class DataQualityAgent:
    """Find and fix deterministic data-quality issues in a text dataset.

    Parameters
    ----------
    config:
        A project YAML path, a project/quality configuration mapping, or ``None``.
        When a project YAML path is supplied, reports default to
        ``<project>/reports/quality``.
    report_dir:
        Optional explicit artifact directory.
    text_column / label_column:
        Column names used for length diagnostics and class balance.  If the
        configured label is absent, ``source_label`` and then ``label`` are tried.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        report_dir: str | Path | None = None,
        text_column: str = "text",
        label_column: str | None = None,
    ) -> None:
        raw_config: dict[str, Any] = {}
        project_root = Path.cwd()
        if isinstance(config, (str, Path)):
            config_path = Path(config).expanduser().resolve()
            with config_path.open("r", encoding="utf-8") as handle:
                raw_config = yaml.safe_load(handle) or {}
            project_root = config_path.parent
        elif config is not None:
            raw_config = dict(config)

        quality_config = raw_config.get("quality", raw_config)
        self.config: dict[str, Any] = dict(quality_config or {})
        self.text_column = str(self.config.get("text_column", text_column))
        self.label_column = self.config.get("target_column", label_column)
        self.iqr_multiplier = float(self.config.get("outlier", {}).get("iqr_multiplier", 1.5))
        self.z_threshold = float(self.config.get("outlier", {}).get("z_threshold", 3.0))
        self.imbalance_threshold = float(self.config.get("imbalance_threshold", 0.5))

        self.strategies = {name: values.copy() for name, values in DEFAULT_STRATEGIES.items()}
        for name, values in self.config.get("strategies", {}).items():
            self.strategies[str(name)] = {**self.strategies.get(str(name), {}), **values}

        configured_report_dir = self.config.get("report_dir")
        chosen_report_dir = report_dir or configured_report_dir or "reports/quality"
        chosen_path = Path(chosen_report_dir).expanduser()
        self.report_dir = chosen_path if chosen_path.is_absolute() else project_root / chosen_path

    # ------------------------------------------------------------------ detection
    def detect_media_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        """Inspect visual asset rows without requiring text or source labels."""
        from agents.media_quality import MediaQualityAgent

        return MediaQualityAgent(self.config).detect_issues(df)

    def fix_media(
        self, df: pd.DataFrame, strategy: str | Mapping[str, Any] = "conservative",
    ) -> pd.DataFrame:
        """Clean visual assets; retained labels and original media stay untouched."""
        from agents.media_quality import MediaQualityAgent

        return MediaQualityAgent(self.config).fix(df, strategy)

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return a JSON-serializable quality report without mutating ``df``."""
        self._validate_frame(df)

        missing_by_column = {column: int(df[column].isna().sum()) for column in df.columns}
        rows_with_missing = int(df.isna().any(axis=1).sum()) if len(df.columns) else 0

        empty_mask = self._empty_text_mask(df)
        duplicate_mask = df.duplicated(keep="first")
        normalized_text = self._normalized_texts(df)
        text_duplicate_mask = normalized_text.ne("") & normalized_text.duplicated(keep="first")
        lengths = self._text_lengths(df)
        iqr = self._iqr_details(lengths)
        zscore = self._zscore_details(lengths)

        primary_label = self._resolve_label_column(df)
        imbalance = self._imbalance_details(df, primary_label)
        target_distributions = {
            column: self._distribution(df[column]) for column in self._target_columns(df)
        }

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            # Keep these top-level values simple for the assignment's public contract.
            "missing": missing_by_column,
            "missing_total": int(sum(missing_by_column.values())),
            "rows_with_missing": rows_with_missing,
            "empty_text": int(empty_mask.sum()),
            "empty_text_indices": self._index_values(df.index[empty_mask]),
            "duplicates": int(duplicate_mask.sum()),
            "duplicate_indices": self._index_values(df.index[duplicate_mask]),
            "text_duplicates": int(text_duplicate_mask.sum()),
            "text_duplicate_indices": self._index_values(df.index[text_duplicate_mask]),
            "outliers": {"iqr": iqr, "zscore": zscore},
            "imbalance": imbalance,
            "target_distributions": target_distributions,
            "text_length": self._length_summary(lengths),
        }

    # --------------------------------------------------------------------- repair
    def fix(
        self,
        df: pd.DataFrame,
        strategy: str | Mapping[str, Any] = "conservative",
    ) -> pd.DataFrame:
        """Return a cleaned copy according to a named or custom strategy.

        Supported actions are:

        - ``missing``: ``drop_required``, ``drop_all``, or ``keep``;
        - ``duplicates``: ``drop`` or ``keep``;
        - ``outliers``: ``clip_iqr``/``truncate_iqr``, ``drop_iqr``, or ``keep``.

        Named ``conservative`` and ``strict`` strategies are loaded from the
        quality section of ``config.yaml``.  A mapping can override the actions
        directly and may include ``required_columns``.
        """
        self._validate_frame(df)
        resolved, strategy_name = self._resolve_strategy(strategy)
        result = df.copy(deep=True)
        original_target_dtypes = {
            column: result[column].dtype for column in self._target_columns(result)
        }
        actions: dict[str, Any] = {"strategy": strategy_name, "input_rows": len(result)}

        missing_action = self._action_name(resolved.get("missing", "keep"))
        if missing_action == "drop_required":
            required = self._required_columns(result, resolved)
            invalid = pd.Series(False, index=result.index)
            for column in required:
                invalid |= result[column].isna()
                if self._is_string_like(result[column]):
                    invalid |= result[column].astype("string").fillna("").str.strip().eq("")
            actions["missing_rows_removed"] = int(invalid.sum())
            actions["required_columns"] = required
            result = result.loc[~invalid].copy()
        elif missing_action == "drop_all":
            invalid = result.isna().any(axis=1)
            if self.text_column in result.columns:
                invalid |= self._empty_text_mask(result)
            actions["missing_rows_removed"] = int(invalid.sum())
            result = result.loc[~invalid].copy()
        elif missing_action in {"keep", "none"}:
            actions["missing_rows_removed"] = 0
        else:
            raise ValueError(f"Unsupported missing-value action: {missing_action!r}")

        duplicates_action = self._action_name(resolved.get("duplicates", "keep"))
        if duplicates_action == "drop":
            if self.text_column in result.columns:
                normalized_text = self._normalized_texts(result)
                duplicate_mask = normalized_text.ne("") & normalized_text.duplicated(keep="first")
                actions["duplicate_basis"] = "normalized_text"
            else:
                duplicate_mask = result.duplicated(keep="first")
                actions["duplicate_basis"] = "full_row"
            actions["duplicate_rows_removed"] = int(duplicate_mask.sum())
            result = result.loc[~duplicate_mask].copy()
        elif duplicates_action in {"keep", "none"}:
            actions["duplicate_rows_removed"] = 0
        else:
            raise ValueError(f"Unsupported duplicate action: {duplicates_action!r}")

        outlier_action = self._action_name(resolved.get("outliers", "keep"))
        actions["outlier_rows_removed"] = 0
        actions["texts_truncated"] = 0
        if outlier_action in {"clip_iqr", "truncate_iqr", "truncate_long"}:
            if self.text_column in result.columns and not result.empty:
                lengths = self._text_lengths(result)
                details = self._iqr_details(lengths)
                upper = details["upper_bound"]
                if upper is not None:
                    max_chars = max(1, math.floor(float(upper)))
                    truncate_mask = lengths > max_chars
                    actions["texts_truncated"] = int(truncate_mask.sum())
                    if truncate_mask.any():
                        result.loc[truncate_mask, self.text_column] = (
                            result.loc[truncate_mask, self.text_column]
                            .astype(str)
                            .str.slice(stop=max_chars)
                        )
                    actions["truncate_at_chars"] = max_chars
        elif outlier_action == "drop_iqr":
            if self.text_column in result.columns and not result.empty:
                lengths = self._text_lengths(result)
                details = self._iqr_details(lengths)
                lower, upper = details["lower_bound"], details["upper_bound"]
                if lower is not None and upper is not None:
                    outlier_mask = (lengths < float(lower)) | (lengths > float(upper))
                    # Missing/blank values not dropped by a custom strategy are not
                    # silently reclassified as statistical outliers.
                    outlier_mask &= lengths.notna()
                    actions["outlier_rows_removed"] = int(outlier_mask.sum())
                    result = result.loc[~outlier_mask].copy()
        elif outlier_action in {"keep", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported outlier action: {outlier_action!r}")

        # Cleaning is allowed to select rows, never to reinterpret target values.
        for column, dtype in original_target_dtypes.items():
            if column in result.columns and result[column].dtype != dtype:
                result[column] = result[column].astype(dtype)

        actions["output_rows"] = len(result)
        actions["rows_removed"] = len(df) - len(result)
        result = result.reset_index(drop=True)
        result.attrs["quality_actions"] = actions
        result.attrs["quality_strategy"] = dict(resolved)
        return result

    # ---------------------------------------------------------------- comparison
    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """Build the required before/after quality table."""
        before = self.detect_issues(df_before)
        after = self.detect_issues(df_after)

        metrics: list[tuple[str, float, float, str]] = [
            ("rows", before["row_count"], after["row_count"], "neutral"),
            ("missing_cells", before["missing_total"], after["missing_total"], "lower"),
            (
                "rows_with_missing",
                before["rows_with_missing"],
                after["rows_with_missing"],
                "lower",
            ),
            ("empty_text", before["empty_text"], after["empty_text"], "lower"),
            ("exact_duplicates", before["duplicates"], after["duplicates"], "lower"),
            (
                "normalized_text_duplicates",
                before["text_duplicates"],
                after["text_duplicates"],
                "lower",
            ),
            (
                "text_length_iqr_outliers",
                before["outliers"]["iqr"]["count"],
                after["outliers"]["iqr"]["count"],
                "lower",
            ),
            (
                "text_length_zscore_outliers",
                before["outliers"]["zscore"]["count"],
                after["outliers"]["zscore"]["count"],
                "lower",
            ),
        ]

        before_ratio = before["imbalance"].get("minority_to_majority_ratio")
        after_ratio = after["imbalance"].get("minority_to_majority_ratio")
        if before_ratio is not None or after_ratio is not None:
            metrics.append(
                (
                    "class_balance_ratio",
                    float(before_ratio or 0.0),
                    float(after_ratio or 0.0),
                    "higher",
                )
            )

        rows: list[dict[str, Any]] = []
        for metric, old, new, direction in metrics:
            old_value, new_value = float(old), float(new)
            change = new_value - old_value
            if direction == "lower":
                improved: bool | None = new_value < old_value
            elif direction == "higher":
                improved = new_value > old_value
            else:
                improved = None
            rows.append(
                {
                    "metric": metric,
                    "before": self._clean_number(old_value),
                    "after": self._clean_number(new_value),
                    "change": self._clean_number(change),
                    "improved": improved,
                }
            )

        primary = before["imbalance"].get("label_column") or after["imbalance"].get("label_column")
        if primary:
            before_counts = before["target_distributions"].get(primary, {}).get("counts", {})
            after_counts = after["target_distributions"].get(primary, {}).get("counts", {})
            before_props = before["target_distributions"].get(primary, {}).get("proportions", {})
            after_props = after["target_distributions"].get(primary, {}).get("proportions", {})
            for label in sorted(set(before_counts) | set(after_counts)):
                old_count, new_count = before_counts.get(label, 0), after_counts.get(label, 0)
                rows.append(
                    {
                        "metric": f"target_count:{label}",
                        "before": int(old_count),
                        "after": int(new_count),
                        "change": int(new_count - old_count),
                        "improved": None,
                    }
                )
                old_share = float(before_props.get(label, 0.0))
                new_share = float(after_props.get(label, 0.0))
                rows.append(
                    {
                        "metric": f"target_share:{label}",
                        "before": old_share,
                        "after": new_share,
                        "change": new_share - old_share,
                        "improved": None,
                    }
                )

        return pd.DataFrame(rows, columns=["metric", "before", "after", "change", "improved"])

    def evaluate_strategies(
        self,
        df: pd.DataFrame,
        strategies: Sequence[str | Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple strategies on the same immutable input.

        The returned mapping contains each cleaned frame, its issue report, and
        the before/after comparison.  This makes it possible to inspect both the
        headline table and individual records without running cleaning twice.
        """
        candidates = list(strategies or ("conservative", "strict"))
        results: dict[str, dict[str, Any]] = {}
        for position, strategy in enumerate(candidates, start=1):
            if isinstance(strategy, str):
                name = strategy
            else:
                name = str(strategy.get("name", f"custom_{position}"))
            cleaned = self.fix(df, strategy)
            results[name] = {
                "dataframe": cleaned,
                "issues": self.detect_issues(cleaned),
                "comparison": self.compare(df, cleaned),
                "actions": dict(cleaned.attrs.get("quality_actions", {})),
            }
        return results

    # ------------------------------------------------------------- report/artifacts
    def generate_visualizations(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Save one diagnostic plot per required quality problem."""
        self._validate_frame(df_before)
        if df_after is not None:
            self._validate_frame(df_after)
        target = Path(output_dir) if output_dir is not None else self.report_dir
        target.mkdir(parents=True, exist_ok=True)
        after = df_after if df_after is not None else df_before
        before_report = self.detect_issues(df_before)
        after_report = self.detect_issues(after)
        paths: dict[str, Path] = {}

        # Missing values by column.
        columns = list(dict.fromkeys([*df_before.columns, *after.columns]))
        before_missing = [before_report["missing"].get(column, 0) for column in columns]
        after_missing = [after_report["missing"].get(column, 0) for column in columns]
        fig, ax = plt.subplots(figsize=(max(7, len(columns) * 0.75), 4.5))
        x = np.arange(len(columns))
        width = 0.38
        ax.bar(x - width / 2, before_missing, width, label="Before", color="#d95f5f")
        ax.bar(x + width / 2, after_missing, width, label="After", color="#4c956c")
        ax.set(title="Missing values by column", ylabel="Missing cells")
        ax.set_xticks(x, columns, rotation=45, ha="right")
        ax.legend()
        paths["missing_values"] = self._save_figure(fig, target / "missing_values.png")

        # Blank strings are a separate text-specific problem.
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(
            ["Before", "After"],
            [before_report["empty_text"], after_report["empty_text"]],
            color=["#d95f5f", "#4c956c"],
        )
        ax.bar_label(bars)
        ax.set(title="Empty review text", ylabel="Rows")
        paths["empty_text"] = self._save_figure(fig, target / "empty_text.png")

        # Exact duplicate rows.
        fig, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(
            ["Before", "After"],
            [before_report["text_duplicates"], after_report["text_duplicates"]],
            color=["#d95f5f", "#4c956c"],
        )
        ax.bar_label(bars)
        ax.set(title="Normalized review-text duplicates", ylabel="Duplicates after first")
        paths["duplicates"] = self._save_figure(fig, target / "duplicates.png")

        # Text length distributions and the original IQR fences.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        before_lengths = self._text_lengths(df_before).dropna()
        after_lengths = self._text_lengths(after).dropna()
        if not before_lengths.empty:
            ax.hist(
                before_lengths, bins=self._hist_bins(before_lengths), alpha=0.55, label="Before"
            )
        if not after_lengths.empty:
            ax.hist(after_lengths, bins=self._hist_bins(after_lengths), alpha=0.55, label="After")
        bounds = before_report["outliers"]["iqr"]
        if bounds["lower_bound"] is not None:
            ax.axvline(bounds["lower_bound"], color="#d97706", linestyle="--", label="IQR fences")
            ax.axvline(bounds["upper_bound"], color="#d97706", linestyle="--")
        ax.set(title="Review-length outliers", xlabel="Characters", ylabel="Rows")
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        paths["outliers"] = self._save_figure(fig, target / "text_length_outliers.png")

        # Class distribution.
        label_column = self._resolve_label_column(df_before) or self._resolve_label_column(after)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        if label_column:
            before_counts = (
                self._distribution(df_before[label_column])["counts"]
                if label_column in df_before
                else {}
            )
            after_counts = (
                self._distribution(after[label_column])["counts"] if label_column in after else {}
            )
            labels_all = sorted(set(before_counts) | set(after_counts))
            positions = np.arange(len(labels_all))
            ax.bar(
                positions - width / 2,
                [before_counts.get(label, 0) for label in labels_all],
                width,
                label="Before",
                color="#457b9d",
            )
            ax.bar(
                positions + width / 2,
                [after_counts.get(label, 0) for label in labels_all],
                width,
                label="After",
                color="#81b29a",
            )
            ax.set_xticks(positions, labels_all)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No target column", ha="center", va="center")
            ax.set_xticks([])
        ax.set(title="Class distribution", ylabel="Rows")
        paths["class_imbalance"] = self._save_figure(fig, target / "class_imbalance.png")
        return paths

    # Alias kept intentionally obvious for notebook users.
    visualizations = generate_visualizations

    def generate_report(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame | None = None,
        *,
        strategy: str | Mapping[str, Any] = "conservative",
        output_dir: str | Path | None = None,
        rationale: str | None = None,
    ) -> dict[str, Path]:
        """Write JSON, Markdown, CSV, and plot artifacts for a cleaning run."""
        target = Path(output_dir) if output_dir is not None else self.report_dir
        target.mkdir(parents=True, exist_ok=True)
        cleaned = self.fix(df_before, strategy) if df_after is None else df_after.copy(deep=True)
        resolved, strategy_name = self._resolve_strategy(strategy)
        before = self.detect_issues(df_before)
        after = self.detect_issues(cleaned)
        comparison = self.compare(df_before, cleaned)
        explanation = rationale or self._strategy_rationale(strategy_name, resolved, before, after)

        comparison_path = target / "before_after.csv"
        comparison.to_csv(comparison_path, index=False)
        plot_paths = self.generate_visualizations(df_before, cleaned, output_dir=target)

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy": strategy_name,
            "strategy_config": resolved,
            "rationale": explanation,
            "before": before,
            "after": after,
            "target_shift": self._target_shift(before, after),
            "comparison": comparison.where(pd.notna(comparison), None).to_dict(orient="records"),
            "artifacts": {
                "comparison": comparison_path.name,
                "plots": {name: path.name for name, path in plot_paths.items()},
            },
        }
        json_path = target / "quality_report.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=self._json_default)

        markdown_path = target / "quality_report.md"
        markdown_path.write_text(
            self._markdown_report(payload, comparison),
            encoding="utf-8",
        )
        return {
            "json": json_path,
            "markdown": markdown_path,
            "comparison": comparison_path,
            **{f"plot_{name}": path for name, path in plot_paths.items()},
        }

    # ------------------------------------------------------------------- internals
    @staticmethod
    def _validate_frame(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

    def _resolve_strategy(self, strategy: str | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        if isinstance(strategy, str):
            if strategy not in self.strategies:
                available = ", ".join(sorted(self.strategies))
                raise ValueError(
                    f"Unknown quality strategy {strategy!r}; choose one of: {available}"
                )
            return dict(self.strategies[strategy]), strategy
        if not isinstance(strategy, Mapping):
            raise TypeError("strategy must be a strategy name or a mapping")
        values = dict(strategy)
        name = str(values.pop("name", "custom"))
        return values, name

    @staticmethod
    def _action_name(value: Any) -> str:
        if isinstance(value, Mapping):
            value = value.get("action", value.get("strategy", "keep"))
        return str(value).strip().lower()

    def _required_columns(self, df: pd.DataFrame, strategy: Mapping[str, Any]) -> list[str]:
        configured = strategy.get("required_columns", self.config.get("required_columns"))
        if configured is None:
            configured = ["record_id", self.text_column, "source"]
            label_column = self._resolve_label_column(df)
            if label_column:
                configured.append(label_column)
        if isinstance(configured, str):
            configured = [configured]
        # A minimal dataframe (e.g. in a unit test/notebook) is valid; absent
        # canonical columns cannot contain missing values and are simply ignored.
        return list(dict.fromkeys(column for column in configured if column in df.columns))

    @staticmethod
    def _is_string_like(series: pd.Series) -> bool:
        return bool(
            pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)
        )

    def _empty_text_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.text_column not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        text = df[self.text_column]
        # Null text is already represented in ``missing``; this mask is only for
        # zero-length/whitespace strings, avoiding double counting.
        return text.notna() & text.astype("string").fillna("").str.strip().eq("")

    def _text_lengths(self, df: pd.DataFrame) -> pd.Series:
        if self.text_column not in df.columns:
            return pd.Series(dtype=float, index=df.index)
        text = df[self.text_column]
        values = text.astype("string").str.len().astype("Float64")
        return values.astype(float)

    def _normalized_texts(self, df: pd.DataFrame) -> pd.Series:
        if self.text_column not in df.columns:
            return pd.Series("", index=df.index, dtype="string")
        return (
            df[self.text_column]
            .astype("string")
            .fillna("")
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _iqr_details(self, lengths: pd.Series) -> dict[str, Any]:
        valid = lengths.dropna()
        if valid.empty:
            return {
                "count": 0,
                "indices": [],
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "multiplier": self.iqr_multiplier,
            }
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        spread = q3 - q1
        lower = max(0.0, q1 - self.iqr_multiplier * spread)
        upper = q3 + self.iqr_multiplier * spread
        mask = (valid < lower) | (valid > upper)
        return {
            "count": int(mask.sum()),
            "indices": self._index_values(valid.index[mask]),
            "q1": q1,
            "q3": q3,
            "iqr": spread,
            "lower_bound": lower,
            "upper_bound": upper,
            "multiplier": self.iqr_multiplier,
        }

    def _zscore_details(self, lengths: pd.Series) -> dict[str, Any]:
        valid = lengths.dropna()
        if valid.empty:
            return {
                "count": 0,
                "indices": [],
                "mean": None,
                "std": None,
                "threshold": self.z_threshold,
            }
        mean = float(valid.mean())
        std = float(valid.std(ddof=0))
        if std == 0.0 or math.isnan(std):
            mask = pd.Series(False, index=valid.index)
        else:
            mask = ((valid - mean) / std).abs() > self.z_threshold
        return {
            "count": int(mask.sum()),
            "indices": self._index_values(valid.index[mask]),
            "mean": mean,
            "std": std,
            "threshold": self.z_threshold,
        }

    @staticmethod
    def _length_summary(lengths: pd.Series) -> dict[str, float | int | None]:
        valid = lengths.dropna()
        if valid.empty:
            return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
        return {
            "count": int(valid.size),
            "min": int(valid.min()),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "max": int(valid.max()),
        }

    def _resolve_label_column(self, df: pd.DataFrame) -> str | None:
        candidates = [self.label_column, "source_label", "label"]
        return next((str(column) for column in candidates if column and column in df.columns), None)

    @staticmethod
    def _target_columns(df: pd.DataFrame) -> list[str]:
        protected = [
            "source_label",
            "label",
            "auto_label",
            "human_label",
            "final_label",
        ]
        return [column for column in protected if column in df.columns]

    @staticmethod
    def _distribution(series: pd.Series) -> dict[str, Any]:
        non_missing = series.dropna()
        counts_series = non_missing.astype(str).value_counts(dropna=False).sort_index()
        total = int(counts_series.sum())
        counts = {str(key): int(value) for key, value in counts_series.items()}
        proportions = {
            key: (float(value) / total if total else 0.0) for key, value in counts.items()
        }
        return {
            "counts": counts,
            "proportions": proportions,
            "missing": int(series.isna().sum()),
            "observed": total,
        }

    def _imbalance_details(self, df: pd.DataFrame, column: str | None) -> dict[str, Any]:
        if column is None:
            return {
                "label_column": None,
                "counts": {},
                "proportions": {},
                "minority_to_majority_ratio": None,
                "is_imbalanced": False,
                "threshold": self.imbalance_threshold,
            }
        distribution = self._distribution(df[column])
        counts = list(distribution["counts"].values())
        if len(counts) == 0:
            ratio = None
            is_imbalanced = False
        elif len(counts) == 1:
            ratio = 0.0
            is_imbalanced = True
        else:
            ratio = float(min(counts) / max(counts)) if max(counts) else 0.0
            is_imbalanced = ratio < self.imbalance_threshold
        return {
            "label_column": column,
            **distribution,
            "minority_to_majority_ratio": ratio,
            "is_imbalanced": is_imbalanced,
            "threshold": self.imbalance_threshold,
        }

    @staticmethod
    def _index_values(index: Iterable[Any]) -> list[Any]:
        values: list[Any] = []
        for value in index:
            if isinstance(value, (np.integer,)):
                values.append(int(value))
            elif isinstance(value, (np.floating,)):
                values.append(float(value))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                values.append(value)
            else:
                values.append(str(value))
        return values

    @staticmethod
    def _clean_number(value: float) -> int | float:
        return int(value) if float(value).is_integer() else float(value)

    @staticmethod
    def _save_figure(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _hist_bins(values: pd.Series) -> int:
        return max(1, min(30, math.ceil(math.sqrt(max(1, len(values))))))

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _target_shift(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
        column = before.get("imbalance", {}).get("label_column") or after.get("imbalance", {}).get(
            "label_column"
        )
        if not column:
            return {"label_column": None, "total_variation": None, "max_abs_share_change": None}
        old = before.get("target_distributions", {}).get(column, {}).get("proportions", {})
        new = after.get("target_distributions", {}).get(column, {}).get("proportions", {})
        labels = set(old) | set(new)
        differences = [
            abs(float(new.get(label, 0.0)) - float(old.get(label, 0.0))) for label in labels
        ]
        return {
            "label_column": column,
            "total_variation": 0.5 * sum(differences),
            "max_abs_share_change": max(differences, default=0.0),
        }

    def _strategy_rationale(
        self,
        strategy_name: str,
        strategy: Mapping[str, Any],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> str:
        shift = self._target_shift(before, after)
        shift_value = shift.get("total_variation")
        shift_text = "not measurable" if shift_value is None else f"{shift_value:.4f}"
        if self._action_name(strategy.get("outliers", "keep")) in {
            "clip_iqr",
            "truncate_iqr",
            "truncate_long",
        }:
            choice = (
                "The conservative strategy removes unusable required-field rows and normalized-"
                "text duplicates, while retaining short reviews and truncating only the unusually "
                "long tail. This preserves more domain coverage for sentiment classification."
            )
        elif self._action_name(strategy.get("outliers", "keep")) == "drop_iqr":
            choice = (
                "The strict strategy removes IQR length anomalies entirely. It is useful when "
                "extreme lengths indicate corrupted records, but it may discard valid terse or "
                "detailed reviews and should therefore be compared with the conservative result."
            )
        else:
            choice = "The custom strategy applies the explicitly configured deterministic actions."
        return (
            f"{choice} Strategy name: {strategy_name}. Target-distribution total variation "
            f"before versus after is {shift_text}; labels themselves were never rewritten."
        )

    @staticmethod
    def _markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(map(str, columns)) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = []
        for values in frame.itertuples(index=False, name=None):
            cells = []
            for value in values:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    rendered = "—"
                elif isinstance(value, float):
                    rendered = f"{value:.4f}"
                else:
                    rendered = str(value)
                cells.append(rendered.replace("|", "\\|"))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, separator, *rows])

    def _markdown_report(self, payload: Mapping[str, Any], comparison: pd.DataFrame) -> str:
        before = payload["before"]
        after = payload["after"]
        plots = payload["artifacts"]["plots"]
        return f"""# Data quality report

Generated: `{payload["generated_at"]}`  
Strategy: `{payload["strategy"]}`

## Summary

- Rows: {before["row_count"]} → {after["row_count"]}
- Missing cells: {before["missing_total"]} → {after["missing_total"]}
- Empty texts: {before["empty_text"]} → {after["empty_text"]}
- Exact duplicates: {before["duplicates"]} → {after["duplicates"]}
- Normalized-text duplicates: {before["text_duplicates"]} → {after["text_duplicates"]}
- IQR length outliers: {before["outliers"]["iqr"]["count"]} → {after["outliers"]["iqr"]["count"]}
- Z-score length outliers: {before["outliers"]["zscore"]["count"]} → {after["outliers"]["zscore"]["count"]}

## Before / after

{self._markdown_table(comparison)}

## Strategy rationale

{payload["rationale"]}

The cleaning stage never imputes, normalizes, or overwrites `label`,
`source_label`, `auto_label`, `human_label`, or `final_label`. A distribution can
change only when a complete record is removed. The measured target-distribution
shift is recorded in `quality_report.json`.

## Visual diagnostics

![Missing values]({plots["missing_values"]})

![Empty text]({plots["empty_text"]})

![Exact duplicates]({plots["duplicates"]})

![Text-length outliers]({plots["outliers"]})

![Class imbalance]({plots["class_imbalance"]})
"""


__all__ = ["DataQualityAgent"]
