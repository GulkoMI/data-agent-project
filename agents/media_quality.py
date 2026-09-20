"""Non-destructive media quality checks that preserve unlabeled assets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd


class MediaQualityAgent:
    """Conservative removes unusable/identical assets; strict also drops IQR tails.

    Darkness, blur, and perceptual similarity are review flags. They can be
    meaningful task examples and never cause silent deletion. No labels are
    required, imputed, or overwritten and no source file is changed.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        raw = dict(config or {})
        self.config = dict(raw.get("quality", raw) or {})
        self.iqr_multiplier = float(self.config.get("outlier", {}).get("iqr_multiplier", 1.5))
        if not math.isfinite(self.iqr_multiplier) or self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive")

    @staticmethod
    def _invalid(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=frame.index)
        for column in ("record_id", "asset_id", "group_id", "media_path", "content_hash"):
            if column not in frame:
                raise ValueError(f"Missing media quality column: {column}")
            mask |= frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if "valid" in frame:
            mask |= ~frame["valid"].fillna(False).astype(bool)
        return mask

    def _outliers(self, frame: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
        combined = pd.Series(False, index=frame.index)
        details: dict[str, Any] = {}
        configured = self.config.get("media_outlier_columns", ["width", "height", "duration_sec"])
        if isinstance(configured, str):
            configured = [configured]
        for column in configured:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.dropna()
            if len(valid) < 4:
                details[column] = {"count": 0, "lower_bound": None, "upper_bound": None}
                continue
            q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
            lower, upper = (
                max(0.0, q1 - self.iqr_multiplier * (q3 - q1)),
                q3 + self.iqr_multiplier * (q3 - q1),
            )
            mask = (values < lower) | (values > upper)
            combined |= mask
            details[column] = {"count": int(mask.sum()), "lower_bound": lower, "upper_bound": upper}
        return combined, details

    def detect_issues(self, frame: pd.DataFrame) -> dict[str, Any]:
        invalid = self._invalid(frame)
        duplicates = frame["content_hash"].notna() & frame["content_hash"].duplicated(keep="first")
        outlier_mask, details = self._outliers(frame.loc[~invalid])
        brightness = pd.to_numeric(frame.get("brightness", pd.Series(dtype=float)), errors="coerce")
        edges = pd.to_numeric(frame.get("edge_variance", pd.Series(dtype=float)), errors="coerce")
        near = frame.get("near_duplicate_candidate", pd.Series(False, index=frame.index)).fillna(
            False
        )
        return {
            "row_count": len(frame),
            "invalid": int(invalid.sum()),
            "invalid_record_ids": frame.loc[invalid, "record_id"].astype(str).tolist(),
            "exact_duplicates": int(duplicates.sum()),
            "near_duplicate_candidates": int(near.sum()),
            "dark_candidates": int(
                (brightness < float(self.config.get("dark_threshold", 10))).sum()
            ),
            "low_detail_candidates": int(
                (edges < float(self.config.get("low_detail_threshold", 5))).sum()
            ),
            "outliers": {"iqr": {"count": int(outlier_mask.sum()), "columns": details}},
            "unlabeled": int(frame["label"].isna().sum()) if "label" in frame else len(frame),
            "note": "Unlabeled, dark, low-detail and perceptually similar assets are retained for review.",
        }

    def fix(
        self, frame: pd.DataFrame, strategy: str | Mapping[str, Any] = "conservative"
    ) -> pd.DataFrame:
        name = (
            str(strategy.get("name", "conservative")) if isinstance(strategy, Mapping) else strategy
        )
        if name not in {"conservative", "strict"}:
            raise ValueError("Media quality strategy must be conservative or strict")
        cleaned = frame.loc[~self._invalid(frame)].copy(deep=True)
        cleaned = cleaned.drop_duplicates(subset=["content_hash"], keep="first")
        if name == "strict":
            outliers, _ = self._outliers(cleaned)
            cleaned = cleaned.loc[~outliers].copy()
        return cleaned.reset_index(drop=True)


__all__ = ["MediaQualityAgent"]
