"""Shared data contracts and filesystem helpers for all agents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LABELS = ("negative", "positive")
REQUIRED_COLUMNS = (
    "record_id",
    "text",
    "audio",
    "image",
    "label",
    "source",
    "source_id",
    "collected_at",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_record_id(source: str, source_id: Any, text: Any) -> str:
    normalized = " ".join(str(text or "").split()).strip().lower()
    payload = f"{source}\x1f{source_id}\x1f{normalized}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def normalize_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "0": "negative",
        "1": "positive",
        "neg": "negative",
        "negative": "negative",
        "false": "negative",
        "pos": "positive",
        "positive": "positive",
        "true": "positive",
    }
    return mapping.get(normalized, normalized or None)


def ensure_unified_schema(frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    """Return a validated copy in the canonical raw-data schema."""
    result = frame.copy()
    for column in REQUIRED_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result = result.loc[
        :,
        list(REQUIRED_COLUMNS)
        + [column for column in result.columns if column not in REQUIRED_COLUMNS],
    ]
    result["text"] = result["text"].fillna("").astype(str)
    result["label"] = result["label"].map(normalize_label)
    if not allow_empty and result.empty:
        raise ValueError("The unified dataset is empty")
    invalid_labels = sorted(set(result["label"].dropna()) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"Unsupported labels: {invalid_labels}")
    duplicated_ids = result["record_id"].duplicated(keep=False)
    if duplicated_ids.any():
        values = result.loc[duplicated_ids, "record_id"].astype(str).head(5).tolist()
        raise ValueError(f"record_id must be unique; duplicates include {values}")
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent(path)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix.lower() == ".jsonl":
        frame.to_json(target, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_csv(target, index=False)
    return target


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".jsonl":
        return pd.read_json(source, orient="records", lines=True)
    return pd.read_csv(source)


def write_json(payload: Any, path: str | Path) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return target


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    processed: Path
    labeled: Path
    review: Path
    cache: Path
    reports: Path
    models: Path

    @classmethod
    def from_config(cls, root: str | Path, config: dict[str, Any]) -> ProjectPaths:
        base = Path(root).resolve()
        paths = config.get("paths", {})
        return cls(
            root=base,
            raw=base / paths.get("raw_dir", "data/raw"),
            processed=base / paths.get("processed_dir", "data/processed"),
            labeled=base / paths.get("labeled_dir", "data/labeled"),
            review=base / paths.get("review_dir", "data/review"),
            cache=base / paths.get("cache_dir", "data/cache"),
            reports=base / paths.get("reports_dir", "reports"),
            models=base / paths.get("models_dir", "models"),
        )

    def create(self) -> None:
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def directories(self) -> Iterable[Path]:
        return (
            self.raw,
            self.processed,
            self.labeled,
            self.review,
            self.cache,
            self.reports,
            self.models,
        )
