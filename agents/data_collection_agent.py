"""Collection agent for a cross-domain sentiment dataset from public sources.

The public methods intentionally return pandas frames so they remain useful as
stand-alone "skills".  ``run`` adds orchestration, fault isolation, persistence,
and a compact EDA report.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from glob import glob
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

try:  # Importing this optional-at-import dependency lazily keeps unit tests light.
    from datasets import load_dataset as hf_load_dataset
except ImportError:  # pragma: no cover - exercised only in minimal installations.
    hf_load_dataset = None

from agents.common import (
    LABELS,
    ProjectPaths,
    ensure_unified_schema,
    load_frame,
    normalize_label,
    read_yaml,
    save_frame,
    stable_record_id,
    utc_now_iso,
    write_json,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def _is_missing(value: Any) -> bool:
    """Scalar-safe missing-value predicate."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if not _is_missing(value) and str(value).strip():
            return value
    return None


def _slug(value: str) -> str:
    cleaned = _SLUG_RE.sub("_", value).strip("_.")
    return cleaned or "source"


class _SimpleSelectorParser(HTMLParser):
    """Small dependency-free parser for tag, ``.class`` and ``#id`` selectors."""

    def __init__(self, selector: str) -> None:
        super().__init__(convert_charrefs=True)
        # For a descendant selector the terminal component is the selected node.
        terminal = re.split(r"\s+|>", selector.strip())[-1]
        match = re.fullmatch(
            r"(?P<tag>[A-Za-z][\w-]*)?(?:#(?P<id>[\w-]+))?(?P<classes>(?:\.[\w-]+)*)",
            terminal,
        )
        if not match or not terminal:
            raise ValueError(
                "The fallback scraper supports tag, .class, #id and tag.class selectors"
            )
        self.tag = (match.group("tag") or "").lower()
        self.element_id = match.group("id")
        self.classes = {item for item in match.group("classes").split(".") if item}
        self._captures: list[dict[str, Any]] = []
        self.values: list[str] = []

    def _matches(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        attributes = dict(attrs)
        actual_classes = set((attributes.get("class") or "").split())
        return (
            (not self.tag or tag.lower() == self.tag)
            and (not self.element_id or attributes.get("id") == self.element_id)
            and self.classes.issubset(actual_classes)
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for capture in self._captures:
            capture["depth"] += 1
        if self._matches(tag, attrs):
            self._captures.append({"depth": 1, "parts": []})

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._matches(tag, attrs):
            self.values.append("")

    def handle_data(self, data: str) -> None:
        for capture in self._captures:
            capture["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        finished: list[dict[str, Any]] = []
        for capture in self._captures:
            capture["depth"] -= 1
            if capture["depth"] == 0:
                finished.append(capture)
        for capture in finished:
            value = " ".join("".join(capture["parts"]).split())
            if value:
                self.values.append(value)
            self._captures.remove(capture)


class DataCollectionAgent:
    """Collect, normalize, persist, and describe review data from multiple sources."""

    def __init__(self, config: str | Path = "config.yaml") -> None:
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if not config_path.exists() and str(config) == "config.yaml":
            config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config_path = config_path.resolve()
        self.config = read_yaml(self.config_path)
        configured_root = Path(str(self.config.get("project", {}).get("root", "."))).expanduser()
        self.root = (
            configured_root if configured_root.is_absolute() else self.config_path.parent / configured_root
        ).resolve()
        self.paths = ProjectPaths.from_config(self.root, self.config)
        self.paths.create()
        self.collection_config = self.config.get("collection", {})
        self.random_seed = int(self.config.get("project", {}).get("random_seed", 42))
        self.last_manifest: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public source skills
    # ------------------------------------------------------------------
    def collect_media(
        self,
        sources: Sequence[Mapping[str, Any] | str] | None = None,
        *,
        modality: str | None = None,
    ) -> pd.DataFrame:
        """Collect local image/video assets without invoking the text/HF workflow.

        Rows retain unassigned labels and source/group provenance; the visual
        runner decides quality, group splitting, and frame preparation next.
        """
        from agents.media_collection import collect_media_sources

        selected = sources if sources is not None else self.collection_config.get("sources", [])
        chosen_modality = modality or self.config.get("project", {}).get("modality", "")
        return collect_media_sources(selected, self.root, str(chosen_modality))

    def scrape(self, url: str, selector: str) -> pd.DataFrame:
        """Download a page and turn matching text elements into canonical rows."""
        timeout = float(self.collection_config.get("request_timeout", 30))
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "DataCollectionAgent/1.0 (educational project)"},
        )
        response.raise_for_status()
        parser = _SimpleSelectorParser(selector)
        parser.feed(response.text)
        source = f"scrape:{urlparse(url).netloc or 'web'}"
        rows = [
            {"text": text, "source_id": str(index), "label": None}
            for index, text in enumerate(parser.values)
        ]
        return self._rows_to_canonical(rows, default_source=source)

    def fetch_api(self, endpoint: str, params: Mapping[str, Any] | None) -> pd.DataFrame:
        """Fetch a conventional JSON API and infer common text/label/id fields."""
        timeout = float(self.collection_config.get("request_timeout", 30))
        response = requests.get(endpoint, params=dict(params or {}), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        records = self._records_from_json(payload)
        source = f"api:{urlparse(endpoint).netloc or 'endpoint'}"
        return self._rows_to_canonical(records, default_source=source)

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        **overrides: Any,
    ) -> pd.DataFrame:
        """Stream a deterministic, schema-configurable sample from Hugging Face.

        Dataset-specific field names, exact row filters, label mappings, and
        metadata passthrough are defined in ``config.yaml``. The iterator stops
        when both canonical label quotas are met, so large datasets are never
        materialized in full on a laptop.
        """
        if source.lower() not in {"hf", "huggingface", "hf_dataset"}:
            raise ValueError(f"Unsupported open-dataset source: {source}")
        if hf_load_dataset is None:
            raise ImportError("Install the 'datasets' package to load Hugging Face datasets")

        spec = self._source_defaults("hf_dataset", name=name)
        spec.update({key: value for key, value in overrides.items() if value is not None})
        split = str(spec.get("split", "train"))
        streaming = bool(spec.get("streaming", True))
        text_field = str(spec.get("text_field", "text"))
        configured_text_fields = spec.get("text_fields")
        if isinstance(configured_text_fields, str):
            text_fields = [configured_text_fields]
        elif configured_text_fields:
            text_fields = [str(field) for field in configured_text_fields]
        else:
            text_fields = []
        text_separator = str(spec.get("text_separator", "\n\n"))
        label_field = str(spec.get("label_field", "label"))
        label_map = spec.get("label_map", {})
        row_filters = spec.get("filters", spec.get("row_filters", {})) or {}
        metadata_fields = spec.get("metadata_fields", spec.get("keep_fields", [])) or []
        sample_size = int(spec.get("sample_size", 200))
        per_label = int(spec.get("per_label") or max(1, sample_size // len(LABELS)))
        scan_limit = int(spec.get("scan_limit", max(per_label * 200, 10_000)))
        buffer_size = int(spec.get("shuffle_buffer", min(max(per_label * 20, 1_000), 20_000)))

        subset = spec.get("subset", spec.get("config_name"))
        local_files = self._resolve_local_dataset_files(spec.get("local_path"))
        if local_files:
            local_format = str(spec.get("local_format", "")).strip().lower()
            if not local_format:
                raise ValueError(
                    f"A local_format is required when local files exist for {name}"
                )
            # ``datasets`` uses the JSON builder for both JSON and JSON Lines.
            local_builder = "json" if local_format in {"jsonl", "ndjson"} else local_format
            dataset = hf_load_dataset(
                local_builder,
                data_files=local_files,
                split=split,
                streaming=streaming,
            )
        else:
            load_args: list[str] = []
            if subset:
                load_args.append(str(subset))
            load_kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
            if spec.get("revision"):
                load_kwargs["revision"] = str(spec["revision"])
            token_env = spec.get("token_env")
            token_setting = spec.get("token")
            if token_env:
                resolved_token = os.environ.get(str(token_env))
                if not resolved_token:
                    raise ValueError(
                        f"Hugging Face token environment variable is empty: {token_env}"
                    )
                load_kwargs["token"] = resolved_token
            elif token_setting is True:
                # ``True`` asks datasets to use the locally configured HF credential.
                load_kwargs["token"] = True
            elif token_setting is not None and token_setting is not False:
                raise ValueError(
                    "Use token: true or token_env; do not store a literal HF token in YAML"
                )

            dataset = hf_load_dataset(name, *load_args, **load_kwargs)
        if hasattr(dataset, "shuffle"):
            shuffle_kwargs: dict[str, Any] = {"seed": self.random_seed}
            if streaming:
                shuffle_kwargs["buffer_size"] = buffer_size
            dataset = dataset.shuffle(**shuffle_kwargs)

        source_name = str(spec.get("id", f"hf:{name}"))
        buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
        for index, raw_row in enumerate(dataset):
            if index >= scan_limit:
                break
            row = dict(raw_row)
            if not self._matches_row_filters(row, row_filters):
                continue
            label = self._mapped_label(row.get(label_field), label_map)
            if label not in buckets or len(buckets[label]) >= per_label:
                continue
            if text_fields:
                text_parts = [
                    str(row[field]).strip()
                    for field in text_fields
                    if field in row and not _is_missing(row[field]) and str(row[field]).strip()
                ]
                text = text_separator.join(text_parts) if text_parts else None
            else:
                text = _first_present(row, [text_field, "text", "content", "review", "body"])
            if text is None:
                continue
            explicit_id = _first_present(
                row,
                [str(spec.get("id_field", "id")), "id", "review_id", "asin"],
            )
            source_id = explicit_id if explicit_id is not None else f"{split}:{index}"
            record = {
                "text": str(text),
                "label": label,
                "source_id": str(source_id),
                "dataset_name": name,
                "dataset_split": split,
            }
            if subset:
                record["dataset_subset"] = str(subset)
            record.update(self._select_metadata(row, metadata_fields))
            buckets[label].append(record)
            if all(len(bucket) >= per_label for bucket in buckets.values()):
                break

        available = min(len(bucket) for bucket in buckets.values())
        if available == 0:
            counts = {label: len(rows) for label, rows in buckets.items()}
            raise ValueError(f"Could not build a two-class sample from {name}: {counts}")
        # Trimming to the smaller bucket preserves balance if scan_limit is reached.
        rows = [row for label in LABELS for row in buckets[label][:available]]
        return self._rows_to_canonical(rows, default_source=source_name)

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate non-empty frames and validate the shared data contract."""
        normalized = [
            self._canonicalize_frame(frame, default_source=f"source_{index}")
            for index, frame in enumerate(sources)
            if frame is not None and not frame.empty
        ]
        if not normalized:
            raise ValueError("No non-empty source frames were supplied")
        return ensure_unified_schema(pd.concat(normalized, ignore_index=True, sort=False))

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, sources: Sequence[Mapping[str, Any]] | None = None) -> pd.DataFrame:
        """Collect configured sources, persist artifacts, and return one dataset.

        An exception in one source is recorded in the manifest and does not stop
        the remaining sources. At least two independently configured sources must
        nevertheless succeed, as required by the assignment.
        """
        configured_sources = self.collection_config.get("sources", [])
        source_specs = list(sources if sources is not None else configured_sources)
        started_at = utc_now_iso()
        frames: list[pd.DataFrame] = []
        source_reports: list[dict[str, Any]] = []

        for index, original_spec in enumerate(source_specs):
            spec = dict(original_spec)
            source_type = str(spec.get("type", "")).lower()
            source_name = str(spec.get("id") or spec.get("name") or f"source_{index + 1}")
            report: dict[str, Any] = {
                "id": source_name,
                "type": source_type,
                "status": "failed",
                "rows": 0,
            }
            try:
                frame = self._collect_source(spec)
                frame = self._canonicalize_frame(frame, default_source=source_name)
                if frame.empty:
                    report.update(status="empty", error="Source returned zero rows")
                else:
                    artifact_info = self._save_source_frame(frame, source_name)
                    report.update(
                        status="ok",
                        rows=len(frame),
                        labels=self._value_counts(frame["label"]),
                        artifacts=artifact_info["paths"],
                    )
                    if artifact_info["errors"]:
                        report["artifact_warnings"] = artifact_info["errors"]
                    frames.append(frame)
            except Exception as error:  # noqa: BLE001 - source-level isolation is intentional.
                report["error"] = f"{type(error).__name__}: {error}"
            source_reports.append(report)

        manifest_path = self.paths.raw / "collection_manifest.json"
        manifest: dict[str, Any] = {
            "status": "running",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "config": str(self.config_path),
            "required_non_empty_sources": 2,
            "successful_sources": len(frames),
            "sources": source_reports,
        }

        if len(frames) < 2:
            manifest["status"] = "failed"
            manifest["error"] = f"At least 2 non-empty sources are required; received {len(frames)}"
            write_json(manifest, manifest_path)
            self.last_manifest = manifest
            failures = "; ".join(
                f"{item['id']}: {item.get('error', item['status'])}" for item in source_reports
            )
            raise RuntimeError(f"{manifest['error']}. {failures}")

        merged = self.merge(frames)
        output_setting = self.collection_config.get("output", "data/raw/reviews_raw.parquet")
        output_path = Path(output_setting)
        if not output_path.is_absolute():
            output_path = self.root / output_path
        merged_artifacts = self._save_in_both_formats(merged, output_path)
        eda_artifacts = self._write_eda(merged)

        manifest.update(
            status="ok",
            finished_at=utc_now_iso(),
            rows=len(merged),
            columns=list(merged.columns),
            label_distribution=self._value_counts(merged["label"]),
            source_distribution=self._value_counts(merged["source"]),
            merged_artifacts=merged_artifacts["paths"],
            eda_artifacts=eda_artifacts,
        )
        if merged_artifacts["errors"]:
            manifest["artifact_warnings"] = merged_artifacts["errors"]
        write_json(manifest, manifest_path)
        self.last_manifest = manifest
        return merged

    # ------------------------------------------------------------------
    # Normalization and dispatch helpers
    # ------------------------------------------------------------------
    def _source_defaults(self, source_type: str, *, name: str | None = None) -> dict[str, Any]:
        for candidate in self.collection_config.get("sources", []):
            if str(candidate.get("type", "")).lower() != source_type:
                continue
            if name is not None and candidate.get("name") != name:
                continue
            return dict(candidate)
        return {}

    def _resolve_local_dataset_files(self, local_path: Any) -> list[str]:
        """Resolve configured local files without changing their logical dataset identity."""
        if local_path is None or local_path == "":
            return []
        configured_paths = (
            list(local_path)
            if isinstance(local_path, Sequence) and not isinstance(local_path, (str, bytes))
            else [local_path]
        )
        matches: set[str] = set()
        for configured_path in configured_paths:
            candidate = Path(str(configured_path)).expanduser()
            if not candidate.is_absolute():
                candidate = self.root / candidate
            matches.update(
                str(Path(match).resolve())
                for match in glob(str(candidate), recursive=True)
                if Path(match).is_file()
            )
        return sorted(matches)

    def _collect_source(self, spec: Mapping[str, Any]) -> pd.DataFrame:
        source_type = str(spec.get("type", "")).lower()
        if source_type in {"hf", "hf_dataset", "huggingface"}:
            name = spec.get("name")
            if not name:
                raise ValueError("A Hugging Face source requires 'name'")
            return self.load_dataset(
                str(name),
                source="hf",
                **{key: value for key, value in spec.items() if key not in {"name", "type"}},
            )
        if source_type == "scrape":
            return self.scrape(str(spec["url"]), str(spec["selector"]))
        if source_type in {"api", "json_api"}:
            return self.fetch_api(str(spec["endpoint"]), spec.get("params"))
        if source_type in {"csv", "file", "parquet", "jsonl"}:
            path = Path(str(spec.get("path", "")))
            if not path.is_absolute():
                path = self.root / path
            return load_frame(path)
        raise ValueError(f"Unsupported source type: {source_type or '<missing>'}")

    @staticmethod
    def _records_from_json(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [dict(item) if isinstance(item, Mapping) else {"text": item} for item in payload]
        if not isinstance(payload, Mapping):
            return [{"text": payload}]
        for key in ("results", "data", "items", "reviews", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [
                    dict(item) if isinstance(item, Mapping) else {"text": item} for item in value
                ]
        return [dict(payload)]

    @staticmethod
    def _mapped_label(value: Any, label_map: Mapping[Any, Any] | None = None) -> str | None:
        mapping = label_map or {}
        sentinel = object()
        mapped: Any = sentinel
        try:
            if value in mapping:
                mapped = mapping[value]
        except TypeError:
            pass
        if mapped is sentinel:
            normalized_key = str(value).strip().casefold()
            for raw_label, canonical_label in mapping.items():
                if str(raw_label).strip().casefold() == normalized_key:
                    mapped = canonical_label
                    break
        if mapped is sentinel:
            mapped = value
        normalized = normalize_label(mapped)
        return normalized if normalized in LABELS else None

    @staticmethod
    def _matches_row_filters(
        row: Mapping[str, Any], filters: Mapping[str, Any]
    ) -> bool:
        if not isinstance(filters, Mapping):
            raise TypeError("HF source filters must be a field-to-value mapping")
        for field, expected in filters.items():
            actual = row.get(str(field))
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _select_metadata(
        row: Mapping[str, Any], metadata_fields: Mapping[str, str] | Sequence[str]
    ) -> dict[str, Any]:
        if isinstance(metadata_fields, Mapping):
            fields = [(str(source), str(target)) for source, target in metadata_fields.items()]
        elif isinstance(metadata_fields, str):
            fields = [(metadata_fields, metadata_fields)]
        else:
            fields = [(str(field), str(field)) for field in metadata_fields]
        return {
            target: row[source]
            for source, target in fields
            if source in row and target not in {"text", "label", "source", "source_id"}
        }

    def _rows_to_canonical(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        default_source: str,
    ) -> pd.DataFrame:
        records = [dict(row) for row in rows]
        if not records:
            return ensure_unified_schema(pd.DataFrame(), allow_empty=True)
        return self._canonicalize_frame(pd.DataFrame.from_records(records), default_source)

    def _canonicalize_frame(self, frame: pd.DataFrame, default_source: str) -> pd.DataFrame:
        result = frame.copy()
        if "text" not in result.columns:
            for candidate in ("review", "content", "body", "description", "title"):
                if candidate in result.columns:
                    result["text"] = result[candidate]
                    break
        if "text" not in result.columns:
            result["text"] = ""

        if "label" not in result.columns:
            if "sentiment" in result.columns:
                result["label"] = result["sentiment"]
            else:
                result["label"] = None
        result["label"] = result["label"].map(self._mapped_label)

        if "source" not in result.columns:
            result["source"] = default_source
        else:
            result["source"] = result["source"].where(result["source"].notna(), default_source)
        if "source_id" not in result.columns:
            id_column = next(
                (name for name in ("id", "review_id") if name in result),
                None,
            )
            result["source_id"] = (
                result[id_column].astype(str) if id_column else result.index.astype(str)
            )
        missing_source_ids = result["source_id"].isna() | result["source_id"].astype(str).eq("")
        result.loc[missing_source_ids, "source_id"] = result.index[missing_source_ids].astype(str)

        if "collected_at" not in result.columns:
            result["collected_at"] = utc_now_iso()
        else:
            result["collected_at"] = result["collected_at"].fillna(utc_now_iso())
        for column in ("audio", "image"):
            if column not in result.columns:
                result[column] = None

        if "record_id" not in result.columns:
            result["record_id"] = None
        missing_ids = result["record_id"].isna() | result["record_id"].astype(str).eq("")
        result.loc[missing_ids, "record_id"] = [
            stable_record_id(source, source_id, text)
            for source, source_id, text in result.loc[
                missing_ids, ["source", "source_id", "text"]
            ].itertuples(index=False, name=None)
        ]
        return ensure_unified_schema(result, allow_empty=True)

    # ------------------------------------------------------------------
    # Persistence and EDA
    # ------------------------------------------------------------------
    def _save_source_frame(self, frame: pd.DataFrame, source_name: str) -> dict[str, Any]:
        path = self.paths.raw / f"source_{_slug(source_name)}.parquet"
        return self._save_in_both_formats(frame, path)

    @staticmethod
    def _value_counts(series: pd.Series) -> dict[str, int]:
        values = series.fillna("unlabeled").astype(str).value_counts().sort_index()
        return {str(key): int(value) for key, value in values.items()}

    @staticmethod
    def _save_in_both_formats(frame: pd.DataFrame, preferred: Path) -> dict[str, Any]:
        parquet_path = preferred.with_suffix(".parquet")
        csv_path = preferred.with_suffix(".csv")
        paths: list[str] = []
        errors: list[str] = []
        for path in (parquet_path, csv_path):
            try:
                save_frame(frame, path)
                paths.append(str(path.resolve()))
            except Exception as error:  # noqa: BLE001 - CSV remains a portable fallback.
                errors.append(f"{path.name}: {type(error).__name__}: {error}")
        if not paths:
            raise OSError(f"Could not persist frame: {'; '.join(errors)}")
        return {"paths": paths, "errors": errors}

    def _write_eda(self, frame: pd.DataFrame) -> list[str]:
        eda_dir = self.paths.reports / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[str] = []

        class_counts = (
            frame["label"]
            .fillna("unlabeled")
            .astype(str)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )
        source_counts = (
            frame["source"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("source")
            .reset_index(name="count")
        )
        lengths = pd.DataFrame(
            {
                "record_id": frame["record_id"],
                "source": frame["source"],
                "char_length": frame["text"].fillna("").astype(str).str.len(),
                "word_count": frame["text"]
                .fillna("")
                .astype(str)
                .map(lambda value: len(_WORD_RE.findall(value))),
            }
        )
        top_n = int(self.collection_config.get("eda", {}).get("top_words", 20))
        words = Counter(
            token
            for text in frame["text"].fillna("").astype(str)
            for token in (match.lower() for match in _WORD_RE.findall(text))
            if token not in _STOP_WORDS
        )
        top_words = pd.DataFrame(words.most_common(top_n), columns=["word", "count"])

        for name, table in (
            ("class_distribution.csv", class_counts),
            ("source_distribution.csv", source_counts),
            ("text_lengths.csv", lengths),
            ("top_20_words.csv", top_words),
        ):
            path = eda_dir / name
            table.to_csv(path, index=False)
            artifacts.append(str(path.resolve()))

        length_summary = {
            "rows": len(lengths),
            "char_length": lengths["char_length"].describe().to_dict(),
            "word_count": lengths["word_count"].describe().to_dict(),
        }
        summary_path = write_json(length_summary, eda_dir / "text_length_summary.json")
        artifacts.append(str(summary_path.resolve()))

        # Plot failures should not invalidate otherwise sound collected data.
        try:
            mpl_dir = self.paths.cache / "matplotlib"
            mpl_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_specs = [
                (class_counts, "label", "count", "Class distribution", "class_distribution.png"),
                (
                    source_counts,
                    "source",
                    "count",
                    "Source distribution",
                    "source_distribution.png",
                ),
                (top_words, "word", "count", f"Top {top_n} words", "top_20_words.png"),
            ]
            for table, x_column, y_column, title, filename in plot_specs:
                figure, axis = plt.subplots(figsize=(8, 4.5))
                if not table.empty:
                    axis.bar(table[x_column].astype(str), table[y_column])
                axis.set_title(title)
                axis.tick_params(axis="x", rotation=45)
                figure.tight_layout()
                path = eda_dir / filename
                figure.savefig(path, dpi=140)
                plt.close(figure)
                artifacts.append(str(path.resolve()))

            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.hist(lengths["word_count"], bins=min(30, max(5, len(lengths))))
            axis.set(title="Review word lengths", xlabel="Words", ylabel="Reviews")
            figure.tight_layout()
            path = eda_dir / "text_length_distribution.png"
            figure.savefig(path, dpi=140)
            plt.close(figure)
            artifacts.append(str(path.resolve()))
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            warning_path = write_json(
                {"plot_warning": f"{type(error).__name__}: {error}"},
                eda_dir / "plot_warning.json",
            )
            artifacts.append(str(warning_path.resolve()))

        return artifacts


__all__ = ["DataCollectionAgent"]
