"""Offline tests for DataCollectionAgent.

Every HTTP/dataset boundary is replaced with a deterministic fake; running this
module never requires Hugging Face connectivity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import agents.data_collection_agent as collection_module
from agents.common import REQUIRED_COLUMNS
from agents.data_collection_agent import DataCollectionAgent


def _write_config(tmp_path: Path, sources: list[dict[str, Any]]) -> Path:
    config = {
        "project": {"random_seed": 17},
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "labeled_dir": "data/labeled",
            "review_dir": "data/review",
            "cache_dir": "data/cache",
            "reports_dir": "reports",
            "models_dir": "models",
        },
        "collection": {
            "output": "data/raw/reviews_raw.parquet",
            "sources": sources,
            "eda": {"top_words": 20},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class _FakeResponse:
    def __init__(self, *, payload: Any = None, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeStream:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.shuffle_calls: list[dict[str, Any]] = []

    def shuffle(self, **kwargs: Any) -> _FakeStream:
        self.shuffle_calls.append(kwargs)
        return self

    def __iter__(self):
        return iter(self.records)


def test_hf_streaming_sample_is_balanced_deterministic_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {
        "id": "amazon_polarity",
        "type": "hf_dataset",
        "name": "mteb/amazon_polarity",
        "split": "train",
        "streaming": True,
        "sample_size": 4,
        "per_label": 2,
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
    }
    agent = DataCollectionAgent(_write_config(tmp_path, [spec]))
    records = [
        {"text": "bad product", "label": 0},
        {"text": "great product", "label": 1},
        {"text": "awful item", "label": 0},
        {"text": "excellent item", "label": 1},
        {"text": "unused extra", "label": 1},
    ]
    streams: list[_FakeStream] = []

    def fake_load_dataset(name: str, *, split: str, streaming: bool) -> _FakeStream:
        assert name == "mteb/amazon_polarity"
        assert split == "train"
        assert streaming is True
        stream = _FakeStream(records)
        streams.append(stream)
        return stream

    monkeypatch.setattr(collection_module, "hf_load_dataset", fake_load_dataset)

    first = agent.load_dataset("mteb/amazon_polarity")
    second = agent.load_dataset("mteb/amazon_polarity")

    assert first["label"].value_counts().to_dict() == {"negative": 2, "positive": 2}
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert list(first.columns[: len(REQUIRED_COLUMNS)]) == list(REQUIRED_COLUMNS)
    assert first["record_id"].is_unique
    assert all(stream.shuffle_calls == [{"seed": 17, "buffer_size": 1000}] for stream in streams)


def test_hf_uses_matching_local_files_and_preserves_logical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "snapshots" / "amazon"
    local_dir.mkdir(parents=True)
    first_file = local_dir / "train-00001.parquet"
    second_file = local_dir / "train-00000.parquet"
    first_file.touch()
    second_file.touch()
    spec = {
        "id": "amazon_polarity",
        "type": "hf_dataset",
        "name": "mteb/amazon_polarity",
        "local_path": "snapshots/amazon/train-*.parquet",
        "local_format": "parquet",
        "split": "train",
        "streaming": True,
        "sample_size": 2,
        "per_label": 1,
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
    }
    agent = DataCollectionAgent(_write_config(tmp_path, [spec]))
    stream = _FakeStream(
        [
            {"text": "bad product", "label": 0},
            {"text": "great product", "label": 1},
        ]
    )
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(name: str, *args: Any, **kwargs: Any) -> _FakeStream:
        calls.append((name, args, kwargs))
        return stream

    monkeypatch.setattr(collection_module, "hf_load_dataset", fake_load_dataset)
    frame = agent.load_dataset("mteb/amazon_polarity")

    assert calls == [
        (
            "parquet",
            (),
            {
                "data_files": [str(second_file.resolve()), str(first_file.resolve())],
                "split": "train",
                "streaming": True,
            },
        )
    ]
    assert frame["source"].eq("amazon_polarity").all()
    assert frame["dataset_name"].eq("mteb/amazon_polarity").all()
    assert "local" not in " ".join(frame["source"].astype(str)).lower()


def test_hf_falls_back_to_remote_name_when_local_files_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {
        "id": "amazon_polarity",
        "type": "hf_dataset",
        "name": "mteb/amazon_polarity",
        "local_path": "snapshots/amazon/train-*.parquet",
        "local_format": "parquet",
        "split": "train",
        "streaming": True,
        "sample_size": 2,
        "per_label": 1,
        "label_map": {0: "negative", 1: "positive"},
    }
    agent = DataCollectionAgent(_write_config(tmp_path, [spec]))
    stream = _FakeStream(
        [
            {"text": "bad product", "label": 0},
            {"text": "great product", "label": 1},
        ]
    )
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(name: str, *args: Any, **kwargs: Any) -> _FakeStream:
        calls.append((name, args, kwargs))
        return stream

    monkeypatch.setattr(collection_module, "hf_load_dataset", fake_load_dataset)
    agent.load_dataset("mteb/amazon_polarity")

    assert calls == [
        (
            "mteb/amazon_polarity",
            (),
            {"split": "train", "streaming": True},
        )
    ]


def test_hf_does_not_retry_remote_when_matching_local_files_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_file = tmp_path / "snapshots" / "steam" / "reviews.jsonl"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("not relevant to mocked loader\n", encoding="utf-8")
    spec = {
        "id": "steam_reviews",
        "type": "hf_dataset",
        "name": "reapxdev/steam-reviews-scraper",
        "local_path": "snapshots/steam/reviews.jsonl",
        "local_format": "json",
    }
    agent = DataCollectionAgent(_write_config(tmp_path, [spec]))
    calls: list[str] = []

    def failing_load_dataset(name: str, *args: Any, **kwargs: Any) -> _FakeStream:
        calls.append(name)
        raise ValueError("invalid local snapshot")

    monkeypatch.setattr(collection_module, "hf_load_dataset", failing_load_dataset)

    with pytest.raises(ValueError, match="invalid local snapshot"):
        agent.load_dataset("reapxdev/steam-reviews-scraper")

    assert calls == ["json"]


def test_hf_steam_schema_filters_language_maps_bool_and_keeps_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steam_spec = {
        "id": "steam_reviews",
        "type": "hf_dataset",
        "name": "reapxdev/steam-reviews-scraper",
        "subset": "test-config",
        "revision": "test-revision",
        "token_env": "TEST_HF_TOKEN",
        "split": "train",
        "streaming": True,
        "text_field": "reviewText",
        "label_field": "votedUp",
        "label_map": {False: "negative", True: "positive"},
        "id_field": "reviewId",
        "filters": {"language": "english"},
        "metadata_fields": ["appId", "appName", "votesUp", "playtimeForeverHours"],
        "sample_size": 4,
        "per_label": 2,
    }
    agent = DataCollectionAgent(_write_config(tmp_path, [steam_spec]))
    records = [
        {
            "reviewId": "skip-de",
            "reviewText": "Tolles Spiel",
            "votedUp": True,
            "language": "german",
            "appId": 10,
            "appName": "Ignored",
            "votesUp": 1,
            "playtimeForeverHours": 2.0,
        },
        {
            "reviewId": "s-neg-1",
            "reviewText": "Broken matchmaking",
            "votedUp": False,
            "language": "english",
            "appId": 10,
            "appName": "Game A",
            "votesUp": 3,
            "playtimeForeverHours": 20.5,
        },
        {
            "reviewId": "s-pos-1",
            "reviewText": "Excellent campaign",
            "votedUp": True,
            "language": "english",
            "appId": 20,
            "appName": "Game B",
            "votesUp": 7,
            "playtimeForeverHours": 12.0,
        },
        {
            "reviewId": "s-neg-2",
            "reviewText": "Constant crashes",
            "votedUp": False,
            "language": "english",
            "appId": 20,
            "appName": "Game B",
            "votesUp": 4,
            "playtimeForeverHours": 5.5,
        },
        {
            "reviewId": "s-pos-2",
            "reviewText": "Creative and enjoyable",
            "votedUp": True,
            "language": "english",
            "appId": 10,
            "appName": "Game A",
            "votesUp": 9,
            "playtimeForeverHours": 30.0,
        },
    ]
    stream = _FakeStream(records)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(name: str, *args: Any, **kwargs: Any) -> _FakeStream:
        calls.append((name, args, kwargs))
        return stream

    monkeypatch.setenv("TEST_HF_TOKEN", "secret-from-environment")
    monkeypatch.setattr(collection_module, "hf_load_dataset", fake_load_dataset)
    frame = agent.load_dataset("reapxdev/steam-reviews-scraper")

    assert frame["label"].value_counts().to_dict() == {"negative": 2, "positive": 2}
    assert frame["source_id"].tolist() == ["s-neg-1", "s-neg-2", "s-pos-1", "s-pos-2"]
    assert "skip-de" not in set(frame["source_id"])
    assert frame["source"].eq("steam_reviews").all()
    assert frame["dataset_subset"].eq("test-config").all()
    assert frame.loc[frame["source_id"] == "s-neg-1", "appName"].item() == "Game A"
    assert {"appId", "appName", "votesUp", "playtimeForeverHours"}.issubset(frame)
    assert calls == [
        (
            "reapxdev/steam-reviews-scraper",
            ("test-config",),
            {
                "split": "train",
                "streaming": True,
                "revision": "test-revision",
                "token": "secret-from-environment",
            },
        )
    ]


def test_scrape_and_generic_api_are_offline_and_return_canonical_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = DataCollectionAgent(_write_config(tmp_path, []))

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        if url.endswith("/page"):
            return _FakeResponse(
                text=(
                    '<div class="review">First <b>review</b></div>'
                    '<div class="other">ignore</div>'
                    '<div class="review">Second review</div>'
                )
            )
        return _FakeResponse(
            payload={
                "results": [
                    {"id": "x1", "body": "API was good", "sentiment": "positive"},
                    {"id": "x2", "body": "API was bad", "sentiment": "negative"},
                ]
            }
        )

    monkeypatch.setattr(collection_module.requests, "get", fake_get)
    scraped = agent.scrape("https://example.test/page", ".review")
    api = agent.fetch_api("https://api.example.test/reviews", {"limit": 2})

    assert scraped["text"].tolist() == ["First review", "Second review"]
    assert scraped["label"].isna().all()
    assert api[["source_id", "label"]].to_dict("records") == [
        {"source_id": "x1", "label": "positive"},
        {"source_id": "x2", "label": "negative"},
    ]
    assert list(api.columns[: len(REQUIRED_COLUMNS)]) == list(REQUIRED_COLUMNS)


def test_run_isolates_failed_source_and_writes_manifest_raw_merged_and_eda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [
        {"id": "amazon", "type": "hf_dataset", "name": "fake/amazon"},
        {"id": "steam", "type": "hf_dataset", "name": "fake/steam"},
        {"id": "broken", "type": "api", "endpoint": "https://offline.invalid"},
    ]
    agent = DataCollectionAgent(_write_config(tmp_path, sources))

    amazon = agent._rows_to_canonical(
        [{"source_id": "a1", "text": "excellent purchase", "label": "positive"}],
        default_source="amazon",
    )
    steam = agent._rows_to_canonical(
        [{"source_id": "s1", "text": "awful game", "label": "negative"}],
        default_source="steam",
    )
    monkeypatch.setattr(
        agent,
        "load_dataset",
        lambda name, *args, **kwargs: amazon if name == "fake/amazon" else steam,
    )
    monkeypatch.setattr(
        agent,
        "fetch_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    merged = agent.run()

    assert len(merged) == 2
    assert set(merged["source"]) == {"amazon", "steam"}
    manifest_path = tmp_path / "data/raw/collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["successful_sources"] == 2
    broken = next(item for item in manifest["sources"] if item["id"] == "broken")
    assert broken["status"] == "failed"
    assert "ConnectionError" in broken["error"]
    assert (tmp_path / "data/raw/source_amazon.csv").exists()
    assert (tmp_path / "data/raw/source_steam.csv").exists()
    assert (tmp_path / "data/raw/reviews_raw.csv").exists()
    assert (tmp_path / "reports/eda/class_distribution.csv").exists()
    assert (tmp_path / "reports/eda/source_distribution.csv").exists()
    assert (tmp_path / "reports/eda/text_lengths.csv").exists()
    assert (tmp_path / "reports/eda/top_20_words.csv").exists()


def test_run_requires_two_non_empty_sources_and_still_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [
        {"id": "only", "type": "hf_dataset", "name": "fake/one"},
        {"id": "empty", "type": "api", "endpoint": "https://offline.invalid"},
    ]
    agent = DataCollectionAgent(_write_config(tmp_path, sources))
    one = agent._rows_to_canonical(
        [{"source_id": "1", "text": "one row", "label": "positive"}],
        default_source="only",
    )
    empty = agent._rows_to_canonical([], default_source="empty")
    monkeypatch.setattr(agent, "load_dataset", lambda *args, **kwargs: one)
    monkeypatch.setattr(agent, "fetch_api", lambda *args, **kwargs: empty)

    with pytest.raises(RuntimeError, match="At least 2 non-empty sources"):
        agent.run()

    manifest = json.loads(
        (tmp_path / "data/raw/collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["successful_sources"] == 1


def test_merge_normalizes_plain_frames_without_mutating_inputs(tmp_path: Path) -> None:
    agent = DataCollectionAgent(_write_config(tmp_path, []))
    first = pd.DataFrame({"id": [1], "review": ["Great"], "sentiment": ["positive"]})
    second = pd.DataFrame({"id": [2], "body": ["Bad"], "sentiment": ["negative"]})
    original_columns = first.columns.tolist()

    merged = agent.merge([first, second])

    assert list(merged.columns[: len(REQUIRED_COLUMNS)]) == list(REQUIRED_COLUMNS)
    assert merged["label"].tolist() == ["positive", "negative"]
    assert merged["record_id"].is_unique
    assert first.columns.tolist() == original_columns
