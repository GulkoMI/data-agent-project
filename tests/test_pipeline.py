from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.runner import PipelineRunner


def _project(tmp_path: Path, rows: int = 120) -> Path:
    config = {
        "project": {"random_seed": 42},
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "labeled_dir": "data/labeled",
            "review_dir": "data/review",
            "cache_dir": "data/cache",
            "reports_dir": "reports",
            "models_dir": "models",
        },
        "collection": {"output": "data/raw/reviews_raw.parquet", "sources": []},
        "quality": {
            "selected_strategy": "conservative",
            "strategies": {
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
            },
            "outlier": {"iqr_multiplier": 1.5, "z_threshold": 3.0},
        },
        "annotation": {
            "backend": "transformers",
            "model": "unused-offline",
            "batch_size": 8,
            "max_length": 128,
            "confidence_threshold": 0.8,
            "review_target": 12,
            "labels": ["negative", "positive"],
            "allow_lexicon_fallback": False,
        },
        "active_learning": {
            "model": "logreg",
            "initial_size": 20,
            "n_iterations": 2,
            "batch_size": 5,
            "strategies": ["entropy", "random"],
            "test_size": 0.2,
            "random_seed": 42,
            "tfidf": {"max_features": 500, "ngram_range": [1, 2]},
        },
        "training": {
            "test_size": 0.2,
            "random_seed": 42,
            "max_features": 500,
            "ngram_range": [1, 2],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    fixture_dir = tmp_path / "data/fixtures"
    fixture_dir.mkdir(parents=True)
    records = []
    for index in range(rows):
        positive = index % 2 == 0
        source = "fixture:amazon" if index % 4 < 2 else "fixture:steam"
        records.append(
            {
                "record_id": f"row-{index:03d}",
                "text": (
                    f"excellent useful enjoyable review number {index}"
                    if positive
                    else f"terrible broken frustrating review number {index}"
                ),
                "audio": None,
                "image": None,
                "label": "positive" if positive else "negative",
                "source": source,
                "source_id": str(index),
                "collected_at": "2026-01-01T00:00:00+00:00",
            }
        )
    pd.DataFrame(records).to_csv(fixture_dir / "reviews_fixture.csv", index=False)
    return config_path


def test_offline_pipeline_requires_exact_review_then_completes(tmp_path: Path) -> None:
    runner = PipelineRunner(_project(tmp_path))
    first = runner.run(offline=True, force=True, review_mode="required")

    assert first.status == "review_required"
    assert not (tmp_path / "models/smoke/sentiment_model.joblib").exists()
    queue = pd.read_csv(first.artifacts["review_queue"])
    assert "source_label" not in queue.columns
    assert "label" not in queue.columns
    tasks = json.loads(Path(first.artifacts["labelstudio_import"]).read_text(encoding="utf-8"))
    assert all("source_label" not in task["data"] for task in tasks)

    partial = queue.iloc[:-1].copy()
    partial["human_label"] = ""
    partial["reviewer"] = "pytest-reviewer"
    partial.to_csv(runner.corrected_path, index=False)
    with pytest.raises(ValueError, match="cover the current queue exactly"):
        runner.run(offline=True, review_mode="required")

    corrected = queue.copy()
    corrected["human_label"] = ""
    corrected["reviewer"] = "pytest-reviewer"
    corrected.to_csv(runner.corrected_path, index=False)
    second = runner.run(offline=True, review_mode="required")

    assert second.status == "completed"
    assert second.metrics["hitl"]["verified"] is True
    assert second.metrics["hitl"]["auto_vs_human"]["n_compared"] == len(queue)
    assert Path(second.artifacts["model"]).exists()
    outer_manifest = json.loads(
        (tmp_path / "reports/smoke/active_learning/outer_split_manifest.json").read_text()
    )
    al_results = json.loads(
        (tmp_path / "reports/smoke/active_learning/al_results.json").read_text()
    )
    al_seen = set(al_results["split_manifest"]["initial_record_ids"])
    al_seen.update(al_results["split_manifest"]["pool_record_ids"])
    al_seen.update(al_results["split_manifest"]["test_record_ids"])
    assert al_seen.isdisjoint(outer_manifest["outer_holdout_record_ids"])


def test_outer_holdout_is_independent_of_reviewer_state(tmp_path: Path) -> None:
    runner = PipelineRunner(_project(tmp_path))
    frame = pd.read_csv(tmp_path / "data/fixtures/reviews_fixture.csv")
    frame["source_label"] = frame["label"]
    frame["final_label"] = frame["label"]
    frame["reviewer"] = pd.NA

    _, _, first_holdout = runner._active_learning(frame)
    reviewed = frame.copy()
    reviewed["reviewer"] = "pytest-reviewer"
    reviewed.loc[0, "final_label"] = "negative"
    _, reviewed_training, second_holdout = runner._active_learning(reviewed)

    assert set(first_holdout["record_id"]) == set(second_holdout["record_id"])
    assert set(reviewed_training["training_label_origin"]) == {"human_review"}
    if "row-000" in set(reviewed_training["record_id"]):
        row = reviewed_training.set_index("record_id").loc["row-000"]
        assert row["training_label"] == "negative"
