import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from agents.al_agent import ActiveLearningAgent
from agents.features import extract_features, feature_matrix
from agents.train_agent import TrainAgent


def vectors(prefix):
    return pd.DataFrame(
        [
            {
                "record_id": f"{prefix}-{index}",
                "group_id": f"{prefix}-group-{index}",
                "features": [float(index % 3), float(index % 3) ** 2],
                "final_label": ["red", "green", "blue"][index % 3],
                "source_label": ["red", "green", "blue"][index % 3],
            }
            for index in range(12)
        ]
    )


def test_multiclass_media_model_saved_and_evaluated_with_group_guard(tmp_path):
    train, test = vectors("train"), vectors("test")
    trainer = TrainAgent({"feature_column": "features", "labels": ["red", "green", "blue"]})
    target = tmp_path / "model.joblib"
    metrics = trainer.train_and_evaluate(
        train, test_df=test, model_path=target, metrics_path=tmp_path / "metrics.json"
    )
    assert np.asarray(metrics["confusion_matrix"]).shape == (3, 3)
    assert metrics["accuracy"] == 1
    model = joblib.load(target)
    assert model.predict(feature_matrix(test)).tolist() == test["source_label"].tolist()
    assert json.loads(target.with_suffix(".metadata.json").read_text())["classes"] == [
        "red",
        "green",
        "blue",
    ]
    test.loc[0, "group_id"] = train.loc[0, "group_id"]
    with pytest.raises(ValueError, match="group_id overlap"):
        trainer.train_media(train, test_df=test, model_path=target)
    with pytest.raises(ValueError, match="requires a non-empty group_id"):
        trainer.train_media(
            train, test_df=vectors("test").drop(columns="group_id"), model_path=target
        )


def test_visual_al_queries_without_source_labels_and_checks_bad_features():
    agent = ActiveLearningAgent(config={"feature_column": "features"})
    train, pool = vectors("train"), vectors("pool")
    agent.fit(train, label_col="final_label")
    pool = pool.drop(columns=["source_label", "final_label"])
    for strategy in ("entropy", "margin", "random"):
        selected = agent.query(pool, strategy, batch_size=4)
        assert len(selected) == len(set(selected)) == 4
    pool.at[0, "features"] = [float("nan"), 0]
    with pytest.raises(ValueError, match="finite"):
        agent.query(pool)


def test_feature_cache_binds_pixels_and_temporal_order(tmp_path):
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (16, 16), "red").save(red)
    Image.new("RGB", (16, 16), "blue").save(blue)
    rows = pd.DataFrame(
        [
            {
                "record_id": "video",
                "modality": "video",
                "frames": [
                    {"path": str(red), "timestamp_sec": 0},
                    {"path": str(blue), "timestamp_sec": 1},
                ],
            }
        ]
    )
    config = {"backend": "statistics", "cache_dir": str(tmp_path / "cache")}
    first = extract_features(rows, config)
    assert first["features"].iloc[0] == extract_features(rows, config)["features"].iloc[0]
    rows.at[0, "frames"] = [
        {"path": str(blue), "timestamp_sec": 0},
        {"path": str(red), "timestamp_sec": 1},
    ]
    reverse = extract_features(rows, config)
    assert reverse["feature_key"].iloc[0] != first["feature_key"].iloc[0]
    assert not np.array_equal(feature_matrix(first), feature_matrix(reverse))
    Image.new("RGB", (16, 16), "green").save(red)
    changed = extract_features(rows, config)
    assert changed["feature_key"].iloc[0] != reverse["feature_key"].iloc[0]


def test_human_interval_controls_training_evidence(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (8, 8)).save(path)
    rows = pd.DataFrame(
        [
            {
                "record_id": "v",
                "modality": "video",
                "human_start_sec": 1,
                "human_end_sec": 2,
                "frames": [{"path": str(path), "timestamp_sec": 0}],
            }
        ]
    )
    with pytest.raises(ValueError, match="No frames inside"):
        extract_features(rows)
