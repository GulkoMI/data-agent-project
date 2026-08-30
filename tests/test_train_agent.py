from __future__ import annotations

import pandas as pd

from agents.train_agent import TrainAgent


def test_final_model_uses_final_labels_and_disjoint_gold_holdout(tmp_path) -> None:
    rows = []
    for index in range(120):
        label = "positive" if index % 2 == 0 else "negative"
        rows.append(
            {
                "record_id": f"id-{index}",
                "text": "great enjoyable product" if label == "positive" else "bad broken product",
                "source_label": label,
                "final_label": label,
            }
        )
    metrics = TrainAgent({"random_seed": 7, "test_size": 0.25}).train_and_evaluate(
        pd.DataFrame(rows),
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
    )
    assert set(metrics["train_record_ids"]).isdisjoint(metrics["test_record_ids"])
    assert metrics["accuracy"] == 1.0
    assert (tmp_path / "model.joblib").exists()


def test_final_model_accepts_explicit_outer_holdout(tmp_path) -> None:
    rows = []
    for index in range(120):
        label = "positive" if index % 2 == 0 else "negative"
        rows.append(
            {
                "record_id": f"id-{index}",
                "text": "great enjoyable product" if label == "positive" else "bad broken product",
                "source_label": label,
                "final_label": label,
            }
        )
    frame = pd.DataFrame(rows)
    metrics = TrainAgent({"random_seed": 7}).train_and_evaluate(
        frame.iloc[:80],
        test_df=frame.iloc[80:],
        model_path=tmp_path / "outer-model.joblib",
        metrics_path=tmp_path / "outer-metrics.json",
    )
    assert metrics["evaluation_mode"] == "explicit_outer_holdout"
    assert set(metrics["train_record_ids"]).isdisjoint(metrics["test_record_ids"])
