from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.contracts import TaskSpec
from hitl_app import load_queue, save_media_decision, save_text_review


def test_text_save_persists_submitted_reviewed_flags_by_id(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("annotation:\n  backend: lexicon\n")
    queue = pd.DataFrame(
        {
            "record_id": ["one", "two"], "text": ["good", "bad"],
            "auto_label": ["positive", "negative"], "confidence": [0.7, 0.8],
            "human_label": ["", ""], "reviewer": ["", ""], "reviewed": [False, False],
        }
    )
    edited = queue.iloc[::-1].copy()
    edited["reviewed"] = True
    edited["reviewer"] = "Human"
    output = tmp_path / "corrected.csv"
    saved = save_text_review(queue, edited, output, config=config)
    assert saved["reviewed"].tolist() == [True, True]
    assert saved["human_label"].tolist() == ["positive", "negative"]
    assert pd.read_csv(output)["reviewed"].tolist() == [True, True]
    assert not queue["reviewed"].any()
    edited.loc[0, "reviewed"] = False
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="every row"):
        save_text_review(queue, edited, output, config=config)
    assert output.read_bytes() == previous


def test_text_load_does_not_treat_false_csv_text_as_confirmation(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.csv"
    queue_path.write_text("record_id,text,auto_label,reviewed,source_label\na,sample,positive,False,negative\n")
    queue = load_queue(queue_path)
    assert not bool(queue.loc[0, "reviewed"])
    assert "source_label" not in queue


def test_media_save_is_partial_and_preserves_previous_decision_events(tmp_path: Path) -> None:
    task = TaskSpec.from_config({"project": {"modality": "image", "labels": ["a", "b"]}})
    queue = pd.DataFrame(
        {
            "record_id": ["one", "two"], "request_id": ["request-one", "request-two"],
            "auto_label": ["a", "b"], "start_sec": [0.0, 0.0], "end_sec": [0.0, 0.0],
        }
    )
    decision = {
        "record_id": "one", "request_id": "request-one", "review_status": "accepted",
        "reviewed": True, "reviewer": "Human", "reviewed_at": "2026-09-21T12:00:00+00:00",
    }
    output = tmp_path / "review_decisions.jsonl"
    first = save_media_decision(queue, decision, task, output)
    assert first["reviewed"].tolist() == [True, False]
    assert first["final_label"].tolist() == ["a", None]
    assert len(output.read_text().splitlines()) == 1
    updated = {
        **decision, "review_status": "edited", "human_label": "b",
        "reviewed_at": "2026-09-21T12:01:00+00:00",
    }
    save_media_decision(queue, updated, task, output)
    latest = json.loads(output.read_text().strip())
    assert latest["human_label"] == "b"
    events = [json.loads(line) for line in (tmp_path / "review_events.jsonl").read_text().splitlines()]
    assert [event["review_status"] for event in events] == ["accepted", "edited"]
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="confirmation"):
        save_media_decision(queue, {**updated, "reviewed": False}, task, output)
    assert output.read_bytes() == previous
    with pytest.raises(ValueError, match="reviewer"):
        save_media_decision(queue, {**updated, "reviewer": ""}, task, output)
    assert output.read_bytes() == previous
