"""Technical fixtures exercise media hand-offs; no demonstration datasets are produced."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from pipeline.media_runner import MediaPipelineRunner, merge_video_intervals
from pipeline.runner import PipelineRunner


def config_file(
    tmp_path: Path, *, count: int = 3, al: bool = False, training: bool = False
) -> Path:
    source = tmp_path / "input"
    source.mkdir()
    for index in range(count):
        Image.new("RGB", (24, 24), (index * 17 % 255, 40, 255 - index * 17 % 255)).save(
            source / f"{index}.png"
        )
    config = {
        "project": {"modality": "image", "labels": ["warm", "cool"], "random_seed": 42},
        "task": {"id": "technical-test", "version": "1", "description": "Classify appearance"},
        "collection": {"sources": [{"type": "local", "path": str(source)}]},
        "training": {"enabled": training, "test_size": 0},
        "features": {"backend": "statistics"},
        "active_learning": {"enabled": al, "initial_size": 2, "n_iterations": 2, "batch_size": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def respond(manifest_path: str, *, count: int | None = None) -> Path:
    manifest = json.loads(Path(manifest_path).read_text())
    requests = manifest["requests"] if count is None else manifest["requests"][:count]
    answers = [
        {
            "record_id": item["record_id"],
            "request_id": item["request_id"],
            "task_id": manifest["task"]["task_id"],
            "task_version": manifest["task"]["version"],
            "label": ["warm", "cool"][index % 2],
            "needs_review": True,
            "reason": "Technical test response",
            "confidence": None,
            "score_type": None,
            "viewed_frames": [frame["timestamp_sec"] for frame in item["frames"]],
        }
        for index, item in enumerate(requests)
    ]
    path = Path(manifest["response_path"])
    path.write_text("".join(json.dumps(row) + "\n" for row in answers))
    return path


def review(queue_path: str, decisions_path: str, *, count: int | None = None) -> None:
    rows = pd.read_json(queue_path, lines=True).to_dict("records")
    if count is not None:
        rows = rows[:count]
    decisions = [
        {
            "record_id": row["record_id"],
            "request_id": row["request_id"],
            "review_status": "accepted",
            "reviewed": True,
            "reviewer": "Test fixture reviewer",
            "reviewed_at": "2026-09-21T00:00:00+00:00",
        }
        for row in rows
    ]
    Path(decisions_path).write_text("".join(json.dumps(row) + "\n" for row in decisions))


def test_partial_annotation_and_review_resume_without_inventing_humans(tmp_path):
    config = config_file(tmp_path)
    runner = MediaPipelineRunner(config)
    first = runner.run()
    assert first.status == "annotation_required", first.message
    assert not Path(first.artifacts["review_queue"]).exists()
    respond(first.artifacts["annotation_requests"], count=1)
    partial = MediaPipelineRunner(config).run()
    assert partial.status == "annotation_required", partial.message
    respond(first.artifacts["annotation_requests"])
    waiting = MediaPipelineRunner(config).run()
    assert waiting.status == "review_required", waiting.message
    queue = pd.read_json(waiting.artifacts["review_queue"], lines=True)
    assert queue["final_label"].isna().all()
    assert not queue["reviewed"].any()
    review(waiting.artifacts["review_queue"], waiting.artifacts["review_decisions"], count=1)
    partial_review = MediaPipelineRunner(config).run()
    assert partial_review.status == "review_required", partial_review.message
    review(waiting.artifacts["review_queue"], waiting.artifacts["review_decisions"])
    done = MediaPipelineRunner(config).run()
    assert done.status == "completed", done.message
    assert done.metrics["hitl"]["verified"]
    assert len(pd.read_json(done.artifacts["labeled_dataset"], lines=True)) == 3
    saved = Path(done.artifacts["labeled_dataset"]).read_bytes()
    assert MediaPipelineRunner(config).run().status == "completed"
    assert Path(done.artifacts["labeled_dataset"]).read_bytes() == saved
    with pytest.raises(ValueError, match="new --run-id"):
        MediaPipelineRunner(config).run(force=True)


def test_config_edit_requires_isolated_new_run(tmp_path):
    config = config_file(tmp_path)
    first = MediaPipelineRunner(config).run()
    document = yaml.safe_load(config.read_text())
    document["task"]["version"] = "2"
    config.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="Configuration changed"):
        MediaPipelineRunner(config).run()
    second = MediaPipelineRunner(config, run_id="v2").run()
    assert second.status == "annotation_required", second.message
    assert first.artifacts["state"] != second.artifacts["state"]


def test_cli_status_is_read_only_and_existing_runner_dispatches_media(tmp_path, monkeypatch, capsys):
    import run_pipeline

    config = config_file(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--config", str(config), "--status"])
    assert run_pipeline.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_started"
    assert not (tmp_path / "data/runs").exists()
    prepared = PipelineRunner(config).run()
    assert prepared.status == "annotation_required", prepared.message


def test_concurrent_run_cannot_replace_state(tmp_path):
    import fcntl

    config = config_file(tmp_path)
    runner = MediaPipelineRunner(config)
    runner.run_dir.mkdir(parents=True)
    with (runner.run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            runner.run()
    assert not runner.state_path.exists()


def test_live_al_requests_new_unlabeled_batches_and_trains(tmp_path):
    config = config_file(tmp_path, count=8, al=True, training=True)
    current = MediaPipelineRunner(config).run()
    all_ids = set()
    for iteration in range(3):
        assert current.status == "annotation_required", current.message
        manifest = json.loads(Path(current.artifacts["annotation_requests"]).read_text())
        ids = {row["record_id"] for row in manifest["requests"]}
        assert len(ids) == 2 and not ids & all_ids
        all_ids |= ids
        respond(current.artifacts["annotation_requests"])
        current = MediaPipelineRunner(config).run()
        assert current.status == "review_required", current.message
        review(current.artifacts["review_queue"], current.artifacts["review_decisions"])
        current = MediaPipelineRunner(config).run()
    assert current.status == "completed", current.message
    assert Path(current.artifacts["model"]).is_file()
    assert current.metrics["training"]["train_rows"] == 6
    assert current.metrics["training"]["evaluation_mode"] == "not_evaluated"
    assert [point["n_labeled"] for point in current.metrics["active_learning"]["history"]] == [
        2,
        4,
        6,
    ]
    assert current.metrics["annotation"]["unqueried_units"] == 2


def test_auto_only_is_never_reported_as_human_review(tmp_path):
    config = config_file(tmp_path)
    first = MediaPipelineRunner(config).run(review_mode="auto-only")
    respond(first.artifacts["annotation_requests"])
    done = MediaPipelineRunner(config).run(review_mode="auto-only")
    assert done.status == "completed_without_hitl", done.message
    assert not done.metrics["hitl"]["verified"]
    assert done.metrics["hitl"]["reviewed_rows"] == 0


def test_intervals_respect_gaps_rejections_and_actual_human_boundaries():
    frame = pd.DataFrame(
        [
            {
                "record_id": "a",
                "asset_id": "v",
                "modality": "video",
                "media_path": "x",
                "start_sec": 0,
                "end_sec": 5,
                "final_label": "walk",
                "human_start_sec": 1,
                "human_end_sec": 5,
            },
            {
                "record_id": "b",
                "asset_id": "v",
                "modality": "video",
                "media_path": "x",
                "start_sec": 5,
                "end_sec": 10,
                "final_label": "walk",
                "human_start_sec": 5,
                "human_end_sec": 8,
            },
            {
                "record_id": "c",
                "asset_id": "v",
                "modality": "video",
                "media_path": "x",
                "start_sec": 10,
                "end_sec": 15,
                "final_label": "walk",
                "human_start_sec": 11,
                "human_end_sec": 15,
            },
        ]
    )
    intervals = merge_video_intervals(frame)
    assert [(part["start_sec"], part["end_sec"]) for part in intervals] == [(1, 8), (11, 15)]
    assert intervals[0]["record_ids"] == ["a", "b"]
