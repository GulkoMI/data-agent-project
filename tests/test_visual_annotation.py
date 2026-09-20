from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from agents.annotation_agent import AnnotationAgent
from agents.contracts import TaskSpec
from agents.media_review import apply_media_review, build_media_review_queue
from agents.visual_annotation import VisualAnnotationAgent


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec.from_config(
        {
            "project": {"modality": "video", "labels": ["moving", "still"]},
            "task": {
                "id": "movement", "version": "2",
                "classes": [
                    {"name": "moving", "definition": "Visible position change"},
                    {"name": "still", "definition": "No visible position change"},
                ],
            },
        }
    )


@pytest.fixture
def units(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index in range(3):
        frames = []
        for position in range(2):
            # Deliberately revealing source names must never occur in the packet.
            path = tmp_path / f"hidden_source_label_still_{index}_{position}.png"
            Image.new("RGB", (16, 12), color=(index * 30, position * 120, 80)).save(path)
            frames.append({"path": str(path), "timestamp_sec": float(index * 2 + position)})
        rows.append(
            {
                "record_id": f"unit-{index}", "asset_id": f"asset-{index}",
                "group_id": f"group-{index}", "content_hash": f"asset-content-{index}",
                "modality": "video", "start_sec": float(index * 2),
                "end_sec": float(index * 2 + 2), "frames": frames,
                "label": "still", "source_label": "still",
            }
        )
    return pd.DataFrame(rows, index=[4, 8, 11])


def _packet(agent: VisualAnnotationAgent, units: pd.DataFrame, tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "requests"
    manifest = agent.prepare_requests(units, directory, batch_id="initial")
    return directory, json.loads(manifest.read_text())


def _answer(request: dict, *, label: str | None = "moving") -> dict:
    return {
        "record_id": request["record_id"], "request_id": request["request_id"],
        "task_id": "movement", "task_version": "2", "label": label,
        "needs_review": label is None, "reason": "Observed visual evidence.",
        "confidence": None, "score_type": None,
        "viewed_frames": [frame["timestamp_sec"] for frame in request["frames"]],
    }


def _write(path: Path, answers: list[dict]) -> Path:
    path.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
    return path


def test_packet_hides_source_labels_and_factory_preserves_text(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = AnnotationAgent.for_task(task)
    assert isinstance(agent, VisualAnnotationAgent)
    directory, manifest = _packet(agent, units, tmp_path)
    contents = (directory / "manifest.json").read_text()
    assert '"source_label"' not in contents
    assert "hidden_source_label" not in contents
    assert "source_path" not in contents
    assert manifest["task"]["classes"][0]["definition"] == "Visible position change"
    for request in manifest["requests"]:
        for frame in request["frames"]:
            assert Path(frame["path"]).is_relative_to(directory / "frames")
            with Image.open(frame["path"]) as pixels:
                assert pixels.size == (16, 12)
    text = AnnotationAgent(config={"backend": "lexicon"})
    assert text.auto_label(pd.DataFrame({"text": ["Wonderful excellent game"]}))["auto_label"].item() == "positive"


def test_partial_out_of_order_import_and_resume_preserve_human_decisions(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    answer = _answer(manifest["requests"][2])
    response = _write(tmp_path / "partial.jsonl", [answer])
    partial = agent.import_responses(units, directory, response)
    assert partial.index.tolist() == [4, 8, 11]
    assert partial["annotation_status"].tolist() == ["pending", "pending", "annotated"]
    assert partial["final_label"].isna().all()
    assert not partial["reviewed"].any()
    assert partial["needs_review"].all()
    reviewed = apply_media_review(
        partial,
        pd.DataFrame({"record_id": ["unit-2"], "review_status": ["accepted"]}), task,
        reviewer="Actual user", reviewed_at="2026-09-21T12:00:00+00:00",
    )
    repeated = agent.import_responses(reviewed, directory, response)
    assert repeated.loc[11, "final_label"] == "moving"
    assert repeated.loc[11, "reviewer"] == "Actual user"
    assert bool(repeated.loc[11, "reviewed"])
    assert repeated.loc[11, "source_label"] == "still"
    assert repeated.loc[11, "review_history"] == reviewed.loc[11, "review_history"]
    rest = _write(
        tmp_path / "remaining.jsonl",
        [_answer(manifest["requests"][1], label=None), _answer(manifest["requests"][0])],
    )
    completed = agent.import_responses(repeated, directory, rest)
    assert completed["annotation_status"].tolist() == ["annotated", "abstained", "annotated"]
    assert len((directory / "accepted_responses.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("record_id", "unknown", "unknown"),
        ("request_id", "old", "Stale"),
        ("task_version", "1", "Stale"),
        ("task_id", "another-task", "Stale"),
        ("label", "unexpected", "Unsupported"),
        ("confidence", float("nan"), "finite"),
        ("confidence", 0.8, "self_reported"),
        ("needs_review", "false", "boolean"),
        ("viewed_frames", [], "evidence"),
        ("viewed_frames", [100.0], "unavailable"),
        ("viewed_frames", [0.0, 0.0], "duplicate"),
        ("end_sec", 3.0, "outside"),
        ("human_label", "still", "unknown"),
    ],
)
def test_invalid_answers_are_atomic(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path, field: str, value: object, error: str
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    valid = _answer(manifest["requests"][1])
    invalid = {**_answer(manifest["requests"][0]), field: value}
    response = _write(tmp_path / "invalid.jsonl", [valid, invalid])
    with pytest.raises(ValueError, match=error):
        agent.import_responses(units, directory, response)
    assert not (directory / "accepted_responses.jsonl").exists()


def test_duplicate_and_conflicting_answers_do_not_replace_accepted_data(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    answer = _answer(manifest["requests"][0])
    response = _write(tmp_path / "responses.jsonl", [answer, answer])
    with pytest.raises(ValueError, match="Duplicate"):
        agent.import_responses(units, directory, response)
    _write(response, [answer])
    agent.import_responses(units, directory, response)
    accepted = (directory / "accepted_responses.jsonl").read_bytes()
    _write(response, [{**answer, "label": "still"}])
    with pytest.raises(ValueError, match="Conflicting"):
        agent.import_responses(units, directory, response)
    assert (directory / "accepted_responses.jsonl").read_bytes() == accepted


def test_pending_frame_refresh_invalidates_old_response(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    old_answer = _answer(manifest["requests"][0])
    denser = units.copy(deep=True)
    new_path = tmp_path / "extra.png"
    Image.new("RGB", (16, 12), color="red").save(new_path)
    denser.at[4, "frames"] = [
        units.loc[4, "frames"][0], {"path": str(new_path), "timestamp_sec": 0.5},
        units.loc[4, "frames"][1],
    ]
    updated = json.loads(agent.prepare_requests(denser, directory, batch_id="initial").read_text())
    assert updated["requests"][0]["request_id"] != old_answer["request_id"]
    response = _write(tmp_path / "old.jsonl", [old_answer])
    with pytest.raises(ValueError, match="Stale"):
        agent.import_responses(denser, directory, response)
    response = _write(tmp_path / "new.jsonl", [_answer(updated["requests"][0])])
    agent.import_responses(denser, directory, response)
    with pytest.raises(ValueError, match="answered request"):
        agent.prepare_requests(units, directory, batch_id="initial")


def test_changed_frames_or_task_invalidate_packet(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    other = TaskSpec.from_config({"project": {"modality": "video", "labels": ["moving", "still"]}})
    with pytest.raises(ValueError, match="Stale"):
        VisualAnnotationAgent(other).import_responses(units, directory)
    packet_frame = Path(manifest["requests"][0]["frames"][0]["path"])
    original_packet_bytes = packet_frame.read_bytes()
    packet_frame.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="Packet frame content changed"):
        agent.import_responses(units, directory)
    with pytest.raises(ValueError, match="Packet frame content changed"):
        agent.prepare_requests(units, directory, batch_id="initial")
    packet_frame.write_bytes(original_packet_bytes)
    agent.prepare_requests(units, directory, batch_id="initial")
    Image.new("RGB", (16, 12), color="purple").save(units.loc[4, "frames"][0]["path"])
    with pytest.raises(ValueError, match="media or sampling changed"):
        agent.import_responses(units, directory)


def test_source_bytes_are_verified_for_image_units(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="blue").save(source)
    task = TaskSpec.from_config({"project": {"modality": "image", "labels": ["blue", "red"]}})
    units = pd.DataFrame([
        {
            "record_id": "opaque-id", "media_path": str(source),
            "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
            "modality": "image", "start_sec": 0.0, "end_sec": 0.0,
            "frames": [{"path": str(source), "timestamp_sec": 0.0}],
        }
    ])
    agent = VisualAnnotationAgent(task)
    directory = tmp_path / "image-request"
    agent.prepare_requests(units, directory, batch_id="images")
    Image.new("RGB", (8, 8), color="red").save(source)
    with pytest.raises(ValueError, match="Source content changed"):
        agent.import_responses(units, directory)


def test_parquet_nested_frames_and_manifest_integrity(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "units.parquet"
    units.to_parquet(path)
    roundtrip = pd.read_parquet(path)
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, roundtrip, tmp_path)
    imported = agent.import_responses(roundtrip, directory)
    assert imported["annotation_status"].tolist() == ["pending"] * 3
    manifest["requests"][0]["start_sec"] = -1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest content changed"):
        agent.import_responses(roundtrip, directory)


def test_abstention_with_no_inspected_frames_remains_unreviewed(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    abstention = {**_answer(manifest["requests"][0], label=None), "viewed_frames": []}
    response = _write(tmp_path / "abstain.jsonl", [abstention])
    result = agent.import_responses(units, directory, response)
    assert result.loc[4, "annotation_status"] == "abstained"
    assert result.loc[4, "auto_label"] is None
    assert result.loc[4, "final_label"] is None
    assert not bool(result.loc[4, "reviewed"])
    assert bool(result.loc[4, "needs_review"])


def test_human_review_explicit_partial_edits_and_history(
    task: TaskSpec, units: pd.DataFrame, tmp_path: Path
) -> None:
    agent = VisualAnnotationAgent(task)
    directory, manifest = _packet(agent, units, tmp_path)
    response = _write(tmp_path / "answers.jsonl", [_answer(row) for row in manifest["requests"]])
    annotated = agent.import_responses(units, directory, response)
    queue = build_media_review_queue(annotated, task)
    changes = pd.DataFrame(
        {
            "record_id": ["unit-0", "unit-1", "unit-2"],
            "review_status": ["edited", "accepted", "pending"],
            "reviewed": [True, True, False], "human_label": ["still", None, None],
            "human_start_sec": [0.5, None, None], "human_end_sec": [1.5, None, None],
        }
    )
    reviewed = apply_media_review(
        queue, changes, task, reviewer="Human", reviewed_at="2026-09-21T12:00:00+00:00"
    )
    assert reviewed["reviewed"].tolist() == [True, True, False]
    assert reviewed["final_label"].tolist() == ["still", "moving", None]
    assert reviewed["review_changed"].tolist() == [True, False, False]
    assert reviewed.loc[4, "human_start_sec"] == 0.5
    assert reviewed.loc[4, "auto_start_sec"] == 0.0
    assert reviewed.loc[4, "source_label"] == "still"
    replay = apply_media_review(
        reviewed, changes, task, reviewer="Human", reviewed_at="2026-09-21T12:00:00+00:00"
    )
    assert len(replay.loc[4, "review_history"]) == 1
    undated_replay = apply_media_review(reviewed, changes, task, reviewer="Human")
    assert len(undated_replay.loc[4, "review_history"]) == 1
    assert undated_replay.loc[4, "reviewed_at"] == reviewed.loc[4, "reviewed_at"]
    rejected = apply_media_review(
        reviewed,
        pd.DataFrame({"record_id": ["unit-0", "unit-1"], "review_status": ["rejected", "uncertain"]}),
        task, reviewer="Human", reviewed_at="2026-09-21T12:01:00+00:00",
    )
    assert rejected["final_label"].isna().all()
    assert not bool(rejected.loc[4, "needs_review"])
    assert bool(rejected.loc[8, "needs_review"])
    assert len(rejected.loc[4, "review_history"]) == 2


@pytest.mark.parametrize(
    ("decision", "error"),
    [
        ({"record_id": "unknown", "review_status": "accepted"}, "unknown"),
        ({"record_id": "unit-0", "review_status": "pending"}, "Invalid"),
        ({"record_id": "unit-0", "review_status": "accepted"}, "absent auto_label"),
        ({"record_id": "unit-0", "review_status": "edited", "human_label": "bad"}, "Unsupported"),
        ({"record_id": "unit-0", "review_status": "edited"}, "requires human_label"),
        (
            {"record_id": "unit-0", "review_status": "edited", "human_label": "still", "human_end_sec": 99},
            "outside",
        ),
    ],
)
def test_invalid_human_decisions_do_not_change_input(
    task: TaskSpec, units: pd.DataFrame, decision: dict, error: str
) -> None:
    queue = build_media_review_queue(units, task)
    with pytest.raises(ValueError, match=error):
        apply_media_review(queue, pd.DataFrame([decision]), task, reviewer="Human")
    assert not queue["reviewed"].any()
    assert queue["final_label"].isna().all()
