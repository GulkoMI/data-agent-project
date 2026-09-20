"""Local decode, provenance and temporal identity tests; no external dataset runs."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from agents.contracts import TaskSpec
from agents.media_collection import collect_media_sources
from agents.media_quality import MediaQualityAgent
from utils.media import prepare_media, sample_video_frames


def _task(modality="image", **overrides):
    return TaskSpec.from_config(
        {
            "project": {"modality": modality, "labels": ["present", "absent"]},
            "task": {"id": "test", "version": "1", **overrides},
        }
    )


def _image(path, *, color="red", size=(20, 20)):
    Image.new("RGB", size, color=color).save(path)
    return path


def test_task_classes_validate_and_roundtrip():
    task = _task(
        classes=[
            {"name": "present", "definition": "Object is visible", "examples": ["A clear object"]},
            {"name": "absent", "definition": "Object not visible", "examples": []},
        ],
        boundary_rules=["Abstain if the object cannot be identified."],
    )
    assert task == TaskSpec.from_config(task.to_dict())
    assert task.labels == ("present", "absent")
    assert task.fingerprint != _task().fingerprint
    with pytest.raises(ValueError, match="distinct"):
        TaskSpec.from_config({"project": {"modality": "image", "labels": ["one", "one"]}})
    with pytest.raises(ValueError, match="match"):
        _task(classes=[{"name": "other"}, {"name": "absent"}])
    with pytest.raises(ValueError, match="image or video"):
        TaskSpec.from_config({"project": {"modality": "boxes", "labels": ["a", "b"]}})


def test_collect_corrupt_and_unlabeled_assets_non_destructively(tmp_path):
    original = _image(tmp_path / "secret_gold_present.png")
    (tmp_path / "broken.png").write_bytes(b"broken media")
    rows = collect_media_sources([{"id": "local", "path": str(tmp_path)}], tmp_path, "image")
    assert len(rows) == 2
    quality = MediaQualityAgent()
    report = quality.detect_issues(rows)
    assert report["invalid"] == 1
    assert report["unlabeled"] == 2
    clean = quality.fix(rows)
    assert len(clean) == 1
    assert clean["label"].isna().all()
    assert original.exists() and (tmp_path / "broken.png").exists()
    again = collect_media_sources([{"id": "local", "path": str(tmp_path)}], tmp_path, "image")
    assert rows["record_id"].tolist() == again["record_id"].tolist()


def test_manifest_reference_labels_hidden_and_duplicate_groups_are_transitive(tmp_path):
    first = _image(tmp_path / "first.png", color="red")
    second = _image(tmp_path / "second.png", color="blue")
    copied = tmp_path / "copied.png"
    shutil.copyfile(first, copied)
    manifest = tmp_path / "assets.jsonl"
    entries = [
        {"path": first.name, "group_id": "session-A", "source_label": "present"},
        {"path": second.name, "group_id": "session-B", "label": "absent"},
        {"path": copied.name, "group_id": "session-B"},
    ]
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
    rows = collect_media_sources([{"type": "manifest", "path": manifest.name}], tmp_path, "image")
    assert len(rows) == 3
    assert rows["label"].isna().all()
    assert rows["source_label"].iloc[:2].tolist() == ["present", "absent"]
    assert rows["group_id"].nunique() == 1
    assert rows["asset_id"].nunique() == 2
    assert len(MediaQualityAgent().fix(rows)) == 2


def test_csv_manifest_and_local_glob(tmp_path):
    _image(tmp_path / "one.png")
    (tmp_path / "manifest.csv").write_text(
        "path,group_id,source_label\none.png,session-1,present\n",
        encoding="utf-8",
    )
    manifested = collect_media_sources(
        [{"type": "manifest", "path": "manifest.csv"}],
        tmp_path,
        "image",
    )
    assert manifested.iloc[0]["source_label"] == "present"
    assert manifested.iloc[0]["group_id"] == "session-1"
    assert len(collect_media_sources([{"path": "*.png"}], tmp_path, "image")) == 1
    with pytest.raises(ValueError, match="local files only"):
        collect_media_sources(["https://example.org/picture.png"], tmp_path, "image")


def test_prepare_images_sanitizes_metadata_and_keeps_stable_identity(tmp_path):
    original = tmp_path / "present_secret.png"
    info = Image.Exif()
    info[270] = "secret source label"
    Image.new("RGB", (40, 20), "blue").save(original, exif=info)
    assets = collect_media_sources([str(original)], tmp_path, "image")
    assets["split"] = "test"
    sparse = prepare_media(assets, _task(), {"max_dimension": 20}, tmp_path / "prepared")
    dense = prepare_media(assets, _task(), {"max_dimension": 40}, tmp_path / "prepared")
    assert sparse["record_id"].tolist() == dense["record_id"].tolist()
    assert sparse.iloc[0]["group_id"] == assets.iloc[0]["group_id"]
    assert sparse.iloc[0]["split"] == "test"
    assert sparse.iloc[0]["start_sec"] == sparse.iloc[0]["end_sec"] == 0.0
    first_frame = sparse.iloc[0]["frames"][0]
    second_frame = dense.iloc[0]["frames"][0]
    assert first_frame["path"] != second_frame["path"]
    assert "present_secret" not in first_frame["path"]
    with Image.open(first_frame["path"]) as frame:
        assert frame.size == (20, 10)
        assert not frame.getexif()
    _image(original, color="green")
    with pytest.raises(ValueError, match="bytes changed"):
        prepare_media(assets, _task(), {}, tmp_path / "prepared")


def test_quality_strict_excludes_iqr_tail_but_keeps_unlabeled_dark_assets(tmp_path):
    for number, side in enumerate([20, 21, 22, 23, 200]):
        _image(tmp_path / f"image{number}.png", color="black", size=(side, side))
    assets = collect_media_sources([str(tmp_path)], tmp_path, "image")
    quality = MediaQualityAgent()
    assert quality.detect_issues(assets)["dark_candidates"] == 5
    assert len(quality.fix(assets, "conservative")) == 5
    strict = quality.fix(assets, "strict")
    assert len(strict) == 4 and strict["width"].max() == 23
    assert strict["label"].isna().all()
    assert len(list(tmp_path.glob("*.png"))) == 5


def _vfr_video(path):
    av = pytest.importorskip("av")
    times_ms = [0, 200, 900, 1600, 1700, 2500]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(times_ms):
            data = np.full((24, 32, 3), (index * 35) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return [value / 1000 for value in times_ms]


def test_video_uses_pts_and_preserves_group_split_and_ids_across_densification(tmp_path):
    video = tmp_path / "video.mkv"
    expected_times = _vfr_video(video)
    assets = collect_media_sources([str(video)], tmp_path, "video")
    assert assets["valid"].all()
    assets["split"] = "test"
    sparse = prepare_media(
        assets,
        _task("video"),
        {
            "clip_duration_sec": 1,
            "frame_interval_sec": 0.5,
        },
        tmp_path / "sparse",
    )
    dense = prepare_media(
        assets,
        _task("video"),
        {
            "clip_duration_sec": 1,
            "frame_interval_sec": 0.1,
        },
        tmp_path / "dense",
    )
    assert len(sparse) == len(dense) == 3
    assert sparse["record_id"].tolist() == dense["record_id"].tolist()
    assert sparse["group_id"].nunique() == 1
    assert sparse["split"].eq("test").all()
    all_times = [
        frame["timestamp_sec"] for row in dense.to_dict(orient="records") for frame in row["frames"]
    ]
    assert all_times == expected_times
    for row in sparse.to_dict(orient="records"):
        assert row["start_sec"] < row["end_sec"]
        assert all(
            row["start_sec"] <= frame["timestamp_sec"] < row["end_sec"] for frame in row["frames"]
        )
    assert sparse.iloc[-1]["end_sec"] >= 2.5


def test_prepare_requires_a_clean_valid_frame(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_text("not a picture")
    rows = collect_media_sources([str(broken)], tmp_path, "image")
    with pytest.raises(ValueError, match="Remove invalid"):
        prepare_media(rows, _task(), {}, tmp_path / "prepared")
    with pytest.raises(ValueError, match="Missing media columns"):
        prepare_media(pd.DataFrame([{"text": "not media"}]), _task(), {}, tmp_path)


def test_reviewed_video_boundaries_resample_only_observed_frames_inside_interval(tmp_path):
    video = tmp_path / "reviewed.mkv"
    _vfr_video(video)
    frames = sample_video_frames(
        video,
        1.5,
        1.8,
        frame_interval_sec=0.01,
        output_dir=tmp_path / "reviewed",
    )
    assert [frame["timestamp_sec"] for frame in frames] == [1.6, 1.7]
    assert all(frame["source_timestamp_sec"] == frame["timestamp_sec"] for frame in frames)
    with pytest.raises(ValueError, match="no decoded frame"):
        sample_video_frames(video, 1.0, 1.1, output_dir=tmp_path / "empty")
    with pytest.raises(ValueError, match="finite"):
        sample_video_frames(video, 2.0, 1.0, output_dir=tmp_path / "empty")


def test_existing_agent_media_adapters_honor_explicit_project_root(tmp_path):
    import yaml

    from agents.data_collection_agent import DataCollectionAgent
    from agents.data_quality_agent import DataQualityAgent

    source = tmp_path / "inputs"
    source.mkdir()
    _image(source / "local.png")
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "images.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"root": "..", "modality": "image", "labels": ["present", "absent"]},
                "collection": {"sources": [{"type": "local", "path": "inputs"}]},
            }
        ),
        encoding="utf-8",
    )
    collection = DataCollectionAgent(config)
    assert collection.root == tmp_path
    rows = collection.collect_media()
    quality = DataQualityAgent(config)
    assert quality.detect_media_issues(rows)["invalid"] == 0
    assert len(quality.fix_media(rows)) == 1
    assert not (configs / "data").exists()


def test_video_pipeline_densifies_resumes_and_trains_on_corrected_intervals(tmp_path):
    import yaml

    from pipeline.media_runner import MediaPipelineRunner

    video = tmp_path / "technical.mkv"
    _vfr_video(video)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"modality": "video", "labels": ["present", "absent"]},
                "task": {"id": "technical-video", "version": "1"},
                "collection": {"sources": [{"path": str(video)}]},
                "sampling": {"clip_duration_sec": 1, "frame_interval_sec": 0.5},
                "training": {"enabled": True, "test_size": 0},
                "features": {"backend": "statistics"},
            }
        )
    )
    first = MediaPipelineRunner(config).run()
    assert first.status == "annotation_required", first.message
    manifest_path = first.artifacts["annotation_requests"]
    old = json.loads(Path(manifest_path).read_text())
    record = old["requests"][0]
    dense = MediaPipelineRunner(config).run(frames=record["record_id"], frame_step=0.1)
    assert dense.status == "annotation_required", dense.message
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["requests"][0]["request_id"] != record["request_id"]
    assert len(manifest["requests"][0]["frames"]) > len(record["frames"])
    answers = [
        {
            "record_id": item["record_id"],
            "request_id": item["request_id"],
            "task_id": "technical-video",
            "task_version": "1",
            "label": ["present", "absent"][index % 2],
            "needs_review": True,
            "reason": "Technical response",
            "confidence": None,
            "score_type": None,
            "viewed_frames": [frame["timestamp_sec"] for frame in item["frames"]],
        }
        for index, item in enumerate(manifest["requests"])
    ]
    responses = Path(manifest["response_path"])
    responses.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
    waiting = MediaPipelineRunner(config).run()
    assert waiting.status == "review_required", waiting.message
    decisions = [
        {
            "record_id": item["record_id"],
            "request_id": item["request_id"],
            "reviewed": True,
            "reviewer": "Test fixture reviewer",
            "review_status": "accepted",
        }
        for item in manifest["requests"]
    ]
    decisions[1].update(
        review_status="edited", human_label="absent", human_start_sec=1.5, human_end_sec=1.8
    )
    Path(waiting.artifacts["review_decisions"]).write_text(
        "".join(json.dumps(row) + "\n" for row in decisions)
    )
    done = MediaPipelineRunner(config).run()
    assert done.status == "completed", done.message
    assert done.metrics["training"]["train_rows"] == 3
    intervals = json.loads(Path(done.artifacts["intervals"]).read_text())
    assert (intervals[1]["start_sec"], intervals[1]["end_sec"]) == (1.5, 1.8)
