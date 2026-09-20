"""A resumable file hand-off to an image-capable agent; no model API is called."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from agents.contracts import TaskSpec


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)
        if jsonl
        else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")  # noqa: TRY004
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path.name}:{number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path.name}:{number}")  # noqa: TRY004
        records.append(value)
    return records


class VisualAnnotationAgent:
    """Prepare viewable requests and validate partial, version-bound JSONL answers.

    ``prepare_requests`` returns ``manifest.json``. Repeating it is safe; only
    unanswered requests may change (for example after denser video sampling).
    ``import_responses`` returns every input row in its original order, including
    pending rows. Importing a partial answer file never marks it complete and
    never represents an agent prediction as human-reviewed.
    """

    def __init__(self, task: TaskSpec) -> None:
        if task.modality not in {"image", "video"}:
            raise ValueError("VisualAnnotationAgent requires an image or video task")
        self.task = task

    def _describe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        required = {"record_id", "content_hash", "modality", "frames"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing annotation input columns: {missing}")
        if df.empty:
            raise ValueError("Cannot prepare an empty annotation batch")
        if df["record_id"].isna().any() or df["record_id"].astype(str).str.strip().eq("").any():
            raise ValueError("record_id must be non-empty")
        if df["record_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate record_id in annotation batch")
        records = []
        checked_sources: dict[str, str] = {}
        for _, row in df.iterrows():
            if row["modality"] != self.task.modality:
                raise ValueError("Input modality does not match task modality")
            start = _number(row.get("start_sec", 0.0), "start_sec")
            end = _number(row.get("end_sec", start), "end_sec")
            if start < 0 or end < start or (self.task.modality == "video" and end <= start):
                raise ValueError("Invalid annotation interval")
            content_hash = str(row["content_hash"])
            media_path = row.get("media_path")
            if isinstance(media_path, (str, Path)) and str(media_path):
                source = Path(media_path).resolve()
                if str(source) not in checked_sources:
                    checked_sources[str(source)] = _file_hash(source)
                if checked_sources[str(source)] != content_hash:
                    raise ValueError(f"Source content changed for record {row['record_id']}")
            frames = row["frames"]
            if isinstance(frames, str):
                frames = json.loads(frames)
            elif hasattr(frames, "tolist"):
                frames = frames.tolist()
            if not isinstance(frames, (list, tuple)) or not frames:
                raise ValueError("Every annotation unit must contain viewable frames")
            described_frames = []
            for frame in frames:
                if not isinstance(frame, Mapping) or "path" not in frame:
                    raise ValueError("Each frame needs path and timestamp_sec")
                timestamp = _number(frame.get("timestamp_sec"), "timestamp_sec")
                if not start <= timestamp <= end:
                    raise ValueError("Frame timestamp lies outside annotation interval")
                path = Path(frame["path"]).expanduser().resolve()
                described_frames.append(
                    {"source_path": str(path), "timestamp_sec": timestamp, "sha256": _file_hash(path)}
                )
            timestamps = [frame["timestamp_sec"] for frame in described_frames]
            if timestamps != sorted(set(timestamps)):
                raise ValueError("Frame timestamps must be unique and chronological")
            identity = {
                "task": self.task.to_dict(),
                "record_id": str(row["record_id"]),
                "content_hash": content_hash,
                "modality": self.task.modality,
                "start_sec": start,
                "end_sec": end,
                "frames": [
                    {"timestamp_sec": frame["timestamp_sec"], "sha256": frame["sha256"]}
                    for frame in described_frames
                ],
            }
            records.append(
                {
                    "record_id": str(row["record_id"]),
                    "request_id": _digest(identity),
                    "modality": self.task.modality,
                    "start_sec": start,
                    "end_sec": end,
                    "frames": described_frames,
                }
            )
        return records

    def prepare_requests(
        self, df: pd.DataFrame, request_dir: Path, *, batch_id: str
    ) -> Path:
        """Export only the task, opaque identifiers and sanitized visual evidence."""
        from PIL import Image, ImageOps

        request_dir = Path(request_dir).expanduser().resolve()
        manifest_path = request_dir / "manifest.json"
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        records = self._describe(df)
        old: dict[str, dict[str, Any]] = {}
        if manifest_path.exists():
            previous = self._load_manifest(request_dir)
            if previous["batch_id"] != batch_id:
                raise ValueError("Existing request directory belongs to another batch")
            old = {row["record_id"]: row for row in previous["requests"]}
            if set(old) != {row["record_id"] for row in records}:
                raise ValueError("Batch record IDs changed; create a new request directory")
            accepted = {row["record_id"] for row in _read_jsonl(request_dir / "accepted_responses.jsonl")}
            for row in records:
                if row["record_id"] in accepted and old[row["record_id"]]["request_id"] != row["request_id"]:
                    raise ValueError("Cannot change an answered request; create a new batch")

        requests = []
        for row in records:
            previous_row = old.get(row["record_id"])
            if previous_row and previous_row["request_id"] == row["request_id"]:
                for frame in previous_row["frames"]:
                    path = Path(frame["path"]).resolve()
                    if not path.is_relative_to(request_dir / "frames"):
                        raise ValueError("Manifest frame path is outside request directory")
                    if not path.is_file() or _file_hash(path) != frame["sha256"]:
                        raise ValueError(f"Packet frame content changed for {row['record_id']}")
                requests.append(previous_row)
                continue
            packet_frames = []
            for index, frame in enumerate(row["frames"]):
                # Re-encode pixels under opaque names to omit source names and EXIF labels.
                target = request_dir / "frames" / row["request_id"] / f"{index:04d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(frame["source_path"]) as source:
                    pixels = ImageOps.exif_transpose(source).convert("RGB")
                    pixels.info.clear()
                    pixels.save(target, format="PNG")
                packet_frames.append(
                    {
                        "path": str(target),
                        "timestamp_sec": frame["timestamp_sec"],
                        "sha256": _file_hash(target),
                    }
                )
            requests.append({**row, "frames": packet_frames})
        manifest = {
            "schema_version": 1,
            "batch_id": batch_id,
            "task": self.task.to_dict(),
            "requests": requests,
            "response_path": str(request_dir / "responses.jsonl"),
        }
        manifest["manifest_id"] = _digest(manifest)
        _atomic_json(manifest_path, manifest)
        return manifest_path

    def _load_manifest(self, request_dir: Path) -> dict[str, Any]:
        manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
        if manifest.get("manifest_id") != _digest(payload):
            raise ValueError("Manifest content changed; rebuild requests from the pipeline")
        if manifest.get("schema_version") != 1 or manifest.get("task") != self.task.to_dict():
            raise ValueError("Stale manifest: task or schema version changed")
        requests = manifest.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("Manifest must contain requests")
        record_ids = [row["record_id"] for row in requests]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Duplicate record_id in manifest")
        return manifest

    def _validate_answer(self, raw: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        required = {
            "record_id", "request_id", "task_id", "task_version", "label", "needs_review",
            "reason", "confidence", "score_type", "viewed_frames",
        }
        allowed = required | {"start_sec", "end_sec"}
        if required - raw.keys() or raw.keys() - allowed:
            raise ValueError(
                f"Invalid response fields: missing={sorted(required - raw.keys())}, "
                f"unknown={sorted(raw.keys() - allowed)}"
            )
        if (
            raw["request_id"] != request["request_id"]
            or raw["task_id"] != self.task.task_id
            or str(raw["task_version"]) != str(self.task.version)
        ):
            raise ValueError(f"Stale response for record {request['record_id']}")
        label = raw["label"]
        if label is not None and (not isinstance(label, str) or label not in self.task.labels):
            raise ValueError(f"Unsupported response label: {label!r}")
        if not isinstance(raw["needs_review"], bool):
            raise ValueError("needs_review must be a JSON boolean")  # noqa: TRY004
        if label is None and not raw["needs_review"]:
            raise ValueError("Abstention requires needs_review=true")
        if not isinstance(raw["reason"], str) or not raw["reason"].strip():
            raise ValueError("A non-empty reason is required")
        confidence = raw["confidence"]
        if confidence is not None:
            confidence = _number(confidence, "confidence")
            if not 0 <= confidence <= 1 or raw["score_type"] != "self_reported":
                raise ValueError("confidence must be in [0, 1] with score_type=self_reported")
        elif raw["score_type"] not in (None, "self_reported"):
            raise ValueError("score_type must be null or self_reported")
        viewed = raw["viewed_frames"]
        if not isinstance(viewed, list):
            raise ValueError("viewed_frames must be a list of timestamps")  # noqa: TRY004
        viewed = [_number(value, "viewed_frames") for value in viewed]
        available = {frame["timestamp_sec"] for frame in request["frames"]}
        if len(set(viewed)) != len(viewed) or not set(viewed) <= available:
            raise ValueError("viewed_frames contains duplicate or unavailable timestamps")
        if label is not None and not viewed:
            raise ValueError("A non-null label requires viewed visual evidence")
        start = _number(raw.get("start_sec", request["start_sec"]), "start_sec")
        end = _number(raw.get("end_sec", request["end_sec"]), "end_sec")
        if not request["start_sec"] <= start <= end <= request["end_sec"]:
            raise ValueError("Response interval lies outside the requested clip")
        if request["modality"] == "video" and start == end:
            raise ValueError("Video response interval must have positive duration")
        return {
            **raw,
            "task_version": str(self.task.version),
            "confidence": confidence,
            "viewed_frames": sorted(viewed),
            "start_sec": start,
            "end_sec": end,
        }

    def import_responses(
        self,
        df: pd.DataFrame,
        request_dir: Path,
        response_path: Path | None = None,
    ) -> pd.DataFrame:
        """Validate the whole input before committing; merge safe partial answers by ID."""
        request_dir = Path(request_dir).expanduser().resolve()
        manifest = self._load_manifest(request_dir)
        requests = {row["record_id"]: row for row in manifest["requests"]}
        described = {row["record_id"]: row for row in self._describe(df)}
        if requests.keys() != described.keys():
            raise ValueError("Input record IDs do not match annotation manifest")
        for record_id, request in requests.items():
            if described[record_id]["request_id"] != request["request_id"]:
                raise ValueError(f"Stale manifest: media or sampling changed for {record_id}")
            for field in ("modality", "start_sec", "end_sec"):
                if described[record_id][field] != request[field]:
                    raise ValueError(f"Manifest {field} changed for {record_id}")
            if [frame["timestamp_sec"] for frame in request["frames"]] != [
                frame["timestamp_sec"] for frame in described[record_id]["frames"]
            ]:
                raise ValueError(f"Manifest frame timestamps changed for {record_id}")
            for frame in request["frames"]:
                path = Path(frame["path"]).resolve()
                if not path.is_relative_to(request_dir / "frames"):
                    raise ValueError("Manifest frame path is outside request directory")
                if _file_hash(path) != frame["sha256"]:
                    raise ValueError(f"Packet frame content changed for {record_id}")

        accepted_path = request_dir / "accepted_responses.jsonl"
        incoming = Path(response_path) if response_path is not None else request_dir / "responses.jsonl"
        if response_path is not None and not incoming.is_file():
            raise FileNotFoundError(incoming)
        accepted: dict[str, dict[str, Any]] = {}
        paths = [accepted_path] if incoming.resolve() == accepted_path else [accepted_path, incoming]
        for path in paths:
            seen = set()
            for raw in _read_jsonl(path):
                record_id = raw.get("record_id")
                if not isinstance(record_id, str) or record_id not in requests:
                    raise ValueError(f"Response contains unknown record_id: {record_id!r}")
                if record_id in seen:
                    raise ValueError(f"Duplicate response record_id: {record_id}")
                seen.add(record_id)
                answer = self._validate_answer(raw, requests[record_id])
                if record_id in accepted and accepted[record_id] != answer:
                    raise ValueError(f"Conflicting response for {record_id}; preserve accepted answer")
                accepted[record_id] = answer

        result = df.copy(deep=True)
        defaults = {
            "annotation_status": "pending", "auto_label": None, "confidence": None,
            "score_type": None, "annotation_backend": "agent_skill", "annotation_reason": "",
            "annotation_needs_review": True, "viewed_frames": None, "request_id": None,
            "auto_start_sec": None, "auto_end_sec": None, "human_label": None,
            "final_label": None, "reviewed": False, "review_status": "pending",
            "needs_review": True,
        }
        for column, default in defaults.items():
            if column not in result:
                result[column] = pd.Series([default] * len(result), index=result.index, dtype=object)
            else:
                # JSON/Parquet readers infer all-null columns as numeric. These
                # fields later hold classes, IDs, lists or nullable decisions.
                result[column] = result[column].astype(object)
                if default is not None:
                    result[column] = result[column].where(result[column].notna(), default)
        for index, row in result.iterrows():
            record_id = str(row["record_id"])
            result.at[index, "request_id"] = requests[record_id]["request_id"]
            if record_id not in accepted:
                continue
            answer = accepted[record_id]
            result.at[index, "annotation_status"] = (
                "annotated" if answer["label"] is not None else "abstained"
            )
            for column, field in (
                ("auto_label", "label"), ("confidence", "confidence"),
                ("score_type", "score_type"), ("annotation_reason", "reason"),
                ("annotation_needs_review", "needs_review"), ("viewed_frames", "viewed_frames"),
                ("auto_start_sec", "start_sec"), ("auto_end_sec", "end_sec"),
            ):
                result.at[index, column] = answer[field]
        _atomic_json(accepted_path, [accepted[key] for key in sorted(accepted)], jsonl=True)
        return result
