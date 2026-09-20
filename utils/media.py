"""Local media inspection and bounded frame extraction using real video PTS.

Pillow and PyAV are imported when their respective modalities are used. There is
no external model call, download, source-file mutation, or shell invocation here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.contracts import TaskSpec

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:  # pragma: no cover - dependency guard
        raise ImportError("Install the project's media extra for image support (Pillow)") from error
    return Image, ImageOps


def _pyav() -> Any:
    try:
        import av
    except ImportError as error:  # pragma: no cover - dependency guard
        raise ImportError(
            "Install the project's media extra for video support (av/PyAV)"
        ) from error
    return av


def _image_statistics(image: Any) -> dict[str, Any]:
    image = image.convert("RGB")
    gray = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float64)
    # These are diagnostics, not automatic reasons to reject dark/blurred scenes.
    gradient = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))
    bits = np.asarray(image.convert("L").resize((9, 8)))
    differences = bits[:, 1:] > bits[:, :-1]
    dhash = sum(int(bit) << index for index, bit in enumerate(differences.flat))
    return {
        "width": int(image.width),
        "height": int(image.height),
        "brightness": float(gray.mean()),
        "edge_variance": gradient,
        "perceptual_hash": f"{dhash:016x}",
    }


def inspect_media(path: str | Path, modality: str) -> dict[str, Any]:
    """Decode enough to inspect a local asset; decoding failures remain auditable."""
    path = Path(path)
    if modality == "image":
        Image, ImageOps = _pillow()
        with Image.open(path) as image:
            image.seek(0)
            image.load()
            stats = _image_statistics(ImageOps.exif_transpose(image))
            return {
                **stats,
                "duration_sec": None,
                "frame_count": int(getattr(image, "n_frames", 1)),
            }
    if modality != "video":
        raise ValueError("modality must be image or video")
    av = _pyav()
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("File contains no video stream")
        stream = container.streams.video[0]
        iterator = container.decode(stream)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Video contains no decodable frames")
        if first.pts is None or first.time_base is None:
            raise ValueError("Video frame has no presentation timestamp")
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        if duration is None or duration <= 0:
            # Unknown-length containers require a scan; keep only one frame at a time.
            last = first
            for frame in iterator:
                last = frame
            duration = _frame_time(last) - _frame_time(first) + _frame_duration(last, stream)
        stats = _image_statistics(first.to_image())
        return {
            **stats,
            "duration_sec": duration,
            "frame_count": int(stream.frames or 0),
            "source_start_sec": _frame_time(first),
        }


def _frame_time(frame: Any) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("Missing frame PTS; cannot safely fabricate timestamps from frame indices")
    timestamp = float(frame.pts * frame.time_base)
    if not math.isfinite(timestamp):
        raise ValueError("Non-finite video timestamp")
    return timestamp


def _frame_duration(frame: Any, stream: Any) -> float:
    duration = getattr(frame, "duration", None)
    if duration is not None and duration > 0 and frame.time_base is not None:
        return float(duration * frame.time_base)
    # Only used to bound the final frame's visible interval, never to invent its PTS.
    if stream.average_rate is not None and stream.average_rate > 0:
        return 1.0 / float(stream.average_rate)
    return 0.001


def _sampling_settings(sampling: Mapping[str, Any]) -> dict[str, Any]:
    settings = {
        "clip_duration_sec": float(sampling.get("clip_duration_sec", 5.0)),
        "frame_interval_sec": float(
            sampling.get("frame_interval_sec", sampling.get("sample_interval_sec", 1.0))
        ),
        "max_frames_per_clip": int(sampling.get("max_frames_per_clip", 8)),
        "max_dimension": int(sampling.get("max_dimension", 1024)),
    }
    if any(not math.isfinite(float(value)) or value <= 0 for value in settings.values()):
        raise ValueError("Media sampling values must be positive finite numbers")
    if settings["max_frames_per_clip"] > 256:
        raise ValueError(
            "max_frames_per_clip must be at most 256; shorten clips for denser sampling"
        )
    return settings


def _save_image(image: Any, path: Path, max_dimension: int) -> dict[str, Any]:
    Image, ImageOps = _pillow()
    sanitized = ImageOps.exif_transpose(image).convert("RGB")
    sanitized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    # Re-encoding strips source metadata, filenames, and orientation ambiguity.
    sanitized.info.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    sanitized.save(temporary, format="JPEG", quality=92)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "width": sanitized.width,
        "height": sanitized.height,
    }


def prepare_media(
    assets: pd.DataFrame,
    task: TaskSpec,
    sampling: Mapping[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Create image/clip units after the caller assigns asset/group splits.

    Video is decoded as a stream. Saved frames are bounded by the configured
    per-clip cap, contain actual PTS, and retain their order. No audio analysis is
    implied. Rows with ``valid=False`` must be removed explicitly by quality.
    """
    settings = _sampling_settings(sampling)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    target = Path(output_dir).expanduser().resolve() / f"sampling-{fingerprint[:16]}"
    prepared: list[dict[str, Any]] = []
    required = {"record_id", "asset_id", "media_path", "content_hash", "group_id", "modality"}
    if not required.issubset(assets.columns):
        raise ValueError(f"Missing media columns: {sorted(required - set(assets.columns))}")
    for row in assets.to_dict(orient="records"):
        if row.get("valid") is False:
            raise ValueError(f"Remove invalid asset before preparation: {row['record_id']}")
        if row["modality"] != task.modality:
            raise ValueError("Asset modality does not match task")
        path = Path(str(row["media_path"]))
        if file_sha256(path) != row["content_hash"]:
            raise ValueError(f"Media bytes changed after collection: {path}")
        # Annotation-unit identity survives denser/resized frame requests. The
        # request fingerprint, rather than record_id, binds sampled evidence.
        identity = str(row["asset_id"])
        if task.modality == "video":
            identity += f":clip_duration_sec={settings['clip_duration_sec']}"
        unit_prefix = hashlib.sha256(identity.encode()).hexdigest()[:32]
        if task.modality == "image":
            Image, _ = _pillow()
            unit_id = f"unit-{unit_prefix}"
            with Image.open(path) as image:
                frame_info = _save_image(
                    image, target / unit_id / "frame-000000.jpg", settings["max_dimension"]
                )
            prepared.append(
                {
                    **row,
                    "record_id": unit_id,
                    "task_id": task.task_id,
                    "task_version": task.version,
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "duration_sec": None,
                    "frames": [{**frame_info, "timestamp_sec": 0.0}],
                    "sampling": settings.copy(),
                }
            )
        else:
            prepared.extend(_prepare_video(path, row, task, settings, target, unit_prefix))
        if file_sha256(path) != row["content_hash"]:
            raise ValueError(f"Media bytes changed during preparation: {path}")
    return pd.DataFrame(prepared)


def _prepare_video(
    path: Path,
    row: dict[str, Any],
    task: TaskSpec,
    settings: dict[str, Any],
    target: Path,
    unit_prefix: str,
) -> list[dict[str, Any]]:
    av = _pyav()
    clip_length = settings["clip_duration_sec"]
    cap = settings["max_frames_per_clip"]
    interval = max(settings["frame_interval_sec"], clip_length / cap)
    clips: dict[int, list[dict[str, Any]]] = {}
    source_start: float | None = None
    previous = -math.inf
    final_end = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            raw_time = _frame_time(frame)
            if source_start is None:
                source_start = raw_time
            timestamp = raw_time - source_start
            if timestamp + 1e-9 < previous:
                raise ValueError("Non-monotonic presentation timestamps cannot be sampled safely")
            previous = timestamp
            final_end = timestamp + _frame_duration(frame, stream)
            clip_index = math.floor(timestamp / clip_length)
            sampled = clips.setdefault(clip_index, [])
            start = clip_index * clip_length
            threshold = (
                start
                + (math.floor((sampled[-1]["timestamp_sec"] - start) / interval) + 1) * interval
                if sampled
                else start
            )
            if len(sampled) >= cap or timestamp + 1e-9 < threshold:
                continue
            unit_id = f"unit-{unit_prefix}-{clip_index:06d}"
            frame_info = _save_image(
                frame.to_image(),
                target / unit_id / f"frame-{len(sampled):06d}.jpg",
                settings["max_dimension"],
            )
            sampled.append(
                {**frame_info, "timestamp_sec": timestamp, "source_timestamp_sec": raw_time}
            )
    if source_start is None:
        raise ValueError("Video contains no decodable frames")
    result = []
    for index, frames in sorted(clips.items()):
        start = index * clip_length
        end = min(start + clip_length, final_end)
        result.append(
            {
                **row,
                "record_id": f"unit-{unit_prefix}-{index:06d}",
                "task_id": task.task_id,
                "task_version": task.version,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "asset_duration_sec": final_end,
                "source_start_sec": source_start,
                "frames": frames,
                "sampling": settings.copy(),
            }
        )
    return result


def sample_video_frames(
    media_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    frame_interval_sec: float = 1.0,
    max_frames: int = 8,
    max_dimension: int = 1024,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Sample a reviewed interval without inventing boundary frames.

    Times are relative to the first decoded PTS, as in ``prepare_media``.
    Decode is sequential, output is bounded, and only frames actually observed
    inside ``[start_sec, end_sec)`` are returned.
    """
    start, end = float(start_sec), float(end_sec)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Video interval must satisfy finite 0 <= start_sec < end_sec")
    settings = _sampling_settings(
        {
            "clip_duration_sec": end - start,
            "frame_interval_sec": frame_interval_sec,
            "max_frames_per_clip": max_frames,
            "max_dimension": max_dimension,
        }
    )
    path = Path(media_path).expanduser().resolve()
    source_hash = file_sha256(path)
    identity = json.dumps(
        {"source_hash": source_hash, "start_sec": start, "end_sec": end, "sampling": settings},
        sort_keys=True,
    )
    directory = (
        Path(output_dir).expanduser().resolve() / hashlib.sha256(identity.encode()).hexdigest()[:32]
    )
    interval = max(settings["frame_interval_sec"], (end - start) / settings["max_frames_per_clip"])
    next_requested = start
    sampled: list[dict[str, Any]] = []
    source_start = None
    previous = -math.inf
    av = _pyav()
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        for frame in container.decode(container.streams.video[0]):
            source_time = _frame_time(frame)
            if source_start is None:
                source_start = source_time
            timestamp = source_time - source_start
            if timestamp + 1e-9 < previous:
                raise ValueError("Non-monotonic presentation timestamps")
            previous = timestamp
            if timestamp >= end or len(sampled) >= settings["max_frames_per_clip"]:
                break
            if timestamp < start or timestamp + 1e-9 < next_requested:
                continue
            frame_info = _save_image(
                frame.to_image(),
                directory / f"frame-{len(sampled):06d}.jpg",
                settings["max_dimension"],
            )
            sampled.append(
                {**frame_info, "timestamp_sec": timestamp, "source_timestamp_sec": source_time}
            )
            next_requested = start + (math.floor((timestamp - start) / interval) + 1) * interval
    if file_sha256(path) != source_hash:
        raise ValueError("Media bytes changed during sampling")
    if not sampled:
        raise ValueError("Reviewed interval contains no decoded frame; adjust boundaries or media")
    return sampled


__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "file_sha256",
    "inspect_media",
    "prepare_media",
    "sample_video_frames",
]
