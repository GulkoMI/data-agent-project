"""Read local visual assets while keeping source labels separate from annotation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from agents.common import utc_now_iso
from utils.media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, file_sha256, inspect_media

ASSET_COLUMNS = [
    "asset_id",
    "record_id",
    "group_id",
    "original_group_id",
    "media_path",
    "content_hash",
    "modality",
    "source",
    "source_id",
    "collected_at",
    "label",
    "source_label",
    "valid",
    "error",
    "width",
    "height",
    "duration_sec",
    "frame_count",
    "brightness",
    "edge_variance",
    "perceptual_hash",
    "near_duplicate_candidate",
]


def _present(value: Any) -> bool:
    return (
        value is not None
        and not (isinstance(value, float) and pd.isna(value))
        and str(value).strip() != ""
    )


def _local_path(value: Any, root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Local media requires a nonempty path")
    if "://" in str(value):
        raise ValueError("Media collection accepts local files only; remote URLs are unsupported")
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict(orient="records")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Manifest line {line_number} must be an object")
                rows.append(row)
        return rows
    raise ValueError("A media manifest must be a local CSV or JSONL file")


def _entries(spec: Mapping[str, Any], root: Path, modality: str) -> list[dict[str, Any]]:
    source_type = str(spec.get("type", "local")).lower()
    if source_type not in {"local", "file", "directory", "glob", "manifest", "local_media"}:
        raise ValueError(f"Unsupported media source type: {source_type}; use local or manifest")
    setting = spec.get("manifest", spec.get("paths", spec.get("path", spec.get("local_path"))))
    if source_type == "manifest" or "manifest" in spec:
        manifest = _local_path(setting, root).resolve()
        records = _read_manifest(manifest)
        return [
            {
                **entry,
                "media_path": str(
                    _local_path(
                        entry.get("media_path", entry.get("path")), manifest.parent
                    ).resolve()
                ),
            }
            for entry in records
        ]
    settings = list(setting) if isinstance(setting, (list, tuple)) else [setting]
    paths: set[Path] = set()
    allowed = IMAGE_EXTENSIONS if modality == "image" else VIDEO_EXTENSIONS
    for value in settings:
        path = _local_path(value, root)
        if path.is_dir():
            pattern = "**/*" if spec.get("recursive", True) else "*"
            paths.update(
                item.resolve()
                for item in path.glob(pattern)
                if item.is_file() and item.suffix.lower() in allowed
            )
        elif any(character in str(path) for character in "*?["):
            paths.update(
                Path(item).resolve()
                for item in glob(str(path), recursive=True)
                if Path(item).is_file() and Path(item).suffix.lower() in allowed
            )
        else:
            # Explicit nonexistent/corrupt files produce invalid asset rows for audit.
            paths.add(path.resolve())
    return [{"media_path": str(path), "group_id": spec.get("group_id")} for path in sorted(paths)]


def _group_assets(frame: pd.DataFrame) -> pd.DataFrame:
    """Union explicit session groups and byte-identical assets before splitting."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for row in frame.to_dict(orient="records"):
        asset = f"asset:{row['asset_id']}"
        find(asset)
        if _present(row.get("original_group_id")):
            union(asset, f"group:{row['original_group_id']}")
    names: dict[str, set[str]] = {}
    assets: dict[str, set[str]] = {}
    for row in frame.to_dict(orient="records"):
        key = find(f"asset:{row['asset_id']}")
        assets.setdefault(key, set()).add(row["asset_id"])
        if _present(row.get("original_group_id")):
            names.setdefault(key, set()).add(str(row["original_group_id"]))
    groups: dict[str, str] = {}
    for key, members in assets.items():
        explicit = sorted(names.get(key, set()))
        if len(explicit) == 1:
            groups[key] = explicit[0]
        elif explicit:
            encoded = json.dumps(explicit, ensure_ascii=False, separators=(",", ":"))
            groups[key] = "group-" + hashlib.sha256(encoded.encode()).hexdigest()[:24]
        else:
            groups[key] = min(members)
    frame["group_id"] = [groups[find(f"asset:{asset}")] for asset in frame["asset_id"]]
    return frame


def collect_media_sources(
    sources: Sequence[Mapping[str, Any] | str],
    root: Path,
    modality: str,
) -> pd.DataFrame:
    """Collect local files/directories/globs or manifests into asset-level rows.

    Manifest paths resolve relative to the manifest. Optional ``group_id`` joins
    camera/session-related files. Optional ``source_label`` (or legacy ``label``)
    is reference metadata only: the new ``label`` field stays null until review.
    A missing dependency raises; a corrupt source becomes ``valid=False``.
    """
    if modality not in {"image", "video"}:
        raise ValueError("modality must be image or video")
    root = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        spec = {"path": source} if isinstance(source, str) else dict(source)
        name = str(spec.get("id", f"local_{index + 1}"))
        for entry in _entries(spec, root, modality):
            path = Path(entry["media_path"])
            source_id = str(entry.get("source_id") or path)
            content_hash = None
            metadata: dict[str, Any] = {}
            error = None
            try:
                content_hash = file_sha256(path)
                metadata = inspect_media(path, modality)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001 - retain per-file decode failures for quality.
                error = f"{type(exc).__name__}: {exc}"
            asset_hash = content_hash or hashlib.sha256(str(path).encode()).hexdigest()
            asset_id = f"asset-{asset_hash[:32]}"
            record_hash = hashlib.sha256(f"{name}\0{source_id}\0{asset_hash}".encode()).hexdigest()
            source_label = entry.get("source_label", entry.get("label"))
            if not _present(source_label):
                source_label = None
            rows.append(
                {
                    "asset_id": asset_id,
                    "record_id": f"asset-row-{record_hash[:32]}",
                    "group_id": None,
                    "original_group_id": entry.get("group_id", spec.get("group_id")),
                    "media_path": str(path.resolve()),
                    "content_hash": content_hash,
                    "modality": modality,
                    "source": name,
                    "source_id": source_id,
                    "collected_at": utc_now_iso(),
                    "label": None,
                    "source_label": source_label,
                    "valid": error is None,
                    "error": error,
                    **metadata,
                }
            )
    frame = pd.DataFrame(rows, columns=ASSET_COLUMNS + ["source_start_sec"])
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)
    frame = _group_assets(frame)
    hashes = frame["perceptual_hash"]
    # This is only a candidate flag. Black/flat scenes may share a dHash and
    # should not be silently deleted or treated as confirmed duplicate footage.
    frame["near_duplicate_candidate"] = hashes.notna() & hashes.duplicated(keep=False)
    return frame


__all__ = ["ASSET_COLUMNS", "collect_media_sources"]
