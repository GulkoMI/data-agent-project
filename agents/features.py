"""Local visual features shared by active learning and final model training.

The statistics backend is a small, dependency-light baseline, not a semantic
action recognizer. Optional pretrained encoders run locally after their weights
have been downloaded. No annotation service or model API is used here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

FEATURE_VERSION = 1


def _frames(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("Visual features require a non-empty frames list")
    return value


def feature_matrix(frame: pd.DataFrame, column: str = "features") -> np.ndarray:
    """Convert stored vectors to one finite, consistently shaped feature matrix."""
    if column not in frame or frame.empty:
        raise ValueError(f"A non-empty {column!r} feature column is required")
    try:
        matrix = np.stack([np.asarray(value, dtype=np.float32) for value in frame[column]])
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature vectors must all have the same numeric shape") from exc
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Feature vectors must be finite, non-empty one-dimensional arrays")
    return matrix


def assert_disjoint_groups(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check both lineage and content, not just the derived sample identifiers."""
    for frame in (train, test):
        if (
            "group_id" not in frame
            or frame["group_id"].isna().any()
            or frame["group_id"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Media evaluation requires a non-empty group_id on every row")
    for column in ("record_id", "group_id", "asset_id", "content_hash"):
        if column in train and column in test:
            overlap = set(train[column].dropna().astype(str)) & set(
                test[column].dropna().astype(str)
            )
            if overlap:
                raise ValueError(f"Train/test {column} overlap: {sorted(overlap)[:5]}")


class VisualFeatureEncoder:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.backend = self.config.get("backend", "statistics")
        if self.backend not in {"statistics", "transformers_image", "videomae"}:
            raise ValueError("features.backend must be statistics, transformers_image, or videomae")
        self.model = None
        self.processor = None
        self.device = None
        self.cache = Path(self.config["cache_dir"]) if self.config.get("cache_dir") else None
        if self.backend != "statistics":
            model = self.config.get("model")
            if not model:
                raise ValueError("Pretrained visual features require features.model")
            if not Path(model).is_dir() and not self.config.get("revision"):
                raise ValueError("Pin features.revision for a remote pretrained model")

    def _identity(self) -> dict[str, Any]:
        identity = {key: value for key, value in self.config.items() if key != "cache_dir"}
        identity.update(backend=self.backend, implementation_version=FEATURE_VERSION)
        model = Path(self.config.get("model", "__no_model__"))
        if self.backend != "statistics" and model.is_dir():
            digests = {}
            for path in sorted(model.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                    with path.open("rb") as handle:
                        digests[str(path.relative_to(model))] = hashlib.file_digest(
                            handle, "sha256"
                        ).hexdigest()
            identity["local_model_files"] = digests
        return identity

    @staticmethod
    def _statistics(images: list[Image.Image]) -> np.ndarray:
        # Four temporal bins retain coarse order; adjacent differences capture
        # changes in appearance. These are explicitly low-level features.
        grids = np.stack(
            [
                np.asarray(image.resize((4, 4)), dtype=np.float32).reshape(-1) / 255.0
                for image in images
            ]
        )
        bins = [part.mean(axis=0) if len(part) else grids[-1] for part in np.array_split(grids, 4)]
        delta = np.abs(np.diff(grids, axis=0)).mean(axis=0) if len(grids) > 1 else grids[0] * 0
        return np.concatenate([*bins, grids.mean(axis=0), grids.std(axis=0), delta])

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModel,
                VideoMAEImageProcessor,
                VideoMAEModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the annotation extra for pretrained visual features"
            ) from exc
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("features.device must be auto, cpu, mps, or cuda")
        self.device = device
        kwargs = {"local_files_only": bool(self.config.get("local_files_only", False))}
        if self.config.get("revision"):
            kwargs["revision"] = self.config["revision"]
        name = self.config["model"]
        if self.backend == "videomae":
            self.processor = VideoMAEImageProcessor.from_pretrained(name, **kwargs)
            self.model = VideoMAEModel.from_pretrained(name, **kwargs)
        else:
            self.processor = AutoImageProcessor.from_pretrained(name, use_fast=False, **kwargs)
            self.model = AutoModel.from_pretrained(name, **kwargs)
        self.model = self.model.to(device).eval()

    def _pretrained(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load_model()
        if self.backend == "videomae":
            count = int(self.model.config.num_frames)
            positions = np.linspace(0, len(images) - 1, count).round().astype(int)
            images = [images[position] for position in positions]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1).float().cpu().numpy()
        if self.backend == "videomae":
            return embeddings[0]
        # An image model encodes frame appearance; pooling is not a video model.
        return np.concatenate([embeddings.mean(axis=0), embeddings.std(axis=0)])

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        identity = self._identity()
        vectors = []
        keys = []
        for row in dataframe.to_dict("records"):
            frames = _frames(row.get("frames"))
            if row.get("modality") == "video":
                start, end = row.get("human_start_sec"), row.get("human_end_sec")
                if (
                    start is not None
                    and end is not None
                    and not pd.isna(start)
                    and not pd.isna(end)
                ):
                    frames = [frame for frame in frames if start <= frame["timestamp_sec"] < end]
                    if not frames:
                        raise ValueError(
                            "No frames inside the human-reviewed interval; resample before training"
                        )
            evidence = []
            for item in frames:
                path = Path(item["path"])
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                evidence.append({"sha256": digest, "timestamp_sec": item.get("timestamp_sec")})
            payload = {
                "encoder": identity,
                "frames": evidence,
                "content_hash": row.get("content_hash"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
            }
            key = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest()
            path = self.cache / f"{key}.npy" if self.cache else None
            if path and path.exists():
                vector = np.load(path, allow_pickle=False)
            else:
                images = []
                for item in frames:
                    with Image.open(item["path"]) as source:
                        images.append(source.convert("RGB"))
                vector = (
                    self._statistics(images)
                    if self.backend == "statistics"
                    else self._pretrained(images)
                )
                vector = np.asarray(vector, dtype=np.float32)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError("Encoder returned invalid visual features")
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.save(handle, vector, allow_pickle=False)
                    temporary.replace(path)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid cached feature vector: {path}")
            vectors.append(vector.tolist())
            keys.append(key)
        result = dataframe.copy()
        result["features"] = vectors
        result["feature_key"] = keys
        return result


def extract_features(dataframe: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    return VisualFeatureEncoder(config).transform(dataframe)
