"""The existing pipeline's resumable image/video branch.

The CLI exchanges files with a vision-capable agent. Human decisions are a
separate boundary; restarting the process never invents annotations or review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.al_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.common import read_yaml, utc_now_iso
from agents.contracts import TaskSpec
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.features import extract_features
from agents.media_review import apply_media_review, build_media_review_queue
from agents.train_agent import TrainAgent
from pipeline.runner import PipelineResult
from utils.media import file_sha256, prepare_media, sample_video_frames


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_json(
        temporary,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
        double_precision=15,
    )
    temporary.replace(path)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_json(
        path,
        orient="records",
        lines=True,
        convert_dates=False,
        precise_float=True,
        dtype={
            "record_id": str,
            "asset_id": str,
            "group_id": str,
        },
    )


@contextmanager
def _run_lock(directory: Path):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another process is updating this media run; retry after it finishes"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _slug(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError(
            "task.id and run-id must use letters, digits, dots, underscores or hyphens"
        )
    return value


def merge_video_intervals(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Merge adjacent verified clip labels, retaining contributing record IDs."""
    intervals: list[dict[str, Any]] = []
    if dataframe.empty or "final_label" not in dataframe:
        return intervals
    for _, row in dataframe.sort_values(["asset_id", "start_sec"]).iterrows():
        if row["modality"] != "video" or pd.isna(row["final_label"]):
            continue
        start = row.get("human_start_sec")
        end = row.get("human_end_sec")
        start = float(row["start_sec"] if pd.isna(start) else start)
        end = float(row["end_sec"] if pd.isna(end) else end)
        previous = intervals[-1] if intervals else None
        if (
            previous
            and previous["asset_id"] == row["asset_id"]
            and previous["label"] == row["final_label"]
            and abs(previous["end_sec"] - start) < 1e-6
        ):
            previous["end_sec"] = end
            previous["record_ids"].append(row["record_id"])
        else:
            intervals.append(
                {
                    "asset_id": row["asset_id"],
                    "media_path": row["media_path"],
                    "start_sec": start,
                    "end_sec": end,
                    "label": row["final_label"],
                    "record_ids": [row["record_id"]],
                    "boundary_precision": "sampled_clip_bounds_or_human_adjusted",
                }
            )
    return intervals


class MediaPipelineRunner:
    def __init__(self, config: str | Path, *, run_id: str = "default") -> None:
        self.config_path = Path(config).expanduser().resolve()
        self.config = read_yaml(self.config_path)
        self.task = TaskSpec.from_config(self.config)
        project_root = self.config.get("project", {}).get("root", ".")
        self.root = (self.config_path.parent / project_root).resolve()
        self.run_id = _slug(run_id)
        self.run_dir = self.root / "data/runs" / _slug(self.task.task_id) / self.run_id
        self.state_path = self.run_dir / "pipeline_state.json"
        self.units_path = self.run_dir / "units.jsonl"
        self.assets_path = self.run_dir / "assets.jsonl"
        self.clean_assets_path = self.run_dir / "clean_assets.jsonl"
        self.seed = int(self.config.get("project", {}).get("random_seed", 42))
        self.al_config = self.config.get("active_learning", {})
        self.training_config = self.config.get("training", {})
        self.al_enabled = bool(self.al_config.get("enabled", False))
        self.config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str, ensure_ascii=False).encode()
        ).hexdigest()
        self.state: dict[str, Any] = {}
        self.annotator = AnnotationAgent.for_task(self.task)

    def status(self) -> PipelineResult:
        if not self.state_path.exists():
            return PipelineResult(
                status="not_started",
                message="Run this configuration to prepare annotation requests.",
            )
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        return self._result(state)

    @staticmethod
    def _result(state: dict[str, Any]) -> PipelineResult:
        return PipelineResult(
            status=state["status"],
            completed_stages=state.get("completed_stages", []),
            artifacts=state.get("artifacts", {}),
            metrics=state.get("metrics", {}),
            message=state.get("message", ""),
        )

    def _save(self, status: str, message: str) -> PipelineResult:
        self.state.update(status=status, message=message, updated_at=utc_now_iso())
        self.state["artifacts"]["state"] = str(self.state_path)
        _json(self.state_path, self.state)
        return self._result(self.state)

    def _stage(self, name: str) -> None:
        if name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(name)

    def _split(self, assets: pd.DataFrame) -> pd.DataFrame:
        assets = assets.copy()
        assets["split"] = "train"
        fraction = float(self.training_config.get("test_size", 0.0))
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("training.test_size must be in [0, 1)")
        if fraction:
            groups = sorted(assets["group_id"].unique())
            if len(groups) < 2:
                raise ValueError(
                    "A held-out split needs at least two source groups; set training.test_size: 0 for annotation/fit-only"
                )
            generator = np.random.default_rng(self.seed)
            count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
            test = set(generator.choice(groups, size=count, replace=False))
            assets.loc[assets["group_id"].isin(test), "split"] = "test"
        return assets

    def _initialize(self) -> None:
        self.state = {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "config_path": str(self.config_path),
            "task": self.task.to_dict(),
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "status": "running",
            "completed_stages": [],
            "artifacts": {},
            "metrics": {},
            "batches": [],
            "history": [],
        }
        if self.al_enabled:
            for name, default, minimum in (
                ("initial_size", 50, 2),
                ("batch_size", 20, 1),
                ("n_iterations", 5, 0),
            ):
                value = self.al_config.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"active_learning.{name} must be an integer >= {minimum}")
            if self.al_config.get("strategy", "entropy") not in {"entropy", "margin", "random"}:
                raise ValueError("active_learning.strategy must be entropy, margin, or random")
        collector = DataCollectionAgent(self.config_path)
        assets = collector.collect_media(modality=self.task.modality)
        _save_frame(assets, self.assets_path)
        self.state["artifacts"]["assets"] = str(self.assets_path)
        self._stage("collection")
        quality = DataQualityAgent(config=self.config)
        before = quality.detect_media_issues(assets)
        strategy = self.config.get("quality", {}).get("selected_strategy", "conservative")
        clean = quality.fix_media(assets, strategy=strategy)
        if clean.empty:
            raise ValueError("No usable media remain after quality checks")
        self.state["metrics"]["quality"] = {
            "strategy": strategy,
            "before": before,
            "after": quality.detect_media_issues(clean),
        }
        clean = self._split(clean)
        _save_frame(clean, self.clean_assets_path)
        self._stage("quality")
        units = prepare_media(
            clean, self.task, self.config.get("sampling", {}), self.run_dir / "media"
        )
        if units.empty:
            raise ValueError("Media preparation returned no annotation units")
        units = build_media_review_queue(units, self.task)
        units["annotation_status"] = "pending"
        _save_frame(units, self.units_path)
        self.state["artifacts"]["units"] = str(self.units_path)
        self._stage("preparation")
        candidates = units[units["split"].eq("train")] if self.al_enabled else units
        if self.al_enabled:
            required = self.al_config.get("initial_size", 50) + self.al_config.get(
                "n_iterations", 5
            ) * self.al_config.get("batch_size", 20)
            if len(candidates) < required:
                raise ValueError(
                    f"Active learning needs {required} training units; found {len(candidates)}. Reduce its budget or disable it."
                )
            candidates = candidates.sample(
                n=self.al_config.get("initial_size", 50), random_state=self.seed
            )
        self._new_batch(candidates["record_id"].astype(str).tolist())

    def _new_batch(self, record_ids: list[str]) -> None:
        batch_id = f"batch-{len(self.state['batches']):03d}"
        self.state["batches"].append({"id": batch_id, "record_ids": record_ids, "complete": False})
        self.state["active_batch"] = batch_id

    def _batch_units(self, batch: dict[str, Any]) -> pd.DataFrame:
        units = _load_frame(self.units_path)
        return units[units["record_id"].isin(batch["record_ids"])].copy().reset_index(drop=True)

    def _update_units(self, rows: pd.DataFrame) -> None:
        all_rows = _load_frame(self.units_path).to_dict("records")
        replacements = {row["record_id"]: row for row in rows.to_dict("records")}
        _save_frame(
            pd.DataFrame([replacements.get(row["record_id"], row) for row in all_rows]),
            self.units_path,
        )

    def _densify(self, record_id: str, step: float, batch: dict[str, Any]) -> None:
        if self.task.modality != "video" or not math.isfinite(step) or step <= 0:
            raise ValueError("Denser frames require video and a positive finite frame-step")
        rows = self._batch_units(batch)
        selected = rows[rows["record_id"].eq(record_id)]
        if selected.empty or selected.iloc[0]["annotation_status"] != "pending":
            raise ValueError("Only an unanswered record in the active batch can request new frames")
        row = selected.iloc[0]
        assets = _load_frame(self.clean_assets_path)
        asset = assets[assets["asset_id"].eq(row["asset_id"])]
        sampling = dict(self.config.get("sampling", {}))
        sampling["frame_interval_sec"] = step
        sampling["max_frames_per_clip"] = max(
            int(sampling.get("max_frames_per_clip", 8)),
            math.ceil((row["end_sec"] - row["start_sec"]) / step) + 1,
        )
        prepared = prepare_media(asset, self.task, sampling, self.run_dir / "media")
        refreshed = prepared[prepared["record_id"].eq(record_id)]
        if len(refreshed) != 1:
            raise ValueError("Resampling could not locate the original clip")
        index = selected.index[0]
        rows.at[index, "frames"] = refreshed.iloc[0]["frames"]
        self.annotator.prepare_requests(
            rows, self.run_dir / "batches" / batch["id"], batch_id=batch["id"]
        )
        self._update_units(rows)

    def _features(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy(deep=True)
        checked_sources: dict[str, str] = {}
        for index, row in rows.iterrows():
            source = str(row["media_path"])
            if source not in checked_sources:
                checked_sources[source] = file_sha256(Path(source))
            if checked_sources[source] != row["content_hash"]:
                raise ValueError("Source media changed after annotation; use a new run-id")
            if row.get("modality") != "video":
                continue
            start, end = row.get("human_start_sec"), row.get("human_end_sec")
            if start is None or end is None or pd.isna(start) or pd.isna(end):
                continue
            if float(start) != float(row["start_sec"]) or float(end) != float(row["end_sec"]):
                sampling = self.config.get("sampling", {})
                rows.at[index, "frames"] = sample_video_frames(
                    row["media_path"],
                    float(start),
                    float(end),
                    frame_interval_sec=float(sampling.get("frame_interval_sec", 1)),
                    max_frames=int(sampling.get("max_frames_per_clip", 8)),
                    max_dimension=int(sampling.get("max_dimension", 1024)),
                    output_dir=self.run_dir / "reviewed_frames",
                )
        config = dict(self.config.get("features", {}))
        config["cache_dir"] = str(self.run_dir / "feature_cache")
        if config.get("model"):
            local = self.root / config["model"]
            if local.is_dir():
                config["model"] = str(local)
        return extract_features(rows, config)

    def _process_batch(
        self,
        batch: dict[str, Any],
        *,
        annotations: Path | None,
        reviews: Path | None,
        review_mode: str,
        reviewer: str | None,
    ) -> PipelineResult | None:
        directory = self.run_dir / "batches" / batch["id"]
        rows = self._batch_units(batch)
        manifest = self.annotator.prepare_requests(rows, directory, batch_id=batch["id"])
        self.state["artifacts"].update(
            annotation_requests=str(manifest),
            annotation_responses=str(directory / "responses.jsonl"),
            review_queue=str(directory / "review_queue.jsonl"),
            review_decisions=str(directory / "review_decisions.jsonl"),
        )
        rows = self.annotator.import_responses(rows, directory, response_path=annotations)
        self._update_units(rows)
        pending = int(rows["annotation_status"].eq("pending").sum())
        if pending:
            return self._save(
                "annotation_required",
                f"{pending} samples await the vision agent. Read {manifest}; import responses with --annotations PATH or rerun after writing responses.jsonl.",
            )
        _save_frame(rows, directory / "review_queue.jsonl")
        if review_mode == "auto-only":
            unchecked = ~rows["reviewed"].fillna(False).astype(bool)
            rows.loc[unchecked, "final_label"] = rows.loc[unchecked, "auto_label"]
            batch["human_verified"] = False
        else:
            if review_mode == "terminal":
                raise ValueError(
                    "Visual review requires viewing images/video: use Streamlit or --reviews PATH"
                )
            decision_path = (
                Path(reviews) if reviews is not None else directory / "review_decisions.jsonl"
            )
            if reviews is not None and not decision_path.is_file():
                raise FileNotFoundError(decision_path)
            if decision_path.is_file():
                digest = hashlib.sha256(decision_path.read_bytes()).hexdigest()
                if digest != batch.get("review_digest"):
                    rows = apply_media_review(rows, decision_path, self.task, reviewer=reviewer)
                    self._update_units(rows)
                    batch["review_digest"] = digest
            _save_frame(rows, directory / "review_queue.jsonl")
            open_rows = ~rows["review_status"].isin(["accepted", "edited", "rejected"])
            if open_rows.any():
                return self._save(
                    "review_required",
                    f"{int(open_rows.sum())} samples need a human decision. Open Streamlit, save reviews, then rerun the same command.",
                )
            batch["human_verified"] = True
        self._update_units(rows)
        batch["complete"] = True
        return None

    def _advance(self) -> bool:
        """Select the next real annotation batch; never reveal reference labels."""
        if not self.al_enabled:
            return False
        units = _load_frame(self.units_path)
        selected_ids = {record for batch in self.state["batches"] for record in batch["record_ids"]}
        current = units[
            units["record_id"].isin(selected_ids)
            & units["final_label"].notna()
            & units["split"].eq("train")
        ]
        iteration = len(self.state["batches"]) - 1
        if len(self.state["history"]) <= iteration:
            self.state["history"].append(
                {
                    "iteration": iteration,
                    "n_labeled": len(current),
                    "queried_count": len(selected_ids),
                    "strategy": self.al_config.get("strategy", "entropy"),
                }
            )
        if iteration >= int(self.al_config.get("n_iterations", 5)):
            self._stage("active_learning")
            return False
        pool = units[~units["record_id"].isin(selected_ids) & units["split"].eq("train")]
        if pool.empty:
            raise ValueError("No unused training samples remain for the requested AL iterations")
        size = min(len(pool), int(self.al_config.get("batch_size", 20)))
        if current["final_label"].nunique() < 2:
            # A blind initial sample need not contain two classes. Acquire a
            # further real batch without pretending a classifier can fit it.
            positions = np.random.default_rng(self.seed + iteration).choice(
                len(pool), size=size, replace=False
            )
            self.state["history"][-1]["query_method"] = "random_until_two_classes"
        else:
            learner = ActiveLearningAgent(
                config={**self.al_config, "feature_column": "features"}, random_seed=self.seed
            )
            learner.fit(self._features(current), label_col="final_label")
            positions = learner.query(
                self._features(pool),
                strategy=self.al_config.get("strategy", "entropy"),
                batch_size=size,
                iteration=iteration,
            )
            self.state["history"][-1]["query_method"] = self.al_config.get("strategy", "entropy")
        self._new_batch(pool.iloc[positions]["record_id"].tolist())
        return True

    def _finish(self) -> PipelineResult:
        units = _load_frame(self.units_path)
        self._stage("annotation")
        human_verified = all(batch.get("human_verified") for batch in self.state["batches"])
        if human_verified:
            self._stage("human_review")
        verified = units["reviewed"].fillna(False).astype(bool)
        self.state["metrics"]["hitl"] = {
            "verified": human_verified,
            "reviewed_rows": int(verified.sum()),
            "changed_rows": int(units["review_changed"].fillna(False).astype(bool).sum()),
        }
        self.state["metrics"]["annotation"] = {
            "total_units": len(units),
            "annotated_units": int(units["annotation_status"].eq("annotated").sum()),
            "abstained_units": int(units["annotation_status"].eq("abstained").sum()),
            "unqueried_units": int(units["annotation_status"].eq("pending").sum()),
        }
        final = units[units["final_label"].notna()].copy()
        final_path = self.run_dir / "labeled.jsonl"
        _save_frame(final, final_path)
        self.state["artifacts"]["labeled_dataset"] = str(final_path)
        if self.task.modality == "video":
            intervals = self.run_dir / "intervals.json"
            _json(intervals, merge_video_intervals(final))
            self.state["artifacts"]["intervals"] = str(intervals)
        if self.training_config.get("enabled", False):
            train = final[final["split"].eq("train")]
            test = units[units["split"].eq("test")]
            reference = None
            # A partially labeled holdout is not silently reduced to an easier subset.
            if not test.empty and "source_label" in test and test["source_label"].notna().all():
                reference = self._features(test)
            model_path = self.run_dir / "model.joblib"
            trainer = TrainAgent(
                config={
                    **self.training_config,
                    "feature_column": "features",
                    "labels": list(self.task.labels),
                    "features": self.config.get("features", {"backend": "statistics"}),
                }
            )
            metrics = trainer.train_media(
                self._features(train), test_df=reference, model_path=model_path
            )
            metrics["reference_origin"] = "supplied_source_label" if reference is not None else None
            self.state["metrics"]["training"] = metrics
            self.state["artifacts"]["model"] = str(model_path)
            self._stage("training")
        if self.al_enabled:
            self.state["metrics"]["active_learning"] = {
                "mode": "live_annotation",
                "history": self.state["history"],
            }
        return self._save(
            "completed" if human_verified else "completed_without_hitl",
            "Visual pipeline completed. Artifacts and enabled training results are saved in this run directory.",
        )

    def run(
        self,
        *,
        annotations: Path | None = None,
        reviews: Path | None = None,
        review_mode: str = "required",
        reviewer: str | None = None,
        force: bool = False,
        offline: bool = False,
        frames: str | None = None,
        frame_step: float | None = None,
    ) -> PipelineResult:
        with _run_lock(self.run_dir):
            return self._run_unlocked(
                annotations=annotations,
                reviews=reviews,
                review_mode=review_mode,
                reviewer=reviewer,
                force=force,
                offline=offline,
                frames=frames,
                frame_step=frame_step,
            )

    def _run_unlocked(
        self,
        *,
        annotations: Path | None,
        reviews: Path | None,
        review_mode: str,
        reviewer: str | None,
        force: bool,
        offline: bool,
        frames: str | None,
        frame_step: float | None,
    ) -> PipelineResult:
        if review_mode not in {"required", "terminal", "auto-only"}:
            raise ValueError("Invalid review mode")
        if offline:
            raise ValueError(
                "--offline is the text fixture. Media are local already; choose statistics or local cached weights."
            )
        if (frames is None) != (frame_step is None):
            raise ValueError("--frames and --frame-step must be supplied together")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("config_hash") != self.config_hash:
                raise ValueError(
                    "Configuration changed. Use a new --run-id; existing annotations are preserved."
                )
            if force:
                raise ValueError(
                    "Media --force does not overwrite saved annotations. Use a new --run-id."
                )
            if self.state["status"] in {"completed", "completed_without_hitl"}:
                if annotations or reviews or frames:
                    raise ValueError(
                        "This run is completed; use a new run-id for a new annotation pass"
                    )
                return self._result(self.state)
        try:
            if not self.state or "preparation" not in self.state.get("completed_stages", []):
                self._initialize()
            if not self.state["batches"]:
                raise ValueError(
                    "Run initialization was incomplete; resolve the earlier error and use a new run-id"
                )
            batch = self.state["batches"][-1]
            if frames:
                self._densify(frames, float(frame_step), batch)
            # Resume may enter after a completed batch but before next query/training.
            if not batch["complete"] or reviews is not None:
                pending = self._process_batch(
                    batch,
                    annotations=annotations,
                    reviews=reviews,
                    review_mode=review_mode,
                    reviewer=reviewer,
                )
                if pending is not None:
                    return pending
                self._save("running", "Batch complete; continuing the pipeline.")
            while self._advance():
                pending = self._process_batch(
                    self.state["batches"][-1],
                    annotations=None,
                    reviews=None,
                    review_mode=review_mode,
                    reviewer=reviewer,
                )
                if pending is not None:
                    return pending
                self._save("running", "Batch complete; continuing the pipeline.")
            return self._finish()
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            if not self.state:
                raise
            return self._save("failed", str(exc))
