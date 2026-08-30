"""End-to-end orchestration for the cross-domain sentiment data project.

The runner intentionally stops at the human-review boundary unless a verified
corrections file exists or an explicit non-HITL smoke-test mode is selected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from agents.al_agent import ActiveLearningAgent
from agents.annotation_agent import AnnotationAgent
from agents.common import (
    ProjectPaths,
    ensure_unified_schema,
    load_frame,
    read_yaml,
    save_frame,
    utc_now_iso,
    write_json,
)
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.train_agent import TrainAgent
from utils.ollama import ollama_chat

ReviewMode = Literal["required", "terminal", "auto-only"]


@dataclass
class PipelineResult:
    """Machine-readable pipeline outcome."""

    status: str
    completed_stages: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completed_stages": self.completed_stages,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "message": self.message,
        }


class PipelineRunner:
    """Run collection, quality, annotation, HITL, AL, and model training."""

    def __init__(self, config: str | Path = "config.yaml") -> None:
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        self.config = read_yaml(self.config_path)
        self.base_paths = ProjectPaths.from_config(self.root, self.config)
        self._configure_runtime(offline=False)

    def _configure_runtime(self, *, offline: bool) -> None:
        """Keep smoke artifacts isolated from live/submission artifacts."""
        self.offline = offline
        if offline:
            smoke = self.root / "data/smoke"
            self.paths = ProjectPaths(
                root=self.root,
                raw=smoke / "raw",
                processed=smoke / "processed",
                labeled=smoke / "labeled",
                review=smoke / "review",
                cache=smoke / "cache",
                reports=self.root / "reports/smoke",
                models=self.root / "models/smoke",
            )
            self.raw_path = self.paths.raw / "reviews_raw.parquet"
        else:
            self.paths = self.base_paths
            self.raw_path = self.root / self.config["collection"]["output"]
        self.paths.create()
        self.clean_path = self.paths.processed / "reviews_clean.parquet"
        self.auto_path = self.paths.labeled / "reviews_auto_labeled.parquet"
        self.queue_path = self.paths.review / "review_queue.csv"
        self.corrected_path = self.paths.review / "review_queue_corrected.csv"
        self.final_path = self.paths.labeled / "reviews_final.parquet"
        self.state_path = self.paths.reports / "pipeline_state.json"

    def run(
        self,
        *,
        offline: bool = False,
        force: bool = False,
        review_mode: ReviewMode = "required",
        reviewer: str | None = None,
    ) -> PipelineResult:
        """Run every available stage and return a precise status.

        ``required`` creates a queue and stops until a reviewed corrections file
        exists. ``terminal`` asks a person to review every queued record.
        ``auto-only`` is for CI and explicitly does not satisfy HITL.
        """
        if review_mode not in {"required", "terminal", "auto-only"}:
            raise ValueError("review_mode must be required, terminal, or auto-only")
        self._configure_runtime(offline=offline)
        result = PipelineResult(status="running")
        started_at = utc_now_iso()
        try:
            raw = self._collect(offline=offline, force=force)
            result.completed_stages.append("collection")
            result.artifacts["raw_dataset"] = str(self.raw_path)

            clean, quality_metrics = self._quality(raw, force=force)
            result.completed_stages.append("quality")
            result.artifacts["clean_dataset"] = str(self.clean_path)
            result.metrics["quality"] = quality_metrics
            llm_report = self._maybe_write_llm_advice(quality_metrics)
            if llm_report is not None:
                result.artifacts["llm_quality_advice"] = str(llm_report)

            labeled, annotation_metrics = self._annotate(clean, force=force)
            result.completed_stages.append("annotation")
            result.artifacts.update(
                auto_labeled_dataset=str(self.auto_path),
                review_queue=str(self.queue_path),
                labelstudio_import=str(self.paths.review / "labelstudio_import.json"),
            )
            result.metrics["annotation"] = annotation_metrics

            final, hitl_metrics = self._human_review(labeled, mode=review_mode, reviewer=reviewer)
            result.metrics["hitl"] = hitl_metrics
            if final is None:
                result.status = "review_required"
                result.message = (
                    f"Automatic labeling is complete. Review {self.queue_path}, "
                    f"save corrections as {self.corrected_path}, "
                    "and rerun the same command."
                )
                self._save_state(result, started_at)
                return result

            result.completed_stages.append(
                "human_review" if hitl_metrics["verified"] else "auto_only"
            )
            result.artifacts["final_dataset"] = str(self.final_path)

            al_metrics, selected_training, outer_holdout = self._active_learning(final)
            result.completed_stages.append("active_learning")
            result.artifacts["learning_curve"] = str(
                self.paths.reports / "active_learning" / "learning_curves.png"
            )
            result.metrics["active_learning"] = al_metrics
            result.artifacts["al_selected_training"] = str(
                self.paths.labeled / "al_selected_training.parquet"
            )
            result.artifacts["outer_holdout"] = str(self.paths.labeled / "outer_holdout.parquet")

            model_metrics = self._train(selected_training, outer_holdout)
            result.completed_stages.append("training")
            result.artifacts["model"] = str(self.paths.models / "sentiment_model.joblib")
            result.metrics["model"] = model_metrics

            self._write_final_reports(final, result, offline=offline)
            result.completed_stages.append("reporting")
            result.artifacts["final_report"] = str(self.paths.reports / "final_report.md")
            result.status = "completed" if hitl_metrics["verified"] else "completed_without_hitl"
            result.message = (
                "Pipeline completed with verified human review."
                if hitl_metrics["verified"]
                else "Smoke pipeline completed, but no human review was claimed."
            )
            self._save_state(result, started_at)
            return result
        except Exception as error:
            result.status = "failed"
            result.message = f"{type(error).__name__}: {error}"
            self._save_state(result, started_at)
            raise

    def _collect(self, *, offline: bool, force: bool) -> pd.DataFrame:
        collection_meta = self.paths.cache / "collection_stage.json"
        collection_config_hash = self._object_fingerprint(self.config["collection"])
        if self.raw_path.exists() and collection_meta.exists() and not force and not offline:
            metadata = json.loads(collection_meta.read_text(encoding="utf-8"))
            if metadata.get("config_fingerprint") == collection_config_hash:
                return ensure_unified_schema(load_frame(self.raw_path))
        if offline:
            fixture = self.root / "data/fixtures/reviews_fixture.csv"
            frame = ensure_unified_schema(load_frame(fixture))
            if frame["source"].nunique() < 2:
                raise ValueError("Offline fixture must represent at least two sources")
            save_frame(frame, self.raw_path)
            save_frame(frame, self.raw_path.with_suffix(".csv"))
            write_json(
                {
                    "status": "ok",
                    "mode": "offline_fixture",
                    "generated_at": utc_now_iso(),
                    "rows": len(frame),
                    "sources": frame["source"].value_counts().to_dict(),
                    "note": "Fixture data is for smoke tests, not the submitted HF datasets.",
                },
                self.paths.raw / "collection_manifest.json",
            )
            return frame
        frame = DataCollectionAgent(self.config_path).run()
        write_json(
            {
                "config_fingerprint": collection_config_hash,
                "output_fingerprint": self._frame_fingerprint(frame),
                "generated_at": utc_now_iso(),
            },
            collection_meta,
        )
        return frame

    def _quality(self, raw: pd.DataFrame, *, force: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
        agent = DataQualityAgent(self.config_path, report_dir=self.paths.reports / "quality")
        selected = str(self.config["quality"].get("selected_strategy", "conservative"))
        meta_path = self.paths.cache / "quality_stage.json"
        expected_meta = {
            "input_fingerprint": self._frame_fingerprint(raw),
            "config_fingerprint": self._object_fingerprint(self.config["quality"]),
        }
        cache_valid = self._metadata_matches(meta_path, expected_meta)
        if self.clean_path.exists() and cache_valid and not force:
            cleaned = load_frame(self.clean_path)
        else:
            cleaned = agent.fix(raw, selected)
            save_frame(cleaned, self.clean_path)
            save_frame(cleaned, self.clean_path.with_suffix(".csv"))
            write_json({**expected_meta, "generated_at": utc_now_iso()}, meta_path)
        strategy_results = agent.evaluate_strategies(raw, ["conservative", "strict"])
        agent.generate_report(raw, cleaned, strategy=selected)
        summary: dict[str, Any] = {
            "selected_strategy": selected,
            "before": agent.detect_issues(raw),
            "after": agent.detect_issues(cleaned),
            "strategies": {},
        }
        for name, payload in strategy_results.items():
            summary["strategies"][name] = {
                "rows": len(payload["dataframe"]),
                "issues": payload["issues"],
                "actions": payload["actions"],
            }
        return cleaned, summary

    def _annotate(self, clean: pd.DataFrame, *, force: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
        annotation_config: str | Path | dict[str, Any] = self.config_path
        if self.offline:
            annotation_config = {
                "annotation": {
                    **self.config["annotation"],
                    "backend": "lexicon",
                    "allow_lexicon_fallback": True,
                }
            }
        agent = AnnotationAgent(modality="text", config=annotation_config)
        meta_path = self.paths.cache / "annotation_stage.json"
        expected_meta = {
            "input_fingerprint": self._frame_fingerprint(clean),
            "config_fingerprint": self._object_fingerprint(self.config["annotation"]),
        }
        cache_valid = self._metadata_matches(meta_path, expected_meta)
        if self.auto_path.exists() and cache_valid and not force:
            labeled = load_frame(self.auto_path)
        else:
            labeled = agent.auto_label(clean)
            save_frame(labeled, self.auto_path)
            save_frame(labeled, self.auto_path.with_suffix(".csv"))
            write_json({**expected_meta, "generated_at": utc_now_iso()}, meta_path)
        queue = agent.flag_for_review(labeled)
        safe_columns = [
            "record_id",
            "text",
            "source",
            "auto_label",
            "confidence",
            "needs_review",
            "review_reason",
        ]
        queue_export = queue.loc[:, safe_columns].copy()
        queue_export["human_label"] = ""
        queue_export["reviewer"] = ""
        save_frame(queue_export, self.queue_path)
        write_json(
            {
                "generated_at": utc_now_iso(),
                "record_ids": sorted(queue_export["record_id"].astype(str).tolist()),
                "rows": len(queue_export),
                "note": "Gold/source labels are intentionally excluded from reviewer artifacts.",
            },
            self.paths.review / "review_manifest.json",
        )
        agent.export_to_labelstudio(queue_export, self.paths.review / "labelstudio_import.json")
        agent.generate_spec(labeled, output_path=self.paths.reports / "annotation_spec.md")
        metrics = agent.check_quality(labeled)
        write_json(metrics, self.paths.reports / "annotation_report.json")
        self._write_annotation_report(metrics, queue)
        return labeled, metrics

    def _human_review(
        self,
        labeled: pd.DataFrame,
        *,
        mode: ReviewMode,
        reviewer: str | None,
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        agent = AnnotationAgent(
            modality="text",
            config=self.config_path,
            backend=lambda texts: [],
        )
        if self.corrected_path.exists():
            corrected = pd.read_csv(self.corrected_path, dtype={"record_id": "string"})
            expected = pd.read_csv(self.queue_path, dtype={"record_id": "string"})
            expected_ids = set(expected["record_id"].astype(str))
            corrected_ids = set(corrected["record_id"].astype(str))
            if corrected_ids != expected_ids or len(corrected) != len(expected):
                missing = sorted(expected_ids - corrected_ids)
                extra = sorted(corrected_ids - expected_ids)
                raise ValueError(
                    "Human review must cover the current queue exactly; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            if "auto_label" not in corrected.columns:
                raise ValueError(
                    "Corrected review file must retain auto_label for stale-queue validation"
                )
            expected_auto = expected.set_index("record_id")["auto_label"].astype(str)
            corrected_auto = corrected.set_index("record_id")["auto_label"].astype(str)
            if not corrected_auto.reindex(expected_auto.index).equals(expected_auto):
                raise ValueError("Corrected review file belongs to a stale auto-label queue")
            final = agent.apply_human_review(labeled, corrected, reviewer=reviewer)
            save_frame(final, self.final_path)
            save_frame(final, self.final_path.with_suffix(".csv"))
            reviewed = (
                final["reviewer"].notna()
                if "reviewer" in final
                else pd.Series(False, index=final.index)
            )
            changed = final.get("review_changed", pd.Series(False, index=final.index)).fillna(False)
            metrics = {
                "verified": True,
                "reviewed_rows": int(reviewed.sum()),
                "changed_labels": int(changed.sum()),
                "reviewer_count": int(final.loc[reviewed, "reviewer"].nunique()),
            }
            post_review_quality = agent.check_quality(final)
            metrics["auto_vs_human"] = post_review_quality["against_human_label"]
            write_json(
                post_review_quality,
                self.paths.reports / "annotation_post_review.json",
            )
            write_json(metrics, self.paths.reports / "hitl_report.json")
            return final, metrics
        if mode == "terminal":
            if not (reviewer or "").strip():
                raise ValueError("--reviewer is required for terminal review")
            corrections = self._terminal_review(reviewer=str(reviewer).strip())
            save_frame(corrections, self.corrected_path)
            return self._human_review(labeled, mode="required", reviewer=reviewer)
        if mode == "auto-only":
            final = labeled.copy()
            final["final_label"] = final["auto_label"]
            save_frame(final, self.final_path)
            save_frame(final, self.final_path.with_suffix(".csv"))
            metrics = {
                "verified": False,
                "reviewed_rows": 0,
                "changed_labels": 0,
                "warning": "No human review was performed; this mode does not satisfy HITL.",
            }
            write_json(metrics, self.paths.reports / "hitl_report.json")
            return final, metrics
        metrics = {
            "verified": False,
            "reviewed_rows": 0,
            "changed_labels": 0,
            "pending_rows": len(load_frame(self.queue_path)),
        }
        write_json(metrics, self.paths.reports / "hitl_report.json")
        return None, metrics

    def _active_learning(
        self, final: pd.DataFrame
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        config = self.config["active_learning"]
        composite_strata = final["source"].astype(str) + "::" + final["source_label"].astype(str)
        outer_test_size = float(self.config["training"].get("test_size", 0.2))
        if len(final) < 100:
            # Four composite source×label strata need at least four holdout rows.
            outer_test_size = max(4, int(len(final) * outer_test_size + 0.999)) / len(final)
        development, outer_holdout = train_test_split(
            final,
            test_size=outer_test_size,
            random_state=int(self.config["training"].get("random_seed", 42)) + 10_000,
            stratify=composite_strata,
        )
        development = development.reset_index(drop=True)
        outer_holdout = outer_holdout.reset_index(drop=True)

        # The 12-row fixture still exercises the complete code path. Live runs
        # retain the assignment contract N=50, five iterations, batch=20.
        if len(development) < 100:
            initial_size = 4
            n_iterations = 2
            batch_size = 1
            inner_test_size = 0.25
        else:
            initial_size = int(config.get("initial_size", 50))
            n_iterations = int(config.get("n_iterations", 5))
            batch_size = int(config.get("batch_size", 20))
            inner_test_size = float(config.get("test_size", 0.2))
        agent = ActiveLearningAgent(
            model=str(config.get("model", "logreg")),
            config=config,
            random_seed=int(config.get("random_seed", 42)),
        )
        # AL strategy selection sees only the development partition. The final
        # holdout remains untouched until TrainAgent evaluation.
        results = agent.compare_strategies(
            development,
            strategies=list(config.get("strategies", ["entropy", "random"])),
            initial_size=initial_size,
            n_iterations=n_iterations,
            batch_size=batch_size,
            test_size=inner_test_size,
            initial_label_col="final_label",
            oracle_label_col="source_label",
        )
        metrics = agent.report(results, self.paths.reports / "active_learning")
        strategy = (
            "entropy" if "entropy" in results["histories"] else next(iter(results["histories"]))
        )
        initial_ids = set(results["split_manifest"]["initial_record_ids"])
        queried_ids: set[str] = set()
        for point in results["histories"][strategy]:
            queried_ids.update(point["selected_record_ids"])
        reviewed_ids = (
            set(development.loc[development["reviewer"].notna(), "record_id"].astype(str))
            if "reviewer" in development.columns
            else set()
        )
        selected_ids = initial_ids | queried_ids | reviewed_ids
        selected_training = development[
            development["record_id"].astype(str).isin(selected_ids)
        ].copy()
        selected_training["training_label"] = selected_training["source_label"]
        selected_training["training_label_origin"] = "source_oracle_after_al_query"
        initial_mask = selected_training["record_id"].astype(str).isin(initial_ids)
        selected_training.loc[initial_mask, "training_label"] = selected_training.loc[
            initial_mask, "final_label"
        ]
        selected_training.loc[initial_mask, "training_label_origin"] = "initial_final_label"
        reviewed_training_mask = selected_training["record_id"].astype(str).isin(reviewed_ids)
        selected_training.loc[reviewed_training_mask, "training_label"] = selected_training.loc[
            reviewed_training_mask, "final_label"
        ]
        selected_training.loc[reviewed_training_mask, "training_label_origin"] = "human_review"
        al_seen_ids = (
            set(results["split_manifest"]["initial_record_ids"])
            | set(results["split_manifest"]["pool_record_ids"])
            | set(results["split_manifest"]["test_record_ids"])
        )
        outer_ids = set(outer_holdout["record_id"].astype(str))
        if al_seen_ids & outer_ids:
            raise RuntimeError("Outer holdout leaked into the active-learning experiment")
        save_frame(selected_training, self.paths.labeled / "al_selected_training.parquet")
        save_frame(outer_holdout, self.paths.labeled / "outer_holdout.parquet")
        outer_manifest = {
            "random_seed": int(self.config["training"].get("random_seed", 42)) + 10_000,
            "development_record_ids": development["record_id"].astype(str).tolist(),
            "outer_holdout_record_ids": outer_holdout["record_id"].astype(str).tolist(),
            "selected_training_record_ids": selected_training["record_id"].astype(str).tolist(),
            "selection_strategy": strategy,
            "human_reviewed_training_record_ids": sorted(reviewed_ids),
            "note": "Outer holdout is never used for AL fit, query, or strategy selection.",
        }
        write_json(
            outer_manifest,
            self.paths.reports / "active_learning" / "outer_split_manifest.json",
        )
        metrics.update(
            development_rows=len(development),
            selected_training_rows=len(selected_training),
            outer_holdout_rows=len(outer_holdout),
            human_reviewed_training_rows=int(reviewed_training_mask.sum()),
            selection_strategy=strategy,
        )
        return metrics, selected_training, outer_holdout

    def _train(
        self, selected_training: pd.DataFrame, outer_holdout: pd.DataFrame
    ) -> dict[str, Any]:
        agent = TrainAgent(self.config["training"])
        return agent.train_and_evaluate(
            selected_training,
            test_df=outer_holdout,
            train_label_col="training_label",
            gold_label_col="source_label",
            model_path=self.paths.models / "sentiment_model.joblib",
            metrics_path=self.paths.reports / "model_metrics.json",
        )

    def _terminal_review(self, *, reviewer: str) -> pd.DataFrame:
        queue = pd.read_csv(self.queue_path, dtype={"record_id": "string"})
        rows: list[dict[str, Any]] = []
        print("\nHuman review: Enter=confirm auto-label, p=positive, n=negative.\n")
        for position, row in queue.iterrows():
            print(f"[{position + 1}/{len(queue)}] {row['record_id']}")
            print(f"auto={row['auto_label']} confidence={float(row['confidence']):.3f}")
            print(str(row["text"]))
            while True:
                answer = input("label [Enter/p/n]: ").strip().lower()
                if answer in {"", "p", "positive", "n", "negative"}:
                    break
                print("Use Enter, p/positive, or n/negative.")
            label = {"p": "positive", "n": "negative"}.get(answer, answer)
            rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "auto_label": str(row["auto_label"]),
                    "human_label": label,
                    "reviewer": reviewer,
                    "reviewed_at": utc_now_iso(),
                }
            )
        return pd.DataFrame(rows)

    def _write_annotation_report(self, metrics: dict[str, Any], queue: pd.DataFrame) -> None:
        path = self.paths.reports / "annotation_report.md"
        lines = [
            "# Annotation report",
            "",
            f"- Auto-label distribution: `{metrics['label_dist']}`",
            f"- Mean confidence: `{metrics['confidence_mean']}`",
            f"- Agreement with source labels: `{metrics['against_source_label']['agreement']}`",
            f"- Cohen's kappa vs source labels: `{metrics['against_source_label']['kappa']}`",
            f"- Rows sent to human review: `{len(queue)}`",
            f"- Annotation backends: `{metrics['backend_distribution']}`",
            f"- Degraded fallback used: `{metrics['degraded_fallback_used']}`",
            "",
            "Source labels are retained only for audit/evaluation and are never overwritten.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _maybe_write_llm_advice(self, quality: dict[str, Any]) -> Path | None:
        """Optional local-Ollama bonus; deterministic pipeline never depends on it."""
        settings = self.config.get("annotation", {}).get("ollama", {})
        if not bool(settings.get("enabled", False)):
            return None
        prompt = (
            "You are a data-quality assistant. Briefly explain the following before/after "
            "metrics for English cross-domain sentiment classification, identify risks, and "
            "justify or challenge the selected cleaning strategy. Do not invent values.\n\n"
            + json.dumps(quality, ensure_ascii=False, default=str)[:20_000]
        )
        answer = ollama_chat(
            prompt,
            model=str(settings.get("model", "gemma3:4b")),
            base_url=str(settings.get("base_url", "http://localhost:11434")),
        )
        if not answer:
            return None
        target = self.paths.reports / "llm_quality_advice.md"
        target.write_text(
            "# Optional LLM quality advice\n\n"
            "> Generated locally by Ollama; this narrative is advisory and does not alter data.\n\n"
            + answer.strip()
            + "\n",
            encoding="utf-8",
        )
        return target

    def _write_final_reports(
        self, final: pd.DataFrame, result: PipelineResult, *, offline: bool
    ) -> None:
        (self.paths.labeled / "DATA_CARD.md").write_text(
            self._data_card(final, offline=offline), encoding="utf-8"
        )
        model = result.metrics["model"]
        hitl = result.metrics["hitl"]
        sources = final["source"].value_counts().to_dict()
        labels = final["final_label"].value_counts().to_dict()
        lines = [
            "# Final Data Project Report",
            "",
            "## 1. Task and dataset",
            "",
            (
                "Binary cross-domain sentiment classification for English Amazon product "
                "and Steam game reviews loaded from two Hugging Face datasets."
            ),
            f"Rows: **{len(final)}**. Sources: `{sources}`. Final labels: `{labels}`.",
            "",
            "## 2. What each agent did",
            "",
            "- DataCollectionAgent loaded and normalized two independent Hugging Face datasets.",
            "- DataQualityAgent diagnosed missing values, duplicates, length outliers, and imbalance; two strategies were compared.",
            "- AnnotationAgent generated sentiment predictions, confidence, a specification, Label Studio tasks, and a review queue.",
            "- ActiveLearningAgent compared entropy, margin, and random on exactly the same split.",
            "- TrainAgent trained TF-IDF + logistic regression and evaluated an untouched source-label holdout.",
            "",
            "## 3. Human-in-the-loop",
            "",
            f"Verified: **{hitl['verified']}**. Reviewed rows: **{hitl['reviewed_rows']}**. Changed labels: **{hitl['changed_labels']}**.",
            (
                f"Auto-vs-human agreement: **{hitl.get('auto_vs_human', {}).get('agreement', 'n/a')}**; "
                f"Cohen's kappa: **{hitl.get('auto_vs_human', {}).get('kappa', 'n/a')}**."
            ),
            "Review provenance is stored per row in `reviewer` and `reviewed_at`.",
            "",
            "## 4. Metrics",
            "",
            f"Final holdout accuracy: **{model['accuracy']:.4f}**; macro F1: **{model['f1_macro']:.4f}**.",
            f"The final model used **{model['train_rows']}** AL-selected/reviewed rows and an untouched outer holdout of **{model['test_rows']}** rows.",
            f"Per-domain holdout metrics: `{model.get('by_source', {})}`.",
            f"Annotation agreement with source labels: **{result.metrics['annotation']['against_source_label']['agreement']}**.",
            "Detailed stage metrics are saved under `reports/`.",
            "",
            "## 5. Retrospective",
            "",
            "What worked: stable IDs, isolated source failures, preserved source/auto/human/final labels, and reproducible AL splits.",
            "Limitations: ratings are weak gold labels, domains differ, confidence is not calibrated, and AL reveals source labels only as a clearly marked simulation oracle after querying.",
            (
                "Next: sample more Steam titles from future dataset snapshots, calibrate "
                "probabilities, and repeat AL with labels from multiple annotators."
            ),
        ]
        (self.paths.reports / "final_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _data_card(self, frame: pd.DataFrame, *, offline: bool) -> str:
        return (
            "\n".join(
                [
                    "# Cross-domain Reviews — Data Card",
                    "",
                    "## Intended use",
                    "",
                    "Educational binary sentiment classification and data-pipeline evaluation.",
                    "",
                    "## Data",
                    "",
                    f"- Rows: {len(frame)}",
                    f"- Sources: `{frame['source'].value_counts().to_dict()}`",
                    f"- Labels: `{frame['final_label'].value_counts().to_dict()}`",
                    f"- Mode: {'offline fixture' if offline else 'Hugging Face dataset snapshots'}",
                    "- Fields: record_id, text, source, source_label, auto_label, confidence, human_label, final_label.",
                    "",
                    "## Provenance and licensing",
                    "",
                    (
                        "Both configured sources are loaded through Hugging Face Datasets: "
                        "`mteb/amazon_polarity` and "
                        "[`reapxdev/steam-reviews-scraper`](https://huggingface.co/datasets/"
                        "reapxdev/steam-reviews-scraper)."
                    ),
                    (
                        "For Steam, `reviewText` supplies the text, `votedUp` supplies the "
                        "negative/positive weak label, and `language` filters English rows. "
                        "No Steam API or Steam token is used. Consult current dataset terms "
                        "before redistributing raw text."
                    ),
                    "",
                    "## Limitations",
                    "",
                    "Ratings are imperfect sentiment labels. Reviews may contain personal information, abuse, sarcasm, language mismatch, and domain-specific bias.",
                ]
            )
            + "\n"
        )

    def _save_state(self, result: PipelineResult, started_at: str) -> None:
        payload = result.as_dict()
        payload.update(
            started_at=started_at,
            finished_at=utc_now_iso(),
            config=str(self.config_path),
        )
        write_json(payload, self.state_path)

    @staticmethod
    def _object_fingerprint(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _frame_fingerprint(cls, frame: pd.DataFrame) -> str:
        columns = [
            column
            for column in ("record_id", "text", "label", "source", "source_id")
            if column in frame.columns
        ]
        stable = frame.loc[:, columns].copy()
        if "record_id" in stable:
            stable = stable.sort_values("record_id", kind="stable")
        payload = stable.fillna("").astype(str).to_csv(index=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_matches(path: Path, expected: dict[str, str]) -> bool:
        if not path.exists():
            return False
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return all(actual.get(key) == value for key, value in expected.items())


__all__ = ["PipelineResult", "PipelineRunner", "ReviewMode"]
