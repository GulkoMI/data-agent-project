"""Command-line entry point for the complete data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.common import read_yaml
from pipeline.runner import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--offline", action="store_true", help="Use the deterministic fixture.")
    parser.add_argument("--force", action="store_true", help="Recompute non-human stages.")
    parser.add_argument(
        "--review-mode",
        choices=["required", "terminal", "auto-only"],
        default="required",
        help="Default stops after creating a human-review queue.",
    )
    parser.add_argument("--reviewer", help="Required for terminal review.")
    parser.add_argument("--run-id", default="default", help="Separate, resumable media run name.")
    parser.add_argument(
        "--annotations", type=Path, help="Import the current agent's media JSONL answers."
    )
    parser.add_argument("--reviews", type=Path, help="Import real human media review decisions.")
    parser.add_argument(
        "--status", action="store_true", help="Read media run status without running stages."
    )
    parser.add_argument("--frames", help="Request denser frames for an unanswered media record ID.")
    parser.add_argument("--frame-step", type=float, help="Seconds between requested video frames.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = read_yaml(args.config)
    modality = config.get("project", {}).get(
        "modality", config.get("task", {}).get("modality", "text")
    )
    if modality in {"image", "video"}:
        from pipeline.media_runner import MediaPipelineRunner

        try:
            runner = MediaPipelineRunner(args.config, run_id=args.run_id)
            result = (
                runner.status()
                if args.status
                else runner.run(
                    offline=args.offline,
                    force=args.force,
                    review_mode=args.review_mode,
                    reviewer=args.reviewer,
                    annotations=args.annotations,
                    reviews=args.reviews,
                    frames=args.frames,
                    frame_step=args.frame_step,
                )
            )
        except (ValueError, OSError, RuntimeError) as exc:
            print(json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))
        return (
            0
            if result.status
            in {
                "completed",
                "completed_without_hitl",
                "annotation_required",
                "review_required",
                "not_started",
            }
            else 1
        )
    if (
        args.annotations
        or args.reviews
        or args.status
        or args.frames
        or args.frame_step is not None
        or args.run_id != "default"
    ):
        raise SystemExit("Media arguments require project.modality: image or video")
    result = PipelineRunner(args.config).run(
        offline=args.offline,
        force=args.force,
        review_mode=args.review_mode,
        reviewer=args.reviewer,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))
    return 0 if result.status in {"completed", "completed_without_hitl", "review_required"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
