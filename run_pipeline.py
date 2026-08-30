"""Command-line entry point for the complete data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
