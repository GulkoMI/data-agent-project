"""Explicit, partial human decisions for visual annotation batches."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents.common import utc_now_iso
from agents.contracts import TaskSpec


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def _ids(frame: pd.DataFrame, name: str) -> None:
    if "record_id" not in frame:
        raise ValueError(f"{name} needs record_id")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} contains empty record_id")
    if frame["record_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate record_id")


def build_media_review_queue(df: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    """Add review fields without confirming predictions or replacing source labels.

    The returned frame retains internal metadata. A UI should display an explicit
    allowlist of fields, excluding source labels and source filenames.
    """
    _ids(df, "Review input")
    result = df.copy(deep=True)
    defaults = {
        "human_label": None, "review_status": "pending", "reviewed": False,
        "reviewer": None, "reviewed_at": None, "review_changed": False,
        "final_label": None, "needs_review": True, "review_note": "",
        "human_start_sec": None, "human_end_sec": None, "review_history": None,
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = pd.Series([default] * len(result), index=result.index, dtype=object)
        else:
            result[column] = result[column].astype(object)
            if default is not None:
                result[column] = result[column].where(result[column].notna(), default)
    if "auto_label" in result:
        labels = result["auto_label"].dropna()
        if not labels.isin(task.labels).all():
            raise ValueError("Review input contains unsupported auto_label")
    return result


def apply_media_review(
    df: pd.DataFrame,
    decisions: pd.DataFrame | str | Path,
    task: TaskSpec,
    *,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
) -> pd.DataFrame:
    """Apply explicit decisions by ID, leaving unsent/unchecked rows untouched.

    Required decision fields: ``record_id`` and ``review_status``. If ``reviewed``
    is present, only checked rows apply. ``accepted`` confirms the original agent
    label and interval; ``edited`` requires ``human_label`` and allows optional
    ``human_start_sec``/``human_end_sec`` inside the prepared clip. Rejected rows
    are excluded from training; uncertain rows remain open for further review.
    """
    result = build_media_review_queue(df, task)
    if isinstance(decisions, (str, Path)):
        path = Path(decisions)
        if path.suffix.lower() == ".jsonl":
            review = pd.read_json(path, lines=True, dtype={"record_id": str})
        else:
            review = pd.read_csv(path, dtype={"record_id": str})
    else:
        review = decisions.copy(deep=True)
    _ids(review, "Decisions")
    unknown = set(review["record_id"].astype(str)) - set(result["record_id"].astype(str))
    if unknown:
        raise ValueError(f"Decisions contain unknown record_id: {sorted(unknown)}")
    if "review_status" not in review:
        raise ValueError("Decisions need explicit review_status")
    if "reviewed" in review:
        checked = review["reviewed"]
        if not checked.map(lambda value: isinstance(value, bool)).all():
            raise ValueError("reviewed must contain booleans")
        review = review.loc[checked].copy()
    lookup = {str(row["record_id"]): index for index, row in result.iterrows()}
    now = reviewed_at or utc_now_iso()
    for _, decision in review.iterrows():
        index = lookup[str(decision["record_id"])]
        old = result.loc[index]
        status = decision["review_status"]
        if status not in {"accepted", "edited", "rejected", "uncertain"}:
            raise ValueError(f"Invalid review_status: {status!r}")
        actor = decision.get("reviewer")
        actor = str(reviewer or "").strip() if _blank(actor) else str(actor).strip()
        if not actor:
            raise ValueError("A non-empty reviewer is required for every decision")
        timestamp = decision.get("reviewed_at")
        timestamp = now if _blank(timestamp) else str(timestamp)
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError("reviewed_at must be an ISO timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        if (
            "request_id" in decision
            and not _blank(decision["request_id"])
            and decision["request_id"] != old.get("request_id")
        ):
            raise ValueError("Stale review decision: request_id changed")

        source_start = float(old.get("start_sec", 0.0))
        source_end = float(old.get("end_sec", source_start))
        start = old.get("auto_start_sec")
        start = source_start if _blank(start) else float(start)
        end = old.get("auto_end_sec")
        end = source_end if _blank(end) else float(end)
        label = None
        if status == "accepted":
            label = old.get("auto_label")
            if _blank(label):
                raise ValueError("Cannot accept an absent auto_label; use edited with a class")
            correction = decision.get("human_label")
            if not _blank(correction) and correction != label:
                raise ValueError("Changed label requires review_status=edited")
            for field, expected in (("human_start_sec", start), ("human_end_sec", end)):
                value = decision.get(field)
                if not _blank(value) and float(value) != expected:
                    raise ValueError("Changed interval requires review_status=edited")
        elif status == "edited":
            label = decision.get("human_label")
            if _blank(label):
                raise ValueError("Edited decision requires human_label")
            for field in ("human_start_sec", "human_end_sec"):
                value = decision.get(field)
                if not _blank(value):
                    if isinstance(value, bool):
                        raise ValueError("Review timestamps must be finite numbers")
                    if field == "human_start_sec":
                        start = float(value)
                    else:
                        end = float(value)
        if label is not None and label not in task.labels:
            raise ValueError(f"Unsupported human_label: {label!r}")
        if not all(math.isfinite(value) for value in (start, end)):
            raise ValueError("Review timestamps must be finite")
        if not source_start <= start <= end <= source_end:
            raise ValueError("Human interval lies outside prepared clip")
        if task.modality == "video" and start == end:
            raise ValueError("Human video interval must have positive duration")

        changed = status in {"accepted", "edited"} and (
            label != old.get("auto_label")
            or start != (source_start if _blank(old.get("auto_start_sec")) else old["auto_start_sec"])
            or end != (source_end if _blank(old.get("auto_end_sec")) else old["auto_end_sec"])
        )
        note = decision.get("review_note", "")
        note = "" if _blank(note) else str(note)
        event = {
            "review_status": status, "human_label": label, "start_sec": start,
            "end_sec": end, "reviewer": actor, "reviewed_at": timestamp, "review_note": note,
            "auto_label": None if _blank(old.get("auto_label")) else old.get("auto_label"),
            "request_id": None if _blank(old.get("request_id")) else old.get("request_id"),
        }
        history = old.get("review_history")
        if isinstance(history, str):
            history = json.loads(history)
        elif hasattr(history, "tolist"):
            history = history.tolist()
        history = list(history) if isinstance(history, list) else []
        # An undated replay must not manufacture a second human review event.
        if (
            history
            and reviewed_at is None
            and _blank(decision.get("reviewed_at"))
            and {key: value for key, value in history[-1].items() if key != "reviewed_at"}
            == {key: value for key, value in event.items() if key != "reviewed_at"}
        ):
            timestamp = history[-1]["reviewed_at"]
            event["reviewed_at"] = timestamp
        # Changed decisions retain previous review events.
        if not history or history[-1] != event:
            history.append(event)
        updates = {
            "human_label": label, "final_label": label,
            "human_start_sec": start if label is not None else None,
            "human_end_sec": end if label is not None else None,
            "review_status": status, "reviewed": True, "reviewer": actor,
            "reviewed_at": timestamp, "review_changed": bool(changed),
            "needs_review": status == "uncertain", "review_note": note,
            "review_history": history,
        }
        for column, value in updates.items():
            result.at[index, column] = value
    return result
