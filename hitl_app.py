"""Streamlit interface for explicit review of the annotation queue."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agents.annotation_agent import AnnotationAgent
from agents.common import save_frame, utc_now_iso
from agents.contracts import TaskSpec
from agents.media_review import apply_media_review

PROJECT_ROOT = Path(__file__).resolve().parent
QUEUE_PATH = PROJECT_ROOT / "data/review/review_queue.csv"
CORRECTED_PATH = PROJECT_ROOT / "data/review/review_queue_corrected.csv"


def load_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    """Load the review queue while preserving stable string identifiers."""
    queue = pd.read_csv(path, dtype={"record_id": "string", "reviewed": "boolean"})
    if "human_label" not in queue.columns:
        queue["human_label"] = ""
    queue["human_label"] = queue["human_label"].fillna("").astype(str)
    if "reviewer" not in queue.columns:
        queue["reviewer"] = ""
    queue["reviewer"] = queue["reviewer"].fillna("").astype(str)
    if "reviewed" not in queue.columns:
        queue["reviewed"] = False
    queue["reviewed"] = queue["reviewed"].fillna(False).astype(bool)
    forbidden = {"label", "source_label", "final_label"} & set(queue.columns)
    if forbidden:
        queue = queue.drop(columns=sorted(forbidden))
    return queue


def save_text_review(
    queue: pd.DataFrame,
    edited: pd.DataFrame,
    output_path: Path,
    *,
    reviewer: str | None = None,
    config: Path | None = None,
) -> pd.DataFrame:
    """Persist only explicitly checked text reviews, including the edited checkbox."""
    if "reviewed" not in edited or not edited["reviewed"].map(
        lambda value: isinstance(value, bool) and value
    ).all():
        raise ValueError("Mark every row as reviewed before saving")
    if set(edited["record_id"].astype(str)) != set(queue["record_id"].astype(str)):
        raise ValueError("Every queued text record must be reviewed before saving")
    agent = AnnotationAgent(modality="text", config=config or PROJECT_ROOT / "config.yaml")
    corrected = agent.apply_human_review(
        queue, edited[["record_id", "human_label", "reviewer", "reviewed"]], reviewer=reviewer
    )
    # The original queue contains False. Use the submitted flags, not the stale queue.
    checked = edited.set_index("record_id")["reviewed"]
    corrected["reviewed"] = corrected["record_id"].map(checked).fillna(False).astype(bool)
    save_frame(corrected, output_path)
    return corrected


def save_media_decision(
    queue: pd.DataFrame,
    decision: dict[str, Any],
    task: TaskSpec,
    output_path: Path,
) -> pd.DataFrame:
    """Validate and atomically upsert one actual human decision without accepting others."""
    rows: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id")
            if record_id in rows:
                raise ValueError("Saved decisions contain duplicate record_id")
            rows[record_id] = row
    if decision.get("reviewed") is not True:
        raise ValueError("Explicit human confirmation is required before saving")
    record_id = decision.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Decision requires record_id")
    rows[record_id] = decision
    reviewed = apply_media_review(queue, pd.DataFrame(list(rows.values())), task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve intermediate user decisions even when changed before a pipeline resume.
    events_path = output_path.with_name("review_events.jsonl")
    events = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    event_line = json.dumps(decision, ensure_ascii=False, allow_nan=False)
    previous_event = events.rstrip().splitlines()[-1] if events.strip() else None
    if previous_event != event_line:
        event_descriptor, event_temporary = tempfile.mkstemp(prefix=".events-", dir=output_path.parent)
        try:
            with os.fdopen(event_descriptor, "w", encoding="utf-8") as stream:
                stream.write(events.rstrip() + "\n" if events.strip() else "")
                stream.write(event_line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(event_temporary, events_path)
        finally:
            if os.path.exists(event_temporary):
                os.unlink(event_temporary)
    descriptor, temporary = tempfile.mkstemp(prefix=".review-", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for key in sorted(rows):
                stream.write(json.dumps(rows[key], ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return reviewed


def _text_review() -> None:
    st.title("Human review: cross-domain sentiment")
    st.write(
        "Read each full review and correct its label when needed. Leaving the label blank "
        "explicitly confirms the model's auto-label. Your reviewer name is required."
    )

    if not QUEUE_PATH.exists():
        st.warning(f"Review queue not found at `{QUEUE_PATH}`. Run the annotation stage first.")
        return

    queue = load_queue()
    default_reviewer = st.sidebar.text_input("Reviewer name")
    if default_reviewer.strip():
        empty_reviewer = queue["reviewer"].str.strip().eq("")
        queue.loc[empty_reviewer, "reviewer"] = default_reviewer.strip()

    st.caption(
        f"{len(queue)} rows · {int(queue['auto_label'].eq('positive').sum())} positive "
        f"auto-labels · {int(queue['auto_label'].eq('negative').sum())} negative auto-labels"
    )
    editable = {"human_label", "reviewer", "reviewed"}
    disabled = [column for column in queue.columns if column not in editable]
    edited = st.data_editor(
        queue,
        hide_index=True,
        use_container_width=True,
        disabled=disabled,
        column_config={
            "human_label": st.column_config.SelectboxColumn(
                "human_label",
                help="Blank confirms auto_label; otherwise choose the corrected label.",
                options=["", "negative", "positive"],
                required=False,
            ),
            "reviewer": st.column_config.TextColumn(
                "reviewer", help="Name of the person who reviewed this row", required=True
            ),
            "reviewed": st.column_config.CheckboxColumn(
                "reviewed",
                help="Check only after reading the full text and confirming/correcting the label.",
                required=True,
            ),
            "confidence": st.column_config.NumberColumn("confidence", format="%.3f"),
            "text": st.column_config.TextColumn("text", width="large"),
        },
        key="review_editor",
    )

    if st.button("Save verified corrections", type="primary"):
        try:
            corrected = save_text_review(
                queue, edited, CORRECTED_PATH, reviewer=default_reviewer or None
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            changed = int(corrected["review_changed"].sum())
            st.success(
                f"Saved {len(corrected)} verified rows to `{CORRECTED_PATH}`; "
                f"{changed} auto-labels were changed."
            )


def _optional(row: pd.Series, field: str, fallback: Any) -> Any:
    value = row.get(field)
    return fallback if value is None or pd.isna(value) else value


def _media_review() -> None:
    st.title("Проверка разметки фото и видео")
    st.write("Просмотрите материал и сохраните собственное решение для выбранного примера.")
    state_paths = sorted((PROJECT_ROOT / "data/runs").glob("*/*/pipeline_state.json"))
    if not state_paths:
        st.info("Нет подготовленных запусков. Сначала запустите пайплайн с конфигурацией фото или видео.")
        return
    state_path = st.sidebar.selectbox(
        "Запуск", state_paths, format_func=lambda path: f"{path.parent.parent.name} / {path.parent.name}"
    )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        task = TaskSpec.from_config(state["task"])
        if state.get("status") in {"completed", "completed_without_hitl"}:
            st.success("Этот запуск завершён. Его решения сохранены и доступны только для чтения.")
            st.info("Для новой разметки используйте новый --run-id; завершённый запуск не принимает исправления.")
            return
        batch_id = state.get("active_batch")
        if not batch_id:
            st.info(f"Текущий статус: {state.get('status', 'unknown')}. Очереди проверки пока нет.")
            return
        batch_dir = state_path.parent / "batches" / str(batch_id)
        queue_path = batch_dir / "review_queue.jsonl"
        if not queue_path.exists():
            st.info(
                f"Текущий статус: {state.get('status', 'unknown')}. "
                "Сначала дождитесь подготовки очереди ручной проверки."
            )
            return
        queue = pd.read_json(queue_path, lines=True, dtype={"record_id": str})
        decisions_path = batch_dir / "review_decisions.jsonl"
        current = (
            apply_media_review(queue, decisions_path, task) if decisions_path.exists() else queue
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Не удалось открыть очередь: {error}")
        return
    if current.empty:
        st.info("Очередь проверки пуста.")
        return
    with st.expander("Спецификация задачи"):
        st.write(task.description or task.task_id)
        for definition in task.classes:
            st.write(f"**{definition.name}** — {definition.definition}")
            for example in definition.examples:
                st.write(f"• {example}")
        for rule in task.boundary_rules:
            st.write(f"• {rule}")
    statuses = current.get("review_status", pd.Series("pending", index=current.index))
    done = int(statuses.isin(["accepted", "edited", "rejected"]).sum())
    st.caption(f"Решения сохранены: {done} / {len(current)} · Пакет: {batch_id}")
    if done == len(current):
        st.success("Все примеры проверены. Продолжите пайплайн, чтобы импортировать решения.")
    reviewer = st.sidebar.text_input("Ваше имя", key="media_reviewer")
    selected_id = st.selectbox("Пример", current["record_id"].astype(str).tolist())
    row = current.loc[current["record_id"].astype(str).eq(selected_id)].iloc[0]
    start = float(row.get("start_sec", 0.0))
    end = float(row.get("end_sec", start))
    frames = row["frames"]
    if isinstance(frames, str):
        frames = json.loads(frames)
    try:
        if task.modality == "video":
            st.video(str(row["media_path"]), start_time=start)
            st.caption(f"Интервал задания: {start:.3f}–{end:.3f} с. Плеер показывает исходное видео.")
            with st.expander("Кадры, предоставленные агенту"):
                columns = st.columns(min(4, len(frames)))
                for position, frame in enumerate(frames):
                    columns[position % len(columns)].image(
                        frame["path"], caption=f"{frame['timestamp_sec']:.3f} с", use_container_width=True
                    )
        else:
            st.image(str(row.get("media_path") or frames[0]["path"]), use_container_width=True)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Не удалось показать материал: {error}")
        return
    auto_label = _optional(row, "auto_label", None)
    st.write(f"Предложение агента: **{auto_label or 'воздержался от ответа'}**")
    st.write(str(_optional(row, "annotation_reason", "")))
    st.caption("Уверенность агента не является откалиброванной вероятностью.")
    action_names = {
        "accepted": "Подтвердить", "edited": "Исправить",
        "rejected": "Отклонить пример", "uncertain": "Недостаточно уверенности",
    }
    # Per-record widget keys prevent an earlier item's class/time from carrying over.
    widget_id = f"{state_path}:{batch_id}:{selected_id}"
    status = st.selectbox(
        "Решение", list(action_names), index=0 if auto_label is not None else 3,
        format_func=lambda value: action_names[value], key=f"action:{widget_id}",
    )
    default_label = (
        _optional(row, "human_label", auto_label) if status == "edited" else auto_label
    )
    options = ["", *task.labels]
    label = st.selectbox(
        "Класс", options, index=options.index(default_label) if default_label in options else 0,
        disabled=status != "edited", key=f"label:{widget_id}:{status}",
    )
    human_start, human_end = start, end
    if task.modality == "video":
        auto_start = float(_optional(row, "auto_start_sec", start))
        auto_end = float(_optional(row, "auto_end_sec", end))
        default_start = float(_optional(row, "human_start_sec", auto_start)) if status == "edited" else auto_start
        default_end = float(_optional(row, "human_end_sec", auto_end)) if status == "edited" else auto_end
        left, right = st.columns(2)
        human_start = left.number_input(
            "Начало, секунды", min_value=start, max_value=end, value=default_start,
            step=0.1, format="%.3f", disabled=status != "edited", key=f"start:{widget_id}:{status}",
        )
        human_end = right.number_input(
            "Конец, секунды", min_value=start, max_value=end, value=default_end,
            step=0.1, format="%.3f", disabled=status != "edited", key=f"end:{widget_id}:{status}",
        )
    note = st.text_area("Комментарий", key=f"note:{widget_id}")
    confirmed = st.checkbox("Я просмотрел материал и подтверждаю это решение", key=f"confirm:{widget_id}")
    if st.button("Сохранить решение", type="primary"):
        decision = {
            "record_id": selected_id, "request_id": row.get("request_id"),
            "review_status": status, "reviewed": confirmed, "reviewer": reviewer.strip(),
            "reviewed_at": utc_now_iso(), "review_note": note,
        }
        if status == "edited":
            decision.update(
                human_label=label, human_start_sec=human_start, human_end_sec=human_end
            )
        try:
            save_media_decision(queue, decision, task, decisions_path)
        except (OSError, ValueError, TypeError) as error:
            st.error(str(error))
        else:
            st.success("Решение сохранено. Можно перейти к следующему примеру.")
    st.caption("После проверки продолжите запуск. Файл содержит только сохранённые вами решения.")
    st.code(
        "python run_pipeline.py --config " + shlex.quote(state["config_path"])
        + " --run-id " + shlex.quote(state_path.parent.name)
        + " --reviews " + shlex.quote(str(decisions_path)),
        language="bash",
    )


def main() -> None:
    st.set_page_config(page_title="Human annotation review", layout="wide")
    mode = st.sidebar.radio("Тип данных", ["Текст", "Фото и видео"])
    if mode == "Текст":
        _text_review()
    else:
        _media_review()


if __name__ == "__main__":
    main()
