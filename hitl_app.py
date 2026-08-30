"""Streamlit interface for explicit review of the annotation queue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from agents.annotation_agent import AnnotationAgent
from agents.common import save_frame

PROJECT_ROOT = Path(__file__).resolve().parent
QUEUE_PATH = PROJECT_ROOT / "data/review/review_queue.csv"
CORRECTED_PATH = PROJECT_ROOT / "data/review/review_queue_corrected.csv"


def load_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    """Load the review queue while preserving stable string identifiers."""
    queue = pd.read_csv(path, dtype={"record_id": "string"})
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


def main() -> None:
    st.set_page_config(page_title="Sentiment HITL review", layout="wide")
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
            if not edited["reviewed"].fillna(False).all():
                raise ValueError("Mark every row as reviewed before saving")
            agent = AnnotationAgent(
                modality="text",
                config=PROJECT_ROOT / "config.yaml",
                backend=lambda texts: [],  # Inference is not used during review merge.
            )
            corrected = agent.apply_human_review(
                queue,
                edited[["record_id", "human_label", "reviewer"]],
                reviewer=default_reviewer or None,
            )
            save_frame(corrected, CORRECTED_PATH)
        except ValueError as exc:
            st.error(str(exc))
        else:
            changed = int(corrected["review_changed"].sum())
            st.success(
                f"Saved {len(corrected)} verified rows to `{CORRECTED_PATH}`; "
                f"{changed} auto-labels were changed."
            )


if __name__ == "__main__":
    main()
