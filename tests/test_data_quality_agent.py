from __future__ import annotations

import json

import pandas as pd
import pandas.testing as pdt
import pytest

from agents.data_quality_agent import DataQualityAgent


def _problem_frame() -> pd.DataFrame:
    regular_positive = "A solid purchase with reliable performance."
    regular_negative = "A poor purchase with unreliable performance."
    rows = [
        {
            "record_id": "p1",
            "text": regular_positive,
            "label": "positive",
            "source_label": "positive",
            "source": "amazon",
        },
        {
            "record_id": "n1",
            "text": regular_negative,
            "label": "negative",
            "source_label": "negative",
            "source": "amazon",
        },
        {
            "record_id": "p2",
            "text": "Excellent product and a delight to use every day.",
            "label": "positive",
            "source_label": "positive",
            "source": "steam",
        },
        {
            "record_id": "n2",
            "text": "Terrible game and frustrating to use every day.",
            "label": "negative",
            "source_label": "negative",
            "source": "steam",
        },
        {
            "record_id": "p3",
            "text": "Fantastic " * 120,
            "label": "positive",
            "source_label": "positive",
            "source": "amazon",
        },
        {
            "record_id": "n3",
            "text": "Horrible " * 120,
            "label": "negative",
            "source_label": "negative",
            "source": "steam",
        },
        {
            "record_id": "blank-positive",
            "text": "   ",
            "label": "positive",
            "source_label": "positive",
            "source": "amazon",
        },
        {
            "record_id": "blank-negative",
            "text": None,
            "label": "negative",
            "source_label": "negative",
            "source": "steam",
        },
    ]
    for number in range(4, 8):
        rows.extend(
            [
                {
                    "record_id": f"p{number}",
                    "text": f"Useful product with dependable results number {number}.",
                    "label": "positive",
                    "source_label": "positive",
                    "source": "amazon",
                },
                {
                    "record_id": f"n{number}",
                    "text": f"Useless product with disappointing results number {number}.",
                    "label": "negative",
                    "source_label": "negative",
                    "source": "steam",
                },
            ]
        )
    # Exact duplicate pairs are balanced so removing them preserves target shares.
    rows.extend([rows[0].copy(), rows[1].copy()])
    return pd.DataFrame(rows)


def test_detects_all_required_problem_types() -> None:
    frame = _problem_frame()
    agent = DataQualityAgent()

    report = agent.detect_issues(frame)

    assert report["missing"]["text"] == 1
    assert report["empty_text"] == 1
    assert report["duplicates"] == 2
    assert report["text_duplicates"] == 2
    assert report["outliers"]["iqr"]["count"] >= 2
    assert report["outliers"]["zscore"]["threshold"] == 3.0
    assert report["imbalance"]["counts"] == {"negative": 9, "positive": 9}
    assert report["imbalance"]["minority_to_majority_ratio"] == 1.0
    # The two protected target columns are reported separately, not conflated.
    assert set(report["target_distributions"]) == {"label", "source_label"}


def test_conservative_fix_is_immutable_and_preserves_target_semantics() -> None:
    frame = _problem_frame()
    snapshot = frame.copy(deep=True)
    original_longest = int(frame["text"].fillna("").str.len().max())
    agent = DataQualityAgent()

    cleaned = agent.fix(frame, "conservative")

    pdt.assert_frame_equal(frame, snapshot)
    assert cleaned["text"].notna().all()
    assert cleaned["text"].str.strip().ne("").all()
    assert not cleaned.duplicated().any()
    assert int(cleaned["text"].str.len().max()) < original_longest
    assert (cleaned["label"] == cleaned["source_label"]).all()
    assert set(cleaned["label"]) == {"positive", "negative"}
    # Balanced bad records were deliberately supplied in pairs; no target skew is introduced.
    assert cleaned["label"].value_counts(normalize=True).to_dict() == {
        "positive": 0.5,
        "negative": 0.5,
    }


def test_strict_and_custom_strategies() -> None:
    frame = _problem_frame()
    agent = DataQualityAgent()

    strict = agent.fix(frame, "strict")
    custom = agent.fix(
        frame,
        {
            "name": "keep_lengths",
            "missing": "drop_required",
            "duplicates": "drop",
            "outliers": "keep",
        },
    )

    assert len(strict) < len(custom)
    assert strict["text"].str.len().max() < custom["text"].str.len().max()
    assert custom.attrs["quality_actions"]["strategy"] == "keep_lengths"
    with pytest.raises(ValueError, match="Unknown quality strategy"):
        agent.fix(frame, "does-not-exist")


def test_compare_and_evaluate_strategies() -> None:
    frame = _problem_frame()
    agent = DataQualityAgent()
    results = agent.evaluate_strategies(frame)

    assert set(results) == {"conservative", "strict"}
    comparison = results["conservative"]["comparison"].set_index("metric")
    assert comparison.loc["empty_text", "after"] == 0
    assert comparison.loc["exact_duplicates", "after"] == 0
    assert comparison.loc["target_share:negative", "change"] == pytest.approx(0.0)
    assert comparison.loc["target_share:positive", "change"] == pytest.approx(0.0)
    assert isinstance(results["strict"]["dataframe"], pd.DataFrame)


def test_report_writes_json_markdown_table_and_every_plot(tmp_path) -> None:
    frame = _problem_frame()
    agent = DataQualityAgent(report_dir=tmp_path)

    artifacts = agent.generate_report(frame, strategy="conservative")

    expected = {
        "json",
        "markdown",
        "comparison",
        "plot_missing_values",
        "plot_empty_text",
        "plot_duplicates",
        "plot_outliers",
        "plot_class_imbalance",
    }
    assert set(artifacts) == expected
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.values())

    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    markdown = artifacts["markdown"].read_text(encoding="utf-8")
    assert payload["strategy"] == "conservative"
    assert payload["target_shift"]["total_variation"] == pytest.approx(0.0)
    assert "## Strategy rationale" in markdown
    assert "| metric | before | after | change | improved |" in markdown
    assert "labels themselves were never rewritten" in payload["rationale"]


def test_normalized_text_duplicates_with_different_ids_are_removed() -> None:
    frame = pd.DataFrame(
        [
            {"record_id": "a", "text": "Great   PRODUCT", "label": "positive"},
            {"record_id": "b", "text": " great product ", "label": "positive"},
            {"record_id": "c", "text": "bad product", "label": "negative"},
        ]
    )
    agent = DataQualityAgent()
    assert agent.detect_issues(frame)["text_duplicates"] == 1
    cleaned = agent.fix(
        frame,
        {"missing": "keep", "duplicates": "drop", "outliers": "keep"},
    )
    assert cleaned["record_id"].tolist() == ["a", "c"]
