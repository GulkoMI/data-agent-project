from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agents.annotation_agent import AnnotationAgent


class FakeBackend:
    name = "fake-sentiment"

    def predict(self, texts: list[str], **_: object) -> list[dict[str, object]]:
        predictions = {
            "p1": ("POSITIVE", 0.97),
            "p2": ("POSITIVE", 0.92),
            "p3": ("NEGATIVE", 0.91),  # deliberate disagreement
            "n1": ("NEGATIVE", 0.96),
            "n2": ("NEGATIVE", 0.58),  # mandatory low confidence
            "n3": ("NEGATIVE", 0.89),
        }
        return [{"label": predictions[text][0], "score": predictions[text][1]} for text in texts]


@pytest.fixture
def source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["p1", "p2", "p3", "n1", "n2", "n3"],
            "text": ["p1", "p2", "p3", "n1", "n2", "n3"],
            "label": ["positive", "positive", "positive", "negative", "negative", "negative"],
            "source": ["fixture"] * 6,
            "source_id": list(range(6)),
        }
    )


@pytest.fixture
def agent(tmp_path: Path) -> AnnotationAgent:
    return AnnotationAgent(
        config={
            "annotation": {
                "confidence_threshold": 0.80,
                "review_target": 4,
                "model": "fake-model",
            }
        },
        backend=FakeBackend(),
    )


def test_auto_label_preserves_source_and_builds_deterministic_queue(
    agent: AnnotationAgent, source_df: pd.DataFrame
) -> None:
    labeled = agent.auto_label(source_df)

    assert labeled["label"].tolist() == source_df["label"].tolist()
    assert labeled["source_label"].tolist() == source_df["label"].tolist()
    assert {
        "auto_label",
        "confidence",
        "needs_review",
        "review_reason",
        "human_label",
        "final_label",
    }.issubset(labeled.columns)
    assert labeled.loc[labeled.record_id.eq("p3"), "review_reason"].item() in {
        "",
        "uncertainty_sample",
    }
    assert labeled.loc[labeled.record_id.eq("n2"), "review_reason"].item() == "low_confidence"

    queue = agent.flag_for_review(labeled)
    assert len(queue) == 4
    assert {"p3", "n2"}.issubset(set(queue["record_id"]))
    assert queue["confidence"].is_monotonic_increasing
    assert (queue["annotation_backend"] == "fake-sentiment").all()


def test_spec_quality_and_label_studio_export(
    agent: AnnotationAgent, source_df: pd.DataFrame, tmp_path: Path
) -> None:
    labeled = agent.auto_label(source_df)
    labeled.loc[labeled.record_id.eq("p3"), "human_label"] = "positive"

    spec_path = agent.generate_spec(labeled, output_path=tmp_path / "annotation_spec.md")
    spec = spec_path.read_text(encoding="utf-8")
    assert "## Class definitions" in spec
    assert "## Boundary cases" in spec
    assert spec.splitlines().count("### `negative`") == 1
    assert spec.splitlines().count("### `positive`") == 1
    assert sum(line.startswith("- `n") for line in spec.splitlines()) >= 3
    assert sum(line.startswith("- `p") for line in spec.splitlines()) >= 3

    metrics = agent.check_quality(labeled)
    assert metrics["against_source_label"]["n_compared"] == 6
    assert metrics["against_source_label"]["agreement"] == pytest.approx(5 / 6)
    assert metrics["against_human_label"]["n_compared"] == 1
    assert metrics["against_human_label"]["agreement"] == 0.0
    assert metrics["label_dist"] == {"negative": 4, "positive": 2}
    assert metrics["confidence_mean"] == pytest.approx(
        (0.97 + 0.92 + 0.91 + 0.96 + 0.58 + 0.89) / 6
    )

    export_path = agent.export_to_labelstudio(
        agent.flag_for_review(labeled), tmp_path / "labelstudio_import.json"
    )
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    assert isinstance(tasks, list)
    assert len(tasks) == 4
    assert set(tasks[0]) == {"data", "predictions"}
    assert all("source_label" not in task["data"] for task in tasks)
    result = tasks[0]["predictions"][0]["result"][0]
    assert result["from_name"] == "sentiment"
    assert result["to_name"] == "review_text"
    assert result["value"]["choices"][0] in {"negative", "positive"}

    xml_path = Path(__file__).parents[1] / "label_studio_config.xml"
    root = ET.parse(xml_path).getroot()
    assert root.tag == "View"
    choices = root.find("Choices")
    assert choices is not None
    assert choices.attrib["name"] == "sentiment"
    assert choices.attrib["toName"] == "review_text"


def test_apply_human_review_validates_and_merges(
    agent: AnnotationAgent, source_df: pd.DataFrame
) -> None:
    labeled = agent.auto_label(source_df)
    corrections = pd.DataFrame(
        {
            "record_id": ["p3", "n2"],
            "corrected_label": ["positive", ""],
        }
    )
    reviewed = agent.apply_human_review(
        labeled,
        corrections,
        reviewer="Student Reviewer",
        reviewed_at="2026-03-21T12:00:00+00:00",
    )

    p3 = reviewed.loc[reviewed.record_id.eq("p3")].iloc[0]
    n2 = reviewed.loc[reviewed.record_id.eq("n2")].iloc[0]
    assert p3["human_label"] == "positive"
    assert p3["final_label"] == "positive"
    assert bool(p3["review_changed"])
    assert n2["human_label"] == n2["auto_label"]
    assert not bool(n2["review_changed"])
    assert p3["reviewer"] == "Student Reviewer"
    assert p3["reviewed_at"] == "2026-03-21T12:00:00+00:00"
    assert not bool(p3["needs_review"])

    with pytest.raises(ValueError, match="reviewer"):
        agent.apply_human_review(labeled, corrections)
    with pytest.raises(ValueError, match="duplicate"):
        agent.apply_human_review(
            labeled,
            pd.DataFrame({"record_id": ["p3", "p3"], "human_label": ["positive", "negative"]}),
            reviewer="Reviewer",
        )
    with pytest.raises(ValueError, match="unknown"):
        agent.apply_human_review(
            labeled,
            pd.DataFrame({"record_id": ["missing"], "human_label": ["positive"]}),
            reviewer="Reviewer",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        agent.apply_human_review(
            labeled,
            pd.DataFrame({"record_id": ["p3"], "human_label": ["neutral"]}),
            reviewer="Reviewer",
        )


def test_stable_ids_and_lexicon_fallback_do_not_require_torch() -> None:
    frame = pd.DataFrame(
        {
            "text": ["A great and wonderful game", "An awful broken product"],
            "label": ["positive", "negative"],
            "source": ["fixture", "fixture"],
            "source_id": ["one", "two"],
        }
    )
    lexicon_agent = AnnotationAgent(
        config={"backend": "lexicon", "confidence_threshold": 0.8, "review_target": 2}
    )
    first = lexicon_agent.auto_label(frame)
    second = lexicon_agent.auto_label(frame)

    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert first["auto_label"].tolist() == ["positive", "negative"]
    assert (first["annotation_backend"] == "lexicon").all()


def test_missing_or_empty_local_model_uses_logical_model_id(tmp_path: Path) -> None:
    logical_model = "owner/sentiment-model"
    missing = tmp_path / "missing-model"
    agent = AnnotationAgent(
        config={"model": logical_model, "local_model_path": str(missing)}
    )

    assert agent._resolve_transformer_model() == (logical_model, False)

    empty = tmp_path / "empty-model"
    empty.mkdir()
    agent = AnnotationAgent(
        config={"model": logical_model, "local_model_path": str(empty)}
    )

    assert agent._resolve_transformer_model() == (logical_model, False)


def test_partial_local_model_fails_without_fallback_or_transformer_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_model = tmp_path / "partial-model"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    agent = AnnotationAgent(
        config={
            "backend": "transformers",
            "model": "owner/sentiment-model",
            "local_model_path": str(local_model),
            "allow_lexicon_fallback": True,
        }
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match="incomplete.*model.safetensors"):
        agent._predict(["A good product"])


def test_complete_local_model_is_loaded_offline_and_keeps_logical_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_model = tmp_path / "complete-model"
    local_model.mkdir()
    for filename in (
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (local_model / filename).write_bytes(b"fixture")

    calls: dict[str, object] = {}
    tokenizer_object = object()
    model_object = object()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> object:
            calls["tokenizer"] = (source, kwargs)
            return tokenizer_object

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> object:
            calls["model"] = (source, kwargs)
            return model_object

    class FakePipeline:
        def __call__(self, texts: list[str], **kwargs: object) -> list[dict[str, object]]:
            calls["inference"] = (texts, kwargs)
            return [{"label": "POSITIVE", "score": 0.99} for _ in texts]

    def fake_pipeline(task: str, **kwargs: object) -> FakePipeline:
        calls["pipeline"] = (task, kwargs)
        return FakePipeline()

    unavailable = SimpleNamespace(is_available=lambda: False)
    fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=unavailable), cuda=unavailable)
    fake_transformers = SimpleNamespace(
        AutoModelForSequenceClassification=FakeModel,
        AutoTokenizer=FakeTokenizer,
        pipeline=fake_pipeline,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    logical_model = "owner/sentiment-model"
    agent = AnnotationAgent(
        config={
            "backend": "transformers",
            "model": logical_model,
            "local_model_path": str(local_model),
            "device": "cpu",
            "allow_lexicon_fallback": False,
        }
    )
    predictions, backend = agent._predict(["A good product"])

    assert predictions == [{"label": "POSITIVE", "score": 0.99}]
    assert backend == f"transformers:{logical_model}"
    expected_source = str(local_model.resolve())
    assert calls["tokenizer"] == (expected_source, {"local_files_only": True})
    assert calls["model"] == (expected_source, {"local_files_only": True})
    task, pipeline_kwargs = calls["pipeline"]
    assert task == "sentiment-analysis"
    assert pipeline_kwargs["tokenizer"] is tokenizer_object
    assert pipeline_kwargs["model"] is model_object
