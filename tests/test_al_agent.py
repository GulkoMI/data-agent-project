from __future__ import annotations

import pandas as pd

from agents.al_agent import ActiveLearningAgent


def _dataset(size: int = 240) -> pd.DataFrame:
    rows = []
    for index in range(size):
        positive = index % 2 == 0
        rows.append(
            {
                "record_id": f"row-{index:03d}",
                "text": (
                    f"excellent fun polished game product number {index}"
                    if positive
                    else f"terrible broken boring game product number {index}"
                ),
                "source_label": "positive" if positive else "negative",
                "final_label": "positive" if positive else "negative",
            }
        )
    return pd.DataFrame(rows)


def test_compare_strategies_uses_disjoint_identical_splits() -> None:
    agent = ActiveLearningAgent(config={"random_seed": 17, "tfidf": {"max_features": 500}})
    result = agent.compare_strategies(
        _dataset(),
        strategies=["entropy", "random"],
        initial_size=50,
        n_iterations=2,
        batch_size=20,
        test_size=0.2,
    )
    manifest = result["split_manifest"]
    initial = set(manifest["initial_record_ids"])
    pool = set(manifest["pool_record_ids"])
    test = set(manifest["test_record_ids"])
    assert not initial & pool
    assert not initial & test
    assert not pool & test
    assert result["histories"]["entropy"][0]["n_labeled"] == 50
    assert result["histories"]["entropy"][-1]["n_labeled"] == 90


def test_entropy_is_deterministic() -> None:
    data = _dataset()
    first = ActiveLearningAgent(random_seed=42).compare_strategies(
        data, strategies=["entropy"], initial_size=50, n_iterations=1, batch_size=20
    )
    second = ActiveLearningAgent(random_seed=42).compare_strategies(
        data, strategies=["entropy"], initial_size=50, n_iterations=1, batch_size=20
    )
    assert first["histories"] == second["histories"]
