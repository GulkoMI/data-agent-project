"""Public agent API with lazy imports to keep lightweight modules lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ActiveLearningAgent": "agents.al_agent",
    "AnnotationAgent": "agents.annotation_agent",
    "DataCollectionAgent": "agents.data_collection_agent",
    "DataQualityAgent": "agents.data_quality_agent",
    "TrainAgent": "agents.train_agent",
}

__all__ = [
    "ActiveLearningAgent",
    "AnnotationAgent",
    "DataCollectionAgent",
    "DataQualityAgent",
    "TrainAgent",
]


def __getattr__(name: str) -> Any:
    """Load an agent only when that public attribute is requested."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
