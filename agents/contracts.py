"""Small, model-independent contracts for configurable visual classification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    definition: str = ""
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "definition": self.definition, "examples": list(self.examples)}


@dataclass(frozen=True)
class TaskSpec:
    """One image/video classification task; uncertainty is a state, not a class.

    A minimal configuration may provide only ``project.modality`` and
    ``project.labels``. Detailed definitions and examples improve annotation but
    are deliberately not tied to a particular course or domain.
    """

    task_id: str
    version: str
    modality: str
    labels: tuple[str, ...]
    description: str = ""
    classes: tuple[ClassDefinition, ...] = ()
    boundary_rules: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TaskSpec:
        project = config.get("project", {}) or {}
        task = config.get("task", config) or {}
        if not isinstance(project, Mapping) or not isinstance(task, Mapping):
            raise TypeError("project and task must be mappings")
        modality = str(project.get("modality", task.get("modality", ""))).strip().lower()
        if modality not in {"image", "video"}:
            raise ValueError("Visual tasks require project.modality: image or video")
        task_id = str(task.get("id", task.get("task_id", f"{modality}_classification"))).strip()
        version = str(task.get("version", "1")).strip()
        if not task_id or not version or task_id in {"None", "null"} or version == "None":
            raise ValueError("task.id and task.version must be nonempty")

        raw_classes = task.get("classes")
        labels_setting = project.get("labels", task.get("labels"))
        classes: list[ClassDefinition] = []
        if raw_classes is not None:
            if not isinstance(raw_classes, (list, tuple)):
                raise ValueError("task.classes must be a list")
            for entry in raw_classes:
                if isinstance(entry, str):
                    entry = {"name": entry}
                if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                    raise TypeError("Each class requires a string name")
                name = entry["name"].strip()
                definition = entry.get("definition", "")
                examples = entry.get("examples", []) or []
                if not isinstance(definition, str) or not isinstance(examples, (list, tuple)):
                    raise TypeError("Class definition must be text and examples must be a list")
                if any(not isinstance(example, str) for example in examples):
                    raise ValueError("Class examples must be strings")
                classes.append(ClassDefinition(name, definition.strip(), tuple(examples)))
            labels = tuple(entry.name for entry in classes)
            if labels_setting is not None:
                configured_labels = cls._labels(labels_setting)
                if configured_labels != labels:
                    raise ValueError("project.labels must match task.classes names and order")
        else:
            labels = cls._labels(labels_setting)
            classes = [ClassDefinition(name) for name in labels]
        cls._labels(labels)
        rules = task.get("boundary_rules", []) or []
        if isinstance(rules, str):
            rules = [rules]
        if not isinstance(rules, (list, tuple)) or any(not isinstance(rule, str) for rule in rules):
            raise ValueError("task.boundary_rules must be text or a list of strings")
        description = task.get("description", "")
        if not isinstance(description, str):
            raise TypeError("task.description must be text")
        return cls(task_id, version, modality, labels, description, tuple(classes), tuple(rules))

    @staticmethod
    def _labels(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise ValueError("At least two string class labels are required")
        labels = tuple(item.strip() for item in value)
        if len(labels) < 2 or any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("At least two distinct, nonempty class labels are required")
        return labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "version": self.version,
            "modality": self.modality,
            "description": self.description,
            "labels": list(self.labels),
            "classes": [definition.to_dict() for definition in self.classes],
            "boundary_rules": list(self.boundary_rules),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["ClassDefinition", "TaskSpec"]
