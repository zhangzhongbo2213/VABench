"""Validated scope for the current task-native interaction-frame dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


DEFAULT_TASK_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "open_vocab_grasp_tasks_22.yaml"
)
ALLOWED_SPLITS = {"train", "generalization"}
ALLOWED_FRAME_TYPES = {"grasp", "place", "handover", "push", "strike"}


@dataclass(frozen=True)
class TaskFrameSpec:
    task: str
    split: str
    family: str
    frame_types: tuple[str, ...]
    base_task: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskFrameSpec":
        task = _nonempty(value.get("task"), "task")
        split = _nonempty(value.get("split"), f"{task}.split")
        family = _nonempty(value.get("family"), f"{task}.family")
        raw_frame_types = value.get("frame_types")
        if not isinstance(raw_frame_types, list) or not raw_frame_types:
            raise ValueError(f"{task}.frame_types must be a non-empty list")
        frame_types = tuple(_nonempty(item, f"{task}.frame_types") for item in raw_frame_types)
        base = value.get("base_task")
        return cls(
            task=task,
            split=split,
            family=family,
            frame_types=frame_types,
            base_task=_nonempty(base, f"{task}.base_task") if base is not None else None,
        )

    @property
    def contributes_grasp_labels(self) -> bool:
        return "grasp" in self.frame_types


@dataclass(frozen=True)
class TaskFrameManifest:
    manifest_id: str
    tasks: tuple[TaskFrameSpec, ...]
    label_policy: Mapping[str, Any]
    source: Path

    @property
    def by_name(self) -> dict[str, TaskFrameSpec]:
        return {task.task: task for task in self.tasks}

    def tasks_for_frame(
        self, frame_type: str, *, splits: Iterable[str] | None = None
    ) -> tuple[TaskFrameSpec, ...]:
        if frame_type not in ALLOWED_FRAME_TYPES:
            raise ValueError(f"unsupported frame type {frame_type!r}")
        allowed_splits = set(splits) if splits is not None else ALLOWED_SPLITS
        return tuple(
            task
            for task in self.tasks
            if frame_type in task.frame_types and task.split in allowed_splits
        )


def load_task_frame_manifest(
    path: str | Path = DEFAULT_TASK_MANIFEST,
) -> TaskFrameManifest:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("task manifest must contain a mapping")
    if value.get("schema_version") != "spatial.task_manifest.v1":
        raise ValueError("unsupported task manifest schema_version")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("task manifest tasks must be a list")
    tasks = tuple(TaskFrameSpec.from_mapping(item) for item in raw_tasks)
    expected_count = int(value.get("task_count", -1))
    if expected_count != len(tasks):
        raise ValueError(
            f"task_count={expected_count} does not match {len(tasks)} task rows"
        )
    names = [task.task for task in tasks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("duplicate tasks in manifest: " + ", ".join(duplicates))
    by_name = {task.task: task for task in tasks}
    for task in tasks:
        if task.split not in ALLOWED_SPLITS:
            raise ValueError(f"{task.task} has unsupported split {task.split!r}")
        unknown_frames = sorted(set(task.frame_types) - ALLOWED_FRAME_TYPES)
        if unknown_frames:
            raise ValueError(
                f"{task.task} has unsupported frame types: " + ", ".join(unknown_frames)
            )
        if task.split == "generalization":
            if task.base_task is None:
                raise ValueError(f"{task.task} must declare base_task")
            if task.base_task not in by_name:
                raise ValueError(
                    f"{task.task} references missing base_task {task.base_task!r}"
                )
            if by_name[task.base_task].split != "train":
                raise ValueError(f"{task.task} base_task must belong to train split")
    label_policy = value.get("label_policy")
    if not isinstance(label_policy, Mapping):
        raise ValueError("task manifest label_policy must be a mapping")
    if label_policy.get("point_annotation") != "automatic":
        raise ValueError("current manifest requires automatic point annotation")
    if label_policy.get("extra_out_of_distribution_hard_samples") is not False:
        raise ValueError("current manifest forbids extra out-of-distribution hard samples")
    if label_policy.get("generalization_tasks_are_training_inputs") is not False:
        raise ValueError("generalization tasks must not be training inputs")
    return TaskFrameManifest(
        manifest_id=_nonempty(value.get("manifest_id"), "manifest_id"),
        tasks=tasks,
        label_policy=dict(label_policy),
        source=source,
    )


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
