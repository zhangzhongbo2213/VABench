"""Validated task-native grasp events for automatic frame extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .task_manifest import DEFAULT_TASK_MANIFEST, load_task_frame_manifest


DEFAULT_GRASP_EVENT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "open_vocab_grasp_events_14.yaml"
)
TARGET_KINDS = {"attribute", "collection_records"}
ARM_KINDS = {"fixed", "task_attribute", "object_x_sign"}
ARMS = {"left", "right"}
DIRECT_GRASP_POST_GOALS = {"lift", "lift_clear_of_support_cube"}


@dataclass(frozen=True)
class GraspEventSpec:
    task: str
    event_id: str
    target: Mapping[str, Any]
    arm: Mapping[str, Any]
    task_stage: str
    post_grasp_goal: str
    coordination_group: str | None = None


@dataclass(frozen=True)
class ResolvedGraspEvent:
    task: str
    event_id: str
    actor: Any
    actor_name: str
    arm: str
    task_stage: str
    post_grasp_goal: str
    coordination_group: str | None
    access: str = "oracle/training_only"


@dataclass(frozen=True)
class GraspEventManifest:
    manifest_id: str
    task_manifest_id: str
    events_by_task: Mapping[str, tuple[GraspEventSpec, ...]]
    label_policy: Mapping[str, Any]
    source: Path

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.events_by_task)

    @property
    def event_count(self) -> int:
        return sum(len(events) for events in self.events_by_task.values())


def load_grasp_event_manifest(
    path: str | Path = DEFAULT_GRASP_EVENT_MANIFEST,
    *,
    task_manifest_path: str | Path = DEFAULT_TASK_MANIFEST,
) -> GraspEventManifest:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("grasp event manifest must contain a mapping")
    if payload.get("schema_version") != "spatial.grasp_event_manifest.v1":
        raise ValueError("unsupported grasp event manifest schema_version")
    task_manifest = load_task_frame_manifest(task_manifest_path)
    if payload.get("task_manifest_id") != task_manifest.manifest_id:
        raise ValueError("grasp event manifest references a different task manifest")
    label_policy = payload.get("label_policy")
    if not isinstance(label_policy, Mapping):
        raise ValueError("grasp event label_policy must be a mapping")
    if label_policy.get("point_annotation") != "automatic_from_successful_trajectory":
        raise ValueError(
            "grasp event points must be labelled from successful trajectories"
        )
    if label_policy.get("manual_metric_point_labels") is not False:
        raise ValueError("manual metric point labels are forbidden")
    if label_policy.get("simulator_contacts_are_training_only") is not True:
        raise ValueError("simulator contacts must remain training-only")

    rows = payload.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("grasp event manifest tasks must be a list")
    events_by_task: dict[str, tuple[GraspEventSpec, ...]] = {}
    event_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("grasp event task rows must be mappings")
        task = _text(row.get("task"), "task")
        if task in events_by_task:
            raise ValueError(f"duplicate grasp event task {task!r}")
        raw_events = row.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError(f"{task}.events must be a non-empty list")
        events = tuple(_parse_event(task, value) for value in raw_events)
        local_ids = [event.event_id for event in events]
        duplicates = sorted(
            {
                event_id
                for event_id in local_ids
                if local_ids.count(event_id) > 1 or event_id in event_ids
            }
        )
        if duplicates:
            raise ValueError("duplicate grasp event IDs: " + ", ".join(duplicates))
        event_ids.update(event.event_id for event in events)
        events_by_task[task] = events

    expected_tasks = {
        spec.task for spec in task_manifest.tasks_for_frame("grasp", splits=["train"])
    }
    actual_tasks = set(events_by_task)
    if actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        extra = sorted(actual_tasks - expected_tasks)
        raise ValueError(
            "grasp event task coverage mismatch; missing="
            + ",".join(missing)
            + "; extra="
            + ",".join(extra)
        )
    return GraspEventManifest(
        manifest_id=_text(payload.get("manifest_id"), "manifest_id"),
        task_manifest_id=task_manifest.manifest_id,
        events_by_task=events_by_task,
        label_policy=dict(label_policy),
        source=source,
    )


def resolve_grasp_events(
    task_or_env: Any,
    task_name: str,
    *,
    manifest: GraspEventManifest | None = None,
) -> tuple[ResolvedGraspEvent, ...]:
    """Resolve event templates after reset without producing metric point labels."""

    resolved_manifest = manifest or load_grasp_event_manifest()
    try:
        specs = resolved_manifest.events_by_task[task_name]
    except KeyError as exc:
        raise ValueError(f"task {task_name!r} has no automatic grasp events") from exc
    task = getattr(task_or_env, "task", task_or_env)
    result: list[ResolvedGraspEvent] = []
    for spec in specs:
        actors = _resolve_actors(task, spec.target)
        for actor_index, actor in enumerate(actors):
            suffix = f"/{actor_index}" if len(actors) > 1 else ""
            result.append(
                ResolvedGraspEvent(
                    task=task_name,
                    event_id=spec.event_id + suffix,
                    actor=actor,
                    actor_name=_actor_name(actor),
                    arm=_resolve_arm(task, actor, spec.arm),
                    task_stage=spec.task_stage,
                    post_grasp_goal=spec.post_grasp_goal,
                    coordination_group=spec.coordination_group,
                )
            )
    return tuple(result)


def select_grasp_event(
    task_or_env: Any,
    task_name: str,
    *,
    task_stage: str | None = None,
    active_arm: str | None = None,
    manifest: GraspEventManifest | None = None,
) -> ResolvedGraspEvent:
    """Select one task-native grasp event without guessing among actors.

    ``task_stage`` and ``active_arm`` are inference-visible query fields.  The
    resolved simulator actor remains training-only and is used solely for
    automatic physical labels.
    """

    events = list(
        resolve_grasp_events(
            task_or_env,
            task_name,
            manifest=manifest,
        )
    )
    if task_stage is not None:
        events = [event for event in events if event.task_stage == task_stage]
    if active_arm is not None:
        if active_arm not in ARMS:
            raise ValueError(f"active arm {active_arm!r} is invalid")
        events = [event for event in events if event.arm == active_arm]
    if not events:
        raise ValueError(
            f"task {task_name!r} has no grasp event matching "
            f"stage={task_stage!r}, arm={active_arm!r}"
        )
    if len(events) != 1:
        descriptions = ", ".join(
            f"{event.event_id}:{event.actor_name}:{event.arm}" for event in events
        )
        raise ValueError(
            f"task {task_name!r} grasp event is ambiguous; provide task_stage "
            f"or active_arm ({descriptions})"
        )
    return events[0]


def grasp_probe_success_scope(event: ResolvedGraspEvent) -> str:
    """Return whether close-and-lift can evaluate the full task or one event."""

    return (
        "full_task"
        if event.post_grasp_goal in DIRECT_GRASP_POST_GOALS
        else "grasp_event"
    )


def _parse_event(task: str, value: Any) -> GraspEventSpec:
    if not isinstance(value, Mapping):
        raise ValueError(f"{task} event rows must be mappings")
    if "expert_contact_point_ids" in value:
        raise ValueError(
            f"{task} event must derive metric contacts from successful trajectories"
        )
    target = value.get("target")
    arm = value.get("arm")
    if not isinstance(target, Mapping) or target.get("kind") not in TARGET_KINDS:
        raise ValueError(f"{task} event target has unsupported kind")
    if not isinstance(arm, Mapping) or arm.get("kind") not in ARM_KINDS:
        raise ValueError(f"{task} event arm has unsupported kind")
    _text(target.get("attribute"), f"{task}.target.attribute")
    if target.get("kind") == "collection_records":
        _text(target.get("actor_key"), f"{task}.target.actor_key")
    arm_kind = arm.get("kind")
    if arm_kind == "fixed" and arm.get("value") not in ARMS:
        raise ValueError(f"{task} fixed arm must be left or right")
    if arm_kind == "task_attribute":
        _text(arm.get("attribute"), f"{task}.arm.attribute")
    if arm_kind == "object_x_sign" and (
        arm.get("negative") not in ARMS or arm.get("nonnegative") not in ARMS
    ):
        raise ValueError(f"{task} object_x_sign arms must be left or right")
    coordination = value.get("coordination_group")
    return GraspEventSpec(
        task=task,
        event_id=_text(value.get("event_id"), f"{task}.event_id"),
        target=dict(target),
        arm=dict(arm),
        task_stage=_text(value.get("task_stage"), f"{task}.task_stage"),
        post_grasp_goal=_text(value.get("post_grasp_goal"), f"{task}.post_grasp_goal"),
        coordination_group=(
            _text(coordination, f"{task}.coordination_group")
            if coordination is not None
            else None
        ),
    )


def _resolve_actors(task: Any, target: Mapping[str, Any]) -> tuple[Any, ...]:
    attribute = str(target["attribute"])
    try:
        value = getattr(task, attribute)
    except AttributeError as exc:
        raise ValueError(f"task has no target attribute {attribute!r}") from exc
    if target["kind"] == "attribute":
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"task target collection {attribute!r} must be a sequence")
    actor_key = str(target["actor_key"])
    actors = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping) or actor_key not in record:
            raise ValueError(
                f"target collection record {index} has no actor key {actor_key!r}"
            )
        actors.append(record[actor_key])
    if not actors:
        raise ValueError(f"task target collection {attribute!r} is empty")
    return tuple(actors)


def _resolve_arm(task: Any, actor: Any, arm: Mapping[str, Any]) -> str:
    kind = arm["kind"]
    if kind == "fixed":
        return str(arm["value"])
    if kind == "task_attribute":
        try:
            value = str(getattr(task, str(arm["attribute"])))
        except AttributeError as exc:
            raise ValueError("task is missing dynamic arm attribute") from exc
        if value not in ARMS:
            raise ValueError(f"resolved dynamic arm {value!r} is invalid")
        return value
    x_position = float(actor.get_pose().p[0])
    return str(arm["negative"] if x_position < 0.0 else arm["nonnegative"])


def _actor_name(actor: Any) -> str:
    name = actor.get_name()
    return _text(name, "resolved actor name")


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
