"""Dataset contract for automatically labelled task-native grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .grasp_candidates import GraspCandidate, GraspIntent
from .task_manifest import TaskFrameSpec


CANDIDATE_SAMPLE_SCHEMA = "spatial.grasp_candidate_sample.v1"
TASK_NATIVE_NEGATIVE_KINDS = {
    "none",
    "along_region_offset",
    "across_closing_offset",
    "vertical_offset",
    "orientation_offset",
    "opening_width_offset",
    "wrong_task_region",
    "one_sided_contact",
    "support_collision",
    "unreachable",
    "failed_close_or_lift",
}
EXPERT_CONTACT_RELATIVE_TOLERANCE = 1.25
EXPERT_CONTACT_ABSOLUTE_TOLERANCE = 0.01
FORBIDDEN_INFERENCE_KEYS = {
    "actor_id",
    "actor_segmentation_id",
    "oracle_geometry",
    "oracle_graph",
    "oracle_object_pose",
    "physics_contacts",
    "contact_truth",
    "execution_success_label",
    "task_success_label",
}


@dataclass(frozen=True)
class CandidateExecutionLabel:
    candidate_id: str
    negative_kind: str
    ik_reachable: bool | None
    collision_free: bool | None
    close_planner_success: bool | None
    lift_planner_success: bool | None
    bilateral_contact: bool | None
    lifted_from_support: bool | None
    task_success: bool | None
    object_height_change_m: float | None = None
    collision_audit_scope: str = "not_evaluated"
    max_non_target_contact_impulse: float | None = None
    minimum_non_target_separation_m: float | None = None
    expert_contact_impulse_baseline: float | None = None
    expert_contact_impulse_limit: float | None = None
    collision_within_expert_baseline: bool | None = None
    source: str = "simulator_execution_probe"

    def __post_init__(self) -> None:
        if self.negative_kind not in TASK_NATIVE_NEGATIVE_KINDS:
            raise ValueError(
                f"unsupported task-native negative kind {self.negative_kind!r}"
            )

    @property
    def execution_success(self) -> bool | None:
        values = (
            self.ik_reachable,
            self.close_planner_success,
            self.lift_planner_success,
            self.bilateral_contact,
            self.lifted_from_support,
            self.task_success,
        )
        if any(value is False for value in values):
            return False
        if all(value is True for value in values):
            return True
        return None

    @property
    def task_native_safe_success(self) -> bool | None:
        if self.execution_success is False:
            return False
        if self.execution_success is None:
            return None
        return self.collision_safety_success

    @property
    def collision_safety_success(self) -> bool | None:
        if self.collision_free is True:
            return True
        if self.collision_within_expert_baseline is not None:
            return bool(self.collision_within_expert_baseline)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "negative_kind": self.negative_kind,
            "ik_reachable": self.ik_reachable,
            "collision_free": self.collision_free,
            "close_planner_success": self.close_planner_success,
            "lift_planner_success": self.lift_planner_success,
            "bilateral_contact": self.bilateral_contact,
            "lifted_from_support": self.lifted_from_support,
            "task_success": self.task_success,
            "execution_success": self.execution_success,
            "collision_safety_success": self.collision_safety_success,
            "task_native_safe_success": self.task_native_safe_success,
            "object_height_change_m": self.object_height_change_m,
            "collision_audit_scope": self.collision_audit_scope,
            "max_non_target_contact_impulse": self.max_non_target_contact_impulse,
            "minimum_non_target_separation_m": self.minimum_non_target_separation_m,
            "expert_contact_impulse_baseline": self.expert_contact_impulse_baseline,
            "expert_contact_impulse_limit": self.expert_contact_impulse_limit,
            "collision_within_expert_baseline": self.collision_within_expert_baseline,
            "source": self.source,
        }


def build_automatic_candidate_sample(
    *,
    task_spec: TaskFrameSpec,
    episode_id: str,
    seed: int,
    step: int,
    world_state_version: int,
    intent: GraspIntent,
    observations: Sequence[Mapping[str, Any]],
    candidates: Sequence[GraspCandidate],
    execution_labels: Iterable[CandidateExecutionLabel],
    oracle_geometry: Mapping[str, Any],
    oracle_candidate_graph: Mapping[str, Any],
    expert_summary_reference: str,
    candidate_preexecution_context: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one frozen-state record without leaking simulator truth to inputs."""

    if not task_spec.contributes_grasp_labels:
        raise ValueError(f"task {task_spec.task!r} does not contribute grasp labels")
    if task_spec.split != "train":
        raise ValueError("generalization tasks cannot be automatic-label training inputs")
    label_by_id = {label.candidate_id: label for label in execution_labels}
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    unknown_labels = sorted(set(label_by_id) - candidate_ids)
    if unknown_labels:
        raise ValueError(
            "execution labels reference unknown candidates: " + ", ".join(unknown_labels)
        )
    missing_labels = sorted(candidate_ids - set(label_by_id))
    if missing_labels:
        raise ValueError(
            "automatic candidate labels are incomplete: " + ", ".join(missing_labels)
        )
    context_by_id = dict(candidate_preexecution_context or {})
    unknown_context = sorted(set(context_by_id) - candidate_ids)
    if unknown_context:
        raise ValueError(
            "preexecution context references unknown candidates: "
            + ", ".join(unknown_context)
        )
    inference_observations = [dict(observation) for observation in observations]
    sample = {
        "schema_version": CANDIDATE_SAMPLE_SCHEMA,
        "sample_id": f"{episode_id}/step{int(step):04d}",
        "task": task_spec.task,
        "task_family": task_spec.family,
        "seed": int(seed),
        "step": int(step),
        "world_state_version": int(world_state_version),
        "inference_visible": {
            "intent": intent.as_dict(),
            "observations": inference_observations,
            "expert_summary_reference": expert_summary_reference,
        },
        "training_only": {
            "access": "oracle/training_only",
            "oracle_geometry": dict(oracle_geometry),
            "oracle_candidate_graph": dict(oracle_candidate_graph),
            "candidate_labels": [
                {
                    "candidate": {
                        **candidate.as_dict(),
                        **(
                            {
                                "preexecution_context": dict(
                                    context_by_id[candidate.candidate_id]
                                )
                            }
                            if candidate.candidate_id in context_by_id
                            else {}
                        ),
                    },
                    "execution": label_by_id[candidate.candidate_id].as_dict(),
                }
                for candidate in candidates
            ],
            "label_generation": {
                "point_annotation": "automatic",
                "semantic_role_source": "expert_summary",
                "metric_frame_source": "simulator_geometry",
                "hard_sample_scope": "current_task_native_only",
            },
        },
    }
    errors = validate_candidate_sample(sample)
    if errors:
        raise ValueError("invalid candidate sample: " + "; ".join(errors))
    return sample


def validate_candidate_sample(sample: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if sample.get("schema_version") != CANDIDATE_SAMPLE_SCHEMA:
        errors.append("unsupported schema_version")
    inference = sample.get("inference_visible")
    training = sample.get("training_only")
    if not isinstance(inference, Mapping):
        errors.append("inference_visible must be a mapping")
        return errors
    if not isinstance(training, Mapping):
        errors.append("training_only must be a mapping")
        return errors
    leaked_keys = sorted(_recursive_keys(inference) & FORBIDDEN_INFERENCE_KEYS)
    if leaked_keys:
        errors.append("oracle/training keys leaked into inference_visible: " + ", ".join(leaked_keys))
    leaked_sources = sorted(
        value
        for value in _recursive_string_values(inference)
        if "oracle" in value.lower() or "simulator_truth" in value.lower()
    )
    if leaked_sources:
        errors.append("oracle source leaked into inference_visible")
    if training.get("access") != "oracle/training_only":
        errors.append("training_only.access must be oracle/training_only")
    labels = training.get("candidate_labels")
    if not isinstance(labels, list) or not labels:
        errors.append("training_only.candidate_labels must be a non-empty list")
    else:
        ids = []
        for row in labels:
            if not isinstance(row, Mapping):
                errors.append("candidate label rows must be mappings")
                continue
            candidate = row.get("candidate")
            execution = row.get("execution")
            if not isinstance(candidate, Mapping) or not isinstance(execution, Mapping):
                errors.append("candidate label row requires candidate and execution mappings")
                continue
            candidate_id = candidate.get("id")
            if candidate_id != execution.get("candidate_id"):
                errors.append("candidate and execution candidate_id mismatch")
            ids.append(candidate_id)
            if execution.get("negative_kind") not in TASK_NATIVE_NEGATIVE_KINDS:
                errors.append("candidate label uses unsupported negative_kind")
        if len(ids) != len(set(ids)):
            errors.append("candidate labels contain duplicate ids")
    return errors


def expert_contact_envelope_limit(
    baseline_impulse: float,
    *,
    relative_tolerance: float = EXPERT_CONTACT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = EXPERT_CONTACT_ABSOLUTE_TOLERANCE,
) -> float:
    if baseline_impulse < 0.0:
        raise ValueError("baseline_impulse must be non-negative")
    if relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be >= 1")
    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative")
    return relative_tolerance * float(baseline_impulse) + absolute_tolerance


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for item in value.values():
            result.update(_recursive_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_recursive_keys(item))
        return result
    return set()


def _recursive_string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_recursive_string_values(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_recursive_string_values(item))
        return result
    return set()
