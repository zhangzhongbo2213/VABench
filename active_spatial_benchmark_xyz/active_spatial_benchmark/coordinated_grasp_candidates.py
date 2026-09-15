"""Joint candidate graphs for coordinated multi-arm grasp events."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .expert_grasp_trace import validate_expert_grasp_event_sample


COORDINATED_CANDIDATE_GRAPH_SCHEMA = "spatial.coordinated_grasp_candidate_graph.v1"


def build_coordinated_candidate_graph(
    event_samples: Sequence[Mapping[str, Any]],
    *,
    graph_variant: str = "base",
    generation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(event_samples) != 2:
        raise ValueError("the first coordinated graph requires exactly two event samples")
    if not isinstance(graph_variant, str) or not graph_variant.strip():
        raise ValueError("graph_variant must be non-empty")
    graph_variant = graph_variant.strip()
    samples = sorted(event_samples, key=_sample_arm)
    if [_sample_arm(sample) for sample in samples] != ["left", "right"]:
        raise ValueError("coordinated graph requires one left and one right event")
    for sample in samples:
        errors = validate_expert_grasp_event_sample(sample)
        if errors:
            raise ValueError("invalid expert event sample: " + "; ".join(errors))
    task_values = {str(sample.get("task")) for sample in samples}
    seed_values = {int(sample.get("seed", -1)) for sample in samples}
    groups = {
        sample["inference_visible"]["query"].get("coordination_group")
        for sample in samples
    }
    targets = {
        sample["inference_visible"]["query"].get("target") for sample in samples
    }
    if len(task_values) != 1 or len(seed_values) != 1:
        raise ValueError("coordinated event samples must share task and seed")
    if len(groups) != 1 or None in groups:
        raise ValueError("coordinated event samples must share a non-null group")
    if len(targets) != 1:
        raise ValueError("coordinated event samples must share the semantic target")

    candidates_by_arm = {
        _sample_arm(sample): sample["training_only"]["candidate_perturbations"]
        for sample in samples
    }
    left_candidates = _executable_candidates(candidates_by_arm["left"])
    right_candidates = _executable_candidates(candidates_by_arm["right"])
    if len(left_candidates) != len(right_candidates) or not left_candidates:
        raise ValueError("coordinated event candidate suites must align")
    for index, (left, right) in enumerate(zip(left_candidates, right_candidates)):
        left_kind = left.get("generation_parameters", {}).get("perturbation_kind")
        right_kind = right.get("generation_parameters", {}).get("perturbation_kind")
        if left_kind != right_kind:
            raise ValueError(f"candidate perturbation kinds diverge at index {index}")

    pair_specs = [(0, 0, "nominal")]
    for index in range(1, len(left_candidates)):
        pair_specs.extend(
            (
                (index, index, "both_same_factor"),
                (index, 0, "left_only_factor"),
                (0, index, "right_only_factor"),
            )
        )
    coordination_group = next(iter(groups))
    joint_candidates = [
        _joint_candidate(
            coordination_group=str(coordination_group),
            joint_index=joint_index,
            left_index=left_index,
            right_index=right_index,
            mode=mode,
            left=left_candidates[left_index],
            right=right_candidates[right_index],
            samples=samples,
        )
        for joint_index, (left_index, right_index, mode) in enumerate(pair_specs)
    ]
    result = {
        "schema_version": COORDINATED_CANDIDATE_GRAPH_SCHEMA,
        "graph_id": (
            f"{next(iter(task_values))}_seed{next(iter(seed_values))}/"
            f"{coordination_group}/{graph_variant}"
        ),
        "graph_variant": graph_variant,
        "task": next(iter(task_values)),
        "seed": next(iter(seed_values)),
        "coordination_group": coordination_group,
        "inference_visible": {
            "query": {
                "type": "propose_coordinated_grasp_candidates",
                "target": next(iter(targets)),
                "task_goal": next(iter(task_values)),
                "required_arms": 2,
                "coordination_group": coordination_group,
                "post_grasp_goal": samples[0]["inference_visible"]["query"].get(
                    "post_grasp_goal"
                ),
            },
            "observations_by_arm": {
                _sample_arm(sample): deepcopy(
                    sample["inference_visible"].get("observations", [])
                )
                for sample in samples
            },
        },
        "training_only": {
            "access": "oracle/training_only",
            "source_event_samples": [sample["sample_id"] for sample in samples],
            "joint_candidates": joint_candidates,
            "generation": {
                "point_annotation": "automatic_from_successful_trajectory",
                "pairing_policy": "nominal_plus_matched_and_unilateral_single_factor",
                "predicted_execution_labels": False,
                **dict(generation_metadata or {}),
            },
        },
    }
    errors = validate_coordinated_candidate_graph(result)
    if errors:
        raise ValueError("invalid coordinated candidate graph: " + "; ".join(errors))
    return result


def validate_coordinated_candidate_graph(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != COORDINATED_CANDIDATE_GRAPH_SCHEMA:
        errors.append("unsupported coordinated candidate graph schema")
    inference = value.get("inference_visible")
    training = value.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        errors.append("coordinated graph requires inference_visible and training_only")
        return errors
    if training.get("access") != "oracle/training_only":
        errors.append("coordinated graph training payload must remain training-only")
    if "joint_candidates" in inference:
        errors.append("oracle joint candidates leaked into inference_visible")
    candidates = training.get("joint_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("joint_candidates must be a non-empty list")
        return errors
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            errors.append("joint candidates must be mappings")
            continue
        ids.append(candidate.get("id"))
        members = candidate.get("members")
        if not isinstance(members, list) or len(members) != 2:
            errors.append("joint candidate must contain two members")
            continue
        if sorted(member.get("arm") for member in members) != ["left", "right"]:
            errors.append("joint candidate must contain left and right members")
        if candidate.get("predicted_execution_success") is not None:
            errors.append("unprobed joint candidate cannot claim execution success")
        if candidate.get("access") != "oracle/training_only":
            errors.append("joint candidate must remain training-only")
    if len(ids) != len(set(ids)):
        errors.append("joint candidate IDs must be unique")
    return errors


def _joint_candidate(
    *,
    coordination_group: str,
    joint_index: int,
    left_index: int,
    right_index: int,
    mode: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_center = _vector(left["frame"]["center_world_m"], "left center")
    right_center = _vector(right["frame"]["center_world_m"], "right center")
    left_approach = _unit(left["frame"]["approach_axis_world"], "left approach")
    right_approach = _unit(right["frame"]["approach_axis_world"], "right approach")
    left_closing = _unit(left["frame"]["closing_axis_world"], "left closing")
    right_closing = _unit(right["frame"]["closing_axis_world"], "right closing")
    span = right_center - left_center
    sample_by_arm = {_sample_arm(sample): sample for sample in samples}
    return {
        "id": f"{coordination_group}.joint_{joint_index:02d}",
        "coordination_group": coordination_group,
        "pairing_mode": mode,
        "members": [
            _member("left", left_index, left, sample_by_arm["left"]),
            _member("right", right_index, right, sample_by_arm["right"]),
        ],
        "pair_geometry": {
            "center_midpoint_world_m": _round((left_center + right_center) / 2.0),
            "center_span_world_m": _round(span),
            "center_separation_m": round(float(np.linalg.norm(span)), 8),
            "approach_axis_dot": round(float(np.dot(left_approach, right_approach)), 8),
            "closing_axis_dot": round(float(np.dot(left_closing, right_closing)), 8),
            "opening_width_sum_m": round(
                float(left["frame"]["opening_width_m"])
                + float(right["frame"]["opening_width_m"]),
                8,
            ),
        },
        "coordination_constraints": {
            "synchronized_pregrasp": True,
            "synchronized_grasp": True,
            "synchronized_close": True,
            "shared_target": True,
            "joint_reachability": "unknown_until_probe",
            "inter_gripper_collision": "unknown_until_probe",
        },
        "predicted_execution_success": None,
        "source": "automatic_coordinated_expert_frame_pairing",
        "access": "oracle/training_only",
    }


def _member(
    arm: str,
    candidate_index: int,
    candidate: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    event = sample["training_only"]["event"]
    return {
        "arm": arm,
        "event_id": event["event_id"],
        "source_sample_id": sample["sample_id"],
        "candidate_index": int(candidate_index),
        "candidate_id": candidate["id"],
        "candidate": deepcopy(dict(candidate)),
    }


def _sample_arm(sample: Mapping[str, Any]) -> str:
    return str(sample.get("training_only", {}).get("event", {}).get("arm", ""))


def _executable_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if candidate.get("generation_parameters", {}).get("perturbation_kind")
        != "opening_width_offset"
    ]


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _unit(value: Any, name: str) -> np.ndarray:
    result = _vector(value, name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return result / norm


def _round(value: np.ndarray) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), 8).tolist()
