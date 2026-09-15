"""Inference-visible translation recovery for failed pre-grasp relations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


TRANSLATION_RELATIONS = (
    "grasp_height_aligned",
    "grasp_region_along_object_axis",
    "object_between_fingers",
)
ORIENTATION_RELATION = "closing_axis_perpendicular_to_object_axis"


def plan_pregrasp_recovery(
    tool_result: Mapping[str, Any],
    *,
    max_translation_step_m: float = 0.045,
    minimum_translation_m: float = 0.003,
    along_target_fraction: float = 0.45,
    force_relations: Sequence[str] = (),
    allow_upward_vertical_recovery: bool = False,
) -> dict[str, Any]:
    """Plan one bounded Cartesian correction using only the returned belief graph.

    The planner deliberately handles one failed relation per call. The caller
    must invalidate the old belief, execute the correction, and diagnose again
    before requesting another step.
    """

    if max_translation_step_m <= 0.0:
        raise ValueError("max_translation_step_m must be positive")
    if minimum_translation_m < 0.0:
        raise ValueError("minimum_translation_m must be non-negative")
    if not 0.0 <= along_target_fraction < 1.0:
        raise ValueError("along_target_fraction must be in [0, 1)")

    relations = tool_result.get("relations", {})
    forced = [
        str(edge_id)
        for edge_id in force_relations
        if edge_id in relations
        and float(relations[edge_id].get("uncertainty", 1.0)) <= 0.35
    ]
    failed = list(dict.fromkeys([*forced, *_failed_relations(tool_result, relations)]))
    graph = tool_result.get("belief_graph", {})
    query_axes = graph.get("query_axes", {})
    candidates = []
    unsupported = []
    safety_rejected = []
    for edge_id in failed:
        relation = relations.get(edge_id, {})
        probability = float(relation.get("probability", 0.5))
        measurement = relation.get("measurement", {})
        planned = _relation_correction(
            edge_id,
            measurement=measurement,
            query_axes=query_axes,
            graph=graph,
            along_target_fraction=along_target_fraction,
            allow_upward_vertical_recovery=allow_upward_vertical_recovery,
        )
        if planned is None:
            unsupported.append(edge_id)
            continue
        if planned.get("safety_rejected", False):
            safety_rejected.append((edge_id, planned))
            continue
        candidates.append((probability, edge_id, relation, planned))

    if safety_rejected:
        edge_id, rejected = safety_rejected[0]
        return {
            "schema_version": "phase8.pregrasp_recovery_plan.v1",
            "status": "unsafe",
            "source_relation": edge_id,
            "failed_relations": failed,
            "unsupported_relations": unsupported,
            "safety_rejected_relations": [item[0] for item in safety_rejected],
            "rejected_translation_world_m": _vector(rejected["translation_world_m"]),
            "reason": rejected["reason"],
            "inference_visible_only": True,
        }

    if not candidates:
        return {
            "schema_version": "phase8.pregrasp_recovery_plan.v1",
            "status": "unsupported" if unsupported else "not_required",
            "source_relation": unsupported[0] if unsupported else None,
            "failed_relations": failed,
            "unsupported_relations": unsupported,
            "reason": (
                "The failed relation requires orientation recovery or lacks a stable inference-visible axis."
                if unsupported
                else "No failed translation relation requires recovery."
            ),
            "inference_visible_only": True,
        }

    probability, edge_id, relation, planned = min(candidates, key=lambda item: item[0])
    raw = np.asarray(planned["translation_world_m"], dtype=np.float64)
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm < minimum_translation_m:
        return {
            "schema_version": "phase8.pregrasp_recovery_plan.v1",
            "status": "not_required",
            "source_relation": edge_id,
            "failed_relations": failed,
            "unsupported_relations": unsupported,
            "reason": "The predicted correction is below the minimum translation deadband.",
            "raw_translation_world_m": _vector(raw),
            "inference_visible_only": True,
        }
    scale = min(1.0, float(max_translation_step_m) / max(raw_norm, 1e-12))
    bounded = raw * scale
    return {
        "schema_version": "phase8.pregrasp_recovery_plan.v1",
        "status": "move",
        "source_relation": edge_id,
        "failed_relations": failed,
        "unsupported_relations": unsupported,
        "relation_probability": round(probability, 6),
        "relation_uncertainty": round(float(relation.get("uncertainty", 1.0)), 6),
        "measurement": dict(relation.get("measurement", {})),
        "axis_world": _vector(planned["axis_world"]),
        "raw_translation_world_m": _vector(raw),
        "translation_world_m": _vector(bounded),
        "translation_norm_m": round(float(np.linalg.norm(bounded)), 6),
        "clamped": scale < 1.0,
        "expected_measurement_after": planned["expected_measurement_after"],
        "reason": planned["reason"],
        "inference_visible_only": True,
    }


def recovery_confirmation_failures(
    tool_result: Mapping[str, Any],
    recovered_relations: Sequence[str],
    *,
    maximum_vertical_offset_m: float = 0.008,
    maximum_closing_axis_offset_m: float = 0.006,
    maximum_along_limit_fraction: float = 0.55,
) -> list[str]:
    """Return previously corrected relations that have not reached a safe interior band."""

    relations = tool_result.get("relations", {})
    failures = []
    for edge_id in dict.fromkeys(str(value) for value in recovered_relations):
        relation = relations.get(edge_id, {})
        if float(relation.get("uncertainty", 1.0)) > 0.35:
            failures.append(edge_id)
            continue
        measurement = relation.get("measurement", {})
        if edge_id == "grasp_height_aligned":
            value = measurement.get("predicted_vertical_offset_m")
            if value is None or abs(float(value)) > maximum_vertical_offset_m:
                failures.append(edge_id)
        elif edge_id == "grasp_region_along_object_axis":
            offset = measurement.get("predicted_axis_offset_m")
            limit = measurement.get("predicted_axis_limit_m")
            if (
                offset is None
                or limit is None
                or abs(float(offset)) > maximum_along_limit_fraction * max(float(limit), 1e-4)
            ):
                failures.append(edge_id)
        elif edge_id == "object_between_fingers":
            value = measurement.get("predicted_closing_axis_offset_m")
            if value is None or abs(float(value)) > maximum_closing_axis_offset_m:
                failures.append(edge_id)
    return failures


def _failed_relations(
    tool_result: Mapping[str, Any],
    relations: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    explicit = [
        str(edge_id)
        for edge_id in tool_result.get("missing_evidence", [])
        if edge_id in relations
    ]
    if explicit:
        supported = [
            edge_id
            for edge_id in explicit
            if edge_id in (*TRANSLATION_RELATIONS, ORIENTATION_RELATION)
            and float(relations[edge_id].get("uncertainty", 1.0)) <= 0.35
        ]
        if supported:
            return supported
    return [
        edge_id
        for edge_id in (*TRANSLATION_RELATIONS, ORIENTATION_RELATION)
        if float(relations.get(edge_id, {}).get("probability", 1.0)) <= 0.20
        and float(relations.get(edge_id, {}).get("uncertainty", 1.0)) <= 0.35
    ]


def _relation_correction(
    edge_id: str,
    *,
    measurement: Mapping[str, Any],
    query_axes: Mapping[str, Any],
    graph: Mapping[str, Any],
    along_target_fraction: float,
    allow_upward_vertical_recovery: bool,
) -> dict[str, Any] | None:
    if edge_id == "grasp_height_aligned":
        if "predicted_vertical_offset_m" not in measurement:
            return None
        axis = _unit(query_axes.get("support_normal_world", [0.0, 0.0, 1.0]))
        offset = float(measurement["predicted_vertical_offset_m"])
        translation = -offset * axis
        if not allow_upward_vertical_recovery and float(np.dot(translation, axis)) > 0.0:
            return {
                "axis_world": axis,
                "translation_world_m": translation,
                "safety_rejected": True,
                "reason": (
                    "Upward pre-grasp recovery is blocked until contact-free motion can be verified."
                ),
            }
        return {
            "axis_world": axis,
            "translation_world_m": translation,
            "expected_measurement_after": {"predicted_vertical_offset_m": 0.0},
            "reason": "Move opposite the measured vertical grasp-center offset.",
        }
    if edge_id == "grasp_region_along_object_axis":
        if "predicted_axis_offset_m" not in measurement:
            return None
        axis = _object_axis(query_axes, graph)
        if axis is None:
            return None
        offset = float(measurement["predicted_axis_offset_m"])
        limit = max(float(measurement.get("predicted_axis_limit_m", 0.05)), 1e-4)
        target = np.sign(offset) * along_target_fraction * limit
        scalar = -(offset - target)
        return {
            "axis_world": axis,
            "translation_world_m": scalar * axis,
            "expected_measurement_after": {"predicted_axis_offset_m": round(float(target), 6)},
            "reason": "Move along the learned object axis toward the interior grasp region.",
        }
    if edge_id == "object_between_fingers":
        if "predicted_closing_axis_offset_m" not in measurement:
            return None
        axis = _axis(query_axes.get("closing_axis_world"))
        if axis is None:
            return None
        offset = float(measurement["predicted_closing_axis_offset_m"])
        return {
            "axis_world": axis,
            "translation_world_m": offset * axis,
            "expected_measurement_after": {"predicted_closing_axis_offset_m": 0.0},
            "reason": "Move the grasp center toward the object along the gripper closing axis.",
        }
    return None


def _object_axis(
    query_axes: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> np.ndarray | None:
    axis = _axis(query_axes.get("object_axis_world"))
    if axis is not None:
        return axis
    nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    start = nodes.get("object.axis_start", {}).get("position_mean_world_m")
    end = nodes.get("object.axis_end", {}).get("position_mean_world_m")
    if start is None or end is None:
        return None
    return _axis(np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64))


def _axis(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return None
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-9 else None


def _unit(value: Sequence[float]) -> np.ndarray:
    result = _axis(value)
    if result is None:
        raise ValueError("expected a non-zero finite 3D axis")
    return result


def _vector(value: Sequence[float] | np.ndarray) -> list[float]:
    return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]
