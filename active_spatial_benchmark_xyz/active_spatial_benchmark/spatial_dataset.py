"""Schemas and label helpers for the Phase 2 multi-view dataset."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np


RELATION_SPECS = {
    "object_between_fingers": {
        "required_nodes": ("object.center", "gripper.finger_a_inner", "gripper.finger_b_inner"),
        "axis": "closing",
        "weight": 1.0,
    },
    "finger_a_contact": {
        "required_nodes": ("object.surface_a", "gripper.finger_a_inner"),
        "axis": "closing",
        "weight": 1.2,
    },
    "finger_b_contact": {
        "required_nodes": ("object.surface_b", "gripper.finger_b_inner"),
        "axis": "closing",
        "weight": 1.2,
    },
    "supported_by_table": {
        "required_nodes": ("object.center", "support.plane_anchor"),
        "axis": "vertical",
        "weight": 0.6,
    },
    "lifted_from_support": {
        "required_nodes": ("object.center", "support.plane_anchor"),
        "axis": "vertical",
        "weight": 0.8,
    },
    "moves_with_gripper": {
        "required_nodes": ("object.center", "gripper.jaw_center"),
        "axis": "temporal",
        "weight": 1.0,
    },
}


def build_view_labels(
    capture: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    relation_specs: Mapping[str, Mapping[str, Any]] = RELATION_SPECS,
) -> dict[str, Any]:
    """Build training-only keypoint and relation labels for one rendered view."""

    node_by_id = {str(node["id"]): node for node in graph["nodes"]}
    projections = capture["projections"]
    visibility = capture["node_visibility"]
    keypoints = []
    for node_id, node in node_by_id.items():
        projection = projections.get(node_id)
        visible = float(visibility.get(node_id, 0.0))
        if projection is None or not projection.get("in_frame", False):
            observation_state = "out_of_frame"
        elif visible > 0.0:
            observation_state = "visible"
        else:
            observation_state = "occluded"
        keypoints.append(
            {
                "id": node_id,
                "semantic_type": node["semantic_type"],
                "position_world_m": list(node["position_mean_world_m"]),
                "pixel_uv": projection["pixel"] if projection else None,
                "camera_depth_m": projection["camera_depth_m"] if projection else None,
                "in_frame": bool(projection and projection.get("in_frame", False)),
                "visibility": visible,
                "observation_state": observation_state,
                "label_source": node["source"],
                "access": "training_only",
            }
        )

    direction = _unit(np.asarray(capture["view_direction_world"], dtype=np.float64))
    closing_axis = _unit(np.asarray(graph["query_axes"]["closing_axis_world"], dtype=np.float64))
    observability = {
        "closing": 1.0 - abs(float(np.dot(direction, closing_axis))),
        "object": 1.0
        - abs(
            float(
                np.dot(
                    direction,
                    _unit(np.asarray(graph["query_axes"]["object_axis_world"], dtype=np.float64)),
                )
            )
        ),
        "vertical": 1.0 - abs(float(direction[2])),
        "temporal": 0.0,
    }
    graph_edges = {str(edge["id"]): edge for edge in graph["edges"]}
    relation_labels = []
    for edge_id, spec in relation_specs.items():
        edge = graph_edges[edge_id]
        required = tuple(spec["required_nodes"])
        alternatives = spec.get("visual_alternatives")
        if alternatives:
            visible_quality = max(
                min(_effective_visibility(node_id, visibility) for node_id in alternative)
                for alternative in alternatives
            )
        else:
            visible_quality = min(_effective_visibility(node_id, visibility) for node_id in required)
        axes = tuple(spec.get("axes", (spec.get("axis"),)))
        axis_observability = min(observability[str(axis)] for axis in axes)
        evidence_quality = float(np.clip(visible_quality * axis_observability, 0.0, 1.0))
        relation_labels.append(
            {
                "id": edge_id,
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation": edge.get("relation"),
                "state": edge.get("state"),
                "probability": edge.get("probability"),
                "required_nodes": list(required),
                "visible_node_quality": round(visible_quality, 6),
                "axis_observability": round(axis_observability, 6),
                "evidence_quality": round(evidence_quality, 6),
                "measurement": edge.get("measurement", {}),
                "access": "training_only",
            }
        )
    return {
        "schema_version": "phase2.view_labels.v1",
        "access": "training_only",
        "frame_id": capture["frame_id"],
        "view": capture["view"],
        "keypoints": keypoints,
        "relations": relation_labels,
    }


def rank_candidate_views(
    captures: list[Mapping[str, Any]],
    labels_by_view: Mapping[str, Mapping[str, Any]],
    *,
    relation_specs: Mapping[str, Mapping[str, Any]] = RELATION_SPECS,
) -> list[dict[str, Any]]:
    """Rank additional views by an explicit Oracle evidence-gain proxy.

    This target is intentionally not called learned error reduction.  It is the
    decrease of a simple relation evidence uncertainty after adding one view to
    the current observation, minus camera motion cost.
    """

    if not captures or captures[0]["view"] != "current":
        raise ValueError("captures must start with the current view")
    current = captures[0]
    current_evidence = _relation_evidence(labels_by_view["current"])
    rows = []
    for capture in captures:
        if capture["view"] == "current":
            continue
        if _same_camera_pose(current, capture):
            continue
        candidate_evidence = _relation_evidence(labels_by_view[str(capture["view"])])
        weighted_gain = 0.0
        weight_sum = 0.0
        relation_gain = {}
        for edge_id, spec in relation_specs.items():
            before = _evidence_uncertainty(current_evidence[edge_id])
            after_evidence = current_evidence[edge_id]
            after_evidence += candidate_evidence[edge_id]
            after = _evidence_uncertainty(after_evidence)
            gain = max(0.0, before - after)
            relation_gain[edge_id] = round(gain, 6)
            weight = float(spec["weight"])
            weighted_gain += weight * gain
            weight_sum += weight
        information_gain = weighted_gain / max(weight_sum, 1e-9)
        move_cost = _move_cost(current, capture)
        utility = information_gain - 0.15 * move_cost
        rows.append(
            {
                "view": capture["view"],
                "oracle_information_gain_proxy": round(information_gain, 6),
                "camera_move_cost": round(move_cost, 6),
                "oracle_utility_proxy": round(utility, 6),
                "relation_gain": relation_gain,
            }
        )
    rows.sort(key=lambda row: row["oracle_utility_proxy"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def oracle_next_tool(
    graph: Mapping[str, Any],
    ranking: list[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.9,
) -> dict[str, Any]:
    """Choose the supervision target after respecting query stop semantics."""

    verdict = str(graph.get("verdict", "uncertain"))
    confidence = float(graph.get("confidence", 0.0))
    if verdict in {"true", "false"} and confidence >= confidence_threshold:
        return {
            "tool": "stop",
            "reason": "The current graph verdict already meets the confidence threshold.",
            "view": None,
        }
    missing = set(graph.get("missing_evidence", []))
    if missing & {"lifted_from_support", "moves_with_gripper"}:
        return {
            "tool": "spatial.controlled_lift_probe",
            "reason": "Static camera motion cannot establish temporal rigid-motion evidence.",
            "view": ranking[0]["view"] if ranking else None,
        }
    if ranking and float(ranking[0]["oracle_utility_proxy"]) > 0.0:
        return {
            "tool": "camera.select_view",
            "reason": "The best additional view has positive evidence utility after motion cost.",
            "view": str(ranking[0]["view"]),
        }
    return {
        "tool": "stop",
        "reason": "No additional camera view has positive utility.",
        "view": None,
    }


def oracle_next_view_from_current_evidence(
    current_labels: Mapping[str, Any],
    ranking: list[Mapping[str, Any]],
    *,
    relation_specs: Mapping[str, Mapping[str, Any]] = RELATION_SPECS,
    required_evidence: float = 0.75,
) -> dict[str, Any]:
    """Label NBV from observable evidence, without consulting query truth.

    ``oracle_next_tool`` is useful for full diagnostic orchestration, where an
    Oracle graph may establish that a temporal probe or stop is appropriate.
    A view-ranker training target has a stricter boundary: query truth must not
    make the current frame appear sufficient.  This helper therefore inspects
    only current-frame evidence quality and candidate evidence gain.
    """

    current_evidence = _relation_evidence(current_labels)
    insufficient = [
        edge_id
        for edge_id in relation_specs
        if current_evidence.get(edge_id, 0.0) < required_evidence
    ]
    if not insufficient:
        return {
            "tool": "stop",
            "reason": "Current-view evidence meets every relation threshold.",
            "view": None,
            "insufficient_relations": [],
            "current_evidence": current_evidence,
        }

    scored = []
    for row in ranking:
        weighted_gain = 0.0
        weight_sum = 0.0
        for edge_id in insufficient:
            weight = float(relation_specs[edge_id]["weight"])
            weighted_gain += weight * float(row.get("relation_gain", {}).get(edge_id, 0.0))
            weight_sum += weight
        targeted_gain = weighted_gain / max(weight_sum, 1e-9)
        targeted_utility = targeted_gain - 0.15 * float(row.get("camera_move_cost", 0.0))
        scored.append(
            {
                "view": str(row["view"]),
                "targeted_information_gain_proxy": round(targeted_gain, 6),
                "camera_move_cost": round(float(row.get("camera_move_cost", 0.0)), 6),
                "targeted_utility_proxy": round(targeted_utility, 6),
            }
        )
    scored.sort(key=lambda row: row["targeted_utility_proxy"], reverse=True)
    if scored and scored[0]["targeted_utility_proxy"] > 0.0:
        return {
            "tool": "camera.select_view",
            "reason": "Current evidence is insufficient and this view best reduces the relevant gaps.",
            "view": scored[0]["view"],
            "insufficient_relations": insufficient,
            "current_evidence": current_evidence,
            "targeted_ranking": scored,
        }
    return {
        "tool": "stop",
        "reason": "Current evidence is insufficient, but no candidate has positive targeted utility.",
        "view": None,
        "insufficient_relations": insufficient,
        "current_evidence": current_evidence,
        "targeted_ranking": scored,
    }


def validate_manifest_record(record: Mapping[str, Any]) -> list[str]:
    errors = []
    required = {
        "sample_id",
        "episode_id",
        "state_id",
        "frame_id",
        "view",
        "world_state_version",
        "inference_visible",
        "training_only",
    }
    missing = required - set(record)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors
    inference = record["inference_visible"]
    training = record["training_only"]
    for field in ("rgb", "depth", "camera"):
        if field not in inference:
            errors.append(f"missing inference_visible.{field}")
    if record.get("query") == "verify_pregrasp" and "robot_kinematics" not in inference:
        errors.append("missing inference_visible.robot_kinematics")
    for field in ("labels", "actor_segmentation", "oracle_target_mask"):
        if field not in training:
            errors.append(f"missing training_only.{field}")
    forbidden_tokens = ("actor_segmentation", "oracle", "ground_truth", "task_actor_id")
    for field, value in inference.items():
        joined = f"{field}:{value}".lower()
        if any(token in joined for token in forbidden_tokens):
            errors.append(f"truth leakage in inference_visible.{field}")
        if PurePosixPath(str(value)).is_absolute():
            errors.append(f"absolute path in inference_visible.{field}")
    return errors


def validate_frame_alignment(
    record: Mapping[str, Any],
    camera_metadata: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> list[str]:
    errors = []
    expected = (int(record["frame_id"]), str(record["view"]), int(record["world_state_version"]))
    camera_value = (
        int(camera_metadata["frame_id"]),
        str(camera_metadata["view"]),
        int(camera_metadata["world_state_version"]),
    )
    label_value = (int(labels["frame_id"]), str(labels["view"]), int(labels["world_state_version"]))
    if camera_value != expected:
        errors.append(f"camera frame mismatch: expected {expected}, got {camera_value}")
    if label_value != expected:
        errors.append(f"label frame mismatch: expected {expected}, got {label_value}")
    return errors


def validate_bias_application(
    applied_bias: Mapping[str, Any],
    *,
    minimum_axis_fraction: float = 0.9,
    maximum_translation_residual_m: float = 0.01,
) -> list[str]:
    """Reject commanded perturbations that were not realized by the robot."""

    if abs(float(applied_bias.get("distance_m", 0.0))) < 1e-9:
        return []
    errors = []
    if not bool(applied_bias.get("planner_success", False)):
        errors.append("bias planner reported failure")
    axis_fraction = float(applied_bias.get("requested_axis_fraction_achieved", 0.0))
    if axis_fraction < minimum_axis_fraction:
        errors.append(
            f"only {axis_fraction} of requested bias was achieved; "
            f"minimum is {minimum_axis_fraction}"
        )
    residual = float(applied_bias.get("translation_residual_m", float("inf")))
    if residual > maximum_translation_residual_m:
        errors.append(
            f"bias translation residual {residual} m exceeds "
            f"{maximum_translation_residual_m} m"
        )
    return errors


def _effective_visibility(node_id: str, visibility: Mapping[str, Any]) -> float:
    if node_id.startswith("gripper."):
        return 1.0
    return float(np.clip(visibility.get(node_id, 0.0), 0.0, 1.0))


def _relation_evidence(labels: Mapping[str, Any]) -> dict[str, float]:
    return {str(item["id"]): float(item["evidence_quality"]) for item in labels["relations"]}


def _evidence_uncertainty(evidence: float) -> float:
    return 1.0 / (1.0 + 4.0 * max(0.0, evidence))


def _move_cost(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    current_position = np.asarray(current["camera_position_world"], dtype=np.float64)
    candidate_position = np.asarray(candidate["camera_position_world"], dtype=np.float64)
    position_delta = float(np.linalg.norm(candidate_position - current_position))
    current_direction = _unit(np.asarray(current["view_direction_world"], dtype=np.float64))
    candidate_direction = _unit(np.asarray(candidate["view_direction_world"], dtype=np.float64))
    angle = math.acos(float(np.clip(np.dot(current_direction, candidate_direction), -1.0, 1.0)))
    return float(np.clip(0.7 * position_delta / 0.8 + 0.3 * angle / math.pi, 0.0, 1.0))


def _same_camera_pose(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    position_tolerance_m: float = 1e-5,
    direction_tolerance_rad: float = 1e-5,
) -> bool:
    first_position = np.asarray(first["camera_position_world"], dtype=np.float64)
    second_position = np.asarray(second["camera_position_world"], dtype=np.float64)
    if float(np.linalg.norm(first_position - second_position)) > position_tolerance_m:
        return False
    first_direction = _unit(np.asarray(first["view_direction_world"], dtype=np.float64))
    second_direction = _unit(np.asarray(second["view_direction_world"], dtype=np.float64))
    angle = math.acos(float(np.clip(np.dot(first_direction, second_direction), -1.0, 1.0)))
    return angle <= direction_tolerance_rad


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("cannot normalize near-zero vector")
    return value / norm
