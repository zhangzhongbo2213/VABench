"""Oracle template and view geometry for the verify_pregrasp query."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .spatial_graph import obb_support_radius, probability_state, unit


PREGRASP_EDGE_IDS = (
    "object_between_fingers",
    "grasp_region_along_object_axis",
    "grasp_height_aligned",
    "closing_axis_perpendicular_to_object_axis",
)

PREGRASP_RELATION_SPECS = {
    "object_between_fingers": {
        "required_nodes": ("object.center", "gripper.finger_a_inner_tip", "gripper.finger_b_inner_tip"),
        "visual_alternatives": (("object.center",), ("object.axis_start", "object.axis_end")),
        "axes": ("closing",),
        "weight": 1.3,
    },
    "grasp_region_along_object_axis": {
        "required_nodes": ("object.axis_start", "object.axis_end", "gripper.grasp_center"),
        # The image/mask encoder can estimate the pen axis while one exact OBB
        # endpoint lies on an occlusion boundary.  The endpoints remain the
        # graph geometry; object.center is an alternative visual support cue.
        "visual_alternatives": (("object.axis_start", "object.axis_end"), ("object.center",)),
        "axes": ("object",),
        "weight": 1.0,
    },
    "grasp_height_aligned": {
        "required_nodes": ("object.center", "gripper.grasp_center"),
        "visual_alternatives": (("object.center",),),
        "axes": ("vertical",),
        "weight": 1.1,
    },
    "closing_axis_perpendicular_to_object_axis": {
        "required_nodes": (
            "object.axis_start",
            "object.axis_end",
            "gripper.finger_a_inner_tip",
            "gripper.finger_b_inner_tip",
        ),
        "visual_alternatives": (("object.axis_start", "object.axis_end"), ("object.center",)),
        "axes": ("closing", "object"),
        "weight": 1.0,
    },
}


@dataclass(frozen=True)
class VerifyPregraspGeometry:
    finger_a_base: np.ndarray
    finger_b_base: np.ndarray
    finger_a_tip: np.ndarray
    finger_b_tip: np.ndarray
    object_center: np.ndarray
    object_rotation: np.ndarray
    object_half_extents: np.ndarray
    object_axis: np.ndarray
    support_z: float


def build_verify_pregrasp_oracle_graph(
    geometry: VerifyPregraspGeometry,
    *,
    world_state_version: int,
) -> dict[str, Any]:
    finger_a_base = np.asarray(geometry.finger_a_base, dtype=np.float64)
    finger_b_base = np.asarray(geometry.finger_b_base, dtype=np.float64)
    finger_a_tip = np.asarray(geometry.finger_a_tip, dtype=np.float64)
    finger_b_tip = np.asarray(geometry.finger_b_tip, dtype=np.float64)
    object_center = np.asarray(geometry.object_center, dtype=np.float64)
    rotation = np.asarray(geometry.object_rotation, dtype=np.float64)
    half_extents = np.asarray(geometry.object_half_extents, dtype=np.float64)
    object_axis = unit(geometry.object_axis)
    closing_axis = unit(finger_b_tip - finger_a_tip)
    grasp_center = (finger_a_tip + finger_b_tip) / 2.0
    jaw_base_center = (finger_a_base + finger_b_base) / 2.0
    object_axis_radius = obb_support_radius(rotation, half_extents, object_axis)

    finger_scalars = sorted(
        [
            float(np.dot(finger_a_tip - grasp_center, closing_axis)),
            float(np.dot(finger_b_tip - grasp_center, closing_axis)),
        ]
    )
    object_scalar = float(np.dot(object_center - grasp_center, closing_axis))
    object_closing_radius = obb_support_radius(rotation, half_extents, closing_axis)
    object_interval = [object_scalar - object_closing_radius, object_scalar + object_closing_radius]
    enclosure_margin = min(
        object_interval[0] - finger_scalars[0],
        finger_scalars[1] - object_interval[1],
    )
    between_probability = _soft_less_equal(-enclosure_margin, limit=0.002, scale=0.004)

    along_offset = float(np.dot(grasp_center - object_center, object_axis))
    along_limit = max(0.025, min(0.065, 0.65 * object_axis_radius))
    along_probability = _soft_less_equal(abs(along_offset), limit=along_limit, scale=0.008)

    vertical_offset = float(grasp_center[2] - object_center[2])
    height_probability = _soft_less_equal(abs(vertical_offset), limit=0.018, scale=0.004)

    axis_angle = math.degrees(
        math.acos(float(np.clip(abs(np.dot(closing_axis, object_axis)), 0.0, 1.0)))
    )
    perpendicular_error = abs(90.0 - axis_angle)
    perpendicular_probability = _soft_less_equal(perpendicular_error, limit=15.0, scale=4.0)

    nodes = [
        _node("gripper.jaw_center", "jaw_base_center", jaw_base_center, "robot_kinematics", world_state_version),
        _node("gripper.finger_a_base", "finger_base", finger_a_base, "robot_kinematics", world_state_version),
        _node("gripper.finger_b_base", "finger_base", finger_b_base, "robot_kinematics", world_state_version),
        _node("gripper.finger_a_inner_tip", "finger_inner_tip", finger_a_tip, "robot_kinematics", world_state_version),
        _node("gripper.finger_b_inner_tip", "finger_inner_tip", finger_b_tip, "robot_kinematics", world_state_version),
        _node("gripper.grasp_center", "grasp_center", grasp_center, "robot_kinematics", world_state_version),
        _node("object.center", "object_center", object_center, "oracle_object_pose", world_state_version),
        _node(
            "object.axis_start",
            "object_axis_endpoint",
            object_center - object_axis * object_axis_radius,
            "oracle_obb",
            world_state_version,
        ),
        _node(
            "object.axis_end",
            "object_axis_endpoint",
            object_center + object_axis * object_axis_radius,
            "oracle_obb",
            world_state_version,
        ),
        _node(
            "support.plane_anchor",
            "support_plane_anchor",
            np.array([object_center[0], object_center[1], geometry.support_z], dtype=np.float64),
            "oracle_support_plane",
            world_state_version,
        ),
    ]
    edges = [
        _edge(
            "object_between_fingers",
            "object.center",
            "gripper.grasp_center",
            "between_fingers",
            between_probability,
            world_state_version,
            {
                "finger_interval_m": _vec(finger_scalars),
                "object_interval_m": _vec(object_interval),
                "enclosure_margin_m": round(enclosure_margin, 6),
            },
        ),
        _edge(
            "grasp_region_along_object_axis",
            "gripper.grasp_center",
            "object.center",
            "within_grasp_region",
            along_probability,
            world_state_version,
            {
                "offset_along_object_axis_m": round(along_offset, 6),
                "allowed_abs_offset_m": round(along_limit, 6),
            },
        ),
        _edge(
            "grasp_height_aligned",
            "gripper.grasp_center",
            "object.center",
            "height_aligned",
            height_probability,
            world_state_version,
            {
                "vertical_offset_m": round(vertical_offset, 6),
                "allowed_abs_offset_m": 0.018,
            },
        ),
        _edge(
            "closing_axis_perpendicular_to_object_axis",
            "gripper.grasp_center",
            "object.center",
            "axis_perpendicular",
            perpendicular_probability,
            world_state_version,
            {
                "axis_angle_deg": round(axis_angle, 6),
                "perpendicular_error_deg": round(perpendicular_error, 6),
                "allowed_error_deg": 15.0,
            },
        ),
    ]

    probabilities = {edge["id"]: float(edge["probability"]) for edge in edges}
    failed = [edge_id for edge_id, probability in probabilities.items() if probability <= 0.2]
    uncertain = [edge_id for edge_id, probability in probabilities.items() if 0.2 < probability < 0.8]
    if failed:
        verdict = "adjust"
        confidence = max(1.0 - probabilities[edge_id] for edge_id in failed)
        summary = "One or more pre-grasp spatial requirements are violated."
    elif uncertain:
        verdict = "uncertain"
        confidence = min(probabilities.values())
        summary = "The pre-grasp geometry is plausible, but at least one relation is not decisive."
    else:
        verdict = "execute"
        confidence = min(probabilities.values())
        summary = "The target is enclosed, centered on the grasp region, height-aligned, and correctly oriented."

    return {
        "schema_version": "phase2.verify_pregrasp.oracle.v1",
        "access": "oracle/training_only",
        "query": "verify_pregrasp",
        "world_state_version": int(world_state_version),
        "coordinate_frame": "world",
        "units": {"position": "m", "angle": "deg"},
        "verdict": verdict,
        "confidence": round(float(confidence), 6),
        "summary": summary,
        "failed_relations": failed,
        "uncertain_relations": uncertain,
        "nodes": nodes,
        "edges": edges,
        "query_axes": {
            "closing_axis_world": _vec(closing_axis),
            "object_axis_world": _vec(object_axis),
            "support_normal_world": [0.0, 0.0, 1.0],
        },
        "kinematic_calibration": {
            "finger_tip_local_offset_m": [0.06, 0.0, 0.0],
            "source": "aloha-agilex collision mesh and URDF",
        },
    }


def score_pregrasp_candidate_views(
    *,
    edge_uncertainty: Mapping[str, float],
    query_axes: Mapping[str, Iterable[float]],
    current_view_direction_world: np.ndarray,
    current_camera_position_world: np.ndarray,
    candidates: Iterable[Mapping[str, Any]],
    visited_views: Iterable[str] = (),
) -> list[dict[str, Any]]:
    visited = set(visited_views)
    closing = unit(np.asarray(query_axes["closing_axis_world"], dtype=np.float64))
    vertical = unit(np.asarray(query_axes.get("support_normal_world", [0.0, 0.0, 1.0]), dtype=np.float64))
    if "object_axis_world" in query_axes:
        object_axis = unit(np.asarray(query_axes["object_axis_world"], dtype=np.float64))
    else:
        # Query-template prior for a horizontal elongated target. This is used
        # only to choose a diagnostic view until both visual axis endpoints are
        # observed; it is never inserted as measured object geometry.
        object_axis = unit(np.cross(vertical, closing))
    current_direction = unit(np.asarray(current_view_direction_world, dtype=np.float64))
    current_position = np.asarray(current_camera_position_world, dtype=np.float64)
    rows = []
    for candidate in candidates:
        view = str(candidate["view"])
        if view in visited:
            continue
        direction = unit(np.asarray(candidate["view_direction_world"], dtype=np.float64))
        position = np.asarray(candidate["camera_position_world"], dtype=np.float64)
        if (
            float(np.linalg.norm(position - current_position)) <= 1e-5
            and math.acos(float(np.clip(np.dot(direction, current_direction), -1.0, 1.0))) <= 1e-5
        ):
            continue
        observability = {
            "closing": 1.0 - abs(float(np.dot(direction, closing))),
            "object": 1.0 - abs(float(np.dot(direction, object_axis))),
            "vertical": 1.0 - abs(float(np.dot(direction, vertical))),
        }
        numerator = 0.0
        denominator = 0.0
        per_relation = {}
        for edge_id, spec in PREGRASP_RELATION_SPECS.items():
            axes = tuple(spec["axes"])
            score = min(observability[axis] for axis in axes)
            uncertainty = float(np.clip(edge_uncertainty.get(edge_id, 1.0), 0.0, 1.0))
            weight = float(spec["weight"]) * uncertainty
            numerator += weight * score
            denominator += weight
            per_relation[edge_id] = round(score, 6)
        relation_score = numerator / max(denominator, 1e-9)
        position_delta = float(np.linalg.norm(position - current_position))
        angle = math.acos(float(np.clip(np.dot(direction, current_direction), -1.0, 1.0)))
        move_cost = float(np.clip(0.7 * position_delta / 0.8 + 0.3 * angle / math.pi, 0.0, 1.0))
        baseline = float(np.clip(angle / (math.pi / 2.0), 0.0, 1.0))
        framing = float(np.clip(candidate.get("framing_score", 1.0), 0.0, 1.0))
        utility = 0.72 * relation_score + 0.13 * baseline + 0.15 * framing - 0.15 * move_cost
        rows.append(
            {
                "view": view,
                "predicted": {
                    "utility": round(utility, 6),
                    "relation_score": round(relation_score, 6),
                    "move_cost": round(move_cost, 6),
                    "baseline_score": round(baseline, 6),
                    "framing_score": round(framing, 6),
                    "per_relation_observability": per_relation,
                },
                "camera_position_world_m": _vec(position),
                "view_direction_world": _vec(direction),
            }
        )
    return sorted(rows, key=lambda row: row["predicted"]["utility"], reverse=True)


def transform_local_point(pose_matrix: np.ndarray, local_point: Iterable[float]) -> np.ndarray:
    pose_matrix = np.asarray(pose_matrix, dtype=np.float64)
    local = np.asarray(tuple(local_point), dtype=np.float64)
    return pose_matrix[:3, :3] @ local + pose_matrix[:3, 3]


def _node(node_id: str, semantic_type: str, position: np.ndarray, source: str, version: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "semantic_type": semantic_type,
        "position_mean_world_m": _vec(position),
        "position_covariance_m2": (np.eye(3) * 1e-8).tolist(),
        "source": source,
        "access": "oracle/training_only",
        "valid_for_world_state": int(version),
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    probability: float,
    version: int,
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    probability = float(np.clip(probability, 0.0, 1.0))
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "state": probability_state(probability),
        "probability": round(probability, 6),
        "uncertainty": round(min(probability, 1.0 - probability), 6),
        "measurement": dict(measurement),
        "evidence_source": "oracle_kinematics_and_object_geometry",
        "valid_for_world_state": int(version),
    }


def _soft_less_equal(value: float, *, limit: float, scale: float) -> float:
    exponent = (float(value) - float(limit)) / max(float(scale), 1e-9)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(exponent)))


def _vec(values: Any) -> list[float]:
    return [round(float(value), 6) for value in np.asarray(values, dtype=np.float64).reshape(-1)]
