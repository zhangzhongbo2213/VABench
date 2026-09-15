from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class VerifyGraspGeometry:
    """Training-only geometry used to validate the verify_grasp graph template."""

    finger_a: np.ndarray
    finger_b: np.ndarray
    object_center: np.ndarray
    object_rotation: np.ndarray
    object_half_extents: np.ndarray
    object_axis: np.ndarray
    support_z: float
    object_start_center_z: float
    finger_a_contact: bool
    finger_b_contact: bool
    support_contact: bool
    gripper_closed: bool
    finger_a_contact_point: np.ndarray | None = None
    finger_b_contact_point: np.ndarray | None = None
    object_displacement: np.ndarray | None = None
    gripper_displacement: np.ndarray | None = None


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        raise ValueError("cannot normalize a near-zero vector")
    return value / norm


def probability_state(probability: float, *, threshold: float = 0.5) -> str:
    return "true" if probability >= threshold else "false"


def closest_point_on_obb(
    point: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Return the closest surface point on an oriented box."""

    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)
    local = rotation.T @ (point - center)
    clipped = np.clip(local, -half_extents, half_extents)
    if np.all(np.abs(local) <= half_extents + EPS):
        face_axis = int(np.argmin(half_extents - np.abs(local)))
        sign = 1.0 if local[face_axis] >= 0.0 else -1.0
        clipped[face_axis] = sign * half_extents[face_axis]
    return center + rotation @ clipped


def obb_support_radius(
    rotation: np.ndarray,
    half_extents: np.ndarray,
    direction: np.ndarray,
) -> float:
    local_direction = np.asarray(rotation, dtype=np.float64).T @ unit(direction)
    return float(np.dot(np.abs(local_direction), np.asarray(half_extents, dtype=np.float64)))


def build_verify_grasp_oracle_graph(
    geometry: VerifyGraspGeometry,
    *,
    world_state_version: int,
) -> dict[str, Any]:
    """Build a sparse oracle graph without retaining a scene point cloud.

    The output is intentionally marked training-only. In a learned system the same
    schema is populated from predicted keypoints and calibrated relation heads.
    """

    finger_a = np.asarray(geometry.finger_a, dtype=np.float64)
    finger_b = np.asarray(geometry.finger_b, dtype=np.float64)
    center = np.asarray(geometry.object_center, dtype=np.float64)
    rotation = np.asarray(geometry.object_rotation, dtype=np.float64)
    half_extents = np.asarray(geometry.object_half_extents, dtype=np.float64)
    closing_axis = unit(finger_b - finger_a)
    jaw_center = (finger_a + finger_b) / 2.0
    object_axis = unit(geometry.object_axis)
    axis_radius = obb_support_radius(rotation, half_extents, object_axis)
    surface_a = (
        np.asarray(geometry.finger_a_contact_point, dtype=np.float64)
        if geometry.finger_a_contact_point is not None
        else closest_point_on_obb(finger_a, center, rotation, half_extents)
    )
    surface_b = (
        np.asarray(geometry.finger_b_contact_point, dtype=np.float64)
        if geometry.finger_b_contact_point is not None
        else closest_point_on_obb(finger_b, center, rotation, half_extents)
    )

    finger_scalars = sorted(
        [float(np.dot(finger_a - jaw_center, closing_axis)), float(np.dot(finger_b - jaw_center, closing_axis))]
    )
    object_scalar = float(np.dot(center - jaw_center, closing_axis))
    object_radius = obb_support_radius(rotation, half_extents, closing_axis)
    object_interval = [object_scalar - object_radius, object_scalar + object_radius]
    enclosure_margin = min(
        object_interval[0] - finger_scalars[0],
        finger_scalars[1] - object_interval[1],
    )
    between_probability = _soft_threshold(enclosure_margin, threshold=-0.002, scale=0.004)

    lifted_delta = float(center[2] - geometry.object_start_center_z)
    lifted = lifted_delta > 0.02 and not geometry.support_contact
    lifted_probability = 1.0 if lifted else 0.0
    contact_a_probability = 1.0 if geometry.finger_a_contact else 0.0
    contact_b_probability = 1.0 if geometry.finger_b_contact else 0.0
    supported_probability = 1.0 if geometry.support_contact else 0.0

    motion_probability: float | None = None
    motion_measurement: dict[str, Any]
    if geometry.object_displacement is None or geometry.gripper_displacement is None:
        motion_measurement = {"reason": "requires observations before and after gripper motion"}
    else:
        object_displacement = np.asarray(geometry.object_displacement, dtype=np.float64)
        gripper_displacement = np.asarray(geometry.gripper_displacement, dtype=np.float64)
        motion_residual = float(np.linalg.norm(object_displacement - gripper_displacement))
        object_motion = float(np.linalg.norm(object_displacement))
        gripper_motion = float(np.linalg.norm(gripper_displacement))
        sufficient_motion = _soft_threshold(min(object_motion, gripper_motion), threshold=0.02, scale=0.005)
        rigid_agreement = 1.0 - _soft_threshold(motion_residual, threshold=0.012, scale=0.004)
        motion_probability = float(np.clip(sufficient_motion * rigid_agreement, 0.0, 1.0))
        motion_measurement = {
            "object_displacement_world_m": _vec(object_displacement),
            "gripper_displacement_world_m": _vec(gripper_displacement),
            "displacement_residual_m": round(motion_residual, 6),
            "object_motion_m": round(object_motion, 6),
            "gripper_motion_m": round(gripper_motion, 6),
        }

    nodes = [
        _node("gripper.jaw_center", "jaw_center", jaw_center, "robot_kinematics", world_state_version),
        _node(
            "gripper.finger_a_inner",
            "finger_inner",
            finger_a,
            "robot_kinematics",
            world_state_version,
        ),
        _node(
            "gripper.finger_b_inner",
            "finger_inner",
            finger_b,
            "robot_kinematics",
            world_state_version,
        ),
        _node("object.center", "object_center", center, "oracle_object_pose", world_state_version),
        _node("object.surface_a", "grasp_surface", surface_a, "oracle_obb", world_state_version),
        _node("object.surface_b", "grasp_surface", surface_b, "oracle_obb", world_state_version),
        _node(
            "object.axis_start",
            "object_axis_endpoint",
            center - object_axis * axis_radius,
            "oracle_obb",
            world_state_version,
        ),
        _node(
            "object.axis_end",
            "object_axis_endpoint",
            center + object_axis * axis_radius,
            "oracle_obb",
            world_state_version,
        ),
        _node(
            "support.plane_anchor",
            "support_plane_anchor",
            np.array([center[0], center[1], geometry.support_z], dtype=np.float64),
            "oracle_support_plane",
            world_state_version,
        ),
    ]

    edges = [
        _edge(
            "object_between_fingers",
            "object.center",
            "gripper.jaw_center",
            "between_fingers",
            between_probability,
            world_state_version,
            measurement={
                "closing_axis_world": _vec(closing_axis),
                "finger_interval_m": _vec(finger_scalars),
                "object_interval_m": _vec(object_interval),
                "enclosure_margin_m": round(enclosure_margin, 6),
            },
        ),
        _edge(
            "finger_a_contact",
            "gripper.finger_a_inner",
            "object.surface_a",
            "contact",
            contact_a_probability,
            world_state_version,
            measurement={
                "contact_point_world_m": _vec(surface_a),
                "contact_point_source": (
                    "oracle_physics_contact_centroid"
                    if geometry.finger_a_contact_point is not None
                    else "oracle_obb_closest_point"
                ),
            },
        ),
        _edge(
            "finger_b_contact",
            "gripper.finger_b_inner",
            "object.surface_b",
            "contact",
            contact_b_probability,
            world_state_version,
            measurement={
                "contact_point_world_m": _vec(surface_b),
                "contact_point_source": (
                    "oracle_physics_contact_centroid"
                    if geometry.finger_b_contact_point is not None
                    else "oracle_obb_closest_point"
                ),
            },
        ),
        _edge(
            "supported_by_table",
            "object.center",
            "support.plane_anchor",
            "supported_by",
            supported_probability,
            world_state_version,
            measurement={"center_height_above_plane_m": round(float(center[2] - geometry.support_z), 6)},
        ),
        _edge(
            "lifted_from_support",
            "object.center",
            "support.plane_anchor",
            "lifted_from_support",
            lifted_probability,
            world_state_version,
            measurement={"center_height_change_m": round(lifted_delta, 6)},
        ),
        (
            {
                "id": "moves_with_gripper",
                "source": "object.center",
                "target": "gripper.jaw_center",
                "relation": "moves_rigidly_with",
                "state": "unknown",
                "probability": None,
                "uncertainty": 1.0,
                "measurement": motion_measurement,
                "evidence_source": "not_observed",
                "valid_for_world_state": world_state_version,
            }
            if motion_probability is None
            else _edge(
                "moves_with_gripper",
                "object.center",
                "gripper.jaw_center",
                "moves_rigidly_with",
                motion_probability,
                world_state_version,
                measurement=motion_measurement,
            )
        ),
    ]

    dual_contact = geometry.finger_a_contact and geometry.finger_b_contact
    gripper_closed = bool(geometry.gripper_closed)
    enclosed = between_probability >= 0.5
    moves_with_gripper = motion_probability is not None and motion_probability >= 0.5
    if enclosed and dual_contact and gripper_closed and lifted and (
        motion_probability is None or moves_with_gripper
    ):
        verdict = "true"
        confidence = 0.98
        missing = [] if motion_probability is not None else ["moves_with_gripper"]
        summary = (
            "The object is enclosed, has bilateral contact, is lifted from the support, "
            "and moves with the gripper."
            if moves_with_gripper
            else "The object is enclosed, has bilateral contact, and is lifted from the support."
        )
    elif not enclosed or not dual_contact or not gripper_closed:
        verdict = "false"
        confidence = 0.97
        missing = []
        summary = "The grasp is not established because enclosure, bilateral contact, or closure is absent."
    elif lifted and motion_probability is not None and not moves_with_gripper:
        verdict = "false"
        confidence = 0.95
        missing = []
        summary = "The object was lifted, but its motion is inconsistent with rigid gripper motion."
    else:
        verdict = "uncertain"
        confidence = 0.72
        missing = ["lifted_from_support", "moves_with_gripper"]
        summary = (
            "The object is enclosed with bilateral contact, but the pre-lift state cannot yet "
            "verify that it will move rigidly with the gripper."
        )

    return {
        "schema_version": "phase1.verify_grasp.oracle.v1",
        "access": "oracle/training_only",
        "query": "verify_grasp",
        "world_state_version": world_state_version,
        "coordinate_frame": "world",
        "units": {"position": "m", "angle": "deg"},
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "missing_evidence": missing,
        "entity_state": {"gripper": {"closed": gripper_closed}},
        "nodes": nodes,
        "edges": edges,
        "query_axes": {
            "closing_axis_world": _vec(closing_axis),
            "object_axis_world": _vec(object_axis),
            "support_normal_world": [0.0, 0.0, 1.0],
        },
    }


def score_candidate_view(
    *,
    view_direction_world: np.ndarray,
    current_view_direction_world: np.ndarray,
    closing_axis_world: np.ndarray,
    edge_uncertainty: Mapping[str, float],
    framing_score: float,
    move_cost: float,
    relation_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Rule-based Phase 1 utility for a candidate discrete view.

    A relation is easiest to measure when its diagnostic axis lies in the image
    plane. The score does not inspect the candidate image; realized visibility is
    recorded separately for oracle analysis.
    """

    direction = unit(view_direction_world)
    current_direction = unit(current_view_direction_world)
    closing_axis = unit(closing_axis_world)
    vertical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    closing_observability = 1.0 - abs(float(np.dot(direction, closing_axis)))
    support_observability = 1.0 - abs(float(np.dot(direction, vertical_axis)))
    relation_weights = dict(
        relation_weights
        or {
            "object_between_fingers": 1.0,
            "finger_a_contact": 1.2,
            "finger_b_contact": 1.2,
            "lifted_from_support": 0.8,
        }
    )
    relation_observability = {
        "object_between_fingers": closing_observability,
        "finger_a_contact": closing_observability,
        "finger_b_contact": closing_observability,
        "lifted_from_support": support_observability,
    }
    numerator = 0.0
    denominator = 0.0
    for edge_id, weight in relation_weights.items():
        uncertainty = float(np.clip(edge_uncertainty.get(edge_id, 0.0), 0.0, 1.0))
        weighted_uncertainty = weight * uncertainty
        numerator += weighted_uncertainty * relation_observability[edge_id]
        denominator += weighted_uncertainty
    relation_score = numerator / denominator if denominator > EPS else 0.0
    view_angle = math.degrees(
        math.acos(float(np.clip(np.dot(direction, current_direction), -1.0, 1.0)))
    )
    baseline_score = float(np.clip(view_angle / 90.0, 0.0, 1.0))
    framing_score = float(np.clip(framing_score, 0.0, 1.0))
    move_cost = float(np.clip(move_cost, 0.0, 1.0))
    utility = 0.65 * relation_score + 0.20 * baseline_score + 0.15 * framing_score - 0.15 * move_cost
    return {
        "utility": round(float(utility), 6),
        "relation_score": round(float(relation_score), 6),
        "closing_axis_observability": round(float(closing_observability), 6),
        "support_axis_observability": round(float(support_observability), 6),
        "baseline_score": round(baseline_score, 6),
        "framing_score": round(framing_score, 6),
        "move_cost": round(move_cost, 6),
    }


def graph_node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(node["id"]): node for node in graph["nodes"]}


def _node(
    node_id: str,
    semantic_type: str,
    position: np.ndarray,
    source: str,
    version: int,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "semantic_type": semantic_type,
        "position_mean_world_m": _vec(position),
        "position_covariance_m2": [
            [1e-8, 0.0, 0.0],
            [0.0, 1e-8, 0.0],
            [0.0, 0.0, 1e-8],
        ],
        "source": source,
        "access": "oracle/training_only",
        "valid_for_world_state": version,
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    probability: float,
    version: int,
    *,
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
        "evidence_source": "oracle_physics_or_geometry",
        "valid_for_world_state": version,
    }


def _soft_threshold(value: float, *, threshold: float, scale: float) -> float:
    exponent = -(value - threshold) / max(scale, EPS)
    exponent = float(np.clip(exponent, -60.0, 60.0))
    return float(1.0 / (1.0 + math.exp(exponent)))


def _vec(values: Any) -> list[float]:
    return [round(float(value), 6) for value in np.asarray(values, dtype=np.float64).reshape(-1)]
