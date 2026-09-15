"""Cross-fitted realized graph-error labels for active view selection."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

import numpy as np

from .expert_event_candidate_dataset import event_intent_from_sample
from .expert_event_view_dataset import RELATION_NAMES


REALIZED_EVENT_VIEW_SAMPLE_SCHEMA = "spatial.realized_event_view_ranking_sample.v1"
UNCERTAINTY_SCALES = {
    "grasp_center_location": 0.03,
    "left_contact": 0.03,
    "right_contact": 0.03,
    "opening_width": 0.02,
    "approach_axis": math.radians(30.0),
    "closing_axis": math.radians(45.0),
}


def build_realized_event_view_sample(
    event: Mapping[str, Any],
    realized_row: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(event["sample_id"])
    if realized_row.get("sample_id") != sample_id:
        raise ValueError("realized view row references a different event sample")
    graphs = realized_row["per_view_prediction_graphs"]
    current_graph = graphs["current"]
    frame_node = _frame_node(current_graph)
    axes = frame_node["attributes"]
    closing = _unit(axes["closing_axis_world"])
    approach = _unit(axes["approach_axis_world"])
    uncertainty = relation_uncertainty_from_grasp_frame_graph(current_graph)
    supervision = {
        str(row["view"]): row for row in event["training_only"]["view_supervision"]
    }
    current_direction = _unit(supervision["current"]["view_direction_world"])
    labels = []
    candidates = []
    for realized_label in realized_row["candidate_view_labels"]:
        view = str(realized_label["view"])
        direction = _unit(supervision[view]["view_direction_world"])
        dot = float(np.clip(np.dot(current_direction, direction), -1.0, 1.0))
        input_features = {
            "current_relation_uncertainty": uncertainty,
            "candidate_closing_axis_observability": float(
                np.clip(1.0 - abs(float(np.dot(direction, closing))), 0.0, 1.0)
            ),
            "candidate_approach_axis_observability": float(
                np.clip(1.0 - abs(float(np.dot(direction, approach))), 0.0, 1.0)
            ),
            "candidate_view_direction_world": np.round(direction, 8).tolist(),
            "angular_diversity": round(float(1.0 - abs(dot)), 8),
            "move_cost": round(float(math.acos(dot) / math.pi), 8),
        }
        target = realized_label["oracle_evaluation_only"]
        labels.append(
            {
                "view": view,
                "input_features": input_features,
                "target": {
                    "relation_realized_error_reduction": deepcopy(
                        target["relation_error_reduction"]
                    ),
                    "normalized_composite_error_reduction": float(
                        target["normalized_composite_error_reduction"]
                    ),
                    "move_cost_penalty": float(
                        target["normalized_composite_error_reduction"]
                        - target["realized_utility"]
                    ),
                    "oracle_utility": float(target["realized_utility"]),
                },
                "access": "oracle/training_only",
            }
        )
        candidates.append(
            {
                "view": view,
                "view_direction_world": input_features[
                    "candidate_view_direction_world"
                ],
                "move_cost": input_features["move_cost"],
                "camera_position_world_m": _optional_vector3(
                    supervision[view].get("camera_position_world_m")
                ),
                "camera_pose_source": supervision[view].get("camera_pose_source"),
                "camera_layout_id": supervision[view].get("camera_layout_id"),
            }
        )
    labels.sort(key=lambda row: row["view"])
    candidates.sort(key=lambda row: row["view"])
    best = max(labels, key=lambda row: float(row["target"]["oracle_utility"]))
    result = {
        "schema_version": REALIZED_EVENT_VIEW_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "task": str(event["task"]),
        "seed": int(event["seed"]),
        "world_state_version": int(event.get("world_state_version", 0)),
        "inference_visible": {
            "intent": event_intent_from_sample(event).as_dict(),
            "current_view": "current",
            "current_observation": deepcopy(
                next(
                    row
                    for row in event["inference_visible"]["observations"]
                    if row["view"] == "current"
                )
            ),
            "current_prediction_graph": deepcopy(current_graph),
            "candidate_views": candidates,
        },
        "training_only": {
            "access": "oracle/training_only",
            "view_labels": labels,
            "oracle_next_action": {
                "action": "acquire_view",
                "view": best["view"],
                "utility": best["target"]["oracle_utility"],
            },
            "label_model": {
                "type": "cross_fitted_realized_learned_grasp_frame_error_reduction",
                "realized_learned_model_error_reduction": True,
                "simulator_truth_used_only_for_target_error": True,
                "input_uncertainty_source": "current_learned_rgbd_grasp_frame_graph",
            },
        },
    }
    errors = validate_realized_event_view_sample(result)
    if errors:
        raise ValueError("invalid realized event view sample: " + "; ".join(errors))
    return result


def relation_uncertainty_from_grasp_frame_graph(
    graph: Mapping[str, Any],
) -> dict[str, float]:
    points = {
        str(node["semantic_type"]): node
        for node in graph["nodes"]
        if node.get("node_type") == "grasp_frame_point"
    }
    point_std = {
        name: float(
            np.sqrt(
                np.trace(np.asarray(points[name]["position_covariance_m2"])) / 3.0
            )
        )
        for name in ("grasp_center", "left_contact", "right_contact")
    }
    frame = _frame_node(graph)
    orientation_covariance = np.asarray(
        frame["attributes"]["orientation_covariance_rad2"], dtype=np.float64
    )
    orientation_std = float(np.sqrt(np.trace(orientation_covariance) / 3.0))
    opening_std = math.sqrt(
        point_std["left_contact"] ** 2 + point_std["right_contact"] ** 2
    )
    raw = {
        "grasp_center_location": point_std["grasp_center"],
        "left_contact": point_std["left_contact"],
        "right_contact": point_std["right_contact"],
        "opening_width": opening_std,
        "approach_axis": orientation_std,
        "closing_axis": orientation_std,
    }
    return {
        name: round(float(value / (value + UNCERTAINTY_SCALES[name])), 8)
        for name, value in raw.items()
    }


def validate_realized_event_view_sample(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != REALIZED_EVENT_VIEW_SAMPLE_SCHEMA:
        errors.append("unsupported realized event view sample schema")
    inference = value.get("inference_visible")
    training = value.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        return errors + ["realized sample requires inference and training payloads"]
    if training.get("access") != "oracle/training_only":
        errors.append("realized targets must remain training-only")
    forbidden = {
        "view_labels",
        "oracle_next_action",
        "oracle_utility",
        "relation_realized_error_reduction",
        "normalized_composite_error_reduction",
    }
    leaked = sorted(_recursive_keys(inference).intersection(forbidden))
    if leaked:
        errors.append("realized targets leaked into inference_visible: " + ", ".join(leaked))
    labels = training.get("view_labels")
    candidates = inference.get("candidate_views")
    if not isinstance(labels, list) or not labels:
        errors.append("realized sample requires view labels")
    elif not isinstance(candidates, list) or {
        str(row.get("view")) for row in labels
    } != {str(row.get("view")) for row in candidates}:
        errors.append("realized labels must align with candidate views")
    else:
        for row in labels:
            uncertainty = row.get("input_features", {}).get(
                "current_relation_uncertainty", {}
            )
            reduction = row.get("target", {}).get(
                "relation_realized_error_reduction", {}
            )
            if set(uncertainty) != set(RELATION_NAMES):
                errors.append("realized input uncertainty schema mismatch")
            if set(reduction) != set(RELATION_NAMES):
                errors.append("realized relation reduction schema mismatch")
    return errors


def _frame_node(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = [node for node in graph["nodes"] if node.get("node_type") == "grasp_frame"]
    if len(rows) != 1:
        raise ValueError("prediction graph must have exactly one grasp frame node")
    return rows[0]


def _unit(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or norm <= 1e-9:
        raise ValueError("view ranker geometry requires a nonzero 3-vector")
    return result / norm


def _optional_vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("optional camera position must be a finite 3-vector")
    return np.round(result, 8).tolist()


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = set(str(key) for key in value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()
