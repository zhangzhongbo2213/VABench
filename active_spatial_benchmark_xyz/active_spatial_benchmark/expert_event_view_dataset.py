"""Relation-level next-view labels for open-vocabulary grasp events."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .expert_grasp_trace import validate_expert_grasp_event_sample
from .expert_event_candidate_dataset import event_intent_from_sample


EXPERT_EVENT_VIEW_SAMPLE_SCHEMA = "spatial.expert_event_view_ranking_sample.v1"
RELATION_SPECS = {
    "grasp_center_location": {"weight": 1.0, "axis": "spatial"},
    "left_contact": {"weight": 1.2, "axis": "closing"},
    "right_contact": {"weight": 1.2, "axis": "closing"},
    "opening_width": {"weight": 1.1, "axis": "closing"},
    "approach_axis": {"weight": 0.8, "axis": "approach"},
    "closing_axis": {"weight": 0.9, "axis": "closing"},
}
RELATION_NAMES = tuple(RELATION_SPECS)
DEFAULT_MEASUREMENT_PRECISION = 4.0
DEFAULT_MOVE_COST_WEIGHT = 0.08


def build_expert_event_view_ranking_sample(
    sample: Mapping[str, Any],
    *,
    current_view: str = "current",
    measurement_precision: float = DEFAULT_MEASUREMENT_PRECISION,
    move_cost_weight: float = DEFAULT_MOVE_COST_WEIGHT,
    minimum_view_utility: float = 0.01,
    max_additional_views: int = 2,
) -> dict[str, Any]:
    """Create oracle expected relation-information-gain labels.

    This is a declared Stage-0 sensor model.  It uses training-only simulator
    visibility to define measurement precision, while inference features use
    only current graph uncertainty, task intent and candidate camera geometry.
    """

    errors = validate_expert_grasp_event_sample(sample)
    if errors:
        raise ValueError("invalid expert grasp event sample: " + "; ".join(errors))
    if (
        measurement_precision <= 0.0
        or move_cost_weight < 0.0
        or max_additional_views < 1
    ):
        raise ValueError("invalid view information-gain settings")
    supervision = sample["training_only"].get("view_supervision", ())
    row_by_view = {str(row.get("view")): row for row in supervision}
    if len(row_by_view) != len(supervision) or current_view not in row_by_view:
        raise ValueError("view supervision must contain unique views and the current view")
    if len(row_by_view) < 2:
        raise ValueError("view ranking requires at least one additional view")
    current = row_by_view[current_view]
    current_evidence = relation_evidence_from_supervision(current)
    current_uncertainty = {
        relation: _posterior_variance(
            measurement_precision * current_evidence[relation]
        )
        for relation in RELATION_NAMES
    }
    current_direction = _unit(current.get("view_direction_world"), "current direction")
    labels = []
    candidate_views = []
    for view, candidate in row_by_view.items():
        if view == current_view:
            continue
        candidate_direction = _unit(
            candidate.get("view_direction_world"), f"{view} direction"
        )
        dot = float(np.clip(np.dot(current_direction, candidate_direction), -1.0, 1.0))
        move_cost = float(math.acos(dot) / math.pi)
        angular_diversity = float(1.0 - abs(dot))
        complementarity = 0.4 + 0.6 * angular_diversity
        candidate_evidence = relation_evidence_from_supervision(candidate)
        relation_gain = {}
        posterior_uncertainty = {}
        weighted_gain = 0.0
        weight_sum = 0.0
        for relation, spec in RELATION_SPECS.items():
            before = current_uncertainty[relation]
            additional_precision = (
                measurement_precision
                * candidate_evidence[relation]
                * complementarity
            )
            after = 1.0 / (1.0 / before + additional_precision)
            gain = 0.5 * math.log(max(before, 1e-12) / max(after, 1e-12))
            relation_gain[relation] = round(float(gain), 8)
            posterior_uncertainty[relation] = round(float(after), 8)
            weight = float(spec["weight"])
            weighted_gain += weight * gain
            weight_sum += weight
        information_gain = weighted_gain / max(weight_sum, 1e-12)
        utility = information_gain - move_cost_weight * move_cost
        input_features = {
            "current_relation_uncertainty": {
                key: round(float(value), 8)
                for key, value in current_uncertainty.items()
            },
            "candidate_closing_axis_observability": float(
                candidate["closing_axis_observability"]
            ),
            "candidate_approach_axis_observability": float(
                candidate["approach_axis_observability"]
            ),
            "candidate_view_direction_world": np.round(candidate_direction, 8).tolist(),
            "angular_diversity": round(angular_diversity, 8),
            "move_cost": round(move_cost, 8),
        }
        candidate_position = _optional_vector3(candidate.get("camera_position_world_m"))
        if candidate_position is not None:
            input_features["candidate_camera_position_world_m"] = candidate_position
        if candidate.get("camera_pose_source") is not None:
            input_features["candidate_camera_pose_source"] = str(
                candidate["camera_pose_source"]
            )
        if candidate.get("camera_layout_id") is not None:
            input_features["candidate_camera_layout_id"] = str(
                candidate["camera_layout_id"]
            )
        candidate_views.append(
            {
                "view": view,
                "view_direction_world": input_features[
                    "candidate_view_direction_world"
                ],
                "move_cost": input_features["move_cost"],
                "camera_position_world_m": candidate_position,
                "camera_pose_source": candidate.get("camera_pose_source"),
                "camera_layout_id": candidate.get("camera_layout_id"),
            }
        )
        labels.append(
            {
                "view": view,
                "input_features": input_features,
                "target": {
                    "relation_information_gain_nats": relation_gain,
                    "posterior_relation_uncertainty": posterior_uncertainty,
                    "oracle_information_gain_nats": round(information_gain, 8),
                    "move_cost_penalty": round(move_cost_weight * move_cost, 8),
                    "oracle_utility": round(utility, 8),
                    "oracle_relation_evidence": {
                        key: round(float(value), 8)
                        for key, value in candidate_evidence.items()
                    },
                },
                "access": "oracle/training_only",
            }
        )
    labels.sort(key=lambda row: str(row["view"]))
    best = max(labels, key=lambda row: float(row["target"]["oracle_utility"]))
    best_utility = float(best["target"]["oracle_utility"])
    intent = event_intent_from_sample(sample).as_dict()
    result = {
        "schema_version": EXPERT_EVENT_VIEW_SAMPLE_SCHEMA,
        "sample_id": str(sample.get("sample_id", "unknown")),
        "task": str(sample.get("task", "unknown")),
        "seed": int(sample.get("seed", 0)),
        "world_state_version": int(sample.get("world_state_version", 0)),
        "inference_visible": {
            "intent": intent,
            "current_view": current_view,
            "current_observation": deepcopy(_observation_for_view(sample, current_view)),
            "candidate_views": sorted(candidate_views, key=lambda row: row["view"]),
        },
        "training_only": {
            "access": "oracle/training_only",
            "view_labels": labels,
            "oracle_next_action": {
                "action": "acquire_view" if best_utility > minimum_view_utility else "stop",
                "view": best["view"] if best_utility > minimum_view_utility else None,
                "utility": round(best_utility, 8),
                "minimum_view_utility": float(minimum_view_utility),
            },
            "oracle_active_sequence": build_oracle_active_view_sequence(
                supervision,
                current_view=current_view,
                measurement_precision=measurement_precision,
                move_cost_weight=move_cost_weight,
                minimum_view_utility=minimum_view_utility,
                max_additional_views=max_additional_views,
            ),
            "label_model": {
                "type": "gaussian_expected_relation_information_gain",
                "measurement_precision": float(measurement_precision),
                "move_cost_weight": float(move_cost_weight),
                "visibility_source": "simulator_projection_and_target_mask_training_only",
                "candidate_feature_source_at_deployment": "learned_rgbd_spatial_graph",
                "realized_learned_model_error_reduction": False,
            },
        },
    }
    validation_errors = validate_expert_event_view_ranking_sample(result)
    if validation_errors:
        raise ValueError("invalid expert event view sample: " + "; ".join(validation_errors))
    return result


def build_oracle_active_view_sequence(
    supervision: Sequence[Mapping[str, Any]],
    *,
    current_view: str = "current",
    measurement_precision: float = DEFAULT_MEASUREMENT_PRECISION,
    move_cost_weight: float = DEFAULT_MOVE_COST_WEIGHT,
    minimum_view_utility: float = 0.01,
    max_additional_views: int = 2,
) -> dict[str, Any]:
    """Greedily update relation precision after every selected frozen view."""

    row_by_view = {str(row.get("view")): row for row in supervision}
    if current_view not in row_by_view or max_additional_views < 1:
        raise ValueError("active view sequence requires current view and positive budget")
    current = row_by_view[current_view]
    precision = {
        relation: measurement_precision * evidence
        for relation, evidence in relation_evidence_from_supervision(current).items()
    }
    acquired = [current_view]
    remaining = set(row_by_view) - {current_view}
    latest_direction = _unit(current["view_direction_world"], "current direction")
    acquired_directions = [latest_direction]
    steps = []
    stop_reason = "view_budget_exhausted"
    while remaining and len(steps) < max_additional_views:
        candidates = []
        uncertainty_before = {
            relation: _posterior_variance(value)
            for relation, value in precision.items()
        }
        for view in sorted(remaining):
            row = row_by_view[view]
            direction = _unit(row["view_direction_world"], f"{view} direction")
            closest_direction_similarity = max(
                abs(float(np.dot(direction, acquired_direction)))
                for acquired_direction in acquired_directions
            )
            angular_diversity = 1.0 - closest_direction_similarity
            complementarity = 0.4 + 0.6 * angular_diversity
            move_dot = float(np.clip(np.dot(latest_direction, direction), -1.0, 1.0))
            move_cost = float(math.acos(move_dot) / math.pi)
            evidence = relation_evidence_from_supervision(row)
            relation_gain = {}
            posterior = {}
            weighted_gain = 0.0
            weight_sum = 0.0
            for relation, spec in RELATION_SPECS.items():
                before = uncertainty_before[relation]
                additional = measurement_precision * evidence[relation] * complementarity
                after = 1.0 / (1.0 / before + additional)
                gain = 0.5 * math.log(max(before, 1e-12) / max(after, 1e-12))
                relation_gain[relation] = round(float(gain), 8)
                posterior[relation] = round(float(after), 8)
                weighted_gain += float(spec["weight"]) * gain
                weight_sum += float(spec["weight"])
            information_gain = weighted_gain / max(weight_sum, 1e-12)
            utility = information_gain - move_cost_weight * move_cost
            candidates.append(
                {
                    "view": view,
                    "utility": round(float(utility), 8),
                    "information_gain_nats": round(float(information_gain), 8),
                    "move_cost": round(move_cost, 8),
                    "complementarity": round(complementarity, 8),
                    "relation_information_gain_nats": relation_gain,
                    "posterior_relation_uncertainty": posterior,
                    "oracle_relation_evidence": {
                        key: round(float(value), 8) for key, value in evidence.items()
                    },
                    "direction": direction,
                }
            )
        best = max(candidates, key=lambda row: float(row["utility"]))
        if float(best["utility"]) <= minimum_view_utility:
            stop_reason = "evidence_sufficient"
            break
        view = str(best["view"])
        evidence = best["oracle_relation_evidence"]
        complementarity = float(best["complementarity"])
        uncertainty_after = best["posterior_relation_uncertainty"]
        steps.append(
            {
                "step": len(steps) + 1,
                "selected_view": view,
                "utility": best["utility"],
                "information_gain_nats": best["information_gain_nats"],
                "move_cost": best["move_cost"],
                "relation_uncertainty_before": {
                    key: round(float(value), 8)
                    for key, value in uncertainty_before.items()
                },
                "relation_information_gain_nats": best[
                    "relation_information_gain_nats"
                ],
                "relation_uncertainty_after": uncertainty_after,
                "remaining_candidate_utilities": {
                    str(row["view"]): row["utility"] for row in candidates
                },
            }
        )
        for relation in RELATION_NAMES:
            precision[relation] += (
                measurement_precision * float(evidence[relation]) * complementarity
            )
        latest_direction = best.pop("direction")
        acquired_directions.append(latest_direction)
        acquired.append(view)
        remaining.remove(view)
    if not remaining:
        stop_reason = "candidate_views_exhausted"
    elif len(steps) < max_additional_views and stop_reason != "evidence_sufficient":
        stop_reason = "evidence_sufficient"
    return {
        "initial_view": current_view,
        "max_additional_views": int(max_additional_views),
        "steps": steps,
        "acquired_views": acquired,
        "final_relation_uncertainty": {
            relation: round(_posterior_variance(value), 8)
            for relation, value in precision.items()
        },
        "stop_reason": stop_reason,
        "access": "oracle/training_only",
    }


def relation_evidence_from_supervision(row: Mapping[str, Any]) -> dict[str, float]:
    projections = row.get("keypoint_projections", {})
    center = _keypoint_quality(projections.get("grasp_center"))
    left = _keypoint_quality(projections.get("left_contact"))
    right = _keypoint_quality(projections.get("right_contact"))
    closing = float(np.clip(row.get("closing_axis_observability", 0.0), 0.0, 1.0))
    approach = float(np.clip(row.get("approach_axis_observability", 0.0), 0.0, 1.0))
    target_scale = float(np.clip(row.get("target_scale_score", 0.0), 0.0, 1.0))
    contact_pair = min(left, right)
    return {
        "grasp_center_location": center * (0.4 + 0.6 * target_scale),
        "left_contact": min(center, left) * closing,
        "right_contact": min(center, right) * closing,
        "opening_width": contact_pair * closing,
        "approach_axis": center * approach,
        "closing_axis": max(contact_pair, 0.5 * center) * closing,
    }


def _optional_vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("optional camera position must be a finite 3-vector")
    return np.round(array, 8).tolist()


def validate_expert_event_view_ranking_sample(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != EXPERT_EVENT_VIEW_SAMPLE_SCHEMA:
        errors.append("unsupported expert event view sample schema")
    inference = value.get("inference_visible")
    training = value.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        errors.append("view ranking sample requires inference and training payloads")
        return errors
    forbidden = {
        "view_labels",
        "oracle_next_action",
        "oracle_active_sequence",
        "oracle_information_gain_nats",
        "oracle_utility",
        "oracle_relation_evidence",
    }
    leaked = sorted(_recursive_keys(inference).intersection(forbidden))
    if leaked:
        errors.append("oracle view labels leaked into inference_visible: " + ", ".join(leaked))
    if training.get("access") != "oracle/training_only":
        errors.append("view ranking training payload must remain training-only")
    labels = training.get("view_labels")
    candidates = inference.get("candidate_views")
    if not isinstance(labels, list) or not labels:
        errors.append("view ranking sample requires view labels")
    elif not isinstance(candidates, list) or {
        str(row.get("view")) for row in labels
    } != {str(row.get("view")) for row in candidates}:
        errors.append("view labels must align with inference candidate views")
    else:
        for row in labels:
            features = row.get("input_features", {})
            target = row.get("target", {})
            uncertainty = features.get("current_relation_uncertainty", {})
            if set(uncertainty) != set(RELATION_NAMES):
                errors.append("view input relation uncertainty schema mismatch")
            gains = target.get("relation_information_gain_nats", {})
            if set(gains) != set(RELATION_NAMES):
                errors.append("view target relation gain schema mismatch")
    return errors


def _keypoint_quality(value: Any) -> float:
    if not isinstance(value, Mapping) or value.get("observation_state") != "target_visible":
        return 0.0
    patch = float(np.clip(value.get("target_mask_patch_fraction", 0.0), 0.0, 1.0))
    return 0.35 + 0.65 * math.sqrt(patch)


def _posterior_variance(measurement_precision: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(measurement_precision)))


def _observation_for_view(sample: Mapping[str, Any], view: str) -> Mapping[str, Any]:
    rows: Sequence[Mapping[str, Any]] = sample["inference_visible"].get("observations", ())
    matches = [row for row in rows if str(row.get("view")) == view]
    if len(matches) != 1:
        raise ValueError(f"event sample must contain exactly one inference observation for {view}")
    return matches[0]


def _unit(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return result / norm


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = set(str(key) for key in value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()
