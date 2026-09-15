"""Zero-training analytic control policy for candidate view selection.

This module is the *control* in the experiment, not a model. It fits no
weights and reads no dataset: every term is either a physical quantity derived
from the belief and a calibrated sensor model, or a declared cost constant that
is written into the returned record. Nothing here is learned, so its score on a
held-out task is by construction an out-of-sample number.

The score has the shape the design fixes:

    score = information_gain - movement_cost - risk_penalty

where ``information_gain`` is the L1 analytic Fisher reduction of the belief
covariance along the task axes (see :mod:`active_belief`), ``movement_cost`` is
the declared cost of travelling to the candidate pose, and ``risk_penalty``
covers predicted invisibility, occlusion, unreachable and unsafe poses.

Two properties are structural rather than tested-in:

* Each candidate is scored independently from belief-side quantities only, so
  the policy accepts any number of candidates in any order and has no notion of
  a view label. Adding or moving a camera changes the inputs, not the contract.
* Missing calibration raises. A view whose information gain could not be
  computed is not scored as zero gain, because "no gain" and "not computed"
  must stay distinguishable.

A learned ranker may be combined with this policy only as a bounded residual
(:func:`blend_with_bounded_residual`), so the deployed ordering can never move
further from the analytic ordering than the declared bound allows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .active_belief import (
    ViewSensorModel,
    analytic_view_information_features,
    depth_dominates_lateral_noise,
    is_nearly_isotropic_noise,
    measurement_noise_anisotropy_ratio,
)
from .view_pose_features import uncertainty_frame

ANALYTIC_CONTROL_SCHEMA = "spatial.analytic_view_control_policy.v1"

# Declared, not learned. These convert physical quantities into one comparable
# scalar; they are recorded with every decision so a reviewer can see the
# exchange rate that produced an ordering.
DEFAULT_INFORMATION_GAIN_SCALE_PER_M = 1.0
DEFAULT_MOVEMENT_COST_WEIGHT = 0.02
DEFAULT_INVISIBILITY_PENALTY = 0.05
DEFAULT_OCCLUSION_PENALTY = 0.05
DEFAULT_UNREACHABLE_PENALTY = 1.0
DEFAULT_UNSAFE_PENALTY = 1.0

# Each relation contributes to the task axis whose uncertainty it represents.
# Spatial position is deliberately shared across all axes; unknown relation
# names remain visible in the belief record but do not silently acquire a
# geometric meaning.
RELATION_AXIS_COMPONENTS: dict[str, tuple[float, float, float]] = {
    "grasp_center_location": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "left_contact": (1.0, 0.0, 0.0),
    "right_contact": (1.0, 0.0, 0.0),
    "opening_width": (1.0, 0.0, 0.0),
    "closing_axis": (1.0, 0.0, 0.0),
    "approach_axis": (0.0, 0.0, 1.0),
    "alignment": (0.5, 0.5, 0.0),
    "between_fingers": (1.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class AnalyticControlCosts:
    """Declared exchange rates between information, motion, and risk."""

    information_gain_scale_per_m: float = DEFAULT_INFORMATION_GAIN_SCALE_PER_M
    movement_cost_weight: float = DEFAULT_MOVEMENT_COST_WEIGHT
    invisibility_penalty: float = DEFAULT_INVISIBILITY_PENALTY
    occlusion_penalty: float = DEFAULT_OCCLUSION_PENALTY
    unreachable_penalty: float = DEFAULT_UNREACHABLE_PENALTY
    unsafe_penalty: float = DEFAULT_UNSAFE_PENALTY

    def __post_init__(self) -> None:
        for name in (
            "information_gain_scale_per_m",
            "movement_cost_weight",
            "invisibility_penalty",
            "occlusion_penalty",
            "unreachable_penalty",
            "unsafe_penalty",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"analytic control cost {name} must be finite and >= 0"
                )
        if self.information_gain_scale_per_m <= 0.0:
            raise ValueError("information gain scale must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "information_gain_scale_per_m": float(self.information_gain_scale_per_m),
            "movement_cost_weight": float(self.movement_cost_weight),
            "invisibility_penalty": float(self.invisibility_penalty),
            "occlusion_penalty": float(self.occlusion_penalty),
            "unreachable_penalty": float(self.unreachable_penalty),
            "unsafe_penalty": float(self.unsafe_penalty),
            "source": "declared_control_constants_not_fitted",
        }


@dataclass(frozen=True)
class AnalyticControlBelief:
    """Belief-side inputs shared by every candidate in one decision."""

    prior_covariance_m2: Sequence[Sequence[float]]
    task_axes_world: Sequence[Sequence[float]]
    sensor_model: ViewSensorModel
    focal_length_px: float
    range_m: float
    relation_uncertainty: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.focal_length_px) or self.focal_length_px <= 0.0:
            raise ValueError("focal length must be finite and positive")
        if not np.isfinite(self.range_m) or self.range_m <= 0.0:
            raise ValueError("range must be finite and positive")
        if len(tuple(self.task_axes_world)) != 3:
            raise ValueError("analytic control needs exactly three task axes")
        for name, value in dict(self.relation_uncertainty).items():
            uncertainty = float(value)
            if not np.isfinite(uncertainty) or not 0.0 <= uncertainty <= 1.0:
                raise ValueError(
                    f"relation uncertainty {name!r} must be finite and in [0, 1]"
                )

    def directional_signal_available(self, range_m: float | None = None) -> bool:
        """Whether calibrated noise is sufficiently anisotropic at a range."""

        return not is_nearly_isotropic_noise(
            sensor_model=self.sensor_model,
            range_m=float(self.range_m if range_m is None else range_m),
            focal_length_px=float(self.focal_length_px),
        )

    @property
    def depth_premise_holds(self) -> bool:
        """Backward-compatible name for directional signal availability."""

        return self.directional_signal_available()

    def as_dict(self) -> dict[str, Any]:
        return {
            "focal_length_px": float(self.focal_length_px),
            "range_m": float(self.range_m),
            "keypoint_std_px": float(self.sensor_model.keypoint_std_px),
            "depth_std_m": float(self.sensor_model.depth_std_m),
            "depth_dominates_lateral_noise": depth_dominates_lateral_noise(
                sensor_model=self.sensor_model,
                range_m=float(self.range_m),
                focal_length_px=float(self.focal_length_px),
            ),
            "directional_signal_available": bool(self.depth_premise_holds),
            "measurement_noise_anisotropy_ratio": measurement_noise_anisotropy_ratio(
                sensor_model=self.sensor_model,
                range_m=float(self.range_m),
                focal_length_px=float(self.focal_length_px),
            ),
            "relation_count": len(dict(self.relation_uncertainty)),
            "relation_uncertainty": {
                str(name): float(value)
                for name, value in self.relation_uncertainty.items()
            },
        }


def analytic_view_control_score(
    *,
    belief: AnalyticControlBelief,
    view_direction_world: Iterable[float],
    movement_cost: float = 0.0,
    predicted_target_visibility: float | None = None,
    occlusion_risk: float | None = None,
    camera_reachable: bool = True,
    camera_safe: bool = True,
    range_m: float | None = None,
    costs: AnalyticControlCosts = AnalyticControlCosts(),
) -> dict[str, Any]:
    """Score one candidate pose. Fits nothing; reads only belief-side inputs.

    ``predicted_target_visibility`` and ``occlusion_risk`` are the L2 layer.
    L2 is not implemented yet, so ``None`` means "not computed" and the
    corresponding penalty is withheld and reported as such -- it is not
    silently replaced by an optimistic or pessimistic constant.
    """

    effective_range_m = float(belief.range_m if range_m is None else range_m)
    if not belief.directional_signal_available(effective_range_m):
        raise ValueError(
            "analytic view control has no directional signal because calibrated "
            "lateral and axial noise are nearly isotropic at this range"
        )
    movement = float(movement_cost)
    if not np.isfinite(movement) or movement < 0.0:
        raise ValueError("movement cost must be finite and >= 0")
    information = analytic_view_information_features(
        prior_covariance=np.asarray(belief.prior_covariance_m2, dtype=np.float64),
        candidate_view_direction_world=view_direction_world,
        task_axes_world=belief.task_axes_world,
        range_m=effective_range_m,
        focal_length_px=float(belief.focal_length_px),
        sensor_model=belief.sensor_model,
    )
    axis_weights, weight_source = _relation_axis_weights(belief, information)
    axis_reductions = np.asarray(
        [
            information["fisher_std_reduction_closing_m"],
            information["fisher_std_reduction_lateral_m"],
            information["fisher_std_reduction_approach_m"],
        ],
        dtype=np.float64,
    )
    gain_m = float(np.dot(axis_weights, axis_reductions))
    information_gain = costs.information_gain_scale_per_m * gain_m
    movement_term = costs.movement_cost_weight * movement
    penalties: dict[str, float] = {}
    withheld: list[str] = []
    if predicted_target_visibility is None:
        withheld.append("predicted_target_visibility")
    else:
        visibility = _unit_interval(
            predicted_target_visibility, "predicted target visibility"
        )
        penalties["invisibility"] = costs.invisibility_penalty * (1.0 - visibility)
    if occlusion_risk is None:
        withheld.append("occlusion_risk")
    else:
        penalties["occlusion"] = costs.occlusion_penalty * _unit_interval(
            occlusion_risk, "occlusion risk"
        )
    if not camera_reachable:
        penalties["unreachable"] = costs.unreachable_penalty
    if not camera_safe:
        penalties["unsafe"] = costs.unsafe_penalty
    risk_penalty = float(sum(penalties.values()))
    score = information_gain - movement_term - risk_penalty
    return {
        "schema_version": ANALYTIC_CONTROL_SCHEMA,
        "score": float(score),
        "information_gain": float(information_gain),
        "relation_weighted_std_reduction_m": gain_m,
        "worst_axis_std_reduction_m": float(
            information["fisher_worst_axis_std_reduction_m"]
        ),
        "task_axis_weights": {
            name: float(value)
            for name, value in zip(("closing", "lateral", "approach"), axis_weights)
        },
        "task_axis_weight_source": weight_source,
        "range_m": effective_range_m,
        "log_volume_reduction_nats": float(
            information["fisher_log_volume_reduction_nats"]
        ),
        "movement_cost": movement,
        "movement_penalty": float(movement_term),
        "risk_penalty": risk_penalty,
        "risk_terms": {name: float(value) for name, value in penalties.items()},
        "withheld_risk_terms": tuple(withheld),
        "predicted_target_visibility": (
            float(predicted_target_visibility)
            if predicted_target_visibility is not None
            else None
        ),
        "occlusion_risk": (
            float(occlusion_risk) if occlusion_risk is not None else None
        ),
        "feasible": bool(camera_reachable and camera_safe),
        "fitted_parameter_count": 0,
        "costs": costs.as_dict(),
    }


def rank_candidates_with_analytic_control(
    *,
    belief: AnalyticControlBelief,
    candidates: Sequence[Mapping[str, Any]],
    costs: AnalyticControlCosts = AnalyticControlCosts(),
) -> dict[str, Any]:
    """Rank an arbitrary number of candidate poses with the control policy.

    Each candidate is a mapping with ``view_direction_world`` and optional
    ``movement_cost``, ``predicted_target_visibility``, ``occlusion_risk``,
    ``camera_reachable``, ``camera_safe``, and an opaque ``view`` label that is
    carried through for reporting but never enters the score.
    """

    if not candidates:
        raise ValueError("analytic view control needs at least one candidate")
    scored = []
    for index, candidate in enumerate(candidates):
        if "view_direction_world" not in candidate:
            raise ValueError("each analytic control candidate needs a view direction")
        candidate_range_m = candidate.get("range_m", candidate.get("candidate_range_m"))
        reachable = candidate.get("camera_reachable")
        safe = candidate.get("camera_safe")
        record = analytic_view_control_score(
            belief=belief,
            view_direction_world=candidate["view_direction_world"],
            movement_cost=float(
                candidate.get("movement_cost", candidate.get("move_cost", 0.0))
            ),
            predicted_target_visibility=candidate.get("predicted_target_visibility"),
            occlusion_risk=candidate.get("occlusion_risk"),
            # Candidate ranking is fail-closed. Catalogue adapters must state
            # that a pose passed their reachability and bounds checks.
            camera_reachable=bool(reachable) if reachable is not None else False,
            camera_safe=bool(safe) if safe is not None else False,
            range_m=(
                float(candidate_range_m) if candidate_range_m is not None else None
            ),
            costs=costs,
        )
        record["candidate_index"] = int(index)
        record["view"] = str(candidate.get("view", f"candidate_{index}"))
        record["view_direction_world"] = _unit_vector(
            candidate["view_direction_world"], "candidate view direction"
        ).tolist()
        if candidate.get("visibility_prior") is not None:
            record["visibility_prior"] = dict(candidate["visibility_prior"])
        if candidate.get("camera_position_source") is not None:
            record["camera_position_source"] = str(
                candidate["camera_position_source"]
            )
        if candidate.get("camera_position_world") is not None:
            record["camera_position_world"] = [
                float(value) for value in candidate["camera_position_world"]
            ]
        scored.append(record)
    # Sort on the score alone; ties break on the candidate's own index so the
    # output is deterministic without ever consulting a view name.
    order = sorted(
        scored,
        key=lambda row: (
            not row["feasible"],
            -row["score"],
            row["candidate_index"],
        ),
    )
    feasible = [row for row in order if row["feasible"]]
    selected = feasible[0] if feasible else None
    return {
        "schema_version": ANALYTIC_CONTROL_SCHEMA,
        "policy": "analytic_control_zero_training",
        "fitted_parameter_count": 0,
        "candidate_count": len(scored),
        "feasible_candidate_count": len(feasible),
        "belief": belief.as_dict(),
        "costs": costs.as_dict(),
        "ranked_candidates": order,
        "status": "selected" if selected is not None else "no_feasible_candidate",
        "selection_reason": (
            "maximum_relation_weighted_information_gain_minus_cost_and_risk"
            if selected is not None
            else "all_candidates_failed_reachability_or_safety_gate"
        ),
        "selected_view": selected["view"] if selected is not None else None,
        "selected_candidate_index": (
            selected["candidate_index"] if selected is not None else None
        ),
        "access": "inference_visible",
    }


def rank_runtime_candidates_with_analytic_control(
    *,
    prior_covariance_m2: Sequence[Sequence[float]],
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
    relation_uncertainty: Mapping[str, float],
    sensor_model: ViewSensorModel,
    focal_length_px: float,
    default_range_m: float,
    belief_centroid_world: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
    costs: AnalyticControlCosts = AnalyticControlCosts(),
) -> dict[str, Any]:
    """Adapt runtime graph state and camera poses to the zero-training policy."""

    centroid = np.asarray(belief_centroid_world, dtype=np.float64)
    if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
        raise ValueError("belief centroid must be a finite 3-vector")
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        position = row.get("camera_position_world")
        if position is not None:
            camera_position = np.asarray(position, dtype=np.float64)
            if camera_position.shape != (3,) or not np.all(
                np.isfinite(camera_position)
            ):
                raise ValueError("candidate camera position must be a finite 3-vector")
            candidate_range = float(np.linalg.norm(camera_position - centroid))
            if candidate_range <= 1e-9:
                raise ValueError(
                    "candidate camera position must differ from belief centroid"
                )
            row["range_m"] = candidate_range
        enriched.append(row)
    belief = AnalyticControlBelief(
        prior_covariance_m2=prior_covariance_m2,
        task_axes_world=uncertainty_frame(
            closing_axis_world=closing_axis_world,
            approach_axis_world=approach_axis_world,
        ),
        sensor_model=sensor_model,
        focal_length_px=float(focal_length_px),
        range_m=float(default_range_m),
        relation_uncertainty=relation_uncertainty,
    )
    return rank_candidates_with_analytic_control(
        belief=belief,
        candidates=enriched,
        costs=costs,
    )


def blend_with_bounded_residual(
    *,
    control: Mapping[str, Any],
    learned_scores: Mapping[str, float],
    residual_bound: float,
) -> dict[str, Any]:
    """Add a clipped learned residual to the analytic control scores.

    The learned model can reorder candidates only within ``residual_bound``, so
    the deployed policy stays anchored to the analytic ordering. Candidates the
    learned model has no score for keep their analytic score, and infeasible
    candidates are never rescued by a residual.
    """

    bound = float(residual_bound)
    if not np.isfinite(bound) or bound < 0.0:
        raise ValueError("residual bound must be finite and >= 0")
    ranked = []
    overridden = 0
    for row in control["ranked_candidates"]:
        raw = learned_scores.get(row["view"])
        applied = 0.0
        if raw is not None and row["feasible"]:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError("learned residual must be finite")
            applied = float(np.clip(value, -bound, bound))
        merged = dict(row)
        merged["analytic_score"] = float(row["score"])
        merged["learned_residual_raw"] = None if raw is None else float(raw)
        merged["learned_residual_applied"] = applied
        merged["score"] = float(row["score"]) + applied
        ranked.append(merged)
    order = sorted(
        ranked,
        key=lambda row: (
            not row["feasible"],
            -row["score"],
            row["candidate_index"],
        ),
    )
    analytic_choice = control["selected_candidate_index"]
    feasible = [row for row in order if row["feasible"]]
    selected = feasible[0] if feasible else None
    if selected is not None and selected["candidate_index"] != analytic_choice:
        overridden = 1
    return {
        "schema_version": ANALYTIC_CONTROL_SCHEMA,
        "policy": "analytic_control_with_bounded_learned_residual",
        "residual_bound": bound,
        "belief": control["belief"],
        "costs": control["costs"],
        "ranked_candidates": order,
        "status": "selected" if selected is not None else "no_feasible_candidate",
        "selected_view": selected["view"] if selected is not None else None,
        "selected_candidate_index": (
            selected["candidate_index"] if selected is not None else None
        ),
        "analytic_selected_view": control["selected_view"],
        "overrode_analytic_choice": bool(overridden),
        "access": "inference_visible",
    }


def _unit_interval(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _relation_axis_weights(
    belief: AnalyticControlBelief,
    information: Mapping[str, float],
) -> tuple[np.ndarray, str]:
    raw = np.zeros(3, dtype=np.float64)
    for relation, uncertainty in belief.relation_uncertainty.items():
        components = RELATION_AXIS_COMPONENTS.get(str(relation))
        if components is not None:
            raw += float(uncertainty) * np.asarray(components, dtype=np.float64)
    if float(np.sum(raw)) > 1e-12:
        return raw / float(np.sum(raw)), "query_relation_uncertainty"
    prior_std = np.asarray(
        [
            information["fisher_prior_std_closing_m"],
            information["fisher_prior_std_lateral_m"],
            information["fisher_prior_std_approach_m"],
        ],
        dtype=np.float64,
    )
    fallback = np.zeros(3, dtype=np.float64)
    fallback[int(np.argmax(prior_std))] = 1.0
    return fallback, "prior_worst_axis_fallback"


def _unit_vector(value: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm
