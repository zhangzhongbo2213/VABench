"""Conservative, inference-visible value-of-information diagnostics.

The active tools already have a geometric information score.  This module
turns that score into an explicit *shadow* stop recommendation: continue only
when a candidate is expected to reduce a currently uncertain relation enough
to justify its movement and declared observation cost.  It is intentionally a
diagnostic layer; it does not grant camera or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


VIEW_VOI_SCHEMA = "spatial.view_value_of_information_shadow.v1"


@dataclass(frozen=True)
class ViewVOIConfig:
    """Declared scales for comparing expected evidence and observation cost."""

    uncertainty_scale_m: float = 0.015
    minimum_net_voi: float = 0.025
    observation_cost_weight: float = 1.0
    withheld_visibility_factor: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "uncertainty_scale_m",
            "minimum_net_voi",
            "observation_cost_weight",
            "withheld_visibility_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.uncertainty_scale_m <= 0.0:
            raise ValueError("uncertainty_scale_m must be positive")
        if self.withheld_visibility_factor > 1.0:
            raise ValueError("withheld_visibility_factor must be <= 1")


def score_view_value_of_information(
    *,
    candidate: Mapping[str, Any],
    relation_uncertainty: Mapping[str, float],
    config: ViewVOIConfig = ViewVOIConfig(),
) -> dict[str, Any]:
    """Score one already-ranked candidate using only runtime-visible fields."""

    uncertainty_values = [
        float(value)
        for value in relation_uncertainty.values()
        if np.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    ]
    uncertainty_need = float(np.mean(uncertainty_values)) if uncertainty_values else 0.0
    info_gain_m = candidate.get(
        "relation_weighted_std_reduction_m",
        candidate.get("worst_axis_std_reduction_m"),
    )
    missing: list[str] = []
    if info_gain_m is None:
        missing.append("information_gain")
        info_gain_m = 0.0
    info_gain_m = float(info_gain_m)
    if not np.isfinite(info_gain_m) or info_gain_m < 0.0:
        raise ValueError("candidate information gain must be finite and >= 0")
    visibility = candidate.get("predicted_target_visibility")
    if visibility is None:
        prior = candidate.get("visibility_prior")
        if isinstance(prior, Mapping):
            visibility = prior.get("predicted_target_visibility")
    if visibility is None:
        missing.append("predicted_target_visibility")
        visibility_factor = float(config.withheld_visibility_factor)
    else:
        visibility_factor = float(visibility)
        if not np.isfinite(visibility_factor) or not 0.0 <= visibility_factor <= 1.0:
            raise ValueError("predicted target visibility must be in [0, 1]")
    feasible = bool(candidate.get("feasible", False))
    if not feasible:
        missing.append("feasible_camera_pose")
    normalized_gain = float(
        np.clip(info_gain_m / config.uncertainty_scale_m, 0.0, 1.0)
    )
    expected_gate_change = float(
        np.clip(normalized_gain * visibility_factor * uncertainty_need, 0.0, 1.0)
    )
    movement_penalty = float(candidate.get("movement_penalty", 0.0))
    risk_penalty = float(candidate.get("risk_penalty", 0.0))
    if not np.isfinite(movement_penalty) or movement_penalty < 0.0:
        raise ValueError("movement penalty must be finite and >= 0")
    if not np.isfinite(risk_penalty) or risk_penalty < 0.0:
        raise ValueError("risk penalty must be finite and >= 0")
    observation_cost = config.observation_cost_weight * (
        movement_penalty + risk_penalty
    )
    net_voi = expected_gate_change - observation_cost
    if not feasible:
        net_voi = float("-inf")
    action = "acquire_view" if net_voi >= config.minimum_net_voi else "stop_for_review"
    if missing:
        action = "stop_for_review"
    reason = (
        "expected_gate_change_exceeds_cost"
        if action == "acquire_view"
        else "missing_runtime_evidence"
        if missing
        else "expected_gate_change_below_cost_threshold"
    )
    return {
        "schema_version": VIEW_VOI_SCHEMA,
        "view": str(candidate.get("view", "")),
        "action": action,
        "reason": reason,
        "expected_gate_change": expected_gate_change,
        "information_gain_m": info_gain_m,
        "normalized_information_gain": normalized_gain,
        "uncertainty_need": uncertainty_need,
        "predicted_target_visibility": (
            float(visibility_factor) if visibility is not None else None
        ),
        "observation_cost": float(observation_cost),
        "movement_penalty": movement_penalty,
        "risk_penalty": risk_penalty,
        "net_voi": net_voi,
        "missing_evidence": missing,
        "execution_authorization": "disabled_shadow_only",
        "calibration_status": "declared_voi_scales_not_fitted",
        "access": "inference_visible",
    }


def rank_views_by_value_of_information(
    *,
    candidates: Sequence[Mapping[str, Any]],
    relation_uncertainty: Mapping[str, float],
    config: ViewVOIConfig = ViewVOIConfig(),
) -> dict[str, Any]:
    """Return a deterministic shadow recommendation and all candidate reasons."""

    rows = [
        score_view_value_of_information(
            candidate=candidate,
            relation_uncertainty=relation_uncertainty,
            config=config,
        )
        for candidate in candidates
    ]
    rows.sort(key=lambda row: (-row["net_voi"], row["view"]))
    selected = next((row for row in rows if row["action"] == "acquire_view"), None)
    return {
        "schema_version": VIEW_VOI_SCHEMA,
        "policy": "declared_runtime_visible_voi_shadow",
        "action": "acquire_view" if selected is not None else "stop_for_review",
        "selected_view": selected["view"] if selected is not None else None,
        "selection_reason": (
            selected["reason"]
            if selected is not None
            else "no_candidate_meets_cost_adjusted_voi_threshold"
        ),
        "candidate_count": len(rows),
        "ranked_candidates": rows,
        "execution_authorization": "disabled_shadow_only",
        "access": "inference_visible",
    }
