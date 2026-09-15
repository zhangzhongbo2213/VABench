"""Task-native grasp candidates perturbed from automatic expert frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import transforms3d as t3d

from .grasp_candidates import CandidateCheck, GraspCandidate
from .spatial_graph import unit


PERTURBATION_PROFILES = ("default", "hard_negative_v1", "far_negative_v1")


@dataclass(frozen=True)
class ExpertFramePerturbationConfig:
    longitudinal_scale_factors: tuple[float, ...] = (-1.0, 1.0)
    closing_offset_scale_factors: tuple[float, ...] = (-0.5, 0.5)
    approach_offset_scale_factors: tuple[float, ...] = (-0.35, 0.35)
    orientation_offsets_deg: tuple[float, ...] = (-15.0, 15.0)
    # Contact-centroid separation is not yet calibrated to RoboTwin's
    # normalized gripper actuator command, so it must not be perturbed as if it
    # were an executable jaw setpoint.
    opening_width_scales: tuple[float, ...] = ()
    pregrasp_distance_m: float = 0.09
    minimum_scale_m: float = 0.012
    maximum_scale_m: float = 0.05
    minimum_opening_width_m: float = 0.008
    maximum_opening_width_m: float = 0.09
    include_nominal: bool = True

    def __post_init__(self) -> None:
        if self.pregrasp_distance_m < 0.0:
            raise ValueError("pregrasp_distance_m must be non-negative")
        if not 0.0 < self.minimum_scale_m <= self.maximum_scale_m:
            raise ValueError("perturbation metric scale bounds are invalid")
        if not 0.0 < self.minimum_opening_width_m < self.maximum_opening_width_m:
            raise ValueError("opening width bounds are invalid")
        if any(value <= 0.0 for value in self.opening_width_scales):
            raise ValueError("opening width scales must be positive")


def resolve_perturbation_profile(profile: str) -> ExpertFramePerturbationConfig:
    """Return a reproducible candidate suite for data collection.

    ``default`` preserves the original nine near-expert candidates. The hard
    profile keeps those candidates and adds larger single-factor deviations so
    physical replay can supply genuine negative and all-negative states.
    """

    normalized = str(profile).strip().lower()
    if normalized == "default":
        return ExpertFramePerturbationConfig()
    if normalized == "hard_negative_v1":
        return ExpertFramePerturbationConfig(
            longitudinal_scale_factors=(-2.5, -1.0, 1.0, 2.5),
            closing_offset_scale_factors=(-2.0, -0.5, 0.5, 2.0),
            approach_offset_scale_factors=(-2.0, -0.35, 0.35, 2.0),
            orientation_offsets_deg=(-45.0, -15.0, 15.0, 45.0),
        )
    if normalized == "far_negative_v1":
        return ExpertFramePerturbationConfig(
            longitudinal_scale_factors=(-2.5, 2.5),
            closing_offset_scale_factors=(-2.0, 2.0),
            approach_offset_scale_factors=(-2.0, 2.0),
            orientation_offsets_deg=(-45.0, 45.0),
            include_nominal=False,
        )
    raise ValueError(
        f"unsupported perturbation profile {profile!r}; "
        f"expected one of {', '.join(PERTURBATION_PROFILES)}"
    )


def generate_expert_frame_perturbations(
    trace: Mapping[str, Any],
    *,
    config: ExpertFramePerturbationConfig | None = None,
) -> list[GraspCandidate]:
    """Generate deterministic, one-factor candidates without manual points."""

    resolved = config or ExpertFramePerturbationConfig()
    frame = trace.get("expert_grasp_frame")
    if not isinstance(frame, Mapping):
        raise ValueError("trace has no expert_grasp_frame")
    if frame.get("valid_for_contact_supervision") is not True:
        raise ValueError("expert frame requires bilateral contact supervision")
    center = _vector(frame.get("center_world_m"), "center_world_m")
    approach = unit(_vector(frame.get("approach_axis_world"), "approach_axis_world"))
    closing = unit(_vector(frame.get("closing_axis_world"), "closing_axis_world"))
    closing = unit(closing - float(np.dot(closing, approach)) * approach)
    longitudinal = unit(np.cross(approach, closing))
    opening_width = float(frame.get("opening_width_m"))
    if not math.isfinite(opening_width) or opening_width <= 0.0:
        raise ValueError("expert frame opening_width_m must be positive")
    expert_left_contact = _vector(
        frame.get("left_contact_world_m"), "left_contact_world_m"
    )
    expert_right_contact = _vector(
        frame.get("right_contact_world_m"), "right_contact_world_m"
    )
    contact_midpoint_offset = (
        expert_left_contact + expert_right_contact
    ) / 2.0 - center
    contact_half_separation = (
        expert_right_contact - expert_left_contact
    ) / 2.0
    metric_scale = float(
        np.clip(opening_width, resolved.minimum_scale_m, resolved.maximum_scale_m)
    )
    position_covariance = _matrix3(
        frame.get("position_covariance_m2"), "position_covariance_m2"
    )
    orientation_covariance = _matrix3(
        frame.get("orientation_covariance_rad2"), "orientation_covariance_rad2"
    )
    event_id = _text(trace.get("event_id"), "event_id")
    actor_name = _text(trace.get("actor_name"), "actor_name")
    semantic_role = str(trace.get("task_stage") or "task_conditioned_grasp_region")
    preapproach = trace.get("preapproach")
    recorded_pregrasp_distance = (
        float(preapproach.get("expert_pregrasp_distance_m", 0.0))
        if isinstance(preapproach, Mapping)
        else 0.0
    )
    pregrasp_distance = (
        recorded_pregrasp_distance
        if recorded_pregrasp_distance > 1e-6
        else resolved.pregrasp_distance_m
    )

    specifications: list[dict[str, Any]] = []
    if resolved.include_nominal:
        specifications.append(
            {
                "kind": "none",
                "center_delta": np.zeros(3, dtype=np.float64),
                "orientation_deg": 0.0,
                "opening_scale": 1.0,
            }
        )
    specifications.extend(
        {
            "kind": "along_region_offset",
            "center_delta": longitudinal * metric_scale * float(factor),
            "orientation_deg": 0.0,
            "opening_scale": 1.0,
            "scale_factor": float(factor),
        }
        for factor in resolved.longitudinal_scale_factors
    )
    specifications.extend(
        {
            "kind": "across_closing_offset",
            "center_delta": closing * metric_scale * float(factor),
            "orientation_deg": 0.0,
            "opening_scale": 1.0,
            "scale_factor": float(factor),
        }
        for factor in resolved.closing_offset_scale_factors
    )
    specifications.extend(
        {
            "kind": "vertical_offset",
            "center_delta": approach * metric_scale * float(factor),
            "orientation_deg": 0.0,
            "opening_scale": 1.0,
            "scale_factor": float(factor),
        }
        for factor in resolved.approach_offset_scale_factors
    )
    specifications.extend(
        {
            "kind": "orientation_offset",
            "center_delta": np.zeros(3, dtype=np.float64),
            "orientation_deg": float(angle),
            "opening_scale": 1.0,
        }
        for angle in resolved.orientation_offsets_deg
    )
    specifications.extend(
        {
            "kind": "opening_width_offset",
            "center_delta": np.zeros(3, dtype=np.float64),
            "orientation_deg": 0.0,
            "opening_scale": float(scale),
        }
        for scale in resolved.opening_width_scales
    )

    result = []
    for index, specification in enumerate(specifications):
        angle_rad = math.radians(float(specification["orientation_deg"]))
        if abs(angle_rad) > 1e-12:
            rotation = t3d.axangles.axangle2mat(approach, angle_rad)
            candidate_closing = unit(rotation @ closing)
        else:
            rotation = np.eye(3, dtype=np.float64)
            candidate_closing = closing.copy()
        candidate_center = center + np.asarray(
            specification["center_delta"], dtype=np.float64
        )
        opening_scale = float(specification["opening_scale"])
        rotated_midpoint_offset = rotation @ contact_midpoint_offset
        rotated_half_separation = rotation @ contact_half_separation
        left_contact = (
            candidate_center
            + rotated_midpoint_offset
            - rotated_half_separation * opening_scale
        )
        right_contact = (
            candidate_center
            + rotated_midpoint_offset
            + rotated_half_separation * opening_scale
        )
        candidate_width = float(np.linalg.norm(right_contact - left_contact))
        pregrasp = candidate_center - approach * pregrasp_distance
        kind = str(specification["kind"])
        generation_parameters = {
            "perturbation_kind": kind,
            "center_delta_world_m": np.round(
                specification["center_delta"], 8
            ).tolist(),
            "longitudinal_offset_m": round(
                float(np.dot(specification["center_delta"], longitudinal)), 8
            ),
            "closing_offset_m": round(
                float(np.dot(specification["center_delta"], closing)), 8
            ),
            "approach_offset_m": round(
                float(np.dot(specification["center_delta"], approach)), 8
            ),
            "metric_scale_m": round(metric_scale, 8),
            "scale_factor": specification.get("scale_factor"),
            "orientation_offset_deg": float(specification["orientation_deg"]),
            "opening_width_scale": float(specification["opening_scale"]),
            "expert_event_id": event_id,
            "pregrasp_distance_m": round(pregrasp_distance, 8),
            "point_annotation": "automatic_from_successful_trajectory",
        }
        checks = _candidate_checks(
            kind=kind,
            opening_width_m=candidate_width,
            config=resolved,
        )
        nominal = kind == "none"
        result.append(
            GraspCandidate(
                candidate_id=f"{event_id}.perturbation_{index:02d}",
                target_id=f"object.{actor_name}",
                region_id=f"expert_event.{event_id}",
                semantic_role=semantic_role,
                center_world_m=candidate_center,
                left_contact_world_m=left_contact,
                right_contact_world_m=right_contact,
                approach_axis_world=approach,
                closing_axis_world=candidate_closing,
                pregrasp_center_world_m=pregrasp,
                opening_width_m=candidate_width,
                generation_parameters=generation_parameters,
                position_covariance_m2=position_covariance,
                orientation_covariance_rad2=orientation_covariance,
                checks=checks,
                score=1.0 if nominal else 0.5,
                score_confidence=0.25,
                source="automatic_successful_expert_frame_perturbation",
                access="oracle/training_only",
            )
        )
    return result


def _candidate_checks(
    *,
    kind: str,
    opening_width_m: float,
    config: ExpertFramePerturbationConfig,
) -> tuple[CandidateCheck, ...]:
    width_feasible = float(
        config.minimum_opening_width_m
        <= opening_width_m
        <= config.maximum_opening_width_m
    )
    return (
        CandidateCheck(
            "semantic_match",
            1.0,
            True,
            "same_registered_expert_event",
        ),
        CandidateCheck(
            "antipodal_geometry",
            None,
            True,
            "requires_execution_probe",
        ),
        CandidateCheck(
            "task_compatibility",
            1.0 if kind == "none" else None,
            False,
            "requires_execution_probe" if kind != "none" else "expert_positive",
        ),
        CandidateCheck(
            "opening_width_feasible",
            width_feasible,
            True,
            "gripper_limits",
            {
                "opening_width_m": round(float(opening_width_m), 8),
                "allowed_range_m": [
                    config.minimum_opening_width_m,
                    config.maximum_opening_width_m,
                ],
            },
        ),
        CandidateCheck("support_clearance", None, False, "requires_execution_probe"),
        CandidateCheck("reachable", None, True, "requires_execution_probe"),
        CandidateCheck("collision_free", None, True, "requires_execution_probe"),
        CandidateCheck(
            "predicted_execution_success",
            None,
            False,
            "requires_execution_probe",
        ),
    )


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _matrix3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
