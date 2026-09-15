"""Inference-visible visibility priors for candidate camera poses.

This module deliberately estimates *projected target evidence*, not simulator
visibility truth.  It uses the current fused grasp frame, candidate camera
poses, and calibrated focal length to predict whether the contact pair can be
separated in a future image.  No actor IDs, meshes, masks, probe outcomes, or
oracle utility enter the calculation.

Occlusion remains a separate field.  Unless an explicit inference-visible
occluder hypothesis is provided, the estimator returns ``None`` for occlusion
risk and callers must keep that penalty withheld.  This distinction prevents
"no occluder model" from becoming a false claim that the target is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


VIEW_VISIBILITY_SCHEMA = "spatial.inference_visible_view_visibility_prior.v1"


@dataclass(frozen=True)
class ViewVisibilityConfig:
    """Declared constants for the projected-evidence prior."""

    minimum_separable_pixels: float = 5.0
    transition_pixels: float = 3.0
    maximum_target_radius_m: float = 0.15
    uncertainty_radius_scale: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_separable_pixels",
            "transition_pixels",
            "maximum_target_radius_m",
            "uncertainty_radius_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def estimate_grasp_frame_view_visibility(
    *,
    target_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
    candidate_camera_position_world: Sequence[float],
    candidate_view_direction_world: Sequence[float],
    focal_length_px: float,
    contact_separation_m: float,
    position_covariance_m2: Sequence[Sequence[float]] | None = None,
    config: ViewVisibilityConfig = ViewVisibilityConfig(),
) -> dict[str, Any]:
    """Estimate future contact-pair visibility from metric geometry.

    The returned probability is a soft, calibrated-*later* prior.  It is
    useful for ranking and audit, but it is not a safety authorization.  The
    contact separation is projected onto the image plane; views looking along
    the closing axis make the two contacts overlap and therefore receive a
    lower probability.
    """

    center = _vector3(target_center_world, "target center")
    closing = _unit(closing_axis_world, "closing axis")
    approach = _unit(approach_axis_world, "approach axis")
    camera = _vector3(candidate_camera_position_world, "camera position")
    view = _unit(candidate_view_direction_world, "view direction")
    focal = float(focal_length_px)
    separation = float(contact_separation_m)
    if not np.isfinite(focal) or focal <= 0.0:
        raise ValueError("focal length must be finite and positive")
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("contact separation must be finite and positive")
    to_target = center - camera
    range_m = float(np.linalg.norm(to_target))
    if range_m <= 1e-9:
        raise ValueError("camera must differ from target center")
    ray = to_target / range_m
    # A candidate's supplied view direction should point toward the target.  We
    # use the measured candidate ray for projection and report the mismatch.
    pointing_error = float(1.0 - np.clip(np.dot(ray, view), -1.0, 1.0))
    closing_projection = float(np.sqrt(max(0.0, 1.0 - np.dot(ray, closing) ** 2)))
    approach_projection = float(np.sqrt(max(0.0, 1.0 - np.dot(ray, approach) ** 2)))
    uncertainty_radius = 0.0
    if position_covariance_m2 is not None:
        covariance = _covariance3(position_covariance_m2)
        uncertainty_radius = float(
            config.uncertainty_radius_scale * np.sqrt(max(0.0, np.max(np.linalg.eigvalsh(covariance))))
        )
    effective_separation = min(
        separation + uncertainty_radius,
        config.maximum_target_radius_m * 2.0,
    )
    projected_separation_px = focal * effective_separation * closing_projection / range_m
    # Logistic transition keeps the score continuous while making the declared
    # pixel requirement visible in the output record.
    visibility = _sigmoid(
        (projected_separation_px - config.minimum_separable_pixels)
        / config.transition_pixels
    )
    visibility *= float(np.exp(-2.0 * pointing_error))
    visibility = float(np.clip(visibility, 0.0, 1.0))
    return {
        "schema_version": VIEW_VISIBILITY_SCHEMA,
        "predicted_target_visibility": visibility,
        "occlusion_risk": None,
        "occlusion_status": "withheld_no_occluder_hypotheses",
        "source": "inference_visible_grasp_frame_projection_prior",
        "access": "inference_visible",
        "range_m": range_m,
        "projected_contact_separation_px": float(projected_separation_px),
        "closing_axis_projection": closing_projection,
        "approach_axis_projection": approach_projection,
        "pointing_error": pointing_error,
        "uncertainty_radius_m": uncertainty_radius,
        "calibration_status": "declared_pixel_transition_not_fitted",
        "config": {
            "minimum_separable_pixels": float(config.minimum_separable_pixels),
            "transition_pixels": float(config.transition_pixels),
            "maximum_target_radius_m": float(config.maximum_target_radius_m),
            "uncertainty_radius_scale": float(config.uncertainty_radius_scale),
        },
    }


def attach_visibility_priors_to_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
    focal_length_px: float,
    contact_separation_m: float,
    position_covariance_m2: Sequence[Sequence[float]] | None = None,
    config: ViewVisibilityConfig = ViewVisibilityConfig(),
) -> list[dict[str, Any]]:
    """Add visibility priors to candidate rows without changing their order."""

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        position = row.get("camera_position_world")
        direction = row.get("view_direction_world")
        if position is None or direction is None:
            # The estimator requires a real candidate pose.  Keep missing pose
            # explicit instead of reconstructing a fake visibility probability.
            row["visibility_prior"] = {
                "schema_version": VIEW_VISIBILITY_SCHEMA,
                "predicted_target_visibility": None,
                "occlusion_risk": None,
                "status": "withheld_missing_candidate_pose",
                "access": "inference_visible",
            }
        else:
            prior = estimate_grasp_frame_view_visibility(
                target_center_world=target_center_world,
                closing_axis_world=closing_axis_world,
                approach_axis_world=approach_axis_world,
                candidate_camera_position_world=position,
                candidate_view_direction_world=direction,
                focal_length_px=focal_length_px,
                contact_separation_m=contact_separation_m,
                position_covariance_m2=position_covariance_m2,
                config=config,
            )
            row["predicted_target_visibility"] = prior[
                "predicted_target_visibility"
            ]
            # Keep occlusion absent: no inference-visible occluder model was
            # supplied, so analytic control will report this term as withheld.
            row["visibility_prior"] = prior
        result.append(row)
    return result


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    result = _vector3(value, name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return result / norm


def _covariance3(value: Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("position covariance must be a finite 3x3 matrix")
    result = (result + result.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(result))) < -1e-9:
        raise ValueError("position covariance must be positive semidefinite")
    return result


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + float(np.exp(-value)))
