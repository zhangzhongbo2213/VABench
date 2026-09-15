"""Continuous, view-label-independent geometry for active-view ranking."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


POSE_FEATURE_NAMES = (
    "candidate_ray_alignment_closing",
    "candidate_ray_alignment_lateral",
    "candidate_ray_alignment_approach",
    "history_parallax_min",
    "history_parallax_mean",
    "history_parallax_max",
    "candidate_position_alignment_closing",
    "candidate_position_alignment_lateral",
    "candidate_position_alignment_approach",
    "candidate_range_m",
    "candidate_position_available",
    "predicted_target_visibility",
    "depth_noise_at_range",
    "occlusion_risk",
    "camera_reachable",
    "camera_safe",
)


def relative_view_geometry(
    candidate_view_direction_world: Sequence[float],
    *,
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
    observed_view_directions_world: Sequence[Sequence[float]] = (),
    candidate_camera_position_world: Sequence[float] | None = None,
    belief_centroid_world: Sequence[float] | None = None,
    predicted_target_visibility: float | None = None,
    depth_noise_at_range: float | None = None,
    occlusion_risk: float | None = None,
    camera_reachable: bool | None = None,
    camera_safe: bool | None = None,
) -> dict[str, float]:
    """Describe a candidate pose relative to the learned belief frame.

    Axis projections are sign-invariant because grasp closing axes and local
    principal axes may be observed with either orientation. The result never
    depends on a view label, candidate index, or simulator target pose.
    """

    direction = _unit(candidate_view_direction_world, "candidate view direction")
    closing, lateral, approach = uncertainty_frame(
        closing_axis_world=closing_axis_world,
        approach_axis_world=approach_axis_world,
    )
    ray_alignment = np.abs(
        np.asarray(
            [
                np.dot(direction, closing),
                np.dot(direction, lateral),
                np.dot(direction, approach),
            ],
            dtype=np.float64,
        )
    )
    parallax = np.asarray(
        [
            1.0
            - abs(
                float(
                    np.dot(
                        direction,
                        _unit(previous, "observed view direction"),
                    )
                )
            )
            for previous in observed_view_directions_world
        ],
        dtype=np.float64,
    )
    if parallax.size == 0:
        parallax = np.zeros(1, dtype=np.float64)

    position_alignment = np.zeros(3, dtype=np.float64)
    candidate_range_m = 0.0
    position_available = 0.0
    if candidate_camera_position_world is not None or belief_centroid_world is not None:
        if candidate_camera_position_world is None or belief_centroid_world is None:
            raise ValueError(
                "candidate camera position and belief centroid must be provided together"
            )
        camera_position = _vector3(
            candidate_camera_position_world, "candidate camera position"
        )
        belief_centroid = _vector3(belief_centroid_world, "belief centroid")
        relative_position = camera_position - belief_centroid
        candidate_range_m = float(np.linalg.norm(relative_position))
        if candidate_range_m <= 1e-9:
            raise ValueError(
                "candidate camera position must differ from belief centroid"
            )
        relative_direction = relative_position / candidate_range_m
        position_alignment = np.abs(
            np.asarray(
                [
                    np.dot(relative_direction, closing),
                    np.dot(relative_direction, lateral),
                    np.dot(relative_direction, approach),
                ],
                dtype=np.float64,
            )
        )
        position_available = 1.0

    result = {
        "candidate_ray_alignment_closing": float(ray_alignment[0]),
        "candidate_ray_alignment_lateral": float(ray_alignment[1]),
        "candidate_ray_alignment_approach": float(ray_alignment[2]),
        "history_parallax_min": float(np.min(parallax)),
        "history_parallax_mean": float(np.mean(parallax)),
        "history_parallax_max": float(np.max(parallax)),
        "candidate_position_alignment_closing": float(position_alignment[0]),
        "candidate_position_alignment_lateral": float(position_alignment[1]),
        "candidate_position_alignment_approach": float(position_alignment[2]),
        "candidate_range_m": candidate_range_m,
        "candidate_position_available": position_available,
        "predicted_target_visibility": _probability_or_default(
            predicted_target_visibility, 0.5
        ),
        "depth_noise_at_range": _nonnegative_or_default(depth_noise_at_range, 0.0),
        "occlusion_risk": _probability_or_default(occlusion_risk, 0.5),
        "camera_reachable": _boolean_or_default(camera_reachable, 0.5),
        "camera_safe": _boolean_or_default(camera_safe, 0.5),
    }
    if set(result) != set(POSE_FEATURE_NAMES):
        raise AssertionError("pose feature schema drifted")
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("relative view geometry must be finite")
    return result


def relative_view_geometry_from_observability(
    input_features: Mapping[str, Any],
) -> dict[str, float]:
    """Adapt existing Stage-0 samples that do not store full camera poses.

    The legacy samples provide absolute axis observability and current-view
    angular diversity. This adapter is intentionally explicit about missing
    range, visibility, and safety fields; it does not invent simulator truth.
    """

    closing_observability = _probability_or_default(
        input_features.get("candidate_closing_axis_observability"), 0.0
    )
    approach_observability = _probability_or_default(
        input_features.get("candidate_approach_axis_observability"), 0.0
    )
    closing_alignment = 1.0 - closing_observability
    approach_alignment = 1.0 - approach_observability
    lateral_alignment = float(
        np.sqrt(max(0.0, 1.0 - closing_alignment**2 - approach_alignment**2))
    )
    parallax = _probability_or_default(input_features.get("angular_diversity"), 0.0)
    return {
        "candidate_ray_alignment_closing": closing_alignment,
        "candidate_ray_alignment_lateral": lateral_alignment,
        "candidate_ray_alignment_approach": approach_alignment,
        "history_parallax_min": parallax,
        "history_parallax_mean": parallax,
        "history_parallax_max": parallax,
        "candidate_position_alignment_closing": 0.0,
        "candidate_position_alignment_lateral": 0.0,
        "candidate_position_alignment_approach": 0.0,
        "candidate_range_m": 0.0,
        "candidate_position_available": 0.0,
        "predicted_target_visibility": _probability_or_default(
            input_features.get("predicted_target_visibility"), 0.5
        ),
        "depth_noise_at_range": _nonnegative_or_default(
            input_features.get("depth_noise_at_range"), 0.0
        ),
        "occlusion_risk": _probability_or_default(
            input_features.get("occlusion_risk"), 0.5
        ),
        "camera_reachable": _boolean_or_default(
            input_features.get("camera_reachable"), 0.5
        ),
        "camera_safe": _boolean_or_default(input_features.get("camera_safe"), 0.5),
    }


def uncertainty_frame(
    *,
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closing = _unit(closing_axis_world, "closing axis")
    approach_raw = _unit(approach_axis_world, "approach axis")
    approach = approach_raw - float(np.dot(approach_raw, closing)) * closing
    norm = float(np.linalg.norm(approach))
    if norm <= 1e-6:
        raise ValueError("closing and approach axes must not be parallel")
    approach /= norm
    lateral = np.cross(approach, closing)
    lateral /= float(np.linalg.norm(lateral))
    return closing, lateral, approach


def camera_position_from_standoff(
    view_direction_world: Sequence[float],
    *,
    belief_centroid_world: Sequence[float],
    standoff_m: float,
) -> list[float]:
    """Place a camera along a candidate ray at a fixed standoff.

    Discrete view catalogues store only a direction. This reconstructs the
    position implied by looking at the current belief centroid from that
    direction, so range-dependent features are defined without reading the
    simulator camera pose. The standoff is a declared layout constant, not a
    measurement, and callers must record it next to the ranking.
    """

    if not np.isfinite(standoff_m) or standoff_m <= 0.0:
        raise ValueError("camera standoff must be finite and positive")
    direction = _unit(view_direction_world, "view direction")
    centroid = _vector3(belief_centroid_world, "belief centroid")
    return (centroid - direction * float(standoff_m)).tolist()


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


def _probability_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("probability feature must be finite and in [0, 1]")
    return result


def _nonnegative_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("nonnegative feature must be finite")
    return result


def _boolean_or_default(value: bool | None, default: float) -> float:
    if value is None:
        return float(default)
    return float(bool(value))
