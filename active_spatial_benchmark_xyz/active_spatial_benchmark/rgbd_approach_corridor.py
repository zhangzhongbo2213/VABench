"""Inference-visible RGB-D checks for a grasp candidate's final approach."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .grasp_candidates import GraspCandidate


@dataclass(frozen=True)
class ApproachCorridorConfig:
    """Conservative geometry thresholds for the observed final approach segment."""

    depth_stride_px: int = 3
    min_depth_m: float = 0.2
    max_depth_m: float = 1.6
    voxel_size_m: float = 0.004
    support_bin_m: float = 0.003
    support_exclusion_m: float = 0.008
    corridor_padding_m: float = 0.012
    minimum_corridor_radius_m: float = 0.022
    maximum_corridor_radius_m: float = 0.055
    minimum_evidence_views: int = 2
    minimum_local_voxels: int = 40
    blocked_voxel_threshold: int = 4


def analyze_rgbd_approach_corridor(
    candidate: GraspCandidate,
    evidence: Sequence[Mapping[str, Any]],
    *,
    config: ApproachCorridorConfig | None = None,
) -> dict[str, Any]:
    """Check observed geometry around the pregrasp-to-grasp center segment.

    This intentionally does not claim mesh-complete collision checking. It uses
    only the RGB-D frames and camera calibration available to inference.
    """

    cfg = config or ApproachCorridorConfig()
    points_by_view: list[np.ndarray] = []
    evidence_rows: list[dict[str, Any]] = []
    for index, value in enumerate(evidence):
        points, row = _backproject_evidence(value, index=index, config=cfg)
        if len(points):
            points_by_view.append(points)
        evidence_rows.append(row)

    if points_by_view:
        points = _voxelize(np.concatenate(points_by_view, axis=0), cfg.voxel_size_m)
    else:
        points = np.empty((0, 3), dtype=np.float64)
    center = np.asarray(candidate.center_world_m, dtype=np.float64)
    pregrasp = np.asarray(candidate.pregrasp_center_world_m, dtype=np.float64)
    segment = center - pregrasp
    segment_length = float(np.linalg.norm(segment))
    contact_span = float(
        np.linalg.norm(candidate.left_contact_world_m - candidate.right_contact_world_m)
    )
    corridor_radius = float(
        np.clip(
            max(
                cfg.minimum_corridor_radius_m,
                candidate.opening_width_m * 0.5 + cfg.corridor_padding_m,
                contact_span * 0.5 + cfg.corridor_padding_m,
            ),
            cfg.minimum_corridor_radius_m,
            cfg.maximum_corridor_radius_m,
        )
    )
    support_z, support_voxels = _estimate_support_z(points, center, cfg)
    local_points = _local_points(points, pregrasp, center, corridor_radius)
    local_voxel_count = int(len(local_points))
    observed_views = {
        row["view"] for row in evidence_rows if row["valid_depth_point_count"] > 0
    }
    coverage_reasons = []
    if len(observed_views) < cfg.minimum_evidence_views:
        coverage_reasons.append("insufficient_distinct_rgbd_views")
    if local_voxel_count < cfg.minimum_local_voxels:
        coverage_reasons.append("insufficient_local_depth_geometry")
    if support_z is None:
        coverage_reasons.append("support_plane_not_observed")
    if segment_length <= 1e-6:
        coverage_reasons.append("degenerate_approach_segment")

    obstacle_points = np.empty((0, 3), dtype=np.float64)
    minimum_clearance = None
    excluded_support_count = 0
    excluded_target_count = 0
    if support_z is not None and segment_length > 1e-6 and len(local_points):
        above_support = local_points[:, 2] > support_z + cfg.support_exclusion_m
        excluded_support_count = int((~above_support).sum())
        elevated = local_points[above_support]
        target_mask = _target_allowance_mask(
            elevated,
            candidate,
            support_z=support_z,
            corridor_radius=corridor_radius,
        )
        excluded_target_count = int(target_mask.sum())
        non_target = elevated[~target_mask]
        distances, progress = _point_segment_distance(non_target, pregrasp, center)
        if len(distances):
            minimum_clearance = float(np.min(distances))
        # Ignore geometry just beyond the grasp center. The target allowance
        # accounts for intended contact near the end of this closed segment.
        obstacle_mask = (
            (progress >= 0.0) & (progress <= 1.0) & (distances <= corridor_radius)
        )
        obstacle_points = non_target[obstacle_mask]

    obstacle_voxel_count = int(len(obstacle_points))
    blocked = obstacle_voxel_count >= cfg.blocked_voxel_threshold
    coverage_sufficient = not coverage_reasons
    clear = coverage_sufficient and not blocked
    return {
        "schema_version": "spatial.rgbd_approach_corridor.v1",
        "state": "pass" if clear else "fail",
        "clear": clear,
        "coverage_sufficient": coverage_sufficient,
        "coverage_reasons": coverage_reasons,
        "segment": {
            "pregrasp_center_world_m": _rounded(pregrasp),
            "grasp_center_world_m": _rounded(center),
            "length_m": round(segment_length, 7),
            "radius_m": round(corridor_radius, 7),
        },
        "support_plane_z_m": round(float(support_z), 7)
        if support_z is not None
        else None,
        "support_plane_voxel_count": int(support_voxels),
        "fused_voxel_count": int(len(points)),
        "local_voxel_count": local_voxel_count,
        "excluded_support_voxel_count": excluded_support_count,
        "excluded_target_allowance_voxel_count": excluded_target_count,
        "obstacle_voxel_count": obstacle_voxel_count,
        "blocked_voxel_threshold": cfg.blocked_voxel_threshold,
        "minimum_non_target_clearance_m": (
            round(minimum_clearance, 7) if minimum_clearance is not None else None
        ),
        "evidence_view_count": len(observed_views),
        "evidence_views": sorted(observed_views),
        "evidence": evidence_rows,
        "source": "inference_visible_multiview_rgbd",
        "coverage": "observed_pregrasp_to_grasp_center_corridor_only",
        "target_exclusion": "candidate_centered_geometric_allowance_without_segmentation",
        "limitation": (
            "This check does not cover hidden surfaces, the full gripper mesh, or "
            "the current-to-pregrasp articulated robot sweep."
        ),
        "access": "inference_visible",
    }


def _backproject_evidence(
    value: Mapping[str, Any],
    *,
    index: int,
    config: ApproachCorridorConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    access = str(value.get("access", "inference_visible"))
    if access not in {"inference_visible", "inference_visible_server_side"}:
        raise ValueError("RGB-D corridor evidence must be inference-visible")
    depth_value = value.get("depth_m")
    if depth_value is None:
        raise ValueError("RGB-D corridor evidence requires depth_m")
    depth = np.asarray(depth_value, dtype=np.float64)
    intrinsic = np.asarray(value.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(value.get("extrinsic_cv"), dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("RGB-D corridor depth_m must be a 2D array")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("RGB-D corridor intrinsic_cv must be finite 3x3")
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :4]
    if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("RGB-D corridor extrinsic_cv must be finite 3x4 or 4x4")
    stride = max(1, int(config.depth_stride_px))
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled = depth[ys, xs]
    valid = (
        np.isfinite(sampled)
        & (sampled >= config.min_depth_m)
        & (sampled <= config.max_depth_m)
    )
    if not np.any(valid):
        points = np.empty((0, 3), dtype=np.float64)
    else:
        d = sampled[valid]
        homogeneous_pixels = np.stack([xs[valid] * d, ys[valid] * d, d], axis=-1)
        camera_points = np.linalg.solve(intrinsic, homogeneous_pixels.T).T
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        points = (rotation.T @ (camera_points - translation).T).T
    return points, {
        "frame_id": int(value.get("frame_id", index)),
        "view": str(value.get("view", f"view_{index}")),
        "valid_depth_point_count": int(len(points)),
        "depth_shape": list(depth.shape),
        "access": access,
    }


def _estimate_support_z(
    points: np.ndarray,
    center: np.ndarray,
    config: ApproachCorridorConfig,
) -> tuple[float | None, int]:
    if not len(points):
        return None, 0
    horizontal = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    candidates = points[
        (horizontal <= 0.18)
        & (points[:, 2] >= center[2] - 0.15)
        & (points[:, 2] <= center[2] - 0.004)
    ]
    if len(candidates) < 20:
        return None, int(len(candidates))
    bins = np.floor(candidates[:, 2] / config.support_bin_m).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[int(np.argmax(counts))]
    in_mode = candidates[np.abs(bins - mode) <= 1, 2]
    return float(np.median(in_mode)), int(len(in_mode))


def _local_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    corridor_radius: float,
) -> np.ndarray:
    if not len(points):
        return points
    lower = np.minimum(start, end) - (corridor_radius + 0.08)
    upper = np.maximum(start, end) + (corridor_radius + 0.08)
    return points[np.all((points >= lower) & (points <= upper), axis=1)]


def _target_allowance_mask(
    points: np.ndarray,
    candidate: GraspCandidate,
    *,
    support_z: float,
    corridor_radius: float,
) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    center = np.asarray(candidate.center_world_m, dtype=np.float64)
    approach = np.asarray(candidate.approach_axis_world, dtype=np.float64)
    approach /= max(float(np.linalg.norm(approach)), 1e-9)
    delta = points - center
    axial = delta @ approach
    radial = np.linalg.norm(delta - axial[:, None] * approach, axis=1)
    object_height_hint = max(float(center[2] - support_z), 0.0)
    axial_height_hint = object_height_hint * abs(float(approach[2]))
    axial_half_extent = float(np.clip(axial_height_hint + 0.018, 0.04, 0.065))
    radial_allowance = float(np.clip(corridor_radius + 0.008, 0.03, 0.06))
    return (np.abs(axial) <= axial_half_extent) & (radial <= radial_allowance)


def _point_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return np.linalg.norm(points - start, axis=1), np.zeros(len(points))
    progress = ((points - start) @ segment) / length_squared
    clipped = np.clip(progress, 0.0, 1.0)
    closest = start + clipped[:, None] * segment
    return np.linalg.norm(points - closest, axis=1), progress


def _voxelize(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if not len(points):
        return points
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        return finite
    indices = np.floor(finite / voxel_size_m).astype(np.int64)
    _, selected = np.unique(indices, axis=0, return_index=True)
    return finite[np.sort(selected)]


def _rounded(value: Any) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), 7).tolist()
