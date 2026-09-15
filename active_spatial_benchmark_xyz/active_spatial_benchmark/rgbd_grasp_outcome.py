"""Post-execution grasp verification from inference-visible RGB-D evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from .appearance_runtime import (
    dense_patch_match,
    descriptor_cosine,
    encode_candidate_appearance,
)
from .env import robotwin_cwd
from .grasp_candidates import GraspCandidate


POST_VIEW_DIRECTIONS = {
    "side": [-1.0, 0.0, 0.0],
    "front_side_45": [-1.0, 1.0, 0.0],
    "side_top_45": [-1.0, 0.0, -1.0],
    "oblique_45": [-1.0, 1.0, -1.0],
    "topdown": [0.0, 0.0, -1.0],
}


def build_target_appearance_signature(
    candidate: GraspCandidate,
    evidence: Sequence[Mapping[str, Any]],
    *,
    support_z_m: float | None,
) -> dict[str, Any]:
    """Summarize target-local RGB-D appearance before physical execution."""

    if support_z_m is None:
        return _unavailable_signature("support_plane_not_observed")
    target_colors = []
    target_points = []
    background_colors = []
    per_view = []
    for index, row in enumerate(evidence):
        if row.get("rgb") is None:
            per_view.append(
                {
                    "frame_id": int(row.get("frame_id", index)),
                    "view": str(row.get("view", f"view_{index}")),
                    "target_sample_count": 0,
                    "background_sample_count": 0,
                    "reason": "aligned_rgb_not_available",
                }
            )
            continue
        points, colors, metadata = _backproject_rgbd(row, index=index, stride=2)
        target_mask = _candidate_region_mask(points, candidate.center_world_m, candidate)
        target_mask &= points[:, 2] > float(support_z_m) + 0.006
        local_distance = np.linalg.norm(
            points - np.asarray(candidate.center_world_m, dtype=np.float64), axis=1
        )
        background_mask = (
            (local_distance <= 0.14)
            & ~target_mask
            & (points[:, 2] >= float(support_z_m) - 0.004)
            & (points[:, 2] <= float(support_z_m) + 0.035)
        )
        target_colors.append(colors[target_mask])
        target_points.append(points[target_mask])
        background_colors.append(colors[background_mask])
        per_view.append(
            {
                **metadata,
                "target_sample_count": int(target_mask.sum()),
                "background_sample_count": int(background_mask.sum()),
            }
        )
    target_rgb = _nonempty_concat(target_colors)
    target_xyz = _nonempty_concat(target_points)
    background_rgb = _nonempty_concat(background_colors)
    if len(target_rgb) < 20:
        return {
            **_unavailable_signature("insufficient_target_local_rgbd_samples"),
            "sample_count": int(len(target_rgb)),
            "evidence": per_view,
        }
    color_center = np.median(target_rgb, axis=0)
    target_features = _color_features(target_rgb)
    color_feature_center = np.median(target_features, axis=0)
    color_distances = np.linalg.norm(
        target_features - color_feature_center, axis=1
    )
    color_mad = float(np.median(color_distances))
    background_center = np.median(background_rgb, axis=0) if len(background_rgb) else None
    background_feature_center = (
        np.median(_color_features(background_rgb), axis=0)
        if len(background_rgb)
        else None
    )
    background_distance = (
        float(np.linalg.norm(color_feature_center - background_feature_center))
        if background_feature_center is not None
        else None
    )
    discriminative = bool(
        background_distance is not None
        and background_distance >= 0.10
        and color_mad <= 0.22
    )
    voxel_count = _voxel_count(target_xyz, 0.004)
    return {
        "schema_version": "spatial.target_appearance_signature.v1",
        "available": True,
        "discriminative": discriminative,
        "color_center_rgb_01": _rounded(color_center),
        "color_feature_center": _rounded(color_feature_center),
        "color_mad": round(color_mad, 7),
        "background_center_rgb_01": (
            _rounded(background_center) if background_center is not None else None
        ),
        "target_background_color_distance": (
            round(background_distance, 7)
            if background_distance is not None
            else None
        ),
        "sample_count": int(len(target_rgb)),
        "voxel_count": voxel_count,
        "position_center_world_m": _rounded(np.median(target_xyz, axis=0)),
        "support_plane_z_m": round(float(support_z_m), 7),
        "evidence_view_count": sum(
            int(row["target_sample_count"] > 0) for row in per_view
        ),
        "evidence": per_view,
        "source": "candidate_local_inference_visible_rgbd",
        "access": "inference_visible",
    }


def verify_postgrasp_outcome(
    env: Any,
    candidate: GraspCandidate,
    signature: Mapping[str, Any] | None,
    *,
    arm: str,
    output_dir: str | Path,
    views: Sequence[str] | None = None,
    appearance_encoder_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Actively observe a lifted gripper and verify target retention."""

    output_dir = Path(output_dir).resolve()
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    initial_pose = deepcopy(env.camera.get_camera().entity.get_pose())
    world_before = _world_fingerprint(env)
    view_ranking = rank_postgrasp_views(candidate)
    selected_views = list(views) if views is not None else [
        row["view"] for row in view_ranking[:3]
    ]
    evidence = []
    history = []
    try:
        for index, view in enumerate(selected_views):
            capture = _capture_view(env, arm, str(view), initial_pose)
            image_path = views_dir / f"{index:02d}_{view}.png"
            depth_path = views_dir / f"{index:02d}_{view}_depth_m.npy"
            camera_path = views_dir / f"{index:02d}_{view}_camera.json"
            imageio.imwrite(image_path, capture["rgb"])
            np.save(depth_path, capture["depth_m"], allow_pickle=False)
            camera_path.write_text(
                _camera_json(capture, view=str(view), frame_id=index),
                encoding="utf-8",
            )
            evidence.append(
                {
                    "frame_id": index,
                    "view": str(view),
                    "rgb": capture["rgb"],
                    "depth_m": capture["depth_m"],
                    "intrinsic_cv": capture["intrinsic_cv"],
                    "extrinsic_cv": capture["extrinsic_cv"],
                    "access": "inference_visible_server_side",
                }
            )
            history.append(
                {
                    "frame_id": index,
                    "view": str(view),
                    "image": str(image_path.relative_to(output_dir)),
                    "depth_m": str(depth_path.relative_to(output_dir)),
                    "camera": str(camera_path.relative_to(output_dir)),
                    "access": "inference_visible_server_side",
                }
            )
    finally:
        with robotwin_cwd():
            env.camera.get_camera().entity.set_pose(initial_pose)
            env.task._update_render()
    jaw_center, finger_points = _gripper_kinematics(env, arm)
    result = analyze_postgrasp_evidence(
        candidate,
        signature or {},
        evidence,
        jaw_center_world_m=jaw_center,
        finger_points_world_m=finger_points,
    )
    if appearance_encoder_config is not None:
        try:
            post_shadow = encode_candidate_appearance(
                candidate,
                evidence,
                center_world_m=jaw_center,
                output_dir=output_dir / "frozen_appearance_shadow",
                encoder=appearance_encoder_config,
                stage="post_execution",
            )
        except Exception as exc:
            post_shadow = {
                "available": False,
                "reason": "shadow_encoder_exception",
                "error_type": type(exc).__name__,
                "access": "inference_visible_server_side",
            }
        pre_shadow = (signature or {}).get("frozen_appearance_shadow", {})
        cosine = None
        if pre_shadow.get("available") and post_shadow.get("available"):
            cosine = descriptor_cosine(
                pre_shadow["aggregate_descriptor"],
                post_shadow["aggregate_descriptor"],
            )
        dense_match = dense_patch_match(pre_shadow, post_shadow)
        result["frozen_appearance_shadow"] = {
            "available": cosine is not None,
            "pre_evidence_view_count": pre_shadow.get("evidence_view_count", 0),
            "post_evidence_view_count": post_shadow.get("evidence_view_count", 0),
            "pre_post_cosine": round(float(cosine), 7) if cosine is not None else None,
            "dense_patch_match": dense_match,
            "pre_signature": pre_shadow,
            "post_signature": post_shadow,
            "controls_verdict": False,
            "reason": "shadow_until_dense_patch_matching_is_calibrated",
            "access": "inference_visible_server_side",
        }
    world_after = _world_fingerprint(env)
    world_delta = _fingerprint_delta(world_before, world_after)
    result.update(
        {
            "history": history,
            "observed_view_sequence": [str(view) for view in selected_views],
            "candidate_view_scores": view_ranking,
            "view_selection_source": "candidate_axis_observability",
            "world_frozen_during_observation": world_delta < 1e-6,
            "world_fingerprint_delta": world_delta,
        }
    )
    (output_dir / "query_result.json").write_text(
        _json_dumps(result) + "\n", encoding="utf-8"
    )
    return result


def analyze_postgrasp_evidence(
    candidate: GraspCandidate,
    signature: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    jaw_center_world_m: Sequence[float],
    finger_points_world_m: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Compare post-lift observations with a pre-execution target signature."""

    jaw_center = np.asarray(jaw_center_world_m, dtype=np.float64)
    support_z = signature.get("support_plane_z_m")
    signature_available = bool(signature.get("available"))
    discriminative = bool(signature.get("discriminative"))
    color_feature_center = np.asarray(
        signature.get("color_feature_center", [np.nan] * 4), dtype=np.float64
    )
    color_mad = float(signature.get("color_mad", 1.0))
    color_threshold = float(np.clip(0.08 + 3.0 * color_mad, 0.14, 0.28))
    expected_matches = []
    original_matches = []
    expected_points_all = []
    per_view = []
    original_center = np.asarray(candidate.center_world_m, dtype=np.float64)
    for index, row in enumerate(evidence):
        points, colors, metadata = _backproject_rgbd(row, index=index, stride=2)
        expected_region = _candidate_region_mask(points, jaw_center, candidate)
        original_region = _candidate_region_mask(points, original_center, candidate)
        if signature_available and np.isfinite(color_feature_center).all():
            color_match = (
                np.linalg.norm(
                    _color_features(colors) - color_feature_center, axis=1
                )
                <= color_threshold
            )
        else:
            color_match = np.zeros(len(points), dtype=bool)
        expected = expected_region & color_match
        original = original_region & color_match
        expected_matches.append(points[expected])
        original_matches.append(points[original])
        expected_points_all.append(points[expected_region])
        per_view.append(
            {
                **metadata,
                "expected_region_sample_count": int(expected_region.sum()),
                "original_region_sample_count": int(original_region.sum()),
                "expected_color_match_count": int(expected.sum()),
                "original_color_match_count": int(original.sum()),
            }
        )
    expected_xyz = _nonempty_concat(expected_matches)
    original_xyz = _nonempty_concat(original_matches)
    expected_region_xyz = _nonempty_concat(expected_points_all)
    expected_voxels = _voxelize(expected_xyz, 0.004)
    original_voxels = _voxelize(original_xyz, 0.004)
    expected_view_count = sum(
        int(row["expected_color_match_count"] >= 3) for row in per_view
    )
    original_view_count = sum(
        int(row["original_color_match_count"] >= 3) for row in per_view
    )
    original_observed_view_count = sum(
        int(row["original_region_sample_count"] >= 3) for row in per_view
    )
    pre_voxels = int(signature.get("voxel_count", 0))
    minimum_match_voxels = int(np.clip(round(pre_voxels * 0.06), 6, 24))
    target_detected = bool(
        discriminative
        and len(expected_voxels) >= minimum_match_voxels
        and expected_view_count >= 2
    )
    post_center = (
        np.median(expected_voxels, axis=0) if len(expected_voxels) else None
    )
    jaw_distance = (
        float(np.linalg.norm(post_center - jaw_center))
        if post_center is not None
        else None
    )
    height_above_support = (
        float(post_center[2] - float(support_z))
        if post_center is not None and support_z is not None
        else None
    )
    near_gripper = bool(jaw_distance is not None and jaw_distance <= 0.075)
    lifted = bool(
        height_above_support is not None and height_above_support >= 0.05
    )
    original_still_occupied = bool(
        discriminative
        and len(original_voxels) >= minimum_match_voxels
        and original_view_count >= 1
    )
    original_vacated = bool(
        discriminative
        and original_observed_view_count >= 1
        and len(original_voxels) <= max(4, int(round(pre_voxels * 0.20)))
    )
    retained = target_detected and near_gripper and lifted
    if retained:
        verdict = "true"
        confidence = _true_confidence(
            expected_voxels=len(expected_voxels),
            minimum_voxels=minimum_match_voxels,
            evidence_views=expected_view_count,
            jaw_distance_m=jaw_distance,
            height_above_support_m=height_above_support,
            original_vacated=original_vacated,
        )
        missing_evidence = []
    elif discriminative and original_still_occupied and not target_detected:
        verdict = "false"
        confidence = float(
            np.clip(len(original_voxels) / max(minimum_match_voxels * 2.0, 1.0), 0.55, 0.98)
        )
        missing_evidence = []
    else:
        verdict = "uncertain"
        confidence = 0.0
        missing_evidence = _missing_outcome_evidence(
            signature_available=signature_available,
            discriminative=discriminative,
            expected_view_count=expected_view_count,
            expected_voxel_count=len(expected_voxels),
            minimum_match_voxels=minimum_match_voxels,
            lifted=lifted,
            near_gripper=near_gripper,
        )
    relations = {
        "appearance_tracked_to_lifted_region": _relation(
            target_detected,
            observable=discriminative,
            measurement={
                "matched_voxel_count": int(len(expected_voxels)),
                "required_voxel_count": minimum_match_voxels,
                "evidence_view_count": expected_view_count,
                "color_threshold": round(color_threshold, 7),
            },
        ),
        "near_gripper_after_lift": _relation(
            near_gripper,
            observable=jaw_distance is not None,
            measurement={"distance_m": _optional_round(jaw_distance)},
        ),
        "lifted_from_support": _relation(
            lifted,
            observable=height_above_support is not None,
            measurement={
                "height_above_support_m": _optional_round(height_above_support),
                "required_height_m": 0.05,
            },
        ),
        "original_target_region_vacated": _relation(
            original_vacated,
            observable=discriminative and original_observed_view_count >= 1,
            measurement={
                "matched_voxel_count": int(len(original_voxels)),
                "matching_view_count": original_view_count,
                "observed_view_count": original_observed_view_count,
            },
        ),
        "grasp_retained_after_lift": _relation(
            retained,
            observable=verdict != "uncertain",
            probability=confidence if verdict == "true" else 1.0 - confidence,
        ),
    }
    graph = _build_outcome_graph(
        candidate=candidate,
        jaw_center=jaw_center,
        finger_points=np.asarray(finger_points_world_m, dtype=np.float64),
        post_center=post_center,
        support_z=support_z,
        relations=relations,
        verdict=verdict,
    )
    return {
        "schema_version": "spatial.rgbd_grasp_outcome.v1",
        "query": "verify_grasp_outcome",
        "candidate_id": candidate.candidate_id,
        "verdict": verdict,
        "confidence": round(float(confidence), 6),
        "relations": relations,
        "missing_evidence": missing_evidence,
        "target_signature_summary": {
            "available": signature_available,
            "discriminative": discriminative,
            "pre_execution_voxel_count": pre_voxels,
            "target_background_color_distance": signature.get(
                "target_background_color_distance"
            ),
        },
        "measurements": {
            "post_target_center_world_m": (
                _rounded(post_center) if post_center is not None else None
            ),
            "jaw_center_world_m": _rounded(jaw_center),
            "expected_match_voxel_count": int(len(expected_voxels)),
            "expected_region_raw_sample_count": int(len(expected_region_xyz)),
            "original_match_voxel_count": int(len(original_voxels)),
            "minimum_match_voxels": minimum_match_voxels,
            "jaw_distance_m": _optional_round(jaw_distance),
            "height_above_support_m": _optional_round(height_above_support),
        },
        "evidence": per_view,
        "grasp_outcome_graph": graph,
        "recommended_action": _recommended_action(verdict),
        "source": "active_multiview_rgbd_appearance_tracking_and_robot_kinematics",
        "oracle_object_geometry_used": False,
        "task_success_label_used": False,
        "limitation": (
            "Appearance tracking can be uncertain for transparent, reflective, textureless, "
            "or gripper-colored targets and does not infer contact force."
        ),
        "access": "inference_visible",
    }


def _capture_view(env: Any, arm: str, view: str, initial_pose: Any) -> dict[str, Any]:
    with robotwin_cwd():
        if view == "current":
            env.camera.get_camera().entity.set_pose(deepcopy(initial_pose))
        else:
            env.camera.view(arm, view)
        env.task._update_render()
        camera = env.camera.get_camera()
        camera.take_picture()
        rgba = camera.get_picture("Color")
        position = camera.get_picture("Position")
    pose = camera.entity.get_pose().to_transformation_matrix().astype(np.float64)
    return {
        "rgb": (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8),
        "depth_m": (-position[..., 2]).astype(np.float32),
        "intrinsic_cv": np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64),
        "extrinsic_cv": np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64),
        "view_direction_world": _unit(pose[:3, 0]).tolist(),
    }


def rank_postgrasp_views(candidate: GraspCandidate) -> list[dict[str, Any]]:
    """Rank lateral diagnostic views from grasp-frame observability."""

    approach = _unit(candidate.approach_axis_world)
    closing = _unit(candidate.closing_axis_world)
    rows = []
    for view, direction_value in POST_VIEW_DIRECTIONS.items():
        direction = _unit(direction_value)
        approach_observability = 1.0 - abs(float(np.dot(direction, approach)))
        closing_observability = 1.0 - abs(float(np.dot(direction, closing)))
        utility = 0.65 * approach_observability + 0.35 * closing_observability
        rows.append(
            {
                "view": view,
                "utility": round(float(utility), 6),
                "approach_observability": round(approach_observability, 6),
                "closing_observability": round(closing_observability, 6),
                "reason": "expose_target_below_jaw_and_reduce_axis_projection_degeneracy",
            }
        )
    return sorted(rows, key=lambda row: (-row["utility"], row["view"]))


def _backproject_rgbd(
    value: Mapping[str, Any], *, index: int, stride: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    access = str(value.get("access", "inference_visible"))
    if access not in {"inference_visible", "inference_visible_server_side"}:
        raise ValueError("grasp outcome RGB-D evidence must be inference-visible")
    depth = np.asarray(value.get("depth_m"), dtype=np.float64)
    rgb = np.asarray(value.get("rgb"))
    intrinsic = np.asarray(value.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(value.get("extrinsic_cv"), dtype=np.float64)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("grasp outcome evidence requires aligned HxW depth and HxWx3 RGB")
    if intrinsic.shape != (3, 3):
        raise ValueError("grasp outcome intrinsic_cv must be 3x3")
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :4]
    if extrinsic.shape != (3, 4):
        raise ValueError("grasp outcome extrinsic_cv must be 3x4 or 4x4")
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled = depth[ys, xs]
    valid = np.isfinite(sampled) & (sampled >= 0.2) & (sampled <= 1.6)
    d = sampled[valid]
    if len(d):
        pixels = np.stack([xs[valid] * d, ys[valid] * d, d], axis=-1)
        camera_points = np.linalg.solve(intrinsic, pixels.T).T
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        points = (rotation.T @ (camera_points - translation).T).T
        colors = np.asarray(rgb[ys[valid], xs[valid]], dtype=np.float64) / 255.0
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.float64)
    return points, colors, {
        "frame_id": int(value.get("frame_id", index)),
        "view": str(value.get("view", f"view_{index}")),
        "valid_depth_sample_count": int(len(points)),
        "access": access,
    }


def _candidate_region_mask(
    points: np.ndarray, center_value: Sequence[float], candidate: GraspCandidate
) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    center = np.asarray(center_value, dtype=np.float64)
    approach = _unit(candidate.approach_axis_world)
    delta = points - center
    axial = delta @ approach
    radial = np.linalg.norm(delta - axial[:, None] * approach, axis=1)
    contact_span = float(
        np.linalg.norm(candidate.left_contact_world_m - candidate.right_contact_world_m)
    )
    radial_limit = float(
        np.clip(max(0.032, candidate.opening_width_m * 0.9, contact_span * 0.9), 0.032, 0.07)
    )
    return (np.abs(axial) <= 0.065) & (radial <= radial_limit)


def _gripper_kinematics(env: Any, arm: str) -> tuple[np.ndarray, list[np.ndarray]]:
    robot = env.task.robot
    entity = getattr(robot, f"{arm}_entity")
    prefix = "fl" if arm == "left" else "fr"
    points = []
    for name in (f"{prefix}_link7", f"{prefix}_link8"):
        link = entity.find_link_by_name(name)
        if link is None:
            break
        pose = getattr(link, "entity_pose", None)
        if pose is None and hasattr(link, "get_pose"):
            pose = link.get_pose()
        if pose is None:
            break
        points.append(np.asarray(pose.p, dtype=np.float64))
    if len(points) == 2:
        return (points[0] + points[1]) / 2.0, points
    return np.asarray(env._gripper_finger_center(arm), dtype=np.float64), points


def _build_outcome_graph(
    *,
    candidate: GraspCandidate,
    jaw_center: np.ndarray,
    finger_points: np.ndarray,
    post_center: np.ndarray | None,
    support_z: float | None,
    relations: Mapping[str, Any],
    verdict: str,
) -> dict[str, Any]:
    nodes = [
        {
            "id": "gripper.jaw_center.post_lift",
            "node_type": "robot_kinematic_keypoint",
            "position_mean_world_m": _rounded(jaw_center),
            "source": "robot_kinematics",
            "access": "inference_visible",
        },
        {
            "id": "support.plane",
            "node_type": "support_plane",
            "position_mean_world_m": (
                [0.0, 0.0, round(float(support_z), 7)]
                if support_z is not None
                else None
            ),
            "source": "pre_execution_rgbd",
            "access": "inference_visible",
        },
        {
            "id": candidate.candidate_id,
            "node_type": "executed_grasp_candidate",
            "access": "inference_visible",
        },
    ]
    for index, point in enumerate(finger_points):
        nodes.append(
            {
                "id": f"gripper.finger_{index}.post_lift",
                "node_type": "robot_kinematic_keypoint",
                "position_mean_world_m": _rounded(point),
                "source": "robot_kinematics",
                "access": "inference_visible",
            }
        )
    if post_center is not None:
        nodes.append(
            {
                "id": "target.appearance_track.post_lift",
                "node_type": "appearance_tracked_keypoint",
                "position_mean_world_m": _rounded(post_center),
                "source": "active_multiview_rgbd_appearance_tracking",
                "access": "inference_visible",
            }
        )
    relation_targets = {
        "appearance_tracked_to_lifted_region": (
            candidate.candidate_id,
            "target.appearance_track.post_lift",
        ),
        "near_gripper_after_lift": (
            "target.appearance_track.post_lift",
            "gripper.jaw_center.post_lift",
        ),
        "lifted_from_support": (
            "target.appearance_track.post_lift",
            "support.plane",
        ),
        "original_target_region_vacated": (
            candidate.candidate_id,
            "support.plane",
        ),
        "grasp_retained_after_lift": (
            candidate.candidate_id,
            "gripper.jaw_center.post_lift",
        ),
    }
    edges = []
    node_ids = {node["id"] for node in nodes}
    for name, relation in relations.items():
        source, target = relation_targets[name]
        if source not in node_ids or target not in node_ids:
            continue
        edges.append(
            {
                **dict(relation),
                "source": source,
                "target": target,
                "relation": name,
            }
        )
    return {
        "schema_version": "spatial.postgrasp_outcome_graph.v1",
        "verdict": verdict,
        "nodes": nodes,
        "edges": edges,
        "access": "inference_visible",
    }


def _relation(
    value: bool,
    *,
    observable: bool = True,
    probability: float | None = None,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not observable:
        state = "unknown"
        resolved_probability = None
    else:
        state = "pass" if value else "fail"
        resolved_probability = float(value) if probability is None else float(probability)
    return {
        "state": state,
        "probability": (
            round(resolved_probability, 6)
            if resolved_probability is not None
            else None
        ),
        "measurement": dict(measurement or {}),
        "source": "active_multiview_rgbd_and_robot_kinematics",
    }


def _missing_outcome_evidence(
    *,
    signature_available: bool,
    discriminative: bool,
    expected_view_count: int,
    expected_voxel_count: int,
    minimum_match_voxels: int,
    lifted: bool,
    near_gripper: bool,
) -> list[str]:
    missing = []
    if not signature_available:
        missing.append("pre_execution_target_appearance_signature")
    elif not discriminative:
        missing.append("target_appearance_distinct_from_support")
    if expected_view_count < 2:
        missing.append("target_match_in_two_post_lift_views")
    if expected_voxel_count < minimum_match_voxels:
        missing.append("sufficient_post_lift_target_voxels")
    if not lifted:
        missing.append("target_height_above_support")
    if not near_gripper:
        missing.append("target_near_gripper")
    return missing


def _true_confidence(
    *,
    expected_voxels: int,
    minimum_voxels: int,
    evidence_views: int,
    jaw_distance_m: float | None,
    height_above_support_m: float | None,
    original_vacated: bool,
) -> float:
    voxel_score = np.clip(expected_voxels / max(2.0 * minimum_voxels, 1.0), 0.0, 1.0)
    view_score = np.clip(evidence_views / 3.0, 0.0, 1.0)
    jaw_score = (
        np.clip(1.0 - float(jaw_distance_m) / 0.075, 0.0, 1.0)
        if jaw_distance_m is not None
        else 0.0
    )
    height_score = (
        np.clip((float(height_above_support_m) - 0.05) / 0.08, 0.0, 1.0)
        if height_above_support_m is not None
        else 0.0
    )
    return float(
        np.clip(
            0.45 + 0.15 * voxel_score + 0.15 * view_score + 0.10 * jaw_score
            + 0.10 * height_score + 0.05 * float(original_vacated),
            0.0,
            0.99,
        )
    )


def _recommended_action(verdict: str) -> dict[str, Any]:
    if verdict == "true":
        return {"action": "continue_task", "reason": "grasp_retained_after_lift"}
    if verdict == "false":
        return {"action": "recover_and_regrasp", "reason": "target_not_lifted_with_gripper"}
    return {
        "action": "request_additional_evidence",
        "reason": "post_lift_grasp_outcome_uncertain",
    }


def _unavailable_signature(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "spatial.target_appearance_signature.v1",
        "available": False,
        "discriminative": False,
        "reason": reason,
        "source": "candidate_local_inference_visible_rgbd",
        "access": "inference_visible",
    }


def _nonempty_concat(values: Sequence[np.ndarray]) -> np.ndarray:
    present = [value for value in values if len(value)]
    return np.concatenate(present, axis=0) if present else np.empty((0, 3), dtype=np.float64)


def _color_features(colors: np.ndarray) -> np.ndarray:
    """Use chromaticity and saturation so exposure changes do not break tracking."""

    values = np.asarray(colors, dtype=np.float64)
    totals = np.maximum(values.sum(axis=1, keepdims=True), 1e-6)
    chromaticity = values / totals
    saturation = (values.max(axis=1) - values.min(axis=1))[:, None]
    return np.concatenate([chromaticity, 0.5 * saturation], axis=1)


def _voxelize(points: np.ndarray, size_m: float) -> np.ndarray:
    if not len(points):
        return points
    indices = np.floor(points / size_m).astype(np.int64)
    _, selected = np.unique(indices, axis=0, return_index=True)
    return points[np.sort(selected)]


def _voxel_count(points: np.ndarray, size_m: float) -> int:
    return int(len(_voxelize(points, size_m)))


def _world_fingerprint(env: Any) -> dict[str, np.ndarray]:
    result = {}
    with robotwin_cwd():
        actors = env.task.scene.get_all_actors()
    for index, actor in enumerate(actors):
        pose = actor.get_pose()
        result[f"{index}:{actor.get_name()}"] = np.concatenate([pose.p, pose.q]).astype(np.float64)
    robot = env.task.robot
    result["robot:left_qpos"] = np.asarray(robot.left_entity.get_qpos(), dtype=np.float64)
    result["robot:right_qpos"] = np.asarray(robot.right_entity.get_qpos(), dtype=np.float64)
    return result


def _fingerprint_delta(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]
) -> float:
    if set(before) != set(after):
        return float("inf")
    return max(
        (float(np.max(np.abs(before[key] - after[key]))) for key in before),
        default=0.0,
    )


def _camera_json(capture: Mapping[str, Any], *, view: str, frame_id: int) -> str:
    return _json_dumps(
        {
            "schema_version": "spatial.inference_camera_calibration.v1",
            "view": view,
            "frame_id": frame_id,
            "intrinsic_cv": np.asarray(capture["intrinsic_cv"]).tolist(),
            "extrinsic_cv": np.asarray(capture["extrinsic_cv"]).tolist(),
            "view_direction_world": capture["view_direction_world"],
            "access": "inference_visible",
        }
    ) + "\n"


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _unit(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or norm <= 1e-9:
        raise ValueError("grasp outcome geometry requires a nonzero 3-vector")
    return result / norm


def _rounded(value: Any) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), 7).tolist()


def _optional_round(value: float | None) -> float | None:
    return round(float(value), 7) if value is not None else None
