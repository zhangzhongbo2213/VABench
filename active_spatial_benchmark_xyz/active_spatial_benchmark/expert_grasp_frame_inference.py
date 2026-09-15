"""Inference, sparse-graph serialization, and fusion for learned grasp frames."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .expert_grasp_frame_dataset import GRASP_FRAME_KEYPOINT_NAMES
from .expert_grasp_frame_model import ExpertGraspFrameNet
from .learned_pregrasp import (
    DEPTH_MEAN_M,
    DEPTH_STD_M,
    backproject_with_covariance,
    softargmax_2d,
)


@torch.no_grad()
def predict_expert_grasp_frame(
    model: ExpertGraspFrameNet,
    rgbd: torch.Tensor,
    intent_embedding: torch.Tensor,
    camera: Mapping[str, Any],
    *,
    original_image_size: tuple[int, int],
    frame_id: str,
    target: str,
    semantic_role: str,
    view: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    output = model(
        rgbd.unsqueeze(0).to(resolved_device),
        intent_embedding.unsqueeze(0).to(resolved_device),
    )
    heatmaps = output["heatmap_logits"]
    xy, covariance = softargmax_2d(heatmaps)
    heatmap_height, heatmap_width = heatmaps.shape[-2:]
    original_height, original_width = original_image_size
    scale_x = (original_width - 1) / max(heatmap_width - 1, 1)
    scale_y = (original_height - 1) / max(heatmap_height - 1, 1)
    xy = xy[0].cpu().numpy()
    covariance = covariance[0].cpu().numpy()
    depth_m = (
        output["depth_normalized"][0].float().cpu().numpy() * DEPTH_STD_M
        + DEPTH_MEAN_M
    ).clip(0.2, 1.6)
    depth_variance = (
        output["depth_log_variance"][0].float().exp().cpu().numpy()
        * DEPTH_STD_M**2
    )
    visibility = torch.sigmoid(output["visibility_logits"])[0].cpu().numpy()
    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
    points = []
    covariances = []
    evidence = []
    for index, name in enumerate(GRASP_FRAME_KEYPOINT_NAMES):
        pixel = np.asarray([xy[index, 0] * scale_x, xy[index, 1] * scale_y])
        pixel_covariance = np.asarray(
            [
                [covariance[index, 0, 0] * scale_x**2, covariance[index, 0, 1] * scale_x * scale_y],
                [covariance[index, 1, 0] * scale_x * scale_y, covariance[index, 1, 1] * scale_y**2],
            ]
        )
        point, point_covariance = backproject_with_covariance(
            pixel,
            float(depth_m[index]),
            pixel_covariance,
            float(max(depth_variance[index], 1e-7)),
            intrinsic,
            extrinsic,
        )
        points.append(point)
        covariances.append(point_covariance)
        evidence.append(
            {
                "name": name,
                "view": view,
                "pixel_uv": np.round(pixel, 4).tolist(),
                "camera_depth_m": round(float(depth_m[index]), 6),
                "visibility": round(float(visibility[index]), 6),
            }
        )
    point_array = np.stack(points)
    closing_axis = _unit(point_array[2] - point_array[1])
    approach_axis = _unit(
        output["approach_axis_world"][0].float().cpu().numpy()
    )
    confidence = float(
        torch.sigmoid(output["frame_confidence_logits"])[0].cpu()
    ) * float(np.mean(visibility))
    orientation_std_rad = math.radians(5.0 + 35.0 * (1.0 - confidence))
    return {
        "schema_version": "spatial.learned_grasp_frame.v1",
        "frame_id": frame_id,
        "target": target,
        "semantic_role": semantic_role,
        "view": view,
        "center_world_m": np.round(point_array[0], 7).tolist(),
        "left_contact_world_m": np.round(point_array[1], 7).tolist(),
        "right_contact_world_m": np.round(point_array[2], 7).tolist(),
        "point_covariance_m2": {
            name: np.round(value, 9).tolist()
            for name, value in zip(GRASP_FRAME_KEYPOINT_NAMES, covariances)
        },
        "approach_axis_world": np.round(approach_axis, 7).tolist(),
        "closing_axis_world": np.round(closing_axis, 7).tolist(),
        "opening_width_m": round(
            float(
                np.clip(
                    np.exp(float(output["opening_width_log_m"][0].float().cpu())),
                    0.005,
                    0.15,
                )
            ),
            7,
        ),
        "contact_separation_m": round(
            float(np.linalg.norm(point_array[2] - point_array[1])), 7
        ),
        "orientation_covariance_rad2": np.round(
            np.eye(3) * orientation_std_rad**2, 8
        ).tolist(),
        "confidence": round(confidence, 6),
        "visibility": round(float(np.mean(visibility)), 6),
        "evidence": evidence,
        "source": "learned_rgbd_frozen_intent_expert_grasp_frame",
        "access": "inference_visible",
    }


def fuse_expert_grasp_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("grasp frame fusion requires observations")
    frames = _align_contact_order(frames)
    first = frames[0]
    frame_weights, fusion_diagnostics = _robust_fusion_weights(frames)
    fused_points = {}
    fused_covariances = {}
    for name, field in zip(
        GRASP_FRAME_KEYPOINT_NAMES,
        ("center_world_m", "left_contact_world_m", "right_contact_world_m"),
    ):
        precision_sum = np.zeros((3, 3), dtype=np.float64)
        weighted_sum = np.zeros(3, dtype=np.float64)
        for frame, frame_weight in zip(frames, frame_weights):
            covariance = np.asarray(frame["point_covariance_m2"][name], dtype=np.float64)
            covariance = covariance + np.eye(3) * 1e-8
            precision = np.linalg.pinv(covariance)
            precision_sum += frame_weight * precision
            weighted_sum += frame_weight * precision @ np.asarray(
                frame[field], dtype=np.float64
            )
        fused_covariance = np.linalg.pinv(precision_sum)
        fused_points[field] = fused_covariance @ weighted_sum
        # The information-form covariance describes sensor noise, but not
        # disagreement between views. Keep that disagreement as explicit
        # epistemic uncertainty so more conflicting observations do not make
        # the graph falsely certain.
        observations = np.stack(
            [np.asarray(frame[field], dtype=np.float64) for frame in frames]
        )
        normalized_weights = frame_weights / max(float(np.sum(frame_weights)), 1e-12)
        deltas = observations - fused_points[field]
        between_covariance = np.einsum(
            "n,ni,nj->ij", normalized_weights, deltas, deltas
        )
        fused_covariance = fused_covariance + between_covariance
        fused_covariances[name] = fused_covariance
    weights = frame_weights
    approach = np.average(
        np.stack([np.asarray(frame["approach_axis_world"]) for frame in frames]),
        axis=0,
        weights=weights,
    )
    closing = _unit(
        fused_points["right_contact_world_m"]
        - fused_points["left_contact_world_m"]
    )
    orientation_variances = np.stack(
        [np.diag(np.asarray(frame["orientation_covariance_rad2"])) for frame in frames]
    )
    fused_orientation_variance = 1.0 / np.sum(
        frame_weights[:, None] / np.maximum(orientation_variances, 1e-8), axis=0
    )
    result = {
        "schema_version": "spatial.learned_grasp_frame.v1",
        "frame_id": str(first["frame_id"]),
        "target": str(first["target"]),
        "semantic_role": str(first["semantic_role"]),
        "view": "fused",
        **{key: np.round(value, 7).tolist() for key, value in fused_points.items()},
        "point_covariance_m2": {
            name: np.round(value, 9).tolist()
            for name, value in fused_covariances.items()
        },
        "approach_axis_world": np.round(_unit(approach), 7).tolist(),
        "closing_axis_world": np.round(closing, 7).tolist(),
        "opening_width_m": round(float(np.average([frame["opening_width_m"] for frame in frames], weights=weights)), 7),
        "contact_separation_m": round(float(np.linalg.norm(fused_points["right_contact_world_m"] - fused_points["left_contact_world_m"])), 7),
        "orientation_covariance_rad2": np.diag(fused_orientation_variance).tolist(),
        "confidence": round(
            float(
                1.0
                - np.prod(
                    [
                        1.0
                        - min(float(frame["confidence"]) * weight, 0.999)
                        for frame, weight in zip(frames, fusion_diagnostics["weight_factors"])
                    ]
                )
            ),
            6,
        ),
        "visibility": round(float(np.average([frame["visibility"] for frame in frames], weights=weights)), 6),
        "evidence": [evidence for frame in frames for evidence in frame.get("evidence", ())],
        "fusion_diagnostics": fusion_diagnostics,
        "source": "robust_information_form_multiview_learned_grasp_frame_fusion",
        "access": "inference_visible",
    }
    return result


def _robust_fusion_weights(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return confidence/visibility weights with deterministic consensus gating.

    A view can be visually confident while still producing a frame that is
    inconsistent with the other views. For the small fixed view catalogue we
    can enumerate mutually consistent frame subsets and use the highest-weight
    majority clique. Views outside a majority clique are strongly suppressed;
    when no majority exists, all observations remain active and their spread
    is preserved as uncertainty instead of forcing a brittle consensus.
    """

    base_weights = np.asarray(
        [
            max(float(frame.get("confidence", 0.0)), 1e-3)
            * max(float(frame.get("visibility", 0.0)), 1e-3)
            for frame in frames
        ],
        dtype=np.float64,
    )
    point_fields = (
        "center_world_m",
        "left_contact_world_m",
        "right_contact_world_m",
    )
    point_stack = np.stack(
        [
            [np.asarray(frame[field], dtype=np.float64) for field in point_fields]
            for frame in frames
        ]
    )
    consensus = np.median(point_stack, axis=0)
    point_residuals_m = np.sqrt(
        np.mean(np.sum((point_stack - consensus[None, ...]) ** 2, axis=-1), axis=-1)
    )
    pairwise_residuals_m = np.sqrt(
        np.mean(
            np.sum(
                (point_stack[:, None, ...] - point_stack[None, ...]) ** 2,
                axis=-1,
            ),
            axis=-1,
        )
    )
    median_residual = float(np.median(point_residuals_m))
    mad = float(np.median(np.abs(point_residuals_m - median_residual)))
    median_opening_width_m = float(
        np.median([float(frame["opening_width_m"]) for frame in frames])
    )
    cutoff_m = float(np.clip(max(0.045, 1.25 * median_opening_width_m), 0.045, 0.075))
    selected_indices = list(range(len(frames)))
    consensus_status = "all_views"
    if len(frames) < 3:
        factors = np.ones(len(frames), dtype=np.float64)
    else:
        selected_indices, competitor_weight_ratio = _maximum_consistent_frame_clique(
            pairwise_residuals_m,
            base_weights,
            cutoff_m=cutoff_m,
        )
        consensus_size = (len(frames) + 1) // 2
        has_dominant_consensus = (
            competitor_weight_ratio is None or competitor_weight_ratio <= 0.95
        )
        if len(selected_indices) >= consensus_size and has_dominant_consensus:
            factors = np.full(len(frames), 0.05, dtype=np.float64)
            factors[selected_indices] = 1.0
            consensus_status = "dominant_consensus"
        else:
            factors = np.ones(len(frames), dtype=np.float64)
            selected_indices = list(range(len(frames)))
            consensus_status = "ambiguous_no_dominant_consensus"
    if len(frames) < 3:
        competitor_weight_ratio = None
    effective_weights = base_weights * factors
    if float(np.sum(effective_weights)) <= 1e-8:
        effective_weights = base_weights.copy()
    rejected = [
        str(frame.get("view", index))
        for index, (frame, factor) in enumerate(zip(frames, factors))
        if factor < 0.25
    ]
    diagnostics = {
        "schema_version": "spatial.grasp_frame_fusion_diagnostics.v1",
        "method": "majority_clique_robust_information_fusion",
        "frame_views": [str(frame.get("view", index)) for index, frame in enumerate(frames)],
        "base_weights": np.round(base_weights, 6).tolist(),
        "weight_factors": np.round(factors, 6).tolist(),
        "effective_weights": np.round(effective_weights, 6).tolist(),
        "point_residuals_m": np.round(point_residuals_m, 7).tolist(),
        "pairwise_frame_residuals_m": np.round(pairwise_residuals_m, 7).tolist(),
        "consensus_point_world_m": {
            field: np.round(value, 7).tolist()
            for field, value in zip(point_fields, consensus)
        },
        "median_residual_m": round(median_residual, 7),
        "mad_residual_m": round(mad, 7),
        "robust_cutoff_m": round(cutoff_m, 7),
        "consensus_status": consensus_status,
        "same_size_competitor_weight_ratio": (
            round(competitor_weight_ratio, 6)
            if competitor_weight_ratio is not None
            else None
        ),
        "selected_consensus_views": [
            str(frames[index].get("view", index)) for index in selected_indices
        ],
        "rejected_views": rejected,
        "access": "inference_visible",
    }
    return effective_weights, diagnostics


def _maximum_consistent_frame_clique(
    pairwise_residuals_m: np.ndarray,
    base_weights: np.ndarray,
    *,
    cutoff_m: float,
) -> tuple[list[int], float | None]:
    """Enumerate the maximum-weight mutually consistent subset.

    The active tool currently has at most six views, so exact enumeration is
    simpler and more auditable than introducing a clustering dependency.
    """

    count = len(base_weights)
    candidates: list[tuple[list[int], float]] = []
    for mask in range(1, 1 << count):
        indices = [index for index in range(count) if mask & (1 << index)]
        if any(
            pairwise_residuals_m[left, right] > cutoff_m
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        ):
            continue
        candidates.append((indices, float(np.sum(base_weights[indices]))))
    maximum_size = max(len(indices) for indices, _ in candidates)
    maximum_cliques = sorted(
        (
            (indices, weight)
            for indices, weight in candidates
            if len(indices) == maximum_size
        ),
        key=lambda item: (item[1], int(0 in item[0])),
        reverse=True,
    )
    best_indices, best_weight = maximum_cliques[0]
    competitor_ratio = (
        maximum_cliques[1][1] / max(best_weight, 1e-12)
        if len(maximum_cliques) > 1
        else None
    )
    return best_indices, competitor_ratio


def _align_contact_order(
    frames: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference_left = np.asarray(frames[0]["left_contact_world_m"], dtype=np.float64)
    reference_right = np.asarray(frames[0]["right_contact_world_m"], dtype=np.float64)
    reference_axis = _unit(
        np.asarray(frames[0]["closing_axis_world"], dtype=np.float64)
    )
    result = []
    for raw_frame in frames:
        frame = dict(raw_frame)
        left = np.asarray(frame["left_contact_world_m"], dtype=np.float64)
        right = np.asarray(frame["right_contact_world_m"], dtype=np.float64)
        direct = np.linalg.norm(left - reference_left) + np.linalg.norm(right - reference_right)
        swapped = np.linalg.norm(right - reference_left) + np.linalg.norm(left - reference_right)
        closing_axis = np.asarray(frame["closing_axis_world"], dtype=np.float64)
        axis_norm = float(np.linalg.norm(closing_axis))
        should_swap = (
            float(np.dot(closing_axis / axis_norm, reference_axis)) < 0.0
            if axis_norm > 1e-8
            else swapped < direct
        )
        if should_swap:
            frame["left_contact_world_m"], frame["right_contact_world_m"] = (
                frame["right_contact_world_m"],
                frame["left_contact_world_m"],
            )
            covariance = dict(frame["point_covariance_m2"])
            covariance["left_contact"], covariance["right_contact"] = (
                covariance["right_contact"],
                covariance["left_contact"],
            )
            frame["point_covariance_m2"] = covariance
            frame["closing_axis_world"] = (
                -np.asarray(frame["closing_axis_world"], dtype=np.float64)
            ).tolist()
        result.append(frame)
    return result


def build_grasp_frame_graph(frame: Mapping[str, Any]) -> dict[str, Any]:
    frame_id = str(frame["frame_id"])
    point_specs = (
        ("center", "grasp_center", "center_world_m", "grasp_center"),
        ("left_contact", "left_contact", "left_contact_world_m", "left_contact"),
        ("right_contact", "right_contact", "right_contact_world_m", "right_contact"),
    )
    nodes = [
        {
            "id": frame_id,
            "node_type": "grasp_frame",
            "semantic_type": frame["semantic_role"],
            "entity_id": frame["target"],
            "attributes": {
                "approach_axis_world": frame["approach_axis_world"],
                "closing_axis_world": frame["closing_axis_world"],
                "opening_width_m": frame["opening_width_m"],
                "confidence": frame["confidence"],
                "orientation_covariance_rad2": frame[
                    "orientation_covariance_rad2"
                ],
            },
            "source": frame["source"],
            "access": "inference_visible",
        }
    ]
    edges = []
    for suffix, semantic_type, field, covariance_key in point_specs:
        node_id = f"{frame_id}.{suffix}"
        nodes.append(
            {
                "id": node_id,
                "node_type": "grasp_frame_point",
                "semantic_type": semantic_type,
                "entity_id": frame["target"],
                "position_mean_world_m": frame[field],
                "position_covariance_m2": frame["point_covariance_m2"][covariance_key],
                "source": frame["source"],
                "access": "inference_visible",
            }
        )
        edges.append(
            {
                "source": frame_id,
                "target": node_id,
                "relation": "has_frame_point",
                "probability": frame["confidence"],
                "access": "inference_visible",
            }
        )
    edges.append(
        {
            "source": f"{frame_id}.left_contact",
            "target": f"{frame_id}.right_contact",
            "relation": "opposed_contact_pair",
            "measurement": {"separation_m": frame["contact_separation_m"]},
            "probability": frame["confidence"],
            "access": "inference_visible",
        }
    )
    return {
        "schema_version": "spatial.task_conditioned_grasp_frame_graph.v1",
        "access": "inference_visible",
        "nodes": nodes,
        "edges": edges,
        "evidence": list(frame.get("evidence", ())),
    }


def _unit(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError("grasp frame axis is degenerate")
    return result / norm
