"""Isolated frozen-vision appearance encoding for inference-time shadow use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from .grasp_candidates import GraspCandidate


def encode_candidate_appearance(
    candidate: GraspCandidate,
    evidence: Sequence[Mapping[str, Any]],
    *,
    center_world_m: Sequence[float],
    output_dir: str | Path,
    encoder: Mapping[str, Any],
    stage: str,
    minimum_world_z_m: float | None = None,
    text_embedding: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Crop candidate-local RGB-D regions and encode them in an external process."""

    output_dir = Path(output_dir).resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    crop_rows = []
    for index, row in enumerate(evidence):
        crop_value = candidate_local_masked_crop(
            candidate,
            row,
            center_world_m=center_world_m,
            minimum_world_z_m=minimum_world_z_m,
        )
        if crop_value is None:
            crop_rows.append(
                {
                    "frame_id": int(row.get("frame_id", index)),
                    "view": str(row.get("view", f"view_{index}")),
                    "available": False,
                    "reason": "candidate_region_not_observed",
                }
            )
            continue
        crop, crop_mask, crop_box, pixel_count = crop_value
        request_id = f"{stage}/{index:02d}_{row.get('view', f'view_{index}')}"
        filename = hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".png"
        image_path = crops_dir / filename
        mask_path = crops_dir / filename.replace(".png", "_mask.png")
        Image.fromarray(crop).save(image_path)
        Image.fromarray((crop_mask.astype(np.uint8) * 255)).save(mask_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        requests.append(
            {
                "id": request_id,
                "image": str(image_path.relative_to(output_dir)),
                "image_sha256": digest,
                "mask": str(mask_path.relative_to(output_dir)),
                "emit_patch_tokens": True,
            }
        )
        crop_rows.append(
            {
                "frame_id": int(row.get("frame_id", index)),
                "view": str(row.get("view", f"view_{index}")),
                "available": True,
                "request_id": request_id,
                "crop_box_xyxy": crop_box,
                "candidate_region_pixel_count": pixel_count,
                "mask": str(mask_path.relative_to(output_dir)),
            }
        )
    if not requests:
        return {
            "available": False,
            "reason": "no_candidate_local_crops",
            "stage": stage,
            "crops": crop_rows,
            "access": "inference_visible",
        }
    request_path = output_dir / "requests.json"
    embedding_path = output_dir / "embeddings.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "spatial.appearance_embedding_requests.v1",
                "requests": requests,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        str(Path(str(encoder["python"])).resolve()),
        str(Path(str(encoder["script"])).resolve()),
        "--requests",
        str(request_path),
        "--output",
        str(embedding_path),
        "--snapshot",
        str(Path(str(encoder["snapshot"])).resolve()),
        "--device",
        str(encoder.get("device", "cuda")),
        "--batch-size",
        str(int(encoder.get("batch_size", 4))),
        "--patch-projection-checkpoint",
        str(Path(str(encoder["projection_checkpoint"])).resolve()),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(encoder.get("timeout_seconds", 120.0)),
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "external_appearance_encoder_failed",
            "stage": stage,
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-1000:],
            "crops": crop_rows,
            "access": "inference_visible",
        }
    store = json.loads(embedding_path.read_text(encoding="utf-8"))
    raw_embeddings = store.get("embeddings", {})
    patch_artifacts = store.get("patch_artifacts", {})
    projection = load_visual_projection(Path(str(encoder["projection_checkpoint"])))
    per_view = []
    projected = []
    for row in crop_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str):
            per_view.append(row)
            continue
        descriptor = np.asarray(raw_embeddings[request_id], dtype=np.float32)
        projected_descriptor = project_visual_descriptor(descriptor, projection)
        projected.append(projected_descriptor)
        per_view.append(
            {
                **row,
                "projected_descriptor": np.round(projected_descriptor, 7).tolist(),
                "patch_artifact": patch_artifacts.get(request_id),
            }
        )
    aggregate = _normalized(np.mean(projected, axis=0))
    result = {
        "schema_version": "spatial.frozen_appearance_signature.v1",
        "available": True,
        "stage": stage,
        "encoder_id": store.get("encoder_id"),
        "projection_checkpoint": str(
            Path(str(encoder["projection_checkpoint"])).resolve()
        ),
        "artifact_root": str(output_dir),
        "descriptor_dimension": int(len(aggregate)),
        "aggregate_descriptor": np.round(aggregate, 7).tolist(),
        "evidence_view_count": len(projected),
        "crops": per_view,
        "mode": "shadow",
        "access": "inference_visible_server_side",
    }
    if text_embedding is not None:
        text_projection = load_text_projection(
            Path(str(encoder["projection_checkpoint"]))
        )
        result["projected_text_descriptor"] = np.round(
            project_text_descriptor(
                np.asarray(text_embedding, dtype=np.float32), text_projection
            ),
            7,
        ).tolist()
    return result


def candidate_local_masked_crop(
    candidate: GraspCandidate,
    evidence: Mapping[str, Any],
    *,
    center_world_m: Sequence[float],
    padding_fraction: float = 0.20,
    minimum_world_z_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], int] | None:
    rgb = np.asarray(evidence.get("rgb"))
    depth = np.asarray(evidence.get("depth_m"), dtype=np.float64)
    intrinsic = np.asarray(evidence.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(evidence.get("extrinsic_cv"), dtype=np.float64)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("appearance crop requires aligned RGB and depth")
    if intrinsic.shape != (3, 3):
        raise ValueError("appearance crop intrinsic must be 3x3")
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :4]
    if extrinsic.shape != (3, 4):
        raise ValueError("appearance crop extrinsic must be 3x4 or 4x4")
    ys, xs = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    valid = np.isfinite(depth) & (depth >= 0.2) & (depth <= 1.6)
    d = depth[valid]
    pixels = np.stack([xs[valid] * d, ys[valid] * d, d], axis=-1)
    camera_points = np.linalg.solve(intrinsic, pixels.T).T
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    points = (rotation.T @ (camera_points - translation).T).T
    region = candidate_region_mask(
        points,
        center_world_m=center_world_m,
        candidate=candidate,
    )
    if minimum_world_z_m is not None:
        region &= points[:, 2] >= float(minimum_world_z_m)
    mask = np.zeros(depth.shape, dtype=bool)
    valid_y = ys[valid]
    valid_x = xs[valid]
    mask[valid_y[region], valid_x[region]] = True
    y_values, x_values = np.nonzero(mask)
    if len(x_values) < 8:
        return None
    x0, x1 = int(x_values.min()), int(x_values.max()) + 1
    y0, y1 = int(y_values.min()), int(y_values.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * padding_fraction))
    x0, x1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad)
    crop = np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8).copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = np.asarray([127, 127, 127], dtype=np.uint8)
    return crop, crop_mask, [x0, y0, x1, y1], int(mask.sum())


def candidate_region_mask(
    points: np.ndarray,
    *,
    center_world_m: Sequence[float],
    candidate: GraspCandidate,
) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    center = np.asarray(center_world_m, dtype=np.float64)
    approach = _normalized(np.asarray(candidate.approach_axis_world, dtype=np.float64))
    delta = points - center
    axial = delta @ approach
    radial = np.linalg.norm(delta - axial[:, None] * approach, axis=1)
    contact_span = float(
        np.linalg.norm(candidate.left_contact_world_m - candidate.right_contact_world_m)
    )
    radial_limit = float(
        np.clip(
            max(0.032, candidate.opening_width_m * 0.9, contact_span * 0.9),
            0.032,
            0.07,
        )
    )
    return (np.abs(axial) <= 0.065) & (radial <= radial_limit)


def load_visual_projection(checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("schema_version") != "spatial.appearance_projection_checkpoint.v1":
        raise ValueError("unsupported appearance projection checkpoint")
    weight = checkpoint.get("state_dict", {}).get("visual.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("appearance projection checkpoint has no visual weight")
    return weight.detach().cpu().numpy().astype(np.float32)


def load_text_projection(checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("schema_version") != "spatial.appearance_projection_checkpoint.v1":
        raise ValueError("unsupported appearance projection checkpoint")
    weight = checkpoint.get("state_dict", {}).get("text.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("appearance projection checkpoint has no text weight")
    return weight.detach().cpu().numpy().astype(np.float32)


def project_visual_descriptor(
    descriptor: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    value = np.asarray(descriptor, dtype=np.float32)
    if value.shape != (projection.shape[1],):
        raise ValueError("appearance descriptor dimension does not match projection")
    return _normalized(projection @ value)


def project_text_descriptor(
    descriptor: np.ndarray, projection: np.ndarray
) -> np.ndarray:
    value = np.asarray(descriptor, dtype=np.float32)
    if value.shape != (projection.shape[1],):
        raise ValueError("text descriptor dimension does not match projection")
    return _normalized(projection @ value)


def descriptor_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.dot(_normalized(left), _normalized(right)))


def dense_patch_match(
    pre_signature: Mapping[str, Any],
    post_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare masked patch tokens with a per-event cross-view threshold."""

    pre_views = _load_patch_views(pre_signature)
    post_views = _load_patch_views(post_signature)
    if len(pre_views) < 2 or not post_views:
        return {
            "available": False,
            "reason": "insufficient_pre_or_post_patch_views",
            "pre_patch_view_count": len(pre_views),
            "post_patch_view_count": len(post_views),
            "controls_verdict": False,
        }
    results = {}
    for key in ("raw_descriptors", "projected_descriptors"):
        if not all(key in row for row in [*pre_views, *post_views]):
            continue
        pre_calibration = []
        for index, row in enumerate(pre_views):
            references = np.concatenate(
                [other[key] for other_index, other in enumerate(pre_views) if other_index != index],
                axis=0,
            )
            pre_calibration.append(
                {
                    "view": row["view"],
                    "score": _patch_match_score(row[key], references),
                }
            )
        threshold = float(
            np.clip(
                min(row["score"] for row in pre_calibration) - 0.05,
                0.30,
                0.95,
            )
        )
        reference = np.concatenate([row[key] for row in pre_views], axis=0)
        post_scores = [
            {
                "view": row["view"],
                "score": _patch_match_score(row[key], reference),
            }
            for row in post_views
        ]
        passing = sum(row["score"] >= threshold for row in post_scores)
        results[key.removesuffix("_descriptors")] = {
            "calibration": [
                {"view": row["view"], "score": round(row["score"], 7)}
                for row in pre_calibration
            ],
            "self_calibrated_threshold": round(threshold, 7),
            "post_views": [
                {"view": row["view"], "score": round(row["score"], 7)}
                for row in post_scores
            ],
            "passing_post_view_count": int(passing),
            "matched_in_two_views": passing >= 2,
        }
    text_value = pre_signature.get("projected_text_descriptor")
    if text_value is not None and all("projected_descriptors" in row for row in post_views):
        text = _normalized(text_value)
        pre_text_scores = [
            {
                "view": row["view"],
                "score": _patch_text_score(row["projected_descriptors"], text),
            }
            for row in pre_views
            if "projected_descriptors" in row
        ]
        if len(pre_text_scores) >= 2:
            threshold = float(
                np.clip(
                    min(row["score"] for row in pre_text_scores) - 0.05,
                    -0.20,
                    0.95,
                )
            )
            post_text_scores = [
                {
                    "view": row["view"],
                    "score": _patch_text_score(row["projected_descriptors"], text),
                }
                for row in post_views
            ]
            passing = sum(row["score"] >= threshold for row in post_text_scores)
            results["projected_text_conditioned"] = {
                "calibration": [
                    {"view": row["view"], "score": round(row["score"], 7)}
                    for row in pre_text_scores
                ],
                "self_calibrated_threshold": round(threshold, 7),
                "post_views": [
                    {"view": row["view"], "score": round(row["score"], 7)}
                    for row in post_text_scores
                ],
                "passing_post_view_count": int(passing),
                "matched_in_two_views": passing >= 2,
            }
    return {
        "schema_version": "spatial.dense_appearance_patch_match.v1",
        "available": bool(results),
        "pre_patch_view_count": len(pre_views),
        "post_patch_view_count": len(post_views),
        "representations": results,
        "controls_verdict": False,
        "reason": "shadow_until_success_and_empty_grasp_separation_is_calibrated",
        "access": "inference_visible_server_side",
    }


def _load_patch_views(signature: Mapping[str, Any]) -> list[dict[str, Any]]:
    root_value = signature.get("artifact_root")
    if not isinstance(root_value, str):
        return []
    root = Path(root_value).resolve()
    rows = []
    for row in signature.get("crops", ()):
        if not isinstance(row, Mapping):
            continue
        artifact = row.get("patch_artifact")
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
            continue
        path = (root / str(artifact["path"])).resolve()
        if root not in path.parents or not path.is_file():
            continue
        values = np.load(path, allow_pickle=False)
        mask = np.asarray(values["mask_fraction"], dtype=np.float32) >= 0.25
        if int(mask.sum()) < 2:
            continue
        value = {"view": str(row.get("view", "unknown"))}
        for key in ("raw_descriptors", "projected_descriptors"):
            if key in values:
                descriptors = np.asarray(values[key], dtype=np.float32)[mask]
                norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                value[key] = descriptors / np.maximum(norms, 1e-8)
        rows.append(value)
    return rows


def _patch_match_score(query: np.ndarray, reference: np.ndarray) -> float:
    if not len(query) or not len(reference):
        return -1.0
    maximum = np.max(query @ reference.T, axis=1)
    count = int(np.clip(round(len(maximum) * 0.20), 2, 16))
    return float(np.mean(np.partition(maximum, -count)[-count:]))


def _patch_text_score(patches: np.ndarray, text: np.ndarray) -> float:
    similarity = patches @ text
    count = int(np.clip(round(len(similarity) * 0.20), 2, 16))
    return float(np.mean(np.partition(similarity, -count)[-count:]))


def _normalized(value: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError("appearance descriptor has zero norm")
    return result / norm
