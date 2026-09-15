"""Automatic Phase 11 dataset for query-conditioned semantic part regions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .grasp_candidate_neural_ranker import IntentEmbeddingStore
from .grasp_candidate_ranker import _has_invalid_dataset_marker
from .learned_pregrasp import (
    DEPTH_MEAN_M,
    DEPTH_STD_M,
    RGB_MEAN,
    RGB_STD,
    gaussian_heatmap,
)


def discover_phase11_sample_paths(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        candidates = (
            sorted(path.rglob("automatic_candidate_sample.json"))
            if path.is_dir()
            else [path]
        )
        result.extend(
            candidate
            for candidate in candidates
            if not _has_invalid_dataset_marker(candidate)
        )
    return sorted(set(path.resolve() for path in result))


class SemanticPartRegionDataset(Dataset):
    def __init__(
        self,
        sample_paths: Iterable[str | Path],
        embedding_store: IntentEmbeddingStore,
        *,
        image_size: tuple[int, int] = (240, 320),
    ) -> None:
        self.sample_paths = discover_phase11_sample_paths(sample_paths)
        self.embedding_store = embedding_store
        self.image_size = tuple(int(value) for value in image_size)
        if not self.sample_paths:
            raise ValueError("semantic part dataset has no Phase 11 samples")
        if min(self.image_size) < 32 or any(value % 4 for value in self.image_size):
            raise ValueError("semantic part image dimensions must be >=32 and divisible by 4")
        self.records = []
        for sample_path in self.sample_paths:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            for observation_index in range(
                len(sample["inference_visible"]["observations"])
            ):
                self.records.append((sample_path, observation_index))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path, observation_index = self.records[index]
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        sample_id = str(sample["sample_id"])
        inference = sample["inference_visible"]
        training = sample["training_only"]
        observation = inference["observations"][observation_index]
        base = sample_path.parent
        rgb = np.asarray(
            Image.open(base / observation["rgb"]).convert("RGB"), dtype=np.uint8
        )
        depth_m = np.load(base / observation["depth"]).astype(np.float32)
        camera = json.loads((base / observation["camera"]).read_text(encoding="utf-8"))
        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/depth mismatch in {sample_path}")
        rgbd = prepare_semantic_part_rgbd(rgb, depth_m, self.image_size)
        region = _oracle_region(training["oracle_candidate_graph"])
        center = np.asarray(region["position_mean_world_m"], dtype=np.float64)
        axis = _unit(np.asarray(region["longitudinal_axis_world"], dtype=np.float64))
        half_length = float(region["half_length_m"])
        radial_extent = _region_radial_extent(training, region_id=str(region["id"]))
        points_world = np.stack(
            [center, center - axis * half_length, center + axis * half_length]
        )
        labels = _project_region_labels(
            points_world,
            depth_m,
            camera,
            resized_image_size=self.image_size,
            radial_half_extent_m=radial_extent,
        )
        return {
            "rgbd": rgbd,
            "intent_embedding": torch.from_numpy(
                embedding_store_vector(
                    self.embedding_store, sample_id, inference["intent"]
                )
            ),
            "heatmaps": labels["heatmaps"],
            "keypoint_xy": labels["keypoint_xy"],
            "keypoint_valid": labels["keypoint_valid"],
            "visibility": labels["visibility"],
            "depth_normalized": labels["depth_normalized"],
            "half_length_log_m": torch.tensor(math.log(half_length), dtype=torch.float32),
            "radial_half_extent_log_m": torch.tensor(
                math.log(radial_extent), dtype=torch.float32
            ),
            "region_present": torch.tensor(1.0, dtype=torch.float32),
            "keypoint_world_m": torch.from_numpy(points_world.astype(np.float32)),
            "camera_intrinsic": torch.from_numpy(
                np.asarray(camera["intrinsic_cv"], dtype=np.float32)
            ),
            "camera_extrinsic": torch.from_numpy(
                np.asarray(camera["extrinsic_cv"], dtype=np.float32)
            ),
            "original_image_size": torch.tensor(
                list(depth_m.shape), dtype=torch.float32
            ),
            "sample_id": sample_id,
            "record_id": f"{sample_id}/view_{observation['view']}",
            "view": str(observation["view"]),
            "task": str(sample["task"]),
            "semantic_role": str(region["semantic_type"]),
            "region_id": str(region["id"]),
            "entity_id": str(training["oracle_geometry"]["object_id"]),
        }


def embedding_store_vector(
    store: IntentEmbeddingStore,
    sample_id: str,
    intent: Mapping[str, Any],
) -> np.ndarray:
    try:
        value = store.for_sample(sample_id)
    except ValueError:
        value = store.for_intent(intent)
    return value.astype(np.float32)


def prepare_semantic_part_rgbd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    image_size: tuple[int, int],
) -> torch.Tensor:
    rgb_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    depth_tensor = torch.from_numpy(depth_m).unsqueeze(0)
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
    )[0]
    depth_tensor = F.interpolate(
        depth_tensor.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
    )[0]
    rgb_tensor = (rgb_tensor - RGB_MEAN[:, None, None]) / RGB_STD[:, None, None]
    depth_tensor = (
        depth_tensor.clamp(0.2, 1.6) - DEPTH_MEAN_M
    ) / DEPTH_STD_M
    return torch.cat([rgb_tensor, depth_tensor], dim=0).float()


def _project_region_labels(
    points_world: np.ndarray,
    depth_m: np.ndarray,
    camera: Mapping[str, Any],
    *,
    resized_image_size: tuple[int, int],
    radial_half_extent_m: float,
) -> dict[str, torch.Tensor]:
    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
    original_height, original_width = depth_m.shape
    resized_height, resized_width = resized_image_size
    heatmap_height, heatmap_width = resized_height // 4, resized_width // 4
    heatmaps = torch.zeros(3, heatmap_height, heatmap_width, dtype=torch.float32)
    keypoint_xy = torch.zeros(3, 2, dtype=torch.float32)
    keypoint_valid = torch.zeros(3, dtype=torch.float32)
    visibility = torch.zeros(3, dtype=torch.float32)
    depth_normalized = torch.zeros(3, dtype=torch.float32)
    tolerance = max(0.012, 1.5 * radial_half_extent_m)
    for index, point in enumerate(points_world):
        camera_point = extrinsic @ np.concatenate([point, [1.0]])
        if camera_point[2] <= 1e-6:
            continue
        pixel_h = intrinsic @ camera_point[:3]
        pixel = pixel_h[:2] / pixel_h[2]
        if not np.all(np.isfinite(pixel)):
            continue
        in_frame = bool(
            0.0 <= pixel[0] < original_width and 0.0 <= pixel[1] < original_height
        )
        if not in_frame:
            continue
        x = int(round(float(pixel[0])))
        y = int(round(float(pixel[1])))
        x0, x1 = max(0, x - 3), min(original_width, x + 4)
        y0, y1 = max(0, y - 3), min(original_height, y + 4)
        local_depth = depth_m[y0:y1, x0:x1]
        valid_depth = local_depth[np.isfinite(local_depth) & (local_depth > 0.0)]
        visible = bool(
            valid_depth.size
            and float(np.min(np.abs(valid_depth - camera_point[2]))) <= tolerance
        )
        resized_x = float(pixel[0]) * (resized_width - 1) / max(original_width - 1, 1)
        resized_y = float(pixel[1]) * (resized_height - 1) / max(original_height - 1, 1)
        heatmap_x = resized_x * (heatmap_width - 1) / max(resized_width - 1, 1)
        heatmap_y = resized_y * (heatmap_height - 1) / max(resized_height - 1, 1)
        keypoint_xy[index] = torch.tensor([heatmap_x, heatmap_y])
        keypoint_valid[index] = float(visible)
        visibility[index] = float(visible)
        depth_normalized[index] = float(
            (camera_point[2] - DEPTH_MEAN_M) / DEPTH_STD_M
        )
        if visible:
            heatmaps[index] = gaussian_heatmap(
                heatmap_height,
                heatmap_width,
                center_x=heatmap_x,
                center_y=heatmap_y,
                sigma=1.6,
            )
    return {
        "heatmaps": heatmaps,
        "keypoint_xy": keypoint_xy,
        "keypoint_valid": keypoint_valid,
        "visibility": visibility,
        "depth_normalized": depth_normalized,
    }


def _oracle_region(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    regions = [
        node for node in graph.get("nodes", ()) if node.get("node_type") == "semantic_region"
    ]
    if len(regions) != 1:
        raise ValueError(
            "current Phase 11 region dataset expects exactly one oracle semantic region"
        )
    return regions[0]


def _region_radial_extent(training: Mapping[str, Any], *, region_id: str) -> float:
    candidates = [
        row["candidate"]
        for row in training.get("candidate_labels", ())
        if row.get("candidate", {}).get("region_id") == region_id
    ]
    if not candidates:
        raise ValueError(f"no candidate frame exists for semantic region {region_id!r}")
    frame = candidates[0]["frame"]
    left = np.asarray(frame["left_contact_world_m"], dtype=np.float64)
    right = np.asarray(frame["right_contact_world_m"], dtype=np.float64)
    radius = 0.5 * float(np.linalg.norm(right - left))
    if radius <= 0.0:
        raise ValueError("semantic region radial extent must be positive")
    return radius


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("semantic region axis is degenerate")
    return value / norm
