"""RGB-D supervision for task-conditioned expert grasp frames."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
import torch

from .expert_event_candidate_dataset import event_intent_from_sample
from .expert_grasp_trace import validate_expert_grasp_event_sample
from .grasp_candidate_neural_ranker import IntentEmbeddingStore
from .learned_pregrasp import DEPTH_MEAN_M, DEPTH_STD_M, gaussian_heatmap
from .semantic_part_region_dataset import (
    embedding_store_vector,
    prepare_semantic_part_rgbd,
)


GRASP_FRAME_KEYPOINT_NAMES = ("grasp_center", "left_contact", "right_contact")


def discover_expert_event_sample_paths(
    paths: Iterable[str | Path],
) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            candidates.extend(sorted(path.rglob("samples/*.json")))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "phase11.expert_grasp_event_catalog.v1":
            if payload.get("status") != "complete":
                raise ValueError("expert event catalog must be complete")
            candidates.extend(Path(row["sample"]).resolve() for row in payload["samples"])
        else:
            candidates.append(path)
    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_expert_grasp_event_sample(payload)
        if errors:
            continue
        sample_id = str(payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"expert event sample {path} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate expert event sample_id {sample_id!r}")
        seen.add(sample_id)
        result.append(path)
    if not result:
        raise ValueError("expert grasp frame dataset has no valid event samples")
    return sorted(result)


class ExpertGraspFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_paths: Iterable[str | Path],
        embedding_store: IntentEmbeddingStore,
        *,
        image_size: tuple[int, int] = (240, 320),
    ) -> None:
        self.sample_paths = discover_expert_event_sample_paths(sample_paths)
        self.embedding_store = embedding_store
        self.image_size = tuple(int(value) for value in image_size)
        if min(self.image_size) < 32 or any(value % 4 for value in self.image_size):
            raise ValueError("grasp frame image dimensions must be >=32 and divisible by 4")
        self.records: list[tuple[Path, int]] = []
        for sample_path in self.sample_paths:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            for observation_index in range(len(sample["inference_visible"]["observations"])):
                self.records.append((sample_path, observation_index))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path, observation_index = self.records[index]
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        inference = sample["inference_visible"]
        training = sample["training_only"]
        observation = inference["observations"][observation_index]
        view = str(observation["view"])
        supervision = _view_supervision(training, view=view)
        base = sample_path.parent.parent
        rgb = np.asarray(Image.open(base / observation["rgb"]).convert("RGB"))
        depth_m = np.load(base / observation["depth"]).astype(np.float32)
        camera = json.loads((base / observation["camera"]).read_text(encoding="utf-8"))
        if rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"RGB/depth mismatch in {sample_path}")
        labels = labels_from_view_supervision(
            supervision,
            original_image_size=depth_m.shape,
            resized_image_size=self.image_size,
        )
        frame = training["expert_grasp_frame"]
        intent = event_intent_from_sample(sample).as_dict()
        points_world = np.stack(
            [
                np.asarray(frame["center_world_m"], dtype=np.float32),
                np.asarray(frame["left_contact_world_m"], dtype=np.float32),
                np.asarray(frame["right_contact_world_m"], dtype=np.float32),
            ]
        )
        sample_id = str(sample["sample_id"])
        return {
            "rgbd": prepare_semantic_part_rgbd(rgb, depth_m, self.image_size),
            "intent_embedding": torch.from_numpy(
                embedding_store_vector(self.embedding_store, sample_id, intent)
            ),
            **labels,
            "keypoint_world_m": torch.from_numpy(points_world),
            "approach_axis_world": torch.tensor(
                frame["approach_axis_world"], dtype=torch.float32
            ),
            "closing_axis_world": torch.tensor(
                frame["closing_axis_world"], dtype=torch.float32
            ),
            "opening_width_log_m": torch.tensor(
                math.log(float(frame["opening_width_m"])), dtype=torch.float32
            ),
            "frame_present": torch.tensor(1.0, dtype=torch.float32),
            "camera_intrinsic": torch.tensor(camera["intrinsic_cv"], dtype=torch.float32),
            "camera_extrinsic": torch.tensor(camera["extrinsic_cv"], dtype=torch.float32),
            "original_image_size": torch.tensor(depth_m.shape, dtype=torch.float32),
            "sample_id": sample_id,
            "record_id": f"{sample_id}/view_{view}",
            "task": str(sample["task"]),
            "seed": int(sample["seed"]),
            "view": view,
            "target": str(intent["target"]),
            "semantic_role": str(intent["preferred_roles"][0]),
        }


def labels_from_view_supervision(
    supervision: Mapping[str, Any],
    *,
    original_image_size: tuple[int, int],
    resized_image_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    original_height, original_width = original_image_size
    resized_height, resized_width = resized_image_size
    heatmap_height, heatmap_width = resized_height // 4, resized_width // 4
    heatmaps = torch.zeros(3, heatmap_height, heatmap_width, dtype=torch.float32)
    keypoint_xy = torch.zeros(3, 2, dtype=torch.float32)
    keypoint_valid = torch.zeros(3, dtype=torch.float32)
    visibility = torch.zeros(3, dtype=torch.float32)
    depth_normalized = torch.zeros(3, dtype=torch.float32)
    projections = supervision["keypoint_projections"]
    for index, name in enumerate(GRASP_FRAME_KEYPOINT_NAMES):
        row = projections[name]
        pixel = row.get("pixel_uv")
        if not row.get("in_frame") or pixel is None:
            continue
        resized_x = float(pixel[0]) * (resized_width - 1) / max(original_width - 1, 1)
        resized_y = float(pixel[1]) * (resized_height - 1) / max(original_height - 1, 1)
        heatmap_x = resized_x * (heatmap_width - 1) / max(resized_width - 1, 1)
        heatmap_y = resized_y * (heatmap_height - 1) / max(resized_height - 1, 1)
        keypoint_xy[index] = torch.tensor([heatmap_x, heatmap_y])
        keypoint_valid[index] = 1.0
        visibility[index] = float(row.get("observation_state") == "target_visible")
        depth_normalized[index] = (
            float(row["camera_depth_m"]) - DEPTH_MEAN_M
        ) / DEPTH_STD_M
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


def _view_supervision(training: Mapping[str, Any], *, view: str) -> Mapping[str, Any]:
    matches = [row for row in training["view_supervision"] if row.get("view") == view]
    if len(matches) != 1:
        raise ValueError(f"expected one view supervision row for {view!r}")
    return matches[0]
