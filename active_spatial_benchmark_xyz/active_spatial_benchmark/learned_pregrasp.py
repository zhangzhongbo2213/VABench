"""Learned RGB-D ObservationGraph baseline for ``verify_pregrasp``.

The inference path consumes only RGB, metric depth, camera calibration, and
robot kinematics. Simulator masks, segmentation, object poses, and Oracle
graphs are used only by :class:`PregraspLearningDataset` to construct targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18

from .active_belief import ObservationGraph
from .pregrasp_graph import PREGRASP_EDGE_IDS


OBJECT_KEYPOINT_IDS = ("object.center", "object.axis_start", "object.axis_end")
ROBOT_NODE_IDS = (
    "gripper.jaw_center",
    "gripper.finger_a_inner_tip",
    "gripper.finger_b_inner_tip",
    "gripper.grasp_center",
)
RELATION_ENDPOINTS = {
    "object_between_fingers": ("object.center", "gripper.grasp_center", "between_fingers"),
    "grasp_region_along_object_axis": ("gripper.grasp_center", "object.center", "within_grasp_region"),
    "grasp_height_aligned": ("gripper.grasp_center", "object.center", "height_aligned"),
    "closing_axis_perpendicular_to_object_axis": (
        "gripper.grasp_center",
        "object.center",
        "axis_perpendicular",
    ),
}
KINEMATICS_DIM = 24
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
DEPTH_MEAN_M = 0.8
DEPTH_STD_M = 0.3


@dataclass(frozen=True)
class LearningRecord:
    sample_id: str
    episode_id: str
    state_id: str
    frame_id: int
    view: str
    world_state_version: int
    rgb_path: Path
    depth_path: Path
    camera_path: Path
    robot_kinematics_path: Path
    labels_path: Path


def load_manifest_records(
    dataset_root: str | Path,
    *,
    episode_ids: Iterable[str] | None = None,
) -> list[LearningRecord]:
    root = Path(dataset_root).resolve()
    allowed = set(episode_ids or ())
    records = []
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("query") != "verify_pregrasp":
            continue
        if allowed and value["episode_id"] not in allowed:
            continue
        inference = value["inference_visible"]
        training = value["training_only"]
        records.append(
            LearningRecord(
                sample_id=str(value["sample_id"]),
                episode_id=str(value["episode_id"]),
                state_id=str(value["state_id"]),
                frame_id=int(value["frame_id"]),
                view=str(value["view"]),
                world_state_version=int(value["world_state_version"]),
                rgb_path=root / inference["rgb"],
                depth_path=root / inference["depth"],
                camera_path=root / inference["camera"],
                robot_kinematics_path=root / inference["robot_kinematics"],
                labels_path=root / training["labels"],
            )
        )
    return records


class PregraspLearningDataset(Dataset):
    """RGB-D training dataset with simulator truth confined to targets."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        episode_ids: Iterable[str] | None = None,
        heatmap_size: tuple[int, int] = (60, 80),
        heatmap_sigma: float = 1.6,
        augment: bool = False,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.records = load_manifest_records(self.root, episode_ids=episode_ids)
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = float(heatmap_sigma)
        self.augment = bool(augment)
        if not self.records:
            raise ValueError("no verify_pregrasp records matched the requested episodes")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        rgbd, camera, kinematics = load_inference_inputs(record)
        if self.augment:
            rgbd = augment_rgbd(rgbd)
        labels = json.loads(record.labels_path.read_text(encoding="utf-8"))
        keypoints = {str(item["id"]): item for item in labels["keypoints"]}
        relations = {str(item["id"]): item for item in labels["relations"]}
        image_height, image_width = rgbd.shape[-2:]
        heatmap_height, heatmap_width = self.heatmap_size
        heatmaps = torch.zeros((len(OBJECT_KEYPOINT_IDS), heatmap_height, heatmap_width), dtype=torch.float32)
        keypoint_xy = torch.zeros((len(OBJECT_KEYPOINT_IDS), 2), dtype=torch.float32)
        keypoint_valid = torch.zeros(len(OBJECT_KEYPOINT_IDS), dtype=torch.float32)
        visibility = torch.zeros(len(OBJECT_KEYPOINT_IDS), dtype=torch.float32)
        depth_normalized = torch.zeros(len(OBJECT_KEYPOINT_IDS), dtype=torch.float32)
        for kp_index, node_id in enumerate(OBJECT_KEYPOINT_IDS):
            item = keypoints[node_id]
            visibility[kp_index] = float(item.get("visibility", 0.0))
            pixel = item.get("pixel_uv")
            depth = item.get("camera_depth_m")
            if (
                pixel is None
                or depth is None
                or not bool(item.get("in_frame", False))
                or float(item.get("visibility", 0.0)) <= 0.0
            ):
                continue
            x = float(pixel[0]) * (heatmap_width - 1) / max(image_width - 1, 1)
            y = float(pixel[1]) * (heatmap_height - 1) / max(image_height - 1, 1)
            keypoint_xy[kp_index] = torch.tensor([x, y], dtype=torch.float32)
            keypoint_valid[kp_index] = 1.0
            depth_normalized[kp_index] = (float(depth) - DEPTH_MEAN_M) / DEPTH_STD_M
            heatmaps[kp_index] = gaussian_heatmap(
                heatmap_height,
                heatmap_width,
                center_x=x,
                center_y=y,
                sigma=self.heatmap_sigma,
            )
        relation_probability = torch.tensor(
            [float(relations[edge_id]["probability"]) for edge_id in PREGRASP_EDGE_IDS],
            dtype=torch.float32,
        )
        relation_evidence = torch.tensor(
            [float(relations[edge_id]["evidence_quality"]) for edge_id in PREGRASP_EDGE_IDS],
            dtype=torch.float32,
        )
        return {
            "rgbd": rgbd,
            "kinematics": kinematics,
            "heatmaps": heatmaps,
            "keypoint_xy": keypoint_xy,
            "keypoint_valid": keypoint_valid,
            "visibility": visibility,
            "depth_normalized": depth_normalized,
            "relation_probability": relation_probability,
            "relation_evidence": relation_evidence,
            "record_index": torch.tensor(index, dtype=torch.long),
            "camera_intrinsic": torch.from_numpy(camera["intrinsic_cv"]).float(),
            "camera_extrinsic": torch.from_numpy(camera["extrinsic_cv"]).float(),
        }


class PregraspObservationNet(nn.Module):
    """Small ResNet-FPN baseline producing keypoints and relation evidence."""

    def __init__(self, *, pretrained_path: str | Path | None = None, freeze_stem: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained_path is not None:
            state = torch.load(Path(pretrained_path), map_location="cpu", weights_only=True)
            rgb_weight = state["conv1.weight"]
            depth_weight = rgb_weight.mean(dim=1, keepdim=True)
            state["conv1.weight"] = torch.cat([rgb_weight, depth_weight], dim=1)
            backbone.load_state_dict(state, strict=True)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lat4 = nn.Conv2d(512, 128, 1)
        self.lat3 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(128, 128, 1)
        self.lat1 = nn.Conv2d(64, 128, 1)
        self.smooth3 = conv_block(128)
        self.smooth2 = conv_block(128)
        self.smooth1 = conv_block(128)
        self.heatmap_head = nn.Sequential(conv_block(128), nn.Conv2d(128, len(OBJECT_KEYPOINT_IDS), 1))
        self.global_head = nn.Sequential(
            nn.Linear(512 + KINEMATICS_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.keypoint_head = nn.Sequential(
            nn.Linear(128 + 128 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.depth_residual_head = nn.Linear(128, 1)
        self.depth_log_variance_head = nn.Linear(128, 1)
        self.visibility_head = nn.Linear(128, 1)
        self.relation_head = nn.Linear(128, len(PREGRASP_EDGE_IDS))
        self.evidence_head = nn.Linear(128, len(PREGRASP_EDGE_IDS))
        if freeze_stem:
            for module in (self.stem, self.layer1):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, rgbd: torch.Tensor, kinematics: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.layer1(self.stem(rgbd))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        p3 = self.smooth3(F.interpolate(self.lat4(x4), size=x3.shape[-2:], mode="bilinear", align_corners=False) + self.lat3(x3))
        p2 = self.smooth2(F.interpolate(p3, size=x2.shape[-2:], mode="bilinear", align_corners=False) + self.lat2(x2))
        p1 = self.smooth1(F.interpolate(p2, size=x1.shape[-2:], mode="bilinear", align_corners=False) + self.lat1(x1))
        heatmap_logits = self.heatmap_head(p1)
        pooled = F.adaptive_avg_pool2d(x4, 1).flatten(1)
        hidden = self.global_head(torch.cat([pooled, kinematics], dim=1))
        heatmap_probability = torch.softmax(
            heatmap_logits.reshape(heatmap_logits.shape[0], len(OBJECT_KEYPOINT_IDS), -1) / 0.25,
            dim=-1,
        )
        local_features = torch.einsum(
            "bkn,bcn->bkc",
            heatmap_probability,
            p1.reshape(p1.shape[0], p1.shape[1], -1),
        )
        depth_at_heatmap = F.interpolate(
            rgbd[:, 3:4],
            size=p1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(rgbd.shape[0], 1, -1)
        sampled_depth = torch.einsum("bkn,bcn->bkc", heatmap_probability, depth_at_heatmap)
        keypoint_hidden = self.keypoint_head(
            torch.cat(
                [
                    local_features,
                    hidden[:, None, :].expand(-1, len(OBJECT_KEYPOINT_IDS), -1),
                    sampled_depth,
                ],
                dim=-1,
            )
        )
        depth_normalized = sampled_depth.squeeze(-1) + self.depth_residual_head(keypoint_hidden).squeeze(-1)
        return {
            "heatmap_logits": heatmap_logits,
            "depth_normalized": depth_normalized,
            "depth_log_variance": self.depth_log_variance_head(keypoint_hidden).squeeze(-1).clamp(-6.0, 3.0),
            "visibility_logits": self.visibility_head(keypoint_hidden).squeeze(-1),
            "relation_logits": self.relation_head(hidden),
            "evidence_logits": self.evidence_head(hidden),
        }


def pregrasp_learning_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_probability = torch.sigmoid(output["heatmap_logits"])
    heatmap_loss = F.mse_loss(heatmap_probability, batch["heatmaps"])
    predicted_xy, _ = softargmax_2d(output["heatmap_logits"])
    valid = batch["keypoint_valid"].unsqueeze(-1)
    coordinate_error = F.smooth_l1_loss(predicted_xy, batch["keypoint_xy"], reduction="none")
    coordinate_loss = (coordinate_error * valid).sum() / valid.sum().clamp_min(1.0)
    depth_error = output["depth_normalized"] - batch["depth_normalized"]
    log_variance = output["depth_log_variance"]
    depth_nll = 0.5 * (torch.exp(-log_variance) * depth_error.square() + log_variance)
    depth_loss = (depth_nll * batch["keypoint_valid"]).sum() / batch["keypoint_valid"].sum().clamp_min(1.0)
    visibility_element = F.binary_cross_entropy_with_logits(
        output["visibility_logits"],
        batch["visibility"],
        reduction="none",
    )
    visibility_weight = torch.where(batch["visibility"] < 0.5, 3.0, 1.0)
    visibility_loss = (visibility_element * visibility_weight).sum() / visibility_weight.sum().clamp_min(1.0)
    axis_valid = (batch["keypoint_valid"][:, 1] * batch["keypoint_valid"][:, 2]).float()
    predicted_axis = predicted_xy[:, 2] - predicted_xy[:, 1]
    target_axis = batch["keypoint_xy"][:, 2] - batch["keypoint_xy"][:, 1]
    axis_cosine = F.cosine_similarity(predicted_axis, target_axis, dim=-1, eps=1e-6).abs()
    axis_direction_loss = ((1.0 - axis_cosine) * axis_valid).sum() / axis_valid.sum().clamp_min(1.0)
    predicted_length = torch.linalg.norm(predicted_axis, dim=-1)
    target_length = torch.linalg.norm(target_axis, dim=-1)
    axis_length_loss = (
        F.smooth_l1_loss(predicted_length, target_length, reduction="none") * axis_valid
    ).sum() / axis_valid.sum().clamp_min(1.0)
    relation_loss = F.binary_cross_entropy_with_logits(output["relation_logits"], batch["relation_probability"])
    evidence_loss = F.binary_cross_entropy_with_logits(output["evidence_logits"], batch["relation_evidence"])
    total = (
        5.0 * heatmap_loss
        + 0.12 * coordinate_loss
        + 0.6 * depth_loss
        + 0.5 * visibility_loss
        + 0.3 * axis_direction_loss
        + 0.03 * axis_length_loss
        + 1.5 * relation_loss
        + 0.8 * evidence_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "depth": float(depth_loss.detach()),
        "visibility": float(visibility_loss.detach()),
        "axis_direction": float(axis_direction_loss.detach()),
        "axis_length": float(axis_length_loss.detach()),
        "relation": float(relation_loss.detach()),
        "evidence": float(evidence_loss.detach()),
    }
    return total, metrics


@torch.no_grad()
def predict_observation_graph(
    model: PregraspObservationNet,
    record: LearningRecord,
    *,
    device: torch.device | str,
) -> tuple[ObservationGraph, dict[str, Any]]:
    rgbd, camera, kinematics = load_inference_inputs(record)
    robot_value = json.loads(record.robot_kinematics_path.read_text(encoding="utf-8"))
    return predict_observation_graph_from_inputs(
        model,
        rgbd=rgbd,
        camera=camera,
        kinematics=kinematics,
        robot_kinematics=robot_value,
        frame_id=record.frame_id,
        view=record.view,
        world_state_version=record.world_state_version,
        device=device,
    )


@torch.no_grad()
def predict_observation_graph_from_inputs(
    model: PregraspObservationNet,
    *,
    rgbd: torch.Tensor,
    camera: Mapping[str, np.ndarray],
    kinematics: torch.Tensor,
    robot_kinematics: Mapping[str, Any],
    frame_id: int,
    view: str,
    world_state_version: int,
    device: torch.device | str,
) -> tuple[ObservationGraph, dict[str, Any]]:
    """Create one learned graph without reading simulator-only labels."""

    model.eval()
    output = model(rgbd.unsqueeze(0).to(device), kinematics.unsqueeze(0).to(device))
    decoded = decode_model_output(output, camera, image_size=(rgbd.shape[-2], rgbd.shape[-1]))
    robot_value = dict(robot_kinematics)
    robot_nodes = [dict(node) for node in robot_value["nodes"]]
    robot_by_id = {str(node["id"]): node for node in robot_nodes}
    closing_axis = np.asarray(robot_value["query_axes"]["closing_axis_world"], dtype=np.float64)
    object_axis = np.asarray(decoded["object_axis_world"], dtype=np.float64)
    query_axes = {
        "closing_axis_world": vector(closing_axis),
        "support_normal_world": list(robot_value["query_axes"]["support_normal_world"]),
    }
    if all(
        decoded["keypoints"][node_id]["observation_state"] == "visible"
        for node_id in ("object.axis_start", "object.axis_end")
    ):
        query_axes["object_axis_world"] = vector(object_axis)
    nodes = []
    for node_id in OBJECT_KEYPOINT_IDS:
        value = decoded["keypoints"][node_id]
        if value["observation_state"] != "visible":
            continue
        nodes.append(
            {
                "id": node_id,
                "semantic_type": "object_center" if node_id == "object.center" else "object_axis_endpoint",
                "position_mean_world_m": value["position_world_m"],
                "position_covariance_m2": value["position_covariance_m2"],
                "visibility": value["visibility"],
                "observation_state": "visible",
                "source": "learned_rgbd",
                "access": "inference_visible",
                "valid_for_world_state": int(world_state_version),
            }
        )
    geometric_relations = {}
    if len(nodes) == len(OBJECT_KEYPOINT_IDS):
        geometric_relations = geometric_pregrasp_relations(
            object_keypoints=decoded["keypoints"],
            robot_nodes=robot_by_id,
            view_direction_world=np.asarray(camera["camera_pose_world"], dtype=np.float64)[:3, 0],
        )
    edges = []
    for edge_id in PREGRASP_EDGE_IDS:
        if edge_id not in geometric_relations:
            continue
        source, target, relation = RELATION_ENDPOINTS[edge_id]
        geometric = geometric_relations[edge_id]
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "relation": relation,
                "probability": geometric["probability"],
                "evidence_weight": geometric["evidence_weight"],
                "measurement": {
                    **geometric["measurement"],
                    "relation_source": "geometry_from_learned_keypoints",
                },
                "valid_for_world_state": int(world_state_version),
            }
        )
    observation = ObservationGraph(
        frame_id=int(frame_id),
        view=str(view),
        nodes=nodes,
        edges=edges,
        query_axes=query_axes,
        source="learned_rgbd/inference_visible",
    )
    decoded["robot_nodes"] = robot_nodes
    decoded["query_axes"] = query_axes
    decoded["geometric_relations"] = geometric_relations
    return observation, decoded


def geometric_pregrasp_relations(
    *,
    object_keypoints: Mapping[str, Mapping[str, Any]],
    robot_nodes: Mapping[str, Mapping[str, Any]],
    view_direction_world: np.ndarray,
    pen_cross_radius_m: float = 0.012,
) -> dict[str, dict[str, Any]]:
    """Compute operational relations from learned object points and FK points."""

    object_center = np.asarray(object_keypoints["object.center"]["position_world_m"], dtype=np.float64)
    axis_start = np.asarray(object_keypoints["object.axis_start"]["position_world_m"], dtype=np.float64)
    axis_end = np.asarray(object_keypoints["object.axis_end"]["position_world_m"], dtype=np.float64)
    object_axis = unit(axis_end - axis_start)
    object_axis_radius = max(0.025, 0.5 * float(np.linalg.norm(axis_end - axis_start)))
    tip_a = np.asarray(robot_nodes["gripper.finger_a_inner_tip"]["position_mean_world_m"], dtype=np.float64)
    tip_b = np.asarray(robot_nodes["gripper.finger_b_inner_tip"]["position_mean_world_m"], dtype=np.float64)
    grasp_center = np.asarray(robot_nodes["gripper.grasp_center"]["position_mean_world_m"], dtype=np.float64)
    closing_axis = unit(tip_b - tip_a)
    finger_scalars = sorted(
        [float(np.dot(tip_a - grasp_center, closing_axis)), float(np.dot(tip_b - grasp_center, closing_axis))]
    )
    object_scalar = float(np.dot(object_center - grasp_center, closing_axis))
    object_interval = [object_scalar - pen_cross_radius_m, object_scalar + pen_cross_radius_m]
    enclosure_margin = min(object_interval[0] - finger_scalars[0], finger_scalars[1] - object_interval[1])
    between_probability = logistic((enclosure_margin + 0.002) / 0.005)

    along_offset = float(np.dot(grasp_center - object_center, object_axis))
    along_limit = max(0.025, min(0.065, 0.65 * object_axis_radius))
    along_probability = logistic((along_limit - abs(along_offset)) / 0.008)

    vertical_offset = float(grasp_center[2] - object_center[2])
    height_probability = logistic((0.018 - abs(vertical_offset)) / 0.005)

    axis_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(closing_axis, object_axis)), 0.0, 1.0))))
    perpendicular_error = abs(90.0 - axis_angle)
    perpendicular_probability = logistic((15.0 - perpendicular_error) / 4.0)

    direction = unit(np.asarray(view_direction_world, dtype=np.float64))
    visibility = {node_id: float(value.get("visibility", 0.0)) for node_id, value in object_keypoints.items()}
    axis_visibility = max(
        min(visibility["object.axis_start"], visibility["object.axis_end"]),
        visibility["object.center"],
    )
    observability = {
        "closing": 1.0 - abs(float(np.dot(direction, closing_axis))),
        "object": 1.0 - abs(float(np.dot(direction, object_axis))),
        "vertical": 1.0 - abs(float(direction[2])),
    }
    evidence = {
        "object_between_fingers": observability["closing"] * visibility["object.center"],
        "grasp_region_along_object_axis": observability["object"] * axis_visibility,
        "grasp_height_aligned": observability["vertical"] * visibility["object.center"],
        "closing_axis_perpendicular_to_object_axis": min(
            observability["closing"], observability["object"]
        )
        * axis_visibility,
    }
    probabilities = {
        "object_between_fingers": between_probability,
        "grasp_region_along_object_axis": along_probability,
        "grasp_height_aligned": height_probability,
        "closing_axis_perpendicular_to_object_axis": perpendicular_probability,
    }
    measurements = {
        "object_between_fingers": {
            "predicted_enclosure_margin_m": round(enclosure_margin, 6),
            "predicted_closing_axis_offset_m": round(object_scalar, 6),
        },
        "grasp_region_along_object_axis": {
            "predicted_axis_offset_m": round(along_offset, 6),
            "predicted_axis_limit_m": round(along_limit, 6),
        },
        "grasp_height_aligned": {
            "predicted_vertical_offset_m": round(vertical_offset, 6),
        },
        "closing_axis_perpendicular_to_object_axis": {
            "predicted_axis_angle_deg": round(axis_angle, 6),
        },
    }
    return {
        edge_id: {
            "probability": round(float(probabilities[edge_id]), 6),
            "evidence_weight": round(float(np.clip(evidence[edge_id], 0.02, 1.0)), 6),
            "measurement": measurements[edge_id],
        }
        for edge_id in PREGRASP_EDGE_IDS
    }


def derived_edges_from_belief_graph(
    graph: Mapping[str, Any],
    *,
    view_direction_world: np.ndarray,
) -> tuple[list[dict[str, Any]], list[float] | None]:
    """Recompute relations from the current fused-node graph."""

    nodes = {str(node["id"]): node for node in graph["nodes"]}
    if any(node_id not in nodes for node_id in OBJECT_KEYPOINT_IDS):
        return [], None
    object_keypoints = {
        node_id: {
            "position_world_m": nodes[node_id]["position_mean_world_m"],
            "visibility": nodes[node_id].get("visibility", 0.0),
        }
        for node_id in OBJECT_KEYPOINT_IDS
    }
    relations = geometric_pregrasp_relations(
        object_keypoints=object_keypoints,
        robot_nodes=nodes,
        view_direction_world=view_direction_world,
    )
    axis_start = np.asarray(nodes["object.axis_start"]["position_mean_world_m"], dtype=np.float64)
    axis_end = np.asarray(nodes["object.axis_end"]["position_mean_world_m"], dtype=np.float64)
    object_axis = vector(unit(axis_end - axis_start))
    edges = []
    for edge_id in PREGRASP_EDGE_IDS:
        source, target, relation = RELATION_ENDPOINTS[edge_id]
        value = relations[edge_id]
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "relation": relation,
                "probability": value["probability"],
                "evidence_weight": value["evidence_weight"],
                "measurement": {
                    **value["measurement"],
                    "relation_source": "geometry_from_fused_learned_keypoints",
                },
                "valid_for_world_state": graph["world_state_version"],
            }
        )
    return edges, object_axis


def load_inference_inputs(
    record: LearningRecord,
) -> tuple[torch.Tensor, dict[str, np.ndarray], torch.Tensor]:
    rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"), dtype=np.uint8)
    depth_m = np.load(record.depth_path).astype(np.float32) / 1000.0
    camera_json = json.loads(record.camera_path.read_text(encoding="utf-8"))
    robot_json = json.loads(record.robot_kinematics_path.read_text(encoding="utf-8"))
    return prepare_inference_inputs(rgb, depth_m, camera_json, robot_json)


def prepare_inference_inputs(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    camera_value: Mapping[str, Any],
    robot_kinematics: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, np.ndarray], torch.Tensor]:
    """Normalize live or recorded inference-visible inputs for the network."""

    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB input, got {rgb.shape}")
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth_m.shape}")
    if rgb.dtype == np.uint8:
        rgb_float = rgb.astype(np.float32) / 255.0
    else:
        rgb_float = rgb.astype(np.float32)
        if float(np.nanmax(rgb_float)) > 1.5:
            rgb_float /= 255.0
    camera = {
        "intrinsic_cv": np.asarray(camera_value["intrinsic_cv"], dtype=np.float64),
        "extrinsic_cv": np.asarray(camera_value["extrinsic_cv"], dtype=np.float64),
        "camera_pose_world": np.asarray(camera_value["camera_pose_world"], dtype=np.float64),
    }
    rgb_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)
    rgb_tensor = (rgb_tensor - RGB_MEAN[:, None, None]) / RGB_STD[:, None, None]
    depth_tensor = torch.from_numpy((np.clip(depth_m, 0.2, 1.6) - DEPTH_MEAN_M) / DEPTH_STD_M).unsqueeze(0)
    rgbd = torch.cat([rgb_tensor, depth_tensor], dim=0).float()
    kinematics = torch.from_numpy(kinematics_vector(robot_kinematics, camera)).float()
    return rgbd, camera, kinematics


def augment_rgbd(rgbd: torch.Tensor) -> torch.Tensor:
    """Photometric and metric-depth noise without changing image geometry."""

    result = rgbd.clone()
    rgb = result[:3] * RGB_STD[:, None, None] + RGB_MEAN[:, None, None]
    brightness = torch.empty(1).uniform_(0.82, 1.18).item()
    contrast = torch.empty(1).uniform_(0.82, 1.18).item()
    saturation = torch.empty(1).uniform_(0.85, 1.15).item()
    rgb = rgb * brightness
    spatial_mean = rgb.mean(dim=(-2, -1), keepdim=True)
    rgb = (rgb - spatial_mean) * contrast + spatial_mean
    gray = rgb.mean(dim=0, keepdim=True)
    rgb = (rgb - gray) * saturation + gray
    rgb = (rgb + torch.randn_like(rgb) * 0.012).clamp(0.0, 1.0)
    result[:3] = (rgb - RGB_MEAN[:, None, None]) / RGB_STD[:, None, None]

    depth_m = result[3] * DEPTH_STD_M + DEPTH_MEAN_M
    depth_scale = torch.empty(1).uniform_(0.99, 1.01).item()
    depth_m = depth_m * depth_scale + torch.randn_like(depth_m) * 0.003
    result[3] = (depth_m.clamp(0.2, 1.6) - DEPTH_MEAN_M) / DEPTH_STD_M
    return result


def kinematics_vector(robot_json: Mapping[str, Any], camera: Mapping[str, np.ndarray]) -> np.ndarray:
    nodes = {str(node["id"]): node for node in robot_json["nodes"]}
    workspace_center = np.array([0.0, -0.08, 0.86], dtype=np.float64)
    values = []
    for node_id in ROBOT_NODE_IDS:
        position = np.asarray(nodes[node_id]["position_mean_world_m"], dtype=np.float64)
        values.extend(((position - workspace_center) / 0.5).tolist())
    closing = unit(np.asarray(robot_json["query_axes"]["closing_axis_world"], dtype=np.float64))
    support = unit(np.asarray(robot_json["query_axes"]["support_normal_world"], dtype=np.float64))
    camera_position = np.asarray(camera["camera_pose_world"], dtype=np.float64)[:3, 3]
    grasp_center = np.asarray(nodes["gripper.grasp_center"]["position_mean_world_m"], dtype=np.float64)
    view_direction = unit(np.asarray(camera["camera_pose_world"], dtype=np.float64)[:3, 0])
    values.extend(closing.tolist())
    values.extend(support.tolist())
    values.extend((camera_position - grasp_center).tolist())
    values.extend(view_direction.tolist())
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (KINEMATICS_DIM,):
        raise ValueError(f"expected {KINEMATICS_DIM} kinematic values, got {result.shape}")
    return result


def decode_model_output(
    output: Mapping[str, torch.Tensor],
    camera: Mapping[str, np.ndarray],
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    heatmap_logits = output["heatmap_logits"][0]
    xy_heatmap, covariance_heatmap = softargmax_2d(heatmap_logits.unsqueeze(0))
    xy_heatmap = xy_heatmap[0].cpu().numpy()
    covariance_heatmap = covariance_heatmap[0].cpu().numpy()
    heatmap_height, heatmap_width = heatmap_logits.shape[-2:]
    image_height, image_width = image_size
    scale_x = (image_width - 1) / max(heatmap_width - 1, 1)
    scale_y = (image_height - 1) / max(heatmap_height - 1, 1)
    depth_normalized = output["depth_normalized"][0].cpu().numpy()
    depth_m = DEPTH_MEAN_M + DEPTH_STD_M * depth_normalized
    depth_variance_m2 = np.exp(output["depth_log_variance"][0].cpu().numpy()) * DEPTH_STD_M**2
    visibility = torch.sigmoid(output["visibility_logits"])[0].cpu().numpy()
    heatmap_probability = torch.softmax(
        heatmap_logits.reshape(len(OBJECT_KEYPOINT_IDS), -1) / 0.25,
        dim=-1,
    )
    normalized_entropy = (
        -(heatmap_probability * (heatmap_probability + 1e-12).log()).sum(dim=-1)
        / np.log(heatmap_probability.shape[-1])
    ).cpu().numpy()
    heatmap_quality = np.clip(1.0 - normalized_entropy, 0.0, 1.0)
    effective_visibility = visibility * np.sqrt(heatmap_quality)
    relation_probability = torch.sigmoid(output["relation_logits"])[0].cpu().numpy()
    relation_evidence = torch.sigmoid(output["evidence_logits"])[0].cpu().numpy()
    keypoints = {}
    positions = {}
    for index, node_id in enumerate(OBJECT_KEYPOINT_IDS):
        pixel = np.array([xy_heatmap[index, 0] * scale_x, xy_heatmap[index, 1] * scale_y], dtype=np.float64)
        covariance_pixel = np.array(
            [
                [covariance_heatmap[index, 0, 0] * scale_x**2, covariance_heatmap[index, 0, 1] * scale_x * scale_y],
                [covariance_heatmap[index, 1, 0] * scale_x * scale_y, covariance_heatmap[index, 1, 1] * scale_y**2],
            ],
            dtype=np.float64,
        )
        position_world, covariance_world = backproject_with_covariance(
            pixel,
            float(np.clip(depth_m[index], 0.2, 1.6)),
            covariance_pixel,
            float(max(depth_variance_m2[index], 1e-7)),
            camera["intrinsic_cv"],
            camera["extrinsic_cv"],
        )
        visibility_scale = 1.0 / max(float(effective_visibility[index]), 0.10)
        covariance_world *= visibility_scale
        positions[node_id] = position_world
        keypoints[node_id] = {
            "pixel_uv": vector(pixel),
            "camera_depth_m": round(float(depth_m[index]), 6),
            "position_world_m": vector(position_world),
            "position_covariance_m2": covariance_world.round(9).tolist(),
            "visibility": round(float(effective_visibility[index]), 6),
            "visibility_head_probability": round(float(visibility[index]), 6),
            "heatmap_quality": round(float(heatmap_quality[index]), 6),
            "observation_state": "visible" if effective_visibility[index] >= 0.45 else "unobserved",
        }
    axis = unit(positions["object.axis_end"] - positions["object.axis_start"])
    return {
        "keypoints": keypoints,
        "object_axis_world": vector(axis),
        "relation_probability": [round(float(value), 6) for value in relation_probability],
        "relation_evidence": [round(float(value), 6) for value in relation_evidence],
        "mean_visibility": round(float(np.mean(effective_visibility)), 6),
    }


def softargmax_2d(
    logits: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, keypoints, height, width = logits.shape
    probability = torch.softmax(
        logits.reshape(batch, keypoints, -1) / max(float(temperature), 1e-3),
        dim=-1,
    ).reshape(batch, keypoints, height, width)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=logits.dtype),
        torch.arange(width, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    mean_x = (probability * x_grid).sum(dim=(-2, -1))
    mean_y = (probability * y_grid).sum(dim=(-2, -1))
    dx = x_grid[None, None] - mean_x[:, :, None, None]
    dy = y_grid[None, None] - mean_y[:, :, None, None]
    var_x = (probability * dx.square()).sum(dim=(-2, -1))
    var_y = (probability * dy.square()).sum(dim=(-2, -1))
    cov_xy = (probability * dx * dy).sum(dim=(-2, -1))
    covariance = torch.stack(
        [torch.stack([var_x, cov_xy], dim=-1), torch.stack([cov_xy, var_y], dim=-1)],
        dim=-2,
    )
    return torch.stack([mean_x, mean_y], dim=-1), covariance


def backproject_with_covariance(
    pixel_uv: np.ndarray,
    depth_m: float,
    covariance_pixel: np.ndarray,
    depth_variance_m2: float,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u, v = float(pixel_uv[0]), float(pixel_uv[1])
    point_camera = np.array(
        [(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m],
        dtype=np.float64,
    )
    transform_world_camera = np.eye(4, dtype=np.float64)
    transform_world_camera[:3, :4] = extrinsic
    transform_camera_world = np.linalg.inv(transform_world_camera)
    point_world = (transform_camera_world @ np.concatenate([point_camera, [1.0]]))[:3]
    covariance_uvd = np.zeros((3, 3), dtype=np.float64)
    covariance_uvd[:2, :2] = covariance_pixel
    covariance_uvd[2, 2] = depth_variance_m2
    jacobian = np.array(
        [
            [depth_m / fx, 0.0, (u - cx) / fx],
            [0.0, depth_m / fy, (v - cy) / fy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    covariance_camera = jacobian @ covariance_uvd @ jacobian.T
    rotation_camera_world = transform_camera_world[:3, :3]
    covariance_world = rotation_camera_world @ covariance_camera @ rotation_camera_world.T
    covariance_world += np.eye(3, dtype=np.float64) * 1e-8
    return point_world, covariance_world


def gaussian_heatmap(
    height: int,
    width: int,
    *,
    center_x: float,
    center_y: float,
    sigma: float,
) -> torch.Tensor:
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.exp(-((x_grid - center_x).square() + (y_grid - center_y).square()) / (2.0 * sigma**2))


def conv_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return value / norm


def logistic(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-value))


def vector(value: Sequence[float] | np.ndarray) -> list[float]:
    return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]
