"""Query-conditioned RGB-D model for local semantic grasp-part regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .learned_pregrasp import (
    DEPTH_MEAN_M,
    DEPTH_STD_M,
    backproject_with_covariance,
    conv_block,
    softargmax_2d,
)


REGION_KEYPOINT_NAMES = ("center", "axis_start", "axis_end")


class SemanticPartRegionNet(nn.Module):
    """ResNet-FPN region detector modulated by a frozen text embedding."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        feature_dim: int = 128,
        freeze_stem: bool = True,
    ) -> None:
        super().__init__()
        if text_embedding_dim < 1:
            raise ValueError("text_embedding_dim must be positive")
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
        self.text_embedding_dim = int(text_embedding_dim)
        self.feature_dim = int(feature_dim)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lat4 = nn.Conv2d(512, feature_dim, 1)
        self.lat3 = nn.Conv2d(256, feature_dim, 1)
        self.lat2 = nn.Conv2d(128, feature_dim, 1)
        self.lat1 = nn.Conv2d(64, feature_dim, 1)
        self.smooth3 = conv_block(feature_dim)
        self.smooth2 = conv_block(feature_dim)
        self.smooth1 = conv_block(feature_dim)
        self.query_film = nn.Sequential(
            nn.Linear(text_embedding_dim, feature_dim * 2),
            nn.Tanh(),
        )
        self.heatmap_head = nn.Sequential(
            conv_block(feature_dim),
            nn.Conv2d(feature_dim, len(REGION_KEYPOINT_NAMES), 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(512 + text_embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.keypoint_head = nn.Sequential(
            nn.Linear(feature_dim + 128 + 1, 128),
            nn.GELU(),
        )
        self.depth_residual_head = nn.Linear(128, 1)
        self.depth_log_variance_head = nn.Linear(128, 1)
        self.visibility_head = nn.Linear(128, 1)
        self.region_geometry_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        if freeze_stem:
            for module in (self.stem, self.layer1):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def load_pregrasp_backbone(self, checkpoint: str | Path) -> list[str]:
        """Transfer only RGB-D backbone weights from Phase 3 perception."""

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        source = payload.get("model_state", payload)
        prefixes = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
        transferred = {
            key: value for key, value in source.items() if key.startswith(prefixes)
        }
        missing, unexpected = self.load_state_dict(transferred, strict=False)
        if unexpected:
            raise ValueError(
                "unexpected Phase 3 backbone keys: " + ", ".join(unexpected)
            )
        loaded = sorted(transferred)
        if not loaded or any(key in missing for key in loaded):
            raise ValueError("Phase 3 checkpoint did not provide a complete RGB-D backbone")
        return loaded

    def forward(
        self, rgbd: torch.Tensor, intent_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if rgbd.ndim != 4 or rgbd.shape[1] != 4:
            raise ValueError("rgbd must have shape [B, 4, H, W]")
        if intent_embedding.shape != (rgbd.shape[0], self.text_embedding_dim):
            raise ValueError("intent_embedding shape does not match the RGB-D batch")
        x1 = self.layer1(self.stem(rgbd))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        p3 = self.smooth3(
            F.interpolate(
                self.lat4(x4), size=x3.shape[-2:], mode="bilinear", align_corners=False
            )
            + self.lat3(x3)
        )
        p2 = self.smooth2(
            F.interpolate(p3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            + self.lat2(x2)
        )
        p1 = self.smooth1(
            F.interpolate(p2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            + self.lat1(x1)
        )
        gamma, beta = self.query_film(intent_embedding).chunk(2, dim=-1)
        p1 = p1 * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        heatmap_logits = self.heatmap_head(p1)
        probabilities = torch.softmax(
            heatmap_logits.reshape(rgbd.shape[0], len(REGION_KEYPOINT_NAMES), -1)
            / 0.25,
            dim=-1,
        )
        local_features = torch.einsum(
            "bkn,bcn->bkc",
            probabilities,
            p1.reshape(p1.shape[0], p1.shape[1], -1),
        )
        normalized_depth = F.interpolate(
            rgbd[:, 3:4], size=p1.shape[-2:], mode="bilinear", align_corners=False
        ).reshape(rgbd.shape[0], 1, -1)
        sampled_depth = torch.einsum(
            "bkn,bcn->bkc", probabilities, normalized_depth
        )
        pooled = F.adaptive_avg_pool2d(x4, 1).flatten(1)
        global_hidden = self.global_head(
            torch.cat([pooled, intent_embedding], dim=-1)
        )
        keypoint_hidden = self.keypoint_head(
            torch.cat(
                [
                    local_features,
                    global_hidden[:, None, :].expand(
                        -1, len(REGION_KEYPOINT_NAMES), -1
                    ),
                    sampled_depth,
                ],
                dim=-1,
            )
        )
        geometry = self.region_geometry_head(global_hidden)
        return {
            "heatmap_logits": heatmap_logits,
            "depth_normalized": sampled_depth.squeeze(-1)
            + self.depth_residual_head(keypoint_hidden).squeeze(-1),
            "depth_log_variance": self.depth_log_variance_head(keypoint_hidden)
            .squeeze(-1)
            .clamp(-6.0, 3.0),
            "visibility_logits": self.visibility_head(keypoint_hidden).squeeze(-1),
            "half_length_log_m": geometry[:, 0],
            "radial_half_extent_log_m": geometry[:, 1],
            "region_confidence_logits": geometry[:, 2],
            "global_hidden": global_hidden,
        }


def semantic_part_region_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_loss = F.mse_loss(
        torch.sigmoid(output["heatmap_logits"]), batch["heatmaps"]
    )
    predicted_xy, _ = softargmax_2d(output["heatmap_logits"])
    valid = batch["keypoint_valid"].unsqueeze(-1)
    coordinate_error = F.smooth_l1_loss(
        predicted_xy, batch["keypoint_xy"], reduction="none"
    )
    coordinate_loss = (coordinate_error * valid).sum() / valid.sum().clamp_min(1.0)
    depth_error = output["depth_normalized"] - batch["depth_normalized"]
    log_variance = output["depth_log_variance"]
    depth_nll = 0.5 * (
        torch.exp(-log_variance) * depth_error.square() + log_variance
    )
    depth_loss = (
        depth_nll * batch["keypoint_valid"]
    ).sum() / batch["keypoint_valid"].sum().clamp_min(1.0)
    visibility_loss = F.binary_cross_entropy_with_logits(
        output["visibility_logits"], batch["visibility"]
    )
    half_length_loss = F.smooth_l1_loss(
        output["half_length_log_m"], batch["half_length_log_m"]
    )
    radial_loss = F.smooth_l1_loss(
        output["radial_half_extent_log_m"],
        batch["radial_half_extent_log_m"],
    )
    confidence_loss = F.binary_cross_entropy_with_logits(
        output["region_confidence_logits"], batch["region_present"]
    )
    total = (
        2.0 * heatmap_loss
        + coordinate_loss
        + depth_loss
        + 0.5 * visibility_loss
        + 0.5 * half_length_loss
        + 0.5 * radial_loss
        + 0.5 * confidence_loss
    )
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "depth_nll": float(depth_loss.detach()),
        "visibility": float(visibility_loss.detach()),
        "half_length": float(half_length_loss.detach()),
        "radial_extent": float(radial_loss.detach()),
        "region_confidence": float(confidence_loss.detach()),
    }


@torch.no_grad()
def predict_semantic_part_region_node(
    model: SemanticPartRegionNet,
    rgbd: torch.Tensor,
    intent_embedding: torch.Tensor,
    camera: Mapping[str, Any],
    *,
    original_image_size: tuple[int, int],
    region_id: str,
    entity_id: str,
    semantic_role: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Convert one model prediction into the shared sparse graph node schema."""

    import numpy as np

    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()
    output = model(
        rgbd.unsqueeze(0).to(resolved_device),
        intent_embedding.unsqueeze(0).to(resolved_device),
    )
    heatmap_logits = output["heatmap_logits"]
    xy, covariance = softargmax_2d(heatmap_logits)
    heatmap_height, heatmap_width = heatmap_logits.shape[-2:]
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
    points_world = []
    covariances_world = []
    evidence = []
    for index, name in enumerate(REGION_KEYPOINT_NAMES):
        pixel = np.array([xy[index, 0] * scale_x, xy[index, 1] * scale_y])
        covariance_pixel = np.array(
            [
                [covariance[index, 0, 0] * scale_x**2, covariance[index, 0, 1] * scale_x * scale_y],
                [covariance[index, 1, 0] * scale_x * scale_y, covariance[index, 1, 1] * scale_y**2],
            ]
        )
        point, covariance_world = backproject_with_covariance(
            pixel,
            float(depth_m[index]),
            covariance_pixel,
            float(max(depth_variance[index], 1e-7)),
            intrinsic,
            extrinsic,
        )
        points_world.append(point)
        covariances_world.append(covariance_world)
        evidence.append(
            {
                "name": name,
                "pixel_uv": np.round(pixel, 4).tolist(),
                "camera_depth_m": round(float(depth_m[index]), 6),
                "visibility": round(float(visibility[index]), 6),
                "position_world_m": np.round(point, 7).tolist(),
            }
        )
    points = np.stack(points_world)
    axis_delta = points[2] - points[1]
    axis_norm = float(np.linalg.norm(axis_delta))
    if axis_norm <= 1e-6:
        raise ValueError("predicted semantic part axis is degenerate")
    axis = axis_delta / axis_norm
    half_length = float(
        np.clip(
            np.exp(float(output["half_length_log_m"][0].float().cpu())),
            0.005,
            0.5,
        )
    )
    radial_extent = float(
        np.clip(
            np.exp(float(output["radial_half_extent_log_m"][0].float().cpu())),
            0.002,
            0.15,
        )
    )
    confidence = float(
        torch.sigmoid(output["region_confidence_logits"])[0].cpu()
    ) * float(np.mean(visibility))
    return {
        "id": region_id,
        "node_type": "semantic_part_region",
        "entity_id": entity_id,
        "semantic_type": semantic_role,
        "position_mean_world_m": np.round(points[0], 7).tolist(),
        "position_covariance_m2": np.round(covariances_world[0], 9).tolist(),
        "longitudinal_axis_world": np.round(axis, 7).tolist(),
        "half_length_m": round(half_length, 7),
        "radial_half_extent_m": round(radial_extent, 7),
        "confidence": round(confidence, 6),
        "visibility": round(float(np.mean(visibility)), 6),
        "keypoint_evidence": evidence,
        "source": "learned_rgbd_text_conditioned_semantic_part_region",
        "access": "inference_visible",
    }
