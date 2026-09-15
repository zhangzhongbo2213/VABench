"""Task-conditioned RGB-D model for sparse metric grasp frames."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .semantic_part_region_model import SemanticPartRegionNet
from .learned_pregrasp import softargmax_2d


class ExpertGraspFrameNet(SemanticPartRegionNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.approach_axis_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self, rgbd: torch.Tensor, intent_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = super().forward(rgbd, intent_embedding)
        output["approach_axis_world"] = F.normalize(
            self.approach_axis_head(output["global_hidden"]), dim=-1, eps=1e-6
        )
        output["opening_width_log_m"] = output["half_length_log_m"]
        output["frame_confidence_logits"] = output["region_confidence_logits"]
        return output


def expert_grasp_frame_loss(
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
    opening_width_loss = F.smooth_l1_loss(
        output["opening_width_log_m"], batch["opening_width_log_m"]
    )
    approach_target = F.normalize(batch["approach_axis_world"], dim=-1)
    approach_loss = (
        1.0 - (output["approach_axis_world"] * approach_target).sum(dim=-1)
    ).mean()
    frame_confidence_loss = F.binary_cross_entropy_with_logits(
        output["frame_confidence_logits"], batch["frame_present"]
    )
    total = (
        2.0 * heatmap_loss
        + coordinate_loss
        + depth_loss
        + 0.5 * visibility_loss
        + 0.5 * opening_width_loss
        + approach_loss
        + 0.5 * frame_confidence_loss
    )
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "depth_nll": float(depth_loss.detach()),
        "visibility": float(visibility_loss.detach()),
        "opening_width": float(opening_width_loss.detach()),
        "approach_axis": float(approach_loss.detach()),
        "frame_confidence": float(frame_confidence_loss.detach()),
    }
