"""Inference-visible features and a compact grasp-outcome probability head."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .pregrasp_graph import PREGRASP_EDGE_IDS


FEATURE_NAMES = tuple(
    [f"{edge_id}.probability" for edge_id in PREGRASP_EDGE_IDS]
    + [f"{edge_id}.uncertainty" for edge_id in PREGRASP_EDGE_IDS]
    + [
        "enclosure_margin_scaled",
        "axis_offset_scaled",
        "axis_offset_abs_scaled",
        "axis_limit_scaled",
        "axis_offset_ratio",
        "vertical_offset_scaled",
        "vertical_offset_abs_scaled",
        "perpendicular_error_scaled",
        "observed_view_fraction",
        "gate_confidence",
        "minimum_edge_view_fraction",
        "mean_edge_view_fraction",
        "gate_execute",
        "gate_adjust",
        "gate_uncertain",
    ]
)
FEATURE_DIM = len(FEATURE_NAMES)


class GraspOutcomeNet(nn.Module):
    """Predict whether closing and lifting from the current belief will succeed."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def extract_outcome_features(
    final_gate: Mapping[str, Any],
    *,
    observed_view_count: int,
) -> np.ndarray:
    relations = final_gate.get("relations", {})
    values: list[float] = []
    for edge_id in PREGRASP_EDGE_IDS:
        relation = relations.get(edge_id, {})
        values.append(float(relation.get("probability", 0.5)))
    for edge_id in PREGRASP_EDGE_IDS:
        relation = relations.get(edge_id, {})
        values.append(float(relation.get("uncertainty", 1.0)))

    between = relations.get("object_between_fingers", {}).get("measurement", {})
    along = relations.get("grasp_region_along_object_axis", {}).get("measurement", {})
    height = relations.get("grasp_height_aligned", {}).get("measurement", {})
    perpendicular = relations.get("closing_axis_perpendicular_to_object_axis", {}).get("measurement", {})
    enclosure_margin = float(between.get("predicted_enclosure_margin_m", 0.0))
    axis_offset = float(along.get("predicted_axis_offset_m", 0.0))
    axis_limit = max(float(along.get("predicted_axis_limit_m", 0.06)), 1e-4)
    vertical_offset = float(height.get("predicted_vertical_offset_m", 0.0))
    axis_angle = float(perpendicular.get("predicted_axis_angle_deg", 90.0))
    edge_view_counts = [
        len(set(relations.get(edge_id, {}).get("evidence_views", [])))
        for edge_id in PREGRASP_EDGE_IDS
    ]
    verdict = str(final_gate.get("verdict", "uncertain"))
    values.extend(
        [
            enclosure_margin / 0.08,
            axis_offset / 0.15,
            abs(axis_offset) / 0.15,
            axis_limit / 0.08,
            min(abs(axis_offset) / axis_limit, 3.0),
            vertical_offset / 0.08,
            abs(vertical_offset) / 0.08,
            min(abs(90.0 - axis_angle) / 45.0, 2.0),
            min(max(int(observed_view_count), 0) / 4.0, 1.5),
            float(final_gate.get("confidence", 0.0)),
            min(edge_view_counts, default=0) / 4.0,
            float(np.mean(edge_view_counts)) / 4.0 if edge_view_counts else 0.0,
            float(verdict == "execute"),
            float(verdict == "adjust"),
            float(verdict == "uncertain"),
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid outcome features with shape {result.shape}")
    return result


def binary_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    threshold: float,
) -> dict[str, float | int]:
    probability = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    prediction = probability >= float(threshold)
    positive = truth == 1
    negative = ~positive
    true_positive = int(np.count_nonzero(prediction & positive))
    false_positive = int(np.count_nonzero(prediction & negative))
    true_negative = int(np.count_nonzero(~prediction & negative))
    false_negative = int(np.count_nonzero(~prediction & positive))
    tpr = true_positive / max(int(np.count_nonzero(positive)), 1)
    tnr = true_negative / max(int(np.count_nonzero(negative)), 1)
    return {
        "count": int(truth.size),
        "threshold": float(threshold),
        "accuracy": float(np.mean(prediction == positive)) if truth.size else 0.0,
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "execution_precision": true_positive / max(true_positive + false_positive, 1),
        "successful_grasp_retention": tpr,
        "failed_grasp_block_rate": tnr,
        "unsafe_execute_count": false_positive,
        "missed_success_count": false_negative,
        "brier": float(np.mean((probability - truth) ** 2)) if truth.size else 0.0,
    }


def choose_safety_threshold(probabilities: Sequence[float], labels: Sequence[int | bool]) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    candidates = sorted(set([0.5, 1.0, *probability.tolist()]))
    rows = [binary_metrics(probability, labels, threshold=value) for value in candidates]
    safe = [row for row in rows if row["unsafe_execute_count"] == 0]
    pool = safe or rows
    best = max(
        pool,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["successful_grasp_retention"]),
            float(row["accuracy"]),
            float(row["threshold"]),
        ),
    )
    return {"selected": best, "candidates": rows, "zero_unsafe_available": bool(safe)}


def load_outcome_checkpoint(path: str, *, device: torch.device) -> tuple[GraspOutcomeNet, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    model = GraspOutcomeNet(**value["model_config"])
    model.load_state_dict(value["model_state"])
    model.to(device)
    model.eval()
    return model, value
