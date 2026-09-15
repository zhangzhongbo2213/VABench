"""Legacy Phase 5/9 pre-grasp rankers retained for result reproducibility.

This module still uses a fixed view-name feature vocabulary. New Phase 11
active-perception work belongs in ``expert_event_view_ranker.py``. Do not use
metrics from this legacy path to claim camera-layout generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .pregrasp_graph import PREGRASP_EDGE_IDS, score_pregrasp_candidate_views


VIEW_NAMES = (
    "current",
    "topdown",
    "side",
    "front_side_45",
    "side_top_45",
    "oblique_45",
)
LEGACY_SELECTOR = True
LEGACY_SELECTOR_SCOPE = "phase1_to_phase10_pregrasp_reproducibility_only"
FEATURE_DIM = len(PREGRASP_EDGE_IDS) * 2 + 5 + len(VIEW_NAMES)
RECOVERY_CONTEXT_FIELDS = (
    "recovery_cycle_fraction",
    "cumulative_recovery_fraction",
    "observed_view_fraction",
    "remaining_view_fraction",
    "gate_confidence",
    "outcome_probability",
    "outcome_uncertainty",
    "post_recovery",
)
ONPOLICY_FEATURE_DIM = FEATURE_DIM + len(RECOVERY_CONTEXT_FIELDS)


@dataclass(frozen=True)
class ViewRankingExample:
    episode_id: str
    state_id: str
    views: tuple[str, ...]
    features: torch.Tensor
    target_utility: torch.Tensor
    candidate_metadata: tuple[Mapping[str, Any], ...] = ()


class ViewUtilityRanker(nn.Module):
    def __init__(self, hidden_dim: int = 64, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        if feature_dim not in (FEATURE_DIM, ONPOLICY_FEATURE_DIM):
            raise ValueError(f"unsupported view feature dimension: {feature_dim}")
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class RecoveryConditionedViewRanker(nn.Module):
    """A bounded residual over the previously validated Phase 5 policy."""

    feature_dim = ONPOLICY_FEATURE_DIM

    def __init__(
        self,
        *,
        base_hidden_dim: int = 64,
        residual_hidden_dim: int = 64,
        residual_scale: float = 0.025,
        override_margin: float = 0.0,
    ) -> None:
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        if override_margin < 0.0:
            raise ValueError("override_margin must be non-negative")
        self.residual_scale = float(residual_scale)
        self.override_margin = float(override_margin)
        self.base_ranker = ViewUtilityRanker(
            hidden_dim=base_hidden_dim, feature_dim=FEATURE_DIM
        )
        self.residual_ranker = ViewUtilityRanker(
            hidden_dim=residual_hidden_dim, feature_dim=ONPOLICY_FEATURE_DIM
        )
        for parameter in self.base_ranker.parameters():
            parameter.requires_grad_(False)
        self.base_ranker.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_ranker.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = self.base_ranker(features[..., :FEATURE_DIM])
        residual = torch.tanh(self.residual_ranker(features)) * self.residual_scale
        return base + residual

    def select_index(self, features: torch.Tensor) -> int:
        """Select with a deployment guard around the validated base policy."""

        scores = self(features)
        learned_index = int(torch.argmax(scores))
        base_scores = self.base_ranker(features[..., :FEATURE_DIM])
        base_index = int(torch.argmax(base_scores))
        if (
            learned_index != base_index
            and float(scores[learned_index] - scores[base_index]) < self.override_margin
        ):
            return base_index
        return learned_index


class ConservativeTrajectoryViewRanker(nn.Module):
    """Bootstrap residual ensemble with a lower-confidence-bound override gate."""

    feature_dim = ONPOLICY_FEATURE_DIM

    def __init__(
        self,
        *,
        base_hidden_dim: int = 64,
        residual_hidden_dim: int = 64,
        ensemble_size: int = 5,
        residual_scale: float = 0.02,
        override_margin: float = 0.003,
        uncertainty_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("trajectory ensemble requires at least two heads")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        if override_margin < 0.0 or uncertainty_penalty < 0.0:
            raise ValueError(
                "override margin and uncertainty penalty must be non-negative"
            )
        self.residual_scale = float(residual_scale)
        self.override_margin = float(override_margin)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.base_ranker = ViewUtilityRanker(
            hidden_dim=base_hidden_dim, feature_dim=FEATURE_DIM
        )
        self.residual_heads = nn.ModuleList(
            ViewUtilityRanker(
                hidden_dim=residual_hidden_dim,
                feature_dim=ONPOLICY_FEATURE_DIM,
            )
            for _ in range(ensemble_size)
        )
        for parameter in self.base_ranker.parameters():
            parameter.requires_grad_(False)
        self.base_ranker.eval()

    @property
    def ensemble_size(self) -> int:
        return len(self.residual_heads)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_ranker.eval()
        return self

    def member_scores(self, features: torch.Tensor) -> torch.Tensor:
        base = self.base_ranker(features[..., :FEATURE_DIM])
        members = [
            base + torch.tanh(head(features)) * self.residual_scale
            for head in self.residual_heads
        ]
        return torch.stack(members, dim=-1)

    def score_statistics(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        members = self.member_scores(features)
        return members.mean(dim=-1), members.std(dim=-1, unbiased=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_statistics(features)[0]

    def selection_details(self, features: torch.Tensor) -> dict[str, Any]:
        mean, std = self.score_statistics(features)
        base_scores = self.base_ranker(features[..., :FEATURE_DIM])
        base_index = int(torch.argmax(base_scores))
        learned_index = int(torch.argmax(mean))
        lcb_advantage = float(
            mean[learned_index]
            - self.uncertainty_penalty * std[learned_index]
            - mean[base_index]
            - self.uncertainty_penalty * std[base_index]
        )
        selected_index = learned_index
        guard_reason = None
        if learned_index != base_index and lcb_advantage < self.override_margin:
            selected_index = base_index
            guard_reason = "insufficient_lcb_advantage"
        return {
            "selected_index": selected_index,
            "base_index": base_index,
            "learned_index": learned_index,
            "mean": mean,
            "std": std,
            "lcb_advantage": lcb_advantage,
            "guard_reason": guard_reason,
        }

    def select_index(self, features: torch.Tensor) -> int:
        return int(self.selection_details(features)["selected_index"])


def build_view_ranking_examples(
    dataset_root: str | Path,
    *,
    episode_ids: Iterable[str] | None = None,
) -> list[ViewRankingExample]:
    root = Path(dataset_root).resolve()
    allowed = set(episode_ids or ())
    examples = []
    for state_path in sorted(root.glob("episodes/*/states/*/oracle/state.json")):
        episode_id = state_path.parents[3].name
        state_id = state_path.parents[1].name
        if allowed and episode_id not in allowed:
            continue
        state_root = state_path.parents[1]
        ranking_value = json.loads(
            (state_root / "oracle" / "candidate_view_ranking.json").read_text(
                encoding="utf-8"
            )
        )
        graph = json.loads(
            (state_root / "oracle" / "spatial_graph.json").read_text(encoding="utf-8")
        )
        current_evidence = ranking_value["oracle_next_tool"]["current_evidence"]
        edge_uncertainty = {
            edge_id: float(np.clip(1.0 - current_evidence.get(edge_id, 0.0), 0.0, 1.0))
            for edge_id in PREGRASP_EDGE_IDS
        }
        cameras = {}
        for camera_path in (state_root / "inference").glob("*_camera.json"):
            value = json.loads(camera_path.read_text(encoding="utf-8"))
            pose = np.asarray(value["camera_pose_world"], dtype=np.float64)
            cameras[str(value["view"])] = {
                "view": str(value["view"]),
                "camera_position_world": pose[:3, 3],
                "view_direction_world": pose[:3, 0],
                "framing_score": 1.0,
            }
        current = cameras["current"]
        rows = score_pregrasp_candidate_views(
            edge_uncertainty=edge_uncertainty,
            query_axes=graph["query_axes"],
            current_view_direction_world=current["view_direction_world"],
            current_camera_position_world=current["camera_position_world"],
            candidates=list(cameras.values()),
            visited_views=("current",),
        )
        target_map = {
            str(row["view"]): float(row["oracle_utility_proxy"])
            for row in ranking_value["rows"]
        }
        rows = [row for row in rows if row["view"] in target_map]
        if len(rows) < 2:
            continue
        examples.append(
            ViewRankingExample(
                episode_id=episode_id,
                state_id=state_id,
                views=tuple(str(row["view"]) for row in rows),
                features=torch.stack(
                    [view_feature_vector(row, edge_uncertainty) for row in rows]
                ),
                target_utility=torch.tensor(
                    [target_map[str(row["view"])] for row in rows], dtype=torch.float32
                ),
            )
        )
    if not examples:
        raise ValueError("no view-ranking examples found")
    return examples


def view_feature_vector(
    row: Mapping[str, Any],
    edge_uncertainty: Mapping[str, float],
    *,
    state_context: Mapping[str, float] | None = None,
    feature_dim: int = FEATURE_DIM,
) -> torch.Tensor:
    predicted = row["predicted"]
    observability = predicted["per_relation_observability"]
    view = str(row["view"])
    values = [
        float(edge_uncertainty.get(edge_id, 1.0)) for edge_id in PREGRASP_EDGE_IDS
    ]
    values.extend(
        float(observability.get(edge_id, 0.0)) for edge_id in PREGRASP_EDGE_IDS
    )
    values.extend(
        [
            float(predicted["relation_score"]),
            float(predicted["move_cost"]),
            float(predicted["baseline_score"]),
            float(predicted["framing_score"]),
            float(predicted["utility"]),
        ]
    )
    values.extend(1.0 if view == name else 0.0 for name in VIEW_NAMES)
    if feature_dim == ONPOLICY_FEATURE_DIM:
        context = state_context or {}
        values.extend(
            float(context.get(field, 0.0)) for field in RECOVERY_CONTEXT_FIELDS
        )
    result = torch.tensor(values, dtype=torch.float32)
    if result.shape != (feature_dim,):
        raise ValueError(f"expected {feature_dim} view features, got {result.shape}")
    return result


@torch.no_grad()
def rerank_candidate_views(
    model: nn.Module,
    rows: list[dict[str, Any]],
    *,
    edge_uncertainty: Mapping[str, float],
    device: torch.device | str,
    state_context: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    model.eval()
    features = torch.stack(
        [
            view_feature_vector(
                row,
                edge_uncertainty,
                state_context=state_context,
                feature_dim=model.feature_dim,
            )
            for row in rows
        ]
    ).to(device)
    score_tensor = model(features)
    scores = score_tensor.cpu().numpy()
    selected_index = int(torch.argmax(score_tensor))
    selection_audit = None
    if isinstance(model, RecoveryConditionedViewRanker):
        base_scores = model.base_ranker(features[..., :FEATURE_DIM])
        base_index = int(torch.argmax(base_scores))
        learned_index = selected_index
        selected_index = model.select_index(features)
        selection_audit = {
            "base_view": str(rows[base_index]["view"]),
            "learned_view": str(rows[learned_index]["view"]),
            "selected_view": str(rows[selected_index]["view"]),
            "override_margin": round(model.override_margin, 6),
            "learned_advantage_over_base": round(
                float(score_tensor[learned_index] - score_tensor[base_index]), 6
            ),
            "guard_applied": bool(selected_index != learned_index),
        }
    elif isinstance(model, ConservativeTrajectoryViewRanker):
        details = model.selection_details(features)
        selected_index = int(details["selected_index"])
        base_index = int(details["base_index"])
        learned_index = int(details["learned_index"])
        selection_audit = {
            "base_view": str(rows[base_index]["view"]),
            "learned_view": str(rows[learned_index]["view"]),
            "selected_view": str(rows[selected_index]["view"]),
            "override_margin": round(model.override_margin, 6),
            "uncertainty_penalty": round(model.uncertainty_penalty, 6),
            "learned_advantage_over_base": round(
                float(details["mean"][learned_index] - details["mean"][base_index]),
                6,
            ),
            "learned_std": round(float(details["std"][learned_index]), 6),
            "base_std": round(float(details["std"][base_index]), 6),
            "lcb_advantage_over_base": round(float(details["lcb_advantage"]), 6),
            "guard_applied": bool(selected_index != learned_index),
            "guard_reason": details["guard_reason"],
        }
    result = []
    for row, score in zip(rows, scores):
        value = dict(row)
        value["predicted"] = dict(row["predicted"])
        value["predicted"]["rule_utility"] = value["predicted"]["utility"]
        value["predicted"]["learned_utility"] = round(float(score), 6)
        value["predicted"]["utility"] = round(float(score), 6)
        if isinstance(model, ConservativeTrajectoryViewRanker):
            value["selector"] = "conservative_trajectory_return_view_ranker"
        elif model.feature_dim == ONPOLICY_FEATURE_DIM:
            value["selector"] = "learned_recovery_conditioned_view_ranker"
        else:
            value["selector"] = "learned_view_utility_ranker"
        if selection_audit is not None:
            value["selection_audit"] = dict(selection_audit)
        result.append(value)
    order = sorted(
        range(len(result)),
        key=lambda index: result[index]["predicted"]["utility"],
        reverse=True,
    )
    order.remove(selected_index)
    order.insert(0, selected_index)
    return [result[index] for index in order]


def load_view_ranker(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> nn.Module:
    value = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = value["model_config"]
    if config.get("model_type") == "recovery_conditioned_residual":
        model = RecoveryConditionedViewRanker(
            base_hidden_dim=int(config["base_hidden_dim"]),
            residual_hidden_dim=int(config["residual_hidden_dim"]),
            residual_scale=float(config["residual_scale"]),
            override_margin=float(config.get("override_margin", 0.0)),
        )
    elif config.get("model_type") == "trajectory_return_ensemble":
        model = ConservativeTrajectoryViewRanker(
            base_hidden_dim=int(config["base_hidden_dim"]),
            residual_hidden_dim=int(config["residual_hidden_dim"]),
            ensemble_size=int(config["ensemble_size"]),
            residual_scale=float(config["residual_scale"]),
            override_margin=float(config["override_margin"]),
            uncertainty_penalty=float(config["uncertainty_penalty"]),
        )
    else:
        model = ViewUtilityRanker(
            hidden_dim=int(config["hidden_dim"]),
            feature_dim=int(config.get("feature_dim", FEATURE_DIM)),
        )
    model.load_state_dict(value["model_state"])
    model.to(device)
    model.eval()
    return model
