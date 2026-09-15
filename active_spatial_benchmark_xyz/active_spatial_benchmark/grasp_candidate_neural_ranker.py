"""Neural query-conditioned ranker with externally generated text embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .grasp_candidate_ranker import (
    AFFORDANCE_PROFILE_FEATURE_NAMES,
    BASE_CANDIDATE_FEATURE_NAMES,
    affordance_profile_feature_vector,
    candidate_base_feature_vector,
)


INTENT_EMBEDDING_SCHEMA = "spatial.intent_embedding_store.v1"
NEURAL_RANKER_SCHEMA = "spatial.query_conditioned_candidate_ranker.v1"


@dataclass(frozen=True)
class IntentEmbeddingStore:
    encoder_id: str
    embedding_dimension: int
    normalized: bool
    embeddings: Mapping[str, np.ndarray]
    text_embeddings: Mapping[str, np.ndarray] | None = None
    sample_text_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.encoder_id, str) or not self.encoder_id.strip():
            raise ValueError("intent embedding encoder_id must be non-empty")
        if self.embedding_dimension < 1:
            raise ValueError("intent embedding dimension must be positive")
        normalized_embeddings: dict[str, np.ndarray] = {}
        for sample_id, raw_value in (self.embeddings or {}).items():
            value = np.asarray(raw_value, dtype=np.float32)
            if value.shape != (self.embedding_dimension,):
                raise ValueError(
                    f"intent embedding {sample_id!r} has shape {value.shape}, "
                    f"expected {(self.embedding_dimension,)}"
                )
            if not np.all(np.isfinite(value)) or float(np.linalg.norm(value)) <= 1e-8:
                raise ValueError(f"intent embedding {sample_id!r} must be finite and nonzero")
            if self.normalized and not np.isclose(
                float(np.linalg.norm(value)), 1.0, atol=1e-3
            ):
                raise ValueError(
                    f"intent embedding {sample_id!r} is declared normalized but has non-unit norm"
                )
            normalized_embeddings[str(sample_id)] = value
        object.__setattr__(self, "encoder_id", self.encoder_id.strip())
        object.__setattr__(self, "embeddings", normalized_embeddings)
        normalized_text_embeddings: dict[str, np.ndarray] = {}
        for digest, raw_value in (self.text_embeddings or {}).items():
            if len(str(digest)) != 64:
                raise ValueError("intent text embedding keys must be SHA-256 digests")
            value = np.asarray(raw_value, dtype=np.float32)
            if value.shape != (self.embedding_dimension,) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError("intent text embedding has an invalid vector")
            if float(np.linalg.norm(value)) <= 1e-8:
                raise ValueError("intent text embedding must be nonzero")
            if self.normalized and not np.isclose(
                float(np.linalg.norm(value)), 1.0, atol=1e-3
            ):
                raise ValueError("normalized intent text embedding has non-unit norm")
            normalized_text_embeddings[str(digest)] = value
        object.__setattr__(self, "text_embeddings", normalized_text_embeddings)
        object.__setattr__(
            self,
            "sample_text_sha256",
            {str(key): str(value) for key, value in (self.sample_text_sha256 or {}).items()},
        )

    def for_sample(self, sample_id: str) -> np.ndarray:
        try:
            return self.embeddings[sample_id].copy()
        except KeyError as exc:
            raise ValueError(
                f"intent embedding store has no vector for sample {sample_id!r}"
            ) from exc

    def for_intent(self, intent: Mapping[str, Any]) -> np.ndarray:
        text = canonical_intent_text(intent)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            return self.text_embeddings[digest].copy()
        except KeyError as exc:
            raise ValueError(
                "intent embedding store has no vector for this canonical intent text; "
                "regenerate the frozen-encoder sidecar instead of using a hash fallback"
            ) from exc

    @classmethod
    def load(cls, path: str | Path) -> "IntentEmbeddingStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != INTENT_EMBEDDING_SCHEMA:
            raise ValueError("unsupported intent embedding store schema")
        return cls(
            encoder_id=str(payload.get("encoder_id", "")),
            embedding_dimension=int(payload.get("embedding_dimension", 0)),
            normalized=bool(payload.get("normalized", False)),
            embeddings=payload.get("embeddings", {}),
            text_embeddings=payload.get("text_embeddings", {}),
            sample_text_sha256=payload.get("sample_text_sha256", {}),
        )


@dataclass(frozen=True)
class NeuralCandidateRankingGroup:
    sample_id: str
    task: str
    seed: int
    candidate_ids: tuple[str, ...]
    candidate_features: np.ndarray
    affordance_features: np.ndarray
    intent_embedding: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        candidates = np.asarray(self.candidate_features, dtype=np.float32)
        affordance = np.asarray(self.affordance_features, dtype=np.float32)
        embedding = np.asarray(self.intent_embedding, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != len(
            BASE_CANDIDATE_FEATURE_NAMES
        ):
            raise ValueError("neural candidate feature matrix has an unexpected shape")
        if affordance.shape != (len(AFFORDANCE_PROFILE_FEATURE_NAMES),):
            raise ValueError("affordance feature vector has an unexpected shape")
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError("intent embedding must be a finite vector")
        if labels.shape != (candidates.shape[0],):
            raise ValueError("neural ranker labels must align with candidates")
        if not np.all(np.isin(labels, (0.0, 1.0))):
            raise ValueError("neural ranker labels must be binary")
        if len(self.candidate_ids) != candidates.shape[0]:
            raise ValueError("neural ranker candidate ids must align with rows")
        object.__setattr__(self, "candidate_features", candidates)
        object.__setattr__(self, "affordance_features", affordance)
        object.__setattr__(self, "intent_embedding", embedding)
        object.__setattr__(self, "labels", labels)

    @property
    def has_pair(self) -> bool:
        return bool(np.any(self.labels == 1.0) and np.any(self.labels == 0.0))


def neural_group_from_sample(
    sample: Mapping[str, Any],
    embedding_store: IntentEmbeddingStore,
    *,
    label_name: str = "execution_success",
) -> NeuralCandidateRankingGroup | None:
    sample_id = str(sample.get("sample_id", ""))
    inference = sample.get("inference_visible", {})
    training = sample.get("training_only", {})
    intent = inference.get("intent", {})
    candidate_ids: list[str] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    for row in training.get("candidate_labels", ()):
        label = row.get("execution", {}).get(label_name)
        if label is None:
            continue
        candidate = row.get("candidate", {})
        candidate_ids.append(str(candidate.get("id")))
        features.append(candidate_base_feature_vector(intent, candidate))
        labels.append(float(bool(label)))
    if not features:
        return None
    return NeuralCandidateRankingGroup(
        sample_id=sample_id,
        task=str(sample.get("task", "unknown")),
        seed=int(sample.get("seed", -1)),
        candidate_ids=tuple(candidate_ids),
        candidate_features=np.stack(features),
        affordance_features=affordance_profile_feature_vector(
            intent.get("affordance_profile")
        ),
        intent_embedding=embedding_store.for_sample(sample_id),
        labels=np.asarray(labels, dtype=np.float32),
    )


class QueryConditionedCandidateRanker(nn.Module):
    def __init__(
        self,
        *,
        text_embedding_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.1,
        candidate_mean: Sequence[float] | None = None,
        candidate_scale: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if text_embedding_dim < 1 or hidden_dim < 8:
            raise ValueError("invalid neural candidate ranker dimensions")
        candidate_dim = len(BASE_CANDIDATE_FEATURE_NAMES)
        profile_dim = len(AFFORDANCE_PROFILE_FEATURE_NAMES)
        mean = np.zeros(candidate_dim, dtype=np.float32) if candidate_mean is None else np.asarray(candidate_mean, dtype=np.float32)
        scale = np.ones(candidate_dim, dtype=np.float32) if candidate_scale is None else np.asarray(candidate_scale, dtype=np.float32)
        if mean.shape != (candidate_dim,) or scale.shape != (candidate_dim,):
            raise ValueError("candidate normalization vectors have an unexpected shape")
        if np.any(scale <= 0.0):
            raise ValueError("candidate normalization scale must be positive")
        self.text_embedding_dim = int(text_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.register_buffer("candidate_mean", torch.from_numpy(mean))
        self.register_buffer("candidate_scale", torch.from_numpy(scale))
        self.candidate_tower = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_tower = nn.Sequential(
            nn.Linear(text_embedding_dim + profile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        intent_embedding: torch.Tensor,
        affordance_features: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be [N, C]")
        count = candidate_features.shape[0]
        query = torch.cat(
            [intent_embedding.reshape(-1), affordance_features.reshape(-1)], dim=0
        )
        query = query.unsqueeze(0).expand(count, -1)
        standardized = (
            candidate_features - self.candidate_mean
        ) / self.candidate_scale
        candidate_hidden = self.candidate_tower(standardized)
        query_hidden = self.query_tower(query)
        fused = torch.cat(
            [
                candidate_hidden,
                query_hidden,
                candidate_hidden * query_hidden,
                torch.abs(candidate_hidden - query_hidden),
            ],
            dim=-1,
        )
        return self.utility_head(fused).squeeze(-1)


def pairwise_candidate_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("pairwise candidate loss requires positive and negative labels")
    differences = positive[:, None] - negative[None, :]
    return F.softplus(-differences).mean()


def listwise_candidate_success_loss(
    scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Negative log probability assigned to the set of safe candidates."""

    positive = scores[labels > 0.5]
    if positive.numel() == 0 or positive.numel() == scores.numel():
        raise ValueError("listwise candidate loss requires positive and negative labels")
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(positive, dim=0)


def candidate_normalization(
    groups: Sequence[NeuralCandidateRankingGroup],
) -> tuple[np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("candidate normalization requires at least one group")
    values = np.concatenate([group.candidate_features for group in groups], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    standard_deviation = values.std(axis=0).astype(np.float32)
    scale = np.where(standard_deviation > 1e-6, standard_deviation, 1.0)
    return mean, scale


def save_neural_ranker_checkpoint(
    path: str | Path,
    model: QueryConditionedCandidateRanker,
    *,
    encoder_id: str,
    training_metadata: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": NEURAL_RANKER_SCHEMA,
        "model_type": "two_tower_query_candidate_interaction_ranker",
        "encoder_id": encoder_id,
        "text_embedding_dim": model.text_embedding_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        "candidate_feature_names": list(BASE_CANDIDATE_FEATURE_NAMES),
        "affordance_feature_names": list(AFFORDANCE_PROFILE_FEATURE_NAMES),
        "state_dict": model.state_dict(),
        "training_metadata": dict(training_metadata),
    }
    torch.save(payload, Path(path))


def load_neural_ranker_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[QueryConditionedCandidateRanker, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema_version") != NEURAL_RANKER_SCHEMA:
        raise ValueError("unsupported neural candidate ranker schema")
    if tuple(payload.get("candidate_feature_names", ())) != BASE_CANDIDATE_FEATURE_NAMES:
        raise ValueError("neural candidate feature schema mismatch")
    if tuple(payload.get("affordance_feature_names", ())) != AFFORDANCE_PROFILE_FEATURE_NAMES:
        raise ValueError("neural affordance feature schema mismatch")
    state = payload["state_dict"]
    model = QueryConditionedCandidateRanker(
        text_embedding_dim=int(payload["text_embedding_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload["dropout"]),
        candidate_mean=state["candidate_mean"].cpu().numpy(),
        candidate_scale=state["candidate_scale"].cpu().numpy(),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def canonical_intent_text(intent: Mapping[str, Any]) -> str:
    """Stable text sent to the declared external frozen text encoder."""

    fields = {
        key: intent.get(key)
        for key in (
            "target",
            "task_goal",
            "preferred_roles",
            "avoided_roles",
            "contact_pattern",
            "approach_relation",
            "closing_axis_relation",
            "grasp_depth_rule",
            "natural_language_constraints",
        )
    }
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
