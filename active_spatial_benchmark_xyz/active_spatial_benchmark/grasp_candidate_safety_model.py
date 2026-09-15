"""Factorized grasp-success and collision-safety candidate model."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .grasp_candidate_ranker import (
    CandidateRankingGroup,
    PairwiseLinearCandidateRanker,
    candidate_feature_vector,
    evaluate_candidate_policy_outputs,
)


FACTORIZED_MODEL_SCHEMA = "spatial.factorized_candidate_outcome_model.v1"
SHADOW_ASSESSMENT_SCHEMA = "spatial.factorized_candidate_shadow_assessment.v1"


class FactorizedCandidateOutcomeModel:
    """Combine independently calibrated success and collision-safety heads."""

    def __init__(
        self,
        *,
        success_head: PairwiseLinearCandidateRanker | None = None,
        collision_safety_head: PairwiseLinearCandidateRanker | None = None,
    ) -> None:
        self.success_head = success_head or PairwiseLinearCandidateRanker()
        self.collision_safety_head = (
            collision_safety_head or PairwiseLinearCandidateRanker()
        )
        self.final_probability_scale: float | None = None
        self.final_probability_bias: float | None = None

    def fit(
        self,
        success_groups: Sequence[CandidateRankingGroup],
        collision_safety_groups: Sequence[CandidateRankingGroup],
        safe_groups: Sequence[CandidateRankingGroup],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
    ) -> dict[str, Any]:
        metrics = {
            "success_head": self.success_head.fit(
                success_groups,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            ),
            "collision_safety_head": self.collision_safety_head.fit(
                collision_safety_groups,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            ),
        }
        metrics["final_safe_probability_calibration"] = (
            self._fit_final_probability_calibration(safe_groups)
        )
        return metrics

    def predict_components(self, features: np.ndarray) -> dict[str, np.ndarray]:
        success = self.success_head.predict_success_probability(features)
        collision_safety = self.collision_safety_head.predict_success_probability(
            features
        )
        raw_safe_success = success * collision_safety
        safe_success = self._calibrated_safe_probability(raw_safe_success)
        return {
            "execution_success_probability": success,
            "collision_safety_probability": collision_safety,
            "raw_factorized_safe_probability": raw_safe_success,
            "task_native_safe_success_probability": safe_success,
        }

    def safe_probability(self, features: np.ndarray) -> np.ndarray:
        return self.predict_components(features)[
            "task_native_safe_success_probability"
        ]

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.safe_probability(features)

    def as_dict(
        self, *, training_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "schema_version": FACTORIZED_MODEL_SCHEMA,
            "model_type": "factorized_execution_success_and_collision_safety",
            "success_head": self.success_head.as_dict(),
            "collision_safety_head": self.collision_safety_head.as_dict(),
            "combination": {
                "method": "probability_product_then_training_only_platt_calibration",
                "assumption": (
                    "conditional independence baseline; deployment gate is measured "
                    "against task_native_safe_success"
                ),
                "final_probability_calibration": {
                    "scale": self.final_probability_scale,
                    "bias": self.final_probability_bias,
                },
            },
            "training_metadata": dict(training_metadata or {}),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FactorizedCandidateOutcomeModel":
        if payload.get("schema_version") != FACTORIZED_MODEL_SCHEMA:
            raise ValueError("unsupported factorized candidate outcome schema")
        model = cls(
            success_head=PairwiseLinearCandidateRanker.from_dict(
                payload.get("success_head", {})
            ),
            collision_safety_head=PairwiseLinearCandidateRanker.from_dict(
                payload.get("collision_safety_head", {})
            ),
        )
        calibration = payload.get("combination", {}).get(
            "final_probability_calibration", {}
        )
        scale = float(calibration.get("scale"))
        bias = float(calibration.get("bias"))
        if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(bias):
            raise ValueError("factorized final probability calibration is invalid")
        model.final_probability_scale = scale
        model.final_probability_bias = bias
        return model

    def _fit_final_probability_calibration(
        self, groups: Sequence[CandidateRankingGroup]
    ) -> dict[str, Any]:
        labels = np.concatenate([group.labels for group in groups])
        if not np.any(labels == 1.0) or not np.any(labels == 0.0):
            raise ValueError("final safe calibration needs positive and negative labels")
        raw = np.concatenate(
            [self._raw_factorized_probability(group.features) for group in groups]
        )
        logits = _probability_logit(raw)
        prevalence = float(np.mean(labels))
        scale = 1.0 / max(float(np.std(logits)), 1e-3)
        bias = float(np.log(prevalence / (1.0 - prevalence)))
        learning_rate = 0.05
        for _ in range(1200):
            probability = _sigmoid(scale * logits + bias)
            residual = probability - labels
            scale -= learning_rate * float(np.mean(residual * logits))
            bias -= learning_rate * float(np.mean(residual))
            scale = max(scale, 1e-6)
        self.final_probability_scale = float(scale)
        self.final_probability_bias = float(bias)
        calibrated = self._calibrated_safe_probability(raw)
        return {
            "method": "platt_logistic_on_factorized_training_logits",
            "candidate_count": len(labels),
            "positive_count": int(np.sum(labels == 1.0)),
            "negative_count": int(np.sum(labels == 0.0)),
            "scale": self.final_probability_scale,
            "bias": self.final_probability_bias,
            "brier_score": float(np.mean(np.square(calibrated - labels))),
        }

    def _raw_factorized_probability(self, features: np.ndarray) -> np.ndarray:
        success = self.success_head.predict_success_probability(features)
        collision = self.collision_safety_head.predict_success_probability(features)
        return success * collision

    def _calibrated_safe_probability(self, raw: np.ndarray) -> np.ndarray:
        if self.final_probability_scale is None or self.final_probability_bias is None:
            raise RuntimeError("factorized model has no final probability calibration")
        logits = _probability_logit(raw)
        return _sigmoid(
            self.final_probability_scale * logits + self.final_probability_bias
        )


def evaluate_factorized_candidate_policy(
    model: FactorizedCandidateOutcomeModel,
    safe_groups: Sequence[CandidateRankingGroup],
    *,
    authorization_threshold: float = 0.5,
) -> dict[str, Any]:
    probabilities = [model.safe_probability(group.features) for group in safe_groups]
    return evaluate_candidate_policy_outputs(
        safe_groups,
        probabilities,
        probabilities,
        authorization_threshold=authorization_threshold,
    )


def score_inference_candidate_set_shadow(
    model: FactorizedCandidateOutcomeModel,
    *,
    intent: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    candidate_graph: Mapping[str, Any],
    view_assessment: Mapping[str, Any],
    authorization_threshold: float = 0.9,
) -> dict[str, Any]:
    """Score inference candidates without connecting the result to execution.

    Candidate ``score`` remains the selection policy.  The factorized model is
    deliberately an authorization opinion only, and all graph inputs are
    passed through the same inference-visible feature allowlist used in
    training.
    """

    if not 0.0 < authorization_threshold < 1.0:
        raise ValueError("authorization threshold must be in (0, 1)")
    if not candidates:
        raise ValueError("shadow candidate assessment needs candidates")
    context = {
        "candidate_graph": candidate_graph,
        "view_assessment": view_assessment,
    }
    candidate_values = [
        dict(candidate) if isinstance(candidate, Mapping) else candidate
        for candidate in candidates
    ]
    features = np.stack(
        [
            candidate_feature_vector(
                intent,
                candidate,
                evidence_context=context,
            )
            for candidate in candidate_values
        ]
    )
    components = model.predict_components(features)
    rows = []
    for index, candidate in enumerate(candidate_values):
        rows.append(
            {
                "candidate_id": str(candidate.get("id", index)),
                "selection_score": float(candidate.get("score", 0.0)),
                "execution_success_probability": float(
                    components["execution_success_probability"][index]
                ),
                "collision_safety_probability": float(
                    components["collision_safety_probability"][index]
                ),
                "raw_factorized_safe_probability": float(
                    components["raw_factorized_safe_probability"][index]
                ),
                "task_native_safe_success_probability": float(
                    components["task_native_safe_success_probability"][index]
                ),
                "would_cross_shadow_threshold": bool(
                    components["task_native_safe_success_probability"][index]
                    >= authorization_threshold
                ),
            }
        )
    selected_index = int(
        np.argmax([float(candidate.get("score", 0.0)) for candidate in candidate_values])
    )
    selected = rows[selected_index]
    return {
        "schema_version": SHADOW_ASSESSMENT_SCHEMA,
        "status": "shadow_only",
        "authorization_connected": False,
        "selection_policy": "candidate_score",
        "authorization_policy": "factorized_safe_probability_shadow",
        "authorization_threshold": float(authorization_threshold),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_safe_probability": selected[
            "task_native_safe_success_probability"
        ],
        "selected_candidate_would_cross_shadow_threshold": selected[
            "would_cross_shadow_threshold"
        ],
        "candidates": rows,
        "graph_evidence_status": _shadow_graph_evidence_status(candidate_graph),
    }


def _shadow_graph_evidence_status(graph: Mapping[str, Any]) -> str:
    fusion = graph.get("multiview_fusion")
    if not isinstance(fusion, Mapping):
        return "missing"
    return str(fusion.get("status", "unknown"))


def _probability_logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
