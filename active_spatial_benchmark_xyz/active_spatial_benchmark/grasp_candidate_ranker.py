"""Leakage-controlled baseline ranker for task-conditioned grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RANKER_SCHEMA = "spatial.grasp_candidate_pairwise_ranker.v6"
LEGACY_RANKER_SCHEMAS = {
    "spatial.grasp_candidate_pairwise_ranker.v1",
    "spatial.grasp_candidate_pairwise_ranker.v2",
    "spatial.grasp_candidate_pairwise_ranker.v3",
    "spatial.grasp_candidate_pairwise_ranker.v4",
    "spatial.grasp_candidate_pairwise_ranker.v5",
}
SUPPORTED_CANDIDATE_SAMPLE_SCHEMAS = {
    "spatial.grasp_candidate_sample.v1",
    "spatial.open_vocab_grasp_candidate_probe_sample.v1",
}
BASE_CANDIDATE_FEATURE_NAMES = (
    "analytic_score",
    "score_confidence",
    "eligible",
    "semantic_match_probability",
    "antipodal_probability",
    "task_compatibility_probability",
    "opening_width_feasible_probability",
    "support_clearance_probability",
    "opening_width_m",
    "candidate_center_world_x_m",
    "candidate_center_world_y_m",
    "candidate_center_world_z_m",
    "pregrasp_center_world_x_m",
    "pregrasp_center_world_y_m",
    "pregrasp_center_world_z_m",
    "approach_axis_world_x",
    "approach_axis_world_y",
    "approach_axis_world_z",
    "closing_axis_world_x",
    "closing_axis_world_y",
    "closing_axis_world_z",
    "absolute_approach_axis_z",
    "absolute_closing_axis_z",
    "absolute_along_region_fraction",
    "vertical_offset_m",
    "non_primary_closing_axis",
    "position_covariance_trace_m2",
    "orientation_covariance_trace_rad2",
    "unknown_constraint_fraction",
    "intent_confidence",
    "intent_constraint_count",
    "longitudinal_offset_m",
    "closing_offset_m",
    "approach_offset_m",
    "center_offset_norm_m",
    "orientation_offset_deg",
    "absolute_orientation_offset_deg",
    "opening_width_scale",
    "opening_width_scale_delta",
    "semantic_role_matches_preference",
    "preexecution_context_available",
    "workspace_bounds_pass",
    "pregrasp_reachable",
    "grasp_reachable",
    "log1p_pregrasp_waypoint_count",
    "log1p_grasp_waypoint_count",
    "corridor_clear",
    "corridor_coverage_sufficient",
    "corridor_evidence_view_fraction",
    "log1p_corridor_local_voxel_count",
    "log1p_corridor_obstacle_voxel_count",
    "corridor_minimum_non_target_clearance_m",
    "support_plane_z_m",
    "candidate_center_support_clearance_m",
    "grasp_control_support_clearance_m",
    "pregrasp_control_support_clearance_m",
    "grasp_control_workspace_x_normalized",
    "grasp_control_workspace_y_normalized",
    "grasp_control_workspace_z_normalized",
    "current_gripper_to_pregrasp_distance_m",
    "planner_collision_check_available",
    "log1p_pregrasp_self_collision_count",
    "log1p_pregrasp_environment_collision_count",
    "log1p_pregrasp_support_collision_count",
    "log1p_grasp_self_collision_count",
    "log1p_grasp_environment_collision_count",
    "log1p_grasp_support_collision_count",
    "planner_self_collision_free",
    "planner_support_collision_free",
    "visual_frame_confidence_probability",
    "contact_width_consistency_probability",
)
INTENT_HASH_DIM = 16
INTENT_INTERACTION_BASIS_NAMES = (
    "absolute_along_region_fraction",
    "vertical_offset_m",
    "non_primary_closing_axis",
    "opening_width_m",
    "task_compatibility_probability",
    "support_clearance_probability",
    "longitudinal_offset_m",
    "closing_offset_m",
    "approach_offset_m",
    "orientation_offset_deg",
    "opening_width_scale_delta",
    "candidate_center_support_clearance_m",
    "absolute_approach_axis_z",
    "absolute_closing_axis_z",
    "current_gripper_to_pregrasp_distance_m",
)
AFFORDANCE_PROFILE_FEATURE_NAMES = (
    "part_shape_slender_cylinder",
    "part_shape_broad_cylinder",
    "part_shape_box",
    "part_shape_handle",
    "part_shape_flat",
    "part_shape_irregular",
    "symmetry_continuous",
    "symmetry_two_fold",
    "symmetry_four_fold",
    "symmetry_asymmetric",
    "centering_strict",
    "centering_moderate",
    "centering_permissive",
    "vertical_strict",
    "vertical_moderate",
    "vertical_permissive",
    "avoid_ends",
    "requires_bilateral_contact",
    "requires_dual_arm",
)
HASH_INTERACTION_FEATURE_NAMES = tuple(
    f"intent_hash_{bucket:02d}_x_{basis}"
    for bucket in range(INTENT_HASH_DIM)
    for basis in INTENT_INTERACTION_BASIS_NAMES
)
AFFORDANCE_INTERACTION_FEATURE_NAMES = tuple(
    f"affordance_{profile}_x_{basis}"
    for profile in AFFORDANCE_PROFILE_FEATURE_NAMES
    for basis in INTENT_INTERACTION_BASIS_NAMES
)
GRAPH_EVIDENCE_FEATURE_NAMES = (
    "graph_evidence_available",
    "fusion_evidence_available",
    "fusion_dominant_consensus",
    "fusion_ambiguous_no_dominant_consensus",
    "fusion_observed_view_count_scaled",
    "fusion_selected_view_fraction",
    "fusion_suppressed_view_fraction",
    "fusion_mean_point_residual_m",
    "fusion_max_point_residual_m",
    "fusion_mean_residual_cutoff_ratio",
    "fusion_max_residual_cutoff_ratio",
    "fusion_competitor_ratio_available",
    "fusion_same_size_competitor_weight_ratio",
    "candidate_identity_entropy",
    "visual_evidence_sufficient",
    "top_candidate_position_std_m",
    "candidate_score_margin",
)
CANDIDATE_FEATURE_NAMES = (
    BASE_CANDIDATE_FEATURE_NAMES
    + HASH_INTERACTION_FEATURE_NAMES
    + AFFORDANCE_INTERACTION_FEATURE_NAMES
    + GRAPH_EVIDENCE_FEATURE_NAMES
)
FORBIDDEN_CANDIDATE_FEATURE_KEYS = {
    "execution",
    "execution_success",
    "task_native_safe_success",
    "task_success",
    "ik_reachable",
    "collision_free",
    "bilateral_contact",
    "lifted_from_support",
}


@dataclass(frozen=True)
class CandidateRankingGroup:
    sample_id: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(CANDIDATE_FEATURE_NAMES):
            raise ValueError("candidate feature matrix has an unexpected shape")
        if labels.shape != (features.shape[0],):
            raise ValueError("candidate labels must align with feature rows")
        if len(self.candidate_ids) != features.shape[0]:
            raise ValueError("candidate ids must align with feature rows")
        if not np.all(np.isin(labels, (0.0, 1.0))):
            raise ValueError("ranker labels must be binary")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)


def candidate_feature_vector(
    intent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    evidence_context: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Encode only fields available before candidate execution.

    This function intentionally uses an allowlist.  Passing a full labelled row
    instead of its ``candidate`` mapping fails immediately rather than silently
    leaking simulator outcomes into the model.
    """

    base = candidate_base_feature_vector(intent, candidate)
    base_by_name = dict(zip(BASE_CANDIDATE_FEATURE_NAMES, base))
    interaction_basis = np.asarray(
        [base_by_name[name] for name in INTENT_INTERACTION_BASIS_NAMES],
        dtype=np.float64,
    )
    intent_hash = _hashed_intent_vector(intent)
    hash_interactions = np.outer(intent_hash, interaction_basis).reshape(-1)
    affordance = affordance_profile_feature_vector(intent.get("affordance_profile"))
    affordance_interactions = np.outer(affordance, interaction_basis).reshape(-1)
    graph_evidence = candidate_graph_evidence_feature_vector(evidence_context)
    result = np.concatenate(
        [base, hash_interactions, affordance_interactions, graph_evidence]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("candidate features must be finite")
    return result


def candidate_base_feature_vector(
    intent: Mapping[str, Any], candidate: Mapping[str, Any]
) -> np.ndarray:
    """Return the allowlisted numeric candidate/query features without text hashing."""

    forbidden = FORBIDDEN_CANDIDATE_FEATURE_KEYS.intersection(candidate)
    if forbidden:
        raise ValueError(
            "execution-only keys passed to candidate feature encoder: "
            + ", ".join(sorted(forbidden))
        )
    checks = candidate.get("checks", {})
    parameters = candidate.get("generation_parameters", {})
    frame = candidate.get("frame", {})
    context = candidate.get("preexecution_context")
    context = context if isinstance(context, Mapping) else {}
    context_available = bool(context)
    preferred_roles = tuple(str(value) for value in intent.get("preferred_roles", ()))
    constraints = intent.get("natural_language_constraints", ())
    unknown = candidate.get("unknown_constraints", ())
    check_count = max(1, len(checks))

    base_values = (
        _number(candidate.get("score")),
        _number(candidate.get("score_confidence")),
        float(bool(candidate.get("eligible", False))),
        _check_probability(checks, "semantic_match"),
        _check_probability(checks, "antipodal_geometry"),
        _check_probability(checks, "task_compatibility"),
        _check_probability(checks, "opening_width_feasible"),
        _check_probability(checks, "support_clearance"),
        _number(
            frame.get("opening_width_m", candidate.get("opening_width_m"))
        ),
        _vector_component(frame.get("center_world_m"), 0),
        _vector_component(frame.get("center_world_m"), 1),
        _vector_component(frame.get("center_world_m"), 2),
        _vector_component(frame.get("pregrasp_center_world_m"), 0),
        _vector_component(frame.get("pregrasp_center_world_m"), 1),
        _vector_component(frame.get("pregrasp_center_world_m"), 2),
        _vector_component(frame.get("approach_axis_world"), 0),
        _vector_component(frame.get("approach_axis_world"), 1),
        _vector_component(frame.get("approach_axis_world"), 2),
        _vector_component(frame.get("closing_axis_world"), 0),
        _vector_component(frame.get("closing_axis_world"), 1),
        _vector_component(frame.get("closing_axis_world"), 2),
        abs(_vector_component(frame.get("approach_axis_world"), 2)),
        abs(_vector_component(frame.get("closing_axis_world"), 2)),
        abs(_number(parameters.get("along_region_fraction"))),
        _number(parameters.get("vertical_offset_m")),
        float(int(parameters.get("closing_axis_index", 0)) != 0),
        _matrix_trace(candidate.get("position_covariance_m2")),
        _matrix_trace(candidate.get("orientation_covariance_rad2")),
        float(len(unknown)) / float(check_count),
        _number(intent.get("confidence"), default=0.5),
        float(len(constraints)),
        _number(parameters.get("longitudinal_offset_m")),
        _number(parameters.get("closing_offset_m")),
        _number(parameters.get("approach_offset_m")),
        _vector_norm(parameters.get("center_delta_world_m")),
        _number(parameters.get("orientation_offset_deg")),
        abs(_number(parameters.get("orientation_offset_deg"))),
        _number(parameters.get("opening_width_scale"), default=1.0),
        _number(parameters.get("opening_width_scale"), default=1.0) - 1.0,
        float(str(candidate.get("semantic_role", "")) in preferred_roles),
        float(context_available),
        _context_boolean(context, "workspace_bounds_pass"),
        _context_boolean(context, "pregrasp_reachable"),
        _context_boolean(context, "grasp_reachable"),
        math.log1p(max(0.0, _number(context.get("pregrasp_waypoint_count")))),
        math.log1p(max(0.0, _number(context.get("grasp_waypoint_count")))),
        _context_boolean(context, "corridor_clear"),
        _context_boolean(context, "corridor_coverage_sufficient"),
        min(1.0, max(0.0, _number(context.get("corridor_evidence_view_count")) / 3.0)),
        math.log1p(max(0.0, _number(context.get("corridor_local_voxel_count")))),
        math.log1p(max(0.0, _number(context.get("corridor_obstacle_voxel_count")))),
        _number(context.get("corridor_minimum_non_target_clearance_m")),
        _number(context.get("support_plane_z_m")),
        _number(context.get("candidate_center_support_clearance_m")),
        _number(context.get("grasp_control_support_clearance_m")),
        _number(context.get("pregrasp_control_support_clearance_m")),
        _vector_component(context.get("grasp_control_workspace_normalized"), 0),
        _vector_component(context.get("grasp_control_workspace_normalized"), 1),
        _vector_component(context.get("grasp_control_workspace_normalized"), 2),
        _number(context.get("current_gripper_to_pregrasp_distance_m")),
        _context_boolean(context, "planner_collision_check_available"),
        math.log1p(
            max(0.0, _number(context.get("pregrasp_self_collision_count")))
        ),
        math.log1p(
            max(0.0, _number(context.get("pregrasp_environment_collision_count")))
        ),
        math.log1p(
            max(0.0, _number(context.get("pregrasp_support_collision_count")))
        ),
        math.log1p(max(0.0, _number(context.get("grasp_self_collision_count")))),
        math.log1p(
            max(0.0, _number(context.get("grasp_environment_collision_count")))
        ),
        math.log1p(
            max(0.0, _number(context.get("grasp_support_collision_count")))
        ),
        _context_boolean(context, "planner_self_collision_free"),
        _context_boolean(context, "planner_support_collision_free"),
        _check_probability(checks, "visual_frame_confidence"),
        _check_probability(checks, "contact_width_consistency"),
    )
    result = np.asarray(base_values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("candidate base features must be finite")
    return result


def ranking_group_from_sample(
    sample: Mapping[str, Any],
    *,
    label_name: str = "task_native_safe_success",
) -> CandidateRankingGroup | None:
    inference = sample.get("inference_visible", {})
    training = sample.get("training_only", {})
    intent = inference.get("intent", {})
    rows = training.get("candidate_labels", ())
    candidate_ids: list[str] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    for row in rows:
        execution = row.get("execution", {})
        label = execution.get(label_name)
        if label is None:
            continue
        candidate = row.get("candidate", {})
        candidate_ids.append(str(candidate.get("id")))
        features.append(
            candidate_feature_vector(
                intent,
                candidate,
                evidence_context=inference,
            )
        )
        labels.append(float(bool(label)))
    if not features:
        return None
    return CandidateRankingGroup(
        sample_id=str(sample.get("sample_id", "unknown")),
        candidate_ids=tuple(candidate_ids),
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.float64),
    )


class PairwiseLinearCandidateRanker:
    """Standardized linear utility trained on successful/failed candidate pairs."""

    def __init__(self) -> None:
        self.feature_mean = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float64)
        self.feature_scale = np.ones(len(CANDIDATE_FEATURE_NAMES), dtype=np.float64)
        self.weights = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float64)
        self.probability_feature_mean = np.zeros(
            len(CANDIDATE_FEATURE_NAMES), dtype=np.float64
        )
        self.probability_feature_scale = np.ones(
            len(CANDIDATE_FEATURE_NAMES), dtype=np.float64
        )
        self.probability_weights: np.ndarray | None = None
        self.probability_bias: float | None = None
        self.fitted = False

    def fit(
        self,
        groups: Sequence[CandidateRankingGroup],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
    ) -> dict[str, Any]:
        if epochs < 1 or learning_rate <= 0.0 or l2 < 0.0:
            raise ValueError("invalid ranker optimization settings")
        eligible_groups = [group for group in groups if _has_positive_and_negative(group)]
        if not eligible_groups:
            raise ValueError("pairwise training needs at least one positive/negative state")
        all_features = np.concatenate([group.features for group in eligible_groups], axis=0)
        self.feature_mean = np.mean(all_features, axis=0)
        scale = np.std(all_features, axis=0)
        self.feature_scale = np.where(scale > 1e-8, scale, 1.0)
        pair_features: list[np.ndarray] = []
        pair_labels: list[float] = []
        for group in eligible_groups:
            standardized = self._standardize(group.features)
            positive = standardized[group.labels == 1.0]
            negative = standardized[group.labels == 0.0]
            for positive_row in positive:
                for negative_row in negative:
                    difference = positive_row - negative_row
                    pair_features.extend((difference, -difference))
                    pair_labels.extend((1.0, 0.0))
        x = np.stack(pair_features)
        y = np.asarray(pair_labels, dtype=np.float64)
        self.weights.fill(0.0)
        for _ in range(epochs):
            probabilities = _sigmoid(x @ self.weights)
            gradient = x.T @ (probabilities - y) / float(len(y)) + l2 * self.weights
            self.weights -= learning_rate * gradient
        self.fitted = True
        calibration = self._fit_probability_calibration(
            groups,
            epochs=epochs,
            learning_rate=min(learning_rate, 0.05),
            l2=max(l2, 0.01),
        )
        metrics = evaluate_candidate_ranker(self, eligible_groups)
        metrics.update(
            {
                "pair_count": len(y) // 2,
                "training_group_count": len(eligible_groups),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "l2": float(l2),
                "probability_calibration": calibration,
            }
        )
        return metrics

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("candidate ranker has not been fitted")
        values = np.asarray(features, dtype=np.float64)
        return self._standardize(values) @ self.weights

    def predict_success_probability(self, features: np.ndarray) -> np.ndarray:
        if self.probability_weights is None or self.probability_bias is None:
            raise RuntimeError("candidate ranker has no probability calibration")
        values = np.asarray(features, dtype=np.float64)
        standardized = (
            values - self.probability_feature_mean
        ) / self.probability_feature_scale
        logits = standardized @ self.probability_weights + self.probability_bias
        return _sigmoid(logits)

    @property
    def has_probability_calibration(self) -> bool:
        return self.probability_weights is not None and self.probability_bias is not None

    def as_dict(self, *, training_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("candidate ranker has not been fitted")
        return {
            "schema_version": RANKER_SCHEMA,
            "model_type": "standardized_pairwise_linear_ranker",
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "probability_calibration": (
                {
                    "method": "pointwise_logistic_on_absolute_candidate_features",
                    "feature_mean": self.probability_feature_mean.tolist(),
                    "feature_scale": self.probability_feature_scale.tolist(),
                    "weights": self.probability_weights.tolist(),
                    "bias": float(self.probability_bias),
                }
                if self.has_probability_calibration
                else None
            ),
            "training_metadata": dict(training_metadata or {}),
            "input_contract": {
                "uses_execution_labels_as_features": False,
                "forbidden_feature_keys": sorted(FORBIDDEN_CANDIDATE_FEATURE_KEYS),
                "candidate_source_requirement": "inference-visible learned graph at deployment",
                "state_evidence": {
                    "feature_names": list(GRAPH_EVIDENCE_FEATURE_NAMES),
                    "source": "allowlisted inference_visible.candidate_graph fields",
                    "training_only_fields_traversed": False,
                },
                "query_conditioning": {
                    "type": "deterministic_hashed_intent_candidate_interactions",
                    "hash_dimension": INTENT_HASH_DIM,
                    "structured_affordance_features": list(
                        AFFORDANCE_PROFILE_FEATURE_NAMES
                    ),
                    "status": "small-data baseline; replace with a learned text encoder",
                },
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseLinearCandidateRanker":
        if payload.get("schema_version") in LEGACY_RANKER_SCHEMAS:
            raise ValueError(
                "legacy candidate-ranker checkpoints lack the current absolute-frame, "
                "support/collision, graph-evidence, or reject-calibration contract; "
                "retrain with v6"
            )
        if payload.get("schema_version") != RANKER_SCHEMA:
            raise ValueError("unsupported candidate ranker schema")
        if tuple(payload.get("feature_names", ())) != CANDIDATE_FEATURE_NAMES:
            raise ValueError("candidate ranker feature schema mismatch")
        model = cls()
        model.feature_mean = _vector(payload.get("feature_mean"))
        model.feature_scale = _vector(payload.get("feature_scale"))
        model.weights = _vector(payload.get("weights"))
        if np.any(model.feature_scale <= 0.0):
            raise ValueError("candidate ranker feature scales must be positive")
        calibration = payload.get("probability_calibration")
        if calibration is not None:
            feature_mean = _vector(calibration.get("feature_mean"))
            feature_scale = _vector(calibration.get("feature_scale"))
            weights = _vector(calibration.get("weights"))
            bias = float(calibration["bias"])
            if np.any(feature_scale <= 0.0) or not np.isfinite(bias):
                raise ValueError("candidate probability calibration is invalid")
            model.probability_feature_mean = feature_mean
            model.probability_feature_scale = feature_scale
            model.probability_weights = weights
            model.probability_bias = bias
        model.fitted = True
        return model

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_scale

    def _fit_probability_calibration(
        self,
        groups: Sequence[CandidateRankingGroup],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
    ) -> dict[str, Any]:
        """Fit an absolute candidate classifier for the reject option.

        Pairwise score differences deliberately remove state-level offsets, so a
        one-dimensional calibration of those scores cannot reliably recognize an
        all-negative state.  This head sees the deployment-available absolute
        candidate/context features while the pairwise head remains responsible for
        within-state ordering.
        """

        features = np.concatenate([group.features for group in groups], axis=0)
        labels = np.concatenate([group.labels for group in groups])
        if not np.any(labels == 1.0) or not np.any(labels == 0.0):
            raise ValueError("probability calibration needs positive and negative labels")
        self.probability_feature_mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        self.probability_feature_scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (
            features - self.probability_feature_mean
        ) / self.probability_feature_scale
        prevalence = float(np.mean(labels))
        weights = np.zeros(standardized.shape[1], dtype=np.float64)
        bias = float(np.log(prevalence / (1.0 - prevalence)))
        for _ in range(epochs):
            probability = _sigmoid(standardized @ weights + bias)
            residual = probability - labels
            gradient = standardized.T @ residual / float(len(labels)) + l2 * weights
            weights -= learning_rate * gradient
            bias -= learning_rate * float(np.mean(residual))
        self.probability_weights = weights
        self.probability_bias = float(bias)
        probability = self.predict_success_probability(features)
        return {
            "method": "pointwise_logistic_on_absolute_candidate_features",
            "candidate_count": len(labels),
            "positive_count": int(np.sum(labels == 1.0)),
            "negative_count": int(np.sum(labels == 0.0)),
            "bias": self.probability_bias,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "brier_score": float(np.mean(np.square(probability - labels))),
        }


def evaluate_candidate_ranker(
    model: PairwiseLinearCandidateRanker,
    groups: Sequence[CandidateRankingGroup],
) -> dict[str, Any]:
    return evaluate_candidate_scores(
        groups,
        [model.score(group.features) for group in groups],
    )


def evaluate_calibrated_candidate_policy(
    model: PairwiseLinearCandidateRanker,
    groups: Sequence[CandidateRankingGroup],
    *,
    authorization_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate ranking plus the ability to abstain when every candidate is unsafe."""

    return evaluate_candidate_policy_outputs(
        groups,
        [model.score(group.features) for group in groups],
        [model.predict_success_probability(group.features) for group in groups],
        authorization_threshold=authorization_threshold,
    )


def evaluate_candidate_policy_outputs(
    groups: Sequence[CandidateRankingGroup],
    scores_by_group: Sequence[np.ndarray],
    probabilities_by_group: Sequence[np.ndarray],
    *,
    authorization_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate externally supplied ranking scores and calibrated probabilities."""

    if not 0.0 < authorization_threshold < 1.0:
        raise ValueError("authorization threshold must be in (0, 1)")
    if len(groups) != len(scores_by_group) or len(groups) != len(
        probabilities_by_group
    ):
        raise ValueError("one score and probability vector are required per group")
    candidate_labels = []
    candidate_probabilities = []
    positive_states = 0
    all_negative_states = 0
    authorized_states = 0
    authorized_safe = 0
    authorized_unsafe = 0
    positive_state_success = 0
    all_negative_rejections = 0
    selections = []
    for group, score_values, probability_values in zip(
        groups, scores_by_group, probabilities_by_group
    ):
        probabilities = np.asarray(probability_values, dtype=np.float64)
        scores = np.asarray(score_values, dtype=np.float64)
        if probabilities.shape != group.labels.shape or scores.shape != group.labels.shape:
            raise ValueError("candidate policy outputs must align with group labels")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("candidate probabilities must be finite values in [0, 1]")
        if not np.all(np.isfinite(scores)):
            raise ValueError("candidate scores must be finite")
        selected = int(np.argmax(scores))
        probability = float(probabilities[selected])
        authorized = probability >= authorization_threshold
        safe = bool(group.labels[selected] == 1.0)
        has_positive = bool(np.any(group.labels == 1.0))
        positive_states += int(has_positive)
        all_negative_states += int(not has_positive)
        authorized_states += int(authorized)
        authorized_safe += int(authorized and safe)
        authorized_unsafe += int(authorized and not safe)
        positive_state_success += int(has_positive and authorized and safe)
        all_negative_rejections += int(not has_positive and not authorized)
        candidate_labels.extend(group.labels.tolist())
        candidate_probabilities.extend(probabilities.tolist())
        selections.append(
            {
                "sample_id": group.sample_id,
                "selected_candidate_id": group.candidate_ids[selected],
                "predicted_safe_probability": probability,
                "authorized": authorized,
                "selected_label": safe,
                "state_has_safe_candidate": has_positive,
            }
        )
    labels = np.asarray(candidate_labels, dtype=np.float64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    return {
        "state_count": len(groups),
        "candidate_count": len(labels),
        "positive_candidate_count": int(np.sum(labels == 1.0)),
        "negative_candidate_count": int(np.sum(labels == 0.0)),
        "positive_state_count": positive_states,
        "all_negative_state_count": all_negative_states,
        "authorized_state_count": authorized_states,
        "authorization_coverage": authorized_states / len(groups) if groups else None,
        "authorized_safe_count": authorized_safe,
        "authorized_unsafe_count": authorized_unsafe,
        "authorized_safe_precision": (
            authorized_safe / authorized_states if authorized_states else None
        ),
        "positive_state_safe_selection_rate": (
            positive_state_success / positive_states if positive_states else None
        ),
        "all_negative_rejection_rate": (
            all_negative_rejections / all_negative_states
            if all_negative_states
            else None
        ),
        "candidate_brier_score": (
            float(np.mean(np.square(probabilities - labels))) if len(labels) else None
        ),
        "authorization_threshold": float(authorization_threshold),
        "selections": selections,
    }


def candidate_ranker_deployment_gate(
    metrics: Mapping[str, Any] | None,
    *,
    minimum_states: int = 10,
    minimum_candidates: int = 50,
    minimum_positive_states: int = 3,
    minimum_all_negative_states: int = 2,
    minimum_authorized_safe_precision: float = 0.9,
    minimum_positive_state_safe_selection_rate: float = 0.8,
    minimum_all_negative_rejection_rate: float = 0.8,
) -> dict[str, Any]:
    """Fail-closed gate for giving a candidate ranker execution authority."""

    checks = []

    def add(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    values = metrics or {}
    add("calibrated_validation_metrics_available", metrics is not None)
    add(
        "minimum_validation_states",
        int(values.get("state_count", 0)) >= minimum_states,
    )
    add(
        "minimum_validation_candidates",
        int(values.get("candidate_count", 0)) >= minimum_candidates,
    )
    add(
        "minimum_positive_states",
        int(values.get("positive_state_count", 0)) >= minimum_positive_states,
    )
    add(
        "minimum_all_negative_states",
        int(values.get("all_negative_state_count", 0))
        >= minimum_all_negative_states,
    )
    precision = values.get("authorized_safe_precision")
    add(
        "authorized_safe_precision",
        precision is not None
        and float(precision) >= minimum_authorized_safe_precision,
        measured=precision,
        required=minimum_authorized_safe_precision,
    )
    positive_rate = values.get("positive_state_safe_selection_rate")
    add(
        "positive_state_safe_selection_rate",
        positive_rate is not None
        and float(positive_rate) >= minimum_positive_state_safe_selection_rate,
        measured=positive_rate,
        required=minimum_positive_state_safe_selection_rate,
    )
    rejection_rate = values.get("all_negative_rejection_rate")
    add(
        "all_negative_rejection_rate",
        rejection_rate is not None
        and float(rejection_rate) >= minimum_all_negative_rejection_rate,
        measured=rejection_rate,
        required=minimum_all_negative_rejection_rate,
    )
    passed = all(check["passed"] for check in checks)
    return {
        "status": "pass" if passed else "blocked",
        "control_mode": "candidate_authority" if passed else "shadow",
        "checks": checks,
        "policy": (
            "ranker may authorize candidates above its calibrated threshold"
            if passed
            else "ranker remains shadow-only; planner hard gates are unchanged"
        ),
    }


def evaluate_candidate_scores(
    groups: Sequence[CandidateRankingGroup],
    scores_by_group: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(groups) != len(scores_by_group):
        raise ValueError("one score vector is required for each candidate group")
    pair_correct = 0
    pair_total = 0
    top1_success = 0
    evaluated_groups = 0
    reciprocal_ranks: list[float] = []
    for group, raw_scores in zip(groups, scores_by_group):
        if not _has_positive_and_negative(group):
            continue
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.shape != group.labels.shape:
            raise ValueError("candidate scores must align with group labels")
        positive_scores = scores[group.labels == 1.0]
        negative_scores = scores[group.labels == 0.0]
        for positive_score in positive_scores:
            for negative_score in negative_scores:
                pair_correct += int(positive_score > negative_score)
                pair_total += 1
        order = np.argsort(scores)[::-1]
        ranked_labels = group.labels[order]
        top1_success += int(ranked_labels[0] == 1.0)
        first_positive = int(np.flatnonzero(ranked_labels == 1.0)[0])
        reciprocal_ranks.append(1.0 / float(first_positive + 1))
        evaluated_groups += 1
    return {
        "evaluated_group_count": evaluated_groups,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
        "top1_success_rate": top1_success / evaluated_groups if evaluated_groups else None,
        "mean_reciprocal_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        ),
    }


def load_candidate_samples(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    sample_names = {
        "automatic_candidate_sample.json",
        "open_vocab_candidate_probe_sample.json",
    }
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            sample_paths = sorted(
                sample_path
                for sample_name in sample_names
                for sample_path in path.rglob(sample_name)
            )
        else:
            sample_paths = [path]
        for sample_path in sample_paths:
            if _has_invalid_dataset_marker(sample_path):
                continue
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") in SUPPORTED_CANDIDATE_SAMPLE_SCHEMAS:
                samples.append(payload)
    return samples


def _has_invalid_dataset_marker(path: Path) -> bool:
    return any(
        (parent / "INVALID_DATASET.json").is_file()
        for parent in (path.parent, *path.parents)
    )


def _check_probability(checks: Mapping[str, Any], name: str) -> float:
    check = checks.get(name, {})
    return _number(check.get("probability"), default=0.5)


def _hashed_intent_vector(intent: Mapping[str, Any]) -> np.ndarray:
    fields: list[str] = []
    for key in (
        "target",
        "task_goal",
        "contact_pattern",
        "approach_relation",
        "closing_axis_relation",
        "grasp_depth_rule",
    ):
        value = intent.get(key)
        if value:
            fields.append(str(value))
    for key in ("preferred_roles", "avoided_roles", "natural_language_constraints"):
        value = intent.get(key, ())
        if isinstance(value, str):
            fields.append(value)
        else:
            fields.extend(str(item) for item in value)
    tokens = re.findall(r"\w+", " ".join(fields).casefold(), flags=re.UNICODE)
    result = np.zeros(INTENT_HASH_DIM, dtype=np.float64)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="little", signed=False)
        bucket = integer % INTENT_HASH_DIM
        sign = 1.0 if (integer >> 8) & 1 else -1.0
        result[bucket] += sign
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0.0 else result


def affordance_profile_feature_vector(value: Any) -> np.ndarray:
    profile = value if isinstance(value, Mapping) else {}
    result = np.zeros(len(AFFORDANCE_PROFILE_FEATURE_NAMES), dtype=np.float64)
    active: list[str] = []
    part_shape = str(profile.get("part_shape", "unknown"))
    if part_shape != "unknown":
        active.append(f"part_shape_{part_shape}")
    symmetry = str(profile.get("symmetry_class", "unknown"))
    if symmetry != "unknown":
        active.append(f"symmetry_{symmetry}")
    centering = str(profile.get("centering_tolerance", "unknown"))
    if centering != "unknown":
        active.append(f"centering_{centering}")
    vertical = str(profile.get("vertical_tolerance", "unknown"))
    if vertical != "unknown":
        active.append(f"vertical_{vertical}")
    index_by_name = {
        name: index for index, name in enumerate(AFFORDANCE_PROFILE_FEATURE_NAMES)
    }
    for name in active:
        if name in index_by_name:
            result[index_by_name[name]] = 1.0
    for name in (
        "avoid_ends",
        "requires_bilateral_contact",
        "requires_dual_arm",
    ):
        boolean = profile.get(name)
        if isinstance(boolean, bool):
            result[index_by_name[name]] = 1.0 if boolean else -1.0
    return result


def candidate_graph_evidence_feature_vector(
    value: Mapping[str, Any] | None,
) -> np.ndarray:
    """Encode only inference-visible state evidence shared by a candidate set.

    Candidate geometry explains which candidate to prefer.  These graph-level
    features explain whether the entire candidate set is trustworthy enough to
    authorize.  The extractor is deliberately allowlisted: simulator outcomes
    and ``training_only`` fields are never traversed.
    """

    context = value if isinstance(value, Mapping) else {}
    graph_value = context.get("candidate_graph")
    graph = graph_value if isinstance(graph_value, Mapping) else {}
    graph_available = bool(graph)

    fusion_value = graph.get("multiview_fusion")
    if not isinstance(fusion_value, Mapping):
        fusion_value = context.get("multiview_fusion")
    fusion = fusion_value if isinstance(fusion_value, Mapping) else {}
    fusion_available = bool(fusion)
    status = str(fusion.get("status", "unknown"))
    factors = _finite_number_list(fusion.get("view_contribution_factors"))
    selected = _text_list(fusion.get("selected_views"))
    suppressed = _text_list(fusion.get("suppressed_views"))
    view_count = len(factors) or len(set((*selected, *suppressed)))
    selected_count = len(selected)
    suppressed_count = len(suppressed)
    denominator = max(1, view_count)

    residual_value = fusion.get("point_residual_by_view_mm")
    if isinstance(residual_value, Mapping):
        residuals_m = [
            0.001 * number
            for number in _finite_number_list(residual_value.values())
        ]
    else:
        residuals_m = []
    mean_residual_m = float(np.mean(residuals_m)) if residuals_m else 0.0
    max_residual_m = float(np.max(residuals_m)) if residuals_m else 0.0
    cutoff_m = 0.001 * max(
        0.0, _number(fusion.get("consistency_cutoff_mm"))
    )
    if cutoff_m > 1e-9:
        mean_ratio = mean_residual_m / cutoff_m
        max_ratio = max_residual_m / cutoff_m
    else:
        mean_ratio = 0.0
        max_ratio = 0.0
    competitor = fusion.get("same_size_competitor_weight_ratio")
    competitor_available = competitor is not None and math.isfinite(
        _number(competitor)
    )

    assessment_value = graph.get("view_assessment")
    if not isinstance(assessment_value, Mapping):
        assessment_value = context.get("view_assessment")
    assessment = assessment_value if isinstance(assessment_value, Mapping) else {}
    candidate_set = _candidate_set_attributes_from_graph(graph)

    result = np.asarray(
        (
            float(graph_available),
            float(fusion_available),
            float(status == "dominant_consensus"),
            float(status == "ambiguous_no_dominant_consensus"),
            min(float(view_count) / 6.0, 1.0),
            float(selected_count) / float(denominator),
            float(suppressed_count) / float(denominator),
            mean_residual_m,
            max_residual_m,
            mean_ratio,
            max_ratio,
            float(competitor_available),
            _number(competitor) if competitor_available else 0.0,
            _number(candidate_set.get("candidate_identity_entropy")),
            float(bool(assessment.get("visual_evidence_sufficient", False))),
            _number(assessment.get("top_candidate_position_std_m")),
            _number(assessment.get("score_margin")),
        ),
        dtype=np.float64,
    )
    if result.shape != (len(GRAPH_EVIDENCE_FEATURE_NAMES),):
        raise AssertionError("graph evidence feature contract is inconsistent")
    if not np.all(np.isfinite(result)):
        raise ValueError("graph evidence features must be finite")
    return result


def _candidate_set_attributes_from_graph(
    graph: Mapping[str, Any],
) -> Mapping[str, Any]:
    for node in graph.get("nodes", ()):
        if (
            isinstance(node, Mapping)
            and node.get("node_type") == "grasp_candidate_set"
            and isinstance(node.get("attributes"), Mapping)
        ):
            return node["attributes"]
    return {}


def _finite_number_list(value: Any) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []
    result = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _text_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []
    return [str(item) for item in value]


def _matrix_trace(value: Any) -> float:
    if value is None:
        return 0.0
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("candidate covariance must be a 3x3 matrix")
    return float(np.trace(matrix))


def _vector_norm(value: Any) -> float:
    if value is None:
        return 0.0
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("candidate center delta must be a finite 3-vector")
    return float(np.linalg.norm(vector))


def _vector_component(value: Any, index: int) -> float:
    if value is None:
        return 0.0
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("candidate vector feature must be a finite 3-vector")
    return float(vector[index])


def _number(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    result = float(value)
    return result if math.isfinite(result) else float(default)


def _context_boolean(context: Mapping[str, Any], name: str) -> float:
    value = context.get(name)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return 0.0


def _has_positive_and_negative(group: CandidateRankingGroup) -> bool:
    return bool(np.any(group.labels == 1.0) and np.any(group.labels == 0.0))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(CANDIDATE_FEATURE_NAMES),):
        raise ValueError("candidate ranker vector shape mismatch")
    return result
