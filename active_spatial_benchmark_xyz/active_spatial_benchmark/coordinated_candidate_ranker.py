"""Leakage-controlled baseline ranker for coordinated grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .coordinated_candidate_dataset import validate_coordinated_candidate_sample


COORDINATED_RANKER_SCHEMA = "spatial.coordinated_candidate_pairwise_ranker.v1"
QUERY_HASH_DIM = 8
PAIR_BASE_FEATURE_NAMES = (
    "pairing_mode_nominal",
    "pairing_mode_both_same_factor",
    "pairing_mode_left_only_factor",
    "pairing_mode_right_only_factor",
    "center_separation_m",
    "center_separation_delta_m",
    "absolute_center_separation_delta_m",
    "center_midpoint_delta_norm_m",
    "approach_axis_dot",
    "approach_axis_dot_delta",
    "absolute_approach_axis_dot_delta",
    "closing_axis_dot",
    "closing_axis_dot_delta",
    "absolute_closing_axis_dot_delta",
    "opening_width_sum_m",
    "opening_width_sum_delta_m",
    "absolute_opening_width_sum_delta_m",
    "left_longitudinal_offset_m",
    "right_longitudinal_offset_m",
    "left_closing_offset_m",
    "right_closing_offset_m",
    "left_approach_offset_m",
    "right_approach_offset_m",
    "left_center_offset_norm_m",
    "right_center_offset_norm_m",
    "left_orientation_offset_deg",
    "right_orientation_offset_deg",
    "mean_center_offset_norm_m",
    "max_center_offset_norm_m",
    "center_offset_asymmetry_m",
    "mean_absolute_orientation_offset_deg",
    "max_absolute_orientation_offset_deg",
    "orientation_offset_asymmetry_deg",
    "member_perturbation_kind_matches",
    "member_scale_factor_matches",
    "position_covariance_trace_sum_m2",
    "orientation_covariance_trace_sum_rad2",
    "unknown_joint_constraint_fraction",
    "query_requires_dual_arm",
)
QUERY_INTERACTION_BASIS_NAMES = (
    "absolute_center_separation_delta_m",
    "center_midpoint_delta_norm_m",
    "absolute_approach_axis_dot_delta",
    "absolute_closing_axis_dot_delta",
    "left_center_offset_norm_m",
    "right_center_offset_norm_m",
    "mean_absolute_orientation_offset_deg",
    "orientation_offset_asymmetry_deg",
)
QUERY_INTERACTION_FEATURE_NAMES = tuple(
    f"query_hash_{bucket:02d}_x_{basis}"
    for bucket in range(QUERY_HASH_DIM)
    for basis in QUERY_INTERACTION_BASIS_NAMES
)
COORDINATED_CANDIDATE_FEATURE_NAMES = (
    PAIR_BASE_FEATURE_NAMES + QUERY_INTERACTION_FEATURE_NAMES
)
FORBIDDEN_INPUT_KEYS = {
    "joint_pregrasp_reachable",
    "joint_grasp_reachable",
    "synchronized_close_success",
    "task_success",
    "joint_execution_success",
    "joint_safe_success",
    "execution",
    "label",
}


@dataclass(frozen=True)
class CoordinatedCandidateRankingGroup:
    sample_id: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(
            COORDINATED_CANDIDATE_FEATURE_NAMES
        ):
            raise ValueError("coordinated candidate feature matrix has an unexpected shape")
        if labels.shape != (features.shape[0],):
            raise ValueError("coordinated candidate labels must align with features")
        if len(self.candidate_ids) != features.shape[0]:
            raise ValueError("coordinated candidate IDs must align with features")
        if not np.all(np.isin(labels, (0.0, 1.0))):
            raise ValueError("coordinated candidate labels must be binary")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)

    @property
    def has_pair(self) -> bool:
        return bool(np.any(self.labels == 1.0) and np.any(self.labels == 0.0))


def coordinated_candidate_feature_vector(
    intent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    nominal_candidate: Mapping[str, Any],
) -> np.ndarray:
    """Encode only task intent and pre-execution joint geometry."""

    leaked = sorted(_recursive_keys(candidate).intersection(FORBIDDEN_INPUT_KEYS))
    if leaked:
        raise ValueError("execution-only keys passed to coordinated encoder: " + ", ".join(leaked))
    if candidate.get("predicted_execution_success") is not None:
        raise ValueError("coordinated candidate cannot contain an execution prediction")
    pair = candidate.get("pair_geometry", {})
    nominal_pair = nominal_candidate.get("pair_geometry", {})
    mode = str(candidate.get("pairing_mode", "unknown"))
    member_by_arm = _members_by_arm(candidate)
    left = member_by_arm["left"].get("candidate", {})
    right = member_by_arm["right"].get("candidate", {})
    left_parameters = left.get("generation_parameters", {})
    right_parameters = right.get("generation_parameters", {})
    left_center_norm = _vector_norm(left_parameters.get("center_delta_world_m"))
    right_center_norm = _vector_norm(right_parameters.get("center_delta_world_m"))
    left_orientation = _number(left_parameters.get("orientation_offset_deg"))
    right_orientation = _number(right_parameters.get("orientation_offset_deg"))
    separation_delta = _number(pair.get("center_separation_m")) - _number(
        nominal_pair.get("center_separation_m")
    )
    midpoint_delta = _vector(pair.get("center_midpoint_world_m")) - _vector(
        nominal_pair.get("center_midpoint_world_m")
    )
    approach_delta = _number(pair.get("approach_axis_dot")) - _number(
        nominal_pair.get("approach_axis_dot")
    )
    closing_delta = _number(pair.get("closing_axis_dot")) - _number(
        nominal_pair.get("closing_axis_dot")
    )
    width_delta = _number(pair.get("opening_width_sum_m")) - _number(
        nominal_pair.get("opening_width_sum_m")
    )
    constraints = candidate.get("coordination_constraints", {})
    unknown_constraints = sum(
        str(value).startswith("unknown") for value in constraints.values()
    )
    constraint_count = max(1, len(constraints))
    left_scale = _number(left_parameters.get("scale_factor"), default=1.0)
    right_scale = _number(right_parameters.get("scale_factor"), default=1.0)
    base_values = np.asarray(
        (
            float(mode == "nominal"),
            float(mode == "both_same_factor"),
            float(mode == "left_only_factor"),
            float(mode == "right_only_factor"),
            _number(pair.get("center_separation_m")),
            separation_delta,
            abs(separation_delta),
            float(np.linalg.norm(midpoint_delta)),
            _number(pair.get("approach_axis_dot")),
            approach_delta,
            abs(approach_delta),
            _number(pair.get("closing_axis_dot")),
            closing_delta,
            abs(closing_delta),
            _number(pair.get("opening_width_sum_m")),
            width_delta,
            abs(width_delta),
            _number(left_parameters.get("longitudinal_offset_m")),
            _number(right_parameters.get("longitudinal_offset_m")),
            _number(left_parameters.get("closing_offset_m")),
            _number(right_parameters.get("closing_offset_m")),
            _number(left_parameters.get("approach_offset_m")),
            _number(right_parameters.get("approach_offset_m")),
            left_center_norm,
            right_center_norm,
            left_orientation,
            right_orientation,
            (left_center_norm + right_center_norm) / 2.0,
            max(left_center_norm, right_center_norm),
            abs(left_center_norm - right_center_norm),
            (abs(left_orientation) + abs(right_orientation)) / 2.0,
            max(abs(left_orientation), abs(right_orientation)),
            abs(abs(left_orientation) - abs(right_orientation)),
            float(
                left_parameters.get("perturbation_kind")
                == right_parameters.get("perturbation_kind")
            ),
            float(np.isclose(left_scale, right_scale)),
            _matrix_trace(left.get("position_covariance_m2"))
            + _matrix_trace(right.get("position_covariance_m2")),
            _matrix_trace(left.get("orientation_covariance_rad2"))
            + _matrix_trace(right.get("orientation_covariance_rad2")),
            float(unknown_constraints) / float(constraint_count),
            float(int(intent.get("required_arms", 0)) == 2),
        ),
        dtype=np.float64,
    )
    base_by_name = dict(zip(PAIR_BASE_FEATURE_NAMES, base_values))
    basis = np.asarray(
        [base_by_name[name] for name in QUERY_INTERACTION_BASIS_NAMES],
        dtype=np.float64,
    )
    interactions = np.outer(_hashed_query_vector(intent), basis).reshape(-1)
    result = np.concatenate((base_values, interactions))
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinated candidate features must be finite")
    return result


def ranking_group_from_coordinated_sample(
    sample: Mapping[str, Any],
    *,
    label_name: str = "joint_safe_success",
) -> CoordinatedCandidateRankingGroup | None:
    errors = validate_coordinated_candidate_sample(sample)
    if errors:
        raise ValueError("invalid coordinated candidate sample: " + "; ".join(errors))
    inference = sample["inference_visible"]
    training = sample["training_only"]
    nominal = training["oracle_joint_candidate_graph"].get("nominal_candidate")
    if not isinstance(nominal, Mapping):
        raise ValueError("coordinated ranker sample requires its nominal reference")
    candidate_ids = []
    features = []
    labels = []
    for row in training.get("candidate_labels", ()):
        label = row.get("execution", {}).get(label_name)
        if label is None:
            continue
        candidate = row.get("candidate", {})
        candidate_ids.append(str(candidate.get("id", "")))
        features.append(
            coordinated_candidate_feature_vector(
                inference.get("intent", {}), candidate, nominal
            )
        )
        labels.append(float(bool(label)))
    if not features:
        return None
    return CoordinatedCandidateRankingGroup(
        sample_id=str(sample.get("sample_id", "unknown")),
        candidate_ids=tuple(candidate_ids),
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.float64),
    )


class PairwiseLinearCoordinatedRanker:
    def __init__(self) -> None:
        size = len(COORDINATED_CANDIDATE_FEATURE_NAMES)
        self.feature_mean = np.zeros(size, dtype=np.float64)
        self.feature_scale = np.ones(size, dtype=np.float64)
        self.weights = np.zeros(size, dtype=np.float64)
        self.fitted = False

    def fit(
        self,
        groups: Sequence[CoordinatedCandidateRankingGroup],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
    ) -> dict[str, Any]:
        if epochs < 1 or learning_rate <= 0.0 or l2 < 0.0:
            raise ValueError("invalid coordinated ranker optimization settings")
        eligible = [group for group in groups if group.has_pair]
        if not eligible:
            raise ValueError("coordinated pairwise training needs a positive/negative state")
        all_features = np.concatenate([group.features for group in eligible], axis=0)
        self.feature_mean = all_features.mean(axis=0)
        scale = all_features.std(axis=0)
        self.feature_scale = np.where(scale > 1e-8, scale, 1.0)
        pair_features = []
        pair_labels = []
        for group in eligible:
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
        for _ in range(epochs):
            probability = _sigmoid(x @ self.weights)
            gradient = x.T @ (probability - y) / float(len(y)) + l2 * self.weights
            self.weights -= learning_rate * gradient
        self.fitted = True
        return {
            **evaluate_coordinated_ranker(self, eligible),
            "pair_count": len(y) // 2,
            "training_group_count": len(eligible),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
        }

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("coordinated ranker has not been fitted")
        return self._standardize(np.asarray(features, dtype=np.float64)) @ self.weights

    def as_dict(self, *, training_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("coordinated ranker has not been fitted")
        return {
            "schema_version": COORDINATED_RANKER_SCHEMA,
            "model_type": "standardized_pairwise_linear_coordinated_ranker",
            "feature_names": list(COORDINATED_CANDIDATE_FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "training_metadata": dict(training_metadata or {}),
            "input_contract": {
                "uses_execution_labels_as_features": False,
                "forbidden_feature_keys": sorted(FORBIDDEN_INPUT_KEYS),
                "candidate_source_requirement": "inference-visible learned joint graph at deployment",
                "candidate_set_reference_required": True,
                "query_conditioning": "deterministic_hashed_query_interactions_smoke_baseline",
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairwiseLinearCoordinatedRanker":
        if value.get("schema_version") != COORDINATED_RANKER_SCHEMA:
            raise ValueError("unsupported coordinated candidate ranker schema")
        if tuple(value.get("feature_names", ())) != COORDINATED_CANDIDATE_FEATURE_NAMES:
            raise ValueError("coordinated candidate ranker feature schema mismatch")
        model = cls()
        model.feature_mean = _ranker_vector(value.get("feature_mean"))
        model.feature_scale = _ranker_vector(value.get("feature_scale"))
        model.weights = _ranker_vector(value.get("weights"))
        if np.any(model.feature_scale <= 0.0):
            raise ValueError("coordinated candidate ranker scales must be positive")
        model.fitted = True
        return model

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.feature_mean) / self.feature_scale


def evaluate_coordinated_ranker(
    model: PairwiseLinearCoordinatedRanker,
    groups: Sequence[CoordinatedCandidateRankingGroup],
) -> dict[str, Any]:
    pair_correct = 0
    pair_total = 0
    top1_success = 0
    evaluated = 0
    for group in groups:
        if not group.has_pair:
            continue
        scores = model.score(group.features)
        comparisons = scores[group.labels == 1.0][:, None] - scores[
            group.labels == 0.0
        ][None, :]
        pair_correct += int(np.sum(comparisons > 0.0))
        pair_total += int(comparisons.size)
        top1_success += int(group.labels[int(np.argmax(scores))] == 1.0)
        evaluated += 1
    return {
        "evaluated_group_count": evaluated,
        "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
        "top1_success_rate": top1_success / evaluated if evaluated else None,
    }


def _members_by_arm(candidate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    members = candidate.get("members", ())
    result = {
        str(member.get("arm")): member
        for member in members
        if isinstance(member, Mapping)
    }
    if set(result) != {"left", "right"}:
        raise ValueError("coordinated candidate requires left and right members")
    return result


def _hashed_query_vector(intent: Mapping[str, Any]) -> np.ndarray:
    fields = []
    for key in (
        "target",
        "task_goal",
        "task_stage",
        "post_grasp_goal",
        "coordination_group",
        "contact_pattern",
    ):
        if intent.get(key):
            fields.append(str(intent[key]))
    for key in ("preferred_roles", "natural_language_constraints"):
        value = intent.get(key, ())
        fields.extend([value] if isinstance(value, str) else [str(item) for item in value])
    result = np.zeros(QUERY_HASH_DIM, dtype=np.float64)
    for token in re.findall(r"\w+", " ".join(fields).casefold(), flags=re.UNICODE):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="little", signed=False)
        result[integer % QUERY_HASH_DIM] += 1.0 if (integer >> 8) & 1 else -1.0
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 0.0 else result


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = set(str(key) for key in value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def _number(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    result = float(value)
    return result if math.isfinite(result) else float(default)


def _vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("coordinated pair geometry requires finite 3-vectors")
    return result


def _vector_norm(value: Any) -> float:
    if value is None:
        return 0.0
    return float(np.linalg.norm(_vector(value)))


def _matrix_trace(value: Any) -> float:
    if value is None:
        return 0.0
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("coordinated candidate covariance must be finite 3x3")
    return float(np.trace(result))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _ranker_vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(COORDINATED_CANDIDATE_FEATURE_NAMES),):
        raise ValueError("coordinated candidate ranker vector shape mismatch")
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinated candidate ranker vectors must be finite")
    return result
