"""Pairwise baseline for relation-information-gain view selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .expert_event_view_dataset import (
    RELATION_NAMES,
    validate_expert_event_view_ranking_sample,
)
from .realized_event_view_dataset import (
    REALIZED_EVENT_VIEW_SAMPLE_SCHEMA,
    validate_realized_event_view_sample,
)
from .active_belief import (
    ANALYTIC_FISHER_FEATURE_NAMES,
    ViewSensorModel,
    analytic_view_information_features,
    depth_dominates_lateral_noise,
    is_nearly_isotropic_noise,
    measurement_noise_anisotropy_ratio,
)
from .view_pose_features import (
    POSE_FEATURE_NAMES,
    camera_position_from_standoff,
    relative_view_geometry,
    relative_view_geometry_from_observability,
    uncertainty_frame,
)


EVENT_VIEW_RANKER_SCHEMA = "spatial.expert_event_view_pairwise_ranker.v2"
LEGACY_EVENT_VIEW_RANKER_SCHEMA = "spatial.expert_event_view_pairwise_ranker.v1"
VIEW_NAMES = (
    "topdown",
    "side",
    "front_side_45",
    "side_top_45",
    "oblique_45",
)
QUERY_HASH_DIM = 8
FEATURE_MODES = (
    "full",
    "task_view_conditioned",
    "no_view_onehot",
    "shuffled_view_names",
    "analytic_only",
    "relation_conditioned",
    "pose_only",
    "pose_conditioned",
    "analytic_fisher",
)
UNCERTAINTY_FEATURE_NAMES = tuple(
    f"current_uncertainty_{name}" for name in RELATION_NAMES
)
LEGACY_GEOMETRY_FEATURE_NAMES = (
    "candidate_closing_axis_observability",
    "candidate_approach_axis_observability",
    "candidate_view_direction_x",
    "candidate_view_direction_y",
    "candidate_view_direction_z",
    "angular_diversity",
    "move_cost",
)
LEGACY_VIEW_IDENTITY_FEATURE_NAMES = tuple(f"view_{name}" for name in VIEW_NAMES)
POSE_GEOMETRY_FEATURE_NAMES = (
    "candidate_closing_axis_observability",
    "candidate_approach_axis_observability",
    *POSE_FEATURE_NAMES,
    "move_cost",
)
FISHER_GEOMETRY_FEATURE_NAMES = (
    *POSE_GEOMETRY_FEATURE_NAMES,
    *ANALYTIC_FISHER_FEATURE_NAMES,
)
LEGACY_QUERY_INTERACTION_BASIS_NAMES = (
    "candidate_closing_axis_observability",
    "candidate_approach_axis_observability",
    "angular_diversity",
    "move_cost",
    "current_uncertainty_opening_width",
    "current_uncertainty_approach_axis",
)
POSE_QUERY_INTERACTION_BASIS_NAMES = (
    "candidate_closing_axis_observability",
    "candidate_approach_axis_observability",
    "history_parallax_mean",
    "move_cost",
    "predicted_target_visibility",
    "current_uncertainty_opening_width",
    "current_uncertainty_approach_axis",
)
ANALYTIC_GEOMETRY_FEATURE_NAMES = (
    "candidate_closing_axis_observability",
    "candidate_approach_axis_observability",
    "angular_diversity",
    "move_cost",
)
ANALYTIC_FEATURE_NAMES = (
    *UNCERTAINTY_FEATURE_NAMES,
    *ANALYTIC_GEOMETRY_FEATURE_NAMES,
)
RELATION_CONDITIONED_FEATURE_NAMES = tuple(
    f"{uncertainty}_x_{geometry}"
    for uncertainty in UNCERTAINTY_FEATURE_NAMES
    for geometry in ANALYTIC_GEOMETRY_FEATURE_NAMES
)
FORBIDDEN_INPUT_KEYS = {
    "target",
    "oracle_utility",
    "oracle_information_gain_nats",
    "oracle_relation_evidence",
    "posterior_relation_uncertainty",
    "relation_information_gain_nats",
}


@dataclass(frozen=True)
class EventViewFeatureConfig:
    """Serializable feature contract for one Step-0 ablation."""

    mode: str = "pose_conditioned"
    view_shuffle_seed: int = 0
    shuffle_phase: str = "evaluation"

    def __post_init__(self) -> None:
        if self.mode not in FEATURE_MODES:
            raise ValueError(f"unsupported event-view feature mode: {self.mode}")
        if self.view_shuffle_seed < 0:
            raise ValueError("view shuffle seed must be non-negative")
        if self.shuffle_phase not in {"training", "evaluation"}:
            raise ValueError("shuffle phase must be 'training' or 'evaluation'")

    @property
    def uses_view_identity(self) -> bool:
        return self.mode in {
            "full",
            "task_view_conditioned",
            "shuffled_view_names",
        }

    @property
    def supports_unseen_view_labels(self) -> bool:
        return not self.uses_view_identity

    @property
    def query_interaction_basis_names(self) -> tuple[str, ...]:
        if self.mode in {"analytic_only", "relation_conditioned", "pose_only"}:
            return ()
        if self.mode in {"pose_conditioned", "analytic_fisher"}:
            return POSE_QUERY_INTERACTION_BASIS_NAMES
        if self.mode == "task_view_conditioned":
            return (
                *LEGACY_QUERY_INTERACTION_BASIS_NAMES,
                *LEGACY_VIEW_IDENTITY_FEATURE_NAMES,
            )
        return LEGACY_QUERY_INTERACTION_BASIS_NAMES

    @property
    def base_feature_names(self) -> tuple[str, ...]:
        if self.mode == "analytic_only":
            return ANALYTIC_FEATURE_NAMES
        if self.mode == "relation_conditioned":
            return ANALYTIC_FEATURE_NAMES + RELATION_CONDITIONED_FEATURE_NAMES
        if self.mode in {"pose_only", "pose_conditioned"}:
            return UNCERTAINTY_FEATURE_NAMES + POSE_GEOMETRY_FEATURE_NAMES
        if self.mode == "analytic_fisher":
            return UNCERTAINTY_FEATURE_NAMES + FISHER_GEOMETRY_FEATURE_NAMES
        result = UNCERTAINTY_FEATURE_NAMES + LEGACY_GEOMETRY_FEATURE_NAMES
        if self.uses_view_identity:
            result += LEGACY_VIEW_IDENTITY_FEATURE_NAMES
        return result

    @property
    def feature_names(self) -> tuple[str, ...]:
        interactions = tuple(
            f"query_hash_{bucket:02d}_x_{basis}"
            for bucket in range(QUERY_HASH_DIM)
            for basis in self.query_interaction_basis_names
        )
        return self.base_feature_names + interactions

    @property
    def applies_view_shuffle(self) -> bool:
        """Whether *this* config permutes names when encoding a sample.

        ``shuffled_view_names`` is an evaluation-time intervention. A single
        global bijection applied to both training and evaluation is only a
        permutation of one-hot columns: the optimizer recovers an exactly
        equivalent weight vector, so the ablation measures nothing. The
        permutation must therefore be absent while fitting and present while
        scoring, which is what :meth:`for_training` and :meth:`for_evaluation`
        express.
        """

        return self.mode == "shuffled_view_names" and self.shuffle_phase == "evaluation"

    def for_training(self) -> "EventViewFeatureConfig":
        """Encoding contract used to fit weights: never shuffled."""

        if self.mode != "shuffled_view_names":
            return self
        return replace(self, shuffle_phase="training")

    def for_evaluation(self) -> "EventViewFeatureConfig":
        """Encoding contract used to score held-out samples: shuffled."""

        if self.mode != "shuffled_view_names":
            return self
        return replace(self, shuffle_phase="evaluation")

    @property
    def shuffled_view_mapping(self) -> dict[str, str]:
        if not self.applies_view_shuffle:
            return {name: name for name in VIEW_NAMES}
        rng = np.random.default_rng(self.view_shuffle_seed)
        shuffled = list(VIEW_NAMES)
        rng.shuffle(shuffled)
        return dict(zip(VIEW_NAMES, shuffled))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "view_shuffle_seed": self.view_shuffle_seed,
            "shuffle_phase": self.shuffle_phase,
            "training_view_name_mapping": self.for_training().shuffled_view_mapping,
            "evaluation_view_name_mapping": (
                self.for_evaluation().shuffled_view_mapping
            ),
            "view_name_mapping": self.shuffled_view_mapping,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventViewFeatureConfig":
        result = cls(
            mode=str(value.get("mode", "")),
            view_shuffle_seed=int(value.get("view_shuffle_seed", 0)),
            shuffle_phase=str(value.get("shuffle_phase", "evaluation")),
        )
        expected_mapping = result.shuffled_view_mapping
        stored_mapping = value.get("view_name_mapping", expected_mapping)
        if dict(stored_mapping) != expected_mapping:
            raise ValueError("event-view shuffled mapping does not match its seed")
        return result


DEFAULT_FEATURE_CONFIG = EventViewFeatureConfig()
EVENT_VIEW_FEATURE_NAMES = DEFAULT_FEATURE_CONFIG.feature_names


@dataclass(frozen=True)
class EventViewRankingGroup:
    sample_id: str
    task: str
    seed: int
    views: tuple[str, ...]
    features: np.ndarray
    target_utility: np.ndarray
    feature_names: tuple[str, ...] = EVENT_VIEW_FEATURE_NAMES
    encoding_phase: str = "evaluation"

    def __post_init__(self) -> None:
        if self.encoding_phase not in {"training", "evaluation"}:
            raise ValueError("encoding phase must be 'training' or 'evaluation'")
        features = np.asarray(self.features, dtype=np.float64)
        utility = np.asarray(self.target_utility, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("event-view feature matrix has an unexpected shape")
        if utility.shape != (features.shape[0],) or not np.all(np.isfinite(utility)):
            raise ValueError("event-view targets must align and be finite")
        if len(self.views) != features.shape[0]:
            raise ValueError("event-view names must align with features")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target_utility", utility)

    @property
    def has_pair(self) -> bool:
        return bool(
            len(self.target_utility) >= 2
            and float(np.max(self.target_utility) - np.min(self.target_utility)) > 1e-9
        )


def event_view_feature_vector(
    intent: Mapping[str, Any],
    view: str,
    input_features: Mapping[str, Any],
    *,
    feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> np.ndarray:
    leaked = sorted(_recursive_keys(input_features).intersection(FORBIDDEN_INPUT_KEYS))
    if leaked:
        raise ValueError(
            "oracle targets passed to event-view encoder: " + ", ".join(leaked)
        )
    uncertainty = input_features.get("current_relation_uncertainty", {})
    by_name = {
        f"current_uncertainty_{name}": _number(uncertainty.get(name), default=1.0)
        for name in RELATION_NAMES
    }
    by_name.update(
        {
            "candidate_closing_axis_observability": _number(
                input_features.get("candidate_closing_axis_observability")
            ),
            "candidate_approach_axis_observability": _number(
                input_features.get("candidate_approach_axis_observability")
            ),
            "angular_diversity": _number(input_features.get("angular_diversity")),
            "move_cost": _number(input_features.get("move_cost")),
        }
    )
    if feature_config.mode == "relation_conditioned":
        by_name.update(
            {
                f"{uncertainty}_x_{geometry}": by_name[uncertainty]
                * by_name[geometry]
                for uncertainty in UNCERTAINTY_FEATURE_NAMES
                for geometry in ANALYTIC_GEOMETRY_FEATURE_NAMES
            }
        )
    if feature_config.mode in {"pose_only", "pose_conditioned", "analytic_fisher"}:
        pose_features = input_features.get("relative_pose_features")
        if not isinstance(pose_features, Mapping):
            pose_features = relative_view_geometry_from_observability(input_features)
        by_name.update(
            {name: _number(pose_features.get(name)) for name in POSE_FEATURE_NAMES}
        )
        if feature_config.mode == "analytic_fisher":
            fisher_features = input_features.get("analytic_fisher_features")
            if not isinstance(fisher_features, Mapping):
                raise ValueError(
                    "analytic_fisher mode requires precomputed Fisher features; "
                    "the caller must supply a calibrated sensor model"
                )
            by_name.update(
                {
                    name: _number(fisher_features.get(name))
                    for name in ANALYTIC_FISHER_FEATURE_NAMES
                }
            )
    elif feature_config.mode not in {"analytic_only", "relation_conditioned"}:
        direction = np.asarray(
            input_features.get("candidate_view_direction_world"), dtype=np.float64
        )
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("candidate view direction must be a finite 3-vector")
        by_name.update(
            {
                "candidate_view_direction_x": float(direction[0]),
                "candidate_view_direction_y": float(direction[1]),
                "candidate_view_direction_z": float(direction[2]),
            }
        )
        if feature_config.uses_view_identity:
            encoded_view = feature_config.shuffled_view_mapping.get(view, view)
            by_name.update(
                {f"view_{name}": float(encoded_view == name) for name in VIEW_NAMES}
            )
    base = np.asarray(
        [by_name[name] for name in feature_config.base_feature_names],
        dtype=np.float64,
    )
    basis = np.asarray(
        [by_name[name] for name in feature_config.query_interaction_basis_names],
        dtype=np.float64,
    )
    interactions = (
        np.outer(_hashed_query_vector(intent), basis).reshape(-1)
        if basis.size
        else np.empty(0, dtype=np.float64)
    )
    result = np.concatenate((base, interactions))
    if result.shape != (len(feature_config.feature_names),) or not np.all(
        np.isfinite(result)
    ):
        raise ValueError("event-view features must be finite")
    return result


@dataclass(frozen=True)
class AnalyticFisherContext:
    """Calibration needed to derive L1 Fisher features for stored samples.

    Stage-0 event samples predate the analytic backbone and store no node
    covariance, so ``analytic_fisher`` training must be given an explicit,
    recorded calibration. The prior covariance here is a declared calibration
    constant shared by the group, not per-sample simulator truth; per-sample
    covariance should replace it once the dataset carries it.
    """

    prior_covariance_m2: tuple[tuple[float, ...], ...]
    sensor_model: ViewSensorModel
    focal_length_px: float
    camera_standoff_m: float
    closing_axis_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_covariance_m2": [list(row) for row in self.prior_covariance_m2],
            "closing_axis_world": list(self.closing_axis_world),
            "approach_axis_world": list(self.approach_axis_world),
            "task_frame_source": "declared_group_calibration_constant",
            "keypoint_std_px": self.sensor_model.keypoint_std_px,
            "depth_std_m": self.sensor_model.depth_std_m,
            "focal_length_px": self.focal_length_px,
            "camera_standoff_m": self.camera_standoff_m,
            "prior_covariance_source": "declared_group_calibration_constant",
            "per_sample_state_override_supported": True,
            "depth_dominates_lateral_noise": depth_dominates_lateral_noise(
                sensor_model=self.sensor_model,
                range_m=self.camera_standoff_m,
                focal_length_px=self.focal_length_px,
            ),
            "measurement_noise_anisotropy_ratio": measurement_noise_anisotropy_ratio(
                sensor_model=self.sensor_model,
                range_m=self.camera_standoff_m,
                focal_length_px=self.focal_length_px,
            ),
            "directional_signal_available": not is_nearly_isotropic_noise(
                sensor_model=self.sensor_model,
                range_m=self.camera_standoff_m,
                focal_length_px=self.focal_length_px,
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalyticFisherContext":
        """Restore the calibration a checkpoint was trained under.

        Deployment must reuse the training calibration rather than a fresh
        guess: fitted weights are only meaningful in the units their features
        were built in. Any missing field raises instead of defaulting.
        """

        required = (
            "prior_covariance_m2",
            "keypoint_std_px",
            "depth_std_m",
            "focal_length_px",
            "camera_standoff_m",
        )
        missing = [name for name in required if value.get(name) is None]
        if missing:
            raise ValueError(
                "analytic Fisher calibration is incomplete: " + ", ".join(missing)
            )
        return cls(
            prior_covariance_m2=tuple(
                tuple(float(entry) for entry in row)
                for row in value["prior_covariance_m2"]
            ),
            sensor_model=ViewSensorModel(
                keypoint_std_px=float(value["keypoint_std_px"]),
                depth_std_m=float(value["depth_std_m"]),
            ),
            focal_length_px=float(value["focal_length_px"]),
            camera_standoff_m=float(value["camera_standoff_m"]),
            closing_axis_world=tuple(
                float(entry)
                for entry in value.get("closing_axis_world", (1.0, 0.0, 0.0))
            ),
            approach_axis_world=tuple(
                float(entry)
                for entry in value.get("approach_axis_world", (0.0, 0.0, -1.0))
            ),
        )


def _fisher_features_for_row(
    row: Mapping[str, Any],
    context: AnalyticFisherContext,
) -> dict[str, float]:
    input_features = row["input_features"]
    closing = _unit_vector(
        input_features.get("closing_axis_world", context.closing_axis_world),
        "closing axis",
    )
    approach = _unit_vector(
        input_features.get("approach_axis_world", context.approach_axis_world),
        "approach axis",
    )
    prior_covariance = np.asarray(
        input_features.get("prior_covariance_m2", context.prior_covariance_m2),
        dtype=np.float64,
    )
    if prior_covariance.shape != (3, 3) or not np.all(
        np.isfinite(prior_covariance)
    ):
        raise ValueError("analytic Fisher prior covariance must be a finite 3x3 matrix")
    candidate_range_m = _number(
        input_features.get("candidate_range_m"),
        default=context.camera_standoff_m,
    )
    if candidate_range_m <= 0.0:
        raise ValueError("analytic Fisher candidate range must be positive")
    focal_length_px = _number(
        input_features.get("focal_length_px"), default=context.focal_length_px
    )
    if focal_length_px <= 0.0:
        raise ValueError("analytic Fisher focal length must be positive")
    return analytic_view_information_features(
        prior_covariance=prior_covariance,
        candidate_view_direction_world=input_features["candidate_view_direction_world"],
        task_axes_world=uncertainty_frame(
            closing_axis_world=closing,
            approach_axis_world=approach,
        ),
        range_m=candidate_range_m,
        focal_length_px=focal_length_px,
        sensor_model=context.sensor_model,
    )


def ranking_group_from_event_view_sample(
    sample: Mapping[str, Any],
    *,
    feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
    analytic_fisher_context: AnalyticFisherContext | None = None,
    phase: str = "evaluation",
) -> EventViewRankingGroup:
    """Encode one sample into a ranking group.

    ``phase`` selects the encoding contract. Weights must be fitted on
    ``"training"`` groups and measured on ``"evaluation"`` groups; for the
    ``shuffled_view_names`` ablation those two encodings differ, which is the
    only way the intervention can change a score.
    """

    if phase not in {"training", "evaluation"}:
        raise ValueError("phase must be 'training' or 'evaluation'")
    feature_config = (
        feature_config.for_training()
        if phase == "training"
        else feature_config.for_evaluation()
    )
    errors = (
        validate_realized_event_view_sample(sample)
        if sample.get("schema_version") == REALIZED_EVENT_VIEW_SAMPLE_SCHEMA
        else validate_expert_event_view_ranking_sample(sample)
    )
    if errors:
        raise ValueError("invalid event-view sample: " + "; ".join(errors))
    intent = sample["inference_visible"]["intent"]
    labels = sample["training_only"]["view_labels"]
    views = tuple(str(row["view"]) for row in labels)
    if feature_config.mode == "analytic_fisher" and analytic_fisher_context is None:
        raise ValueError(
            "analytic_fisher training requires an explicit AnalyticFisherContext"
        )
    rows = []
    for row in labels:
        input_features = dict(row["input_features"])
        if analytic_fisher_context is not None:
            input_features["analytic_fisher_features"] = _fisher_features_for_row(
                row, analytic_fisher_context
            )
        rows.append((str(row["view"]), input_features))
    features = np.stack(
        [
            event_view_feature_vector(
                intent,
                view,
                input_features,
                feature_config=feature_config,
            )
            for view, input_features in rows
        ]
    )
    target = np.asarray(
        [float(row["target"]["oracle_utility"]) for row in labels],
        dtype=np.float64,
    )
    return EventViewRankingGroup(
        sample_id=str(sample.get("sample_id", "unknown")),
        task=str(sample.get("task", "unknown")),
        seed=int(sample.get("seed", 0)),
        views=views,
        features=features,
        target_utility=target,
        feature_names=feature_config.feature_names,
        encoding_phase=phase,
    )


class PairwiseLinearEventViewRanker:
    def __init__(
        self,
        *,
        feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
    ) -> None:
        self.feature_config = feature_config
        self.feature_names = feature_config.feature_names
        # Calibration the checkpoint was trained under. Populated by from_dict;
        # None for a freshly constructed model.
        self.analytic_fisher_context: AnalyticFisherContext | None = None
        size = len(self.feature_names)
        self.feature_mean = np.zeros(size, dtype=np.float64)
        self.feature_scale = np.ones(size, dtype=np.float64)
        self.weights = np.zeros(size, dtype=np.float64)
        self.fitted = False

    def fit(
        self,
        groups: Sequence[EventViewRankingGroup],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        pair_weight_power: float = 0.0,
    ) -> dict[str, Any]:
        if epochs < 1 or learning_rate <= 0.0 or l2 < 0.0 or pair_weight_power < 0.0:
            raise ValueError("invalid event-view ranker optimization settings")
        eligible = [group for group in groups if group.has_pair]
        if not eligible:
            raise ValueError(
                "event-view pairwise training needs distinct target utilities"
            )
        if any(group.feature_names != self.feature_names for group in eligible):
            raise ValueError(
                "event-view groups and model use different feature contracts"
            )
        if self.feature_config.mode == "shuffled_view_names" and any(
            group.encoding_phase != "training" for group in eligible
        ):
            raise ValueError(
                "shuffled_view_names weights must be fitted on training-phase "
                "encodings; fitting on the shuffled encoding makes the ablation "
                "a no-op column permutation"
            )
        all_features = np.concatenate([group.features for group in eligible], axis=0)
        self.feature_mean = all_features.mean(axis=0)
        scale = all_features.std(axis=0)
        self.feature_scale = np.where(scale > 1e-8, scale, 1.0)
        differences = []
        labels = []
        pair_weights = []
        for group in eligible:
            standardized = self._standardize(group.features)
            for left in range(len(group.views)):
                for right in range(left + 1, len(group.views)):
                    utility_delta = (
                        group.target_utility[left] - group.target_utility[right]
                    )
                    if abs(float(utility_delta)) <= 1e-9:
                        continue
                    difference = standardized[left] - standardized[right]
                    label = float(utility_delta > 0.0)
                    pair_weight = abs(float(utility_delta)) ** pair_weight_power
                    differences.extend((difference, -difference))
                    labels.extend((label, 1.0 - label))
                    pair_weights.extend((pair_weight, pair_weight))
        if not differences:
            raise ValueError("event-view pairwise training found no strict ordering")
        x = np.stack(differences)
        y = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(pair_weights, dtype=np.float64)
        weights /= float(np.mean(weights))
        for _ in range(epochs):
            probability = _sigmoid(x @ self.weights)
            gradient = (
                x.T @ ((probability - y) * weights) / float(np.sum(weights))
                + l2 * self.weights
            )
            self.weights -= learning_rate * gradient
        self.fitted = True
        return {
            **evaluate_event_view_ranker(
                self, eligible, require_evaluation_phase=False
            ),
            "pair_count": len(y) // 2,
            "training_group_count": len(eligible),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "pair_weight_power": float(pair_weight_power),
        }

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("event-view ranker has not been fitted")
        return self._standardize(np.asarray(features, dtype=np.float64)) @ self.weights

    def as_dict(
        self, *, training_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("event-view ranker has not been fitted")
        return {
            "schema_version": EVENT_VIEW_RANKER_SCHEMA,
            "model_type": "standardized_pairwise_linear_event_view_ranker",
            "feature_config": self.feature_config.as_dict(),
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "training_metadata": dict(training_metadata or {}),
            "input_contract": {
                "uses_oracle_view_targets_as_features": False,
                "forbidden_feature_keys": sorted(FORBIDDEN_INPUT_KEYS),
                "current_uncertainty_source_at_deployment": "learned_rgbd_spatial_graph",
                "candidate_geometry_source": "known_camera_pose_and_learned_grasp_axes",
                "query_conditioning": "deterministic_hashed_query_interactions_smoke_baseline",
                "feature_mode": self.feature_config.mode,
                "uses_view_identity": self.feature_config.uses_view_identity,
                "supports_unseen_view_labels": (
                    self.feature_config.supports_unseen_view_labels
                ),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairwiseLinearEventViewRanker":
        if value.get("schema_version") == LEGACY_EVENT_VIEW_RANKER_SCHEMA:
            raise ValueError(
                "legacy v1 event-view checkpoints are not compatible with the v2 "
                "feature contract; retrain with an explicit feature mode"
            )
        if value.get("schema_version") != EVENT_VIEW_RANKER_SCHEMA:
            raise ValueError("unsupported event-view ranker schema")
        config_value = value.get("feature_config")
        if not isinstance(config_value, Mapping):
            raise ValueError("event-view ranker checkpoint has no feature_config")
        feature_config = EventViewFeatureConfig.from_dict(config_value)
        if tuple(value.get("feature_names", ())) != feature_config.feature_names:
            raise ValueError("event-view ranker feature schema mismatch")
        model = cls(feature_config=feature_config)
        model.feature_mean = _ranker_vector(
            value.get("feature_mean"), len(model.feature_names)
        )
        model.feature_scale = _ranker_vector(
            value.get("feature_scale"), len(model.feature_names)
        )
        model.weights = _ranker_vector(value.get("weights"), len(model.feature_names))
        if np.any(model.feature_scale <= 0.0):
            raise ValueError("event-view ranker scales must be positive")
        metadata = value.get("training_metadata")
        calibration = (
            metadata.get("analytic_fisher_context")
            if isinstance(metadata, Mapping)
            else None
        )
        if isinstance(calibration, Mapping):
            model.analytic_fisher_context = AnalyticFisherContext.from_dict(calibration)
        elif feature_config.mode == "analytic_fisher":
            # Refuse at load rather than crash at the first runtime call: an
            # analytic_fisher checkpoint without its calibration cannot be
            # scored, and a silent fallback would deploy the wrong units.
            raise ValueError(
                "analytic_fisher checkpoint has no recorded analytic_fisher_context; "
                "retrain with explicit sensor calibration before deployment"
            )
        model.fitted = True
        return model

    def _standardize(self, value: np.ndarray) -> np.ndarray:
        return (value - self.feature_mean) / self.feature_scale


def evaluate_event_view_ranker(
    model: PairwiseLinearEventViewRanker,
    groups: Sequence[EventViewRankingGroup],
    *,
    require_evaluation_phase: bool = True,
) -> dict[str, Any]:
    if (
        require_evaluation_phase
        and model.feature_config.mode == "shuffled_view_names"
        and any(group.encoding_phase != "evaluation" for group in groups)
    ):
        raise ValueError(
            "shuffled_view_names metrics must be measured on evaluation-phase "
            "encodings, otherwise the permutation is never applied"
        )
    correct = 0
    pair_count = 0
    top1 = 0
    regrets = []
    for group in groups:
        if not group.has_pair:
            continue
        scores = model.score(group.features)
        for left in range(len(group.views)):
            for right in range(left + 1, len(group.views)):
                target_delta = group.target_utility[left] - group.target_utility[right]
                if abs(float(target_delta)) <= 1e-9:
                    continue
                score_delta = scores[left] - scores[right]
                correct += int(np.sign(score_delta) == np.sign(target_delta))
                pair_count += 1
        selected = int(np.argmax(scores))
        oracle = int(np.argmax(group.target_utility))
        top1 += int(selected == oracle)
        regrets.append(
            float(group.target_utility[oracle] - group.target_utility[selected])
        )
    count = len(regrets)
    return {
        "evaluated_group_count": count,
        "pairwise_accuracy": correct / pair_count if pair_count else None,
        "top1_accuracy": top1 / count if count else None,
        "mean_view_regret": float(np.mean(regrets)) if regrets else None,
    }


def score_runtime_candidate_views(
    model: PairwiseLinearEventViewRanker,
    intent: Mapping[str, Any],
    *,
    current_relation_uncertainty: Mapping[str, float],
    closing_axis_world: Sequence[float],
    approach_axis_world: Sequence[float],
    observed_view_directions_world: Sequence[Sequence[float]],
    candidate_views: Sequence[Mapping[str, Any]],
    belief_centroid_world: Sequence[float] | None = None,
    prior_covariance_m2: Sequence[Sequence[float]] | None = None,
    sensor_model: ViewSensorModel | None = None,
    focal_length_px: float | None = None,
    camera_standoff_m: float | None = None,
) -> list[dict[str, Any]]:
    """Score candidate camera poses from learned graph state only.

    ``belief_centroid_world`` and ``camera_standoff_m`` together reconstruct a
    candidate camera position, which is what makes the range and position
    features non-degenerate. ``prior_covariance_m2`` plus a calibrated
    ``sensor_model`` additionally enable the L1 Fisher features required by
    ``analytic_fisher`` mode. All four are belief-side or declared layout
    quantities; none is simulator truth.
    """

    if not observed_view_directions_world:
        raise ValueError("runtime event-view scoring requires an observed view")
    needs_fisher = model.feature_config.mode == "analytic_fisher"
    # Deployment reuses the calibration recorded in the checkpoint unless the
    # caller overrides it explicitly, so a tool does not have to restate --
    # or guess -- the sensor parameters the weights were fitted under.
    calibration = getattr(model, "analytic_fisher_context", None)
    if calibration is not None:
        if prior_covariance_m2 is None:
            prior_covariance_m2 = calibration.prior_covariance_m2
        if sensor_model is None:
            sensor_model = calibration.sensor_model
        if focal_length_px is None:
            focal_length_px = calibration.focal_length_px
        if camera_standoff_m is None:
            camera_standoff_m = calibration.camera_standoff_m
    if needs_fisher and (
        prior_covariance_m2 is None
        or sensor_model is None
        or focal_length_px is None
        or belief_centroid_world is None
        or camera_standoff_m is None
    ):
        raise ValueError(
            "analytic_fisher scoring requires prior covariance, calibrated sensor "
            "model, focal length, belief centroid, and camera standoff"
        )
    closing = _unit_vector(closing_axis_world, "closing axis")
    approach = _unit_vector(approach_axis_world, "approach axis")
    observed = [
        _unit_vector(value, "observed view direction")
        for value in observed_view_directions_world
    ]
    feature_rows = []
    output_rows = []
    for candidate in candidate_views:
        view = str(candidate.get("view", ""))
        direction = _unit_vector(
            candidate.get("view_direction_world"), "candidate view direction"
        )
        closest_similarity = max(
            abs(float(np.dot(direction, previous))) for previous in observed
        )
        move_dot = float(np.clip(np.dot(direction, observed[-1]), -1.0, 1.0))
        input_features = {
            "current_relation_uncertainty": {
                name: float(current_relation_uncertainty[name])
                for name in RELATION_NAMES
            },
            "candidate_closing_axis_observability": float(
                np.clip(1.0 - abs(float(np.dot(direction, closing))), 0.0, 1.0)
            ),
            "candidate_approach_axis_observability": float(
                np.clip(1.0 - abs(float(np.dot(direction, approach))), 0.0, 1.0)
            ),
            "candidate_view_direction_world": direction.tolist(),
            "angular_diversity": float(1.0 - closest_similarity),
            "move_cost": float(math.acos(move_dot) / math.pi),
        }
        candidate_position = candidate.get("camera_position_world")
        position_source = "measured_camera_extrinsic"
        if (
            candidate_position is None
            and belief_centroid_world is not None
            and camera_standoff_m is not None
        ):
            # Degenerate on purpose and labelled as such: a position rebuilt from
            # a standoff is collinear with the view ray, so the position and
            # range features are a deterministic function of the direction and
            # add no information. Only a measured extrinsic breaks that tie.
            position_source = "reconstructed_from_declared_standoff_degenerate"
            candidate_position = camera_position_from_standoff(
                direction,
                belief_centroid_world=belief_centroid_world,
                standoff_m=camera_standoff_m,
            )
        input_features["relative_pose_features"] = relative_view_geometry(
            direction,
            closing_axis_world=closing,
            approach_axis_world=approach,
            observed_view_directions_world=observed,
            candidate_camera_position_world=(
                candidate_position
                if candidate_position is not None and belief_centroid_world is not None
                else None
            ),
            belief_centroid_world=(
                belief_centroid_world
                if candidate_position is not None and belief_centroid_world is not None
                else None
            ),
            predicted_target_visibility=candidate.get("predicted_target_visibility"),
            depth_noise_at_range=candidate.get("depth_noise_at_range"),
            occlusion_risk=candidate.get("occlusion_risk"),
            camera_reachable=candidate.get("camera_reachable"),
            camera_safe=candidate.get("camera_safe"),
        )
        if needs_fisher:
            assert sensor_model is not None and focal_length_px is not None
            relative_pose = input_features["relative_pose_features"]
            fisher_range_m = (
                float(relative_pose["candidate_range_m"])
                if relative_pose["candidate_position_available"] > 0.5
                else float(camera_standoff_m)
            )
            input_features[
                "analytic_fisher_features"
            ] = analytic_view_information_features(
                prior_covariance=np.asarray(prior_covariance_m2, dtype=np.float64),
                candidate_view_direction_world=direction,
                task_axes_world=uncertainty_frame(
                    closing_axis_world=closing,
                    approach_axis_world=approach,
                ),
                range_m=fisher_range_m,
                focal_length_px=float(focal_length_px),
                sensor_model=sensor_model,
                measurement_probability=float(
                    input_features["relative_pose_features"][
                        "predicted_target_visibility"
                    ]
                ),
            )
        feature_rows.append(
            event_view_feature_vector(
                intent,
                view,
                input_features,
                feature_config=model.feature_config,
            )
        )
        output_rows.append(
            {
                "view": view,
                "camera_position_source": (
                    position_source if candidate_position is not None else "unavailable"
                ),
                "input_features": input_features,
                "feature_mode": model.feature_config.mode,
                "fisher_range_source": (
                    "candidate_layout_pose"
                    if input_features["relative_pose_features"][
                        "candidate_position_available"
                    ]
                    > 0.5
                    else "declared_standoff_fallback"
                ),
            }
        )
    if not feature_rows:
        return []
    feature_matrix = np.stack(feature_rows)
    scores = model.score(feature_matrix)
    probability_predictor = getattr(model, "predict_safe_probability", None)
    safe_probabilities = (
        probability_predictor(feature_matrix)
        if callable(probability_predictor)
        else None
    )
    ood_predictor = getattr(model, "predict_ood_score", None)
    ood_scores = ood_predictor(feature_matrix) if callable(ood_predictor) else None
    for index, (row, score) in enumerate(zip(output_rows, scores)):
        row["learned_score"] = round(float(score), 8)
        if safe_probabilities is not None:
            row["predicted_safe_candidate_probability"] = round(
                float(safe_probabilities[index]), 8
            )
        if ood_scores is not None:
            row["ood_score"] = round(float(ood_scores[index]), 8)
    output_rows.sort(key=lambda row: float(row["learned_score"]), reverse=True)
    return output_rows


def _hashed_query_vector(intent: Mapping[str, Any]) -> np.ndarray:
    fields = []
    for key in (
        "target",
        "task_goal",
        "task_stage",
        "post_grasp_goal",
        "contact_pattern",
        "approach_relation",
        "closing_axis_relation",
    ):
        if intent.get(key):
            fields.append(str(intent[key]))
    for key in ("preferred_roles", "natural_language_constraints"):
        value = intent.get(key, ())
        fields.extend(
            [value] if isinstance(value, str) else [str(item) for item in value]
        )
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


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _ranker_vector(value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("event-view ranker vector has an unexpected shape")
    return result


def _unit_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return result / norm
