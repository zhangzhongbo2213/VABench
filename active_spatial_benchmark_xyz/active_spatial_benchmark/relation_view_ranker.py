"""Relation-decomposed learned policy for auditable next-view selection.

The existing event-view ranker predicts one opaque composite utility.  This
module instead fits one pairwise head per graph relation, calibrates every head
back to normalized error-reduction units, and aggregates the heads with an
explicit query weight vector before subtracting camera motion cost.

Oracle relation reductions are consumed only by :meth:`fit` and offline
evaluation. Runtime scoring delegates feature construction to the same
inference-visible graph/camera adapter as the scalar ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .active_belief import ViewSensorModel
from .expert_event_view_dataset import RELATION_NAMES
from .expert_event_view_ranker import (
    DEFAULT_FEATURE_CONFIG,
    EventViewFeatureConfig,
    EventViewRankingGroup,
    PairwiseLinearEventViewRanker,
    ranking_group_from_event_view_sample,
    score_runtime_candidate_views,
)
from .realized_event_view_dataset import REALIZED_EVENT_VIEW_SAMPLE_SCHEMA


RELATION_VIEW_RANKER_SCHEMA = "spatial.relation_decomposed_view_ranker.v1"
RELATION_ERROR_SCALES = {
    "grasp_center_location": 0.03,
    "left_contact": 0.03,
    "right_contact": 0.03,
    "opening_width": 0.02,
    "approach_axis": 30.0,
    "closing_axis": 45.0,
}
RELATION_ERROR_WEIGHTS = {
    "grasp_center_location": 1.0,
    "left_contact": 1.2,
    "right_contact": 1.2,
    "opening_width": 1.1,
    "approach_axis": 0.8,
    "closing_axis": 0.9,
}


@dataclass(frozen=True)
class RelationHeadCalibration:
    """Positive affine map from pairwise score to normalized error reduction."""

    scale: float
    offset: float
    score_std: float
    target_std: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (self.scale, self.offset, self.score_std, self.target_std)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("relation-head calibration values must be finite")
        if self.scale <= 0.0 or self.score_std <= 0.0 or self.target_std <= 0.0:
            raise ValueError("relation-head calibration scales must be positive")
        if self.sample_count < 2:
            raise ValueError("relation-head calibration needs at least two rows")

    def apply(self, values: np.ndarray) -> np.ndarray:
        return self.offset + self.scale * np.asarray(values, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "offset": float(self.offset),
            "score_std": float(self.score_std),
            "target_std": float(self.target_std),
            "sample_count": int(self.sample_count),
            "method": "positive_standard_deviation_match",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RelationHeadCalibration":
        return cls(
            scale=float(value["scale"]),
            offset=float(value["offset"]),
            score_std=float(value["score_std"]),
            target_std=float(value["target_std"]),
            sample_count=int(value["sample_count"]),
        )


class RelationDecomposedViewRanker:
    """Six calibrated pairwise heads plus an explicit relation aggregator."""

    def __init__(
        self,
        *,
        feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
        move_cost_weight: float = 0.08,
        uncertainty_conditioned_weights: bool = False,
        uncertainty_weight_floor: float = 0.1,
    ) -> None:
        if not np.isfinite(move_cost_weight) or move_cost_weight < 0.0:
            raise ValueError("move cost weight must be finite and non-negative")
        if not 0.0 <= uncertainty_weight_floor <= 1.0:
            raise ValueError("uncertainty weight floor must be in [0, 1]")
        self.feature_config = feature_config
        self.move_cost_weight = float(move_cost_weight)
        self.uncertainty_conditioned_weights = bool(
            uncertainty_conditioned_weights
        )
        self.uncertainty_weight_floor = float(uncertainty_weight_floor)
        self.heads = {
            relation: PairwiseLinearEventViewRanker(
                feature_config=self.feature_config
            )
            for relation in RELATION_NAMES
        }
        self.calibrations: dict[str, RelationHeadCalibration] = {}
        self.fitted = False

    def fit(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        pair_weight_power: float = 0.0,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("relation-decomposed training needs realized samples")
        encoded = [
            _relation_groups_from_sample(
                sample,
                feature_config=self.feature_config,
                phase="training",
            )
            for sample in samples
        ]
        head_metrics: dict[str, Any] = {}
        for relation in RELATION_NAMES:
            groups = [row[relation] for row in encoded]
            model = self.heads[relation]
            optimization = model.fit(
                groups,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                pair_weight_power=pair_weight_power,
            )
            scores = np.concatenate([model.score(group.features) for group in groups])
            targets = np.concatenate([group.target_utility for group in groups])
            score_std = float(np.std(scores))
            target_std = float(np.std(targets))
            if score_std <= 1e-9 or target_std <= 1e-9:
                raise ValueError(
                    f"relation {relation!r} has degenerate score calibration"
                )
            scale = target_std / score_std
            offset = float(np.mean(targets) - scale * np.mean(scores))
            calibration = RelationHeadCalibration(
                scale=scale,
                offset=offset,
                score_std=score_std,
                target_std=target_std,
                sample_count=len(targets),
            )
            self.calibrations[relation] = calibration
            head_metrics[relation] = {
                "optimization": optimization,
                "calibration": calibration.as_dict(),
            }
        self.fitted = True
        return {
            "head_metrics": head_metrics,
            "training_sample_count": len(samples),
            "feature_config": self.feature_config.as_dict(),
            "move_cost_weight": self.move_cost_weight,
            "uncertainty_conditioned_weights": self.uncertainty_conditioned_weights,
            "uncertainty_weight_floor": self.uncertainty_weight_floor,
        }

    def score_sample(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Score stored candidates without exposing their targets to features."""

        self._require_fitted()
        groups = _relation_groups_from_sample(
            sample,
            feature_config=self.feature_config,
            phase="evaluation",
        )
        views = groups[RELATION_NAMES[0]].views
        raw_by_relation = {
            relation: self.heads[relation].score(groups[relation].features)
            for relation in RELATION_NAMES
        }
        predicted = {
            relation: self.calibrations[relation].apply(raw_by_relation[relation])
            for relation in RELATION_NAMES
        }
        first_label = sample["training_only"]["view_labels"][0]
        uncertainty = first_label["input_features"]["current_relation_uncertainty"]
        move_cost_by_view = {
            str(row["view"]): float(row["input_features"]["move_cost"])
            for row in sample["training_only"]["view_labels"]
        }
        return self._aggregate_rows(
            views=views,
            predicted_by_relation=predicted,
            raw_by_relation=raw_by_relation,
            current_relation_uncertainty=uncertainty,
            move_cost_by_view=move_cost_by_view,
        )

    def score_runtime(
        self,
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
        query_relation_weights: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Score arbitrary camera poses from inference-visible graph state."""

        self._require_fitted()
        rows_by_relation: dict[str, dict[str, Mapping[str, Any]]] = {}
        for relation in RELATION_NAMES:
            rows = score_runtime_candidate_views(
                self.heads[relation],
                intent,
                current_relation_uncertainty=current_relation_uncertainty,
                closing_axis_world=closing_axis_world,
                approach_axis_world=approach_axis_world,
                observed_view_directions_world=observed_view_directions_world,
                candidate_views=candidate_views,
                belief_centroid_world=belief_centroid_world,
                prior_covariance_m2=prior_covariance_m2,
                sensor_model=sensor_model,
                focal_length_px=focal_length_px,
                camera_standoff_m=camera_standoff_m,
            )
            rows_by_relation[relation] = {
                str(row["view"]): row for row in rows
            }
        views = tuple(str(row["view"]) for row in candidate_views)
        expected = set(views)
        for relation, rows in rows_by_relation.items():
            if set(rows) != expected:
                raise ValueError(
                    f"relation head {relation!r} scored a different candidate set"
                )
        raw_by_relation = {
            relation: np.asarray(
                [rows_by_relation[relation][view]["learned_score"] for view in views],
                dtype=np.float64,
            )
            for relation in RELATION_NAMES
        }
        predicted = {
            relation: self.calibrations[relation].apply(raw_by_relation[relation])
            for relation in RELATION_NAMES
        }
        move_cost_by_view = {
            view: float(
                rows_by_relation[RELATION_NAMES[0]][view]["input_features"][
                    "move_cost"
                ]
            )
            for view in views
        }
        result = self._aggregate_rows(
            views=views,
            predicted_by_relation=predicted,
            raw_by_relation=raw_by_relation,
            current_relation_uncertainty=current_relation_uncertainty,
            move_cost_by_view=move_cost_by_view,
            query_relation_weights=query_relation_weights,
        )
        for row in result:
            reference = rows_by_relation[RELATION_NAMES[0]][str(row["view"])]
            row["input_features"] = reference["input_features"]
            row["feature_mode"] = self.feature_config.mode
            row["camera_position_source"] = reference.get("camera_position_source")
        return result

    def _aggregate_rows(
        self,
        *,
        views: Sequence[str],
        predicted_by_relation: Mapping[str, np.ndarray],
        raw_by_relation: Mapping[str, np.ndarray],
        current_relation_uncertainty: Mapping[str, float],
        move_cost_by_view: Mapping[str, float],
        query_relation_weights: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        weights = self.relation_weights(
            current_relation_uncertainty,
            query_relation_weights=query_relation_weights,
        )
        weight_sum = float(sum(weights.values()))
        rows = []
        for index, view in enumerate(views):
            relation_gain = {
                relation: float(predicted_by_relation[relation][index])
                for relation in RELATION_NAMES
            }
            contributions = {
                relation: weights[relation] * relation_gain[relation] / weight_sum
                for relation in RELATION_NAMES
            }
            predicted_composite_gain = float(sum(contributions.values()))
            move_cost = float(move_cost_by_view[view])
            move_penalty = self.move_cost_weight * move_cost
            dominant = max(
                RELATION_NAMES,
                key=lambda relation: abs(contributions[relation]),
            )
            rows.append(
                {
                    "view": str(view),
                    "learned_score": predicted_composite_gain - move_penalty,
                    "predicted_normalized_relation_gain": relation_gain,
                    "raw_relation_head_score": {
                        relation: float(raw_by_relation[relation][index])
                        for relation in RELATION_NAMES
                    },
                    "relation_weights": weights,
                    "relation_contributions": contributions,
                    "predicted_composite_error_reduction": predicted_composite_gain,
                    "move_cost": move_cost,
                    "move_cost_penalty": move_penalty,
                    "dominant_relation_reason": dominant,
                    "score_decomposition": (
                        "weighted_predicted_relation_error_reduction_minus_move_cost"
                    ),
                    "access": "inference_visible_prediction",
                }
            )
        rows.sort(key=lambda row: float(row["learned_score"]), reverse=True)
        return rows

    def relation_weights(
        self,
        current_relation_uncertainty: Mapping[str, float],
        *,
        query_relation_weights: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        unknown = set(query_relation_weights or ()) - set(RELATION_NAMES)
        if unknown:
            raise ValueError(
                "unknown query relation weights: " + ", ".join(sorted(unknown))
            )
        result = {}
        for relation in RELATION_NAMES:
            base = float(
                (query_relation_weights or {}).get(
                    relation, RELATION_ERROR_WEIGHTS[relation]
                )
            )
            if not np.isfinite(base) or base < 0.0:
                raise ValueError("query relation weights must be finite and non-negative")
            if self.uncertainty_conditioned_weights:
                uncertainty = float(current_relation_uncertainty[relation])
                if not np.isfinite(uncertainty) or not 0.0 <= uncertainty <= 1.0:
                    raise ValueError("relation uncertainty must be in [0, 1]")
                base *= max(uncertainty, self.uncertainty_weight_floor)
            result[relation] = base
        if sum(result.values()) <= 0.0:
            raise ValueError("at least one query relation weight must be positive")
        return result

    def as_dict(
        self, *, training_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_fitted()
        return {
            "schema_version": RELATION_VIEW_RANKER_SCHEMA,
            "model_type": "relation_decomposed_calibrated_pairwise_linear_ranker",
            "feature_config": self.feature_config.as_dict(),
            "move_cost_weight": self.move_cost_weight,
            "uncertainty_conditioned_weights": self.uncertainty_conditioned_weights,
            "uncertainty_weight_floor": self.uncertainty_weight_floor,
            "relation_error_scales": dict(RELATION_ERROR_SCALES),
            "default_relation_weights": dict(RELATION_ERROR_WEIGHTS),
            "heads": {
                relation: {
                    "ranker": self.heads[relation].as_dict(),
                    "calibration": self.calibrations[relation].as_dict(),
                }
                for relation in RELATION_NAMES
            },
            "training_metadata": dict(training_metadata or {}),
            "input_contract": {
                "uses_oracle_targets_as_features": False,
                "runtime_inputs": (
                    "learned graph uncertainty/axes, task intent, observed camera "
                    "directions, and candidate camera poses"
                ),
                "per_relation_predictions_exposed": True,
                "supports_query_relation_weights": True,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RelationDecomposedViewRanker":
        if value.get("schema_version") != RELATION_VIEW_RANKER_SCHEMA:
            raise ValueError("unsupported relation-decomposed view ranker schema")
        stored_scales = {
            str(name): float(scale)
            for name, scale in value.get("relation_error_scales", {}).items()
        }
        if stored_scales != RELATION_ERROR_SCALES:
            raise ValueError("relation error scale schema mismatch")
        feature_config = EventViewFeatureConfig.from_dict(value["feature_config"])
        model = cls(
            feature_config=feature_config,
            move_cost_weight=float(value["move_cost_weight"]),
            uncertainty_conditioned_weights=bool(
                value.get("uncertainty_conditioned_weights", False)
            ),
            uncertainty_weight_floor=float(
                value.get("uncertainty_weight_floor", 0.1)
            ),
        )
        heads = value.get("heads")
        if not isinstance(heads, Mapping) or set(heads) != set(RELATION_NAMES):
            raise ValueError("relation-decomposed checkpoint head schema mismatch")
        for relation in RELATION_NAMES:
            row = heads[relation]
            model.heads[relation] = PairwiseLinearEventViewRanker.from_dict(
                row["ranker"]
            )
            if model.heads[relation].feature_config != feature_config:
                raise ValueError("relation head feature configuration mismatch")
            model.calibrations[relation] = RelationHeadCalibration.from_dict(
                row["calibration"]
            )
        model.fitted = True
        return model

    def _require_fitted(self) -> None:
        if not self.fitted or set(self.calibrations) != set(RELATION_NAMES):
            raise RuntimeError("relation-decomposed view ranker has not been fitted")


def evaluate_relation_view_ranker(
    model: RelationDecomposedViewRanker,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate composite choices and per-relation prediction error."""

    top1 = 0
    regrets = []
    relation_absolute_errors = {relation: [] for relation in RELATION_NAMES}
    predictions = []
    for sample in samples:
        rows = model.score_sample(sample)
        labels = {
            str(row["view"]): row["target"]
            for row in sample["training_only"]["view_labels"]
        }
        selected = str(rows[0]["view"])
        utility = {
            view: float(target["oracle_utility"])
            for view, target in labels.items()
        }
        oracle = max(utility, key=utility.get)
        top1 += int(selected == oracle)
        regrets.append(utility[oracle] - utility[selected])
        for row in rows:
            view = str(row["view"])
            target_reduction = labels[view]["relation_realized_error_reduction"]
            for relation in RELATION_NAMES:
                target = float(target_reduction[relation]) / RELATION_ERROR_SCALES[
                    relation
                ]
                prediction = float(
                    row["predicted_normalized_relation_gain"][relation]
                )
                relation_absolute_errors[relation].append(abs(prediction - target))
        predictions.append(
            {
                "sample_id": str(sample["sample_id"]),
                "selected_view": selected,
                "oracle_view": oracle,
                "regret": utility[oracle] - utility[selected],
                "ranked_views": rows,
            }
        )
    count = len(samples)
    return {
        "state_count": count,
        "top1_accuracy": top1 / count if count else None,
        "mean_view_regret": float(np.mean(regrets)) if regrets else None,
        "per_relation_normalized_mae": {
            relation: float(np.mean(values)) if values else None
            for relation, values in relation_absolute_errors.items()
        },
        "predictions": predictions,
    }


def _relation_groups_from_sample(
    sample: Mapping[str, Any],
    *,
    feature_config: EventViewFeatureConfig,
    phase: str,
) -> dict[str, EventViewRankingGroup]:
    if sample.get("schema_version") != REALIZED_EVENT_VIEW_SAMPLE_SCHEMA:
        raise ValueError(
            "relation-decomposed ranker requires realized graph-error samples"
        )
    shared = ranking_group_from_event_view_sample(
        sample,
        feature_config=feature_config,
        phase=phase,
    )
    labels = sample["training_only"]["view_labels"]
    if tuple(str(row["view"]) for row in labels) != shared.views:
        raise ValueError("relation targets do not align with encoded candidate views")
    result = {}
    for relation in RELATION_NAMES:
        target = np.asarray(
            [
                float(row["target"]["relation_realized_error_reduction"][relation])
                / RELATION_ERROR_SCALES[relation]
                for row in labels
            ],
            dtype=np.float64,
        )
        result[relation] = EventViewRankingGroup(
            sample_id=shared.sample_id,
            task=shared.task,
            seed=shared.seed,
            views=shared.views,
            features=shared.features,
            target_utility=target,
            feature_names=shared.feature_names,
            encoding_phase=shared.encoding_phase,
        )
    return result
