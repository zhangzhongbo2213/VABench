"""Downstream-safe view ranking with an explicit stop-for-review option.

The existing event-view ranker learns a relative realized-utility ordering.  A
relative score cannot tell whether every available view is unhelpful.  This
module keeps that within-state ranker, then adds an absolute probability head
trained on whether a view produced at least one physically safe grasp
candidate.  The absolute head is calibrated on held-out states and is used only
to choose between ``acquire_view`` and ``stop_for_review``.

Physical probe labels are accepted only while constructing training/evaluation
groups. Runtime scoring consumes the same sparse-graph and camera-pose feature
contract as :mod:`expert_event_view_ranker`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .expert_event_view_ranker import (
    AnalyticFisherContext,
    DEFAULT_FEATURE_CONFIG,
    EventViewFeatureConfig,
    EventViewRankingGroup,
    PairwiseLinearEventViewRanker,
    ranking_group_from_event_view_sample,
)


SAFE_CANDIDATE_VIEW_RANKER_SCHEMA = "phase16.safe_candidate_view_ranker.v3"
PREVIOUS_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA = "phase16.safe_candidate_view_ranker.v2"
LEGACY_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA = "phase15.safe_candidate_view_ranker.v1"
SAFE_VIEW_TARGET_MODES = (
    "safe_candidate_available",
    "safe_top1",
    "hybrid",
)


@dataclass(frozen=True)
class SafeCandidateViewGroup:
    """One frozen state with per-view downstream physical labels."""

    ranking_group: EventViewRankingGroup
    safe_candidate_available: np.ndarray
    safe_top1: np.ndarray
    oracle_utility: np.ndarray
    domain: str = "unspecified"

    def __post_init__(self) -> None:
        size = len(self.ranking_group.views)
        safe = np.asarray(self.safe_candidate_available, dtype=np.float64)
        top1 = np.asarray(self.safe_top1, dtype=np.float64)
        utility = np.asarray(self.oracle_utility, dtype=np.float64)
        for name, value in (
            ("safe_candidate_available", safe),
            ("safe_top1", top1),
            ("oracle_utility", utility),
        ):
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} labels must align with candidate views")
        if np.any((safe < 0.0) | (safe > 1.0)):
            raise ValueError("safe-candidate labels must be binary")
        if np.any((top1 < 0.0) | (top1 > 1.0)):
            raise ValueError("safe-top1 labels must be binary")
        if np.any(top1 > safe):
            raise ValueError("safe_top1 implies safe_candidate_available")
        domain = str(self.domain).strip()
        if not domain:
            raise ValueError("safe-view domain must be a non-empty string")
        object.__setattr__(self, "safe_candidate_available", safe)
        object.__setattr__(self, "safe_top1", top1)
        object.__setattr__(self, "oracle_utility", utility)
        object.__setattr__(self, "domain", domain)

    @property
    def sample_id(self) -> str:
        return self.ranking_group.sample_id

    @property
    def task(self) -> str:
        return self.ranking_group.task

    @property
    def seed(self) -> int:
        return self.ranking_group.seed

    @property
    def views(self) -> tuple[str, ...]:
        return self.ranking_group.views

    @property
    def features(self) -> np.ndarray:
        return self.ranking_group.features

    @property
    def has_safe_view(self) -> bool:
        return bool(np.any(self.safe_candidate_available == 1.0))


def safe_candidate_view_group_from_sample(
    sample: Mapping[str, Any],
    *,
    feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
    phase: str = "evaluation",
    domain: str = "unspecified",
    analytic_fisher_context: AnalyticFisherContext | None = None,
) -> SafeCandidateViewGroup:
    """Build a group while keeping all physical labels outside the encoder."""

    ranking_group = ranking_group_from_event_view_sample(
        sample,
        feature_config=feature_config,
        analytic_fisher_context=analytic_fisher_context,
        phase=phase,
    )
    normalized_domain = str(domain).strip()
    if not normalized_domain:
        raise ValueError("safe-view domain must be a non-empty string")
    if normalized_domain != "unspecified":
        ranking_group = replace(
            ranking_group,
            sample_id=f"{normalized_domain}:{ranking_group.sample_id}",
        )
    labels = sample.get("training_only", {}).get("view_labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("safe-view training requires training_only.view_labels")
    by_view = {str(row.get("view")): row.get("target", {}) for row in labels}
    missing = [view for view in ranking_group.views if view not in by_view]
    if missing:
        raise ValueError(f"safe-view labels are missing views: {missing}")
    return SafeCandidateViewGroup(
        ranking_group=ranking_group,
        safe_candidate_available=np.asarray(
            [
                bool(by_view[view].get("safe_candidate_available"))
                for view in ranking_group.views
            ]
        ),
        safe_top1=np.asarray(
            [bool(by_view[view].get("safe_top1")) for view in ranking_group.views]
        ),
        oracle_utility=np.asarray(
            [float(by_view[view]["oracle_utility"]) for view in ranking_group.views]
        ),
        domain=normalized_domain,
    )


class SafeCandidateViewRanker:
    """Pairwise safe-view ranker plus an absolute safe-availability classifier."""

    def __init__(
        self,
        *,
        feature_config: EventViewFeatureConfig = DEFAULT_FEATURE_CONFIG,
        target_mode: str = "safe_candidate_available",
    ) -> None:
        if target_mode not in SAFE_VIEW_TARGET_MODES:
            raise ValueError(f"unsupported safe-view target mode: {target_mode}")
        self.ranker = PairwiseLinearEventViewRanker(feature_config=feature_config)
        self.target_mode = target_mode
        size = len(feature_config.feature_names)
        self.probability_feature_mean = np.zeros(size, dtype=np.float64)
        self.probability_feature_scale = np.ones(size, dtype=np.float64)
        self.probability_weights = np.zeros(size, dtype=np.float64)
        self.probability_bias = 0.0
        self.probability_fitted = False
        self.probability_weighting = "class"
        self.acquisition_threshold: float | None = None
        self.domain_acquisition_thresholds: dict[str, float] = {}
        self.task_acquisition_thresholds: dict[str, float] = {}
        self.task_domain_acquisition_thresholds: dict[str, dict[str, float]] = {}
        self.threshold_precedence = (
            "task_domain_pair_then_domain_then_task_then_global"
        )
        self.ood_threshold: float | None = None
        self.calibration_report: dict[str, Any] | None = None

    @property
    def feature_config(self) -> EventViewFeatureConfig:
        return self.ranker.feature_config

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.ranker.feature_names

    @property
    def analytic_fisher_context(self) -> Any:
        return self.ranker.analytic_fisher_context

    def fit(
        self,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        epochs: int = 1200,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        pair_weight_power: float = 0.0,
        probability_epochs: int = 2000,
        probability_learning_rate: float = 0.05,
        probability_l2: float = 1e-3,
        probability_weighting: str = "class_task_domain",
        listwise_epochs: int = 0,
        listwise_learning_rate: float = 0.02,
        listwise_l2: float = 1e-3,
        listwise_weight: float = 1.0,
    ) -> dict[str, Any]:
        if not groups:
            raise ValueError("safe-view ranker needs at least one training group")
        ranking_groups = [self._ranking_target(group) for group in groups]
        ranking = self.ranker.fit(
            ranking_groups,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            pair_weight_power=pair_weight_power,
        )
        listwise = self._refine_listwise_ranking(
            groups,
            epochs=listwise_epochs,
            learning_rate=listwise_learning_rate,
            l2=listwise_l2,
            loss_weight=listwise_weight,
        )
        probability = self._fit_probability_head(
            groups,
            epochs=probability_epochs,
            learning_rate=probability_learning_rate,
            l2=probability_l2,
            weighting=probability_weighting,
        )
        return {
            "ranking": ranking,
            "listwise_safe_ranking": listwise,
            "safe_probability": probability,
        }

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.ranker.score(features)

    def predict_safe_probability(self, features: np.ndarray) -> np.ndarray:
        if not self.probability_fitted:
            raise RuntimeError("safe-view probability head has not been fitted")
        value = np.asarray(features, dtype=np.float64)
        standardized = (
            value - self.probability_feature_mean
        ) / self.probability_feature_scale
        return _sigmoid(standardized @ self.probability_weights + self.probability_bias)

    def predict_ood_score(self, features: np.ndarray) -> np.ndarray:
        """Return an inference-only diagonal Mahalanobis RMS distance."""

        if not self.probability_fitted:
            raise RuntimeError("safe-view probability head has not been fitted")
        value = np.asarray(features, dtype=np.float64)
        standardized = (
            value - self.probability_feature_mean
        ) / self.probability_feature_scale
        return np.sqrt(np.mean(np.square(standardized), axis=-1))

    def calibrate_acquisition_threshold(
        self,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        minimum_safe_precision: float = 0.9,
        minimum_all_negative_stop_rate: float = 0.8,
        ood_quantile: float = 0.99,
        ood_margin: float = 0.05,
        domain_aware: bool = True,
        minimum_domain_calibration_states: int = 5,
        task_aware: bool = True,
        minimum_task_calibration_states: int = 5,
        task_domain_aware: bool = True,
        minimum_task_domain_calibration_states: int = 5,
        minimum_specialized_all_negative_states: int = 2,
    ) -> dict[str, Any]:
        """Calibrate OOD rejection and conservative acquisition thresholds."""

        if not groups:
            raise ValueError("safe-view threshold calibration needs held-out groups")
        if not 0.0 < ood_quantile <= 1.0 or ood_margin < 0.0:
            raise ValueError("invalid OOD calibration settings")
        if minimum_domain_calibration_states < 1:
            raise ValueError("minimum domain calibration states must be positive")
        if minimum_task_calibration_states < 1:
            raise ValueError("minimum task calibration states must be positive")
        if minimum_task_domain_calibration_states < 1:
            raise ValueError(
                "minimum task-domain calibration states must be positive"
            )
        if minimum_specialized_all_negative_states < 1:
            raise ValueError(
                "minimum specialized all-negative states must be positive"
            )

        selected_ood_scores = np.asarray(
            [
                float(
                    self.predict_ood_score(group.features)[
                        np.argmax(self.score(group.features))
                    ]
                )
                for group in groups
            ],
            dtype=np.float64,
        )
        self.ood_threshold = float(
            np.quantile(selected_ood_scores, ood_quantile) + ood_margin
        )

        global_calibration = self._select_acquisition_threshold(
            groups,
            minimum_safe_precision=minimum_safe_precision,
            minimum_all_negative_stop_rate=minimum_all_negative_stop_rate,
        )
        self.acquisition_threshold = float(global_calibration["threshold"])
        self.domain_acquisition_thresholds = {}
        self.task_acquisition_thresholds = {}
        self.task_domain_acquisition_thresholds = {}
        domain_reports: dict[str, Any] = {}
        if domain_aware:
            for domain in sorted({group.domain for group in groups}):
                domain_groups = [group for group in groups if group.domain == domain]
                if (
                    domain == "unspecified"
                    or len(domain_groups) < minimum_domain_calibration_states
                    or _all_negative_state_count(domain_groups)
                    < minimum_specialized_all_negative_states
                ):
                    domain_reports[domain] = {
                        "status": "fallback_to_global_insufficient_coverage",
                        "state_count": len(domain_groups),
                        "all_negative_state_count": _all_negative_state_count(
                            domain_groups
                        ),
                        "threshold": self.acquisition_threshold,
                    }
                    continue
                calibration = self._select_acquisition_threshold(
                    domain_groups,
                    minimum_safe_precision=minimum_safe_precision,
                    minimum_all_negative_stop_rate=minimum_all_negative_stop_rate,
                )
                if calibration["status"] == "calibrated":
                    self.domain_acquisition_thresholds[domain] = float(
                        calibration["threshold"]
                    )
                else:
                    calibration = {
                        **calibration,
                        "status": "fallback_to_global_no_safe_threshold",
                        "threshold": self.acquisition_threshold,
                    }
                domain_reports[domain] = calibration

        task_domain_reports: dict[str, dict[str, Any]] = {}
        if task_domain_aware:
            pairs = sorted({(group.domain, group.task) for group in groups})
            for domain, task in pairs:
                pair_groups = [
                    group
                    for group in groups
                    if group.domain == domain and group.task == task
                ]
                if (
                    domain == "unspecified"
                    or len(pair_groups) < minimum_task_domain_calibration_states
                    or _all_negative_state_count(pair_groups)
                    < minimum_specialized_all_negative_states
                ):
                    calibration = {
                        "status": "fallback_to_less_specific_insufficient_coverage",
                        "state_count": len(pair_groups),
                        "all_negative_state_count": _all_negative_state_count(
                            pair_groups
                        ),
                        "threshold": None,
                    }
                else:
                    calibration = self._select_acquisition_threshold(
                        pair_groups,
                        minimum_safe_precision=minimum_safe_precision,
                        minimum_all_negative_stop_rate=(
                            minimum_all_negative_stop_rate
                        ),
                    )
                    if calibration["status"] == "calibrated":
                        self.task_domain_acquisition_thresholds.setdefault(
                            domain, {}
                        )[task] = float(calibration["threshold"])
                    else:
                        calibration = {
                            **calibration,
                            "status": "fallback_to_less_specific_no_safe_threshold",
                            "threshold": None,
                        }
                task_domain_reports.setdefault(domain, {})[task] = calibration

        task_reports: dict[str, Any] = {}
        if task_aware:
            for task in sorted({group.task for group in groups}):
                task_groups = [group for group in groups if group.task == task]
                if (
                    len(task_groups) < minimum_task_calibration_states
                    or _all_negative_state_count(task_groups)
                    < minimum_specialized_all_negative_states
                ):
                    task_reports[task] = {
                        "status": "fallback_to_global_insufficient_coverage",
                        "state_count": len(task_groups),
                        "all_negative_state_count": _all_negative_state_count(
                            task_groups
                        ),
                        "threshold": self.acquisition_threshold,
                    }
                    continue
                calibration = self._select_acquisition_threshold(
                    task_groups,
                    minimum_safe_precision=minimum_safe_precision,
                    minimum_all_negative_stop_rate=minimum_all_negative_stop_rate,
                )
                if calibration["status"] == "calibrated":
                    self.task_acquisition_thresholds[task] = float(
                        calibration["threshold"]
                    )
                else:
                    calibration = {
                        **calibration,
                        "status": "fallback_to_global_no_safe_threshold",
                        "threshold": self.acquisition_threshold,
                    }
                task_reports[task] = calibration

        metrics = evaluate_safe_view_policy(self, groups)
        self.calibration_report = {
            "status": global_calibration["status"],
            "minimum_safe_precision": float(minimum_safe_precision),
            "minimum_all_negative_stop_rate": float(minimum_all_negative_stop_rate),
            "metrics": metrics,
            "global_calibration": global_calibration,
            "domain_calibration": domain_reports,
            "task_calibration": task_reports,
            "task_domain_calibration": task_domain_reports,
            "domain_aware": bool(domain_aware),
            "task_aware": bool(task_aware),
            "task_domain_aware": bool(task_domain_aware),
            "minimum_specialized_all_negative_states": int(
                minimum_specialized_all_negative_states
            ),
            "ood": {
                "method": "selected_candidate_diagonal_mahalanobis_rms",
                "quantile": float(ood_quantile),
                "margin": float(ood_margin),
                "threshold": self.ood_threshold,
                "selected_score_min": float(np.min(selected_ood_scores)),
                "selected_score_max": float(np.max(selected_ood_scores)),
            },
            "calibration_state_count": len(groups),
        }
        return dict(self.calibration_report)

    def _select_acquisition_threshold(
        self,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        minimum_safe_precision: float,
        minimum_all_negative_stop_rate: float,
    ) -> dict[str, Any]:
        selected_probabilities = [
            float(
                self.predict_safe_probability(group.features)[
                    np.argmax(self.score(group.features))
                ]
            )
            for group in groups
        ]
        candidates = []
        for threshold in sorted({0.0, 1.0, *selected_probabilities}):
            metrics = evaluate_safe_view_policy(
                self,
                groups,
                acquisition_threshold=threshold,
                use_domain_thresholds=False,
            )
            precision = metrics["acquired_safe_precision"]
            stop_rate = metrics["all_negative_stop_rate"]
            eligible = bool(
                precision is not None
                and precision >= minimum_safe_precision
                and stop_rate is not None
                and stop_rate >= minimum_all_negative_stop_rate
            )
            candidates.append(
                (eligible, metrics["acquisition_coverage"], -threshold, metrics)
            )
        eligible_rows = [row for row in candidates if row[0]]
        if eligible_rows:
            _, _, _, selected_metrics = max(
                eligible_rows, key=lambda row: (row[1], row[2])
            )
            status = "calibrated"
        else:
            selected_metrics = evaluate_safe_view_policy(
                self,
                groups,
                acquisition_threshold=1.0,
                use_domain_thresholds=False,
            )
            status = "blocked_no_threshold_meets_safety_gates"
        return {
            "status": status,
            "threshold": float(selected_metrics["acquisition_threshold"]),
            "metrics": selected_metrics,
            "state_count": len(groups),
        }

    def policy_decision(
        self,
        features: np.ndarray,
        views: Sequence[str],
        *,
        acquisition_threshold: float | None = None,
        ood_threshold: float | None = None,
        domain: str | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        if len(views) == 0:
            return {
                "action": "stop_for_review",
                "view": None,
                "reason": "no_candidate_views",
            }
        threshold_source = "override"
        joint_threshold = (
            self.task_domain_acquisition_thresholds.get(domain, {}).get(task)
            if domain is not None and task is not None
            else None
        )
        if acquisition_threshold is None and joint_threshold is not None:
            threshold = joint_threshold
            threshold_source = "task_domain"
        elif (
            acquisition_threshold is None
            and self.threshold_precedence
            == "task_domain_pair_then_domain_then_task_then_global"
            and domain is not None
            and domain in self.domain_acquisition_thresholds
        ):
            threshold = self.domain_acquisition_thresholds[domain]
            threshold_source = "domain"
        elif (
            acquisition_threshold is None
            and task is not None
            and task in self.task_acquisition_thresholds
        ):
            threshold = self.task_acquisition_thresholds[task]
            threshold_source = "task"
        elif acquisition_threshold is None and domain is not None:
            threshold = self.domain_acquisition_thresholds.get(
                domain, self.acquisition_threshold
            )
            threshold_source = (
                "domain" if domain in self.domain_acquisition_thresholds else "global"
            )
        else:
            threshold = (
                self.acquisition_threshold
                if acquisition_threshold is None
                else float(acquisition_threshold)
            )
            if acquisition_threshold is None:
                threshold_source = "global"
        if threshold is None:
            raise RuntimeError(
                "safe-view acquisition threshold has not been calibrated"
            )
        scores = self.score(features)
        probabilities = self.predict_safe_probability(features)
        ood_scores = self.predict_ood_score(features)
        selected = int(np.argmax(scores))
        selected_ood = float(ood_scores[selected])
        effective_ood_threshold = (
            self.ood_threshold if ood_threshold is None else float(ood_threshold)
        )
        is_ood = bool(
            effective_ood_threshold is not None
            and selected_ood > effective_ood_threshold
        )
        acquire = bool(not is_ood and float(probabilities[selected]) >= threshold)
        if is_ood:
            reason = "out_of_distribution"
        elif acquire:
            reason = "predicted_safe_candidate_available"
        else:
            reason = "no_safe_view_predicted"
        return {
            "action": "acquire_view" if acquire else "stop_for_review",
            "view": str(views[selected]) if acquire else None,
            "ranked_view": str(views[selected]),
            "learned_score": float(scores[selected]),
            "predicted_safe_candidate_probability": float(probabilities[selected]),
            "acquisition_threshold": float(threshold),
            "acquisition_threshold_source": threshold_source,
            "ood_score": selected_ood,
            "ood_threshold": effective_ood_threshold,
            "domain": domain or "unspecified",
            "task": task,
            "reason": reason,
        }

    def as_dict(
        self, *, training_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.probability_fitted or self.acquisition_threshold is None:
            raise RuntimeError("safe-view ranker must be fitted and calibrated")
        ranker_training_metadata = {}
        fisher_context = getattr(self.ranker, "analytic_fisher_context", None)
        if fisher_context is not None:
            ranker_training_metadata["analytic_fisher_context"] = (
                fisher_context.as_dict()
            )
        return {
            "schema_version": SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
            "model_type": "domain_aware_safe_view_ranker_with_ood_rejection",
            "target_mode": self.target_mode,
            "ranking_model": self.ranker.as_dict(
                training_metadata=ranker_training_metadata
            ),
            "probability_feature_mean": self.probability_feature_mean.tolist(),
            "probability_feature_scale": self.probability_feature_scale.tolist(),
            "probability_weights": self.probability_weights.tolist(),
            "probability_bias": float(self.probability_bias),
            "probability_weighting": self.probability_weighting,
            "acquisition_threshold": float(self.acquisition_threshold),
            "domain_acquisition_thresholds": dict(self.domain_acquisition_thresholds),
            "task_acquisition_thresholds": dict(self.task_acquisition_thresholds),
            "task_domain_acquisition_thresholds": {
                domain: dict(thresholds)
                for domain, thresholds in self.task_domain_acquisition_thresholds.items()
            },
            "threshold_precedence": self.threshold_precedence,
            "ood_threshold": self.ood_threshold,
            "calibration_report": self.calibration_report,
            "training_metadata": dict(training_metadata or {}),
            "input_contract": {
                "runtime_features": "inference_visible_sparse_graph_and_camera_geometry",
                "runtime_domain": "declared_collection_or_deployment_regime_only",
                "physical_probe_labels_runtime_visible": False,
                "execution_authorization": "disabled_shadow_only",
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SafeCandidateViewRanker":
        schema = value.get("schema_version")
        if schema not in {
            SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
            PREVIOUS_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
            LEGACY_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
        }:
            raise ValueError("unsupported safe-candidate view ranker schema")
        ranking_payload = value.get("ranking_model")
        if not isinstance(ranking_payload, Mapping):
            raise ValueError("safe-view checkpoint has no ranking model")
        ranking_model = PairwiseLinearEventViewRanker.from_dict(ranking_payload)
        model = cls(
            feature_config=ranking_model.feature_config,
            target_mode=str(value.get("target_mode", "")),
        )
        model.ranker = ranking_model
        model.threshold_precedence = str(
            value.get(
                "threshold_precedence",
                (
                    "task_domain_pair_then_domain_then_task_then_global"
                    if schema == SAFE_CANDIDATE_VIEW_RANKER_SCHEMA
                    else "task_then_domain_then_global"
                ),
            )
        )
        if model.threshold_precedence not in {
            "task_domain_pair_then_domain_then_task_then_global",
            "task_then_domain_then_global",
        }:
            raise ValueError("safe-view threshold precedence is invalid")
        size = len(model.feature_names)
        model.probability_feature_mean = _vector(
            value.get("probability_feature_mean"), size
        )
        model.probability_feature_scale = _vector(
            value.get("probability_feature_scale"), size
        )
        model.probability_weights = _vector(value.get("probability_weights"), size)
        if np.any(model.probability_feature_scale <= 0.0):
            raise ValueError("safe-view probability scales must be positive")
        model.probability_bias = float(value.get("probability_bias"))
        model.acquisition_threshold = float(value.get("acquisition_threshold"))
        if not np.isfinite(model.probability_bias) or not (
            0.0 <= model.acquisition_threshold <= 1.0
        ):
            raise ValueError("safe-view checkpoint calibration is invalid")
        calibration = value.get("calibration_report")
        model.calibration_report = (
            dict(calibration) if isinstance(calibration, Mapping) else None
        )
        model.probability_weighting = str(value.get("probability_weighting", "class"))
        domain_thresholds = value.get("domain_acquisition_thresholds", {})
        if not isinstance(domain_thresholds, Mapping):
            raise ValueError("safe-view domain thresholds must be a mapping")
        model.domain_acquisition_thresholds = {
            str(domain): float(threshold)
            for domain, threshold in domain_thresholds.items()
        }
        if any(
            not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
            for threshold in model.domain_acquisition_thresholds.values()
        ):
            raise ValueError("safe-view domain threshold is invalid")
        task_thresholds = value.get("task_acquisition_thresholds", {})
        if not isinstance(task_thresholds, Mapping):
            raise ValueError("safe-view task thresholds must be a mapping")
        model.task_acquisition_thresholds = {
            str(task): float(threshold)
            for task, threshold in task_thresholds.items()
        }
        if any(
            not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
            for threshold in model.task_acquisition_thresholds.values()
        ):
            raise ValueError("safe-view task threshold is invalid")
        joint_thresholds = value.get("task_domain_acquisition_thresholds", {})
        if not isinstance(joint_thresholds, Mapping):
            raise ValueError("safe-view task-domain thresholds must be a mapping")
        model.task_domain_acquisition_thresholds = {}
        for domain, task_values in joint_thresholds.items():
            if not isinstance(task_values, Mapping):
                raise ValueError(
                    "safe-view task-domain threshold entries must be mappings"
                )
            normalized = {
                str(task): float(threshold)
                for task, threshold in task_values.items()
            }
            if any(
                not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
                for threshold in normalized.values()
            ):
                raise ValueError("safe-view task-domain threshold is invalid")
            model.task_domain_acquisition_thresholds[str(domain)] = normalized
        raw_ood_threshold = value.get("ood_threshold")
        model.ood_threshold = (
            None if raw_ood_threshold is None else float(raw_ood_threshold)
        )
        if model.ood_threshold is not None and (
            not np.isfinite(model.ood_threshold) or model.ood_threshold < 0.0
        ):
            raise ValueError("safe-view OOD threshold is invalid")
        model.probability_fitted = True
        return model

    def _ranking_target(self, group: SafeCandidateViewGroup) -> EventViewRankingGroup:
        if self.target_mode == "safe_candidate_available":
            target = group.safe_candidate_available
        elif self.target_mode == "safe_top1":
            target = group.safe_top1
        else:
            utility = group.oracle_utility
            span = float(np.max(utility) - np.min(utility))
            normalized = (
                (utility - np.min(utility)) / span
                if span > 1e-9
                else np.zeros_like(utility)
            )
            target = (
                2.0 * group.safe_candidate_available
                + 0.5 * group.safe_top1
                + 0.05 * normalized
            )
        source = group.ranking_group
        return EventViewRankingGroup(
            sample_id=source.sample_id,
            task=source.task,
            seed=source.seed,
            views=source.views,
            features=source.features,
            target_utility=target,
            feature_names=source.feature_names,
            encoding_phase=source.encoding_phase,
        )

    def _refine_listwise_ranking(
        self,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
        loss_weight: float,
    ) -> dict[str, Any]:
        """Refine pairwise weights by assigning probability mass to safe views."""

        if epochs < 0 or learning_rate <= 0.0 or l2 < 0.0 or loss_weight < 0.0:
            raise ValueError("invalid listwise ranking optimization settings")
        if epochs == 0 or loss_weight == 0.0:
            return {"status": "disabled", "epochs": int(epochs)}

        bucket_counts: dict[tuple[str, str], int] = {}
        for group in groups:
            key = (group.task, group.domain)
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
        group_weights = np.asarray(
            [1.0 / bucket_counts[(group.task, group.domain)] for group in groups],
            dtype=np.float64,
        )
        group_weights /= float(np.mean(group_weights))

        final_loss = 0.0
        for _ in range(epochs):
            gradient = np.zeros_like(self.ranker.weights)
            weighted_loss = 0.0
            for group, group_weight in zip(groups, group_weights):
                standardized = self.ranker._standardize(group.features)
                score_probability = _softmax(standardized @ self.ranker.weights)
                if group.has_safe_view:
                    target = group.safe_candidate_available.astype(np.float64)
                    target /= float(np.sum(target))
                else:
                    best = np.isclose(
                        group.oracle_utility,
                        float(np.max(group.oracle_utility)),
                    ).astype(np.float64)
                    target = best / float(np.sum(best))
                gradient += group_weight * (
                    standardized.T @ (score_probability - target)
                )
                weighted_loss -= group_weight * float(
                    np.sum(target * np.log(np.clip(score_probability, 1e-12, 1.0)))
                )
            gradient /= float(np.sum(group_weights))
            gradient = loss_weight * gradient + l2 * self.ranker.weights
            self.ranker.weights -= learning_rate * gradient
            final_loss = weighted_loss / float(np.sum(group_weights))

        selected_safe = []
        for group in groups:
            selected = int(np.argmax(self.score(group.features)))
            selected_safe.append(bool(group.safe_candidate_available[selected]))
        return {
            "status": "fitted",
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "loss_weight": float(loss_weight),
            "final_cross_entropy": float(final_loss),
            "selected_safe_candidate_available_rate": float(np.mean(selected_safe)),
            "group_weighting": "inverse_task_domain_frequency",
        }

    def _fit_probability_head(
        self,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
        weighting: str,
    ) -> dict[str, Any]:
        if epochs < 1 or learning_rate <= 0.0 or l2 < 0.0:
            raise ValueError("invalid safe-view probability optimization settings")
        if weighting not in {"class", "class_task_domain"}:
            raise ValueError(
                f"unsupported safe-view probability weighting: {weighting}"
            )
        features = np.concatenate([group.features for group in groups], axis=0)
        labels = np.concatenate([group.safe_candidate_available for group in groups])
        positive = int(np.sum(labels == 1.0))
        negative = int(np.sum(labels == 0.0))
        if positive == 0 or negative == 0:
            raise ValueError(
                "safe-view probability training needs positive and negative views"
            )
        self.probability_feature_mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        self.probability_feature_scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (
            features - self.probability_feature_mean
        ) / self.probability_feature_scale
        if weighting == "class":
            sample_weights = np.where(
                labels == 1.0,
                len(labels) / (2.0 * positive),
                len(labels) / (2.0 * negative),
            )
        else:
            buckets = []
            for group in groups:
                buckets.extend(
                    (int(label), group.task, group.domain)
                    for label in group.safe_candidate_available
                )
            bucket_counts: dict[tuple[int, str, str], int] = {}
            for bucket in buckets:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            sample_weights = np.asarray(
                [1.0 / bucket_counts[bucket] for bucket in buckets],
                dtype=np.float64,
            )
            sample_weights /= float(np.mean(sample_weights))
        weighted_prevalence = float(
            np.sum(sample_weights * labels) / np.sum(sample_weights)
        )
        weighted_prevalence = float(np.clip(weighted_prevalence, 1e-6, 1.0 - 1e-6))
        self.probability_bias = float(
            np.log(weighted_prevalence / (1.0 - weighted_prevalence))
        )
        self.probability_weights = np.zeros_like(self.probability_weights)
        self.probability_weighting = weighting
        for _ in range(epochs):
            probability = _sigmoid(
                standardized @ self.probability_weights + self.probability_bias
            )
            residual = (probability - labels) * sample_weights
            gradient = standardized.T @ residual / float(np.sum(sample_weights))
            gradient += l2 * self.probability_weights
            self.probability_weights -= learning_rate * gradient
            self.probability_bias -= learning_rate * float(
                np.sum(residual) / np.sum(sample_weights)
            )
        self.probability_fitted = True
        probability = self.predict_safe_probability(features)
        return {
            "candidate_view_count": len(labels),
            "positive_view_count": positive,
            "negative_view_count": negative,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "weighting": weighting,
            "domain_state_counts": {
                domain: sum(group.domain == domain for group in groups)
                for domain in sorted({group.domain for group in groups})
            },
            "brier_score": float(np.mean(np.square(probability - labels))),
        }


def evaluate_safe_view_policy(
    model: SafeCandidateViewRanker,
    groups: Sequence[SafeCandidateViewGroup],
    *,
    acquisition_threshold: float | None = None,
    ood_threshold: float | None = None,
    use_domain_thresholds: bool = True,
) -> dict[str, Any]:
    threshold = (
        model.acquisition_threshold
        if acquisition_threshold is None
        else float(acquisition_threshold)
    )
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "safe-view evaluation needs an acquisition threshold in [0, 1]"
        )
    rows = []
    for group in groups:
        decision = model.policy_decision(
            group.features,
            group.views,
            acquisition_threshold=(
                acquisition_threshold
                if not use_domain_thresholds or acquisition_threshold is not None
                else None
            ),
            ood_threshold=ood_threshold,
            domain=group.domain if use_domain_thresholds else None,
            task=group.task,
        )
        selected = group.views.index(str(decision["ranked_view"]))
        acquired = decision["action"] == "acquire_view"
        rows.append(
            {
                "sample_id": group.sample_id,
                "task": group.task,
                "seed": group.seed,
                "domain": group.domain,
                "selected_view": group.views[selected],
                "selected_safe_candidate_available": bool(
                    group.safe_candidate_available[selected]
                ),
                "selected_safe_top1": bool(group.safe_top1[selected]),
                "predicted_safe_candidate_probability": decision[
                    "predicted_safe_candidate_probability"
                ],
                "acquisition_threshold": decision["acquisition_threshold"],
                "acquisition_threshold_source": decision[
                    "acquisition_threshold_source"
                ],
                "ood_score": decision["ood_score"],
                "ood_threshold": decision["ood_threshold"],
                "action": decision["action"],
                "stop_reason": None if acquired else decision["reason"],
                "state_has_safe_view": group.has_safe_view,
            }
        )
    summary = _summarize_policy_rows(rows)
    summary.update(
        {
            "acquisition_threshold": float(threshold),
            "domain_acquisition_thresholds": dict(model.domain_acquisition_thresholds),
            "task_acquisition_thresholds": dict(model.task_acquisition_thresholds),
            "task_domain_acquisition_thresholds": {
                domain: dict(thresholds)
                for domain, thresholds in model.task_domain_acquisition_thresholds.items()
            },
            "ood_threshold": model.ood_threshold
            if ood_threshold is None
            else ood_threshold,
            "per_domain": {
                domain: _summarize_policy_rows(
                    [row for row in rows if row["domain"] == domain]
                )
                for domain in sorted({str(row["domain"]) for row in rows})
            },
            "rows": rows,
        }
    )
    return summary


def _summarize_policy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    acquired = [row for row in rows if row["action"] == "acquire_view"]
    positive_states = [row for row in rows if row["state_has_safe_view"]]
    all_negative = [row for row in rows if not row["state_has_safe_view"]]
    return {
        "state_count": len(rows),
        "positive_state_count": len(positive_states),
        "all_negative_state_count": len(all_negative),
        "selected_safe_candidate_available_rate": _mean_bool(
            rows, "selected_safe_candidate_available"
        ),
        "selected_safe_top1_rate": _mean_bool(rows, "selected_safe_top1"),
        "acquired_state_count": len(acquired),
        "acquisition_coverage": len(acquired) / len(rows) if rows else None,
        "acquired_safe_precision": _mean_bool(
            acquired, "selected_safe_candidate_available"
        ),
        "positive_state_safe_acquisition_rate": (
            sum(
                row["action"] == "acquire_view"
                and row["selected_safe_candidate_available"]
                for row in positive_states
            )
            / len(positive_states)
            if positive_states
            else None
        ),
        "all_negative_stop_rate": (
            sum(row["action"] == "stop_for_review" for row in all_negative)
            / len(all_negative)
            if all_negative
            else None
        ),
        "out_of_distribution_stop_count": sum(
            row.get("stop_reason") == "out_of_distribution" for row in rows
        ),
        "no_safe_view_predicted_stop_count": sum(
            row.get("stop_reason") == "no_safe_view_predicted" for row in rows
        ),
    }


def _all_negative_state_count(groups: Sequence[SafeCandidateViewGroup]) -> int:
    return sum(not group.has_safe_view for group in groups)


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row[key]) for row in rows]))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - float(np.max(value))
    exponential = np.exp(np.clip(shifted, -40.0, 40.0))
    return exponential / float(np.sum(exponential))


def _vector(value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("safe-view ranker vector has an unexpected shape")
    return result
