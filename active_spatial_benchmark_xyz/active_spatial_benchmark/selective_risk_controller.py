"""Finite-sample selective-risk veto for the safe-view ranker.

The controller never changes the ranked view.  It can only veto an acquisition
that the existing ranker would make.  Calibration consumes physical probe
labels offline; runtime scoring consumes model scores and sparse-graph inputs
only.  A controller with no calibration satisfying its risk contract is
disabled and rejects every acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .safe_candidate_view_ranker import (
    SafeCandidateViewGroup,
    SafeCandidateViewRanker,
)


SELECTIVE_RISK_CONTROLLER_SCHEMA = "phase17.selective_risk_controller.v1"


@dataclass(frozen=True)
class SelectiveRiskEvidence:
    selected_index: int
    selected_probability: float
    score_margin: float
    relation_disagreement: float
    ood_score: float
    risk_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "selected_probability": self.selected_probability,
            "score_margin": self.score_margin,
            "relation_disagreement": self.relation_disagreement,
            "ood_score": self.ood_score,
            "risk_score": self.risk_score,
        }


class SelectiveRiskController:
    """Conservative acquisition veto calibrated on held-out physical labels."""

    def __init__(
        self,
        *,
        unsafe_risk_budget: float = 0.10,
        confidence_level: float = 0.95,
        minimum_all_negative_stop_rate: float = 0.80,
        minimum_calibration_states: int = 20,
        proposal_probability_threshold: float | None = None,
    ) -> None:
        if not 0.0 < unsafe_risk_budget < 1.0:
            raise ValueError("unsafe risk budget must be in (0, 1)")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence level must be in (0, 1)")
        if not 0.0 <= minimum_all_negative_stop_rate <= 1.0:
            raise ValueError("all-negative stop rate must be in [0, 1]")
        if minimum_calibration_states < 1:
            raise ValueError("minimum calibration states must be positive")
        if proposal_probability_threshold is not None and not (
            0.0 <= proposal_probability_threshold <= 1.0
        ):
            raise ValueError("proposal probability threshold must be in [0, 1]")
        self.unsafe_risk_budget = float(unsafe_risk_budget)
        self.confidence_level = float(confidence_level)
        self.minimum_all_negative_stop_rate = float(minimum_all_negative_stop_rate)
        self.minimum_calibration_states = int(minimum_calibration_states)
        # ``None`` preserves the ranker's calibrated absolute proposal gate.
        # A lower threshold is useful during risk calibration: it exposes the
        # full ranked proposal stream to the selective controller instead of
        # conflating proposal generation with risk authorization.
        self.proposal_probability_threshold = (
            None
            if proposal_probability_threshold is None
            else float(proposal_probability_threshold)
        )
        self.risk_threshold: float | None = None
        self.enabled = False
        self.calibration_report: dict[str, Any] | None = None
        self.feature_mode: str | None = None
        self.feature_names: tuple[str, ...] | None = None
        self.risk_model_mode = "legacy_max"
        self.risk_feature_names: tuple[str, ...] | None = None
        self.risk_feature_mean: np.ndarray | None = None
        self.risk_feature_scale: np.ndarray | None = None
        self.risk_weights: np.ndarray | None = None
        self.risk_bias = 0.0
        self.risk_fit_state_count = 0
        self.risk_fit_positive_state_count = 0
        self.risk_fit_status: str | None = None

    def fit(
        self,
        model: SafeCandidateViewRanker,
        groups: Sequence[SafeCandidateViewGroup],
        *,
        risk_fit_groups: Sequence[SafeCandidateViewGroup] | None = None,
    ) -> dict[str, Any]:
        if len(groups) < self.minimum_calibration_states:
            raise ValueError(
                "selective-risk calibration has too few states: "
                f"{len(groups)} < {self.minimum_calibration_states}"
            )
        threshold_ids = [str(group.sample_id) for group in groups]
        if len(set(threshold_ids)) != len(threshold_ids):
            raise ValueError("threshold-calibration groups contain duplicate sample_ids")
        self.feature_mode = model.feature_config.mode
        self.feature_names = tuple(model.feature_names)
        self._clear_risk_model()
        self.risk_fit_state_count = 0
        self.risk_fit_positive_state_count = 0
        self.risk_fit_status = None
        records = self._records_for_groups(model, groups)
        if risk_fit_groups is not None:
            if not risk_fit_groups:
                raise ValueError("risk-fit groups must be non-empty")
            risk_ids = [str(group.sample_id) for group in risk_fit_groups]
            if len(set(risk_ids)) != len(risk_ids):
                raise ValueError("risk-fit groups contain duplicate sample_ids")
            if set(risk_ids) & set(threshold_ids):
                raise ValueError(
                    "risk-fit and threshold-calibration groups must be disjoint"
                )
            risk_records = self._records_for_groups(model, risk_fit_groups)
            self.risk_fit_state_count = len(risk_records)
            self.risk_fit_positive_state_count = sum(
                not bool(row["all_negative"]) for row in risk_records
            )
            self._fit_contextual_risk_model(risk_records)
            if self.risk_model_mode != "contextual_logistic_unsafe_probability":
                raise ValueError(
                    "risk-fit split cannot fit contextual unsafe-risk head: "
                    f"{self.risk_fit_status or 'unknown_reason'}"
                )
        for record in records:
            record["risk_score"] = self._risk_score_from_record(record)

        thresholds = sorted({0.0, 1.0, *(float(r["risk_score"]) for r in records)})
        candidates = []
        for threshold in thresholds:
            acquired = [
                row
                for row in records
                if row["base_acquire"] and row["risk_score"] <= threshold
            ]
            unsafe_count = sum(bool(row["unsafe"]) for row in acquired)
            upper = clopper_pearson_upper_bound(
                unsafe_count,
                len(acquired),
                confidence_level=self.confidence_level,
            )
            negative = [row for row in records if row["all_negative"]]
            negative_acquired = sum(
                row["base_acquire"]
                and row["risk_score"] <= threshold
                for row in negative
            )
            stop_rate = (
                1.0 - negative_acquired / len(negative) if negative else None
            )
            positive = [row for row in records if not row["all_negative"]]
            safe_acquisition = sum(
                row["base_acquire"]
                and row["risk_score"] <= threshold
                and row["safe_acquisition"]
                for row in positive
            )
            positive_rate = safe_acquisition / len(positive) if positive else None
            eligible = bool(
                acquired
                and upper <= self.unsafe_risk_budget
                and stop_rate is not None
                and stop_rate >= self.minimum_all_negative_stop_rate
            )
            candidates.append(
                {
                    "threshold": float(threshold),
                    "acquired_state_count": len(acquired),
                    "unsafe_count": unsafe_count,
                    "unsafe_risk_upper_bound": upper,
                    "all_negative_stop_rate": stop_rate,
                    "positive_state_safe_acquisition_rate": positive_rate,
                    "eligible": eligible,
                }
            )

        eligible = [row for row in candidates if row["eligible"]]
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (
                    float(row["positive_state_safe_acquisition_rate"]),
                    -float(row["threshold"]),
                ),
            )
            self.risk_threshold = float(selected["threshold"])
            self.enabled = True
            status = "calibrated"
        else:
            selected = {
                "threshold": None,
                "acquired_state_count": 0,
                "unsafe_count": 0,
                "unsafe_risk_upper_bound": 1.0,
                "all_negative_stop_rate": None,
                "positive_state_safe_acquisition_rate": None,
                "eligible": False,
            }
            self.risk_threshold = None
            self.enabled = False
            status = "blocked_no_threshold_meets_risk_gates"
        self.calibration_report = {
            "status": status,
            "state_count": len(groups),
            "positive_state_count": sum(group.has_safe_view for group in groups),
            "all_negative_state_count": sum(not group.has_safe_view for group in groups),
            "unsafe_risk_budget": self.unsafe_risk_budget,
            "confidence_level": self.confidence_level,
            "minimum_all_negative_stop_rate": self.minimum_all_negative_stop_rate,
            "minimum_calibration_states": self.minimum_calibration_states,
            "selected": selected,
            "frontier": candidates,
            "physical_probe_labels": "calibration_only",
            "runtime_camera_authority": False,
            "risk_model": self._risk_model_report(),
        }
        return dict(self.calibration_report)

    def _records_for_groups(
        self,
        model: SafeCandidateViewRanker,
        groups: Sequence[SafeCandidateViewGroup],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for group in groups:
            evidence = self.evidence(model, group.features)
            decision = model.policy_decision(
                group.features,
                group.views,
                acquisition_threshold=self.proposal_probability_threshold,
                domain=group.domain,
                task=group.task,
            )
            selected = evidence.selected_index
            records.append(
                {
                    "legacy_risk_score": evidence.risk_score,
                    "selected_probability": evidence.selected_probability,
                    "score_margin": evidence.score_margin,
                    "relation_disagreement": evidence.relation_disagreement,
                    "ood_score": evidence.ood_score,
                    "base_acquire": decision["action"] == "acquire_view",
                    "unsafe": not bool(group.safe_candidate_available[selected]),
                    "all_negative": not group.has_safe_view,
                    "task": group.task,
                    "domain": group.domain,
                    "view": group.views[selected],
                    "safe_acquisition": bool(
                        decision["action"] == "acquire_view"
                        and group.safe_candidate_available[selected]
                    ),
                }
            )
        return records

    def _fit_contextual_risk_model(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Fit a small calibration-only unsafe-probability head.

        The base ranker remains responsible for view ordering.  This head only
        estimates whether the selected view is unsafe, using runtime-visible
        scalar evidence and typed task/domain/view context.  It is deliberately
        conservative and falls back to the historical max-risk rule when the
        calibration split has a single class or lacks context diversity.
        """

        labels = np.asarray([float(bool(row["unsafe"])) for row in records])
        if len(records) < 8 or len(np.unique(labels)) < 2:
            self._clear_risk_model()
            self.risk_fit_status = "insufficient_state_or_label_diversity"
            return
        tasks = sorted({str(row["task"]) for row in records})
        domains = sorted({str(row["domain"]) for row in records})
        views = sorted({str(row["view"]) for row in records})
        names = (
            "probability_risk",
            "score_margin",
            "relation_disagreement",
            "ood_score",
            *(f"task={value}" for value in tasks),
            *(f"domain={value}" for value in domains),
            *(f"view={value}" for value in views),
        )
        matrix = np.asarray(
            [
                [
                    1.0 - float(row["selected_probability"]),
                    float(row["score_margin"]),
                    float(row["relation_disagreement"]),
                    float(row["ood_score"]),
                    *[float(str(row["task"]) == value) for value in tasks],
                    *[float(str(row["domain"]) == value) for value in domains],
                    *[float(str(row["view"]) == value) for value in views],
                ]
                for row in records
            ],
            dtype=np.float64,
        )
        mean = np.mean(matrix, axis=0)
        scale = np.where(np.std(matrix, axis=0) > 1e-8, np.std(matrix, axis=0), 1.0)
        standardized = (matrix - mean) / scale
        positive = max(float(np.sum(labels)), 1.0)
        negative = max(float(len(labels) - np.sum(labels)), 1.0)
        sample_weights = np.where(
            labels == 1.0,
            len(labels) / (2.0 * positive),
            len(labels) / (2.0 * negative),
        )
        prevalence = float(np.sum(sample_weights * labels) / np.sum(sample_weights))
        bias = float(
            np.log(
                np.clip(prevalence, 1e-6, 1.0 - 1e-6)
                / np.clip(1.0 - prevalence, 1e-6, 1.0)
            )
        )
        weights = np.zeros(matrix.shape[1], dtype=np.float64)
        for _ in range(1200):
            probability = _sigmoid(standardized @ weights + bias)
            residual = (probability - labels) * sample_weights
            gradient = standardized.T @ residual / float(np.sum(sample_weights))
            gradient += 1e-2 * weights
            weights -= 0.05 * gradient
            bias -= 0.05 * float(np.sum(residual) / np.sum(sample_weights))
        self.risk_model_mode = "contextual_logistic_unsafe_probability"
        self.risk_feature_names = tuple(names)
        self.risk_feature_mean = mean
        self.risk_feature_scale = scale
        self.risk_weights = weights
        self.risk_bias = bias
        self.risk_fit_status = "fitted"

    def _clear_risk_model(self) -> None:
        self.risk_model_mode = "legacy_max"
        self.risk_feature_names = None
        self.risk_feature_mean = None
        self.risk_feature_scale = None
        self.risk_weights = None
        self.risk_bias = 0.0

    def _risk_score_from_record(self, record: Mapping[str, Any]) -> float:
        if self.risk_model_mode != "contextual_logistic_unsafe_probability":
            return float(np.clip(float(record["legacy_risk_score"]), 0.0, 1.0))
        vector = self._risk_vector(
            selected_probability=float(record["selected_probability"]),
            score_margin=float(record["score_margin"]),
            relation_disagreement=float(record["relation_disagreement"]),
            ood_score=float(record["ood_score"]),
            task=str(record["task"]),
            domain=str(record["domain"]),
            view=str(record["view"]),
        )
        standardized = (vector - self.risk_feature_mean) / self.risk_feature_scale
        logit = standardized @ self.risk_weights + self.risk_bias
        return float(_sigmoid(np.asarray([logit]))[0])

    def _risk_vector(
        self,
        *,
        selected_probability: float,
        score_margin: float,
        relation_disagreement: float,
        ood_score: float,
        task: str,
        domain: str,
        view: str,
    ) -> np.ndarray:
        if self.risk_feature_names is None:
            raise RuntimeError("contextual risk model has no feature contract")
        values = {
            "probability_risk": 1.0 - selected_probability,
            "score_margin": score_margin,
            "relation_disagreement": relation_disagreement,
            "ood_score": ood_score,
        }
        for name in self.risk_feature_names:
            if name.startswith("task="):
                values[name] = float(name == f"task={task}")
            elif name.startswith("domain="):
                values[name] = float(name == f"domain={domain}")
            elif name.startswith("view="):
                values[name] = float(name == f"view={view}")
        return np.asarray([values[name] for name in self.risk_feature_names], dtype=np.float64)

    def _risk_model_report(self) -> dict[str, Any]:
        return {
            "mode": self.risk_model_mode,
            "feature_names": list(self.risk_feature_names or ()),
            "fitted": self.risk_weights is not None,
            "risk_fit_state_count": self.risk_fit_state_count,
            "risk_fit_positive_state_count": self.risk_fit_positive_state_count,
            "risk_fit_status": self.risk_fit_status,
        }

    def evidence(
        self, model: SafeCandidateViewRanker, features: np.ndarray
    ) -> SelectiveRiskEvidence:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != len(model.feature_names):
            raise ValueError("selective-risk features have an unexpected shape")
        scores = model.score(value)
        probabilities = model.predict_safe_probability(value)
        ood = model.predict_ood_score(value)
        selected = int(np.argmax(scores))
        if len(scores) > 1:
            ordered = np.sort(scores)
            margin = float(ordered[-1] - ordered[-2])
        else:
            margin = 0.0
        disagreement = _relation_disagreement(
            model,
            model.ranker._standardize(value),
            selected,
        )
        ood_reference = model.ood_threshold
        ood_risk = (
            min(1.0, float(ood[selected]) / max(float(ood_reference), 1e-6))
            if ood_reference is not None
            else 0.0
        )
        margin_risk = 1.0 / (1.0 + max(margin, 0.0))
        probability_risk = 1.0 - float(probabilities[selected])
        risk = max(probability_risk, disagreement, ood_risk, margin_risk)
        return SelectiveRiskEvidence(
            selected_index=selected,
            selected_probability=float(probabilities[selected]),
            score_margin=margin,
            relation_disagreement=disagreement,
            ood_score=float(ood[selected]),
            risk_score=float(np.clip(risk, 0.0, 1.0)),
        )

    def policy_decision(
        self,
        model: SafeCandidateViewRanker,
        features: np.ndarray,
        views: Sequence[str],
        *,
        domain: str | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        base = model.policy_decision(
            features,
            views,
            acquisition_threshold=self.proposal_probability_threshold,
            domain=domain,
            task=task,
        )
        evidence = self.evidence(model, features)
        selected_index = int(evidence.selected_index)
        if self.risk_weights is not None and self.risk_feature_names is not None:
            contextual_risk = self._risk_score_from_record(
                {
                    "legacy_risk_score": evidence.risk_score,
                    "selected_probability": evidence.selected_probability,
                    "score_margin": evidence.score_margin,
                    "relation_disagreement": evidence.relation_disagreement,
                    "ood_score": evidence.ood_score,
                    "task": task or "unspecified",
                    "domain": domain or "unspecified",
                    "view": str(views[selected_index]),
                }
            )
            evidence = SelectiveRiskEvidence(
                selected_index=evidence.selected_index,
                selected_probability=evidence.selected_probability,
                score_margin=evidence.score_margin,
                relation_disagreement=evidence.relation_disagreement,
                ood_score=evidence.ood_score,
                risk_score=contextual_risk,
            )
        result = dict(base)
        result["selective_risk"] = evidence.as_dict()
        result["selective_risk_enabled"] = self.enabled
        if base["action"] != "acquire_view":
            return result
        if not self.enabled or self.risk_threshold is None:
            result.update(
                action="stop_for_review",
                view=None,
                reason="selective_risk_controller_unavailable",
            )
        elif evidence.risk_score > self.risk_threshold:
            result.update(
                action="stop_for_review",
                view=None,
                reason="selective_risk_bound_exceeded",
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        if self.feature_mode is None or self.feature_names is None:
            raise RuntimeError("selective-risk controller has not been fitted")
        return {
            "schema_version": SELECTIVE_RISK_CONTROLLER_SCHEMA,
            "model_type": "finite_sample_selective_risk_veto",
            "unsafe_risk_budget": self.unsafe_risk_budget,
            "confidence_level": self.confidence_level,
            "minimum_all_negative_stop_rate": self.minimum_all_negative_stop_rate,
            "minimum_calibration_states": self.minimum_calibration_states,
            "proposal_probability_threshold": self.proposal_probability_threshold,
            "risk_threshold": self.risk_threshold,
            "enabled": self.enabled,
            "feature_mode": self.feature_mode,
            "feature_names": list(self.feature_names),
            "calibration_report": self.calibration_report,
            "risk_model_mode": self.risk_model_mode,
            "risk_feature_names": list(self.risk_feature_names or ()),
            "risk_feature_mean": (
                None
                if self.risk_feature_mean is None
                else self.risk_feature_mean.tolist()
            ),
            "risk_feature_scale": (
                None
                if self.risk_feature_scale is None
                else self.risk_feature_scale.tolist()
            ),
            "risk_weights": (
                None if self.risk_weights is None else self.risk_weights.tolist()
            ),
            "risk_bias": float(self.risk_bias),
            "risk_fit_state_count": self.risk_fit_state_count,
            "risk_fit_positive_state_count": self.risk_fit_positive_state_count,
            "risk_fit_status": self.risk_fit_status,
            "input_contract": {
                "runtime_features": "sparse_graph_scores_camera_geometry",
                "physical_probe_labels_runtime_visible": False,
                "camera_authority": "disabled_shadow_only",
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectiveRiskController":
        if value.get("schema_version") != SELECTIVE_RISK_CONTROLLER_SCHEMA:
            raise ValueError("unsupported selective-risk controller schema")
        controller = cls(
            unsafe_risk_budget=float(value["unsafe_risk_budget"]),
            confidence_level=float(value["confidence_level"]),
            minimum_all_negative_stop_rate=float(
                value["minimum_all_negative_stop_rate"]
            ),
            minimum_calibration_states=int(value["minimum_calibration_states"]),
            proposal_probability_threshold=(
                None
                if value.get("proposal_probability_threshold") is None
                else float(value["proposal_probability_threshold"])
            ),
        )
        controller.risk_threshold = (
            None
            if value.get("risk_threshold") is None
            else float(value["risk_threshold"])
        )
        if controller.risk_threshold is not None and not (
            np.isfinite(controller.risk_threshold)
            and 0.0 <= controller.risk_threshold <= 1.0
        ):
            raise ValueError("selective-risk threshold must be in [0, 1]")
        controller.enabled = bool(value.get("enabled", False))
        controller.feature_mode = str(value.get("feature_mode", ""))
        controller.feature_names = tuple(str(x) for x in value.get("feature_names", ()))
        controller.calibration_report = (
            dict(value["calibration_report"])
            if isinstance(value.get("calibration_report"), Mapping)
            else None
        )
        controller.risk_model_mode = str(value.get("risk_model_mode", "legacy_max"))
        controller.risk_fit_state_count = int(value.get("risk_fit_state_count", 0))
        controller.risk_fit_positive_state_count = int(
            value.get("risk_fit_positive_state_count", 0)
        )
        controller.risk_fit_status = (
            None
            if value.get("risk_fit_status") is None
            else str(value.get("risk_fit_status"))
        )
        raw_names = value.get("risk_feature_names", ())
        controller.risk_feature_names = tuple(str(x) for x in raw_names)
        raw_mean = value.get("risk_feature_mean")
        raw_scale = value.get("risk_feature_scale")
        raw_weights = value.get("risk_weights")
        if raw_mean is not None and raw_scale is not None and raw_weights is not None:
            controller.risk_feature_mean = np.asarray(raw_mean, dtype=np.float64)
            controller.risk_feature_scale = np.asarray(raw_scale, dtype=np.float64)
            controller.risk_weights = np.asarray(raw_weights, dtype=np.float64)
            if (
                not controller.risk_feature_names
                or controller.risk_feature_mean.ndim != 1
                or controller.risk_feature_mean.shape
                != controller.risk_feature_scale.shape
                or controller.risk_feature_mean.shape != controller.risk_weights.shape
                or len(controller.risk_feature_names)
                != len(controller.risk_feature_mean)
                or np.any(controller.risk_feature_scale <= 0.0)
                or not np.all(np.isfinite(controller.risk_feature_mean))
                or not np.all(np.isfinite(controller.risk_feature_scale))
                or not np.all(np.isfinite(controller.risk_weights))
            ):
                raise ValueError("contextual risk model vectors are invalid")
            controller.risk_bias = float(value.get("risk_bias", 0.0))
            if not np.isfinite(controller.risk_bias):
                raise ValueError("contextual risk model bias is invalid")
        elif controller.risk_model_mode != "legacy_max":
            raise ValueError("contextual risk model is missing fitted vectors")
        if not controller.feature_mode or not controller.feature_names:
            raise ValueError("selective-risk controller feature contract is missing")
        if controller.enabled and controller.risk_threshold is None:
            raise ValueError("enabled selective-risk controller has no threshold")
        return controller


def evaluate_selective_risk_policy(
    model: SafeCandidateViewRanker,
    controller: SelectiveRiskController,
    groups: Sequence[SafeCandidateViewGroup],
) -> dict[str, Any]:
    rows = []
    for group in groups:
        decision = controller.policy_decision(
            model,
            group.features,
            group.views,
            domain=group.domain,
            task=group.task,
        )
        selected = int(decision["selective_risk"]["selected_index"])
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
                "action": decision["action"],
                "reason": decision.get("reason"),
                "selective_risk": decision["selective_risk"],
                "state_has_safe_view": group.has_safe_view,
            }
        )
    acquired = [row for row in rows if row["action"] == "acquire_view"]
    positives = [row for row in rows if row["state_has_safe_view"]]
    negatives = [row for row in rows if not row["state_has_safe_view"]]
    return {
        "state_count": len(rows),
        "positive_state_count": len(positives),
        "all_negative_state_count": len(negatives),
        "acquired_state_count": len(acquired),
        "acquisition_coverage": len(acquired) / len(rows) if rows else None,
        "acquired_safe_precision": (
            sum(row["selected_safe_candidate_available"] for row in acquired)
            / len(acquired)
            if acquired
            else None
        ),
        "positive_state_safe_acquisition_rate": (
            sum(
                row["action"] == "acquire_view"
                and row["selected_safe_candidate_available"]
                for row in positives
            )
            / len(positives)
            if positives
            else None
        ),
        "all_negative_stop_rate": (
            sum(row["action"] == "stop_for_review" for row in negatives)
            / len(negatives)
            if negatives
            else None
        ),
        "rows": rows,
    }


def clopper_pearson_upper_bound(
    failures: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Exact one-sided Clopper--Pearson upper bound for Bernoulli risk.

    The implementation uses a monotone binomial CDF and bisection, avoiding a
    mandatory scipy dependency while retaining the exact finite-sample
    guarantee.  For zero failures this reduces to ``1-alpha**(1/n)``.
    """

    if failures < 0 or trials < 0 or failures > trials:
        raise ValueError("invalid Bernoulli counts")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")
    if trials == 0:
        return 1.0
    alpha = 1.0 - confidence_level
    if failures == 0:
        return float(1.0 - alpha ** (1.0 / trials))
    if failures >= trials:
        return 1.0
    low, high = failures / trials, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(failures, trials, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return float(high)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if k >= n else 0.0
    complement = 1.0 - probability
    return float(
        np.clip(
        sum(
            math.comb(n, index)
            * probability**index
            * complement ** (n - index)
            for index in range(k + 1)
        ),
        0.0,
        1.0,
        )
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _relation_disagreement(
    model: SafeCandidateViewRanker,
    standardized: np.ndarray,
    selected: int,
) -> float:
    names = model.feature_names
    weights = np.asarray(model.ranker.weights, dtype=np.float64)
    relations = {}
    for index, name in enumerate(names):
        marker = "_x_"
        if marker not in name or not name.startswith("current_uncertainty_"):
            continue
        relation, geometry = name.split(marker, 1)
        relations.setdefault(relation, []).append(index)
    if not relations:
        return 0.0
    disagreements = 0
    for indices in relations.values():
        contribution = standardized[:, indices] @ weights[indices]
        if int(np.argmax(contribution)) != selected:
            disagreements += 1
    return float(disagreements / len(relations))


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
