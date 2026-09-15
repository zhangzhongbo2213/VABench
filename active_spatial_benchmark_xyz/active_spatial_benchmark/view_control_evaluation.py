"""Calibration and evaluation helpers for active view control.

The calibration estimates the effective image-plane and axial error of a
learned 3D grasp-frame observation model.  It intentionally includes median
bias in the reported noise scale: a systematic error is not free information
that a Fisher controller may assume another view will remove.

All policy metrics consume training/evaluation-only realized utility labels.
The selected policy itself must be computed from ``inference_visible`` fields;
the helpers keep target lookup separate so callers can audit that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VIEW_SENSOR_CALIBRATION_SCHEMA = "spatial.view_sensor_calibration.v1"
ACTIVE_VIEW_BENCHMARK_SCHEMA = "spatial.active_view_policy_benchmark.v1"


@dataclass(frozen=True)
class ViewResidual:
    """One grasp-frame point residual in calibrated camera coordinates."""

    pixel_error_x: float
    pixel_error_y: float
    depth_error_m: float
    camera_depth_m: float
    focal_length_px: float
    sample_id: str = ""
    view: str = ""
    semantic_type: str = ""

    def __post_init__(self) -> None:
        values = (
            self.pixel_error_x,
            self.pixel_error_y,
            self.depth_error_m,
            self.camera_depth_m,
            self.focal_length_px,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("view residual values must be finite")
        if self.camera_depth_m <= 0.0 or self.focal_length_px <= 0.0:
            raise ValueError("camera depth and focal length must be positive")


def fit_view_sensor_calibration(
    residuals: Sequence[ViewResidual],
    *,
    minimum_point_samples: int = 60,
    near_isotropic_ratio: float = 1.25,
) -> dict[str, Any]:
    """Fit a conservative effective sensor model from held-out residuals.

    The 68.27th absolute-error percentile is used as a robust one-sigma scale.
    X/Y use the larger component scale, avoiding a favorable result caused by
    one easy image axis.  Bias, RMSE and tail error remain visible in the
    report and can be used for stricter downstream gates.
    """

    if minimum_point_samples < 1:
        raise ValueError("minimum point sample count must be positive")
    if near_isotropic_ratio <= 1.0:
        raise ValueError("near-isotropic ratio must be greater than one")
    rows = tuple(residuals)
    if not rows:
        raise ValueError("sensor calibration requires residual samples")
    x = np.asarray([row.pixel_error_x for row in rows], dtype=np.float64)
    y = np.asarray([row.pixel_error_y for row in rows], dtype=np.float64)
    depth = np.asarray([row.depth_error_m for row in rows], dtype=np.float64)
    ranges = np.asarray([row.camera_depth_m for row in rows], dtype=np.float64)
    focal = np.asarray([row.focal_length_px for row in rows], dtype=np.float64)
    keypoint_x_std = _absolute_quantile_scale(x)
    keypoint_y_std = _absolute_quantile_scale(y)
    keypoint_std_px = max(keypoint_x_std, keypoint_y_std)
    depth_std_m = _absolute_quantile_scale(depth)
    median_range_m = float(np.median(ranges))
    median_focal_px = float(np.median(focal))
    lateral_std_m = median_range_m * keypoint_std_px / median_focal_px
    lower = min(lateral_std_m, depth_std_m)
    anisotropy_ratio = max(lateral_std_m, depth_std_m) / max(lower, 1e-12)
    enough_samples = len(rows) >= minimum_point_samples
    directional_signal = anisotropy_ratio >= near_isotropic_ratio
    reasons = []
    if not enough_samples:
        reasons.append(
            f"point_sample_count={len(rows)} is below minimum={minimum_point_samples}"
        )
    if not directional_signal:
        reasons.append(
            "effective lateral and axial errors are nearly isotropic at the "
            "median calibration range"
        )
    result = {
        "schema_version": VIEW_SENSOR_CALIBRATION_SCHEMA,
        "access": "training_only_calibration",
        "estimator": {
            "name": "absolute_error_quantile_effective_sigma",
            "quantile": 0.6827,
            "includes_systematic_bias": True,
            "image_axis_reduction": "max(x_scale, y_scale)",
        },
        "point_sample_count": len(rows),
        "unique_sample_count": len({row.sample_id for row in rows if row.sample_id}),
        "view_counts": _count_strings(row.view for row in rows),
        "semantic_type_counts": _count_strings(
            row.semantic_type for row in rows
        ),
        "sensor_model": {
            "keypoint_std_px": keypoint_std_px,
            "depth_std_m": depth_std_m,
            "keypoint_x_std_px": keypoint_x_std,
            "keypoint_y_std_px": keypoint_y_std,
        },
        "residual_statistics": {
            "pixel_x": _residual_statistics(x),
            "pixel_y": _residual_statistics(y),
            "pixel_radial": _residual_statistics(np.hypot(x, y)),
            "depth_m": _residual_statistics(depth),
        },
        "camera_statistics": {
            "range_m": _distribution_statistics(ranges),
            "focal_length_px": _distribution_statistics(focal),
        },
        "directional_signal_audit": {
            "median_range_m": median_range_m,
            "median_focal_length_px": median_focal_px,
            "effective_lateral_std_m": lateral_std_m,
            "effective_axial_std_m": depth_std_m,
            "measurement_noise_anisotropy_ratio": anisotropy_ratio,
            "required_ratio": near_isotropic_ratio,
            "directional_signal_available": directional_signal,
        },
        "stratified_sensor_models": {
            "by_view": _fit_strata(
                rows,
                key=lambda row: row.view,
                minimum_point_samples=max(3, minimum_point_samples // 4),
                near_isotropic_ratio=near_isotropic_ratio,
            ),
            "by_semantic_type": _fit_strata(
                rows,
                key=lambda row: row.semantic_type,
                minimum_point_samples=max(3, minimum_point_samples // 4),
                near_isotropic_ratio=near_isotropic_ratio,
            ),
        },
        "eligibility": {
            "eligible_for_analytic_control": bool(
                enough_samples and directional_signal
            ),
            "minimum_point_samples": int(minimum_point_samples),
            "reasons": reasons,
        },
    }
    return result


def selected_policy_metrics(
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-state selections against realized utility targets."""

    valid = [row for row in selections if row.get("selected_view") is not None]
    regrets = [float(row["regret"]) for row in valid]
    utilities = [float(row["selected_utility"]) for row in valid]
    top1 = [bool(row["top1_correct"]) for row in valid]
    return {
        "state_count": len(selections),
        "selected_state_count": len(valid),
        "selection_coverage": len(valid) / max(len(selections), 1),
        "top1_accuracy": float(np.mean(top1)) if top1 else None,
        "mean_view_regret": float(np.mean(regrets)) if regrets else None,
        "median_view_regret": float(np.median(regrets)) if regrets else None,
        "p95_view_regret": float(np.quantile(regrets, 0.95)) if regrets else None,
        "mean_selected_utility": float(np.mean(utilities)) if utilities else None,
        "selected_view_counts": _count_strings(
            str(row["selected_view"]) for row in valid
        ),
    }


def exact_random_policy_metrics(
    utility_by_state: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Compute the exact uniform-random expectation without Monte Carlo noise."""

    expected_regrets = []
    expected_utilities = []
    top1_probabilities = []
    candidate_counts = []
    for utility in utility_by_state:
        values = np.asarray(list(utility.values()), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            continue
        best = float(np.max(values))
        expected_regrets.append(float(np.mean(best - values)))
        expected_utilities.append(float(np.mean(values)))
        top1_probabilities.append(float(np.mean(np.isclose(values, best))))
        candidate_counts.append(int(values.size))
    return {
        "state_count": len(utility_by_state),
        "selected_state_count": len(expected_regrets),
        "selection_coverage": len(expected_regrets) / max(len(utility_by_state), 1),
        "top1_accuracy": (
            float(np.mean(top1_probabilities)) if top1_probabilities else None
        ),
        "mean_view_regret": (
            float(np.mean(expected_regrets)) if expected_regrets else None
        ),
        "median_view_regret": (
            float(np.median(expected_regrets)) if expected_regrets else None
        ),
        "p95_view_regret": (
            float(np.quantile(expected_regrets, 0.95))
            if expected_regrets
            else None
        ),
        "mean_selected_utility": (
            float(np.mean(expected_utilities)) if expected_utilities else None
        ),
        "mean_candidate_count": (
            float(np.mean(candidate_counts)) if candidate_counts else None
        ),
        "selected_view_counts": {},
    }


def analytic_deployment_gate(
    *,
    calibration: Mapping[str, Any],
    analytic_metrics: Mapping[str, Any] | None,
    random_metrics: Mapping[str, Any],
    fixed_metrics: Mapping[str, Mapping[str, Any]],
    evaluated_task_count: int,
    minimum_states: int = 10,
    minimum_tasks: int = 5,
    minimum_regret_improvement: float = 1e-4,
    minimum_selected_view_count: int = 2,
) -> dict[str, Any]:
    """Decide whether analytic control may replace shadow logging.

    Passing requires calibrated directional signal and strict improvement over
    both exact random and the best fixed-view baseline on the held-out split.
    It also rejects a policy that collapses to one fixed view.
    """

    if minimum_regret_improvement <= 0.0:
        raise ValueError("minimum regret improvement must be positive")
    if minimum_selected_view_count < 2:
        raise ValueError("deployment gate must require at least two selected views")

    checks: list[dict[str, Any]] = []
    eligible = bool(
        calibration.get("eligibility", {}).get(
            "eligible_for_analytic_control", False
        )
    )
    checks.append(_gate_check("calibrated_directional_signal", eligible))
    state_count = int((analytic_metrics or {}).get("state_count", 0))
    checks.append(_gate_check("minimum_evaluation_states", state_count >= minimum_states))
    checks.append(
        _gate_check("minimum_evaluation_tasks", evaluated_task_count >= minimum_tasks)
    )
    if analytic_metrics is None or analytic_metrics.get("mean_view_regret") is None:
        checks.append(_gate_check("analytic_policy_produced_selections", False))
    else:
        analytic_regret = float(analytic_metrics["mean_view_regret"])
        random_regret = float(random_metrics["mean_view_regret"])
        fixed_regrets = [
            float(metrics["mean_view_regret"])
            for metrics in fixed_metrics.values()
            if metrics.get("mean_view_regret") is not None
        ]
        best_fixed_regret = min(fixed_regrets) if fixed_regrets else float("inf")
        checks.append(
            _gate_check(
                "strictly_outperforms_uniform_random",
                analytic_regret
                <= random_regret - minimum_regret_improvement,
                analytic=analytic_regret,
                baseline=random_regret,
                required_improvement=minimum_regret_improvement,
            )
        )
        checks.append(
            _gate_check(
                "strictly_outperforms_best_fixed_view",
                analytic_regret
                <= best_fixed_regret - minimum_regret_improvement,
                analytic=analytic_regret,
                baseline=best_fixed_regret,
                required_improvement=minimum_regret_improvement,
            )
        )
        checks.append(
            _gate_check(
                "full_selection_coverage",
                float(analytic_metrics.get("selection_coverage", 0.0)) >= 1.0,
            )
        )
        selected_view_count = len(
            analytic_metrics.get("selected_view_counts", {})
        )
        checks.append(
            _gate_check(
                "does_not_collapse_to_one_fixed_view",
                selected_view_count >= minimum_selected_view_count,
                selected_view_count=selected_view_count,
                required_selected_view_count=minimum_selected_view_count,
            )
        )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "status": "pass" if passed else "blocked",
        "control_mode": "analytic" if passed else "shadow",
        "checks": checks,
        "policy": (
            "analytic may control candidate ordering on the evaluated scope"
            if passed
            else "analytic remains shadow-only; legacy gated control is unchanged"
        ),
    }


def learned_deployment_gate(
    *,
    learned_metrics: Mapping[str, Any] | None,
    random_metrics: Mapping[str, Any],
    fixed_metrics: Mapping[str, Mapping[str, Any]],
    evaluated_task_count: int,
    minimum_states: int = 10,
    minimum_tasks: int = 5,
    minimum_regret_improvement: float = 1e-4,
    minimum_selected_view_count: int = 2,
    checkpoint_load_error: str | None = None,
) -> dict[str, Any]:
    """Gate a learned ranker on a held-out realized-utility split.

    A learned checkpoint is eligible only when it produces runtime selections
    from inference-visible graph features and beats both exact random and the
    best fixed view. The distinct-view check prevents a ranker from being
    reported as active perception after collapsing to one default camera.
    """

    if minimum_regret_improvement <= 0.0:
        raise ValueError("minimum regret improvement must be positive")
    if minimum_selected_view_count < 2:
        raise ValueError("deployment gate must require at least two selected views")

    checks: list[dict[str, Any]] = []
    checks.append(
        _gate_check(
            "learned_checkpoint_loaded",
            learned_metrics is not None and checkpoint_load_error is None,
            error=checkpoint_load_error,
        )
    )
    state_count = int((learned_metrics or {}).get("state_count", 0))
    checks.append(_gate_check("minimum_evaluation_states", state_count >= minimum_states))
    checks.append(
        _gate_check("minimum_evaluation_tasks", evaluated_task_count >= minimum_tasks)
    )
    if learned_metrics is None or learned_metrics.get("mean_view_regret") is None:
        checks.append(_gate_check("learned_policy_produced_selections", False))
    else:
        learned_regret = float(learned_metrics["mean_view_regret"])
        random_regret = float(random_metrics["mean_view_regret"])
        fixed_regrets = [
            float(metrics["mean_view_regret"])
            for metrics in fixed_metrics.values()
            if metrics.get("mean_view_regret") is not None
        ]
        best_fixed_regret = min(fixed_regrets) if fixed_regrets else float("inf")
        checks.append(
            _gate_check(
                "strictly_outperforms_uniform_random",
                learned_regret <= random_regret - minimum_regret_improvement,
                learned=learned_regret,
                baseline=random_regret,
                required_improvement=minimum_regret_improvement,
            )
        )
        checks.append(
            _gate_check(
                "strictly_outperforms_best_fixed_view",
                learned_regret <= best_fixed_regret - minimum_regret_improvement,
                learned=learned_regret,
                baseline=best_fixed_regret,
                required_improvement=minimum_regret_improvement,
            )
        )
        checks.append(
            _gate_check(
                "full_selection_coverage",
                float(learned_metrics.get("selection_coverage", 0.0)) >= 1.0,
            )
        )
        selected_view_count = len(learned_metrics.get("selected_view_counts", {}))
        checks.append(
            _gate_check(
                "does_not_collapse_to_one_fixed_view",
                selected_view_count >= minimum_selected_view_count,
                selected_view_count=selected_view_count,
                required_selected_view_count=minimum_selected_view_count,
            )
        )
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "status": "pass" if passed else "blocked",
        "control_mode": "learned_gated" if passed else "shadow",
        "checks": checks,
        "policy": (
            "learned ranker may order candidate views on the evaluated scope"
            if passed
            else "learned ranker remains shadow-only; legacy gated control is unchanged"
        ),
    }


def guarded_learned_view(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_view: str,
    override_margin: float,
) -> tuple[str | None, dict[str, Any]]:
    """Apply an explicit score-margin fallback to a learned proposal.

    The guard is intentionally score-space only; it does not inspect realized
    utility. If the baseline is unavailable or rows are malformed, it fails
    closed with no learned override.
    """

    if override_margin < 0.0:
        raise ValueError("override margin must be non-negative")
    if not ranked_rows:
        return None, {"status": "failed_closed", "reason": "empty_ranked_rows"}
    by_view = {
        str(row.get("view")): row
        for row in ranked_rows
        if row.get("view") is not None
    }
    if baseline_view not in by_view:
        return None, {
            "status": "failed_closed",
            "reason": "baseline_view_missing",
            "baseline_view": baseline_view,
        }
    top = max(ranked_rows, key=lambda row: float(row["learned_score"]))
    top_view = str(top["view"])
    baseline_score = float(by_view[baseline_view]["learned_score"])
    top_score = float(top["learned_score"])
    score_margin = top_score - baseline_score
    override = top_view != baseline_view and score_margin >= override_margin
    selected = top_view if override else baseline_view
    return selected, {
        "status": "override" if override else "fallback",
        "baseline_view": baseline_view,
        "proposal_view": top_view,
        "selected_view": selected,
        "proposal_score": top_score,
        "baseline_score": baseline_score,
        "score_margin": score_margin,
        "override_margin": float(override_margin),
    }


def selection_record(
    *,
    sample_id: str,
    task: str,
    selected_view: str | None,
    utility_by_view: Mapping[str, float],
    reason: str,
) -> dict[str, Any]:
    """Build one auditable policy selection row."""

    if not utility_by_view:
        raise ValueError("selection record requires candidate utilities")
    oracle_utility = max(float(value) for value in utility_by_view.values())
    oracle_views = sorted(
        view
        for view, value in utility_by_view.items()
        if np.isclose(float(value), oracle_utility)
    )
    if selected_view is None or selected_view not in utility_by_view:
        return {
            "sample_id": sample_id,
            "task": task,
            "selected_view": None,
            "selected_utility": None,
            "oracle_views": oracle_views,
            "oracle_utility": oracle_utility,
            "regret": None,
            "top1_correct": False,
            "reason": reason,
        }
    selected_utility = float(utility_by_view[selected_view])
    return {
        "sample_id": sample_id,
        "task": task,
        "selected_view": selected_view,
        "selected_utility": selected_utility,
        "oracle_views": oracle_views,
        "oracle_utility": oracle_utility,
        "regret": oracle_utility - selected_utility,
        "top1_correct": selected_view in oracle_views,
        "reason": reason,
    }


def _absolute_quantile_scale(values: np.ndarray) -> float:
    return float(np.quantile(np.abs(values), 0.6827))


def _fit_strata(
    rows: Sequence[ViewResidual],
    *,
    key: Any,
    minimum_point_samples: int,
    near_isotropic_ratio: float,
) -> dict[str, Any]:
    """Summarize calibration strata without changing the global control gate."""

    grouped: dict[str, list[ViewResidual]] = {}
    for row in rows:
        name = str(key(row))
        if name:
            grouped.setdefault(name, []).append(row)
    result: dict[str, Any] = {}
    for name, group in sorted(grouped.items()):
        x = np.asarray([row.pixel_error_x for row in group], dtype=np.float64)
        y = np.asarray([row.pixel_error_y for row in group], dtype=np.float64)
        depth = np.asarray([row.depth_error_m for row in group], dtype=np.float64)
        ranges = np.asarray([row.camera_depth_m for row in group], dtype=np.float64)
        focal = np.asarray([row.focal_length_px for row in group], dtype=np.float64)
        keypoint_x_std = _absolute_quantile_scale(x)
        keypoint_y_std = _absolute_quantile_scale(y)
        keypoint_std = max(keypoint_x_std, keypoint_y_std)
        depth_std = _absolute_quantile_scale(depth)
        lateral_std = float(np.median(ranges)) * keypoint_std / float(np.median(focal))
        lower = min(lateral_std, depth_std)
        ratio = max(lateral_std, depth_std) / max(lower, 1e-12)
        result[name] = {
            "point_sample_count": len(group),
            "sensor_model": {
                "keypoint_std_px": keypoint_std,
                "depth_std_m": depth_std,
                "keypoint_x_std_px": keypoint_x_std,
                "keypoint_y_std_px": keypoint_y_std,
            },
            "median_range_m": float(np.median(ranges)),
            "median_focal_length_px": float(np.median(focal)),
            "effective_lateral_std_m": lateral_std,
            "effective_axial_std_m": depth_std,
            "measurement_noise_anisotropy_ratio": ratio,
            "directional_signal_available": bool(
                len(group) >= minimum_point_samples and ratio >= near_isotropic_ratio
            ),
            "minimum_point_samples": int(minimum_point_samples),
        }
    return result


def _residual_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_about_mean": float(np.std(values)),
        "rmse_about_zero": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(np.abs(values))),
        "absolute_p68": _absolute_quantile_scale(values),
        "absolute_p95": float(np.quantile(np.abs(values), 0.95)),
    }


def _distribution_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _count_strings(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _gate_check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}
