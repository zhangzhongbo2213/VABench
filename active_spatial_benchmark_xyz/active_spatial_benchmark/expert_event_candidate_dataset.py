"""Merge physical candidate probes into training-only event samples."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .expert_grasp_trace import validate_expert_grasp_event_sample
from .grasp_candidate_dataset import (
    CANDIDATE_SAMPLE_SCHEMA,
    TASK_NATIVE_NEGATIVE_KINDS,
    validate_candidate_sample,
)
from .grasp_candidates import GraspIntent


LABELLED_EVENT_SAMPLE_SCHEMA = "spatial.labelled_expert_grasp_event_sample.v1"
AGGREGATED_LABELLED_EVENT_SAMPLE_SCHEMA = (
    "spatial.labelled_expert_grasp_event_sample.v2"
)
SUPPORTED_LABELLED_EVENT_SAMPLE_SCHEMAS = {
    LABELLED_EVENT_SAMPLE_SCHEMA,
    AGGREGATED_LABELLED_EVENT_SAMPLE_SCHEMA,
}
REQUIRED_LABEL_FIELDS = {
    "ik_reachable",
    "close_planner_success",
    "collision_free_at_pregrasp_and_grasp_endpoints",
    "bilateral_contact_after_close",
    "task_success",
    "task_native_execution_success",
    "task_native_safe_success",
}
AGGREGATED_BOOLEAN_LABEL_FIELDS = REQUIRED_LABEL_FIELDS | {
    "collision_within_expert_baseline",
    "strict_endpoint_safe_success",
}


def merge_candidate_probe_reports(
    sample: Mapping[str, Any],
    probe_reports: Sequence[Mapping[str, Any]],
    *,
    sample_sha256: str,
    report_references: Sequence[str] | None = None,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    errors = validate_expert_grasp_event_sample(sample)
    if errors:
        raise ValueError("invalid source event sample: " + "; ".join(errors))
    candidates = sample["training_only"]["candidate_perturbations"]
    candidate_by_index = {index: value for index, value in enumerate(candidates)}
    labels: dict[int, dict[str, Any]] = {}
    references = list(report_references or [""] * len(probe_reports))
    if len(references) != len(probe_reports):
        raise ValueError("report_references must align with probe_reports")
    for report, reference in zip(probe_reports, references):
        if report.get("schema_version") != "phase11.expert_event_candidate_probe.v1":
            raise ValueError("unsupported candidate probe report schema")
        settings = report.get("settings")
        if not isinstance(settings, Mapping) or settings.get("sample_sha256") != sample_sha256:
            raise ValueError("candidate probe report references a different source sample")
        for row in report.get("results", []):
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                continue
            index = int(row["candidate_index"])
            if index not in candidate_by_index:
                raise ValueError(f"probe label references unknown candidate index {index}")
            expected_id = candidate_by_index[index]["id"]
            if row.get("candidate_id") != expected_id:
                raise ValueError("probe candidate ID does not match source sample")
            label = row.get("label")
            if not isinstance(label, Mapping):
                raise ValueError("complete probe row has no label")
            missing = sorted(REQUIRED_LABEL_FIELDS - set(label))
            if missing:
                raise ValueError("probe label is missing fields: " + ", ".join(missing))
            merged_row = {
                "candidate_index": index,
                "candidate_id": expected_id,
                "perturbation_kind": row.get("perturbation_kind"),
                "label": dict(label),
                "probe_report": reference,
                "probe_result": row.get("result"),
                "access": "oracle/training_only",
            }
            previous = labels.get(index)
            if previous is not None and previous["label"] != merged_row["label"]:
                raise ValueError(f"conflicting physical labels for candidate {index}")
            labels[index] = merged_row

    labelled_indices = sorted(labels)
    unlabelled_indices = sorted(set(candidate_by_index) - set(labelled_indices))
    if require_complete_coverage and unlabelled_indices:
        raise ValueError(
            "candidate execution labels are incomplete: "
            + ", ".join(str(index) for index in unlabelled_indices)
        )
    result = deepcopy(dict(sample))
    result["schema_version"] = LABELLED_EVENT_SAMPLE_SCHEMA
    result["source_sample"] = {
        "schema_version": sample["schema_version"],
        "sha256": sample_sha256,
    }
    training = result["training_only"]
    training["candidate_execution_labels"] = [
        labels[index] for index in labelled_indices
    ]
    training["label_coverage"] = {
        "candidate_count": len(candidates),
        "labelled_candidate_count": len(labelled_indices),
        "unlabelled_candidate_indices": unlabelled_indices,
        "complete": not unlabelled_indices,
    }
    training["label_generation"][
        "candidate_execution_labels"
    ] = "deterministic_expert_event_replay_candidate_override"
    validation_errors = validate_labelled_event_sample(result)
    if validation_errors:
        raise ValueError("invalid labelled event sample: " + "; ".join(validation_errors))
    return result


def validate_labelled_event_sample(sample: Mapping[str, Any]) -> list[str]:
    errors = []
    schema_version = sample.get("schema_version")
    if schema_version not in SUPPORTED_LABELLED_EVENT_SAMPLE_SCHEMAS:
        errors.append("unsupported labelled sample schema_version")
    inference = sample.get("inference_visible")
    training = sample.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        errors.append("labelled sample requires inference_visible and training_only")
        return errors
    if "candidate_execution_labels" in inference:
        errors.append("candidate execution labels leaked into inference_visible")
    labels = training.get("candidate_execution_labels")
    coverage = training.get("label_coverage")
    if not isinstance(labels, list):
        errors.append("candidate_execution_labels must be a list")
    if not isinstance(coverage, Mapping):
        errors.append("label_coverage must be a mapping")
    elif isinstance(labels, list):
        if int(coverage.get("labelled_candidate_count", -1)) != len(labels):
            errors.append("label coverage count does not match labels")
        complete = not coverage.get("unlabelled_candidate_indices")
        if bool(coverage.get("complete")) != complete:
            errors.append("label coverage complete flag is inconsistent")
        if schema_version == AGGREGATED_LABELLED_EVENT_SAMPLE_SCHEMA:
            for row in labels:
                label = row.get("label", {}) if isinstance(row, Mapping) else {}
                trial_count = int(label.get("trial_count", 0))
                probabilities = label.get("probabilities")
                unstable = label.get("unstable_boolean_fields")
                if trial_count < 1:
                    errors.append("aggregated labels require trial_count >= 1")
                if not isinstance(probabilities, Mapping):
                    errors.append("aggregated labels require probabilities")
                if not isinstance(unstable, list):
                    errors.append("aggregated labels require unstable_boolean_fields")
                else:
                    for field_name in unstable:
                        if label.get(field_name) is not None:
                            errors.append(
                                "unstable aggregated boolean fields must have null hard labels"
                            )
    return errors


def aggregate_candidate_probe_reports(
    sample: Mapping[str, Any],
    probe_reports: Sequence[Mapping[str, Any]],
    *,
    sample_sha256: str,
    report_references: Sequence[str] | None = None,
    minimum_trials_per_candidate: int = 2,
) -> dict[str, Any]:
    """Aggregate repeated physical trials without inventing stable hard labels."""

    errors = validate_expert_grasp_event_sample(sample)
    if errors:
        raise ValueError("invalid source event sample: " + "; ".join(errors))
    if minimum_trials_per_candidate < 1:
        raise ValueError("minimum_trials_per_candidate must be positive")
    candidates = sample["training_only"]["candidate_perturbations"]
    candidate_by_index = {index: value for index, value in enumerate(candidates)}
    references = list(report_references or [""] * len(probe_reports))
    if len(references) != len(probe_reports):
        raise ValueError("report_references must align with probe_reports")
    trials_by_index: dict[int, list[tuple[Mapping[str, Any], str]]] = {
        index: [] for index in candidate_by_index
    }
    for report, reference in zip(probe_reports, references):
        if report.get("schema_version") != "phase11.expert_event_candidate_probe.v1":
            raise ValueError("unsupported candidate probe report schema")
        settings = report.get("settings")
        if not isinstance(settings, Mapping) or settings.get("sample_sha256") != sample_sha256:
            raise ValueError("candidate probe report references a different source sample")
        seen_in_report: set[int] = set()
        for row in report.get("results", []):
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                continue
            index = int(row["candidate_index"])
            if index in seen_in_report:
                raise ValueError(f"probe report repeats candidate index {index}")
            seen_in_report.add(index)
            if index not in candidate_by_index:
                raise ValueError(f"probe label references unknown candidate index {index}")
            expected_id = candidate_by_index[index]["id"]
            if row.get("candidate_id") != expected_id:
                raise ValueError("probe candidate ID does not match source sample")
            label = row.get("label")
            if not isinstance(label, Mapping):
                raise ValueError("complete probe row has no label")
            missing = sorted(REQUIRED_LABEL_FIELDS - set(label))
            if missing:
                raise ValueError("probe label is missing fields: " + ", ".join(missing))
            trials_by_index[index].append((row, reference))

    insufficient = [
        index
        for index, trials in trials_by_index.items()
        if len(trials) < minimum_trials_per_candidate
    ]
    if insufficient:
        raise ValueError(
            "candidate repetition coverage is incomplete: "
            + ", ".join(str(index) for index in insufficient)
        )
    aggregated_rows = []
    uncertain_indices = []
    for index in sorted(candidate_by_index):
        trials = trials_by_index[index]
        raw_labels = [row["label"] for row, _ in trials]
        hard_labels: dict[str, Any] = {}
        probabilities: dict[str, float] = {}
        unstable = []
        for field_name in sorted(AGGREGATED_BOOLEAN_LABEL_FIELDS):
            values = [label.get(field_name) for label in raw_labels]
            if any(value is None for value in values):
                hard_labels[field_name] = None
                unstable.append(field_name)
                continue
            booleans = [bool(value) for value in values]
            probabilities[field_name] = sum(booleans) / float(len(booleans))
            if all(value == booleans[0] for value in booleans):
                hard_labels[field_name] = booleans[0]
            else:
                hard_labels[field_name] = None
                unstable.append(field_name)
        impulses = _finite_numbers(
            label.get("max_non_target_contact_impulse") for label in raw_labels
        )
        baselines = _finite_numbers(
            label.get("expert_contact_impulse_baseline") for label in raw_labels
        )
        limits = _finite_numbers(
            label.get("expert_contact_impulse_limit") for label in raw_labels
        )
        if unstable:
            uncertain_indices.append(index)
        first_row = trials[0][0]
        label = {
            **hard_labels,
            "collision_audit_scope": raw_labels[0].get("collision_audit_scope"),
            "max_non_target_contact_impulse": max(impulses, default=None),
            "expert_contact_impulse_baseline": (
                float(np.mean(baselines)) if baselines else None
            ),
            "expert_contact_impulse_limit": (
                float(np.mean(limits)) if limits else None
            ),
            "trial_count": len(trials),
            "probabilities": probabilities,
            "unstable_boolean_fields": unstable,
            "impulse_statistics": _numeric_statistics(impulses),
            "source": "repeated_deterministic_event_replay_aggregation",
        }
        aggregated_rows.append(
            {
                "candidate_index": index,
                "candidate_id": candidate_by_index[index]["id"],
                "perturbation_kind": first_row.get("perturbation_kind"),
                "label": label,
                "probe_reports": [reference for _, reference in trials],
                "probe_results": [row.get("result") for row, _ in trials],
                "access": "oracle/training_only",
            }
        )
    result = deepcopy(dict(sample))
    result["schema_version"] = AGGREGATED_LABELLED_EVENT_SAMPLE_SCHEMA
    result["source_sample"] = {
        "schema_version": sample["schema_version"],
        "sha256": sample_sha256,
    }
    training = result["training_only"]
    training["candidate_execution_labels"] = aggregated_rows
    training["label_coverage"] = {
        "candidate_count": len(candidates),
        "labelled_candidate_count": len(aggregated_rows),
        "unlabelled_candidate_indices": [],
        "complete": True,
        "minimum_trials_per_candidate": int(minimum_trials_per_candidate),
        "uncertain_candidate_indices": uncertain_indices,
    }
    training["label_generation"][
        "candidate_execution_labels"
    ] = "repeated_deterministic_event_replay_aggregation"
    validation_errors = validate_labelled_event_sample(result)
    if validation_errors:
        raise ValueError("invalid aggregated event sample: " + "; ".join(validation_errors))
    return result


def load_and_merge_candidate_probe_reports(
    sample_path: str | Path,
    report_paths: Sequence[str | Path],
    *,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    resolved_sample = Path(sample_path).resolve()
    sample_bytes = resolved_sample.read_bytes()
    sample = json.loads(sample_bytes)
    resolved_reports = [Path(path).resolve() for path in report_paths]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in resolved_reports]
    return merge_candidate_probe_reports(
        sample,
        reports,
        sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),
        report_references=[str(path) for path in resolved_reports],
        require_complete_coverage=require_complete_coverage,
    )


def load_and_aggregate_candidate_probe_reports(
    sample_path: str | Path,
    report_paths: Sequence[str | Path],
    *,
    minimum_trials_per_candidate: int = 2,
) -> dict[str, Any]:
    resolved_sample = Path(sample_path).resolve()
    sample_bytes = resolved_sample.read_bytes()
    sample = json.loads(sample_bytes)
    resolved_reports = [Path(path).resolve() for path in report_paths]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in resolved_reports]
    return aggregate_candidate_probe_reports(
        sample,
        reports,
        sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),
        report_references=[str(path) for path in resolved_reports],
        minimum_trials_per_candidate=minimum_trials_per_candidate,
    )


def convert_labelled_event_to_candidate_sample(
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert physical event probes to the ranker's leakage-safe schema.

    Metric candidate frames and execution outcomes remain training-only.  The
    inference side contains the same RGB-D evidence and the semantic intent
    that a VLM supplies at deployment.
    """

    errors = validate_labelled_event_sample(sample)
    if errors:
        raise ValueError("invalid labelled event sample: " + "; ".join(errors))
    training = sample["training_only"]
    coverage = training["label_coverage"]
    if coverage.get("complete") is not True:
        raise ValueError("ranker conversion requires complete physical label coverage")
    candidates = training.get("candidate_perturbations", ())
    labels = training.get("candidate_execution_labels", ())
    label_by_id = {str(row["candidate_id"]): row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("candidate execution labels contain duplicate IDs")
    rows = []
    excluded_candidate_ids = []
    for raw_candidate in candidates:
        candidate = _candidate_with_relative_frame_features(raw_candidate)
        candidate_id = str(candidate.get("id", ""))
        if candidate_id not in label_by_id:
            raise ValueError(f"candidate {candidate_id!r} has no physical label")
        physical = label_by_id[candidate_id]
        label = physical["label"]
        perturbation_kind = candidate.get("generation_parameters", {}).get(
            "perturbation_kind"
        )
        if (
            perturbation_kind == "opening_width_offset"
            and label.get("opening_width_command_applied") is not True
        ):
            excluded_candidate_ids.append(candidate_id)
            continue
        negative_kind = str(
            physical.get("perturbation_kind")
            or candidate.get("generation_parameters", {}).get(
                "perturbation_kind", "none"
            )
        )
        if negative_kind not in TASK_NATIVE_NEGATIVE_KINDS:
            raise ValueError(
                f"unsupported physical perturbation kind {negative_kind!r}"
            )
        rows.append(
            {
                "candidate": candidate,
                "execution": {
                    "candidate_id": candidate_id,
                    "negative_kind": negative_kind,
                    "ik_reachable": label.get("ik_reachable"),
                    "collision_free": label.get(
                        "collision_free_at_pregrasp_and_grasp_endpoints"
                    ),
                    "close_planner_success": label.get("close_planner_success"),
                    "lift_planner_success": None,
                    "bilateral_contact": label.get(
                        "bilateral_contact_after_close"
                    ),
                    "lifted_from_support": None,
                    "task_success": label.get("task_success"),
                    "execution_success": label.get(
                        "task_native_execution_success"
                    ),
                    "task_native_safe_success": label.get(
                        "task_native_safe_success"
                    ),
                    "task_native_execution_success_probability": label.get(
                        "probabilities", {}
                    ).get("task_native_execution_success"),
                    "task_native_safe_success_probability": label.get(
                        "probabilities", {}
                    ).get("task_native_safe_success"),
                    "label_trial_count": label.get("trial_count", 1),
                    "unstable_boolean_fields": label.get(
                        "unstable_boolean_fields", []
                    ),
                    "collision_audit_scope": label.get("collision_audit_scope"),
                    "max_non_target_contact_impulse": label.get(
                        "max_non_target_contact_impulse"
                    ),
                    "expert_contact_impulse_baseline": label.get(
                        "expert_contact_impulse_baseline"
                    ),
                    "expert_contact_impulse_limit": label.get(
                        "expert_contact_impulse_limit"
                    ),
                    "collision_within_expert_baseline": label.get(
                        "collision_within_expert_baseline"
                    ),
                    "source": label.get("source"),
                },
            }
        )
    inference = sample["inference_visible"]
    event = training.get("event", {})
    result = {
        "schema_version": CANDIDATE_SAMPLE_SCHEMA,
        "sample_id": str(sample.get("sample_id", "unknown")),
        "task": str(sample.get("task", "unknown")),
        "seed": int(sample.get("seed", 0)),
        "world_state_version": int(sample.get("world_state_version", 0)),
        "inference_visible": {
            "intent": event_intent_from_sample(sample).as_dict(),
            "observations": deepcopy(inference.get("observations", [])),
            "observation_phase": inference.get("observation_phase"),
            "expert_summary_reference": (
                f"event_manifest:{sample.get('task', 'unknown')}/"
                f"{event.get('event_id', 'unknown')}"
            ),
        },
        "training_only": {
            "access": "oracle/training_only",
            "oracle_geometry": {
                "expert_grasp_frame": deepcopy(training.get("expert_grasp_frame")),
            },
            "oracle_candidate_graph": {
                "source": "automatic_successful_expert_frame_perturbations",
                "candidate_count": len(rows),
            },
            "candidate_labels": rows,
            "label_generation": {
                "point_annotation": "automatic",
                "semantic_role_source": "task_event_manifest",
                "metric_frame_source": "successful_expert_trajectory",
                "execution_label_source": "deterministic_physical_replay",
                "hard_sample_scope": "current_task_native_only",
                "excluded_candidate_ids": excluded_candidate_ids,
                "opening_width_policy": (
                    "exclude_until_metric_width_to_actuator_calibration"
                ),
            },
        },
    }
    validation_errors = validate_candidate_sample(result)
    if validation_errors:
        raise ValueError(
            "converted candidate sample is invalid: "
            + "; ".join(validation_errors)
        )
    return result


def event_intent_from_sample(sample: Mapping[str, Any]) -> GraspIntent:
    inference = sample.get("inference_visible", {})
    training = sample.get("training_only", {})
    query = inference.get("query", {})
    event = training.get("event", {})
    target = query.get("target") or event.get("actor_name")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("event sample cannot resolve a semantic target name")
    task_stage = str(query.get("task_stage") or "initial_grasp")
    preferred_roles = query.get("preferred_roles") or [task_stage]
    required_arms = query.get("required_arms")
    if required_arms is None:
        required_arms = 2 if query.get("coordination_group") else 1
    source = str(
        query.get("source")
        or (
            "task_event_manifest_legacy_upgrade"
            if not query.get("target")
            else "task_instruction_or_vlm_at_deployment"
        )
    )
    return GraspIntent(
        target=target,
        task_goal=str(query.get("task_goal") or sample.get("task") or "grasp"),
        task_stage=task_stage,
        post_grasp_goal=str(query.get("post_grasp_goal") or "unspecified"),
        preferred_roles=tuple(str(value) for value in preferred_roles),
        required_arms=int(required_arms),
        active_arm=str(query.get("active_arm") or event.get("arm") or "right"),
        source=source,
        confidence=float(query.get("confidence", 1.0)),
    )


def _candidate_with_relative_frame_features(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = deepcopy(dict(value))
    frame = candidate.get("frame")
    parameters = candidate.get("generation_parameters")
    if not isinstance(frame, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("event candidate requires frame and generation_parameters")
    parameters = dict(parameters)
    delta = _vector3(parameters.get("center_delta_world_m"), "center_delta_world_m")
    approach = _unit_vector(frame.get("approach_axis_world"), "approach_axis_world")
    closing = _unit_vector(frame.get("closing_axis_world"), "closing_axis_world")
    closing = _unit_vector(
        closing - float(np.dot(closing, approach)) * approach,
        "orthogonal closing_axis_world",
    )
    longitudinal = _unit_vector(
        np.cross(approach, closing), "longitudinal_axis_world"
    )
    parameters.setdefault(
        "longitudinal_offset_m", float(np.dot(delta, longitudinal))
    )
    parameters.setdefault("closing_offset_m", float(np.dot(delta, closing)))
    parameters.setdefault("approach_offset_m", float(np.dot(delta, approach)))
    parameters.setdefault("orientation_offset_deg", 0.0)
    parameters.setdefault("opening_width_scale", 1.0)
    candidate["generation_parameters"] = parameters
    return candidate


def _vector3(value: Any, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must be a finite 3-vector")
    return result


def _unit_vector(value: Any, field_name: str) -> np.ndarray:
    result = _vector3(value, field_name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{field_name} must be nonzero")
    return result / norm


def _finite_numbers(values: Any) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _numeric_statistics(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "maximum": None, "mean": None, "stddev": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "stddev": float(np.std(array)),
    }
