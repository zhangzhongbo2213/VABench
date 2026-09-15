"""Physical labels for coordinated grasp-candidate graphs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .coordinated_grasp_candidates import (
    COORDINATED_CANDIDATE_GRAPH_SCHEMA,
    validate_coordinated_candidate_graph,
)


LABELLED_COORDINATED_GRAPH_SCHEMA = (
    "spatial.labelled_coordinated_grasp_candidate_graph.v1"
)
COORDINATED_CANDIDATE_SAMPLE_SCHEMA = (
    "spatial.coordinated_grasp_candidate_sample.v1"
)
REQUIRED_JOINT_LABEL_FIELDS = {
    "joint_pregrasp_reachable",
    "joint_grasp_reachable",
    "synchronized_close_success",
    "per_arm",
    "task_success",
    "joint_execution_success",
    "joint_safe_success",
}


def merge_coordinated_probe_reports(
    graph: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    *,
    graph_sha256: str,
    graph_path: str | None = None,
    report_references: Sequence[str] | None = None,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    errors = validate_coordinated_candidate_graph(graph)
    if errors:
        raise ValueError("invalid coordinated graph: " + "; ".join(errors))
    candidates = graph["training_only"]["joint_candidates"]
    candidate_by_index = {index: value for index, value in enumerate(candidates)}
    references = list(report_references or [""] * len(reports))
    if len(references) != len(reports):
        raise ValueError("report_references must align with reports")
    labels = {}
    for report, reference in zip(reports, references):
        if report.get("schema_version") != "phase11.coordinated_candidate_probe.v1":
            raise ValueError("unsupported coordinated probe report schema")
        settings = report.get("settings")
        if not isinstance(settings, Mapping):
            raise ValueError("coordinated probe report has no settings")
        reported_digest = settings.get("graph_sha256")
        if reported_digest is not None:
            if reported_digest != graph_sha256:
                raise ValueError("coordinated probe report references a different graph")
        elif graph_path is None or Path(str(settings.get("graph"))).resolve() != Path(
            graph_path
        ).resolve():
            raise ValueError("legacy coordinated probe report graph path does not match")
        for row in report.get("results", []):
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                continue
            index = int(row["candidate_index"])
            if index not in candidate_by_index:
                raise ValueError(f"unknown joint candidate index {index}")
            expected_id = candidate_by_index[index]["id"]
            if row.get("candidate_id") != expected_id:
                raise ValueError("joint candidate ID does not match graph")
            label = row.get("label")
            if not isinstance(label, Mapping):
                raise ValueError("complete joint probe row has no label")
            missing = sorted(REQUIRED_JOINT_LABEL_FIELDS - set(label))
            if missing:
                raise ValueError("joint probe label is missing: " + ", ".join(missing))
            merged = {
                "candidate_index": index,
                "candidate_id": expected_id,
                "pairing_mode": row.get("pairing_mode"),
                "label": deepcopy(dict(label)),
                "probe_report": reference,
                "probe_result": row.get("result"),
                "access": "oracle/training_only",
            }
            previous = labels.get(index)
            if previous is not None and previous["label"] != merged["label"]:
                raise ValueError(f"conflicting joint labels for candidate {index}")
            labels[index] = merged
    labelled_indices = sorted(labels)
    unlabelled_indices = sorted(set(candidate_by_index) - set(labelled_indices))
    if require_complete_coverage and unlabelled_indices:
        raise ValueError("coordinated candidate labels are incomplete")
    result = deepcopy(dict(graph))
    result["schema_version"] = LABELLED_COORDINATED_GRAPH_SCHEMA
    training = result["training_only"]
    training["source_candidate_graph"] = {
        "schema_version": graph["schema_version"],
        "sha256": graph_sha256,
    }
    training["joint_candidate_execution_labels"] = [
        labels[index] for index in labelled_indices
    ]
    training["label_coverage"] = {
        "candidate_count": len(candidates),
        "labelled_candidate_count": len(labelled_indices),
        "unlabelled_candidate_indices": unlabelled_indices,
        "complete": not unlabelled_indices,
        "positive_count": sum(
            labels[index]["label"]["joint_safe_success"] is True
            for index in labelled_indices
        ),
        "negative_count": sum(
            labels[index]["label"]["joint_safe_success"] is False
            for index in labelled_indices
        ),
    }
    validation_errors = validate_labelled_coordinated_graph(result)
    if validation_errors:
        raise ValueError(
            "invalid labelled coordinated graph: " + "; ".join(validation_errors)
        )
    return result


def validate_labelled_coordinated_graph(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != LABELLED_COORDINATED_GRAPH_SCHEMA:
        errors.append("unsupported labelled coordinated graph schema")
    base_graph = deepcopy(dict(value))
    base_graph["schema_version"] = COORDINATED_CANDIDATE_GRAPH_SCHEMA
    errors.extend(validate_coordinated_candidate_graph(base_graph))
    inference = value.get("inference_visible")
    training = value.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        errors.append("labelled coordinated graph requires inference and training payloads")
        return errors
    if "joint_candidate_execution_labels" in inference:
        errors.append("joint execution labels leaked into inference_visible")
    labels = training.get("joint_candidate_execution_labels")
    coverage = training.get("label_coverage")
    if not isinstance(labels, list) or not isinstance(coverage, Mapping):
        errors.append("joint labels and coverage are required")
    elif int(coverage.get("labelled_candidate_count", -1)) != len(labels):
        errors.append("joint label coverage count is inconsistent")
    return errors


def convert_labelled_coordinated_graph_to_ranker_sample(
    value: Mapping[str, Any],
    *,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    """Build a leakage-controlled joint-candidate ranking sample.

    Partial physical coverage is intentional during boundary search.  A ranker
    may consume the sample once its selected target contains at least one
    positive and one negative; unprobed candidates never receive inferred
    labels here.
    """

    errors = validate_labelled_coordinated_graph(value)
    if errors:
        raise ValueError("invalid labelled coordinated graph: " + "; ".join(errors))
    training = value["training_only"]
    coverage = training["label_coverage"]
    if require_complete_coverage and coverage.get("complete") is not True:
        raise ValueError("coordinated ranker conversion requires complete coverage")
    candidates = training.get("joint_candidates", ())
    candidate_by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("coordinated candidates contain duplicate IDs")
    rows = []
    for physical in training.get("joint_candidate_execution_labels", ()):
        candidate_id = str(physical.get("candidate_id", ""))
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"joint label references unknown candidate {candidate_id!r}")
        label = physical.get("label")
        if not isinstance(label, Mapping):
            raise ValueError("joint candidate label must be a mapping")
        rows.append(
            {
                "candidate": deepcopy(candidate),
                "execution": {
                    key: deepcopy(label.get(key))
                    for key in sorted(REQUIRED_JOINT_LABEL_FIELDS)
                }
                | {
                    "candidate_id": candidate_id,
                    "source": label.get("source"),
                },
            }
        )
    if not rows:
        raise ValueError("coordinated ranker sample requires physical labels")
    query = value["inference_visible"].get("query", {})
    result = {
        "schema_version": COORDINATED_CANDIDATE_SAMPLE_SCHEMA,
        "sample_id": str(value.get("graph_id", "unknown")),
        "task": str(value.get("task", "unknown")),
        "seed": int(value.get("seed", 0)),
        "inference_visible": {
            "intent": _coordinated_intent(query),
            "observations_by_arm": deepcopy(
                value["inference_visible"].get("observations_by_arm", {})
            ),
        },
        "training_only": {
            "access": "oracle/training_only",
            "oracle_joint_candidate_graph": {
                "source_graph_id": value.get("graph_id"),
                "graph_variant": value.get("graph_variant"),
                "candidate_count": len(candidates),
                "nominal_candidate": deepcopy(candidates[0]),
            },
            "candidate_labels": rows,
            "label_coverage": deepcopy(coverage),
            "label_generation": {
                "point_annotation": "automatic",
                "metric_frame_source": "successful_expert_trajectory",
                "execution_label_source": "synchronized_dual_arm_physical_replay",
                "hard_sample_scope": "current_task_native_only",
                "unprobed_candidates_are_negative": False,
            },
        },
    }
    validation_errors = validate_coordinated_candidate_sample(result)
    if validation_errors:
        raise ValueError(
            "converted coordinated candidate sample is invalid: "
            + "; ".join(validation_errors)
        )
    return result


def validate_coordinated_candidate_sample(value: Mapping[str, Any]) -> list[str]:
    errors = []
    if value.get("schema_version") != COORDINATED_CANDIDATE_SAMPLE_SCHEMA:
        errors.append("unsupported coordinated candidate sample schema")
    inference = value.get("inference_visible")
    training = value.get("training_only")
    if not isinstance(inference, Mapping) or not isinstance(training, Mapping):
        errors.append("coordinated ranker sample requires inference and training payloads")
        return errors
    forbidden = REQUIRED_JOINT_LABEL_FIELDS | {
        "joint_candidate_execution_labels",
        "candidate_labels",
        "oracle_joint_candidate_graph",
    }
    leaked = sorted(_recursive_keys(inference).intersection(forbidden))
    if leaked:
        errors.append("joint execution data leaked into inference_visible: " + ", ".join(leaked))
    intent = inference.get("intent")
    if not isinstance(intent, Mapping):
        errors.append("coordinated ranker sample requires an inference-visible intent")
    elif int(intent.get("required_arms", 0)) != 2:
        errors.append("coordinated intent must require two arms")
    if training.get("access") != "oracle/training_only":
        errors.append("coordinated ranker training payload must remain training-only")
    rows = training.get("candidate_labels")
    coverage = training.get("label_coverage")
    if not isinstance(rows, list) or not rows:
        errors.append("coordinated ranker sample requires candidate labels")
    else:
        ids = []
        for row in rows:
            candidate = row.get("candidate", {}) if isinstance(row, Mapping) else {}
            execution = row.get("execution", {}) if isinstance(row, Mapping) else {}
            ids.append(candidate.get("id"))
            if execution.get("candidate_id") != candidate.get("id"):
                errors.append("coordinated execution candidate ID does not match input")
            if candidate.get("predicted_execution_success") is not None:
                errors.append("candidate input cannot contain a predicted execution label")
            missing = sorted(REQUIRED_JOINT_LABEL_FIELDS - set(execution))
            if missing:
                errors.append("coordinated execution label is missing: " + ", ".join(missing))
        if len(ids) != len(set(ids)):
            errors.append("coordinated ranker candidate IDs must be unique")
        if not isinstance(coverage, Mapping):
            errors.append("coordinated ranker sample requires label coverage")
        elif int(coverage.get("labelled_candidate_count", -1)) != len(rows):
            errors.append("coordinated ranker label coverage is inconsistent")
    return errors


def _coordinated_intent(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target": str(query.get("target") or "unknown"),
        "task_goal": str(query.get("task_goal") or "coordinated_grasp"),
        "task_stage": "coordinated_dual_grasp",
        "post_grasp_goal": str(query.get("post_grasp_goal") or "unspecified"),
        "preferred_roles": ["coordinated_dual_grasp"],
        "required_arms": 2,
        "coordination_group": str(query.get("coordination_group") or "unknown"),
        "contact_pattern": "bilateral_contact_per_gripper",
        "natural_language_constraints": [
            "both grasp frames must be jointly reachable",
            "both grippers must close synchronously",
            "the shared target must remain stable during the post-grasp action",
        ],
        "source": "task_instruction_or_vlm_at_deployment",
        "confidence": 1.0,
    }


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


def load_and_merge_coordinated_probe_reports(
    graph_path: str | Path,
    report_paths: Sequence[str | Path],
    *,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    resolved_graph = Path(graph_path).resolve()
    graph_bytes = resolved_graph.read_bytes()
    graph = json.loads(graph_bytes)
    resolved_reports = [Path(path).resolve() for path in report_paths]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in resolved_reports]
    return merge_coordinated_probe_reports(
        graph,
        reports,
        graph_sha256=hashlib.sha256(graph_bytes).hexdigest(),
        graph_path=str(resolved_graph),
        report_references=[str(path) for path in resolved_reports],
        require_complete_coverage=require_complete_coverage,
    )
