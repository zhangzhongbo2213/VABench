"""Leakage-controlled samples produced by probing inference candidate sets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


OPEN_VOCAB_CANDIDATE_PROBE_SCHEMA = "spatial.open_vocab_grasp_candidate_probe_sample.v1"
FORBIDDEN_INFERENCE_KEYS = {
    "actor_id",
    "actor_segmentation_id",
    "oracle_geometry",
    "oracle_graph",
    "oracle_candidate_graph",
    "oracle_object_pose",
    "physics_contacts",
    "contact_truth",
    "execution_success",
    "execution_success_label",
    "task_success_label",
    "task_native_execution_success",
    "task_native_safe_success",
    "candidate_execution_labels",
}


def build_open_vocab_candidate_probe_sample(
    *,
    query_result: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    task: str,
    seed: int,
    source_query: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Pair an inference candidate graph with training-only physical outcomes.

    ``case_id`` distinguishes several probes of the same task and seed. Without
    it every case collapses to one sample_id and the dataset audit discards all
    but the first as a duplicate, silently shrinking the evaluation set.
    """

    candidates = query_result.get("top_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("query result must contain non-empty top_candidates")
    candidate_by_id = {
        str(candidate.get("id")): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("id")
    }
    if len(candidate_by_id) != len(candidates):
        raise ValueError("top_candidates must have unique non-empty IDs")
    labels_by_id: dict[str, dict[str, Any]] = {}
    probe_status_by_id: dict[str, str] = {}
    for row in probe_rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe rows must be mappings")
        candidate_id = str(row.get("candidate_id", "")).strip()
        if candidate_id not in candidate_by_id:
            raise ValueError(f"probe row references unknown candidate {candidate_id!r}")
        if candidate_id in labels_by_id:
            raise ValueError(f"probe rows repeat candidate {candidate_id!r}")
        label = row.get("label")
        if not isinstance(label, Mapping):
            raise ValueError(f"probe row for {candidate_id!r} has no label")
        labels_by_id[candidate_id] = _normalise_label(candidate_id, label)
        probe_status_by_id[candidate_id] = str(row.get("status", "complete"))
    missing = sorted(set(candidate_by_id) - set(labels_by_id))
    if missing:
        raise ValueError(
            "candidate probe coverage is incomplete: " + ", ".join(missing)
        )

    visible = {
        "intent": deepcopy(query_result.get("intent", {})),
        "candidate_graph": deepcopy(query_result.get("candidate_graph", {})),
        "top_candidates": deepcopy(candidates),
        "candidate_ranking": deepcopy(query_result.get("candidate_ranking", {})),
        "view_assessment": deepcopy(query_result.get("view_assessment", {})),
        "observed_view_sequence": list(query_result.get("observed_view_sequence", ())),
        "evidence_frames": list(query_result.get("evidence_frames", ())),
        "evidence_views": list(query_result.get("evidence_views", ())),
        "world_frozen": query_result.get("world_frozen"),
        "source_query": str(source_query),
    }
    sample = {
        "schema_version": OPEN_VOCAB_CANDIDATE_PROBE_SCHEMA,
        "sample_id": _probe_sample_id(task=task, seed=seed, case_id=case_id),
        "case_id": None if case_id is None else str(case_id),
        "task": str(task),
        "seed": int(seed),
        "inference_visible": visible,
        "training_only": {
            "access": "oracle/training_only",
            "candidate_labels": [
                {
                    "candidate": deepcopy(candidate_by_id[candidate_id]),
                    "execution": labels_by_id[candidate_id],
                    "probe_status": probe_status_by_id[candidate_id],
                }
                for candidate_id in candidate_by_id
            ],
            "label_generation": {
                "point_annotation": "automatic",
                "metric_frame_source": "inference_candidate_frame",
                "execution_label_source": "same_seed_physical_probe",
                "hard_sample_scope": "current_task_native_only",
                "unprobed_candidates_are_unknown": True,
            },
        },
    }
    errors = validate_open_vocab_candidate_probe_sample(sample)
    if errors:
        raise ValueError(
            "invalid open-vocabulary candidate probe sample: " + "; ".join(errors)
        )
    return sample


def validate_open_vocab_candidate_probe_sample(
    sample: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if sample.get("schema_version") != OPEN_VOCAB_CANDIDATE_PROBE_SCHEMA:
        errors.append("unsupported schema_version")
    visible = sample.get("inference_visible")
    training = sample.get("training_only")
    if not isinstance(visible, Mapping):
        errors.append("inference_visible must be a mapping")
        return errors
    if not isinstance(training, Mapping):
        errors.append("training_only must be a mapping")
        return errors
    leaked_keys = sorted(_recursive_keys(visible) & FORBIDDEN_INFERENCE_KEYS)
    if leaked_keys:
        errors.append(
            "execution/training keys leaked into inference_visible: "
            + ", ".join(leaked_keys)
        )
    leaked_values = sorted(
        value
        for value in _recursive_string_values(visible)
        if "oracle/training_only" in value.lower() or "simulator_truth" in value.lower()
    )
    if leaked_values:
        errors.append("oracle source leaked into inference_visible")
    if training.get("access") != "oracle/training_only":
        errors.append("training_only.access must be oracle/training_only")
    rows = training.get("candidate_labels")
    if not isinstance(rows, list) or not rows:
        errors.append("training_only.candidate_labels must be a non-empty list")
        return errors
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("candidate label rows must be mappings")
            continue
        candidate = row.get("candidate")
        execution = row.get("execution")
        if not isinstance(candidate, Mapping) or not isinstance(execution, Mapping):
            errors.append("candidate label row requires candidate and execution")
            continue
        candidate_id = str(candidate.get("id", ""))
        ids.append(candidate_id)
        if candidate_id != str(execution.get("candidate_id", "")):
            errors.append("candidate and execution IDs do not match")
        if _outcome_state(execution) not in {"success", "failure", "unknown"}:
            errors.append(f"unsupported outcome state for {candidate_id}")
    if len(ids) != len(set(ids)):
        errors.append("candidate labels contain duplicate IDs")
    return errors


def candidate_outcome_state(label: Mapping[str, Any]) -> str:
    """Return success/failure/unknown without treating unprobed as negative."""

    return _outcome_state(label)


def _normalise_label(candidate_id: str, label: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(label)
    result["candidate_id"] = candidate_id
    result["outcome_state"] = _outcome_state(result)
    return result


def _outcome_state(label: Mapping[str, Any]) -> str:
    if "task_native_safe_success" in label:
        value = label.get("task_native_safe_success")
    elif "task_native_execution_success" in label:
        value = label.get("task_native_execution_success")
    else:
        value = label.get("execution_success")
    if value is True:
        return "success"
    if value is False:
        return "failure"
    return "unknown"


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()


def _recursive_string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for child in value.values():
            result.update(_recursive_string_values(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_string_values(child))
        return result
    return set()


def _probe_sample_id(*, task: str, seed: int, case_id: str | None) -> str:
    suffix = "candidate_set" if case_id is None else str(case_id).strip()
    if not suffix:
        raise ValueError("open-vocabulary probe case id must not be blank")
    return f"{task}/seed{int(seed)}/{suffix}"
