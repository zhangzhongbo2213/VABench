"""Auditing and quality gates for open-vocabulary candidate probe samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .open_vocab_candidate_probe import (
    validate_open_vocab_candidate_probe_sample,
)

SAMPLE_FILENAME = "open_vocab_candidate_probe_sample.json"


def discover_open_vocab_candidate_samples(
    paths: Iterable[str | Path],
) -> list[Path]:
    """Return deterministic sample paths without traversing unrelated artifacts."""

    result: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_file():
            if path.name == SAMPLE_FILENAME:
                result.add(path)
            continue
        if path.is_dir():
            result.update(path.rglob(SAMPLE_FILENAME))
    return sorted(result)


def audit_open_vocab_candidate_dataset(
    paths: Iterable[str | Path],
    *,
    task_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize samples and report structural/leakage errors without guessing labels."""

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    outcome_counts = {state: 0 for state in ("success", "failure", "unknown")}
    task_summary: dict[str, dict[str, Any]] = {}
    for path in discover_open_vocab_candidate_samples(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": f"read_error: {exc}"})
            continue
        validation_errors = validate_open_vocab_candidate_probe_sample(payload)
        if validation_errors:
            errors.append(
                {"path": str(path), "error": "; ".join(validation_errors)}
            )
            continue
        sample_id = str(payload.get("sample_id", ""))
        if sample_id in sample_ids:
            errors.append({"path": str(path), "error": f"duplicate sample_id: {sample_id}"})
            continue
        sample_ids.add(sample_id)
        task = str(payload.get("task", ""))
        split = str((task_splits or {}).get(task, "unknown"))
        if task_splits is not None and task not in task_splits:
            errors.append({"path": str(path), "error": f"task absent from manifest: {task}"})
        rows = payload["training_only"]["candidate_labels"]
        task_entry = task_summary.setdefault(
            task,
            {
                "split": split,
                "sample_count": 0,
                "candidate_count": 0,
                "outcome_counts": {state: 0 for state in outcome_counts},
            },
        )
        task_entry["sample_count"] += 1
        task_entry["candidate_count"] += len(rows)
        for row in rows:
            state = str(row["execution"].get("outcome_state", "unknown"))
            if state not in outcome_counts:
                errors.append(
                    {"path": str(path), "error": f"unsupported outcome_state: {state}"}
                )
                continue
            outcome_counts[state] += 1
            task_entry["outcome_counts"][state] += 1
        samples.append(payload)

    candidate_count = sum(
        int(value["candidate_count"]) for value in task_summary.values()
    )
    return {
        "schema_version": "phase11.open_vocab_candidate_dataset_audit.v1",
        "sample_count": len(samples),
        "candidate_count": candidate_count,
        "outcome_counts": outcome_counts,
        "task_summary": task_summary,
        "errors": errors,
        "sample_paths": [str(path) for path in discover_open_vocab_candidate_samples(paths)],
    }


def evaluate_open_vocab_candidate_gate(
    audit: Mapping[str, Any],
    *,
    required_tasks: Sequence[str] = (),
    min_success: int = 1,
    min_failure: int = 1,
    max_unknown_fraction: float = 1.0,
    require_balanced_task: bool = False,
) -> dict[str, Any]:
    """Return an explicit gate decision; unknown labels never count as failures."""

    if min_success < 0 or min_failure < 0:
        raise ValueError("minimum outcome counts must be non-negative")
    if not 0.0 <= max_unknown_fraction <= 1.0:
        raise ValueError("max_unknown_fraction must be in [0, 1]")
    errors = [str(row.get("error")) for row in audit.get("errors", ())]
    counts = dict(audit.get("outcome_counts", {}))
    success = int(counts.get("success", 0))
    failure = int(counts.get("failure", 0))
    unknown = int(counts.get("unknown", 0))
    labelled = success + failure + unknown
    unknown_fraction = unknown / labelled if labelled else 1.0
    if int(audit.get("sample_count", 0)) == 0:
        errors.append("no valid open-vocabulary candidate samples found")
    if success < min_success:
        errors.append(f"success_count={success} is below minimum {min_success}")
    if failure < min_failure:
        errors.append(f"failure_count={failure} is below minimum {min_failure}")
    if unknown_fraction > max_unknown_fraction:
        errors.append(
            f"unknown_fraction={unknown_fraction:.6f} exceeds {max_unknown_fraction:.6f}"
        )
    task_summary = audit.get("task_summary", {})
    for task in required_tasks:
        entry = task_summary.get(task)
        if not isinstance(entry, Mapping) or int(entry.get("sample_count", 0)) == 0:
            errors.append(f"required task has no sample: {task}")
            continue
        if require_balanced_task:
            task_counts = entry.get("outcome_counts", {})
            if not int(task_counts.get("success", 0)):
                errors.append(f"required task has no success label: {task}")
            if not int(task_counts.get("failure", 0)):
                errors.append(f"required task has no failure label: {task}")
    return {
        "schema_version": "phase11.open_vocab_candidate_dataset_gate.v1",
        "status": "pass" if not errors else "blocked",
        "errors": errors,
        "required_tasks": list(required_tasks),
        "policy": {
            "min_success": min_success,
            "min_failure": min_failure,
            "max_unknown_fraction": max_unknown_fraction,
            "require_balanced_task": require_balanced_task,
        },
        "counts": {
            "sample_count": int(audit.get("sample_count", 0)),
            "candidate_count": int(audit.get("candidate_count", 0)),
            "success": success,
            "failure": failure,
            "unknown": unknown,
            "unknown_fraction": unknown_fraction,
        },
    }
