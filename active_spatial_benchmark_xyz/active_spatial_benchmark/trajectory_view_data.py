"""Trajectory-return labels for recovery-conditioned active view selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .learned_view_selector import (
    ONPOLICY_FEATURE_DIM,
    ViewRankingExample,
    view_feature_vector,
)


RETURN_WEIGHTS = {
    "task_success": 1.0,
    "unsafe_execute": -1.0,
    "recovery_safety_violation": -1.0,
    "terminal_error": -1.0,
    "future_view": -0.02,
    "future_recovery_step": -0.05,
    "initial_camera_move_cost": -0.05,
}


RECONSTRUCTION_EXCLUSION_MARKERS = (
    "state reconstruction failed",
    "reconstructed recipe is not an active-view decision state",
)


def is_reconstruction_exclusion(error: str) -> bool:
    """Return whether a replay failure should exclude one state, not fail a shard."""

    normalized = str(error).lower()
    return any(marker in normalized for marker in RECONSTRUCTION_EXCLUSION_MARKERS)


def trajectory_return(
    *,
    task_success: bool,
    action_executed: bool,
    recovery_safety_violation: bool,
    terminal_error: bool,
    future_view_count: int,
    future_recovery_step_count: int,
    initial_camera_move_cost: float,
) -> dict[str, Any]:
    """Calculate a safety-dominant return from the selected-view decision onward."""

    if future_view_count < 1:
        raise ValueError("a trajectory branch must execute at least one future view")
    if future_recovery_step_count < 0:
        raise ValueError("future_recovery_step_count must be non-negative")
    camera_cost = float(np.clip(initial_camera_move_cost, 0.0, 1.0))
    unsafe_execute = bool(action_executed and not task_success)
    components = {
        "task_success": RETURN_WEIGHTS["task_success"] * float(task_success),
        "unsafe_execute": RETURN_WEIGHTS["unsafe_execute"] * float(unsafe_execute),
        "recovery_safety_violation": RETURN_WEIGHTS["recovery_safety_violation"]
        * float(recovery_safety_violation),
        "terminal_error": RETURN_WEIGHTS["terminal_error"] * float(terminal_error),
        "future_views": RETURN_WEIGHTS["future_view"] * int(future_view_count),
        "future_recovery_steps": RETURN_WEIGHTS["future_recovery_step"]
        * int(future_recovery_step_count),
        "initial_camera_move_cost": RETURN_WEIGHTS["initial_camera_move_cost"]
        * camera_cost,
    }
    return {
        "return": round(float(sum(components.values())), 6),
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "unsafe_execute": unsafe_execute,
        "weights": dict(RETURN_WEIGHTS),
    }


def validate_trajectory_branch(
    value: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
) -> list[str]:
    errors = []
    if value.get("schema_version") != "phase9b.trajectory_view_branch.v1":
        errors.append("unexpected schema_version")
    inference = value.get("inference_visible", {})
    serialized = json.dumps(inference).lower()
    if any(token in serialized for token in ("oracle", "actor_id", "segmentation", "task_success")):
        errors.append("training truth leaked into inference_visible")
    training = value.get("training_only", {})
    outcome = training.get("trajectory_outcome", {})
    score = training.get("trajectory_return", {})
    if not np.isfinite(float(score.get("return", np.nan))):
        errors.append("non-finite trajectory return")
    if int(outcome.get("future_view_count", 0)) < 1:
        errors.append("branch did not execute a future view")
    if str(value.get("forced_view")) != str(inference.get("candidate_row", {}).get("view")):
        errors.append("forced view/candidate row mismatch")
    replay = training.get("reconstruction_audit", {})
    if not bool(replay.get("passed", False)):
        errors.append("state reconstruction audit failed")
    try:
        expected_score = trajectory_return(
            task_success=bool(outcome["physical_task_success"]),
            action_executed=bool(outcome["action_executed"]),
            recovery_safety_violation=bool(outcome["recovery_safety_violation"]),
            terminal_error=bool(outcome["terminal_error"]),
            future_view_count=int(outcome["future_view_count"]),
            future_recovery_step_count=int(outcome["future_recovery_step_count"]),
            initial_camera_move_cost=float(
                inference["candidate_row"]["predicted"]["move_cost"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"incomplete trajectory return inputs: {exc}")
    else:
        if not np.isclose(
            float(score.get("return", np.nan)),
            float(expected_score["return"]),
            atol=1e-6,
        ):
            errors.append("trajectory return does not match trajectory outcome")
        if bool(score.get("unsafe_execute")) != bool(expected_score["unsafe_execute"]):
            errors.append("unsafe_execute return audit mismatch")
    source_path = value.get("source_sample")
    if source_path is None or Path(str(source_path)).is_absolute():
        errors.append("source_sample must be relative")
    elif dataset_root is not None and not (Path(dataset_root) / str(source_path)).is_file():
        errors.append("source_sample is missing")
    return errors


def build_trajectory_view_ranking_examples(
    dataset_root: str | Path,
    *,
    seeds: Iterable[int] | None = None,
) -> list[ViewRankingExample]:
    root = Path(dataset_root).resolve()
    allowed = {int(seed) for seed in (seeds or ())}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("branches/**/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if allowed and int(value["seed"]) not in allowed:
            continue
        errors = validate_trajectory_branch(value, dataset_root=root)
        if errors:
            raise ValueError(f"invalid trajectory branch {path}: {errors}")
        grouped.setdefault(str(value["sample_id"]), []).append(value)
    examples = []
    for sample_id, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        rows.sort(key=lambda row: str(row["forced_view"]))
        views = [str(row["forced_view"]) for row in rows]
        if len(set(views)) != len(views):
            raise ValueError(f"duplicate forced view in paired state {sample_id}")
        inference = rows[0]["inference_visible"]
        shared_fields = ("visited_views", "edge_uncertainty", "state_context", "belief_before")
        for row in rows[1:]:
            other = row["inference_visible"]
            if any(other.get(field) != inference.get(field) for field in shared_fields):
                raise ValueError(f"inference context mismatch in paired state {sample_id}")
        examples.append(
            ViewRankingExample(
                episode_id=f"grasp_single_pen_seed{int(rows[0]['seed'])}",
                state_id=sample_id,
                views=tuple(views),
                features=torch.stack(
                    [
                        view_feature_vector(
                            row["inference_visible"]["candidate_row"],
                            inference["edge_uncertainty"],
                            state_context=inference["state_context"],
                            feature_dim=ONPOLICY_FEATURE_DIM,
                        )
                        for row in rows
                    ]
                ),
                target_utility=torch.tensor(
                    [
                        float(row["training_only"]["trajectory_return"]["return"])
                        for row in rows
                    ],
                    dtype=torch.float32,
                ),
                candidate_metadata=tuple(
                    {
                        "physical_task_success": bool(
                            row["training_only"]["trajectory_outcome"][
                                "physical_task_success"
                            ]
                        ),
                        "unsafe_execute": bool(
                            row["training_only"]["trajectory_outcome"].get(
                                "unsafe_execute",
                                row["training_only"]["trajectory_outcome"].get(
                                    "action_executed", False
                                )
                                and not row["training_only"]["trajectory_outcome"].get(
                                    "physical_task_success", False
                                ),
                            )
                        ),
                        "recovery_safety_violation": bool(
                            row["training_only"]["trajectory_outcome"][
                                "recovery_safety_violation"
                            ]
                        ),
                        "terminal_error": bool(
                            row["training_only"]["trajectory_outcome"]["terminal_error"]
                        ),
                        "stop_reason": str(
                            row["training_only"]["trajectory_outcome"].get(
                                "stop_reason", "unknown"
                            )
                        ),
                    }
                    for row in rows
                ),
            )
        )
    if not examples:
        raise ValueError("no paired trajectory-return examples found")
    return examples
