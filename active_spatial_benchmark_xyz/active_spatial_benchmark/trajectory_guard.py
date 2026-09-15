"""Calibration and paired evaluation for trajectory-return view guards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .learned_view_selector import ConservativeTrajectoryViewRanker, ViewRankingExample


@torch.no_grad()
def collect_guard_decisions(
    model: ConservativeTrajectoryViewRanker,
    examples: Iterable[ViewRankingExample],
    *,
    device: torch.device | str,
) -> list[dict[str, Any]]:
    """Materialize proposal/base comparisons without applying a guard threshold."""

    model.eval()
    decisions = []
    for example in examples:
        features = example.features.to(device)
        target = example.target_utility.detach().cpu().numpy()
        details = model.selection_details(features)
        base_index = int(details["base_index"])
        proposal_index = int(details["learned_index"])
        oracle_index = int(np.argmax(target))
        metadata = example.candidate_metadata
        if metadata and len(metadata) != len(example.views):
            raise ValueError(
                f"candidate metadata/view mismatch for {example.state_id}: "
                f"{len(metadata)} != {len(example.views)}"
            )
        decisions.append(
            {
                "episode_id": example.episode_id,
                "state_id": example.state_id,
                "base_index": base_index,
                "proposal_index": proposal_index,
                "oracle_index": oracle_index,
                "base_view": example.views[base_index],
                "proposal_view": example.views[proposal_index],
                "oracle_view": example.views[oracle_index],
                "base_return": float(target[base_index]),
                "proposal_return": float(target[proposal_index]),
                "oracle_return": float(target[oracle_index]),
                "lcb_advantage": float(details["lcb_advantage"]),
                "base_metadata": dict(metadata[base_index]) if metadata else {},
                "proposal_metadata": dict(metadata[proposal_index]) if metadata else {},
            }
        )
    if not decisions:
        raise ValueError("guard calibration requires at least one paired decision")
    return decisions


def evaluate_guard_margin(
    decisions: Sequence[Mapping[str, Any]],
    margin: float,
) -> dict[str, Any]:
    """Evaluate a fixed LCB override threshold on paired trajectory returns."""

    if margin < 0.0:
        raise ValueError("guard margin must be non-negative")
    return_deltas = []
    selected_regrets = []
    base_regrets = []
    improved = regressed = tied = proposed = executed = selected_top1 = base_top1 = 0
    base_unsafe = selected_unsafe = 0
    base_safety = selected_safety = 0
    new_unsafe = new_safety = 0
    selected_rows = []
    for decision in decisions:
        is_proposal = int(decision["proposal_index"]) != int(decision["base_index"])
        execute_override = is_proposal and float(decision["lcb_advantage"]) >= margin
        proposed += int(is_proposal)
        executed += int(execute_override)
        selected_prefix = "proposal" if execute_override else "base"
        selected_return = float(decision[f"{selected_prefix}_return"])
        base_return = float(decision["base_return"])
        delta = selected_return - base_return
        return_deltas.append(delta)
        improved += int(delta > 1e-8)
        regressed += int(delta < -1e-8)
        tied += int(abs(delta) <= 1e-8)
        oracle_return = float(decision["oracle_return"])
        selected_regrets.append(oracle_return - selected_return)
        base_regrets.append(oracle_return - base_return)
        selected_index = int(decision[f"{selected_prefix}_index"])
        selected_top1 += int(selected_index == int(decision["oracle_index"]))
        base_top1 += int(int(decision["base_index"]) == int(decision["oracle_index"]))
        base_meta = decision.get("base_metadata", {})
        selected_meta = decision.get(f"{selected_prefix}_metadata", {})
        base_unsafe += int(bool(base_meta.get("unsafe_execute", False)))
        selected_unsafe += int(bool(selected_meta.get("unsafe_execute", False)))
        base_safety += int(bool(base_meta.get("recovery_safety_violation", False)))
        selected_safety += int(
            bool(selected_meta.get("recovery_safety_violation", False))
        )
        new_unsafe += int(
            bool(selected_meta.get("unsafe_execute", False))
            and not bool(base_meta.get("unsafe_execute", False))
        )
        new_safety += int(
            bool(selected_meta.get("recovery_safety_violation", False))
            and not bool(base_meta.get("recovery_safety_violation", False))
        )
        selected_rows.append(
            {
                "state_id": str(decision["state_id"]),
                "base_view": str(decision["base_view"]),
                "proposal_view": str(decision["proposal_view"]),
                "selected_view": str(decision[f"{selected_prefix}_view"]),
                "override_executed": bool(execute_override),
                "lcb_advantage": float(decision["lcb_advantage"]),
                "return_delta_vs_base": float(delta),
            }
        )
    count = len(decisions)
    return {
        "margin": float(margin),
        "state_count": count,
        "proposed_override_count": proposed,
        "executed_override_count": executed,
        "override_coverage": executed / count,
        "improved_decision_count": improved,
        "regressed_decision_count": regressed,
        "tied_decision_count": tied,
        "total_return_delta_vs_base": float(np.sum(return_deltas)),
        "mean_return_delta_vs_base": float(np.mean(return_deltas)),
        "base_mean_regret": float(np.mean(base_regrets)),
        "selected_mean_regret": float(np.mean(selected_regrets)),
        "base_top1_accuracy": base_top1 / count,
        "selected_top1_accuracy": selected_top1 / count,
        "base_unsafe_execute_count": base_unsafe,
        "selected_unsafe_execute_count": selected_unsafe,
        "new_unsafe_execute_count": new_unsafe,
        "base_recovery_safety_violation_count": base_safety,
        "selected_recovery_safety_violation_count": selected_safety,
        "new_recovery_safety_violation_count": new_safety,
        "return_deltas": return_deltas,
        "selections": selected_rows,
    }


def calibrate_guard_margin(
    decisions: Sequence[Mapping[str, Any]],
    margins: Iterable[float],
    *,
    max_regressed_decisions: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose a threshold on calibration data without consulting test states."""

    unique_margins = sorted({float(value) for value in margins})
    if not unique_margins:
        raise ValueError("at least one guard margin is required")
    evaluations = [evaluate_guard_margin(decisions, margin) for margin in unique_margins]
    feasible = [
        row
        for row in evaluations
        if row["regressed_decision_count"] <= max_regressed_decisions
        and row["new_unsafe_execute_count"] == 0
        and row["new_recovery_safety_violation_count"] == 0
    ]
    if not feasible:
        raise ValueError("no guard margin satisfies calibration safety constraints")
    best = max(
        feasible,
        key=lambda row: (
            row["total_return_delta_vs_base"],
            row["improved_decision_count"],
            row["executed_override_count"],
            -row["margin"],
        ),
    )
    return float(best["margin"]), evaluations


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the mean paired return delta with deterministic sampling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("paired bootstrap requires a non-empty one-dimensional sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 1000
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
    }
