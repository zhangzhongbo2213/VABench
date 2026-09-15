"""Phase 9 on-policy counterfactual view labels and dataset loading."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .learned_view_selector import (
    ONPOLICY_FEATURE_DIM,
    ViewRankingExample,
    view_feature_vector,
)
from .pregrasp_graph import PREGRASP_EDGE_IDS, PREGRASP_RELATION_SPECS


def counterfactual_view_target(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    oracle_graph: Mapping[str, Any],
    *,
    move_cost: float,
) -> dict[str, float]:
    """Score one actually rendered candidate using training-only Oracle truth."""

    truth = {str(edge["id"]): float(edge["probability"]) for edge in oracle_graph["edges"]}
    before_relations = before.get("relations", {})
    after_relations = after.get("relations", {})
    total_weight = 0.0
    error_reduction = 0.0
    uncertainty_reduction = 0.0
    for edge_id in PREGRASP_EDGE_IDS:
        weight = float(PREGRASP_RELATION_SPECS[edge_id]["weight"])
        before_edge = before_relations.get(edge_id, {})
        after_edge = after_relations.get(edge_id, {})
        before_error = abs(float(before_edge.get("probability", 0.5)) - truth[edge_id])
        after_error = abs(float(after_edge.get("probability", 0.5)) - truth[edge_id])
        error_reduction += weight * (before_error - after_error)
        uncertainty_reduction += weight * (
            float(before_edge.get("uncertainty", 1.0))
            - float(after_edge.get("uncertainty", 1.0))
        )
        total_weight += weight
    error_reduction /= max(total_weight, 1e-9)
    uncertainty_reduction /= max(total_weight, 1e-9)
    before_decision = _decision_score(str(before.get("verdict", "uncertain")), oracle_graph)
    after_decision = _decision_score(str(after.get("verdict", "uncertain")), oracle_graph)
    decision_gain = after_decision - before_decision
    cost = float(np.clip(move_cost, 0.0, 1.0))
    utility = (
        0.45 * error_reduction
        + 0.35 * uncertainty_reduction
        + 0.20 * decision_gain
        - 0.15 * cost
    )
    return {
        "oracle_relation_error_reduction": round(float(error_reduction), 6),
        "observed_uncertainty_reduction": round(float(uncertainty_reduction), 6),
        "oracle_decision_gain": round(float(decision_gain), 6),
        "camera_move_cost": round(cost, 6),
        "target_utility": round(float(utility), 6),
    }


def build_onpolicy_view_ranking_examples(
    dataset_root: str | Path,
    *,
    seeds: Iterable[int] | None = None,
) -> list[ViewRankingExample]:
    root = Path(dataset_root).resolve()
    allowed = {int(seed) for seed in (seeds or ())}
    examples = []
    for sample_path in sorted(root.glob("samples/**/*.json")):
        value = json.loads(sample_path.read_text(encoding="utf-8"))
        if allowed and int(value["seed"]) not in allowed:
            continue
        errors = validate_onpolicy_view_sample(value, dataset_root=root)
        if errors:
            raise ValueError(f"invalid on-policy sample {sample_path}: {errors}")
        inference = value["inference_visible"]
        targets = {
            str(row["view"]): float(row["target_utility"])
            for row in value["training_only"]["candidate_targets"]
        }
        rows = [
            row for row in inference["candidate_rows"] if str(row["view"]) in targets
        ]
        examples.append(
            ViewRankingExample(
                episode_id=f"grasp_single_pen_seed{int(value['seed'])}",
                state_id=str(value["sample_id"]),
                views=tuple(str(row["view"]) for row in rows),
                features=torch.stack(
                    [
                        view_feature_vector(
                            row,
                            inference["edge_uncertainty"],
                            state_context=inference["state_context"],
                            feature_dim=ONPOLICY_FEATURE_DIM,
                        )
                        for row in rows
                    ]
                ),
                target_utility=torch.tensor(
                    [targets[str(row["view"])] for row in rows], dtype=torch.float32
                ),
            )
        )
    if not examples:
        raise ValueError("no on-policy view-ranking examples found")
    return examples


def validate_onpolicy_view_sample(
    value: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
) -> list[str]:
    errors = []
    if value.get("schema_version") != "phase9.onpolicy_view_sample.v1":
        errors.append("unexpected schema_version")
    inference = value.get("inference_visible", {})
    training = value.get("training_only", {})
    if any(token in json.dumps(inference).lower() for token in ("oracle", "actor_id", "segmentation")):
        errors.append("training truth leaked into inference_visible")
    candidate_rows = list(inference.get("candidate_rows", []))
    target_rows = list(training.get("candidate_targets", []))
    views = [str(row.get("view")) for row in candidate_rows]
    target_views = [str(row.get("view")) for row in target_rows]
    if len(candidate_rows) < 2:
        errors.append("fewer than two candidate rows")
    if len(views) != len(set(views)):
        errors.append("duplicate candidate view")
    if set(views) != set(target_views):
        errors.append("candidate/target view mismatch")
    for row in target_rows:
        if not np.isfinite(float(row.get("target_utility", np.nan))):
            errors.append(f"non-finite target for {row.get('view')}")
        path = row.get("counterfactual_rgb")
        if path is None or PurePosixPath(str(path)).is_absolute():
            errors.append(f"invalid counterfactual path for {row.get('view')}")
        elif dataset_root is not None and not (Path(dataset_root) / str(path)).is_file():
            errors.append(f"missing counterfactual image for {row.get('view')}")
    selected = value.get("selected_view")
    if selected is not None and str(selected) not in set(views):
        errors.append("selected view is not a candidate")
    if float(training.get("world_fingerprint_delta", 1.0)) > 1e-6:
        errors.append("world changed during counterfactual rendering")
    return errors


def _decision_score(verdict: str, oracle_graph: Mapping[str, Any]) -> float:
    oracle = str(oracle_graph.get("verdict", "uncertain"))
    if verdict == oracle:
        return 1.0
    if verdict == "uncertain":
        return 0.25
    return 0.0
