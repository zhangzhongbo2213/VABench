"""Reproducible split and coverage audit for the Phase 10 tenfold dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


BASELINE_PAIRED_STATES = {
    "train": 37,
    "validation": 5,
    "calibration": 14,
    "test": 32,
}


def inclusive_range(first: int, last: int) -> list[int]:
    return list(range(first, last + 1))


def build_phase10_10x_plan() -> dict[str, Any]:
    """Return the fixed scene matrix used for the tenfold experiment."""

    splits = {
        "train": inclusive_range(42, 101),
        "validation": inclusive_range(102, 111),
        "calibration": inclusive_range(112, 131),
        "test": inclusive_range(132, 179),
    }
    expected_states_per_seed = 88.0 / 12.0
    split_rows = {}
    for name, seeds in splits.items():
        baseline = BASELINE_PAIRED_STATES[name]
        split_rows[name] = {
            "seeds": seeds,
            "seed_count": len(seeds),
            "baseline_paired_states": baseline,
            "minimum_paired_states": baseline * 10,
            "expected_paired_states": round(len(seeds) * expected_states_per_seed),
        }
    plan = {
        "schema_version": "phase10.tenfold_collection_plan.v1",
        "task": "grasp_single_pen",
        "config": "demo_clean",
        "scale_factor": 10,
        "source_dataset": "runs/phase10_10x/onpolicy_seed42_187_v1",
        "trajectory_dataset": "runs/phase10_10x/trajectory_seed42_187_v1",
        "scene_randomization": {
            "pen_position_x_m": [0.18, 0.26],
            "pen_position_y_m": [-0.20, -0.10],
            "pen_yaw_deg": [0.0, 30.0],
            "randomization_source": "RoboTwin task seed",
        },
        "cases": [
            {
                "id": "aligned",
                "bias_mode": "none",
                "bias_m": [0.0],
            },
            {
                "id": "along_bias",
                "bias_mode": "along_object",
                "bias_m": [0.08, -0.10, 0.12, -0.14],
            },
            {
                "id": "across_bias",
                "bias_mode": "across_object",
                "bias_m": [0.10, -0.12, 0.14, -0.16],
            },
            {
                "id": "vertical_bias",
                "bias_mode": "vertical",
                "bias_m": [0.02, 0.03, 0.04, 0.06],
            },
        ],
        "paired_state_filter": {
            "recovery_cycles": [0, 1, 2],
            "view_steps": [0, 2],
            "candidate_limit": 2,
        },
        "candidate_views": [
            "current",
            "topdown",
            "side",
            "front_side_45",
            "side_top_45",
            "oblique_45",
        ],
        "splits": split_rows,
        "reserve_seeds": inclusive_range(180, 187),
        "acceptance": {
            "split_paired_state_floor": "10x the Phase 10 v1 split count",
            "max_pairwise_start_fingerprint_delta": 0.001,
            "validation_error_count": 0,
            "seed_overlap_between_splits": 0,
        },
    }
    validate_phase10_10x_plan(plan)
    return plan


def validate_phase10_10x_plan(plan: Mapping[str, Any]) -> None:
    seen: set[int] = set()
    for name, row in plan["splits"].items():
        seeds = [int(seed) for seed in row["seeds"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed inside {name} split")
        overlap = seen & set(seeds)
        if overlap:
            raise ValueError(f"seed leakage in {name}: {sorted(overlap)}")
        seen.update(seeds)
        expected_floor = BASELINE_PAIRED_STATES[name] * int(plan["scale_factor"])
        if int(row["minimum_paired_states"]) != expected_floor:
            raise ValueError(f"incorrect tenfold floor for {name}")
    reserve = {int(seed) for seed in plan.get("reserve_seeds", ())}
    overlap = seen & reserve
    if overlap:
        raise ValueError(f"reserve seed leakage: {sorted(overlap)}")


def seed_shards(seeds: Iterable[int], shard_size: int) -> list[list[int]]:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    values = [int(seed) for seed in seeds]
    return [values[index : index + shard_size] for index in range(0, len(values), shard_size)]


def audit_trajectory_dataset(
    dataset_root: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Count valid paired state files without loading training tensors."""

    root = Path(dataset_root).resolve()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed = []
    for path in sorted(root.glob("branches/**/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            sample_id = str(value["sample_id"])
            seed = int(value["seed"])
            forced_view = str(value["forced_view"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            malformed.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
            continue
        grouped[sample_id].append(
            {
                "seed": seed,
                "forced_view": forced_view,
                "inference_visible": value.get("inference_visible"),
                "path": str(path),
            }
        )

    paired_by_seed: Counter[int] = Counter()
    duplicate_view_states = []
    cross_seed_states = []
    inference_context_mismatch_states = []
    shared_fields = ("visited_views", "edge_uncertainty", "state_context", "belief_before")
    for sample_id, rows in grouped.items():
        views = [row["forced_view"] for row in rows]
        seeds = {int(row["seed"]) for row in rows}
        if len(views) != len(set(views)):
            duplicate_view_states.append(sample_id)
            continue
        if len(seeds) != 1:
            cross_seed_states.append(sample_id)
            continue
        if all(row["inference_visible"] is not None for row in rows):
            inference = rows[0]["inference_visible"]
            if any(
                any(
                    row["inference_visible"].get(field) != inference.get(field)
                    for field in shared_fields
                )
                for row in rows[1:]
            ):
                inference_context_mismatch_states.append(sample_id)
                continue
        if len(rows) >= 2:
            paired_by_seed[next(iter(seeds))] += 1

    split_metrics = {}
    all_passed = (
        not malformed
        and not duplicate_view_states
        and not cross_seed_states
        and not inference_context_mismatch_states
    )
    for name, row in plan["splits"].items():
        seeds = [int(seed) for seed in row["seeds"]]
        actual = sum(paired_by_seed[seed] for seed in seeds)
        minimum = int(row["minimum_paired_states"])
        passed = actual >= minimum
        split_metrics[name] = {
            "seed_count": len(seeds),
            "paired_state_count": actual,
            "minimum_paired_states": minimum,
            "scale_vs_baseline": actual / max(int(row["baseline_paired_states"]), 1),
            "passed": passed,
            "missing_seed_count": sum(paired_by_seed[seed] == 0 for seed in seeds),
        }
        all_passed = all_passed and passed

    return {
        "schema_version": "phase10.tenfold_dataset_audit.v1",
        "dataset_root": str(root),
        "paired_state_count": int(sum(paired_by_seed.values())),
        "paired_state_count_by_seed": {
            str(seed): int(count) for seed, count in sorted(paired_by_seed.items())
        },
        "splits": split_metrics,
        "reserve": {
            "seeds": [int(seed) for seed in plan.get("reserve_seeds", ())],
            "paired_state_count": sum(
                paired_by_seed[int(seed)] for seed in plan.get("reserve_seeds", ())
            ),
        },
        "malformed_branch_count": len(malformed),
        "malformed_branches": malformed,
        "duplicate_view_state_count": len(duplicate_view_states),
        "cross_seed_state_count": len(cross_seed_states),
        "inference_context_mismatch_state_count": len(
            inference_context_mismatch_states
        ),
        "inference_context_mismatch_states": inference_context_mismatch_states,
        "passed": all_passed,
    }
