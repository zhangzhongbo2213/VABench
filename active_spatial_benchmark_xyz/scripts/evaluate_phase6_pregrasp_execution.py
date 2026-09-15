from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from active_spatial_benchmark import InteractiveRoboTwinEnv
from active_spatial_benchmark.env import robotwin_cwd
from active_spatial_benchmark.pregrasp_tool import LearnedPregraspTool

from render_spatial_graph_only import render_sheet
from run_phase1_5_pregrasp_active_loop import (
    apply_pregrasp_bias,
    build_pregrasp_graph,
    camera_pose_catalog,
    capture_inference_view,
    execute_close,
    execute_to_pregrasp,
    live_robot_kinematics,
)
from run_phase1_pen_cases import grasp_motion_reference
from run_phase1_spatial_graph_demo import (
    build_graph as build_verify_grasp_graph,
    clone_camera_pose,
    fingerprint_delta,
    object_obb,
    restore_camera_pose,
    world_fingerprint,
)


POLICIES = (
    "always_execute",
    "single_view",
    "fixed_views",
    "active_rule",
    "active_learned",
)
FIXED_VIEW_SEQUENCE = ("topdown", "side_top_45", "front_side_45")
CASES = (
    {"id": "aligned", "bias_mode": "none", "magnitudes_m": (0.0,)},
    {"id": "along_bias", "bias_mode": "along_object", "magnitudes_m": (0.085, -0.10)},
    {"id": "across_bias", "bias_mode": "across_object", "magnitudes_m": (0.11, -0.13)},
    {"id": "vertical_bias", "bias_mode": "vertical", "magnitudes_m": (0.04, 0.055)},
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned pre-grasp gating through real close-and-lift execution without GPT/VLM."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--view-ranker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 23, 24, 25])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--max-additional-views", type=int, default=3)
    parser.add_argument("--required-confidence", type=float, default=0.75)
    parser.add_argument("--lift-m", type=float, default=0.12)
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials from partial_results.json.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule_tool = LearnedPregraspTool(
        args.checkpoint,
        calibration=args.calibration,
        device=device,
        required_confidence=args.required_confidence,
        minimum_evidence_views=2,
    )
    single_view_tool = LearnedPregraspTool(
        args.checkpoint,
        calibration=args.calibration,
        device=device,
        required_confidence=args.required_confidence,
        minimum_evidence_views=1,
    )
    learned_tool = LearnedPregraspTool(
        args.checkpoint,
        calibration=args.calibration,
        device=device,
        required_confidence=args.required_confidence,
        minimum_evidence_views=2,
        view_ranker_checkpoint=args.view_ranker,
    )

    results = load_partial_results(args.output_dir) if args.resume else []
    completed_keys = {
        (int(row["seed"]), str(row["case"]["id"]), str(row["policy"]))
        for row in results
        if "error" not in row
    }
    for seed in args.seeds:
        for case in CASES:
            resolved_case = resolve_case(case, seed)
            for policy in args.policies:
                trial_dir = args.output_dir / f"seed{seed}" / resolved_case["id"] / policy
                key = (int(seed), str(resolved_case["id"]), str(policy))
                if key in completed_keys:
                    print(
                        f"skipping completed seed={seed} case={resolved_case['id']} policy={policy}",
                        flush=True,
                    )
                    continue
                print(
                    f"running seed={seed} case={resolved_case['id']} "
                    f"bias={resolved_case['bias_m']:+.3f} policy={policy}",
                    flush=True,
                )
                try:
                    result = run_trial(
                        seed=seed,
                        config=args.config,
                        case=resolved_case,
                        policy=policy,
                        output_dir=trial_dir,
                        single_view_tool=single_view_tool,
                        rule_tool=rule_tool,
                        learned_tool=learned_tool,
                        max_additional_views=args.max_additional_views,
                        lift_m=args.lift_m,
                    )
                except Exception as exc:
                    result = {
                        "seed": seed,
                        "case": resolved_case,
                        "policy": policy,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    (trial_dir / "trial_report.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                results.append(result)
                if "error" not in result:
                    completed_keys.add(key)
                write_progress(args.output_dir, args, results)

    report = summarize_results(results, args)
    report["audit"] = audit_results(results, args)
    (args.output_dir / "execution_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_comparison_image(args.output_dir / "execution_validation_comparison.png", report)
    write_representative_overview(args.output_dir / "representative_trials.png", results)
    write_readme(args.output_dir / "README.md", report)
    print(f"output_dir: {args.output_dir.resolve()}")
    for policy, row in report["policy_metrics"].items():
        print(
            f"{policy:<16} success={row['task_success_rate']:.3f} "
            f"unsafe={row['unsafe_execute_rate']:.3f} "
            f"retention={row.get('successful_grasp_retention', 0.0):.3f} "
            f"views={row['mean_observed_views']:.2f}"
        )


def resolve_case(case: Mapping[str, Any], seed: int) -> dict[str, Any]:
    magnitudes = tuple(float(value) for value in case["magnitudes_m"])
    bias_m = magnitudes[int(seed) % len(magnitudes)]
    return {
        "id": str(case["id"]),
        "bias_mode": str(case["bias_mode"]),
        "bias_m": bias_m,
    }


def run_trial(
    *,
    seed: int,
    config: str,
    case: Mapping[str, Any],
    policy: str,
    output_dir: Path,
    single_view_tool: LearnedPregraspTool,
    rule_tool: LearnedPregraspTool,
    learned_tool: LearnedPregraspTool,
    max_additional_views: int,
    lift_m: float,
    force_execute_for_outcome_label: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "observed_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    env = InteractiveRoboTwinEnv(
        task_name="grasp_single_pen",
        config_name=config,
        active_arm="right",
        max_steps=1000,
        output_dir=output_dir / "env",
        save_images=False,
    )
    try:
        env.reset(seed=seed)
        object_start_center = object_obb(env)[0]
        pregrasp_actions = execute_to_pregrasp(env)
        bias = apply_pregrasp_bias(env, str(case["bias_mode"]), float(case["bias_m"]))
        initial_camera_pose = clone_camera_pose(env)
        pre_action_fingerprint = world_fingerprint(env)
        pre_action_truth = build_pregrasp_graph(env, object_start_center, world_state_version=1)
        pre_action_truth["access"] = "oracle/evaluation_only"
        (output_dir / "pre_action_oracle_evaluation_only.json").write_text(
            json.dumps(pre_action_truth, indent=2), encoding="utf-8"
        )

        observed_views: list[str] = []
        gate_history: list[dict[str, Any]] = []
        gate = {
            "verdict": "execute",
            "confidence": 1.0,
            "reason": "always_execute baseline bypasses spatial diagnosis",
            "action": "allow_gripper_close",
        }
        if policy != "always_execute":
            if policy == "single_view":
                tool = single_view_tool
            else:
                tool = learned_tool if policy == "active_learned" else rule_tool
            candidates = camera_pose_catalog(env, initial_camera_pose)
            current_capture = capture_inference_view(env, "current", initial_camera_pose)
            robot = live_robot_kinematics(env, current_capture, world_state_version=1)
            tool.start_query(robot_kinematics=robot, world_state_version=1)
            fixed_index = 0
            for frame_id in range(max_additional_views + 1):
                result = tool.observe(
                    rgb=current_capture["rgb"],
                    depth_m=current_capture["depth_mm"] / 1000.0,
                    camera=camera_value(current_capture),
                    robot_kinematics=robot,
                    view=str(current_capture["view"]),
                    frame_id=frame_id,
                    candidates=candidates,
                )
                observed_views.append(str(current_capture["view"]))
                imageio.imwrite(views_dir / f"{frame_id:02d}_{current_capture['view']}.png", current_capture["rgb"])
                gate = {
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                    "missing_evidence": result["missing_evidence"],
                    "recommended_action": result["recommended_action"],
                    "relations": result["relations"],
                }
                gate_history.append(
                    {
                        "frame_id": frame_id,
                        "view": current_capture["view"],
                        "gate": gate,
                        "candidate_view_scores": result["candidate_view_scores"],
                    }
                )
                if policy == "single_view" or gate["verdict"] in {"execute", "adjust"}:
                    break
                if frame_id >= max_additional_views:
                    break
                if policy == "fixed_views":
                    while fixed_index < len(FIXED_VIEW_SEQUENCE) and FIXED_VIEW_SEQUENCE[fixed_index] in observed_views:
                        fixed_index += 1
                    if fixed_index >= len(FIXED_VIEW_SEQUENCE):
                        break
                    next_view = FIXED_VIEW_SEQUENCE[fixed_index]
                    fixed_index += 1
                else:
                    scores = result["candidate_view_scores"]
                    if not scores:
                        break
                    next_view = str(scores[0]["view"])
                current_capture = capture_inference_view(env, next_view, initial_camera_pose)
                robot = live_robot_kinematics(env, current_capture, world_state_version=1)

        restore_camera_pose(env, initial_camera_pose)
        observation_delta = fingerprint_delta(pre_action_fingerprint, world_fingerprint(env))
        if observation_delta > 1e-6:
            raise RuntimeError(f"world changed during diagnostic observations: {observation_delta}")

        execute_action = gate["verdict"] == "execute"
        physical_probe_executed = execute_action or bool(force_execute_for_outcome_label)
        close_result = None
        lift_result = None
        post_close_graph = None
        post_lift_graph = None
        physical_success = False
        if physical_probe_executed:
            close_result = execute_close(env)
            motion_reference = grasp_motion_reference(env)
            post_close_graph = build_verify_grasp_graph(
                env,
                object_start_center,
                world_state_version=2,
            )
            post_close_graph["access"] = "oracle/evaluation_only"
            with robotwin_cwd():
                lift_plan = env.task.move_by_displacement(arm_tag=env.task.arm_tag, z=float(lift_m))
                lift_ok = bool(env.task.move(lift_plan, save_freq=None))
                env.task._update_render()
                physical_success = bool(env.task.check_success())
            lift_result = {"distance_m": float(lift_m), "planner_success": lift_ok}
            post_lift_graph = build_verify_grasp_graph(
                env,
                object_start_center,
                motion_reference=motion_reference,
                world_state_version=3,
            )
            post_lift_graph["access"] = "oracle/evaluation_only"
            (output_dir / "post_close_oracle_evaluation_only.json").write_text(
                json.dumps(post_close_graph, indent=2), encoding="utf-8"
            )
            (output_dir / "post_lift_oracle_evaluation_only.json").write_text(
                json.dumps(post_lift_graph, indent=2), encoding="utf-8"
            )
            render_sheet(post_lift_graph, output_dir / "post_lift_spatial_graph.png")

        post_capture = capture_inference_view(env, "current", initial_camera_pose)
        imageio.imwrite(output_dir / "post_decision_rgb.png", post_capture["rgb"])
        object_center_final = object_obb(env)[0]
        result = {
            "schema_version": "phase6.pregrasp_execution_trial.v1",
            "seed": int(seed),
            "case": dict(case),
            "policy": policy,
            "pregrasp_actions": pregrasp_actions,
            "bias_application": bias,
            "pre_action_truth": {
                "verdict": pre_action_truth["verdict"],
                "failed_relations": pre_action_truth["failed_relations"],
            },
            "observed_view_sequence": observed_views,
            "observed_view_count": len(observed_views),
            "gate_history": gate_history,
            "final_gate": gate,
            "action_executed": execute_action,
            "physical_probe_executed": physical_probe_executed,
            "forced_outcome_probe": bool(force_execute_for_outcome_label and not execute_action),
            "close_result": close_result,
            "lift_result": lift_result,
            "physical_task_success": physical_success,
            "object_height_change_m": round(float(object_center_final[2] - object_start_center[2]), 6),
            "post_close_verify_grasp": compact_graph_verdict(post_close_graph),
            "post_lift_verify_grasp": compact_graph_verdict(post_lift_graph),
            "world_frozen_during_diagnosis": observation_delta <= 1e-6,
            "world_fingerprint_delta_during_diagnosis": observation_delta,
            "anti_cheating": {
                "gpt_or_vlm_used": False,
                "candidate_images_rendered_before_selection": False,
                "oracle_pregrasp_graph_used_by_policy": False,
                "oracle_contacts_or_object_pose_used_by_policy": False,
                "oracle_fields_used_for_evaluation_only": True,
                "learned_inputs": ["rgb", "metric_depth", "camera_calibration", "robot_kinematics"],
            },
            "artifacts": {
                "pre_action_truth": str((output_dir / "pre_action_oracle_evaluation_only.json").resolve()),
                "post_decision_rgb": str((output_dir / "post_decision_rgb.png").resolve()),
                "post_lift_graph": (
                    str((output_dir / "post_lift_spatial_graph.png").resolve())
                    if post_lift_graph
                    else None
                ),
            },
        }
        (output_dir / "trial_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        env.close()


def camera_value(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "camera_pose_world": capture["camera_pose_world"],
        "intrinsic_cv": capture["intrinsic_cv"],
        "extrinsic_cv": capture["extrinsic_cv"],
    }


def compact_graph_verdict(graph: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if graph is None:
        return None
    return {
        "verdict": graph["verdict"],
        "confidence": graph["confidence"],
        "missing_evidence": graph["missing_evidence"],
        "relations": {
            str(edge["id"]): {
                "state": edge["state"],
                "probability": edge.get("probability"),
            }
            for edge in graph["edges"]
        },
    }


def summarize_results(results: Iterable[Mapping[str, Any]], args: Any | None = None) -> dict[str, Any]:
    completed = [row for row in results if "error" not in row]
    failures = [row for row in results if "error" in row]
    baseline_by_key = {
        trial_key(row): bool(row["physical_task_success"])
        for row in completed
        if row["policy"] == "always_execute"
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[str(row["policy"])].append(row)

    policy_metrics = {}
    for policy, rows in grouped.items():
        executed = [row for row in rows if row["action_executed"]]
        successes = [row for row in rows if row["physical_task_success"]]
        unsafe = [row for row in executed if not row["physical_task_success"]]
        paired = [(row, baseline_by_key[trial_key(row)]) for row in rows if trial_key(row) in baseline_by_key]
        possible_successes = sum(int(value) for _, value in paired)
        possible_failures = sum(int(not value) for _, value in paired)
        correct_blocks = sum(int(not row["action_executed"] and not value) for row, value in paired)
        missed_successes = sum(int(not row["action_executed"] and value) for row, value in paired)
        retained_successes = sum(int(row["physical_task_success"] and value) for row, value in paired)
        correct_decisions = sum(int(bool(row["action_executed"]) == bool(value)) for row, value in paired)
        true_positive_rate = ratio(retained_successes, possible_successes)
        true_negative_rate = ratio(correct_blocks, possible_failures)
        policy_metrics[policy] = {
            "trials": len(rows),
            "action_execution_rate": ratio(len(executed), len(rows)),
            "task_success_rate": ratio(len(successes), len(rows)),
            "execution_precision": ratio(len(successes), len(executed)),
            "unsafe_execute_count": len(unsafe),
            "unsafe_execute_rate": ratio(len(unsafe), len(rows)),
            "correct_block_count": correct_blocks,
            "failed_grasp_block_rate": ratio(correct_blocks, possible_failures),
            "missed_success_count": missed_successes,
            "successful_grasp_retention": true_positive_rate,
            "decision_accuracy": ratio(correct_decisions, len(paired)),
            "balanced_decision_accuracy": round((true_positive_rate + true_negative_rate) / 2.0, 6),
            "mean_observed_views": round(float(np.mean([row["observed_view_count"] for row in rows])), 4),
            "mean_additional_views": round(
                float(np.mean([max(0, int(row["observed_view_count"]) - 1) for row in rows])), 4
            ),
            "verdict_distribution": dict(Counter(row["final_gate"]["verdict"] for row in rows)),
        }

    by_case = {}
    for case_id in sorted({str(row["case"]["id"]) for row in completed}):
        by_case[case_id] = {}
        for policy in grouped:
            rows = [row for row in grouped[policy] if row["case"]["id"] == case_id]
            if rows:
                by_case[case_id][policy] = {
                    "trials": len(rows),
                    "executed": sum(int(row["action_executed"]) for row in rows),
                    "successes": sum(int(row["physical_task_success"]) for row in rows),
                    "verdicts": dict(Counter(row["final_gate"]["verdict"] for row in rows)),
                }

    baseline_rows = [row for row in completed if row["policy"] == "always_execute"]
    pregrasp_truth_outcomes = {}
    for verdict in ("execute", "adjust", "uncertain"):
        rows = [
            row
            for row in baseline_rows
            if row.get("pre_action_truth", {}).get("verdict") == verdict
        ]
        if rows:
            pregrasp_truth_outcomes[verdict] = {
                "trials": len(rows),
                "physical_successes": sum(int(row["physical_task_success"]) for row in rows),
                "physical_success_rate": ratio(
                    sum(int(row["physical_task_success"]) for row in rows), len(rows)
                ),
            }

    return {
        "schema_version": "phase6.pregrasp_execution_validation.v1",
        "task": "grasp_single_pen",
        "query": "verify_pregrasp_then_execute",
        "gpt_or_vlm_used": False,
        "seeds": list(getattr(args, "seeds", [])),
        "policies": list(getattr(args, "policies", grouped.keys())),
        "trial_count": len(completed),
        "failed_trial_count": len(failures),
        "failures": failures,
        "counterfactual_reference": "always_execute physical close-and-lift outcome for the same seed and bias",
        "counterfactual_outcomes": {
            "successful": sum(int(row["physical_task_success"]) for row in baseline_rows),
            "failed": sum(int(not row["physical_task_success"]) for row in baseline_rows),
        },
        "pregrasp_oracle_verdict_vs_physical_outcome": pregrasp_truth_outcomes,
        "policy_metrics": policy_metrics,
        "case_metrics": by_case,
        "limits": [
            "No recovery motion is attempted after adjust or uncertain; blocked trials count as zero task success.",
            "Post-action contact and pose fields are simulator evaluation truth, not inference inputs.",
            "This evaluates one object family and one task; it is not evidence of cross-object generalization.",
        ],
    }


def audit_results(results: Iterable[Mapping[str, Any]], args: Any) -> dict[str, Any]:
    rows = list(results)
    errors = []
    expected_keys = {
        (int(seed), str(case["id"]), str(policy))
        for seed in args.seeds
        for case in CASES
        for policy in args.policies
    }
    completed = [row for row in rows if "error" not in row]
    actual_keys = {
        (int(row["seed"]), str(row["case"]["id"]), str(row["policy"]))
        for row in completed
    }
    missing = sorted(expected_keys - actual_keys)
    duplicates = len(completed) - len(actual_keys)
    if missing:
        errors.append(f"missing {len(missing)} expected trial keys")
    if duplicates:
        errors.append(f"found {duplicates} duplicate completed trials")
    failed_rows = [row for row in rows if "error" in row]
    if failed_rows:
        errors.append(f"found {len(failed_rows)} failed trials")

    unfrozen = [row for row in completed if not row.get("world_frozen_during_diagnosis", False)]
    if unfrozen:
        errors.append(f"world changed during diagnosis in {len(unfrozen)} trials")
    leakage = [
        row
        for row in completed
        if row.get("anti_cheating", {}).get("gpt_or_vlm_used") is not False
        or row.get("anti_cheating", {}).get("oracle_pregrasp_graph_used_by_policy") is not False
        or row.get("anti_cheating", {}).get("oracle_contacts_or_object_pose_used_by_policy") is not False
    ]
    if leakage:
        errors.append(f"anti-cheating contract failed in {len(leakage)} trials")

    executed = [row for row in completed if row.get("action_executed")]
    inconsistent = []
    for row in executed:
        graph = row.get("post_lift_verify_grasp") or {}
        graph_success = graph.get("verdict") == "true"
        if graph_success != bool(row.get("physical_task_success")):
            inconsistent.append(row)
    if inconsistent:
        errors.append(f"task success and post-lift graph disagree in {len(inconsistent)} trials")

    return {
        "passed": not errors,
        "errors": errors,
        "expected_trial_count": len(expected_keys),
        "completed_trial_count": len(completed),
        "unique_trial_count": len(actual_keys),
        "missing_trial_count": len(missing),
        "failed_trial_count": len(failed_rows),
        "world_frozen_trial_count": len(completed) - len(unfrozen),
        "anti_cheating_pass_count": len(completed) - len(leakage),
        "executed_trial_count": len(executed),
        "post_lift_consistency_count": len(executed) - len(inconsistent),
    }


def trial_key(row: Mapping[str, Any]) -> tuple[int, str, float]:
    return (int(row["seed"]), str(row["case"]["id"]), round(float(row["case"]["bias_m"]), 6))


def ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def write_progress(output_dir: Path, args: Any, results: list[Mapping[str, Any]]) -> None:
    (output_dir / "partial_results.json").write_text(
        json.dumps({"seeds": args.seeds, "policies": args.policies, "results": results}, indent=2),
        encoding="utf-8",
    )


def load_partial_results(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "partial_results.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    results = value.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"invalid partial results: {path}")
    return [dict(row) for row in results]


def write_comparison_image(path: Path, report: Mapping[str, Any]) -> None:
    rows = [(policy, report["policy_metrics"][policy]) for policy in POLICIES if policy in report["policy_metrics"]]
    width = 1320
    row_height = 56
    height = 100 + row_height * len(rows)
    image = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Phase 6: real close-and-lift validation (no GPT/VLM)", fill=(20, 24, 28))
    headers = (
        "policy",
        "decision acc",
        "balanced acc",
        "success",
        "exec precision",
        "unsafe",
        "fail block",
        "success retain",
        "views",
    )
    xs = (24, 215, 340, 475, 580, 725, 815, 930, 1090)
    for x, value in zip(xs, headers):
        draw.text((x, 62), value, fill=(70, 74, 78))
    for index, (policy, values) in enumerate(rows):
        y = 92 + index * row_height
        fill = (232, 237, 240) if index % 2 == 0 else (242, 244, 245)
        draw.rectangle((12, y - 8, width - 12, y + 38), fill=fill)
        cells = (
            policy,
            f"{100 * values['decision_accuracy']:.1f}%",
            f"{100 * values['balanced_decision_accuracy']:.1f}%",
            f"{100 * values['task_success_rate']:.1f}%",
            f"{100 * values['execution_precision']:.1f}%",
            f"{values['unsafe_execute_count']}",
            f"{100 * values['failed_grasp_block_rate']:.1f}%",
            f"{100 * values['successful_grasp_retention']:.1f}%",
            f"{values['mean_observed_views']:.2f}",
        )
        for x, value in zip(xs, cells):
            draw.text((x, y), value, fill=(25, 30, 34))
    image.save(path)


def write_representative_overview(path: Path, results: Iterable[Mapping[str, Any]]) -> None:
    completed = [row for row in results if "error" not in row]
    if not completed:
        return
    first_seed = min(int(row["seed"]) for row in completed)
    selected = [
        row for row in completed
        if int(row["seed"]) == first_seed and row["policy"] in {"always_execute", "active_learned"}
    ]
    panels = []
    for row in selected:
        image_path = Path(row["artifacts"]["post_decision_rgb"])
        if not image_path.is_absolute():
            continue
        panel = Image.open(image_path).convert("RGB").resize((512, 288))
        canvas = Image.new("RGB", (512, 344), (18, 22, 26))
        canvas.paste(panel, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 296), f"{row['case']['id']} | {row['policy']}", fill=(255, 255, 255))
        draw.text(
            (10, 318),
            f"gate={row['final_gate']['verdict']} execute={row['action_executed']} success={row['physical_task_success']}",
            fill=(205, 215, 222),
        )
        panels.append(canvas)
    if not panels:
        return
    sheet = Image.new("RGB", (1024, 344 * ((len(panels) + 1) // 2)), (10, 12, 14))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 512, (index // 2) * 344))
    sheet.save(path)


def write_readme(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Phase 6 pre-grasp execution validation",
        "",
        "No GPT or VLM is used. Each execute decision performs a real gripper close and 0.12 m lift in RoboTwin.",
        "Simulator object poses and contacts are used only after the policy decision for evaluation.",
        "",
        "| Policy | Decision accuracy | Balanced accuracy | Task success | Execution precision | Unsafe executes | Failed-grasp block | Success retention | Mean views |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        if policy not in report["policy_metrics"]:
            continue
        row = report["policy_metrics"][policy]
        lines.append(
            f"| `{policy}` | {100 * row['decision_accuracy']:.1f}% | "
            f"{100 * row['balanced_decision_accuracy']:.1f}% | "
            f"{100 * row['task_success_rate']:.1f}% | "
            f"{100 * row['execution_precision']:.1f}% | {row['unsafe_execute_count']} | "
            f"{100 * row['failed_grasp_block_rate']:.1f}% | "
            f"{100 * row['successful_grasp_retention']:.1f}% | {row['mean_observed_views']:.2f} |"
        )
    lines.extend(["", "See `execution_validation_report.json` for per-case metrics and limitations."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
