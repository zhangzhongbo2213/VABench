from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from active_spatial_benchmark import InteractiveRoboTwinEnv
from active_spatial_benchmark.env import robotwin_cwd
from active_spatial_benchmark.pregrasp_tool import LearnedPregraspTool
from active_spatial_benchmark.recovery_planner import (
    TRANSLATION_RELATIONS,
    plan_pregrasp_recovery,
    recovery_confirmation_failures,
)

from evaluate_phase6_pregrasp_execution import camera_value, compact_graph_verdict
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


POLICIES = ("always_execute", "active_no_recovery", "active_recovery")
CASES = (
    {"id": "aligned", "bias_mode": "none", "magnitudes_m": (0.0,)},
    {"id": "along_bias", "bias_mode": "along_object", "magnitudes_m": (0.08, -0.10, 0.12, -0.14)},
    {"id": "across_bias", "bias_mode": "across_object", "magnitudes_m": (0.10, -0.12, 0.14, -0.16)},
    {"id": "vertical_bias", "bias_mode": "vertical", "magnitudes_m": (0.02, 0.03, 0.04, 0.06)},
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diagnose-adjust-reobserve-close-lift recovery without GPT/VLM."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--view-ranker", type=Path, required=True)
    parser.add_argument("--outcome-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[38, 39, 40, 41])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--cases", nargs="+", choices=[row["id"] for row in CASES])
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--max-additional-views", type=int, default=3)
    parser.add_argument("--max-recovery-steps", type=int, default=4)
    parser.add_argument("--max-translation-step-m", type=float, default=0.045)
    parser.add_argument("--required-confidence", type=float, default=0.75)
    parser.add_argument("--lift-m", type=float, default=0.12)
    parser.add_argument("--disable-outcome-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-safety-violations", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [row for row in CASES if not args.cases or row["id"] in set(args.cases)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tool = LearnedPregraspTool(
        args.checkpoint,
        calibration=args.calibration,
        device=device,
        required_confidence=args.required_confidence,
        minimum_evidence_views=2,
        view_ranker_checkpoint=args.view_ranker,
        outcome_checkpoint=args.outcome_checkpoint,
    )
    results = load_results(args.output_dir) if args.resume else []
    if args.retry_safety_violations:
        results = [
            row
            for row in results
            if not bool(row.get("recovery_object_motion_safety_violation", False))
        ]
    completed = {
        (int(row["seed"]), str(row["case"]["id"]), str(row["policy"]))
        for row in results
        if "error" not in row
    }
    for seed in args.seeds:
        for case in selected_cases:
            resolved = resolve_case(case, seed)
            for policy in args.policies:
                key = (int(seed), str(resolved["id"]), str(policy))
                if key in completed:
                    print(f"reuse seed={seed} case={resolved['id']} policy={policy}", flush=True)
                    continue
                trial_dir = args.output_dir / f"seed{seed}" / resolved["id"] / policy
                print(
                    f"run seed={seed} case={resolved['id']} bias={resolved['bias_m']:+.3f} policy={policy}",
                    flush=True,
                )
                try:
                    result = run_trial(
                        seed=seed,
                        config=args.config,
                        case=resolved,
                        policy=policy,
                        output_dir=trial_dir,
                        tool=tool,
                        max_additional_views=args.max_additional_views,
                        max_recovery_steps=args.max_recovery_steps,
                        max_translation_step_m=args.max_translation_step_m,
                        lift_m=args.lift_m,
                        allow_outcome_gate=not args.disable_outcome_gate,
                    )
                except Exception as exc:
                    result = {
                        "schema_version": "phase8.recovery_trial_error.v1",
                        "seed": int(seed),
                        "case": resolved,
                        "policy": policy,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    trial_dir.mkdir(parents=True, exist_ok=True)
                    (trial_dir / "trial_report.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                results.append(result)
                if "error" not in result:
                    completed.add(key)
                write_progress(args.output_dir, args, results)

    report = write_progress(args.output_dir, args, results)
    print(f"output_dir: {args.output_dir.resolve()}")
    for policy, metrics in report["policy_metrics"].items():
        retention = (
            f"{metrics['successful_grasp_retention']:.3f}"
            if metrics["successful_baseline_state_count"]
            else "n/a"
        )
        print(
            f"{policy:<20} success={metrics['task_success_rate']:.3f} "
            f"unsafe={metrics['unsafe_execute_count']} retention={retention} "
            f"views={metrics['mean_observed_views']:.2f} recovery={metrics['recovery_success_count']}",
            flush=True,
        )


def resolve_case(case: Mapping[str, Any], seed: int) -> dict[str, Any]:
    magnitudes = tuple(float(value) for value in case["magnitudes_m"])
    index = (int(seed) - 38) % len(magnitudes)
    return {
        "id": str(case["id"]),
        "bias_mode": str(case["bias_mode"]),
        "bias_m": magnitudes[index],
    }


def run_trial(
    *,
    seed: int,
    config: str,
    case: Mapping[str, Any],
    policy: str,
    output_dir: Path,
    tool: LearnedPregraspTool,
    max_additional_views: int,
    max_recovery_steps: int,
    max_translation_step_m: float,
    lift_m: float,
    allow_outcome_gate: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        oracle_pre_action = build_pregrasp_graph(env, object_start_center, world_state_version=1)
        oracle_pre_action["access"] = "oracle/evaluation_only"
        (output_dir / "pre_action_oracle_evaluation_only.json").write_text(
            json.dumps(oracle_pre_action, indent=2), encoding="utf-8"
        )

        cycles: list[dict[str, Any]] = []
        recovery_steps: list[dict[str, Any]] = []
        total_views = 0
        cumulative_recovery_m = 0.0
        world_state_version = 1
        execute_reason = "always_execute"
        execute_action = policy == "always_execute"
        final_result: dict[str, Any] | None = None
        if policy != "always_execute":
            for recovery_index in range(max_recovery_steps + 1):
                cycle = diagnose(
                    env=env,
                    tool=tool,
                    initial_camera_pose=initial_camera_pose,
                    world_state_version=world_state_version,
                    recovery_cycle=recovery_index,
                    cumulative_recovery_m=cumulative_recovery_m,
                    output_dir=output_dir / f"cycle_{recovery_index:02d}",
                    max_additional_views=max_additional_views,
                )
                cycles.append(cycle)
                total_views += int(cycle["observed_view_count"])
                final_result = cycle["final_result"]
                verdict = str(final_result["verdict"])
                outcome = final_result.get("grasp_success_if_execute") or {}
                # A correction along one learned axis can expose or introduce an
                # error on another. Once recovery starts, re-confirm the entire
                # translatable pre-grasp state before allowing close/lift.
                recovered_relations = TRANSLATION_RELATIONS if recovery_steps else ()
                confirmation_failures = recovery_confirmation_failures(
                    final_result,
                    recovered_relations,
                )
                cycle["recovery_confirmation_failures"] = confirmation_failures
                if verdict == "execute" and not confirmation_failures:
                    execute_action = True
                    execute_reason = "pregrasp_gate_execute"
                    break
                if (
                    policy == "active_recovery"
                    and allow_outcome_gate
                    and verdict == "uncertain"
                    and not confirmation_failures
                    and bool(outcome.get("supports_execute", False))
                ):
                    execute_action = True
                    execute_reason = "calibrated_outcome_gate_execute"
                    break
                if policy != "active_recovery" or recovery_index >= max_recovery_steps:
                    execute_reason = f"blocked_{verdict}"
                    break
                plan = plan_pregrasp_recovery(
                    final_result,
                    max_translation_step_m=max_translation_step_m,
                    force_relations=confirmation_failures,
                )
                if plan["status"] != "move":
                    execute_reason = f"recovery_{plan['status']}"
                    recovery_steps.append({"index": recovery_index, "plan": plan, "execution": None})
                    break
                execution = execute_recovery_translation(env, plan["translation_world_m"])
                recovery_steps.append(
                    {"index": recovery_index, "plan": plan, "execution": execution}
                )
                cumulative_recovery_m += float(
                    np.linalg.norm(execution["achieved_translation_world_m"])
                )
                world_state_version += 1
                if not execution["planner_success"]:
                    execute_reason = "recovery_planner_failed"
                    break

        restore_camera_pose(env, initial_camera_pose)
        close_result = None
        lift_result = None
        post_close_graph = None
        post_lift_graph = None
        physical_success = False
        if execute_action:
            close_result = execute_close(env)
            motion_reference = grasp_motion_reference(env)
            post_close_graph = build_verify_grasp_graph(
                env,
                object_start_center,
                world_state_version=world_state_version + 1,
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
                world_state_version=world_state_version + 2,
            )
            post_lift_graph["access"] = "oracle/evaluation_only"
            (output_dir / "post_close_oracle_evaluation_only.json").write_text(
                json.dumps(post_close_graph, indent=2), encoding="utf-8"
            )
            (output_dir / "post_lift_oracle_evaluation_only.json").write_text(
                json.dumps(post_lift_graph, indent=2), encoding="utf-8"
            )
            render_sheet(post_lift_graph, output_dir / "post_lift_spatial_graph.png")

        object_center_final = object_obb(env)[0]
        result = {
            "schema_version": "phase8.recovery_execution_trial.v1",
            "seed": int(seed),
            "case": dict(case),
            "policy": policy,
            "pregrasp_actions": pregrasp_actions,
            "bias_application": bias,
            "initial_oracle_verdict_evaluation_only": oracle_pre_action["verdict"],
            "initial_oracle_failed_relations_evaluation_only": oracle_pre_action["failed_relations"],
            "diagnostic_cycles": cycles,
            "recovery_steps": recovery_steps,
            "recovery_attempted": bool(recovery_steps),
            "recovery_step_count": sum(
                int(item.get("execution") is not None) for item in recovery_steps
            ),
            "observed_view_count": total_views,
            "initial_gate_verdict": (
                cycles[0]["final_result"]["verdict"] if cycles else "always_execute"
            ),
            "final_gate_verdict": (
                final_result["verdict"] if final_result is not None else "always_execute"
            ),
            "final_outcome_advisory": (
                final_result.get("grasp_success_if_execute") if final_result is not None else None
            ),
            "execute_reason": execute_reason,
            "action_executed": execute_action,
            "close_result": close_result,
            "lift_result": lift_result,
            "physical_task_success": physical_success,
            "object_height_change_m": round(
                float(object_center_final[2] - object_start_center[2]), 6
            ),
            "post_close_verify_grasp": compact_graph_verdict(post_close_graph),
            "post_lift_verify_grasp": compact_graph_verdict(post_lift_graph),
            "recovery_object_motion_safety_violation": any(
                float(item["execution"]["object_translation_m"]) > 0.005
                for item in recovery_steps
                if item.get("execution") is not None
            ),
            "anti_cheating": {
                "gpt_or_vlm_used": False,
                "candidate_images_rendered_before_selection": False,
                "oracle_pregrasp_graph_used_by_policy": False,
                "oracle_contacts_or_object_pose_used_by_policy": False,
                "recovery_plan_inference_visible_only": True,
                "old_belief_reused_after_robot_motion": False,
                "learned_inputs": [
                    "rgb",
                    "metric_depth",
                    "camera_calibration",
                    "robot_kinematics",
                    "fused_relation_measurements",
                ],
            },
        }
        (output_dir / "trial_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        write_recovery_visualization(output_dir / "recovery_trajectory.png", result, output_dir)
        return result
    finally:
        env.close()


def diagnose(
    *,
    env: InteractiveRoboTwinEnv,
    tool: LearnedPregraspTool,
    initial_camera_pose: Any,
    world_state_version: int,
    recovery_cycle: int = 0,
    cumulative_recovery_m: float = 0.0,
    output_dir: Path,
    max_additional_views: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "observed_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    restore_camera_pose(env, initial_camera_pose)
    before = world_fingerprint(env)
    candidates = camera_pose_catalog(env, initial_camera_pose)
    capture = capture_inference_view(env, "current", initial_camera_pose)
    robot = live_robot_kinematics(env, capture, world_state_version=world_state_version)
    tool.start_query(
        robot_kinematics=robot,
        world_state_version=world_state_version,
        query_context={
            "recovery_cycle": float(recovery_cycle),
            "cumulative_recovery_translation_m": float(cumulative_recovery_m),
        },
    )
    history = []
    final_result = None
    for frame_id in range(max_additional_views + 1):
        result = tool.observe(
            rgb=capture["rgb"],
            depth_m=capture["depth_mm"] / 1000.0,
            camera=camera_value(capture),
            robot_kinematics=robot,
            view=str(capture["view"]),
            frame_id=world_state_version * 100 + frame_id,
            candidates=candidates,
        )
        image_path = views_dir / f"{frame_id:02d}_{capture['view']}.png"
        depth_path = views_dir / f"{frame_id:02d}_{capture['view']}_depth_mm.npy"
        depth_m_path = views_dir / f"{frame_id:02d}_{capture['view']}_depth_m.npy"
        camera_path = views_dir / f"{frame_id:02d}_{capture['view']}_camera.json"
        imageio.imwrite(image_path, capture["rgb"])
        np.save(depth_path, capture["depth_mm"])
        np.save(depth_m_path, capture["depth_mm"] / 1000.0)
        camera_path.write_text(
            json.dumps(
                {
                    **{
                        key: np.asarray(value).tolist()
                        for key, value in camera_value(capture).items()
                    },
                    "view": str(capture["view"]),
                    "access": "inference_visible",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history.append(
            {
                "frame_id": int(frame_id),
                "view": str(capture["view"]),
                "image": str(image_path.resolve()),
                "depth": str(depth_path.resolve()),
                "depth_m": str(depth_m_path.resolve()),
                "camera": str(camera_path.resolve()),
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "relations": result["relations"],
                "grasp_success_if_execute": result.get("grasp_success_if_execute"),
                "candidate_view_scores": result["candidate_view_scores"],
                "belief_graph": result["belief_graph"],
                "evidence_views": result["evidence_views"],
                "missing_evidence": result["missing_evidence"],
                "recommended_action": result["recommended_action"],
            }
        )
        final_result = result
        if result["verdict"] in {"execute", "adjust"} or frame_id >= max_additional_views:
            break
        ranking = result["candidate_view_scores"]
        if not ranking:
            break
        capture = capture_inference_view(env, str(ranking[0]["view"]), initial_camera_pose)
        robot = live_robot_kinematics(env, capture, world_state_version=world_state_version)
    restore_camera_pose(env, initial_camera_pose)
    delta = fingerprint_delta(before, world_fingerprint(env))
    if delta > 1e-6:
        raise RuntimeError(f"world changed during diagnostic cycle: {delta}")
    assert final_result is not None
    return {
        "schema_version": "phase8.diagnostic_cycle.v1",
        "world_state_version": int(world_state_version),
        "recovery_cycle": int(recovery_cycle),
        "cumulative_recovery_translation_m": round(float(cumulative_recovery_m), 6),
        "observed_view_count": len(history),
        "observed_view_sequence": [item["view"] for item in history],
        "history": history,
        "final_result": compact_tool_result(final_result),
        "world_frozen": True,
        "world_fingerprint_delta": delta,
    }


def execute_recovery_translation(
    env: InteractiveRoboTwinEnv,
    translation_world_m: list[float],
) -> dict[str, Any]:
    translation = np.asarray(translation_world_m, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("recovery translation must be a finite 3-vector")
    arm_pose_before = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    object_before = object_obb(env)[0]
    target = arm_pose_before.copy()
    target[:3] += translation
    with robotwin_cwd():
        planner_success = bool(
            env.task.move(env.task.move_to_pose("right", target.tolist()), save_freq=None)
        )
        env.task._update_render()
    arm_pose_after = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    object_after = object_obb(env)[0]
    achieved = arm_pose_after[:3] - arm_pose_before[:3]
    return {
        "requested_translation_world_m": vector(translation),
        "achieved_translation_world_m": vector(achieved),
        "translation_residual_m": round(float(np.linalg.norm(translation - achieved)), 6),
        "planner_success": planner_success,
        "object_translation_m": round(float(np.linalg.norm(object_after - object_before)), 6),
    }


def compact_tool_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result["schema_version"],
        "query": result["query"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "relations": result["relations"],
        "evidence_frames": result["evidence_frames"],
        "evidence_views": result["evidence_views"],
        "missing_evidence": result["missing_evidence"],
        "recommended_action": result["recommended_action"],
        "stop_reason": result["stop_reason"],
        "candidate_view_scores": result["candidate_view_scores"],
        "grasp_success_if_execute": result.get("grasp_success_if_execute"),
        "belief_graph": result["belief_graph"],
    }


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "partial_results.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value.get("results", []))


def write_progress(
    output_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    partial = {
        "schema_version": "phase8.recovery_partial_results.v1",
        "settings": settings(args),
        "results": results,
    }
    (output_dir / "partial_results.json").write_text(
        json.dumps(partial, indent=2), encoding="utf-8"
    )
    report = summarize(results, args)
    (output_dir / "recovery_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_summary_image(output_dir / "recovery_validation_summary.png", report)
    return report


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    valid = [row for row in results if "error" not in row]
    errors = [row for row in results if "error" in row]
    baseline = {
        (int(row["seed"]), str(row["case"]["id"])): bool(row["physical_task_success"])
        for row in valid
        if row["policy"] == "always_execute"
    }
    policy_metrics = {}
    for policy in args.policies:
        rows = [row for row in valid if row["policy"] == policy]
        baseline_pairs = [
            (row, baseline[(int(row["seed"]), str(row["case"]["id"]))])
            for row in rows
            if (int(row["seed"]), str(row["case"]["id"])) in baseline
        ]
        success_available = sum(int(label) for _, label in baseline_pairs)
        retained = sum(int(label and row["physical_task_success"]) for row, label in baseline_pairs)
        failed_available = sum(int(not label) for _, label in baseline_pairs)
        unsafe = sum(
            int(row["action_executed"] and not row["physical_task_success"])
            for row in rows
        )
        policy_metrics[policy] = {
            "trial_count": len(rows),
            "task_success_rate": mean_bool(rows, "physical_task_success"),
            "action_execute_rate": mean_bool(rows, "action_executed"),
            "unsafe_execute_count": unsafe,
            "unsafe_execute_rate": unsafe / max(len(rows), 1),
            "successful_baseline_state_count": success_available,
            "successful_grasp_retention": retained / max(success_available, 1),
            "failed_baseline_state_count": failed_available,
            "mean_observed_views": float(
                np.mean([row["observed_view_count"] for row in rows]) if rows else 0.0
            ),
            "mean_recovery_steps": float(
                np.mean([row["recovery_step_count"] for row in rows]) if rows else 0.0
            ),
            "recovery_attempt_count": sum(int(row["recovery_attempted"]) for row in rows),
            "recovery_success_count": sum(
                int(row["recovery_attempted"] and row["physical_task_success"])
                for row in rows
            ),
            "initially_blocked_recovered_count": sum(
                int(
                    row["initial_gate_verdict"] != "execute"
                    and row["physical_task_success"]
                )
                for row in rows
            ),
            "recovery_object_motion_safety_violations": sum(
                int(row["recovery_object_motion_safety_violation"]) for row in rows
            ),
            "initial_gate_distribution": dict(Counter(row["initial_gate_verdict"] for row in rows)),
            "execute_reason_distribution": dict(Counter(row["execute_reason"] for row in rows)),
        }
    expected = len(args.seeds) * len(args.policies) * len(args.cases or CASES)
    audit_errors = []
    if errors:
        audit_errors.append(f"{len(errors)} trial(s) failed")
    if len(valid) != expected:
        audit_errors.append(f"incomplete evaluation: {len(valid)}/{expected} trials")
    for row in valid:
        if any(not cycle["world_frozen"] for cycle in row["diagnostic_cycles"]):
            audit_errors.append(f"world changed during diagnosis: {row['seed']} {row['case']['id']} {row['policy']}")
        if not row["anti_cheating"]["recovery_plan_inference_visible_only"]:
            audit_errors.append(f"recovery leakage contract failed: {row['seed']} {row['case']['id']}")
        if row["recovery_object_motion_safety_violation"]:
            audit_errors.append(
                f"object moved during recovery: {row['seed']} {row['case']['id']} {row['policy']}"
            )
    return {
        "schema_version": "phase8.recovery_validation_report.v1",
        "settings": settings(args),
        "completed_trials": len(valid),
        "expected_trials": expected,
        "failed_trials": len(errors),
        "policy_metrics": policy_metrics,
        "case_outcomes": [
            {
                "seed": row["seed"],
                "case": row["case"],
                "policy": row["policy"],
                "initial_gate": row["initial_gate_verdict"],
                "recovery_steps": row["recovery_step_count"],
                "execute_reason": row["execute_reason"],
                "executed": row["action_executed"],
                "success": row["physical_task_success"],
                "views": row["observed_view_count"],
            }
            for row in valid
        ],
        "errors": errors,
        "audit": {"passed": not audit_errors, "errors": audit_errors},
    }


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seeds": list(args.seeds),
        "policies": list(args.policies),
        "cases": list(args.cases or [row["id"] for row in CASES]),
        "max_additional_views": int(args.max_additional_views),
        "max_recovery_steps": int(args.max_recovery_steps),
        "max_translation_step_m": float(args.max_translation_step_m),
        "lift_m": float(args.lift_m),
        "outcome_gate_enabled": not bool(args.disable_outcome_gate),
    }


def mean_bool(rows: list[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows]) if rows else 0.0)


def write_recovery_visualization(path: Path, report: Mapping[str, Any], output_dir: Path) -> None:
    images = []
    labels = []
    for cycle_index, cycle in enumerate(report["diagnostic_cycles"]):
        history = cycle["history"]
        if not history:
            continue
        selected = [history[0]] if len(history) == 1 else [history[0], history[-1]]
        for item in selected:
            image = Image.open(item["image"]).convert("RGB").resize((480, 360), Image.Resampling.BILINEAR)
            images.append(image)
            labels.append(
                f"cycle {cycle_index} | {item['view']} | {item['verdict']} {item['confidence']:.3f}"
            )
    if not images:
        return
    width = 960
    rows = (len(images) + 1) // 2
    sheet = Image.new("RGB", (width, rows * 410 + 90), (246, 247, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % 2) * 480
        y = (index // 2) * 410
        sheet.paste(image, (x, y))
        draw.rectangle((x, y + 360, x + 480, y + 410), fill=(16, 23, 29))
        draw.text((x + 12, y + 374), label, fill=(245, 247, 248))
    footer_y = rows * 410
    draw.text(
        (14, footer_y + 20),
        f"policy={report['policy']} recovery_steps={report['recovery_step_count']} "
        f"executed={report['action_executed']} success={report['physical_task_success']}",
        fill=(23, 33, 43),
    )
    sheet.save(path)


def write_summary_image(path: Path, report: Mapping[str, Any]) -> None:
    policies = tuple(report.get("settings", {}).get("policies", POLICIES))
    width, height = 1500, max(260, 190 + 105 * len(policies))
    image = Image.new("RGB", (width, height), (246, 247, 245))
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), "Phase 8: diagnose -> adjust -> re-observe -> close -> lift", fill=(20, 28, 36))
    headers = ("policy", "success", "unsafe", "retention", "views", "recovery", "object motion")
    xs = (24, 310, 480, 630, 830, 980, 1190)
    for x, header in zip(xs, headers):
        draw.text((x, 78), header, fill=(75, 85, 94))
    for row_index, policy in enumerate(policies):
        metrics = report["policy_metrics"].get(policy, {})
        y = 118 + row_index * 105
        fill = (231, 236, 238) if row_index % 2 == 0 else (240, 242, 242)
        draw.rectangle((14, y - 15, width - 14, y + 70), fill=fill)
        baseline_count = int(metrics.get("successful_baseline_state_count", 0))
        values = (
            policy,
            f"{100 * metrics.get('task_success_rate', 0.0):.1f}%",
            str(metrics.get("unsafe_execute_count", 0)),
            (
                f"{100 * metrics.get('successful_grasp_retention', 0.0):.1f}%"
                if baseline_count
                else "n/a"
            ),
            f"{metrics.get('mean_observed_views', 0.0):.2f}",
            f"{metrics.get('recovery_success_count', 0)}/{metrics.get('recovery_attempt_count', 0)}",
            str(metrics.get("recovery_object_motion_safety_violations", 0)),
        )
        for x, value in zip(xs, values):
            draw.text((x, y + 10), value, fill=(20, 28, 36))
    audit = report["audit"]
    draw.text(
        (24, height - 52),
        f"audit={'PASS' if audit['passed'] else 'FAIL'} | completed={report['completed_trials']}/{report['expected_trials']}",
        fill=(23, 133, 83) if audit["passed"] else (196, 73, 64),
    )
    image.save(path)


def vector(value: np.ndarray) -> list[float]:
    return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]


if __name__ == "__main__":
    main()
