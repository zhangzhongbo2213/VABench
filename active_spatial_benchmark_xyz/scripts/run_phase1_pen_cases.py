from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_spatial_benchmark import InteractiveRoboTwinEnv
from active_spatial_benchmark.env import robotwin_cwd

from render_spatial_graph_only import render_perspective, render_sheet
from run_phase1_spatial_graph_demo import (
    CANDIDATE_VIEWS,
    build_graph,
    capture_view,
    clone_camera_pose,
    draw_graph_overlay,
    fingerprint_delta,
    grasp_motion_reference,
    grasp_target_actor,
    inferred_current_edge_uncertainty,
    object_obb,
    restore_camera_pose,
    score_views,
    world_fingerprint,
    write_contact_sheet,
    write_view_artifacts,
)


CASES = (
    {
        "id": "01_aligned_open",
        "phase": "aligned_open_at_grasp",
        "description": "The fingers straddle the pen at the final grasp pose, but the gripper is still open.",
    },
    {
        "id": "02_closed_before_lift",
        "phase": "closed_before_lift",
        "description": "The gripper is closed around the pen while it is still supported by the table.",
    },
    {
        "id": "03_lifted_success",
        "phase": "lifted_success",
        "description": "The closed gripper has lifted the pen, adding temporal rigid-motion evidence.",
    },
    {
        "id": "04_empty_close",
        "phase": "closed_at_pregrasp_offset",
        "description": "The gripper closes at the pre-grasp offset before reaching the pen.",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze several fixed grasp_single_pen states with Phase 1 graphs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--output-dir", default="runs/phase1_spatial_graph/grasp_single_pen_seed0_cases")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for version, case in enumerate(CASES, start=1):
        results.append(run_case(case, args.seed, args.config, output_dir / case["id"], version))

    write_comparison(output_dir / "pen_cases_comparison.png", results)
    write_summary(output_dir / "README.md", args.seed, results)
    (output_dir / "cases_summary.json").write_text(
        json.dumps({"task": "grasp_single_pen", "seed": args.seed, "cases": [result["summary"] for result in results]}, indent=2),
        encoding="utf-8",
    )

    print(f"output_dir: {output_dir}")
    for result in results:
        summary = result["summary"]
        print(
            f"  {summary['case']:<24} verdict={summary['verdict']:<9} "
            f"confidence={summary['confidence']:.2f} selected={summary['selected_view']}"
        )


def run_case(
    case: dict[str, str],
    seed: int,
    config: str,
    output_dir: Path,
    world_state_version: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
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
        start_center = object_obb(env)[0]
        actions = grasp_actions(env)
        executed, motion_reference = execute_case(env, case["phase"], actions)
        frozen_before = world_fingerprint(env)
        graph = build_graph(
            env,
            start_center,
            motion_reference=motion_reference,
            world_state_version=world_state_version,
        )
        graph["diagnostic_context"] = {
            "task": "grasp_single_pen",
            "object": "pen",
            "phase": case["phase"],
            "case_description": case["description"],
        }
        graph_path = output_dir / "oracle_spatial_graph.json"
        graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

        initial_pose = clone_camera_pose(env)
        captures = [capture_view(env, "current", initial_pose, graph)]
        current_uncertainty = inferred_current_edge_uncertainty(graph, captures[0])
        for view in CANDIDATE_VIEWS[1:]:
            captures.append(capture_view(env, view, initial_pose, graph))
        restore_camera_pose(env, initial_pose)

        frozen_delta = fingerprint_delta(frozen_before, world_fingerprint(env))
        if frozen_delta > 1e-6:
            raise RuntimeError(f"world state changed during {case['id']} candidate rendering: {frozen_delta}")
        scores = score_views(captures, graph, current_uncertainty)
        score_by_view = {row["view"]: row for row in scores}
        selected_view = max(scores, key=lambda row: row["predicted"]["utility"])["view"]
        oracle_best = max(scores, key=lambda row: row["visibility_aware_oracle_proxy"])["view"]
        next_tool = orchestrator_next_tool(graph, selected_view)
        for capture in captures:
            capture["selected_by_rule"] = capture["view"] == selected_view
            capture["oracle_best"] = capture["view"] == oracle_best
            write_view_artifacts(views_dir, capture, graph, score_by_view[capture["view"]])

        write_contact_sheet(output_dir / "multiview_contact_sheet.png", captures, graph, score_by_view)
        render_sheet(graph, output_dir / "spatial_graph_only.png")
        render_perspective(graph, output_dir / "spatial_graph_3d.png")
        selected_capture = next(capture for capture in captures if capture["view"] == selected_view)
        selected_overlay = draw_graph_overlay(selected_capture, graph, score_by_view[selected_view])
        Image.fromarray(selected_overlay).save(output_dir / "selected_view_graph.png")

        edge_states = {edge["id"]: edge["state"] for edge in graph["edges"]}
        summary = {
            "case": case["id"],
            "phase": case["phase"],
            "description": case["description"],
            "verdict": graph["verdict"],
            "confidence": graph["confidence"],
            "summary": graph["summary"],
            "edge_states": edge_states,
            "missing_evidence": graph["missing_evidence"],
            "selected_view": selected_view,
            "orchestrator_next_tool": next_tool,
            "visibility_proxy_best_view": oracle_best,
            "world_frozen_across_views": frozen_delta <= 1e-6,
            "executed_actions": executed,
            "directory": case["id"],
        }
        (output_dir / "case_report.json").write_text(
            json.dumps({**summary, "scores": scores}, indent=2), encoding="utf-8"
        )
        return {"summary": summary, "selected_overlay": selected_overlay}
    finally:
        env.close()


def grasp_actions(env: InteractiveRoboTwinEnv):
    with robotwin_cwd():
        arm_tag, actions = env.task.grasp_actor(
            grasp_target_actor(env),
            arm_tag=env.task.arm_tag,
            pre_grasp_dis=0.09,
        )
    if not actions or str(actions[-1].action) != "gripper":
        raise RuntimeError("grasp_single_pen expert did not produce a closing action")
    return arm_tag, actions


def execute_case(env: InteractiveRoboTwinEnv, phase: str, action_bundle) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    arm_tag, actions = action_bundle
    if phase == "aligned_open_at_grasp":
        selected_actions = actions[:-1]
    elif phase in {"closed_before_lift", "lifted_success"}:
        selected_actions = actions
    elif phase == "closed_at_pregrasp_offset":
        selected_actions = [actions[0], actions[-1]]
    else:
        raise ValueError(f"unknown phase {phase}")

    executed = []
    for source_index, action in enumerate(selected_actions):
        with robotwin_cwd():
            success = bool(env.task.move((arm_tag, [action]), save_freq=None))
            env.task._update_render()
        executed.append(
            {
                "index": source_index,
                "action_type": str(action.action),
                "target_gripper_position": getattr(action, "target_gripper_pos", None),
                "planner_success": success,
            }
        )

    motion_reference = None
    if phase == "lifted_success":
        motion_reference = grasp_motion_reference(env)
        with robotwin_cwd():
            lift = env.task.move_by_displacement(arm_tag=env.task.arm_tag, z=0.12)
            success = bool(env.task.move(lift, save_freq=None))
            env.task._update_render()
        executed.append(
            {"index": len(executed), "action_type": "lift_z_0.12", "planner_success": success}
        )
    return executed, motion_reference


def orchestrator_next_tool(graph: dict[str, Any], diagnostic_view: str) -> dict[str, Any]:
    if graph["verdict"] in {"true", "false"} and graph["confidence"] >= 0.9:
        return {
            "tool": "stop",
            "reason": "The graph verdict already meets the confidence threshold.",
            "camera_view": None,
        }
    missing = set(graph.get("missing_evidence", []))
    if missing & {"lifted_from_support", "moves_with_gripper"}:
        return {
            "tool": "spatial.controlled_lift_probe",
            "reason": "A static camera view cannot establish rigid co-motion; apply a small lift and observe height and relative motion.",
            "camera_view": diagnostic_view,
        }
    return {
        "tool": "camera.select_view",
        "reason": "Acquire a view that reduces the remaining relation uncertainty.",
        "camera_view": diagnostic_view,
    }


def write_comparison(path: Path, results: list[dict[str, Any]]) -> None:
    panels = []
    for result in results:
        summary = result["summary"]
        panel = Image.fromarray(result["selected_overlay"]).convert("RGB")
        draw = ImageDraw.Draw(panel)
        banner_height = 68
        draw.rectangle([0, panel.height - banner_height, panel.width, panel.height], fill=(12, 16, 20))
        draw.text((10, panel.height - 61), summary["phase"].replace("_", " "), fill=(255, 255, 255))
        draw.text(
            (10, panel.height - 40),
            f"verdict={summary['verdict']} ({summary['confidence']:.2f}) | diagnostic view={summary['selected_view']}",
            fill=(220, 220, 220),
        )
        draw.text(
            (10, panel.height - 20),
            " | ".join(f"{key}={value}" for key, value in summary["edge_states"].items()),
            fill=(180, 205, 225),
        )
        panels.append(panel)

    width, height = panels[0].size
    sheet = Image.new("RGB", (width * 2, height * 2), (24, 24, 24))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * width, (index // 2) * height))
    sheet.save(path)


def write_summary(path: Path, seed: int, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 1 grasp_single_pen case analysis",
        "",
        f"Task: `grasp_single_pen`, seed: `{seed}`. No GPT/VLM call is used.",
        "Each physical state is frozen while the six discrete candidate views are rendered.",
        "",
        "| Case | Verdict | Between | Contact A | Contact B | Supported | Lifted | Moves with gripper | Next tool | Diagnostic view |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        summary = result["summary"]
        edges = summary["edge_states"]
        lines.append(
            f"| `{summary['phase']}` | `{summary['verdict']}` | {edges['object_between_fingers']} | "
            f"{edges['finger_a_contact']} | {edges['finger_b_contact']} | {edges['supported_by_table']} | "
            f"{edges['lifted_from_support']} | {edges['moves_with_gripper']} | "
            f"`{summary['orchestrator_next_tool']['tool']}` | `{summary['selected_view']}` |"
        )
    lines.extend(
        [
            "",
            "`pen_cases_comparison.png` compares the rule-selected view for all four states.",
            "Each case directory contains its six-view contact sheet, graph-only rendering, JSON graph, depth, and oracle actor IDs.",
            "Physics contacts, actor IDs, object poses, and temporal motion are oracle/training-only evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
