from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_spatial_benchmark import InteractiveRoboTwinEnv
from active_spatial_benchmark.env import robotwin_cwd
from active_spatial_benchmark.grasp_event_manifest import (
    load_grasp_event_manifest,
    select_grasp_event,
)
from active_spatial_benchmark.spatial_graph import (
    VerifyGraspGeometry,
    build_verify_grasp_oracle_graph,
    graph_node_map,
    score_candidate_view,
)


CANDIDATE_VIEWS = (
    "current",
    "topdown",
    "side",
    "front_side_45",
    "side_top_45",
    "oblique_45",
)
EDGE_COLORS = {
    "true": (50, 205, 90),
    "false": (235, 75, 75),
    "unknown": (255, 190, 55),
}
NODE_COLORS = {
    "robot_kinematics": (60, 210, 255),
    "oracle_physics_contact": (60, 210, 255),
    "oracle_object_pose": (255, 90, 205),
    "oracle_obb": (255, 140, 60),
    "oracle_support_plane": (100, 230, 120),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed-state Phase 1 oracle spatial-graph and view-selection demo."
    )
    parser.add_argument("--task", default="grasp_single_bottle")
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="runs/phase1_spatial_graph/grasp_single_bottle_seed0_close",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    env = InteractiveRoboTwinEnv(
        task_name=args.task,
        config_name=args.config,
        active_arm="right",
        max_steps=1000,
        output_dir=output_dir / "env",
        save_images=False,
    )
    try:
        env.reset(seed=args.seed)
        object_start_center = object_obb(env)[0]
        executed_actions = execute_through_gripper_close(env)
        frozen_before = world_fingerprint(env)
        graph = build_graph(env, object_start_center)
        graph_path = output_dir / "oracle_spatial_graph.json"
        graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

        initial_pose = clone_camera_pose(env)
        initial_frame = capture_view(env, "current", initial_pose, graph)
        current_uncertainty = inferred_current_edge_uncertainty(graph, initial_frame)
        captures = [initial_frame]
        for view in CANDIDATE_VIEWS[1:]:
            captures.append(capture_view(env, view, initial_pose, graph))
        restore_camera_pose(env, initial_pose)

        frozen_after = world_fingerprint(env)
        frozen_delta = fingerprint_delta(frozen_before, frozen_after)
        if frozen_delta > 1e-6:
            raise RuntimeError(
                f"world state changed while rendering candidate views: delta={frozen_delta}"
            )

        scores = score_views(captures, graph, current_uncertainty)
        score_by_view = {row["view"]: row for row in scores}
        selected_view = max(scores, key=lambda row: row["predicted"]["utility"])["view"]
        realized_best_view = max(
            scores, key=lambda row: row["visibility_aware_oracle_proxy"]
        )["view"]

        for capture in captures:
            capture["selected_by_rule"] = capture["view"] == selected_view
            capture["oracle_best"] = capture["view"] == realized_best_view
            write_view_artifacts(
                views_dir, capture, graph, score_by_view[capture["view"]]
            )

        report = {
            "schema_version": "phase1.spatial_graph_demo.v1",
            "access": "oracle/training_only",
            "task": args.task,
            "config": args.config,
            "seed": args.seed,
            "fixed_phase": "closed_before_lift",
            "executed_expert_actions": executed_actions,
            "world_state_version": graph["world_state_version"],
            "world_frozen_across_views": frozen_delta <= 1e-6,
            "world_fingerprint_max_delta": frozen_delta,
            "graph": graph_path.name,
            "current_edge_uncertainty": current_uncertainty,
            "selected_view": selected_view,
            "visibility_aware_oracle_proxy_best_view": realized_best_view,
            "selector_matches_visibility_proxy_best": selected_view
            == realized_best_view,
            "view_budget": 1,
            "scores": scores,
            "artifacts": {
                "contact_sheet": "phase1_contact_sheet.png",
                "summary": "README.md",
                "views": {
                    capture["view"]: f"views/{capture['view']}_graph.png"
                    for capture in captures
                },
            },
        }
        (output_dir / "view_selection.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        write_contact_sheet(
            output_dir / "phase1_contact_sheet.png", captures, graph, score_by_view
        )
        write_readme(output_dir / "README.md", graph, report)

        print(f"output_dir: {output_dir}")
        print("fixed_phase: closed_before_lift")
        print(f"graph_verdict: {graph['verdict']} ({graph['confidence']:.2f})")
        print(f"selected_view: {selected_view}")
        print(f"visibility_aware_oracle_proxy_best_view: {realized_best_view}")
        print(f"world_frozen_across_views: {report['world_frozen_across_views']}")
        for row in sorted(
            scores, key=lambda item: item["predicted"]["utility"], reverse=True
        ):
            print(
                f"  {row['view']:<18} predicted={row['predicted']['utility']:.3f} "
                f"visibility_proxy={row['visibility_aware_oracle_proxy']:.3f} "
                f"visible={row['realized_keypoint_visibility']:.2f}"
            )
    finally:
        env.close()


def execute_through_gripper_close(env: InteractiveRoboTwinEnv) -> list[dict[str, Any]]:
    actor = grasp_target_actor(env)
    with robotwin_cwd():
        arm_tag, actions = env.task.grasp_actor(
            actor,
            arm_tag=env.task.arm_tag,
            pre_grasp_dis=grasp_pre_distance(env),
        )
    rows = []
    for index, action in enumerate(actions):
        action_type = str(getattr(action, "action", "unknown"))
        target_gripper_pos = getattr(action, "target_gripper_pos", None)
        with robotwin_cwd():
            ok = bool(env.task.move((arm_tag, [action]), save_freq=None))
            env.task._update_render()
        phase = "pre_grasp" if index == 0 else "final_grasp"
        if action_type == "gripper":
            phase = "close" if float(target_gripper_pos) <= 0.2 else "open"
        rows.append(
            {
                "index": index,
                "phase": phase,
                "action_type": action_type,
                "target_gripper_position": target_gripper_pos,
                "planner_success": ok,
            }
        )
        if phase == "close":
            return rows
    raise RuntimeError("expert grasp sequence did not contain a close action")


def build_graph(
    env: InteractiveRoboTwinEnv,
    object_start_center: np.ndarray,
    *,
    motion_reference: dict[str, np.ndarray] | None = None,
    world_state_version: int = 1,
) -> dict[str, Any]:
    object_center, object_rotation, object_half_extents = object_obb(env)
    finger_a = link_position(env, "fr_link7")
    finger_b = link_position(env, "fr_link8")
    actor = grasp_target_actor(env)
    actor_name = actor.get_name()
    contact_pairs = scene_contact_pairs(env)
    object_axis = object_rotation[:, int(np.argmax(object_half_extents))]
    support_z = float(
        object_start_center[2]
        - projected_obb_radius(object_rotation, object_half_extents, [0, 0, 1])
    )
    object_displacement = None
    gripper_displacement = None
    if motion_reference is not None:
        object_displacement = object_center - np.asarray(
            motion_reference["object_center"], dtype=np.float64
        )
        gripper_displacement = (finger_a + finger_b) / 2.0 - np.asarray(
            motion_reference["jaw_center"], dtype=np.float64
        )
    geometry = VerifyGraspGeometry(
        finger_a=finger_a,
        finger_b=finger_b,
        object_center=object_center,
        object_rotation=object_rotation,
        object_half_extents=object_half_extents,
        object_axis=object_axis,
        support_z=support_z,
        object_start_center_z=float(object_start_center[2]),
        finger_a_contact=pair_present(contact_pairs, actor_name, "fr_link7"),
        finger_b_contact=pair_present(contact_pairs, actor_name, "fr_link8"),
        support_contact=pair_present(contact_pairs, actor_name, "table"),
        gripper_closed=bool(env.task.is_right_gripper_close()),
        finger_a_contact_point=contact_centroid(env, actor_name, "fr_link7"),
        finger_b_contact_point=contact_centroid(env, actor_name, "fr_link8"),
        object_displacement=object_displacement,
        gripper_displacement=gripper_displacement,
    )
    graph = build_verify_grasp_oracle_graph(
        geometry, world_state_version=world_state_version
    )
    graph["oracle_metadata"] = {
        "task_actor": actor_name,
        "task_actor_id": int(actor.actor.per_scene_id),
        "finger_entity_ids": {
            "gripper.finger_a_inner": entity_id_for_link(env, "fr_link7"),
            "gripper.finger_b_inner": entity_id_for_link(env, "fr_link8"),
        },
        "contact_pairs": [
            list(pair) for pair in sorted(contact_pairs) if actor_name in pair
        ],
        "gripper_closed": geometry.gripper_closed,
        "note": "All oracle_metadata fields are forbidden as learned-model inputs.",
    }
    return graph


def grasp_motion_reference(env: InteractiveRoboTwinEnv) -> dict[str, np.ndarray]:
    object_center = object_obb(env)[0]
    finger_a = link_position(env, "fr_link7")
    finger_b = link_position(env, "fr_link8")
    return {
        "object_center": object_center,
        "jaw_center": (finger_a + finger_b) / 2.0,
    }


def capture_view(
    env: InteractiveRoboTwinEnv,
    view: str,
    current_pose,
    graph: dict[str, Any],
) -> dict[str, Any]:
    if view == "current":
        restore_camera_pose(env, current_pose)
    else:
        with robotwin_cwd():
            env.camera.view(env.active_arm, view)
            env.task._update_render()
    camera = env.camera.get_camera()
    with robotwin_cwd():
        camera.take_picture()
        rgba = camera.get_picture("Color")
        position = camera.get_picture("Position")
        segmentation = camera.get_picture("Segmentation")
    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    depth_mm = (-position[..., 2] * 1000.0).astype(np.float32)
    actor_segmentation = segmentation[..., 1].astype(np.uint32)
    pose_matrix = camera.entity.get_pose().to_transformation_matrix().astype(np.float64)
    intrinsic = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64)
    extrinsic = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
    projections = project_graph_nodes(
        graph, intrinsic, extrinsic, rgb.shape[1], rgb.shape[0]
    )
    expected_ids = expected_actor_ids(graph)
    visibility = {
        node_id: projected_node_visibility(
            actor_segmentation, projection, expected_ids.get(node_id)
        )
        for node_id, projection in projections.items()
    }
    object_id = int(graph["oracle_metadata"]["task_actor_id"])
    object_pixel_fraction = float(
        np.count_nonzero(actor_segmentation == object_id) / actor_segmentation.size
    )
    keypoint_visibility = (
        float(np.mean(list(visibility.values()))) if visibility else 0.0
    )
    target = np.asarray(
        graph_node_map(graph)["gripper.jaw_center"]["position_mean_world_m"],
        dtype=np.float64,
    )
    camera_position = pose_matrix[:3, 3]
    forward = normalized(pose_matrix[:3, 0])
    framing_score = projected_framing_score(projections, rgb.shape[1], rgb.shape[0])
    return {
        "view": view,
        "rgb": rgb,
        "depth_mm": depth_mm,
        "actor_segmentation": actor_segmentation,
        "camera_pose_world": pose_matrix,
        "intrinsic_cv": intrinsic,
        "extrinsic_cv": extrinsic,
        "camera_position_world": camera_position,
        "view_direction_world": forward,
        "target_distance_m": float(np.linalg.norm(target - camera_position)),
        "projections": projections,
        "node_visibility": visibility,
        "realized_keypoint_visibility": keypoint_visibility,
        "object_pixel_fraction": object_pixel_fraction,
        "framing_score": framing_score,
    }


def inferred_current_edge_uncertainty(
    graph: dict[str, Any], capture: dict[str, Any]
) -> dict[str, float]:
    closing_axis = np.asarray(
        graph["query_axes"]["closing_axis_world"], dtype=np.float64
    )
    direction = normalized(capture["view_direction_world"])
    closing_observability = 1.0 - abs(
        float(np.dot(direction, normalized(closing_axis)))
    )
    support_observability = 1.0 - abs(float(direction[2]))
    visibility = capture["node_visibility"]

    def relation_uncertainty(node_ids: tuple[str, ...], observability: float) -> float:
        realized_visibility = min(
            float(visibility.get(node_id, 0.0)) for node_id in node_ids
        )
        evidence = observability * (0.35 + 0.65 * realized_visibility)
        return round(float(np.clip(1.0 - evidence, 0.05, 1.0)), 6)

    return {
        "object_between_fingers": relation_uncertainty(
            ("object.center", "gripper.finger_a_inner", "gripper.finger_b_inner"),
            closing_observability,
        ),
        "finger_a_contact": relation_uncertainty(
            ("object.surface_a", "gripper.finger_a_inner"),
            closing_observability,
        ),
        "finger_b_contact": relation_uncertainty(
            ("object.surface_b", "gripper.finger_b_inner"),
            closing_observability,
        ),
        "lifted_from_support": relation_uncertainty(
            ("object.center", "support.plane_anchor"),
            support_observability,
        ),
    }


def score_views(
    captures: list[dict[str, Any]],
    graph: dict[str, Any],
    edge_uncertainty: dict[str, float],
) -> list[dict[str, Any]]:
    current = captures[0]
    current_direction = current["view_direction_world"]
    current_position = current["camera_position_world"]
    closing_axis = np.asarray(
        graph["query_axes"]["closing_axis_world"], dtype=np.float64
    )
    relation_weights = view_relation_weights(graph)
    rows = []
    for capture in captures:
        position_delta = float(
            np.linalg.norm(capture["camera_position_world"] - current_position)
        )
        direction_angle = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(
                            normalized(capture["view_direction_world"]),
                            normalized(current_direction),
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        move_cost = float(
            np.clip(
                0.7 * position_delta / 0.8 + 0.3 * direction_angle / 120.0, 0.0, 1.0
            )
        )
        predicted = score_candidate_view(
            view_direction_world=capture["view_direction_world"],
            current_view_direction_world=current_direction,
            closing_axis_world=closing_axis,
            edge_uncertainty=edge_uncertainty,
            framing_score=capture["framing_score"],
            move_cost=move_cost,
            relation_weights=relation_weights,
        )
        realized = (
            0.65 * predicted["relation_score"] * capture["realized_keypoint_visibility"]
            + 0.20 * predicted["baseline_score"]
            + 0.15 * min(1.0, capture["object_pixel_fraction"] / 0.015)
            - 0.15 * predicted["move_cost"]
        )
        rows.append(
            {
                "view": capture["view"],
                "predicted": predicted,
                "visibility_aware_oracle_proxy": round(float(realized), 6),
                "realized_keypoint_visibility": round(
                    float(capture["realized_keypoint_visibility"]), 6
                ),
                "object_pixel_fraction": round(
                    float(capture["object_pixel_fraction"]), 6
                ),
                "camera_position_world_m": rounded(capture["camera_position_world"]),
                "view_direction_world": rounded(capture["view_direction_world"]),
            }
        )
    return rows


def view_relation_weights(graph: dict[str, Any]) -> dict[str, float]:
    missing = set(graph.get("missing_evidence", []))
    if missing & {"lifted_from_support", "moves_with_gripper"}:
        return {
            "object_between_fingers": 0.15,
            "finger_a_contact": 0.20,
            "finger_b_contact": 0.20,
            "lifted_from_support": 3.0,
        }
    return {
        "object_between_fingers": 1.0,
        "finger_a_contact": 1.2,
        "finger_b_contact": 1.2,
        "lifted_from_support": 0.8,
    }


def write_view_artifacts(
    views_dir: Path,
    capture: dict[str, Any],
    graph: dict[str, Any],
    score: dict[str, Any],
) -> None:
    view = capture["view"]
    imageio.imwrite(views_dir / f"{view}_rgb.png", capture["rgb"])
    np.save(views_dir / f"{view}_depth_mm.npy", capture["depth_mm"])
    np.save(views_dir / f"{view}_actor_ids.npy", capture["actor_segmentation"])
    imageio.imwrite(
        views_dir / f"{view}_depth.png", colorize_depth(capture["depth_mm"])
    )
    imageio.imwrite(
        views_dir / f"{view}_actor_ids.png",
        colorize_actor_ids(capture["actor_segmentation"]),
    )
    overlay = draw_graph_overlay(capture, graph, score)
    imageio.imwrite(views_dir / f"{view}_graph.png", overlay)
    metadata = {
        "view": view,
        "camera_pose_world": capture["camera_pose_world"].tolist(),
        "intrinsic_cv": capture["intrinsic_cv"].tolist(),
        "extrinsic_cv": capture["extrinsic_cv"].tolist(),
        "node_visibility": capture["node_visibility"],
        "object_pixel_fraction": capture["object_pixel_fraction"],
        "score": score,
        "files": {
            "rgb": f"{view}_rgb.png",
            "graph": f"{view}_graph.png",
            "depth_mm": f"{view}_depth_mm.npy",
            "depth_visualization": f"{view}_depth.png",
            "raw_actor_ids": f"{view}_actor_ids.npy",
            "actor_id_visualization": f"{view}_actor_ids.png",
        },
    }
    (views_dir / f"{view}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def draw_graph_overlay(
    capture: dict[str, Any], graph: dict[str, Any], score: dict[str, Any]
) -> np.ndarray:
    draw_scale = 2
    source_image = Image.fromarray(capture["rgb"]).convert("RGB")
    image = source_image.resize(
        (source_image.width * draw_scale, source_image.height * draw_scale),
        Image.Resampling.BILINEAR,
    )
    draw = ImageDraw.Draw(image)
    projections = capture["projections"]
    node_by_id = graph_node_map(graph)
    for edge in graph["edges"]:
        source = projections.get(edge["source"])
        target = projections.get(edge["target"])
        if source is None or target is None:
            continue
        color = EDGE_COLORS.get(str(edge["state"]), EDGE_COLORS["unknown"])
        source_pixel = tuple(draw_scale * value for value in source["pixel"])
        target_pixel = tuple(draw_scale * value for value in target["pixel"])
        draw.line([source_pixel, target_pixel], fill=color, width=3)
    for node_id, projection in projections.items():
        x, y = (draw_scale * value for value in projection["pixel"])
        node = node_by_id[node_id]
        color = NODE_COLORS.get(str(node["source"]), (255, 255, 255))
        visible = capture["node_visibility"].get(node_id, 0.0) > 0.0
        radius = 7 if visible else 6
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
        label = short_node_label(node_id)
        draw.text(
            (x + 9, y - 8), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0)
        )
    panel_height = 61
    draw.rectangle([0, 0, image.width, panel_height], fill=(0, 0, 0))
    flags = []
    if capture.get("selected_by_rule"):
        flags.append("SELECTED")
    if capture.get("oracle_best"):
        flags.append("VIS-PROXY BEST")
    flag_text = " | ".join(flags)
    draw.text((7, 5), f"{capture['view']}  {flag_text}", fill=(255, 255, 255))
    draw.text(
        (7, 23),
        f"pred={score['predicted']['utility']:.3f}  vis-proxy={score['visibility_aware_oracle_proxy']:.3f}  "
        f"kp-vis={score['realized_keypoint_visibility']:.2f}",
        fill=(230, 230, 230),
    )
    draw.text(
        (7, 39),
        "cyan=gripper, magenta/orange=object, green/red/yellow=edge state",
        fill=(190, 190, 190),
    )
    return np.asarray(image)


def write_contact_sheet(
    path: Path,
    captures: list[dict[str, Any]],
    graph: dict[str, Any],
    score_by_view: dict[str, dict[str, Any]],
) -> None:
    panels = [
        Image.fromarray(
            draw_graph_overlay(capture, graph, score_by_view[capture["view"]])
        )
        for capture in captures
    ]
    width, height = panels[0].size
    sheet = Image.new("RGB", (width * 3, height * 2), (25, 25, 25))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * width, (index // 3) * height))
    sheet.save(path)


def write_readme(path: Path, graph: dict[str, Any], report: dict[str, Any]) -> None:
    scores = sorted(
        report["scores"], key=lambda row: row["predicted"]["utility"], reverse=True
    )
    lines = [
        "# Phase 1 Oracle Spatial Graph Demo",
        "",
        "This run freezes `grasp_single_bottle` after the scripted gripper close and before lift.",
        "No GPT/VLM call or benchmark evaluation is used.",
        "",
        "## Result",
        "",
        f"- Graph verdict: `{graph['verdict']}` ({graph['confidence']:.2f})",
        f"- Summary: {graph['summary']}",
        f"- Rule-selected next view: `{report['selected_view']}`",
        f"- Visibility-aware oracle proxy best: `{report['visibility_aware_oracle_proxy_best_view']}`",
        f"- World frozen across candidate renders: `{report['world_frozen_across_views']}`",
        f"- Sparse graph size: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges",
        "- Dense point cloud retained: `false`",
        "",
        "## View ranking",
        "",
        "| View | Predicted utility | Visibility-aware oracle proxy | Keypoint visibility | Object pixels |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scores:
        lines.append(
            f"| `{row['view']}` | {row['predicted']['utility']:.3f} | "
            f"{row['visibility_aware_oracle_proxy']:.3f} | {row['realized_keypoint_visibility']:.2f} | "
            f"{100.0 * row['object_pixel_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `oracle_spatial_graph.json`: GPT-readable sparse nodes, relations, confidence and evidence state.",
            "- `view_selection.json`: rule score and visibility-aware oracle proxy for every candidate.",
            "- `phase1_contact_sheet.png`: all projected graph views in one image.",
            "- `views/*_depth_mm.npy`: metric depth buffers, in millimeters.",
            "- `views/*_actor_ids.npy`: raw actor IDs for oracle evaluation only.",
            "",
            "All actor IDs, object poses and physics contacts are marked `oracle/training_only` and must not be supplied to a learned inference tool.",
            "The visibility-aware oracle proxy uses raw segmentation to validate Phase 1 view geometry; it is not a learned relation-error reduction metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def object_obb(
    env: InteractiveRoboTwinEnv,
    *,
    actor=None,
    task_stage: str | None = None,
    active_arm: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actor = actor or grasp_target_actor(
        env,
        task_stage=task_stage,
        active_arm=active_arm,
    )
    pose = actor.get_pose().to_transformation_matrix().astype(np.float64)
    if actor is getattr(env.task, "cube", None):
        if hasattr(env.task, "cube_half_size_xyz"):
            half_extents = np.asarray(env.task.cube_half_size_xyz, dtype=np.float64)
        elif hasattr(env.task, "cube_half_size"):
            half_extents = np.full(3, float(env.task.cube_half_size), dtype=np.float64)
        else:
            raise RuntimeError(
                "primitive cube geometry requires cube_half_size or cube_half_size_xyz"
            )
        return pose[:3, 3].copy(), pose[:3, :3], half_extents
    scale = np.asarray(actor.config["scale"], dtype=np.float64)
    local_center = np.asarray(actor.config["center"], dtype=np.float64) * scale
    half_extents = np.asarray(actor.config["extents"], dtype=np.float64) * scale / 2.0
    center = pose[:3, :3] @ local_center + pose[:3, 3]
    return center, pose[:3, :3], half_extents


def projected_obb_radius(
    rotation: np.ndarray, half_extents: np.ndarray, direction: Any
) -> float:
    local_direction = rotation.T @ normalized(np.asarray(direction, dtype=np.float64))
    return float(np.dot(np.abs(local_direction), half_extents))


def link_position(env: InteractiveRoboTwinEnv, name: str) -> np.ndarray:
    link = env.task.robot.right_entity.find_link_by_name(name)
    if link is None:
        raise RuntimeError(f"missing robot link {name}")
    pose = getattr(link, "entity_pose", None)
    if pose is None:
        pose = link.get_pose()
    return np.asarray(pose.p, dtype=np.float64)


def entity_id_for_link(env: InteractiveRoboTwinEnv, name: str) -> int:
    link = env.task.robot.right_entity.find_link_by_name(name)
    return int(link.entity.per_scene_id)


def scene_contact_pairs(env: InteractiveRoboTwinEnv) -> set[tuple[str, str]]:
    pairs = set()
    for contact in env.task.scene.get_contacts():
        first = str(contact.bodies[0].entity.name)
        second = str(contact.bodies[1].entity.name)
        pairs.add(tuple(sorted((first, second))))
    return pairs


def contact_centroid(
    env: InteractiveRoboTwinEnv, first: str, second: str
) -> np.ndarray | None:
    points = []
    expected = {first, second}
    for contact in env.task.scene.get_contacts():
        names = {str(contact.bodies[0].entity.name), str(contact.bodies[1].entity.name)}
        if names != expected:
            continue
        points.extend(
            np.asarray(point.position, dtype=np.float64) for point in contact.points
        )
    if not points:
        return None
    return np.mean(np.stack(points), axis=0)


def pair_present(pairs: set[tuple[str, str]], first: str, second: str) -> bool:
    return tuple(sorted((first, second))) in pairs


def clone_camera_pose(env: InteractiveRoboTwinEnv):
    pose = env.camera.get_camera().entity.get_pose()
    return type(pose)(pose.p.copy(), pose.q.copy())


def restore_camera_pose(env: InteractiveRoboTwinEnv, pose) -> None:
    with robotwin_cwd():
        env.camera.get_camera().entity.set_pose(deepcopy(pose))
        env.task._update_render()


def project_graph_nodes(
    graph: dict[str, Any],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    result = {}
    for node in graph["nodes"]:
        point = np.asarray(node["position_mean_world_m"], dtype=np.float64)
        point_camera = extrinsic @ np.concatenate([point, [1.0]])
        if point_camera[2] <= 1e-6:
            continue
        pixel_h = intrinsic @ point_camera[:3]
        pixel = pixel_h[:2] / pixel_h[2]
        if not np.all(np.isfinite(pixel)):
            continue
        result[str(node["id"])] = {
            "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)],
            "in_frame": bool(0 <= pixel[0] < width and 0 <= pixel[1] < height),
            "camera_depth_m": round(float(point_camera[2]), 6),
        }
    return result


def expected_actor_ids(graph: dict[str, Any]) -> dict[str, int]:
    object_id = int(graph["oracle_metadata"]["task_actor_id"])
    finger_ids = graph["oracle_metadata"]["finger_entity_ids"]
    result = {
        str(node["id"]): object_id
        for node in graph["nodes"]
        if str(node["id"]).startswith("object.")
    }
    for node_id, entity_id in finger_ids.items():
        result[str(node_id)] = int(entity_id)
    return result


def projected_node_visibility(
    actor_ids: np.ndarray,
    projection: dict[str, Any],
    expected_actor_id: int | None,
    *,
    radius: int = 9,
) -> float:
    if not projection["in_frame"]:
        return 0.0
    if expected_actor_id is None:
        return 1.0
    x, y = (int(round(value)) for value in projection["pixel"])
    y0, y1 = max(0, y - radius), min(actor_ids.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(actor_ids.shape[1], x + radius + 1)
    patch = actor_ids[y0:y1, x0:x1]
    return 1.0 if np.any(patch == expected_actor_id) else 0.0


def projected_framing_score(
    projections: dict[str, dict[str, Any]], width: int, height: int
) -> float:
    required = (
        "gripper.finger_a_inner",
        "gripper.finger_b_inner",
        "object.center",
        "object.axis_start",
        "object.axis_end",
    )
    in_frame = [
        projections.get(node_id, {}).get("in_frame", False) for node_id in required
    ]
    if not in_frame:
        return 0.0
    score = float(np.mean(in_frame))
    center_projection = projections.get("object.center")
    if center_projection and center_projection["in_frame"]:
        x, y = center_projection["pixel"]
        normalized_radius = math.hypot(
            (x - width / 2) / width, (y - height / 2) / height
        )
        score *= float(np.clip(1.1 - normalized_radius, 0.2, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def world_fingerprint(env: InteractiveRoboTwinEnv) -> dict[str, list[float]]:
    actor_pose = grasp_target_actor(env).get_pose()
    gripper_pose = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    return {
        "target_object": np.concatenate([actor_pose.p, actor_pose.q])
        .astype(np.float64)
        .tolist(),
        "gripper": gripper_pose.tolist(),
        "finger_a": link_position(env, "fr_link7").tolist(),
        "finger_b": link_position(env, "fr_link8").tolist(),
    }


def grasp_target_actor(
    env: InteractiveRoboTwinEnv,
    *,
    task_stage: str | None = None,
    active_arm: str | None = None,
):
    task_name = getattr(env, "task_name", None)
    if isinstance(task_name, str) and task_name:
        manifest = load_grasp_event_manifest()
        if task_name in manifest.events_by_task:
            resolved_arm = active_arm
            if resolved_arm is None and getattr(env, "active_arm", None) in {
                "left",
                "right",
            }:
                resolved_arm = str(env.active_arm)
            return select_grasp_event(
                env,
                task_name,
                task_stage=task_stage,
                active_arm=resolved_arm,
                manifest=manifest,
            ).actor
    for attribute in ("bottle", "pen", "cube"):
        actor = getattr(env.task, attribute, None)
        if actor is not None:
            return actor
    raise RuntimeError(
        "cannot resolve a task-native grasp actor from the event manifest"
    )


def grasp_pre_distance(env: InteractiveRoboTwinEnv) -> float:
    return 0.09 if hasattr(env.task, "pen") else 0.10


def fingerprint_delta(
    before: dict[str, list[float]], after: dict[str, list[float]]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(before[key], dtype=np.float64)
                    - np.asarray(after[key], dtype=np.float64)
                )
            )
        )
        for key in before
    )


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    normalized_depth = np.zeros_like(depth_mm, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth_mm[valid], [2, 98])
        scaled = np.clip((depth_mm - low) / max(float(high - low), 1.0), 0.0, 1.0)
        normalized_depth[valid] = (255 * (1.0 - scaled[valid])).astype(np.uint8)
    return cv2.cvtColor(
        cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )


def colorize_actor_ids(actor_ids: np.ndarray) -> np.ndarray:
    ids = actor_ids.astype(np.uint64)
    return np.stack(
        [
            ((ids * 37 + 17) % 255).astype(np.uint8),
            ((ids * 67 + 29) % 255).astype(np.uint8),
            ((ids * 97 + 43) % 255).astype(np.uint8),
        ],
        axis=-1,
    )


def short_node_label(node_id: str) -> str:
    return {
        "gripper.jaw_center": "jaw",
        "gripper.finger_a_inner": "fA",
        "gripper.finger_b_inner": "fB",
        "object.center": "obj",
        "object.surface_a": "sA",
        "object.surface_b": "sB",
        "object.axis_start": "axis-",
        "object.axis_end": "axis+",
        "support.plane_anchor": "table",
    }.get(node_id, node_id)


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("cannot normalize near-zero vector")
    return vector / norm


def rounded(values: Any) -> list[float]:
    return [
        round(float(value), 6)
        for value in np.asarray(values, dtype=np.float64).reshape(-1)
    ]


if __name__ == "__main__":
    main()
