from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from active_spatial_benchmark import InteractiveRoboTwinEnv
from active_spatial_benchmark.active_belief import ObservationGraph, SpatialBelief
from active_spatial_benchmark.env import robotwin_cwd
from active_spatial_benchmark.learned_pregrasp import (
    PregraspObservationNet,
    derived_edges_from_belief_graph,
    predict_observation_graph_from_inputs,
    prepare_inference_inputs,
)
from active_spatial_benchmark.learned_view_selector import load_view_ranker, rerank_candidate_views
from active_spatial_benchmark.pregrasp_graph import (
    PREGRASP_EDGE_IDS,
    PREGRASP_RELATION_SPECS,
    VerifyPregraspGeometry,
    build_verify_pregrasp_oracle_graph,
    score_pregrasp_candidate_views,
    transform_local_point,
)

from run_phase1_pen_cases import (
    capture_view,
    grasp_actions,
    grasp_target_actor,
    object_obb,
    world_fingerprint,
)
from render_pregrasp_spatial_graph import render as render_pregrasp_spatial_graph
from run_phase1_spatial_graph_demo import (
    CANDIDATE_VIEWS,
    clone_camera_pose,
    fingerprint_delta,
    entity_id_for_link,
    link_position,
    normalized,
    projected_obb_radius,
    restore_camera_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1.5 active multi-view pre-grasp loop.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument(
        "--output-dir",
        default="runs/phase1_5_active_loop/grasp_single_pen_seed0_pregrasp",
    )
    parser.add_argument("--max-additional-views", type=int, default=2)
    parser.add_argument("--required-confidence", type=float, default=0.80)
    parser.add_argument(
        "--observation-adapter",
        choices=("oracle", "learned"),
        default="oracle",
        help="Use the Phase 1.5 Oracle sensor simulation or the Phase 3 RGB-D model.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Required for --observation-adapter learned.")
    parser.add_argument("--calibration", type=Path, help="Optional learned per-relation gate calibration JSON.")
    parser.add_argument("--view-ranker", type=Path, help="Optional learned candidate-view ranker checkpoint.")
    parser.add_argument(
        "--allow-learned-close",
        action="store_true",
        help="Permit a learned execute verdict to close the gripper. Disabled by default.",
    )
    parser.add_argument("--no-close", action="store_true", help="Only test the gate; do not execute close.")
    parser.add_argument(
        "--bias-mode",
        choices=("none", "along_object", "across_object", "vertical"),
        default="none",
    )
    parser.add_argument("--bias-m", type=float, default=0.0)
    args = parser.parse_args()
    if args.observation_adapter == "learned" and args.checkpoint is None:
        parser.error("--checkpoint is required with --observation-adapter learned")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "observed_views"
    views_dir.mkdir(parents=True, exist_ok=True)

    env = InteractiveRoboTwinEnv(
        task_name="grasp_single_pen",
        config_name=args.config,
        active_arm="right",
        max_steps=1000,
        output_dir=output_dir / "env",
        save_images=False,
    )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learned_model = None
        learned_checkpoint = None
        gate_calibration = None
        learned_view_ranker = None
        if args.calibration is not None:
            gate_calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            if "relation_thresholds" not in gate_calibration:
                raise ValueError(f"calibration lacks relation_thresholds: {args.calibration}")
        if args.observation_adapter == "learned":
            learned_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            learned_model = PregraspObservationNet(
                pretrained_path=None,
                freeze_stem=learned_checkpoint["model_config"]["freeze_stem"],
            )
            learned_model.load_state_dict(learned_checkpoint["model_state"])
            learned_model.to(device)
            if args.view_ranker is not None:
                learned_view_ranker = load_view_ranker(args.view_ranker, device=device)
        env.reset(seed=args.seed)
        start_center = object_obb(env)[0]
        executed = execute_to_pregrasp(env)
        bias_result = apply_pregrasp_bias(env, args.bias_mode, args.bias_m)
        initial_pose = clone_camera_pose(env)
        frozen_before = world_fingerprint(env)

        # The graph is always saved as hidden evaluation truth. In learned mode,
        # none of its object nodes, relations, masks, or actor IDs enter inference.
        oracle_graph = build_pregrasp_graph(env, start_center, world_state_version=1)
        oracle_graph["diagnostic_context"] = {
            "task": "grasp_single_pen",
            "phase": "pregrasp_before_close",
            "query": "verify_pregrasp",
        }
        (output_dir / "oracle_world_state_training_only.json").write_text(
            json.dumps(oracle_graph, indent=2), encoding="utf-8"
        )

        if args.observation_adapter == "oracle":
            initial_capture = capture_view(env, "current", initial_pose, oracle_graph)
            robot_kinematics = robot_kinematics_from_graph(oracle_graph, initial_capture, 1)
        else:
            initial_capture = capture_inference_view(env, "current", initial_pose)
            robot_kinematics = live_robot_kinematics(env, initial_capture, 1)
        candidates = camera_pose_catalog(env, initial_pose)
        belief = SpatialBelief(
            query="verify_pregrasp",
            world_state_version=1,
            required_edge_ids=PREGRASP_EDGE_IDS,
            query_axes=robot_kinematics["query_axes"],
            deterministic_nodes=robot_kinematics["nodes"],
        )
        history = []
        observed = []
        current_capture = initial_capture
        current_observation, graph_after_current, decoded = acquire_observation(
            adapter=args.observation_adapter,
            capture=current_capture,
            oracle_graph=oracle_graph,
            robot_kinematics=robot_kinematics,
            frame_id=0,
            belief=belief,
            learned_model=learned_model,
            device=device,
        )
        if decoded is not None:
            attach_learned_projections(current_capture, decoded, robot_kinematics)
        observed.append(current_capture)
        history.append(step_record("current", current_observation, graph_after_current, None, decoded=decoded))
        write_observed_view(views_dir, current_capture, current_observation)

        for step_index in range(1, args.max_additional_views + 1):
            gate = belief.pregrasp_gate(
                required_confidence=args.required_confidence,
                relation_thresholds=(gate_calibration or {}).get("relation_thresholds"),
            )
            if gate["verdict"] in {"execute", "adjust"}:
                break
            rows = score_pregrasp_candidate_views(
                edge_uncertainty=belief.edge_uncertainty(),
                query_axes=belief.query_axes,
                current_view_direction_world=current_capture["view_direction_world"],
                current_camera_position_world=current_capture["camera_position_world"],
                candidates=candidates,
                visited_views=[item["view"] for item in observed],
            )
            if learned_view_ranker is not None:
                rows = rerank_candidate_views(
                    learned_view_ranker,
                    rows,
                    edge_uncertainty=belief.edge_uncertainty(),
                    device=device,
                )
            if not rows:
                break
            selected = rows[0]
            selected_view = selected["view"]
            if args.observation_adapter == "oracle":
                selected_capture = capture_view(env, selected_view, initial_pose, oracle_graph)
            else:
                selected_capture = capture_inference_view(env, selected_view, initial_pose)
            selected_capture["selected_by_belief"] = True
            selected_capture["selection_step"] = step_index
            selected_capture["selector_rows"] = rows
            robot_kinematics = (
                robot_kinematics_from_graph(oracle_graph, selected_capture, 1)
                if args.observation_adapter == "oracle"
                else live_robot_kinematics(env, selected_capture, 1)
            )
            observation, graph_after, decoded = acquire_observation(
                adapter=args.observation_adapter,
                capture=selected_capture,
                oracle_graph=oracle_graph,
                robot_kinematics=robot_kinematics,
                frame_id=step_index,
                belief=belief,
                learned_model=learned_model,
                device=device,
            )
            if decoded is not None:
                attach_learned_projections(selected_capture, decoded, robot_kinematics)
            observed.append(selected_capture)
            history.append(step_record(selected_view, observation, graph_after, rows, decoded=decoded))
            current_capture = selected_capture
            write_observed_view(views_dir, selected_capture, observation)

        gate = belief.pregrasp_gate(
            required_confidence=args.required_confidence,
            relation_thresholds=(gate_calibration or {}).get("relation_thresholds"),
        )
        frozen_after_observation = world_fingerprint(env)
        observation_delta = fingerprint_delta(frozen_before, frozen_after_observation)
        close_result = None
        learned_close_allowed = args.observation_adapter != "learned" or args.allow_learned_close
        if gate["verdict"] == "execute" and not args.no_close and learned_close_allowed:
            close_result = execute_close(env)

        report = {
            "schema_version": "phase3.active_pregrasp_loop.v1",
            "access": (
                "oracle_sensor_adapter/training_only"
                if args.observation_adapter == "oracle"
                else "learned_rgbd/inference_visible"
            ),
            "task": "grasp_single_pen",
            "query": "verify_pregrasp",
            "seed": args.seed,
            "fixed_state": "final_pregrasp_before_close",
            "commanded_bias": bias_result,
            "observation_adapter": args.observation_adapter,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_epoch": learned_checkpoint.get("epoch") if learned_checkpoint else None,
            "calibration": str(args.calibration.resolve()) if args.calibration else None,
            "view_selector": "learned" if learned_view_ranker is not None else "rule",
            "view_ranker": str(args.view_ranker.resolve()) if args.view_ranker else None,
            "max_additional_views": args.max_additional_views,
            "observed_view_sequence": [capture["view"] for capture in observed],
            "history": history,
            "final_gate": gate,
            "close_executed": close_result is not None,
            "learned_close_safety_lock": args.observation_adapter == "learned" and not args.allow_learned_close,
            "close_result": close_result,
            "world_frozen_during_observation": observation_delta <= 1e-6,
            "world_fingerprint_delta_during_observation": observation_delta,
            "anti_cheating": {
                "candidate_images_rendered_before_selection": False,
                "belief_initialized_from_full_oracle_graph": False,
                "oracle_used_only_to_simulate_sensor_labels": args.observation_adapter == "oracle",
                "oracle_object_truth_used_by_learned_adapter": False,
                "learned_inputs": (
                    ["rgb", "metric_depth", "camera_calibration", "robot_kinematics"]
                    if args.observation_adapter == "learned"
                    else None
                ),
            },
            "artifacts": {
                "belief_final": "belief_final.json",
                "belief_final_graph": "belief_final_graph.png",
                "transition": "belief_transition.png",
                "oracle_world_state": "oracle_world_state_training_only.json",
            },
        }
        belief_final = belief.to_graph()
        (output_dir / "belief_final.json").write_text(json.dumps(belief_final, indent=2), encoding="utf-8")
        (output_dir / "active_loop_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        visual_graph = deepcopy(belief_final)
        visual_graph["verdict"] = gate["verdict"]
        visual_graph["confidence"] = gate["confidence"]
        failed_relations = set(gate.get("failed_relations", []))
        for edge in visual_graph.get("edges", []):
            if edge["id"] in failed_relations:
                edge["decision_state"] = "false"
        visual_graph["diagnostic_context"] = oracle_graph["diagnostic_context"]
        render_pregrasp_spatial_graph(visual_graph, output_dir / "belief_final_graph.png")
        write_transition(output_dir / "belief_transition.png", observed, history, gate)
        write_readme(output_dir / "README.md", report)

        print(f"output_dir: {output_dir}")
        print(f"observed_view_sequence: {report['observed_view_sequence']}")
        for item in history:
            print(
                f"  step={item['step']} view={item['view']:<14} "
                + " ".join(
                    f"{edge_id}={values['probability']:.3f}/{values['uncertainty']:.3f}"
                    for edge_id, values in item["relations"].items()
                )
            )
        print(f"final_gate: {gate['verdict']} ({gate['confidence']:.3f}) -> {gate['action']}")
        print(f"world_frozen_during_observation: {report['world_frozen_during_observation']}")
    finally:
        env.close()


def execute_to_pregrasp(env: InteractiveRoboTwinEnv) -> list[dict[str, Any]]:
    arm_tag, actions = grasp_actions(env)
    rows = []
    for index, action in enumerate(actions[:-1]):
        with robotwin_cwd():
            ok = bool(env.task.move((arm_tag, [action]), save_freq=None))
            env.task._update_render()
        rows.append({"index": index, "action": str(action.action), "planner_success": ok})
    return rows


class CloseActionUnavailable(RuntimeError):
    """Raised when the task planner cannot produce a gripper-close action."""


def execute_close(env: InteractiveRoboTwinEnv) -> dict[str, Any]:
    actor = grasp_target_actor(env)
    with robotwin_cwd():
        arm_tag, actions = env.task.grasp_actor(actor, arm_tag=env.task.arm_tag, pre_grasp_dis=0.09)
    if not actions:
        raise CloseActionUnavailable("grasp planner returned no close action")
    close_action = actions[-1]
    with robotwin_cwd():
        ok = bool(env.task.move((arm_tag, [close_action]), save_freq=None))
        env.task._update_render()
    return {"action": str(close_action.action), "planner_success": ok, "gripper_closed": bool(env.task.is_right_gripper_close())}


def apply_pregrasp_bias(env: InteractiveRoboTwinEnv, mode: str, distance_m: float) -> dict[str, Any]:
    if mode == "none" or abs(float(distance_m)) < 1e-9:
        return {
            "mode": "none",
            "distance_m": 0.0,
            "requested_translation_world_m": [0.0, 0.0, 0.0],
            "achieved_translation_world_m": [0.0, 0.0, 0.0],
            "translation_residual_m": 0.0,
            "requested_axis_fraction_achieved": 1.0,
            "planner_success": True,
        }
    _, rotation, half_extents = object_obb(env)
    object_axis = normalized(rotation[:, int(np.argmax(half_extents))])
    object_axis_xy = normalized(np.array([object_axis[0], object_axis[1], 0.0], dtype=np.float64))
    across_axis = normalized(np.cross(np.array([0.0, 0.0, 1.0]), object_axis_xy))
    if mode == "along_object":
        translation = object_axis_xy * float(distance_m)
    elif mode == "across_object":
        translation = across_axis * float(distance_m)
    elif mode == "vertical":
        translation = np.array([0.0, 0.0, float(distance_m)], dtype=np.float64)
    else:
        raise ValueError(f"unsupported bias mode {mode}")
    pose_before = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    target_pose = pose_before.copy()
    target_pose[:3] += translation
    with robotwin_cwd():
        planner_success = bool(env.task.move(env.task.move_to_pose("right", target_pose.tolist()), save_freq=None))
        env.task._update_render()
    pose_after = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    achieved = pose_after[:3] - pose_before[:3]
    requested_norm_sq = float(np.dot(translation, translation))
    axis_fraction = float(np.dot(achieved, translation) / requested_norm_sq) if requested_norm_sq > 1e-12 else 1.0
    return {
        "mode": mode,
        "distance_m": float(distance_m),
        "requested_translation_world_m": [round(float(value), 6) for value in translation],
        "achieved_translation_world_m": [round(float(value), 6) for value in achieved],
        "translation_residual_m": round(float(np.linalg.norm(translation - achieved)), 6),
        "requested_axis_fraction_achieved": round(axis_fraction, 6),
        "planner_success": planner_success,
    }


def build_pregrasp_graph(
    env: InteractiveRoboTwinEnv,
    object_start_center: np.ndarray,
    *,
    world_state_version: int,
) -> dict[str, Any]:
    object_center, object_rotation, object_half_extents = object_obb(env)
    object_axis = object_rotation[:, int(np.argmax(object_half_extents))]
    finger_poses = {}
    for name in ("fr_link7", "fr_link8"):
        link = env.task.robot.right_entity.find_link_by_name(name)
        pose = getattr(link, "entity_pose", None)
        if pose is None:
            pose = link.get_pose()
        finger_poses[name] = pose.to_transformation_matrix().astype(np.float64)
    support_z = float(
        object_start_center[2]
        - projected_obb_radius(object_rotation, object_half_extents, [0.0, 0.0, 1.0])
    )
    graph = build_verify_pregrasp_oracle_graph(
        VerifyPregraspGeometry(
            finger_a_base=finger_poses["fr_link7"][:3, 3],
            finger_b_base=finger_poses["fr_link8"][:3, 3],
            finger_a_tip=transform_local_point(finger_poses["fr_link7"], [0.06, 0.0, 0.0]),
            finger_b_tip=transform_local_point(finger_poses["fr_link8"], [0.06, 0.0, 0.0]),
            object_center=object_center,
            object_rotation=object_rotation,
            object_half_extents=object_half_extents,
            object_axis=object_axis,
            support_z=support_z,
        ),
        world_state_version=world_state_version,
    )
    actor = grasp_target_actor(env)
    finger_a_id = entity_id_for_link(env, "fr_link7")
    finger_b_id = entity_id_for_link(env, "fr_link8")
    graph["oracle_metadata"] = {
        "task_actor": actor.get_name(),
        "task_actor_id": int(actor.actor.per_scene_id),
        "finger_entity_ids": {
            "gripper.finger_a_base": finger_a_id,
            "gripper.finger_b_base": finger_b_id,
            "gripper.finger_a_inner_tip": finger_a_id,
            "gripper.finger_b_inner_tip": finger_b_id,
        },
        "note": "All oracle_metadata fields are forbidden as learned-model inputs.",
    }
    return graph


def camera_pose_catalog(env: InteractiveRoboTwinEnv, initial_pose) -> list[dict[str, Any]]:
    rows = []
    for view in CANDIDATE_VIEWS:
        restore_camera_pose(env, initial_pose)
        if view != "current":
            with robotwin_cwd():
                env.camera.view(env.active_arm, view)
        pose = env.camera.get_camera().entity.get_pose().to_transformation_matrix().astype(np.float64)
        rows.append(
            {
                "view": view,
                "camera_position_world": pose[:3, 3].tolist(),
                "view_direction_world": normalized(pose[:3, 0]).tolist(),
                "framing_score": 1.0,
            }
        )
    restore_camera_pose(env, initial_pose)
    return rows


def deterministic_robot_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for node in graph["nodes"]:
        if node["id"].startswith("gripper."):
            node = dict(node)
            node["source"] = "robot_kinematics"
            node["position_covariance_m2"] = (np.eye(3) * 1e-8).tolist()
            node["visibility"] = 1.0
            nodes.append(node)
    return nodes


def robot_kinematics_from_graph(
    graph: dict[str, Any],
    capture: dict[str, Any],
    world_state_version: int,
) -> dict[str, Any]:
    return {
        "schema_version": "phase3.robot_kinematics.v1",
        "access": "inference_visible",
        "frame_id": int(capture.get("frame_id", 0)),
        "view": capture["view"],
        "world_state_version": int(world_state_version),
        "nodes": deterministic_robot_nodes(graph),
        "query_axes": {
            "closing_axis_world": graph["query_axes"]["closing_axis_world"],
            "support_normal_world": graph["query_axes"]["support_normal_world"],
        },
    }


def live_robot_kinematics(
    env: InteractiveRoboTwinEnv,
    capture: dict[str, Any],
    world_state_version: int,
) -> dict[str, Any]:
    """Build inference-visible gripper points without object simulator truth."""

    poses = {}
    for name in ("fr_link7", "fr_link8"):
        link = env.task.robot.right_entity.find_link_by_name(name)
        pose = getattr(link, "entity_pose", None)
        if pose is None:
            pose = link.get_pose()
        poses[name] = pose.to_transformation_matrix().astype(np.float64)
    base_a = poses["fr_link7"][:3, 3]
    base_b = poses["fr_link8"][:3, 3]
    tip_a = transform_local_point(poses["fr_link7"], [0.06, 0.0, 0.0])
    tip_b = transform_local_point(poses["fr_link8"], [0.06, 0.0, 0.0])
    jaw_center = (base_a + base_b) / 2.0
    grasp_center = (tip_a + tip_b) / 2.0
    closing_axis = normalized(tip_b - tip_a)

    def node(node_id: str, semantic_type: str, position: np.ndarray) -> dict[str, Any]:
        return {
            "id": node_id,
            "semantic_type": semantic_type,
            "position_mean_world_m": [round(float(value), 6) for value in position],
            "position_covariance_m2": (np.eye(3) * 1e-8).tolist(),
            "source": "robot_kinematics",
            "access": "inference_visible",
            "visibility": 1.0,
            "valid_for_world_state": int(world_state_version),
        }

    return {
        "schema_version": "phase3.robot_kinematics.v1",
        "access": "inference_visible",
        "frame_id": int(capture.get("frame_id", 0)),
        "view": capture["view"],
        "world_state_version": int(world_state_version),
        "nodes": [
            node("gripper.jaw_center", "jaw_base_center", jaw_center),
            node("gripper.finger_a_base", "finger_base", base_a),
            node("gripper.finger_b_base", "finger_base", base_b),
            node("gripper.finger_a_inner_tip", "finger_inner_tip", tip_a),
            node("gripper.finger_b_inner_tip", "finger_inner_tip", tip_b),
            node("gripper.grasp_center", "grasp_center", grasp_center),
        ],
        "query_axes": {
            "closing_axis_world": [round(float(value), 6) for value in closing_axis],
            "support_normal_world": [0.0, 0.0, 1.0],
        },
    }


def capture_inference_view(
    env: InteractiveRoboTwinEnv,
    view: str,
    current_pose,
) -> dict[str, Any]:
    """Capture only fields permitted at learned-model inference time."""

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
    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    depth_mm = (-position[..., 2] * 1000.0).astype(np.float32)
    pose_matrix = camera.entity.get_pose().to_transformation_matrix().astype(np.float64)
    intrinsic = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64)
    extrinsic = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
    return {
        "view": view,
        "rgb": rgb,
        "depth_mm": depth_mm,
        "camera_pose_world": pose_matrix,
        "intrinsic_cv": intrinsic,
        "extrinsic_cv": extrinsic,
        "camera_position_world": pose_matrix[:3, 3],
        "view_direction_world": normalized(pose_matrix[:3, 0]),
        "framing_score": 1.0,
        "projections": {},
    }


def acquire_observation(
    *,
    adapter: str,
    capture: dict[str, Any],
    oracle_graph: dict[str, Any],
    robot_kinematics: dict[str, Any],
    frame_id: int,
    belief: SpatialBelief,
    learned_model: PregraspObservationNet | None,
    device: torch.device,
) -> tuple[ObservationGraph, dict[str, Any], dict[str, Any] | None]:
    if adapter == "oracle":
        observation = observation_from_capture(capture, oracle_graph, frame_id=frame_id)
        return observation, belief.update(observation), None
    if learned_model is None:
        raise ValueError("learned observation adapter requires a loaded checkpoint")
    camera_value = {
        "camera_pose_world": capture["camera_pose_world"],
        "intrinsic_cv": capture["intrinsic_cv"],
        "extrinsic_cv": capture["extrinsic_cv"],
    }
    rgbd, camera, kinematics = prepare_inference_inputs(
        capture["rgb"],
        capture["depth_mm"] / 1000.0,
        camera_value,
        robot_kinematics,
    )
    observation, decoded = predict_observation_graph_from_inputs(
        learned_model,
        rgbd=rgbd,
        camera=camera,
        kinematics=kinematics,
        robot_kinematics=robot_kinematics,
        frame_id=frame_id,
        view=capture["view"],
        world_state_version=belief.world_state_version,
        device=device,
    )
    node_observation = ObservationGraph(
        frame_id=observation.frame_id,
        view=observation.view,
        nodes=observation.nodes,
        edges=[],
        query_axes=observation.query_axes,
        source=observation.source,
    )
    belief_graph = belief.update(node_observation)
    edges, object_axis = derived_edges_from_belief_graph(
        belief_graph,
        view_direction_world=capture["view_direction_world"],
    )
    if object_axis is not None:
        belief.query_axes["object_axis_world"] = object_axis
    if edges:
        belief_graph = belief.replace_derived_edges(edges, frame_id=frame_id, view=capture["view"])
    return observation, belief_graph, decoded


def attach_learned_projections(
    capture: dict[str, Any],
    decoded: dict[str, Any],
    robot_kinematics: dict[str, Any],
) -> None:
    projections = {
        node_id: {"pixel": value["pixel_uv"], "in_frame": True}
        for node_id, value in decoded["keypoints"].items()
    }
    intrinsic = np.asarray(capture["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(capture["extrinsic_cv"], dtype=np.float64)
    width, height = capture["rgb"].shape[1], capture["rgb"].shape[0]
    for node in robot_kinematics["nodes"]:
        point = np.asarray(node["position_mean_world_m"], dtype=np.float64)
        camera_point = extrinsic @ np.concatenate([point, [1.0]])
        if camera_point[2] <= 1e-6:
            continue
        pixel = intrinsic @ camera_point[:3]
        u, v = float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])
        projections[node["id"]] = {
            "pixel": [u, v],
            "in_frame": 0 <= u < width and 0 <= v < height,
        }
    capture["projections"] = projections


def observation_from_capture(capture: dict[str, Any], graph: dict[str, Any], *, frame_id: int) -> ObservationGraph:
    visible = capture["node_visibility"]
    observation_nodes = []
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id.startswith("gripper."):
            continue
        visibility = float(visible.get(node_id, 0.0))
        if visibility <= 0.0:
            continue
        observed = dict(node)
        observed["source"] = "oracle_rgbd_sensor_simulation"
        observed["visibility"] = visibility
        # Simulate a single-view RGB-D keypoint uncertainty.  This is the only
        # part that will later be replaced by the learned keypoint head.
        observed["position_covariance_m2"] = (np.eye(3) * (0.004 / max(visibility, 0.25)) ** 2).tolist()
        observation_nodes.append(observed)

    relations = {edge["id"]: edge for edge in graph["edges"]}
    closing_axis = np.asarray(graph["query_axes"]["closing_axis_world"], dtype=np.float64)
    object_axis = np.asarray(graph["query_axes"]["object_axis_world"], dtype=np.float64)
    direction = normalized(np.asarray(capture["view_direction_world"], dtype=np.float64))
    observability = {
        "closing": 1.0 - abs(float(np.dot(direction, normalized(closing_axis)))),
        "object": 1.0 - abs(float(np.dot(direction, normalized(object_axis)))),
        "vertical": 1.0 - abs(float(direction[2])),
    }
    observation_edges = []
    for edge_id, spec in PREGRASP_RELATION_SPECS.items():
        relation = relations[edge_id]
        visibility_quality = max(
            min(float(visible.get(node_id, 0.0)) for node_id in alternative)
            for alternative in spec["visual_alternatives"]
        )
        relation_observability = min(observability[axis] for axis in spec["axes"])
        evidence = float(np.clip(relation_observability * visibility_quality, 0.0, 1.0))
        if evidence <= 0.05:
            continue
        # Probability is the calibrated relation estimate; evidence_weight is
        # the only term that controls how strongly this view updates belief.
        # Shrinking both would count view uncertainty twice and suppress clear
        # negative evidence.
        measured_probability = float(relation.get("probability", 0.5))
        observation_edges.append(
            {
                "id": edge_id,
                "source": relation["source"],
                "target": relation["target"],
                "relation": relation["relation"],
                "probability": measured_probability,
                "evidence_weight": evidence,
                "measurement": {
                    "view_observability": round(relation_observability, 6),
                    "visible_node_quality": round(visibility_quality, 6),
                },
            }
        )
    return ObservationGraph(
        frame_id=frame_id,
        view=capture["view"],
        nodes=observation_nodes,
        edges=observation_edges,
        query_axes=graph["query_axes"],
        source="oracle_sensor_adapter/training_only",
    )


def step_record(
    view: str,
    observation: ObservationGraph,
    graph: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    *,
    decoded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edge_map = {edge["id"]: edge for edge in graph["edges"]}
    relations = {
        edge_id: {
            "probability": float(edge_map[edge_id]["probability"]),
            "uncertainty": float(edge_map[edge_id]["uncertainty"]),
            "evidence_weight": float(edge_map[edge_id]["evidence_weight"]),
            "state": edge_map[edge_id]["state"],
        }
        for edge_id in PREGRASP_EDGE_IDS
    }
    return {
        "step": observation.frame_id,
        "view": view,
        "observed_node_count": len(observation.nodes),
        "observed_edge_ids": [item["id"] for item in observation.edges],
        "relations": relations,
        "evidence_weight": round(sum(item["evidence_weight"] for item in relations.values()), 6),
        "candidate_scores_before_capture": rows,
        "belief_graph": graph,
        "decoded_learned_keypoints": decoded,
    }


def write_observed_view(path: Path, capture: dict[str, Any], observation: ObservationGraph) -> None:
    imageio.imwrite(path / f"{capture['view']}_rgb.png", capture["rgb"])
    np.save(path / f"{capture['view']}_depth_mm.npy", capture["depth_mm"])
    (path / f"{capture['view']}_observation.json").write_text(
        json.dumps(observation.as_dict(), indent=2), encoding="utf-8"
    )


def write_transition(path: Path, captures: list[dict[str, Any]], history: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    panels = []
    for capture, item in zip(captures, history):
        image = Image.fromarray(capture["rgb"]).convert("RGB").resize((640, 480), Image.Resampling.BILINEAR)
        draw_belief_overlay(image, capture, item["belief_graph"])
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, image.width, 78], fill=(10, 15, 20))
        draw.text((12, 10), f"G{item['step']}  acquired view: {item['view']}", fill=(255, 255, 255))
        draw.text((12, 32), f"nodes={item['observed_node_count']}  evidence={item['evidence_weight']:.2f}", fill=(210, 220, 230))
        weakest = min(item["relations"].items(), key=lambda pair: pair[1]["probability"])
        draw.text(
            (12, 54),
            f"weakest={weakest[0]}  probability={weakest[1]['probability']:.3f}  uncertainty={weakest[1]['uncertainty']:.3f}",
            fill=(210, 220, 230),
        )
        panels.append(image)
    width = 640
    height = 480
    sheet = Image.new("RGB", (width * len(panels), height + 70), (28, 32, 38))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * width, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, height + 18), f"Final pre-grasp gate: {gate['verdict']} ({gate['confidence']:.3f}) -> {gate['action']}", fill=(255, 255, 255))
    sheet.save(path)


def draw_belief_overlay(image: Image.Image, capture: dict[str, Any], graph: dict[str, Any]) -> None:
    """Draw only the belief-visible nodes and its current relation."""

    draw = ImageDraw.Draw(image)
    scale_x = image.width / capture["rgb"].shape[1]
    scale_y = image.height / capture["rgb"].shape[0]
    projections = capture.get("projections", {})
    visible_ids = {node["id"] for node in graph.get("nodes", [])}
    node_colors = {
        "gripper.jaw_center": (40, 220, 255),
        "object.center": (255, 80, 190),
        "object.axis_start": (255, 160, 50),
        "object.axis_end": (255, 160, 50),
    }
    for node_id in visible_ids:
        projection = projections.get(node_id)
        if not projection:
            continue
        x = projection["pixel"][0] * scale_x
        y = projection["pixel"][1] * scale_y
        color = node_colors.get(node_id, (245, 245, 245))
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color, outline=(10, 15, 20), width=2)
        draw.text((x + 10, y - 8), node_id.split(".")[-1], fill=color, stroke_width=2, stroke_fill=(10, 15, 20))
    edge = next((edge for edge in graph.get("edges", []) if edge["id"] == "object_between_fingers"), None)
    if edge:
        source = projections.get(edge.get("source"))
        target = projections.get(edge.get("target"))
        if source and target:
            probability = float(edge.get("probability", 0.5))
            color = (35, 205, 100) if probability >= 0.8 else (235, 75, 75) if probability <= 0.2 else (255, 190, 55)
            start = (source["pixel"][0] * scale_x, source["pixel"][1] * scale_y)
            end = (target["pixel"][0] * scale_x, target["pixel"][1] * scale_y)
            draw.line([start, end], fill=color, width=4)


def write_readme(path: Path, report: dict[str, Any]) -> None:
    adapter = report.get("observation_adapter", "oracle")
    lines = [
        "# Active multi-view pre-grasp loop",
        "",
        "This run freezes `grasp_single_pen` at the final pre-grasp pose before closing.",
        "The loop acquires `current`, scores candidate camera poses, acquires only the selected next view, and fuses observations into one SpatialBelief.",
        "No GPT/VLM call is used.",
        "",
        f"- Observation adapter: `{adapter}`",
        f"- View sequence: `{report['observed_view_sequence']}`",
        f"- Final gate: `{report['final_gate']['verdict']}` ({report['final_gate']['confidence']:.3f})",
        f"- Action: `{report['final_gate']['action']}`",
        f"- World frozen during observations: `{report['world_frozen_during_observation']}`",
        f"- Candidate images rendered before selection: `{report['anti_cheating']['candidate_images_rendered_before_selection']}`",
        f"- Belief initialized from full Oracle graph: `{report['anti_cheating']['belief_initialized_from_full_oracle_graph']}`",
        "",
        "`belief_transition.png` shows the acquired-view sequence and relation confidence update.",
        "`belief_final_graph.png` shows the inference-visible fused sparse graph without an RGB background.",
        "`belief_final.json` contains only inference-visible fused nodes, edges, and evidence history.",
        "`oracle_world_state_training_only.json` is retained solely as hidden evaluation truth and must not be model input.",
    ]
    if adapter == "learned":
        lines.extend(
            [
                "",
                "The learned adapter consumes only RGB, metric depth, camera calibration, and robot kinematics.",
                "Segmentation, actor IDs, object pose, and Oracle relations are not passed to the model or belief.",
                f"Learned close safety lock: `{report.get('learned_close_safety_lock', True)}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
