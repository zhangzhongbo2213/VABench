from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import mplib
import numpy as np
import transforms3d as t3d


ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROBOTWIN_ROOT / "active_spatial_benchmark_xyz"
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT))

from active_spatial_benchmark import InteractiveRoboTwinEnv  # noqa: E402
from active_spatial_benchmark.env import robotwin_cwd  # noqa: E402
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC  # noqa: E402
from envs.utils.transforms import cal_quat_dis  # noqa: E402


DEFAULT_TASKS = (
    "grasp_single_bottle_upright",
    "grasp_single_bottle",
    "grasp_single_cube",
    "grasp_single_pen",
    "grasp_pen_leaning_cube",
    "place_cube_in_bowl",
    "place_single_bottle_upright",
    "place_single_cube",
    "place_cube_on_cube",
    "click_bell_right",
    "beat_block_hammer_right",
    "place_shoe",
    "handover_mic",
    "handover_horizontal_block",
    "handover_block",
    "handover_cube_to_target",
    "lift_pot",
)


@dataclass(frozen=True)
class GraspLiftSafetyProfile:
    method: str
    actor_attr: str
    object_layout: str
    required_active_arm: str
    pre_grasp_distance_mm: float
    commanded_lift_mm: float
    min_verified_lift_mm: float
    min_right_arm_plan_segments: int = 3
    min_left_arm_plan_segments: int = 0
    arm_plan_policy: str = "right_only"
    verification_mode: str = "grasp_lift"
    required_gripper_state: str = "closed"
    container_attr: str | None = None
    max_target_container_xy_offset_mm: float | None = None
    verification_method_override: str | None = None
    contact_actor_attr: str | None = None


TASK_SAFETY_PROFILES = {
    "grasp_single_bottle_upright_generalization": GraspLiftSafetyProfile(
        method="upright_heldout_container_right_grasp_lift_v1",
        actor_attr="bottle",
        object_layout="held-out upright bottle or shampoo container on table",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
    ),
    "grasp_single_bottle_upright": GraspLiftSafetyProfile(
        method="upright_bottle_right_grasp_lift_v1",
        actor_attr="bottle",
        object_layout="upright bottle on table",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
    ),
    "grasp_single_bottle_generalization": GraspLiftSafetyProfile(
        method="horizontal_heldout_container_right_grasp_lift_v1",
        actor_attr="bottle",
        object_layout="held-out horizontal bottle or shampoo container with randomized yaw",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        verification_method_override="full-planner-expert",
    ),
    "grasp_single_bottle": GraspLiftSafetyProfile(
        method="horizontal_bottle_right_grasp_lift_v1",
        actor_attr="bottle",
        object_layout="horizontal bottle on table with randomized yaw",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
    ),
    "grasp_single_cube": GraspLiftSafetyProfile(
        method="cube_right_grasp_lift_v1",
        actor_attr="cube",
        object_layout="cube on table with randomized yaw",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
    ),
    "grasp_single_cube_generalization": GraspLiftSafetyProfile(
        method="heldout_box_right_full_grasp_lift_v1",
        actor_attr="cube",
        object_layout="specified held-out box-like object among visible distractors",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        verification_method_override="full-planner-expert",
    ),
    "grasp_single_pen": GraspLiftSafetyProfile(
        method="horizontal_pen_right_grasp_lift_v1",
        actor_attr="pen",
        object_layout="horizontal pen on table with randomized yaw",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
    ),
    "grasp_pen_leaning_cube": GraspLiftSafetyProfile(
        method="leaning_pen_right_full_grasp_lift_v1",
        actor_attr="pen",
        object_layout="pen leaning diagonally against a static support cube",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        verification_method_override="full-planner-expert",
    ),
    "place_cube_in_bowl": GraspLiftSafetyProfile(
        method="cube_in_bowl_right_full_pick_place_v2",
        actor_attr="cube",
        object_layout="cube and bowl on table with randomized positions",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_in_container",
        required_gripper_state="open",
        container_attr="bowl",
        max_target_container_xy_offset_mm=55.0,
        verification_method_override="full-planner-expert",
    ),
    "place_cube_in_bowl_generalization": GraspLiftSafetyProfile(
        method="heldout_box_in_bowl_right_full_pick_place_v1",
        actor_attr="cube",
        object_layout="specified held-out box and larger target bowl with distractors",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_in_container",
        required_gripper_state="open",
        container_attr="bowl",
        max_target_container_xy_offset_mm=45.0,
        verification_method_override="full-planner-expert",
    ),
    "place_single_bottle_upright_generalization": GraspLiftSafetyProfile(
        method="upright_heldout_container_right_direct_pick_place_v1",
        actor_attr="bottle",
        object_layout="held-out upright container and green target pad on table",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="target_pad",
        max_target_container_xy_offset_mm=60.0,
    ),
    "place_single_bottle_upright": GraspLiftSafetyProfile(
        method="upright_bottle_right_direct_pick_place_v2",
        actor_attr="bottle",
        object_layout="upright bottle and green target pad on table",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="target_pad",
        max_target_container_xy_offset_mm=60.0,
    ),
    "place_single_cube": GraspLiftSafetyProfile(
        method="cube_right_place_target_v1",
        actor_attr="cube",
        object_layout="cube and green target pad on table",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="target_pad",
        max_target_container_xy_offset_mm=50.0,
        verification_method_override="full-planner-expert",
    ),
    "place_single_cube_generalization": GraspLiftSafetyProfile(
        method="heldout_box_right_place_target_v1",
        actor_attr="cube",
        object_layout="specified held-out box, valid pad, decoy pad, and obstacle",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="target_pad",
        max_target_container_xy_offset_mm=70.0,
        verification_method_override="full-planner-expert",
    ),
    "place_cube_on_cube": GraspLiftSafetyProfile(
        method="cube_on_cube_right_full_pick_place_v1",
        actor_attr="stack_cube",
        object_layout="red cube and green supporting cube on table",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="base_cube",
        max_target_container_xy_offset_mm=25.0,
        verification_method_override="full-planner-expert",
    ),
    "place_cube_on_cube_generalization": GraspLiftSafetyProfile(
        method="heldout_box_on_rotated_support_right_full_pick_place_v1",
        actor_attr="stack_cube",
        object_layout="specified held-out box and rotated unequal support with distractors",
        required_active_arm="right",
        pre_grasp_distance_mm=90.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=6,
        verification_mode="place_on_target",
        required_gripper_state="open",
        container_attr="base_cube",
        max_target_container_xy_offset_mm=52.0,
        verification_method_override="full-planner-expert",
    ),
    "click_bell_right": GraspLiftSafetyProfile(
        method="click_bell_right_full_press_v1",
        actor_attr="bell",
        object_layout="static bell in the right-arm workspace",
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=0.0,
        min_verified_lift_mm=0.0,
        min_right_arm_plan_segments=3,
        verification_mode="press_contact",
        required_gripper_state="closed",
        verification_method_override="full-planner-expert",
    ),
    "beat_block_hammer_right": GraspLiftSafetyProfile(
        method="beat_block_hammer_right_full_tool_contact_v1",
        actor_attr="hammer",
        object_layout="hammer and static red block in the right-arm workspace",
        required_active_arm="right",
        pre_grasp_distance_mm=120.0,
        commanded_lift_mm=100.0,
        min_verified_lift_mm=40.0,
        min_right_arm_plan_segments=5,
        verification_mode="tool_contact",
        required_gripper_state="closed",
        verification_method_override="full-planner-expert",
        contact_actor_attr="block",
    ),
    "place_shoe": GraspLiftSafetyProfile(
        method="shoe_selected_arm_full_pick_place_v1",
        actor_attr="shoe",
        object_layout="shoe and blue target mat in the dual-arm workspace",
        required_active_arm="both",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=70.0,
        min_verified_lift_mm=0.0,
        min_right_arm_plan_segments=4,
        min_left_arm_plan_segments=4,
        arm_plan_policy="select_by_object_x",
        verification_mode="place_by_selected_arm",
        required_gripper_state="both_open",
        container_attr="target_block",
        max_target_container_xy_offset_mm=55.0,
        verification_method_override="full-planner-expert",
    ),
    "handover_mic": GraspLiftSafetyProfile(
        method="microphone_dual_arm_handover_v2",
        actor_attr="microphone",
        object_layout="microphone initialized horizontally in either arm workspace and presented vertically",
        required_active_arm="both",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=3,
        min_left_arm_plan_segments=3,
        arm_plan_policy="both",
        verification_mode="dual_handover",
        required_gripper_state="handover",
        verification_method_override="full-planner-expert",
    ),
    "handover_horizontal_block": GraspLiftSafetyProfile(
        method="horizontal_block_vertical_dual_arm_handover_v2",
        actor_attr="block",
        object_layout="horizontal 3x3x20 cm block initialized in either arm workspace and presented vertically",
        required_active_arm="both",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=3,
        min_left_arm_plan_segments=3,
        arm_plan_policy="both",
        verification_mode="dual_handover",
        required_gripper_state="handover",
        verification_method_override="full-planner-expert",
    ),
    "handover_block": GraspLiftSafetyProfile(
        method="block_left_to_right_handover_place_v1",
        actor_attr="box",
        object_layout="red block in the left workspace and blue target pad in the right workspace",
        required_active_arm="both",
        pre_grasp_distance_mm=70.0,
        commanded_lift_mm=100.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=4,
        min_left_arm_plan_segments=4,
        arm_plan_policy="both",
        verification_mode="handover_place",
        required_gripper_state="both_open",
        container_attr="target_box",
        max_target_container_xy_offset_mm=45.0,
        verification_method_override="full-planner-expert",
    ),
    "handover_cube_to_target": GraspLiftSafetyProfile(
        method="dynamic_cube_two_stage_relay_place_v1",
        actor_attr="cube",
        object_layout="red cube and green target pad on opposite sides with a yellow middle transfer pad",
        required_active_arm="both",
        pre_grasp_distance_mm=80.0,
        commanded_lift_mm=100.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=4,
        min_left_arm_plan_segments=4,
        arm_plan_policy="both",
        verification_mode="handover_place",
        required_gripper_state="both_open",
        container_attr="target_pad",
        max_target_container_xy_offset_mm=45.0,
        verification_method_override="full-planner-expert",
    ),
    "lift_pot": GraspLiftSafetyProfile(
        method="pot_dual_arm_synchronized_lift_v1",
        actor_attr="pot",
        object_layout="pot centered between two arms with two grasp handles",
        required_active_arm="both",
        pre_grasp_distance_mm=35.0,
        commanded_lift_mm=100.0,
        min_verified_lift_mm=50.0,
        min_right_arm_plan_segments=3,
        min_left_arm_plan_segments=3,
        arm_plan_policy="both",
        verification_mode="dual_lift",
        required_gripper_state="both_closed",
        verification_method_override="full-planner-expert",
    ),
    "put_everything_in_basket": GraspLiftSafetyProfile(
        method="five_object_right_full_pick_place_v1",
        actor_attr="target_objects",
        object_layout=(
            "five randomized cubes, bottles, and pens placed one by one in a large basket"
        ),
        required_active_arm="right",
        pre_grasp_distance_mm=100.0,
        commanded_lift_mm=120.0,
        min_verified_lift_mm=30.0,
        min_right_arm_plan_segments=3,
        verification_mode="multi_object_place_in_container",
        required_gripper_state="open",
        verification_method_override="full-planner-expert",
    ),
}

PROBE_OUTPUT_DIR = Path("/tmp/agent_safe_seed_probe")
DIRECT_GRASP_METHOD = "direct-grasp-lift"
FULL_EXPERT_METHOD = "full-planner-expert"
CONTACT_PRIORITY = (5, 1, 4, 0, 2, 6, 7, 3)
CONTACT_TO_GRIPPER_ROTATION = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
GRIPPER_CENTER_OFFSET_M = 0.12
DIRECT_IK_THRESHOLD_M = 0.003
MAX_EE_POSITION_ERROR_MM = 3.0
MAX_PRECLOSE_OBJECT_DISPLACEMENT_MM = 5.0
GRIPPER_CLOSE_STEPS = 200
PHYSICS_SETTLE_STEPS = 30
POST_TELEPORT_SETTLE_STEPS = 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select evaluation seeds with physical grasp verification, without saving "
            "trajectories or observations."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--active-arm", choices=["left", "right", "both"], default="right")
    parser.add_argument("--candidate-seeds", nargs="+", type=int, default=list(range(10, 50)))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--verification-method",
        choices=[DIRECT_GRASP_METHOD, FULL_EXPERT_METHOD],
        default=DIRECT_GRASP_METHOD,
        help=(
            "direct-grasp-lift solves the final grasp IK, initializes only the robot at "
            "that qpos, closes physically, and verifies the post-grasp outcome. "
            "full-planner-expert preserves RoboTwin's original play_once seed filter."
        ),
    )
    parser.add_argument(
        "--min-object-lift-mm",
        type=float,
        help=(
            "Override the task profile's minimum target-object Z increase. "
            "All four built-in grasp profiles default to 50 mm; placement profiles ignore it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".agent/runs/data/safe_eval_seed_sets",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("count must be positive")
    if args.min_object_lift_mm is not None and args.min_object_lift_mm <= 0:
        raise ValueError("min-object-lift-mm must be positive")
    candidates = list(dict.fromkeys(args.candidate_seeds))
    if len(candidates) < args.count:
        raise ValueError("candidate seed count is smaller than requested output count")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing_tasks: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("config") == args.config
            and existing_manifest.get("active_arm") == args.active_arm
        ):
            existing_tasks = dict(existing_manifest.get("tasks", {}))
    verification_level = verification_level_for(args.verification_method)
    summary: dict[str, Any] = {
        "format": "robotwin_safe_eval_seed_sets_v4",
        "requested_verification_method": args.verification_method,
        "verification_method": args.verification_method,
        "verification_level": verification_level,
        "criteria": criteria_for(args.verification_method),
        "config": args.config,
        "active_arm": args.active_arm,
        "requested_count_per_task": args.count,
        "candidate_seeds": candidates,
        "tasks": existing_tasks,
    }

    for task in args.tasks:
        profile = safety_profile(task)
        verification_method = (
            profile.verification_method_override or args.verification_method
        )
        min_object_lift_mm = profile.min_verified_lift_mm
        if profile.verification_mode == "grasp_lift" and args.min_object_lift_mm is not None:
            min_object_lift_mm = args.min_object_lift_mm
        task_result = select_for_task(
            task=task,
            config=args.config,
            active_arm=args.active_arm,
            profile=profile,
            candidates=candidates,
            count=args.count,
            min_object_lift_mm=min_object_lift_mm,
            verification_method=verification_method,
            output_path=output_dir / f"{task}.json",
        )
        summary["tasks"][task] = {
            "selected_seeds": task_result["selected_seeds"],
            "probe_count": len(task_result["probes"]),
            "verification_method": verification_method,
            "verification_level": verification_level_for(verification_method),
            "selection_criteria": task_result["selection_criteria"],
            "safety_profile": task_result["safety_profile"],
            "result_file": f"{task}.json",
        }
        summary = merge_and_write_manifest(manifest_path, summary)

    print(json.dumps(summary, indent=2), flush=True)


def select_for_task(
    *,
    task: str,
    config: str,
    active_arm: str,
    profile: GraspLiftSafetyProfile,
    candidates: list[int],
    count: int,
    min_object_lift_mm: float,
    verification_method: str,
    output_path: Path,
    resume: bool = False,
    abort_on_infrastructure_error: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": "robotwin_safe_eval_seed_set_v4",
        "verification_method": verification_method,
        "verification_level": verification_level_for(verification_method),
        "task": task,
        "config": config,
        "active_arm": active_arm,
        "actor_attr": profile.actor_attr,
        "safety_profile": asdict(profile),
        "selection_criteria": selection_criteria(
            profile,
            min_object_lift_mm,
            verification_method,
        ),
        "min_object_lift_mm": min_object_lift_mm,
        "trajectory_saved": False,
        "images_saved": False,
        "requested_count": count,
        "candidate_seeds": candidates,
        "selected_seeds": [],
        "probes": [],
        "infrastructure_failures": [],
        "complete": False,
    }

    if resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_infrastructure = list(previous.get("infrastructure_failures", []))
        same_verification_method = (
            previous.get("verification_method") == verification_method
        )
        previous_probes = (
            list(previous.get("probes", [])) if same_verification_method else []
        )
        valid_probes = []
        for probe in previous_probes:
            if is_infrastructure_probe_failure(probe):
                previous_infrastructure.append(
                    {"seed": probe.get("seed"), "error": probe.get("error")}
                )
            else:
                valid_probes.append(probe)
        result["probes"] = valid_probes
        result["infrastructure_failures"] = previous_infrastructure
        result["selected_seeds"] = [
            int(probe["seed"]) for probe in valid_probes if probe.get("ready")
        ]
        result["complete"] = len(result["selected_seeds"]) >= count
        if result["complete"]:
            write_json(output_path, result)
            return result

    evaluated_seeds = {int(probe["seed"]) for probe in result["probes"]}
    for seed in candidates:
        if seed in evaluated_seeds:
            continue
        print(f"probe task={task} seed={seed}", flush=True)
        if profile.verification_mode == "multi_object_place_in_container":
            probe_func = probe_multi_object_place_task
        else:
            probe_func = (
                probe_direct_grasp_task
                if verification_method == DIRECT_GRASP_METHOD
                else probe_full_expert_task
            )
        probe = probe_func(
            task=task,
            config=config,
            active_arm=active_arm,
            profile=profile,
            seed=seed,
            min_object_lift_mm=min_object_lift_mm,
        )
        if is_infrastructure_probe_failure(probe):
            result["infrastructure_failures"].append(
                {"seed": seed, "error": probe.get("error")}
            )
            write_json(output_path, result)
            if abort_on_infrastructure_error:
                raise InfrastructureProbeError(
                    f"infrastructure failure for {task} seed={seed}: "
                    f"{probe.get('error')}"
                )
            continue
        result["probes"].append(probe)
        evaluated_seeds.add(seed)
        if probe.get("ready"):
            result["selected_seeds"].append(seed)
            print(
                f"accepted task={task} seed={seed} "
                f"count={len(result['selected_seeds'])}/{count}",
                flush=True,
            )
        else:
            reported_lift_mm = (
                probe.get("peak_object_lift_mm")
                if profile.verification_mode != "grasp_lift"
                else probe.get("object_lift_mm")
            )
            print(
                f"rejected task={task} seed={seed} "
                f"task_success={probe.get('task_success')} "
                f"lift_mm={reported_lift_mm} "
                f"planner={probe.get('planner_success')} "
                f"error={probe.get('error')}",
                flush=True,
            )
        result["complete"] = len(result["selected_seeds"]) >= count
        write_json(output_path, result)
        if result["complete"]:
            break

    if not result["complete"]:
        raise RuntimeError(
            f"only found {len(result['selected_seeds'])}/{count} safe seeds for {task}"
        )
    return result


class InfrastructureProbeError(RuntimeError):
    pass


def is_infrastructure_probe_failure(probe: dict[str, Any]) -> bool:
    error = str(probe.get("error") or "").lower()
    return any(
        marker in error
        for marker in (
            "failed to find a supported physical device",
            "cuda out of memory",
            "cuda driver error",
            "cuda initialization error",
        )
    )


def probe_multi_object_place_task(
    *,
    task: str,
    config: str,
    active_arm: str,
    profile: GraspLiftSafetyProfile,
    seed: int,
    min_object_lift_mm: float,
) -> dict[str, Any]:
    if active_arm != profile.required_active_arm:
        return failed_probe(
            seed,
            f"profile {profile.method} requires active_arm={profile.required_active_arm}",
            profile,
        )

    probe_dir = probe_output_dir(task, seed)
    object_results: list[dict[str, Any]] = []
    initial_layout_names: list[str] | None = None
    all_initial_poses_finite = True
    all_final_poses_finite = True
    try:
        for object_index in range(5):
            object_probe_dir = probe_dir / f"object_{object_index}"
            shutil.rmtree(object_probe_dir, ignore_errors=True)
            env = None
            env = InteractiveRoboTwinEnv(
                task_name=task,
                config_name=config,
                active_arm=active_arm,
                max_steps=1000,
                output_dir=object_probe_dir,
                save_images=False,
            )
            try:
                with robotwin_cwd():
                    env.reset(seed=seed)
                env.task.need_plan = True
                records = sorted(
                    env.task.target_objects,
                    key=lambda record: record["actor"].get_name(),
                )
                if len(records) != 5:
                    raise ValueError(f"expected 5 target objects, got {len(records)}")
                layout_names = [record["actor"].get_name() for record in records]
                if initial_layout_names is None:
                    initial_layout_names = layout_names
                elif layout_names != initial_layout_names:
                    raise RuntimeError("seeded target-object layout is not deterministic")

                record = records[object_index]
                actor = record["actor"]
                actor_name = actor.get_name()
                initial_xyz, initial_quat = actor_pose_arrays(actor)
                initial_pose_finite = finite_pose(initial_xyz, initial_quat)
                initially_contained = bool(env.task._is_record_contained(record))

                candidate = solve_direct_grasp_candidate(env, actor, profile)
                arm_qpos, actual_gripper_pose = initialize_robot_at_direct_grasp(
                    env,
                    candidate,
                )
                target_gripper_pose = np.asarray(
                    candidate["target_gripper_pose"],
                    dtype=np.float64,
                )
                ee_position_error_mm = float(
                    np.linalg.norm(
                        actual_gripper_pose[:3] - target_gripper_pose[:3]
                    )
                    * 1000.0
                )
                before_close_xyz, _ = actor_pose_arrays(actor)
                preclose_object_displacement_mm = float(
                    np.linalg.norm(before_close_xyz - initial_xyz) * 1000.0
                )
                initialization_ok = bool(
                    ee_position_error_mm <= MAX_EE_POSITION_ERROR_MM
                    and preclose_object_displacement_mm
                    <= MAX_PRECLOSE_OBJECT_DISPLACEMENT_MM
                )

                close_right_gripper_physically(env, arm_qpos)
                contact_count = len(
                    env.task.get_gripper_actor_contact_position(actor_name)
                )
                env.task.move(
                    env.task.move_by_displacement(
                        arm_tag=env.task.arm_tag,
                        z=profile.commanded_lift_mm / 1000.0,
                    )
                )
                env.task._update_lift_state()
                lifted_xyz, _ = actor_pose_arrays(actor)
                lift_mm = float((lifted_xyz[2] - initial_xyz[2]) * 1000.0)
                lifted = bool(
                    env.task.was_lifted.get(actor_name, False)
                    and lift_mm >= min_object_lift_mm
                )

                env.task._transfer_record_to_xy(
                    record,
                    env.task._safety_probe_target_xy(record),
                )

                if env.task.plan_success:
                    actor_aabb = env.task._world_aabb(record)
                    bottom_z = float(actor_aabb[0][2])
                    desired_bottom_z = env.task.basket_floor_top_z + 0.008
                    lower_by = float(
                        np.clip(desired_bottom_z - bottom_z, -0.25, 0.0)
                    )
                    if lower_by < -0.002:
                        env.task.move(
                            env.task.move_by_displacement(
                                arm_tag=env.task.arm_tag,
                                z=lower_by,
                            )
                        )
                if env.task.plan_success:
                    env.task.move(env.task.open_gripper(arm_tag=env.task.arm_tag))
                    settle_physics(env, max(PHYSICS_SETTLE_STEPS, 100))

                final_xyz, final_quat = actor_pose_arrays(actor)
                final_pose_finite = finite_pose(final_xyz, final_quat)
                containment_status = env.task._record_containment_status(record)
                contained = bool(env.task._is_record_contained(record))
                right_gripper_open = bool(env.task.is_right_gripper_open())
                waypoint_counts = planner_waypoint_counts(
                    getattr(env.task, "right_joint_path", [])
                )
                planner_success = bool(
                    env.task.plan_success
                    and waypoint_counts
                    and all(count > 0 for count in waypoint_counts)
                )
                physical_grasp_verified = bool(contact_count > 0 or lifted)
                ready = bool(
                    initial_pose_finite
                    and final_pose_finite
                    and not initially_contained
                    and initialization_ok
                    and candidate["collision_free"]
                    and physical_grasp_verified
                    and lifted
                    and planner_success
                    and contained
                    and right_gripper_open
                )
                all_initial_poses_finite &= initial_pose_finite
                all_final_poses_finite &= final_pose_finite
                object_results.append(
                    {
                        "object": actor_name,
                        "kind": record["kind"],
                        "orientation": record.get("orientation"),
                        "ready": ready,
                        "initially_contained": initially_contained,
                        "direct_ik_success": True,
                        "collision_free": candidate["collision_free"],
                        "initialization_ok": initialization_ok,
                        "ee_position_error_mm": round(ee_position_error_mm, 6),
                        "preclose_object_displacement_mm": round(
                            preclose_object_displacement_mm,
                            6,
                        ),
                        "gripper_contact_count_after_close": contact_count,
                        "physical_grasp_verified": physical_grasp_verified,
                        "object_lift_mm": round(lift_mm, 3),
                        "object_lifted": lifted,
                        "planner_success": planner_success,
                        "right_arm_waypoint_counts": waypoint_counts,
                        "contained": contained,
                        "containment_status": containment_status,
                        "right_gripper_open": right_gripper_open,
                    }
                )
                print(
                    f"safe-probe task={task} seed={seed} "
                    f"object={actor_name} ready={ready} "
                    f"lift_mm={lift_mm:.1f} contained={contained}",
                    flush=True,
                )
                if not ready:
                    break
            finally:
                if env is not None:
                    env.close()
                shutil.rmtree(object_probe_dir, ignore_errors=True)

        ready_count = sum(int(result["ready"]) for result in object_results)
        ready = ready_count == 5
        return {
            "seed": seed,
            "ready": ready,
            "active_arm": active_arm,
            "actor_attr": "target_objects",
            "expert_execution": (
                "five independent seeded physical pick-lift-place verifications"
            ),
            "safety_method": profile.method,
            "safety_profile": asdict(profile),
            "initial_target_pose_finite": all_initial_poses_finite,
            "final_target_pose_finite": all_final_poses_finite,
            "planner_success": ready,
            "task_success": ready,
            "right_gripper_closed": False,
            "right_gripper_open": ready,
            "left_gripper_closed": False,
            "left_gripper_open": True,
            "required_gripper_state": profile.required_gripper_state,
            "required_gripper_state_ok": ready,
            "complete_right_arm_plan_chain": ready,
            "complete_left_arm_plan_chain": True,
            "arm_plan_policy": profile.arm_plan_policy,
            "arm_plan_policy_ok": ready,
            "selected_arm": "right",
            "left_arm_plan_unused": True,
            "object_lifted": ready,
            "multi_object_lifted_count": sum(
                int(result["object_lifted"]) for result in object_results
            ),
            "multi_object_contained_count": sum(
                int(result["contained"]) for result in object_results
            ),
            "multi_object_ready_count": ready_count,
            "multi_object_status": object_results,
            "object_lift_mm": min(
                (result["object_lift_mm"] for result in object_results),
                default=None,
            ),
            "peak_object_lift_mm": max(
                (result["object_lift_mm"] for result in object_results),
                default=None,
            ),
            "min_object_lift_mm": min_object_lift_mm,
            "task_specific_outcome": ready,
            "trajectory_saved": False,
            "images_saved": False,
            "error": None,
        }
    except Exception as exc:
        probe = failed_probe(seed, f"{type(exc).__name__}: {exc}", profile)
        probe["multi_object_status"] = object_results
        probe["multi_object_ready_count"] = sum(
            int(result.get("ready", False)) for result in object_results
        )
        return probe
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def probe_direct_grasp_task(
    *,
    task: str,
    config: str,
    active_arm: str,
    profile: GraspLiftSafetyProfile,
    seed: int,
    min_object_lift_mm: float,
) -> dict[str, Any]:
    if active_arm != profile.required_active_arm:
        return failed_probe(
            seed,
            f"profile {profile.method} requires active_arm={profile.required_active_arm}",
            profile,
        )

    probe_dir = probe_output_dir(task, seed)
    shutil.rmtree(probe_dir, ignore_errors=True)
    env = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=probe_dir,
        save_images=False,
    )
    try:
        with robotwin_cwd():
            env.reset(seed=seed)

        actor_name, actor = resolve_actor(env, profile.actor_attr)
        initial_xyz, initial_quat = actor_pose_arrays(actor)
        initial_pose_finite = finite_pose(initial_xyz, initial_quat)
        candidate = solve_direct_grasp_candidate(env, actor, profile)
        arm_qpos, actual_gripper_pose = initialize_robot_at_direct_grasp(
            env,
            candidate,
        )

        target_gripper_pose = np.asarray(candidate["target_gripper_pose"], dtype=np.float64)
        ee_position_error_mm = float(
            np.linalg.norm(actual_gripper_pose[:3] - target_gripper_pose[:3]) * 1000.0
        )
        before_close_xyz, _ = actor_pose_arrays(actor)
        preclose_object_displacement_mm = float(
            np.linalg.norm(before_close_xyz - initial_xyz) * 1000.0
        )
        initialization_ok = bool(
            ee_position_error_mm <= MAX_EE_POSITION_ERROR_MM
            and preclose_object_displacement_mm <= MAX_PRECLOSE_OBJECT_DISPLACEMENT_MM
        )

        close_right_gripper_physically(env, arm_qpos)
        gripper_contact_count_after_close = len(
            env.task.get_gripper_actor_contact_position(actor.get_name())
        )

        if profile.verification_mode == "grasp_lift":
            post_grasp = execute_grasp_lift_verification(env, profile)
        else:
            post_grasp = execute_place_in_container_verification(
                env,
                actor,
                profile,
            )

        settle_physics(env, PHYSICS_SETTLE_STEPS)
        final_xyz, final_quat = actor_pose_arrays(actor)
        if multi_object_records is not None:
            final_object_poses = {
                record["actor"].get_name(): actor_pose_arrays(record["actor"])
                for record in multi_object_records
            }
            final_pose_finite = all(
                finite_pose(xyz, quat)
                for xyz, quat in final_object_poses.values()
            )
        else:
            final_object_poses = None
            final_pose_finite = finite_pose(final_xyz, final_quat)
        object_lift_mm = float((final_xyz[2] - initial_xyz[2]) * 1000.0)
        peak_object_lift_mm = max(object_lift_mm, post_grasp["peak_object_lift_mm"])
        object_lifted = bool(peak_object_lift_mm >= min_object_lift_mm)
        planner_success = bool(post_grasp["planner_success"])
        task_success = bool(env.task.check_success())
        right_gripper_closed = bool(env.task.is_right_gripper_close())
        right_gripper_open = bool(env.task.is_right_gripper_open())
        required_gripper_state_ok = bool(
            right_gripper_closed
            if profile.required_gripper_state == "closed"
            else right_gripper_open
        )

        target_container_xy_offset_mm = None
        target_container_xy_ok = True
        if profile.container_attr is not None:
            _, container = resolve_actor(env, profile.container_attr)
            container_xyz, container_quat = actor_pose_arrays(container)
            final_pose_finite = bool(
                final_pose_finite and finite_pose(container_xyz, container_quat)
            )
            target_container_xy_offset_mm = float(
                np.linalg.norm(final_xyz[:2] - container_xyz[:2]) * 1000.0
            )
            if profile.max_target_container_xy_offset_mm is not None:
                target_container_xy_ok = bool(
                    target_container_xy_offset_mm
                    <= profile.max_target_container_xy_offset_mm
                )

        press_contact_verified = bool(
            profile.verification_mode == "press_contact"
            and task_success
            and getattr(env.task, "stage_success_tag", False)
        )
        if profile.verification_mode == "grasp_lift":
            task_specific_outcome = object_lifted
        elif profile.verification_mode == "press_contact":
            task_specific_outcome = press_contact_verified
        else:
            task_specific_outcome = object_lifted and target_container_xy_ok
        physical_grasp_verified = bool(
            gripper_contact_count_after_close > 0 or object_lifted
        )
        ready = bool(
            initial_pose_finite
            and final_pose_finite
            and initialization_ok
            and candidate["collision_free"]
            and physical_grasp_verified
            and planner_success
            and task_success
            and required_gripper_state_ok
            and task_specific_outcome
        )
        return {
            "seed": seed,
            "ready": ready,
            "active_arm": env.active_arm,
            "actor_attr": actor_name,
            "expert_execution": "direct final-grasp IK/qpos + physical close + post-grasp task",
            "safety_method": DIRECT_GRASP_METHOD,
            "safety_profile": asdict(profile),
            "initial_target_pose_finite": initial_pose_finite,
            "final_target_pose_finite": final_pose_finite,
            "direct_ik_success": True,
            "grasp_qpos_solver": candidate["solver"],
            "solver_waypoint_count_not_executed": candidate[
                "solver_waypoint_count_not_executed"
            ],
            "direct_ik_candidate_count": candidate["ik_candidate_count"],
            "collision_free_ik_candidate_count": candidate[
                "collision_free_candidate_count"
            ],
            "selected_contact_id": candidate["contact_id"],
            "selected_rotation_id": candidate["rotation_id"],
            "selected_max_joint_delta_rad": round(
                candidate["max_joint_delta_rad"],
                6,
            ),
            "selected_self_collision_count": candidate["self_collision_count"],
            "selected_environment_collision_count": candidate[
                "environment_collision_count"
            ],
            "selected_allowed_contact_count": candidate["allowed_contact_count"],
            "selected_blocking_collision_count": candidate[
                "blocking_collision_count"
            ],
            "collision_free": candidate["collision_free"],
            "ee_position_error_mm": round(ee_position_error_mm, 6),
            "max_ee_position_error_mm": MAX_EE_POSITION_ERROR_MM,
            "preclose_object_displacement_mm": round(
                preclose_object_displacement_mm,
                6,
            ),
            "max_preclose_object_displacement_mm": (
                MAX_PRECLOSE_OBJECT_DISPLACEMENT_MM
            ),
            "initialization_ok": initialization_ok,
            "gripper_contact_count_after_close": gripper_contact_count_after_close,
            "physical_grasp_verified": physical_grasp_verified,
            "planner_success": planner_success,
            "task_success": task_success,
            "right_gripper_closed": right_gripper_closed,
            "right_gripper_open": right_gripper_open,
            "required_gripper_state": profile.required_gripper_state,
            "required_gripper_state_ok": required_gripper_state_ok,
            "post_grasp_waypoint_counts": post_grasp["waypoint_counts"],
            "object_lifted": object_lifted,
            "object_lift_mm": round(object_lift_mm, 3),
            "peak_object_lift_mm": round(peak_object_lift_mm, 3),
            "min_object_lift_mm": min_object_lift_mm,
            "task_specific_outcome": task_specific_outcome,
            "press_contact_verified": press_contact_verified,
            "target_container_xy_offset_mm": (
                round(target_container_xy_offset_mm, 3)
                if target_container_xy_offset_mm is not None
                else None
            ),
            "target_container_xy_ok": target_container_xy_ok,
            "trajectory_saved": False,
            "images_saved": False,
            "error": None,
        }
    except Exception as exc:
        return failed_probe(seed, f"{type(exc).__name__}: {exc}", profile)
    finally:
        env.close()
        shutil.rmtree(probe_dir, ignore_errors=True)


def solve_direct_grasp_candidate(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    profile: GraspLiftSafetyProfile,
) -> dict[str, Any]:
    robot = env.task.robot
    planner = robot.right_mplib_planner.planner
    start_qpos = np.asarray(robot.right_entity.get_qpos(), dtype=np.float32)
    move_group_indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
    preferred_direction = robot.get_grasp_perfect_direction("right")
    root_pose = robot.right_entity.get_root_pose()

    contact_ids = [index for index, _ in actor.iter_contact_points()]
    ordered_contact_ids = [index for index in CONTACT_PRIORITY if index in contact_ids]
    ordered_contact_ids.extend(
        index for index in contact_ids if index not in ordered_contact_ids
    )
    candidates: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    ik_candidate_count = 0
    collision_free_candidate_count = 0

    for contact_rank, contact_id in enumerate(ordered_contact_ids):
        contact_matrix = actor.get_contact_point(contact_id, "matrix")
        if contact_matrix is None:
            continue
        grasp_matrix = np.asarray(contact_matrix, dtype=np.float64) @ CONTACT_TO_GRIPPER_ROTATION
        grasp_rotation = grasp_matrix[:3, :3]
        grasp_position = grasp_matrix[:3, 3] + grasp_rotation @ np.array(
            [-GRIPPER_CENTER_OFFSET_M, 0.0, 0.0],
            dtype=np.float64,
        )
        origin_pose = grasp_position.tolist() + t3d.quaternions.mat2quat(
            grasp_rotation
        ).tolist()
        contact_center = actor.get_contact_point(contact_id, "list")
        target_poses = robot.create_target_pose_list(
            origin_pose,
            contact_center,
            "right",
        )

        for rotation_id, target_pose in enumerate(target_poses):
            top_down_distance = float(
                cal_quat_dis(
                    target_pose[-4:],
                    GRASP_DIRECTION_DIC["top_down_little_left"],
                )
            )
            side_distance = float(
                cal_quat_dis(
                    target_pose[-4:],
                    GRASP_DIRECTION_DIC[preferred_direction],
                )
            )
            target_record = {
                "target_gripper_pose": np.asarray(target_pose, dtype=np.float64),
                "contact_id": int(contact_id),
                "contact_rank": contact_rank,
                "rotation_id": rotation_id,
                "orientation_distance": min(top_down_distance, side_distance),
            }
            target_records.append(target_record)
            endlink_world_pose = robot._trans_from_gripper_to_endlink(
                target_pose,
                arm_tag="right",
            )
            endlink_base_pose = root_pose.inv() * endlink_world_pose
            status, solved_qpos = planner.IK(
                mplib.Pose(endlink_base_pose.p, endlink_base_pose.q),
                start_qpos,
                n_init_qpos=8,
                threshold=DIRECT_IK_THRESHOLD_M,
                return_closest=True,
            )
            if status != "Success" or solved_qpos is None:
                continue
            ik_candidate_count += 1
            solved_qpos = np.asarray(solved_qpos, dtype=np.float32)
            collision = final_grasp_collision_summary(
                env,
                actor,
                planner,
                solved_qpos,
            )
            self_collision_count = collision["self_collision_count"]
            environment_collision_count = collision["environment_collision_count"]
            collision_free = collision["blocking_collision_count"] == 0
            if not collision_free:
                continue
            collision_free_candidate_count += 1

            max_joint_delta_rad = float(
                np.max(
                    np.abs(
                        solved_qpos[move_group_indices]
                        - start_qpos[move_group_indices]
                    )
                )
            )
            candidates.append(
                {
                    "qpos": solved_qpos,
                    **target_record,
                    "max_joint_delta_rad": max_joint_delta_rad,
                    "self_collision_count": self_collision_count,
                    "environment_collision_count": environment_collision_count,
                    "allowed_contact_count": collision["allowed_contact_count"],
                    "blocking_collision_count": collision[
                        "blocking_collision_count"
                    ],
                    "collision_free": collision_free,
                    "solver": "mplib-direct-ik",
                    "solver_waypoint_count_not_executed": 0,
                }
            )

    if not candidates:
        fallback = solve_curobo_grasp_qpos_fallback(
            env,
            actor,
            profile,
            start_qpos,
            move_group_indices,
            planner,
            target_records,
        )
        fallback["ik_candidate_count"] = ik_candidate_count
        fallback["collision_free_candidate_count"] = int(
            fallback["collision_free"]
        )
        return fallback
    selected = min(
        candidates,
        key=lambda item: (
            item["orientation_distance"],
            item["contact_rank"],
            item["max_joint_delta_rad"],
        ),
    )
    selected["ik_candidate_count"] = ik_candidate_count
    selected["collision_free_candidate_count"] = collision_free_candidate_count
    return selected


def solve_curobo_grasp_qpos_fallback(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    profile: GraspLiftSafetyProfile,
    start_qpos: np.ndarray,
    move_group_indices: np.ndarray,
    collision_planner: Any,
    target_records: list[dict[str, Any]],
) -> dict[str, Any]:
    pregrasp_pose, grasp_pose = env.task.choose_grasp_pose(
        actor,
        arm_tag=env.task.arm_tag,
        pre_dis=profile.pre_grasp_distance_mm / 1000.0,
        target_dis=0.0,
    )
    del pregrasp_pose
    grasp_pose_array = np.asarray(grasp_pose, dtype=np.float64)
    if grasp_pose_array.shape != (7,) or not np.all(np.isfinite(grasp_pose_array)):
        raise RuntimeError("neither MPlib IK nor CuRobo produced a valid grasp pose")
    endpoint_plan = env.task.robot.right_plan_path(
        grasp_pose_array.tolist(),
        last_qpos=start_qpos,
    )
    if endpoint_plan.get("status") != "Success":
        raise RuntimeError(
            "no direct MPlib IK and CuRobo could not solve the final grasp endpoint"
        )

    arm_qpos = np.asarray(endpoint_plan["position"], dtype=np.float32)[-1]
    solved_qpos = np.asarray(start_qpos, dtype=np.float32).copy()
    solved_qpos[move_group_indices] = arm_qpos
    collision = final_grasp_collision_summary(
        env,
        actor,
        collision_planner,
        solved_qpos,
    )
    self_collision_count = collision["self_collision_count"]
    environment_collision_count = collision["environment_collision_count"]
    collision_free = collision["blocking_collision_count"] == 0
    if not collision_free:
        raise RuntimeError(
            "CuRobo final grasp qpos failed MPlib collision validation "
            f"(self={self_collision_count}, environment={environment_collision_count})"
        )

    nearest_target = min(
        target_records,
        key=lambda item: pose_distance(
            item["target_gripper_pose"],
            grasp_pose_array,
        ),
    )
    return {
        **nearest_target,
        "target_gripper_pose": grasp_pose_array,
        "qpos": solved_qpos,
        "max_joint_delta_rad": float(
            np.max(
                np.abs(
                    solved_qpos[move_group_indices]
                    - start_qpos[move_group_indices]
                )
            )
        ),
        "self_collision_count": self_collision_count,
        "environment_collision_count": environment_collision_count,
        "allowed_contact_count": collision["allowed_contact_count"],
        "blocking_collision_count": collision["blocking_collision_count"],
        "collision_free": collision_free,
        "solver": "curobo-final-path-endpoint",
        "solver_waypoint_count_not_executed": int(
            len(endpoint_plan.get("position", []))
        ),
    }


def final_grasp_collision_summary(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    planner: Any,
    qpos: np.ndarray,
) -> dict[str, int]:
    self_collisions = list(planner.check_for_self_collision(qpos))
    environment_collisions = list(planner.check_for_env_collision(qpos))
    gripper_links = {
        joint[0].child_link.get_name() for joint in env.task.robot.right_gripper
    }
    allowed_contacts = [
        collision
        for collision in environment_collisions
        if allowed_final_grasp_contact(collision, actor, gripper_links)
    ]
    blocking_collision_count = (
        len(self_collisions) + len(environment_collisions) - len(allowed_contacts)
    )
    return {
        "self_collision_count": len(self_collisions),
        "environment_collision_count": len(environment_collisions),
        "allowed_contact_count": len(allowed_contacts),
        "blocking_collision_count": blocking_collision_count,
    }


def allowed_final_grasp_contact(
    collision: Any,
    actor: Any,
    gripper_links: set[str],
) -> bool:
    link1 = str(getattr(collision, "link_name1", ""))
    link2 = str(getattr(collision, "link_name2", ""))
    object1 = str(getattr(collision, "object_name1", ""))
    object2 = str(getattr(collision, "object_name2", ""))
    if link1 in gripper_links:
        other_names = (link2, object2)
    elif link2 in gripper_links:
        other_names = (link1, object1)
    else:
        return False

    normalized_other = " ".join(other_names).lower()
    actor_name = str(actor.get_name()).lower()
    target_contact = bool(actor_name and actor_name in normalized_other)
    support_contact = "table" in normalized_other
    return target_contact or support_contact


def pose_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(
        np.linalg.norm(first[:3] - second[:3])
        + cal_quat_dis(first[-4:], second[-4:])
    )


def initialize_robot_at_direct_grasp(
    env: InteractiveRoboTwinEnv,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    robot = env.task.robot
    planner = robot.right_mplib_planner.planner
    move_group_indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
    full_qpos = np.asarray(robot.right_entity.get_qpos(), dtype=np.float32).copy()
    solved_qpos = np.asarray(candidate["qpos"], dtype=np.float32)
    full_qpos[move_group_indices] = solved_qpos[move_group_indices]
    arm_qpos = full_qpos[move_group_indices].copy()

    robot.right_entity.set_qpos(full_qpos)
    robot.right_entity.set_qvel(np.zeros(robot.right_entity.dof, dtype=np.float32))
    actual_gripper_pose = np.asarray(
        env.task.get_arm_pose("right"),
        dtype=np.float64,
    )
    for _ in range(POST_TELEPORT_SETTLE_STEPS):
        hold_right_arm(env, arm_qpos, gripper_value=1.0)
    return arm_qpos, actual_gripper_pose


def close_right_gripper_physically(
    env: InteractiveRoboTwinEnv,
    arm_qpos: np.ndarray,
) -> None:
    previous = 1.0
    for value in np.linspace(1.0, 0.0, GRIPPER_CLOSE_STEPS + 1)[1:]:
        hold_right_arm(
            env,
            arm_qpos,
            gripper_value=float(value),
            gripper_eps=float(value - previous),
        )
        previous = float(value)
    for _ in range(PHYSICS_SETTLE_STEPS):
        hold_right_arm(env, arm_qpos, gripper_value=0.0, gripper_eps=-0.01)


def open_right_gripper_physically(env: InteractiveRoboTwinEnv) -> None:
    robot = env.task.robot
    planner = robot.right_mplib_planner.planner
    indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
    arm_qpos = np.asarray(robot.right_entity.get_qpos(), dtype=np.float32)[indices]
    previous = 0.0
    for value in np.linspace(0.0, 1.0, GRIPPER_CLOSE_STEPS + 1)[1:]:
        hold_right_arm(
            env,
            arm_qpos,
            gripper_value=float(value),
            gripper_eps=float(value - previous),
        )
        previous = float(value)
    settle_physics(env, PHYSICS_SETTLE_STEPS)


def hold_right_arm(
    env: InteractiveRoboTwinEnv,
    arm_qpos: np.ndarray,
    *,
    gripper_value: float,
    gripper_eps: float | None = None,
) -> None:
    robot = env.task.robot
    robot.set_arm_joints(
        np.asarray(arm_qpos, dtype=np.float32),
        np.zeros(len(arm_qpos), dtype=np.float32),
        "right",
    )
    if gripper_eps is None:
        robot.set_gripper(gripper_value, "right")
    else:
        robot.set_gripper(gripper_value, "right", gripper_eps)
    env.task.scene.step()


def execute_grasp_lift_verification(
    env: InteractiveRoboTwinEnv,
    profile: GraspLiftSafetyProfile,
) -> dict[str, Any]:
    actor = getattr(env.task, profile.actor_attr)
    before_lift_xyz, _ = actor_pose_arrays(actor)
    target_pose = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    target_pose[2] += profile.commanded_lift_mm / 1000.0
    waypoint_count = plan_and_execute_right_pose(env, target_pose)
    after_lift_xyz, _ = actor_pose_arrays(actor)
    return {
        "planner_success": waypoint_count > 0,
        "waypoint_counts": [waypoint_count],
        "peak_object_lift_mm": float(
            (after_lift_xyz[2] - before_lift_xyz[2]) * 1000.0
        ),
    }


def execute_place_in_container_verification(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    profile: GraspLiftSafetyProfile,
) -> dict[str, Any]:
    if profile.container_attr is None:
        raise ValueError("placement profile requires container_attr")
    _, container = resolve_actor(env, profile.container_attr)
    initial_actor_xyz, _ = actor_pose_arrays(actor)
    waypoint_counts: list[int] = []

    lift_pose = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    lift_pose[2] += profile.commanded_lift_mm / 1000.0
    waypoint_counts.append(plan_and_execute_right_pose(env, lift_pose))
    lifted_actor_xyz, _ = actor_pose_arrays(actor)
    peak_object_lift_mm = float(
        (lifted_actor_xyz[2] - initial_actor_xyz[2]) * 1000.0
    )
    update_lift_state = getattr(env.task, "_update_lift_state", None)
    if callable(update_lift_state):
        update_lift_state()

    container_xyz, _ = actor_pose_arrays(container)
    waypoint_counts.extend(
        execute_segmented_actor_translation(
            env,
            actor,
            target_xyz=container_xyz,
            axes=(0, 1),
            max_step_m=0.06,
        )
    )

    if profile.verification_mode == "place_on_target":
        target_actor_z = float(
            container_xyz[2]
            + getattr(env.task, "target_half_height", 0.0)
            + getattr(env.task, "object_support_offset", 0.05)
        )
    else:
        target_actor_z = float(container_xyz[2] + 0.05)
    lower_target_xyz, _ = actor_pose_arrays(actor)
    lower_target_xyz[2] = target_actor_z
    waypoint_counts.extend(
        execute_segmented_actor_translation(
            env,
            actor,
            target_xyz=lower_target_xyz,
            axes=(2,),
            max_step_m=0.05,
        )
    )
    open_right_gripper_physically(env)

    retreat_pose = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
    retreat_pose[2] += 0.05 if profile.verification_mode == "place_on_target" else 0.10
    waypoint_counts.append(plan_and_execute_right_pose(env, retreat_pose))
    settle_physics(env, PHYSICS_SETTLE_STEPS * 2)
    return {
        "planner_success": bool(
            waypoint_counts and all(count > 0 for count in waypoint_counts)
        ),
        "waypoint_counts": waypoint_counts,
        "peak_object_lift_mm": peak_object_lift_mm,
    }


def execute_segmented_actor_translation(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    *,
    target_xyz: np.ndarray,
    axes: tuple[int, ...],
    max_step_m: float,
    tolerance_m: float = 0.002,
    max_segments: int = 16,
) -> list[int]:
    waypoint_counts: list[int] = []
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    for _ in range(max_segments):
        actor_xyz, _ = actor_pose_arrays(actor)
        remaining = np.zeros(3, dtype=np.float64)
        remaining[list(axes)] = target_xyz[list(axes)] - actor_xyz[list(axes)]
        max_axis_distance = float(np.max(np.abs(remaining[list(axes)])))
        if max_axis_distance <= tolerance_m:
            break
        step = remaining * min(1.0, max_step_m / max_axis_distance)
        target_pose = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
        target_pose[:3] += step
        waypoint_count = plan_and_execute_right_pose(env, target_pose)
        waypoint_counts.append(waypoint_count)
        if waypoint_count <= 0:
            break
    return waypoint_counts


def plan_and_execute_right_pose(
    env: InteractiveRoboTwinEnv,
    target_pose: np.ndarray,
) -> int:
    current_qpos = np.asarray(
        env.task.robot.right_entity.get_qpos(),
        dtype=np.float32,
    )
    plan = env.task.robot.right_plan_path(
        np.asarray(target_pose, dtype=np.float64).tolist(),
        last_qpos=current_qpos,
    )
    if plan.get("status") != "Success":
        return 0
    env.task.take_dense_action(
        {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": plan,
            "right_gripper": None,
        },
        save_freq=None,
    )
    return int(len(plan.get("position", [])))


def settle_physics(env: InteractiveRoboTwinEnv, steps: int) -> None:
    for _ in range(steps):
        env.task.scene.step()


def probe_full_expert_task(
    *,
    task: str,
    config: str,
    active_arm: str,
    profile: GraspLiftSafetyProfile,
    seed: int,
    min_object_lift_mm: float,
) -> dict[str, Any]:
    if active_arm != profile.required_active_arm:
        return failed_probe(
            seed,
            f"profile {profile.method} requires active_arm={profile.required_active_arm}",
            profile,
        )
    probe_dir = probe_output_dir(task, seed)
    shutil.rmtree(probe_dir, ignore_errors=True)
    env = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=probe_dir,
        save_images=False,
    )
    try:
        env.reset(seed=seed)
        multi_object_records = None
        if profile.verification_mode == "multi_object_place_in_container":
            multi_object_records = list(env.task.target_objects)
            if len(multi_object_records) != 5:
                raise ValueError(
                    f"expected 5 target objects, got {len(multi_object_records)}"
                )
            actor = multi_object_records[0]["actor"]
            actor_name = "target_objects"
            initial_object_poses = {
                record["actor"].get_name(): actor_pose_arrays(record["actor"])
                for record in multi_object_records
            }
            initial_xyz, initial_quat = initial_object_poses[actor.get_name()]
            initial_pose_finite = all(
                finite_pose(xyz, quat)
                for xyz, quat in initial_object_poses.values()
            )
        else:
            actor_name, actor = resolve_actor(env, profile.actor_attr)
            initial_xyz, initial_quat = actor_pose_arrays(actor)
            initial_pose_finite = finite_pose(initial_xyz, initial_quat)

        env.task.need_plan = True
        with robotwin_cwd():
            expert_result = env.task.play_once()

        final_xyz, final_quat = actor_pose_arrays(actor)
        if multi_object_records is not None:
            final_object_poses = {
                record["actor"].get_name(): actor_pose_arrays(record["actor"])
                for record in multi_object_records
            }
            final_pose_finite = all(
                finite_pose(xyz, quat)
                for xyz, quat in final_object_poses.values()
            )
        else:
            final_object_poses = None
            final_pose_finite = finite_pose(final_xyz, final_quat)
        object_lift_mm = float((final_xyz[2] - initial_xyz[2]) * 1000.0)
        planner_success = bool(getattr(env.task, "plan_success", True))
        task_success = bool(env.task.check_success())
        right_gripper_closed = bool(env.task.is_right_gripper_close())
        right_gripper_open = bool(env.task.is_right_gripper_open())
        left_gripper_closed = bool(env.task.is_left_gripper_close())
        left_gripper_open = bool(env.task.is_left_gripper_open())
        required_gripper_state_ok = gripper_state_matches(
            env.task,
            profile.required_gripper_state,
            right_closed=right_gripper_closed,
            right_open=right_gripper_open,
            left_closed=left_gripper_closed,
            left_open=left_gripper_open,
        )
        right_waypoint_counts = planner_waypoint_counts(getattr(env.task, "right_joint_path", []))
        left_waypoint_counts = planner_waypoint_counts(getattr(env.task, "left_joint_path", []))
        giver_arm = (
            str(env.task.grasp_arm_tag)
            if hasattr(env.task, "grasp_arm_tag")
            else None
        )
        receiver_arm = (
            str(env.task.handover_arm_tag)
            if hasattr(env.task, "handover_arm_tag")
            else None
        )
        vertical_presentation_observed = (
            bool(env.task.vertical_presentation_observed)
            if hasattr(env.task, "vertical_presentation_observed")
            else None
        )
        handover_completed = (
            bool(env.task.handover_completed)
            if hasattr(env.task, "handover_completed")
            else None
        )
        middle_release_observed = (
            bool(env.task.middle_release_observed)
            if hasattr(env.task, "middle_release_observed")
            else None
        )
        receiver_grasp_observed = (
            bool(env.task.receiver_grasp_observed)
            if hasattr(env.task, "receiver_grasp_observed")
            else None
        )
        plan_validation = validate_plan_chains(
            profile,
            initial_xyz=initial_xyz,
            right_waypoint_counts=right_waypoint_counts,
            left_waypoint_counts=left_waypoint_counts,
        )
        complete_right_arm_plan_chain = plan_validation["right_complete"]
        complete_left_arm_plan_chain = plan_validation["left_complete"]
        arm_plan_policy_ok = plan_validation["policy_ok"]
        selected_arm = plan_validation["selected_arm"]
        left_arm_plan_unused = not left_waypoint_counts
        peak_object_lift_mm = task_peak_object_lift_mm(env.task, object_lift_mm)
        object_lifted = bool(peak_object_lift_mm >= min_object_lift_mm)
        multi_object_lifted_count = None
        multi_object_contained_count = None
        multi_object_status = None
        if multi_object_records is not None:
            multi_object_status = {
                record["actor"].get_name(): {
                    "lifted": bool(
                        env.task.was_lifted.get(record["actor"].get_name(), False)
                    ),
                    "contained": bool(env.task._is_record_contained(record)),
                }
                for record in multi_object_records
            }
            multi_object_lifted_count = sum(
                int(status["lifted"]) for status in multi_object_status.values()
            )
            multi_object_contained_count = sum(
                int(status["contained"]) for status in multi_object_status.values()
            )
            object_lifted = multi_object_lifted_count == len(multi_object_records)
            peak_object_lift_mm = max(
                (
                    final_object_poses[name][0][2]
                    - initial_object_poses[name][0][2]
                )
                * 1000.0
                for name in initial_object_poses
            )
        target_container_xy_offset_mm = None
        target_container_xy_ok = True
        if profile.container_attr is not None:
            _, container = resolve_actor(env, profile.container_attr)
            container_xyz, container_quat = actor_pose_arrays(container)
            final_pose_finite = bool(
                final_pose_finite and finite_pose(container_xyz, container_quat)
            )
            target_container_xy_offset_mm = float(
                np.linalg.norm(final_xyz[:2] - container_xyz[:2]) * 1000.0
            )
            if profile.max_target_container_xy_offset_mm is not None:
                target_container_xy_ok = bool(
                    target_container_xy_offset_mm
                    <= profile.max_target_container_xy_offset_mm
                )
        press_contact_verified = bool(
            profile.verification_mode == "press_contact"
            and task_success
            and getattr(env.task, "stage_success_tag", False)
        )
        tool_contact_verified = False
        if profile.verification_mode == "tool_contact":
            if profile.contact_actor_attr is None:
                raise ValueError("tool_contact profile requires contact_actor_attr")
            _, contact_actor = resolve_actor(env, profile.contact_actor_attr)
            contact_xyz, contact_quat = actor_pose_arrays(contact_actor)
            final_pose_finite = bool(
                final_pose_finite and finite_pose(contact_xyz, contact_quat)
            )
            tool_contact_verified = bool(
                task_success
                and getattr(env.task, "hammer_lifted_with_grasp", False)
                and env.task.check_actors_contact(
                    actor.get_name(),
                    contact_actor.get_name(),
                )
            )
        if profile.verification_mode == "grasp_lift":
            task_specific_outcome = object_lifted
        elif profile.verification_mode == "press_contact":
            task_specific_outcome = press_contact_verified
        elif profile.verification_mode == "tool_contact":
            task_specific_outcome = object_lifted and tool_contact_verified
        elif profile.verification_mode == "place_by_selected_arm":
            task_specific_outcome = target_container_xy_ok
        elif profile.verification_mode in {"dual_handover", "dual_lift"}:
            task_specific_outcome = object_lifted
        elif profile.verification_mode == "handover_place":
            task_specific_outcome = object_lifted and target_container_xy_ok
        elif profile.verification_mode == "multi_object_place_in_container":
            task_specific_outcome = bool(
                multi_object_lifted_count == 5
                and multi_object_contained_count == 5
            )
        else:
            task_specific_outcome = object_lifted and target_container_xy_ok
        ready = bool(
            initial_pose_finite
            and final_pose_finite
            and planner_success
            and task_success
            and required_gripper_state_ok
            and arm_plan_policy_ok
            and task_specific_outcome
        )
        return {
            "seed": seed,
            "ready": ready,
            "active_arm": env.active_arm,
            "actor_attr": actor_name,
            "expert_execution": "task.play_once",
            "safety_method": profile.method,
            "safety_profile": asdict(profile),
            "initial_target_pose_finite": initial_pose_finite,
            "final_target_pose_finite": final_pose_finite,
            "planner_success": planner_success,
            "task_success": task_success,
            "right_gripper_closed": right_gripper_closed,
            "right_gripper_open": right_gripper_open,
            "left_gripper_closed": left_gripper_closed,
            "left_gripper_open": left_gripper_open,
            "required_gripper_state": profile.required_gripper_state,
            "required_gripper_state_ok": required_gripper_state_ok,
            "complete_right_arm_plan_chain": complete_right_arm_plan_chain,
            "complete_left_arm_plan_chain": complete_left_arm_plan_chain,
            "arm_plan_policy": profile.arm_plan_policy,
            "arm_plan_policy_ok": arm_plan_policy_ok,
            "selected_arm": selected_arm,
            "left_arm_plan_unused": left_arm_plan_unused,
            "giver_arm": giver_arm,
            "receiver_arm": receiver_arm,
            "vertical_presentation_observed": vertical_presentation_observed,
            "handover_completed": handover_completed,
            "middle_release_observed": middle_release_observed,
            "receiver_grasp_observed": receiver_grasp_observed,
            "right_arm_waypoint_counts": right_waypoint_counts,
            "left_arm_waypoint_counts": left_waypoint_counts,
            "object_lifted": object_lifted,
            "multi_object_lifted_count": multi_object_lifted_count,
            "multi_object_contained_count": multi_object_contained_count,
            "multi_object_status": multi_object_status,
            "object_lift_mm": round(object_lift_mm, 3),
            "peak_object_lift_mm": round(peak_object_lift_mm, 3),
            "min_object_lift_mm": min_object_lift_mm,
            "task_specific_outcome": task_specific_outcome,
            "press_contact_verified": press_contact_verified,
            "tool_contact_verified": tool_contact_verified,
            "target_container_xy_offset_mm": (
                round(target_container_xy_offset_mm, 3)
                if target_container_xy_offset_mm is not None
                else None
            ),
            "target_container_xy_ok": target_container_xy_ok,
            "expert_returned_mapping": isinstance(expert_result, dict),
            "trajectory_saved": False,
            "images_saved": False,
            "error": None,
        }
    except Exception as exc:
        return failed_probe(seed, f"{type(exc).__name__}: {exc}", profile)
    finally:
        env.close()
        shutil.rmtree(probe_dir, ignore_errors=True)


def actor_pose_arrays(actor: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = actor.get_pose()
    return (
        np.asarray(pose.p, dtype=np.float64),
        np.asarray(pose.q, dtype=np.float64),
    )


def task_peak_object_lift_mm(task: Any, final_object_lift_mm: float) -> float:
    if hasattr(task, "peak_object_lift_mm"):
        return float(task.peak_object_lift_mm)
    if hasattr(task, "peak_cube_lift_mm"):
        return float(task.peak_cube_lift_mm)
    if hasattr(task, "hammer_lift_peak") and hasattr(task, "hammer_start_height"):
        return float((task.hammer_lift_peak - task.hammer_start_height) * 1000.0)
    return float(final_object_lift_mm)


def finite_pose(xyz: np.ndarray, quat: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(xyz)) and np.all(np.isfinite(quat)))


def planner_waypoint_counts(paths: Any) -> list[int]:
    if not isinstance(paths, list):
        return []
    return [int(len(path.get("position", []))) for path in paths if isinstance(path, dict)]


def complete_plan_chain(counts: list[int], required_segments: int) -> bool:
    if required_segments <= 0:
        return True
    return bool(
        len(counts) >= required_segments
        and all(count > 0 for count in counts[:required_segments])
    )


def validate_plan_chains(
    profile: GraspLiftSafetyProfile,
    *,
    initial_xyz: np.ndarray,
    right_waypoint_counts: list[int],
    left_waypoint_counts: list[int],
) -> dict[str, Any]:
    right_complete = complete_plan_chain(
        right_waypoint_counts,
        profile.min_right_arm_plan_segments,
    )
    left_complete = complete_plan_chain(
        left_waypoint_counts,
        profile.min_left_arm_plan_segments,
    )
    selected_arm = None
    if profile.arm_plan_policy == "right_only":
        policy_ok = right_complete and not left_waypoint_counts
        selected_arm = "right"
    elif profile.arm_plan_policy == "both":
        policy_ok = right_complete and left_complete
        selected_arm = "both"
    elif profile.arm_plan_policy == "select_by_object_x":
        selected_arm = "left" if float(initial_xyz[0]) < 0 else "right"
        selected_complete = left_complete if selected_arm == "left" else right_complete
        unused_counts = right_waypoint_counts if selected_arm == "left" else left_waypoint_counts
        policy_ok = selected_complete and not unused_counts
    else:
        raise ValueError(f"unknown arm plan policy: {profile.arm_plan_policy}")
    return {
        "right_complete": right_complete,
        "left_complete": left_complete,
        "policy_ok": bool(policy_ok),
        "selected_arm": selected_arm,
    }


def gripper_state_matches(
    task: Any,
    required_state: str,
    *,
    right_closed: bool,
    right_open: bool,
    left_closed: bool,
    left_open: bool,
) -> bool:
    if required_state == "closed":
        return right_closed
    if required_state == "open":
        return right_open
    if required_state == "both_open":
        return left_open and right_open
    if required_state == "both_closed":
        return left_closed and right_closed
    if required_state == "handover":
        giver = str(getattr(task, "grasp_arm_tag", ""))
        receiver = str(getattr(task, "handover_arm_tag", ""))
        states = {
            "left": {"closed": left_closed, "open": left_open},
            "right": {"closed": right_closed, "open": right_open},
        }
        return bool(
            giver in states
            and receiver in states
            and states[giver]["open"]
            and states[receiver]["closed"]
        )
    raise ValueError(f"unknown required gripper state: {required_state}")


def safety_profile(task: str) -> GraspLiftSafetyProfile:
    try:
        return TASK_SAFETY_PROFILES[task]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_SAFETY_PROFILES))
        raise ValueError(f"no safety profile for {task!r}; supported: {supported}") from exc


def failed_probe(
    seed: int,
    error: str,
    profile: GraspLiftSafetyProfile,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "ready": False,
        "safety_method": profile.method,
        "safety_profile": asdict(profile),
        "task_success": False,
        "right_gripper_closed": False,
        "right_gripper_open": False,
        "left_gripper_closed": False,
        "left_gripper_open": False,
        "required_gripper_state": profile.required_gripper_state,
        "required_gripper_state_ok": False,
        "complete_right_arm_plan_chain": False,
        "complete_left_arm_plan_chain": False,
        "arm_plan_policy": profile.arm_plan_policy,
        "arm_plan_policy_ok": False,
        "selected_arm": None,
        "left_arm_plan_unused": False,
        "object_lifted": False,
        "task_specific_outcome": False,
        "press_contact_verified": False,
        "tool_contact_verified": False,
        "planner_success": False,
        "direct_ik_success": False,
        "gripper_contact_count_after_close": 0,
        "object_lift_mm": None,
        "peak_object_lift_mm": None,
        "error": error,
    }


def verification_level_for(verification_method: str) -> str:
    if verification_method == DIRECT_GRASP_METHOD:
        return "direct_final_grasp_qpos_physical_outcome_v1"
    return "full_planner_expert_task_v1"


def criteria_for(verification_method: str) -> dict[str, Any]:
    common = {
        "environment_reset": True,
        "target_pose_finite_before_and_after": True,
        "task_success": True,
        "required_final_gripper_state": "task_profile",
        "task_specific_outcome": True,
        "min_object_lift_mm": "task_profile_or_cli_override",
        "trajectory_saved": False,
        "images_saved": False,
    }
    if verification_method == DIRECT_GRASP_METHOD:
        return {
            **common,
            "final_grasp_ik": True,
            "final_grasp_qpos_self_collision_free": True,
            "final_grasp_qpos_has_no_blocking_collision": True,
            "allowed_final_grasp_contacts": (
                "target object or support table with gripper links only"
            ),
            "only_robot_qpos_initialized": True,
            "target_object_pose_modified": False,
            "physical_gripper_close": True,
            "physical_grasp_evidence": (
                "gripper-object contact after close or verified target-object lift"
            ),
            "post_grasp_planner_success": True,
            "full_home_to_grasp_approach_required": False,
        }
    return {
        **common,
        "full_planner_expert_execution": True,
        "planner_success": True,
        "task_profile_arm_plan_chain": True,
        "task_profile_gripper_state": True,
    }


def selection_criteria(
    profile: GraspLiftSafetyProfile,
    min_object_lift_mm: float,
    verification_method: str,
) -> str:
    if verification_method == DIRECT_GRASP_METHOD:
        common = (
            "reset && collision-free final-grasp IK && initialize only right-arm qpos "
            "&& target object remains undisturbed before physical close "
            "&& physical grasp evidence && post-grasp planner success "
            "&& task_success && finite target poses"
        )
    else:
        common = (
            "reset && full planner expert execution "
            "&& task-profile arm plan chains && planner_success "
            "&& task_success && finite target poses"
        )
    if profile.verification_mode == "place_by_selected_arm":
        return (
            f"{common} && select arm from object side && other arm unused "
            f"&& both grippers open && target on mat within "
            f"{profile.max_target_container_xy_offset_mm:g} mm XY offset"
        )
    if profile.verification_mode == "dual_handover":
        object_name = (
            "horizontal block"
            if profile.actor_attr == "block"
            else "microphone"
        )
        return (
            f"{common} && complete plans for both arms "
            f"&& {object_name} lift >= {min_object_lift_mm:g} mm "
            "&& observed vertical presentation before receiver grasp "
            "&& giver open && receiver closed && handover task success"
        )
    if profile.verification_mode == "dual_lift":
        return (
            f"{common} && complete plans for both arms && both grippers closed "
            f"&& pot lift >= {min_object_lift_mm:g} mm && dual-handle task success"
        )
    if profile.verification_mode == "handover_place":
        if profile.actor_attr == "cube":
            return (
                f"{common} && complete plans for both arms "
                "&& dynamically assign giver from cube side and receiver from target side "
                "&& observed middle-pad set-down, giver release, and separate receiver pickup "
                "&& both grippers open "
                f"&& cube peak lift >= {min_object_lift_mm:g} mm "
                f"&& cube on target pad within {profile.max_target_container_xy_offset_mm:g} mm XY offset"
            )
        return (
            f"{common} && complete plans for both arms "
            "&& observed left-to-right block handover && both grippers open "
            f"&& block peak lift >= {min_object_lift_mm:g} mm "
            f"&& block on target pad within {profile.max_target_container_xy_offset_mm:g} mm XY offset"
        )
    if profile.verification_mode == "press_contact":
        return (
            f"{common} && right gripper closed "
            "&& bell top-center contact triggered task success"
        )
    if profile.verification_mode == "tool_contact":
        return (
            f"{common} && right gripper closed "
            f"&& tool grasp-lift >= {min_object_lift_mm:g} mm "
            "&& retained tool grasp && tool target contact"
        )
    if profile.verification_mode == "grasp_lift":
        return (
            f"{common} && right gripper closed "
            f"&& target object lift >= {min_object_lift_mm:g} mm"
        )
    if profile.verification_mode == "multi_object_place_in_container":
        return (
            f"{common} && complete right-arm plan chain "
            f"&& all 5 objects individually lifted >= {min_object_lift_mm:g} mm "
            "&& all 5 objects contained in basket && right gripper open"
        )
    target_relation = (
        "target inside container"
        if profile.verification_mode == "place_in_container"
        else "target on designated area"
    )
    return (
        f"{common} && target object peak lift >= {min_object_lift_mm:g} mm "
        f"&& right gripper open && {target_relation} "
        f"within {profile.max_target_container_xy_offset_mm:g} mm XY offset"
    )


def resolve_actor(env: InteractiveRoboTwinEnv, actor_attr: str | None) -> tuple[str, Any]:
    candidates = (actor_attr,) if actor_attr else ("pen", "cube", "bottle", "object", "block", "box")
    for name in candidates:
        if name and hasattr(env.task, name):
            actor = getattr(env.task, name)
            if actor is not None:
                return str(name), actor
    raise ValueError(f"could not find target actor for task {env.task_name}")


def probe_output_dir(task: str, seed: int) -> Path:
    return PROBE_OUTPUT_DIR / f"{task}_seed{seed}_pid{os.getpid()}"


def merge_and_write_manifest(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        merged = dict(summary)
        merged_tasks: dict[str, Any] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("config") == summary.get("config")
                and existing.get("active_arm") == summary.get("active_arm")
            ):
                merged_tasks.update(existing.get("tasks", {}))
        merged_tasks.update(summary.get("tasks", {}))
        merged["tasks"] = merged_tasks
        write_json(path, merged)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return merged


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
