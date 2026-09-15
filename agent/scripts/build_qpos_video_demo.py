from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import imageio.v2 as imageio


ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROBOTWIN_ROOT / "active_spatial_benchmark_xyz"
AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_SRC))

from active_spatial_benchmark import InteractiveRoboTwinEnv  # noqa: E402
from active_spatial_benchmark.env import robotwin_cwd  # noqa: E402
from build_observation_eepose_expert_demos import capture_view, set_center_high_camera  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan once and record an RGB-only replay driven by RoboTwin planner qpos paths."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--active-arm", choices=["left", "right", "both"], default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", choices=["center_high", "head_camera"], default="center_high")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--sample-frequency", type=int, default=8)
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
        help="Keep source capture indices divisible by N; no special final frame is added.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Source-sequence FPS before temporal thinning; output FPS is divided by temporal-stride.",
    )
    args = parser.parse_args()
    if args.active_arm is None:
        args.active_arm = (
            "both"
            if args.task.lower()
            in {
                "place_shoe",
                "handover_mic",
                "handover_horizontal_block",
                "handover_block",
                "handover_cube_to_target",
                "lift_pot",
            }
            else "right"
        )

    if args.sample_frequency < 1:
        raise ValueError("sample-frequency must be positive")
    if args.temporal_stride < 1:
        raise ValueError("temporal-stride must be positive")
    if args.fps < 1:
        raise ValueError("fps must be positive")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    result = build_qpos_video(
        output_dir,
        task=args.task,
        config=args.config,
        seed=args.seed,
        active_arm=args.active_arm,
        view=args.view,
        width=args.width,
        height=args.height,
        sample_frequency=args.sample_frequency,
        temporal_stride=args.temporal_stride,
        source_fps=args.fps,
    )
    print(json.dumps(result, indent=2), flush=True)


def build_qpos_video(
    output_dir: Path,
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str,
    view: str,
    width: int,
    height: int,
    sample_frequency: int,
    temporal_stride: int,
    source_fps: int,
) -> dict[str, Any]:
    print("stage=plan_qpos_paths", flush=True)
    planned_paths, plan_summary = plan_task_paths(
        output_dir,
        task=task,
        config=config,
        seed=seed,
        active_arm=active_arm,
    )

    print("stage=replay_qpos_paths", flush=True)
    video_path = output_dir / f"{view}_qpos_replay.mp4"
    output_fps = max(1, round(source_fps / temporal_stride))
    replay = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=output_dir,
        save_images=False,
    )
    writer = imageio.get_writer(
        video_path,
        fps=output_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    frame_count = 0
    source_frame_count = 0

    try:
        replay.reset(seed=seed)
        if view == "center_high":
            set_center_high_camera(replay)

        replay.task.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": deepcopy(planned_paths["left"]),
                "right_joint_path": deepcopy(planned_paths["right"]),
            }
        )
        replay.task.save_freq = sample_frequency

        def capture_rgb() -> None:
            nonlocal frame_count, source_frame_count
            if source_frame_count % temporal_stride == 0:
                writer.append_data(capture_view(replay, view, width, height))
                frame_count += 1
            source_frame_count += 1

        replay.task._take_picture = capture_rgb
        capture_rgb()
        with robotwin_cwd():
            replay.task.play_once()
        capture_rgb()

        success = bool(replay.task.check_success())
        planner_success = bool(getattr(replay.task, "plan_success", True))
        if not planner_success or not success:
            raise RuntimeError(
                f"qpos replay failed: planner_success={planner_success}, success={success}"
            )
    finally:
        writer.close()
        replay.close()

    result = {
        "task": task,
        "config": config,
        "seed": seed,
        "active_arm": active_arm,
        "success": True,
        "planner_success": True,
        "execution": "robotwin_two_pass_planner_qpos_replay",
        "arm_control": "planner position/velocity arrays applied as joint drive targets",
        "gripper_control": "normalized gripper path applied as gripper joint drive targets",
        "video": video_path.name,
        "view": view,
        "frame_count": frame_count,
        "source_frame_count": source_frame_count,
        "image_size": [width, height],
        "source_fps": source_fps,
        "fps": output_fps,
        "sample_frequency_physics_steps": sample_frequency,
        "temporal_stride": temporal_stride,
        "temporal_sampling_rule": "keep source capture indices divisible by temporal_stride",
        "special_final_frame_retained": False,
        "plan_summary": plan_summary,
        "stored_qpos_trajectory": False,
        "stored_eepose": False,
        "stored_actions": False,
    }
    if task.lower() == "put_everything_in_basket":
        result.update(
            {
                "object_settle_policy": (
                    "after each qpos-driven grasp/lift/transfer/release, "
                    "stabilize the released object at its planned basket target"
                ),
                "pure_physics_object_transport": False,
            }
        )
    elif task.lower() == "handover_horizontal_block":
        result.update(
            {
                "object_type": "horizontal rectangular block",
                "object_full_size_m": [0.03, 0.03, 0.2],
                "initial_object_orientation": "long axis horizontal",
                "required_handover_orientation": "long axis approximately world vertical",
                "role_policy": "giver selected from initial object side; opposite arm receives",
                "handover_sequence": "grasp, lift, verticalize, receiver grasp, giver release",
            }
        )
    elif task.lower() == "handover_mic":
        result.update(
            {
                "object_type": "microphone",
                "initial_object_orientation": "long axis horizontal",
                "required_handover_orientation": "long axis approximately world vertical",
                "role_policy": "giver selected from initial object side; opposite arm receives",
                "handover_sequence": "grasp, lift, verticalize, receiver grasp, giver release",
            }
        )
    elif task.lower() == "handover_cube_to_target":
        result.update(
            {
                "object_type": "cube",
                "role_policy": "giver selected from cube side; receiver selected from target side",
                "handover_sequence": "giver grasp, lift, middle-pad set-down and release, receiver grasp and lift, receiver place",
                "completion": "receiver places cube on the opposite green target pad and both grippers release",
            }
        )
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def plan_task_paths(
    output_dir: Path,
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    env = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=output_dir,
        save_images=False,
    )
    try:
        env.reset(seed=seed)
        env.task.need_plan = True
        with robotwin_cwd():
            env.task.play_once()

        planner_success = bool(getattr(env.task, "plan_success", True))
        success = bool(env.task.check_success())
        if not planner_success or not success:
            containment = None
            if hasattr(env.task, "target_objects") and hasattr(
                env.task,
                "_record_containment_status",
            ):
                containment = {
                    record["actor"].get_name(): env.task._record_containment_status(
                        record
                    )
                    for record in env.task.target_objects
                }
            raise RuntimeError(
                "planning pass failed: "
                f"planner_success={planner_success}, success={success}, "
                f"expert_stage={getattr(env.task, 'expert_stage', 'unknown')}, "
                f"containment={containment}"
            )

        left_paths = deepcopy(env.task.left_joint_path)
        right_paths = deepcopy(env.task.right_joint_path)
        summary = {
            "left_segment_count": len(left_paths),
            "right_segment_count": len(right_paths),
            "left_waypoint_counts": waypoint_counts(left_paths),
            "right_waypoint_counts": waypoint_counts(right_paths),
        }
        return {"left": left_paths, "right": right_paths}, summary
    finally:
        env.close()


def waypoint_counts(paths: list[dict[str, Any]]) -> list[int]:
    return [int(len(path.get("position", []))) for path in paths]


if __name__ == "__main__":
    main()
