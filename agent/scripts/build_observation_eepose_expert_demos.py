from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
import transforms3d as t3d


ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROBOTWIN_ROOT / "active_spatial_benchmark_xyz"
AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(AGENT_SRC))

from active_spatial_benchmark import InteractiveRoboTwinEnv  # noqa: E402
from active_spatial_benchmark.camera_control import look_at_pose  # noqa: E402
from active_spatial_benchmark.env import robotwin_cwd  # noqa: E402
from agent.robotwin.geometry import gripper_finger_link_midpoint  # noqa: E402
from agent.robotwin.visuals import resize_to_fit  # noqa: E402
from scripts.test_action_suite import (  # noqa: E402
    VisualRecorder,
    clone_camera_pose,
    gripper_pose,
    gripper_value,
    reset_camera,
    write_h264_video,
)


DEFAULT_VIEWS = ("center_high",)
DEFAULT_ACTOR_ATTRS = ("pen", "cube", "bottle", "object", "block", "box")
SPARSE_KEYFRAME_EXECUTIONS = {
    "keyframe-eepose-grasp",
    "validated-keyframe-eepose-grasp",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record clean image + observed eepose expert trajectories from any RoboTwin play_once task."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-demos", type=int, help="Number of successful demos to retain from the seed candidates.")
    parser.add_argument(
        "--attempts-per-seed",
        type=int,
        default=1,
        help="Reinitialize and retry the same seed after a failed expert execution.",
    )
    parser.add_argument("--active-arm", choices=["left", "right"], required=True)
    parser.add_argument(
        "--expert-execution",
        choices=[
            "task-play-once",
            "continuous-eepose-grasp",
            "keyframe-eepose-grasp",
            "validated-keyframe-eepose-grasp",
            "validated-temporal-eepose-grasp",
        ],
        default="task-play-once",
        help=(
            "Internal expert execution strategy. Validated keyframes check the complete "
            "pre-grasp, insertion, and lift chain before execution. Stored trajectories remain observation-only."
        ),
    )
    parser.add_argument("--actor-attr", help="Actor attribute used by continuous-eepose-grasp; auto-detected by default.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-frequency", type=int, default=15)
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
        help="Keep source observation indices divisible by N; no special final observation is added.",
    )
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument(
        "--frame-layout",
        choices=["nested", "flat"],
        default="nested",
        help="Store each sample in its own directory or place all images directly under frames/.",
    )
    parser.add_argument("--visual-width", type=int, default=960)
    parser.add_argument("--visual-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--safe-seed-candidates", nargs="*", type=int, default=list(range(10, 30)))
    parser.add_argument(
        "--skip-safe-seed",
        action="store_true",
        help="Do not probe a separate evaluation seed. Useful for supplemental single-seed trajectories.",
    )
    args = parser.parse_args()

    if args.sample_frequency < 1:
        raise ValueError("sample-frequency must be at least 1")
    if args.temporal_stride < 1:
        raise ValueError("temporal-stride must be at least 1")
    if args.attempts_per_seed < 1:
        raise ValueError("attempts-per-seed must be at least 1")
    output_fps = (
        args.fps
        if args.expert_execution in SPARSE_KEYFRAME_EXECUTIONS
        else max(1, round(args.fps / args.temporal_stride))
    )

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    required_demos = args.num_demos if args.num_demos is not None else len(args.seeds)
    if required_demos < 1:
        raise ValueError("num-demos must be at least 1")
    demos = []
    for seed in args.seeds:
        if len(demos) >= required_demos:
            break
        demo_dir = root / f"seed_{seed}"
        demo_path = demo_dir / "expert_demo.json"
        demo = json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.exists() else None
        if demo and demo.get("success") and demo.get("planner_success"):
            print(f"reuse seed={seed} success=True samples={demo.get('sample_count')}", flush=True)
        else:
            for attempt in range(1, args.attempts_per_seed + 1):
                if demo_dir.exists():
                    shutil.rmtree(demo_dir)
                print(f"collect seed={seed} attempt={attempt}/{args.attempts_per_seed}", flush=True)
                demo = record_demo(
                    demo_dir,
                    task=args.task,
                    config=args.config,
                    seed=seed,
                    active_arm=args.active_arm,
                    sample_frequency=args.sample_frequency,
                    temporal_stride=args.temporal_stride,
                    views=tuple(args.views),
                    width=args.visual_width,
                    height=args.visual_height,
                    fps=output_fps,
                    expert_execution=args.expert_execution,
                    actor_attr=args.actor_attr,
                    frame_layout=args.frame_layout,
                )
                if demo["success"] and demo["planner_success"]:
                    break
                print(
                    f"retry seed={seed} attempt={attempt} success={demo['success']} "
                    f"planner={demo['planner_success']} samples={demo['sample_count']}",
                    flush=True,
                )
        if not demo["success"] or not demo["planner_success"]:
            print(
                f"discard seed={seed} success={demo['success']} planner={demo['planner_success']} "
                f"samples={demo['sample_count']}",
                flush=True,
            )
            shutil.rmtree(demo_dir)
            continue
        demos.append(demo)
        print(
            f"accepted seed={seed} success=True planner=True samples={demo['sample_count']} "
            f"count={len(demos)}/{required_demos}",
            flush=True,
        )

    if len(demos) < required_demos:
        raise RuntimeError(
            f"only collected {len(demos)} successful demos from {len(args.seeds)} candidates; "
            f"need {required_demos}"
        )

    selected_seeds = [int(item["seed"]) for item in demos]

    if args.skip_safe_seed:
        safe_seed = {"seed": None, "skipped": True, "probes": []}
    else:
        safe_seed = choose_safe_seed(
            task=args.task,
            config=args.config,
            active_arm=args.active_arm,
            candidates=[seed for seed in args.safe_seed_candidates if seed not in set(selected_seeds)],
            expert_execution=args.expert_execution,
            actor_attr=args.actor_attr,
        )
        if safe_seed["seed"] is None:
            raise RuntimeError(f"no safe evaluation seed found: {safe_seed}")

    manifest = {
        "task": args.task,
        "config": args.config,
        "active_arm": args.active_arm,
        "seeds": selected_seeds,
        "candidate_seeds": args.seeds,
        "type": "image_observed_eepose_expert",
        "observation_only": True,
        "contains_actions": False,
        "contains_target_eepose": False,
        "contains_object_state": False,
        "image_style": {"text_label": False, "eepose_overlay": False},
        "views": args.views,
        "frame_layout": args.frame_layout,
        "video_fps": output_fps,
        "demos": [f"seed_{item['seed']}/expert_demo.json" for item in demos],
        "policy": "expert_task_policy.md",
        "safe_eval_seed": safe_seed,
    }
    if args.expert_execution in SPARSE_KEYFRAME_EXECUTIONS:
        manifest.update({"sampling_mode": "sparse_executed_frames", "frames_per_demo": 5})
    else:
        manifest.update(
            {
                "sample_frequency": args.sample_frequency,
                "temporal_stride": args.temporal_stride,
                "effective_sample_frequency": args.sample_frequency * args.temporal_stride,
                "temporal_sampling_rule": "keep source observation indices divisible by temporal_stride",
                "special_frames_retained": False,
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "safe_eval_seed.json").write_text(json.dumps(safe_seed, indent=2), encoding="utf-8")
    write_policy(
        root,
        args.task,
        selected_seeds,
        safe_seed["seed"],
        sparse_keyframes=args.expert_execution in SPARSE_KEYFRAME_EXECUTIONS,
    )

    print(f"expert_demo_dir: {root}")
    for demo in demos:
        print(
            f"seed={demo['seed']} success={demo['success']} planner={demo['planner_success']} "
            f"samples={demo['sample_count']}"
        )
    print(f"safe_eval_seed: {safe_seed['seed']}")


def record_demo(
    output_dir: Path,
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str,
    sample_frequency: int,
    temporal_stride: int,
    views: tuple[str, ...],
    width: int,
    height: int,
    fps: int,
    expert_execution: str,
    actor_attr: str | None,
    frame_layout: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=output_dir,
        save_images=False,
    )
    rows: list[dict[str, Any]] = []
    frames_by_view: dict[str, list[np.ndarray]] = {view: [] for view in views}

    try:
        env.reset(seed=seed)
        if "center_high" in views:
            set_center_high_camera(env)
        capture_sample(
            output_dir,
            env,
            rows,
            frames_by_view,
            views=views,
            width=width,
            height=height,
            frame_layout=frame_layout,
        )
        if expert_execution in SPARSE_KEYFRAME_EXECUTIONS:
            env.task.save_freq = None

            def keyframe_callback() -> None:
                capture_sample(
                    output_dir,
                    env,
                    rows,
                    frames_by_view,
                    views=views,
                    width=width,
                    height=height,
                    frame_layout=frame_layout,
                )

            with robotwin_cwd():
                run_expert(
                    env,
                    expert_execution=expert_execution,
                    actor_attr=actor_attr,
                    waypoint_callback=keyframe_callback,
                )
            sampling = {
                "mode": "sparse_executed_frames",
                "frame_count": len(rows),
                "video_fps": fps,
            }
        else:
            env.task.save_freq = sample_frequency
            source_observation_index = 0

            def observation_sample_callback() -> None:
                nonlocal source_observation_index
                source_observation_index += 1
                if source_observation_index % temporal_stride == 0:
                    capture_sample(
                        output_dir,
                        env,
                        rows,
                        frames_by_view,
                        views=views,
                        width=width,
                        height=height,
                        frame_layout=frame_layout,
                    )

            env.task._take_picture = observation_sample_callback
            with robotwin_cwd():
                run_expert(env, expert_execution=expert_execution, actor_attr=actor_attr)
            sampling = {
                "source_sample_frequency": sample_frequency,
                "temporal_stride": temporal_stride,
                "effective_sample_frequency": sample_frequency * temporal_stride,
                "source_observation_count": source_observation_index + 1,
                "retained_source_indices": list(range(0, source_observation_index + 1, temporal_stride)),
                "special_frames_retained": False,
                "video_fps": fps,
            }

        success = bool(env.task.check_success())
        planner_success = bool(getattr(env.task, "plan_success", True))
        videos = {}
        for view, frames in frames_by_view.items():
            path = output_dir / f"{view}_replay.mp4"
            write_h264_video(path, frames, fps=fps)
            videos[view] = path.name

        demo = {
            "id": f"seed_{seed}",
            "task": task,
            "config": config,
            "seed": seed,
            "active_arm": env.active_arm,
            "type": "image_observed_eepose_expert",
            "observation_only": True,
            "contains_actions": False,
            "contains_target_eepose": False,
            "contains_object_state": False,
            "image_style": {"text_label": False, "eepose_overlay": False},
            "sampling": sampling,
            "sample_count": len(rows),
            "trajectory": "eepose_trajectory.jsonl",
            "videos": videos,
            "success": success,
            "planner_success": planner_success,
            "last_sampled_state": rows[-1]["state"],
        }
        (output_dir / "expert_demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
        with (output_dir / "eepose_trajectory.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        return demo
    finally:
        env.close()


def capture_sample(
    output_dir: Path,
    env: InteractiveRoboTwinEnv,
    rows: list[dict[str, Any]],
    frames_by_view: dict[str, list[np.ndarray]],
    *,
    views: tuple[str, ...],
    width: int,
    height: int,
    frame_layout: str,
) -> None:
    step = len(rows)
    frame_dir = output_dir / "frames"
    if frame_layout == "nested":
        frame_dir = frame_dir / f"{step:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for view in views:
        frame = capture_view(env, view, width, height)
        if frame_layout == "flat":
            filename = f"{step:05d}.png" if len(views) == 1 else f"{step:05d}_{view}.png"
            rel_path = Path("frames") / filename
        else:
            rel_path = Path("frames") / frame_dir.name / f"{view}.png"
        imageio.imwrite(output_dir / rel_path, frame)
        frames_by_view[view].append(frame)
        images[view] = str(rel_path)
    rows.append({"step": step, "images": images, "state": observed_eepose(env)})


def capture_view(env: InteractiveRoboTwinEnv, view: str, width: int, height: int) -> np.ndarray:
    saved_pose = clone_camera_pose(env)
    try:
        recorder_view = "active" if view in {"head_camera", "center_high"} else view
        recorder = VisualRecorder(
            Path("."),
            enabled=False,
            view_mode=recorder_view,
            overlay_eepose=False,
            visual_width=width,
            visual_height=height,
        )
        frame = recorder.capture(env)
        return resize_to_fit(frame, width, height) if view == "center_high" else frame
    finally:
        reset_camera(env, saved_pose)


def set_center_high_camera(env: InteractiveRoboTwinEnv) -> None:
    midpoint = gripper_finger_link_midpoint(env)
    gripper = np.asarray(midpoint if midpoint is not None else gripper_pose(env)[:3], dtype=np.float64)
    side = 0.0 if env.active_arm == "both" else (1.0 if env.active_arm == "right" else -1.0)
    position = np.array([0.0, -0.58, 1.48], dtype=np.float64)
    target = np.array([0.17 * side, gripper[1] - 0.02, max(0.86, gripper[2] - 0.05)], dtype=np.float64)
    pose = look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    env.camera.get_camera().entity.set_pose(pose)
    env.task._update_render()


def observed_eepose(env: InteractiveRoboTwinEnv) -> dict[str, Any]:
    pose = np.asarray(gripper_pose(env), dtype=np.float64)
    aperture = float(gripper_value(env))
    return {
        "eepose_xyz": rounded(pose[:3], 6),
        "eepose_quat_wxyz": rounded(pose[3:], 6),
        "eepose_rpy_deg": rounded(np.rad2deg(t3d.euler.quat2euler(pose[3:])), 3),
        "gripper_value": round(aperture, 6),
        "eepose_8d_xyz_quat_gripper": rounded(np.concatenate([pose, [aperture]]), 6),
    }


def choose_safe_seed(
    *,
    task: str,
    config: str,
    active_arm: str,
    candidates: list[int],
    expert_execution: str,
    actor_attr: str | None,
) -> dict[str, Any]:
    probes = []
    for seed in candidates:
        probe = probe_seed(
            task=task,
            config=config,
            active_arm=active_arm,
            seed=seed,
            expert_execution=expert_execution,
            actor_attr=actor_attr,
        )
        probes.append(probe)
        if probe.get("success") and probe.get("planner_success"):
            return {"seed": seed, "success": True, "planner_success": True, "probes": probes}
    return {"seed": None, "success": False, "planner_success": False, "probes": probes}


def probe_seed(
    *,
    task: str,
    config: str,
    active_arm: str,
    seed: int,
    expert_execution: str,
    actor_attr: str | None,
) -> dict[str, Any]:
    env = InteractiveRoboTwinEnv(
        task_name=task,
        config_name=config,
        active_arm=active_arm,
        max_steps=1000,
        output_dir=Path("/tmp/agent_observation_seed_probe"),
        save_images=False,
    )
    try:
        env.reset(seed=seed)
        with robotwin_cwd():
            run_expert(env, expert_execution=expert_execution, actor_attr=actor_attr)
        return {
            "seed": seed,
            "success": bool(env.task.check_success()),
            "planner_success": bool(getattr(env.task, "plan_success", True)),
            "active_arm": env.active_arm,
        }
    except Exception as exc:
        return {"seed": seed, "success": False, "planner_success": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def run_expert(
    env: InteractiveRoboTwinEnv,
    *,
    expert_execution: str,
    actor_attr: str | None,
    waypoint_callback=None,
) -> None:
    if expert_execution == "task-play-once":
        env.task.play_once()
        return

    actor = resolve_actor(env, actor_attr)
    if expert_execution in {"validated-keyframe-eepose-grasp", "validated-temporal-eepose-grasp"}:
        chain = plan_validated_grasp_chain(env, actor)
        if chain is None:
            env.task.plan_success = False
            return
        save_freq = -1 if expert_execution == "validated-temporal-eepose-grasp" else None
        execute_validated_grasp_chain(
            env,
            actor,
            chain,
            waypoint_callback=waypoint_callback,
            save_freq=save_freq,
        )
        return

    _, planner_actions = env.task.grasp_actor(actor, arm_tag=env.task.arm_tag, pre_grasp_dis=0.09)
    move_targets = [
        np.asarray(action.target_pose, dtype=np.float64)
        for action in planner_actions
        if getattr(action, "action", "") == "move"
    ]
    if len(move_targets) < 2:
        env.task.plan_success = False
        return

    sparse = expert_execution == "keyframe-eepose-grasp"
    save_freq = None if sparse else -1
    for target_pose in move_targets[:2]:
        if not env.task.move(env.task.move_to_pose(env.active_arm, target_pose.tolist()), save_freq=save_freq):
            return
        if waypoint_callback is not None:
            waypoint_callback()
    if not env.task.move(env.task.close_gripper(env.active_arm), save_freq=save_freq):
        return
    if waypoint_callback is not None:
        waypoint_callback()

    lift_target = np.asarray(gripper_pose(env), dtype=np.float64)
    lift_target[2] += 0.12
    if env.task.move(env.task.move_to_pose(env.active_arm, lift_target.tolist()), save_freq=save_freq):
        if waypoint_callback is not None:
            waypoint_callback()


def plan_validated_grasp_chain(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    *,
    pre_grasp_distance: float = 0.09,
    lift_height: float = 0.12,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Find a grasp with a reachable pre-grasp, insertion, and lift chain."""
    if env.active_arm != "right":
        raise ValueError("validated-keyframe-eepose-grasp currently supports the right arm only")

    contact_ids = [index for index, _ in actor.iter_contact_points()]
    preferred_order = [5, 1, 4, 0, 2, 6, 7, 3]
    ordered_ids = [index for index in preferred_order if index in contact_ids]
    ordered_ids.extend(index for index in contact_ids if index not in ordered_ids)

    initial_qpos = np.asarray(env.task.robot.right_entity.get_qpos()).copy()
    for contact_id in ordered_ids:
        try:
            pre_pose, grasp_pose = env.task.choose_grasp_pose(
                actor,
                arm_tag=env.task.arm_tag,
                pre_dis=pre_grasp_distance,
                target_dis=0.0,
                contact_point_id=contact_id,
            )
            if not valid_pose(pre_pose) or not valid_pose(grasp_pose):
                continue

            pre_plan = env.task.robot.right_plan_path(pre_pose, last_qpos=initial_qpos)
            if pre_plan.get("status") != "Success":
                continue
            pre_qpos = qpos_after_right_plan(env, initial_qpos, pre_plan)

            grasp_plan = env.task.robot.right_plan_path(
                grasp_pose,
                constraint_pose=[1, 1, 1, 0, 0, 0],
                last_qpos=pre_qpos,
            )
            if grasp_plan.get("status") != "Success":
                continue
            grasp_qpos = qpos_after_right_plan(env, pre_qpos, grasp_plan)

            lift_pose = np.asarray(grasp_pose, dtype=np.float64).copy()
            lift_pose[2] += lift_height
            lift_plan = env.task.robot.right_plan_path(lift_pose.tolist(), last_qpos=grasp_qpos)
            if lift_plan.get("status") != "Success":
                continue
            return pre_plan, grasp_plan, lift_plan
        except Exception as exc:
            print(
                f"validated candidate contact={contact_id} rejected: {type(exc).__name__}: {exc}",
                flush=True,
            )
    return None


def execute_validated_grasp_chain(
    env: InteractiveRoboTwinEnv,
    actor: Any,
    chain: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    *,
    waypoint_callback=None,
    save_freq=None,
) -> None:
    pre_plan, grasp_plan, lift_plan = chain
    initial_actor_pose = actor.get_pose()
    for plan in (pre_plan, grasp_plan):
        execute_right_arm_plan(env, plan, save_freq=save_freq)
        if waypoint_callback is not None:
            waypoint_callback()
        if not actor_undisturbed_for_grasp(actor, initial_actor_pose):
            env.task.plan_success = False
            return

    gripper_plan = env.task.set_gripper(right_pos=0.0, set_tag="right")
    env.task.take_dense_action(
        {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": None,
            "right_gripper": gripper_plan,
        },
        save_freq=save_freq,
    )
    if waypoint_callback is not None:
        waypoint_callback()
    if not actor_undisturbed_for_grasp(actor, initial_actor_pose):
        env.task.plan_success = False
        return

    execute_right_arm_plan(env, lift_plan, save_freq=save_freq)
    if waypoint_callback is not None:
        waypoint_callback()


def execute_right_arm_plan(env: InteractiveRoboTwinEnv, plan: dict[str, Any], *, save_freq=None) -> None:
    env.task.take_dense_action(
        {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": plan,
            "right_gripper": None,
        },
        save_freq=save_freq,
    )


def qpos_after_right_plan(
    env: InteractiveRoboTwinEnv,
    start_qpos: np.ndarray,
    plan: dict[str, Any],
) -> np.ndarray:
    result = np.asarray(start_qpos).copy()
    final_arm_qpos = np.asarray(plan["position"])[-1]
    all_joint_names = [joint.get_name() for joint in env.task.robot.right_entity.get_active_joints()]
    for arm_index, joint_name in enumerate(env.task.robot.right_arm_joints_name):
        result[all_joint_names.index(joint_name)] = final_arm_qpos[arm_index]
    return result


def valid_pose(pose: Any) -> bool:
    if pose is None:
        return False
    values = np.asarray(pose, dtype=np.float64)
    return values.shape == (7,) and bool(np.all(np.isfinite(values)))


def actor_undisturbed_for_grasp(
    actor: Any,
    initial_pose: Any,
    *,
    max_xy_displacement: float = 0.025,
    max_tilt_deg: float = 15.0,
) -> bool:
    current_pose = actor.get_pose()
    initial_xy = np.asarray(initial_pose.p, dtype=np.float64)[:2]
    current_xy = np.asarray(current_pose.p, dtype=np.float64)[:2]
    if np.linalg.norm(current_xy - initial_xy) > max_xy_displacement:
        return False

    initial_rotation = t3d.quaternions.quat2mat(np.asarray(initial_pose.q, dtype=np.float64))
    current_rotation = t3d.quaternions.quat2mat(np.asarray(current_pose.q, dtype=np.float64))
    initial_long_axis = initial_rotation[:, 1]
    current_long_axis = current_rotation[:, 1]
    alignment = abs(float(np.dot(initial_long_axis, current_long_axis)))
    return alignment >= float(np.cos(np.deg2rad(max_tilt_deg)))


def resolve_actor(env: InteractiveRoboTwinEnv, actor_attr: str | None) -> Any:
    candidates = (actor_attr,) if actor_attr else DEFAULT_ACTOR_ATTRS
    for name in candidates:
        if name and hasattr(env.task, name):
            return getattr(env.task, name)
    raise AttributeError(f"could not resolve grasp actor from attributes: {', '.join(candidates)}")


def write_policy(root: Path, task: str, seeds: list[int], safe_seed: int | None, *, sparse_keyframes: bool) -> None:
    lines = [
        f"EXPERT OBSERVATION DATA: {task}",
        f"- Demonstration seeds: {', '.join(str(seed) for seed in seeds)}",
        (f"- Safe evaluation seed: {safe_seed}" if safe_seed is not None else "- Separate safe evaluation seed: not probed"),
        "- Each trajectory contains only clean RGB observations and the observed robot eepose/gripper state at the same sample.",
        (
            "- Each demo contains five sparse executed observations without semantic phase labels."
            if sparse_keyframes
            else "- Temporal sampling is uniform: only source observation indices divisible by the configured stride are retained; no semantic or final frame is added."
        ),
        "- No action names, action primitives, target eeposes, object coordinates, simulator functional points, or contact geometry are stored.",
        "- Infer task strategy, grasp choice, motion direction, and tool use from the image/eepose sequence itself.",
        (
            "- The safe evaluation seed was checked without saving a trajectory and is not a demonstration seed."
            if safe_seed is not None
            else "- This supplemental trajectory does not include a separate evaluation-seed probe."
        ),
    ]
    (root / "expert_task_policy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rounded(values: Any, digits: int) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values)]


if __name__ == "__main__":
    main()
