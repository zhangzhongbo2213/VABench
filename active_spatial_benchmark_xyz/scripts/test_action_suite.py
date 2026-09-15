from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import transforms3d as t3d
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_spatial_benchmark import InteractiveRoboTwinEnv, describe_action, list_available_actions, parse_action
from active_spatial_benchmark.camera_control import look_at_pose
from active_spatial_benchmark.env import robotwin_cwd


CORRESPONDENCE_CAMERA_SETUPS = [
    "identity",
    "camera.look_at_workspace",
    "camera.yaw_right.medium",
    "camera.yaw_left.medium",
    "camera.pitch_up.medium",
    "camera.pitch_down.medium",
    "camera.look_at_gripper",
]

CORRESPONDENCE_GRIPPER_ACTIONS = [
    "gripper.image_left.small",
    "gripper.image_right.small",
    "gripper.image_up.small",
    "gripper.image_down.small",
    "gripper.depth_forward.small",
    "gripper.depth_backward.small",
    "gripper.lift_up.small",
    "gripper.lift_down.small",
]

BOUNDARY_FEEDBACK_ACTIONS = [
    "gripper.depth_forward.large",
    "gripper.lift_down.medium",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full interactive action validation suite.")
    parser.add_argument("--task", default="place_empty_cup")
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--active-arm", choices=["left", "right", "both"], default=None)
    parser.add_argument("--output-dir", default="runs/action_suite")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--reference-lift",
        type=float,
        default=0.12,
        help="Lift the neutral gripper test pose before validating directional actions.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Disable before/after PNG and H.264 MP4 generation.",
    )
    parser.add_argument(
        "--visual-view",
        choices=["active", "topdown", "side", "gripper_follow"],
        default="gripper_follow",
        help="Camera used only for saved PNG/MP4 artifacts; action tests still use the active camera.",
    )
    parser.add_argument(
        "--no-eepose-overlay",
        action="store_true",
        help="Disable projected finger-center/GC, XYZ axes, orientation axes, and eepose text overlays.",
    )
    parser.add_argument("--visual-width", type=int, default=1280, help="Width for saved observer-view PNG/MP4 frames.")
    parser.add_argument("--visual-height", type=int, default=960, help="Height for saved observer-view PNG/MP4 frames.")
    args = parser.parse_args()
    if args.visual_width <= 0 or args.visual_height <= 0:
        raise ValueError("--visual-width and --visual-height must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = InteractiveRoboTwinEnv(
        task_name=args.task,
        config_name=args.config,
        active_arm=args.active_arm,
        max_steps=args.max_steps,
        output_dir=output_dir,
        save_images=False,
    )

    try:
        env.reset(seed=args.seed)
        with robotwin_cwd():
            raw_reference_gripper_pose = np.asarray(env.task.get_arm_pose(env.active_arm), dtype=np.float64)
        reference_gripper_pose = raw_reference_gripper_pose.copy()
        reference_gripper_pose[2] += args.reference_lift
        reset_gripper(env, reference_gripper_pose)
        reference_camera_pose = clone_camera_pose(env)
        visuals = VisualRecorder(
            output_dir,
            enabled=not args.no_visuals,
            view_mode=args.visual_view,
            overlay_eepose=not args.no_eepose_overlay,
            visual_width=args.visual_width,
            visual_height=args.visual_height,
        )

        schema_rows = build_schema_rows()
        runtime_rows = run_runtime_action_table(env, reference_camera_pose, reference_gripper_pose, visuals)
        correspondence_rows = run_correspondence_table(env, reference_camera_pose, reference_gripper_pose, visuals)
        feedback_rows = run_boundary_feedback_table(
            env,
            reference_camera_pose,
            raw_reference_gripper_pose,
            visuals,
        )
        visuals.write_videos()

        write_markdown(output_dir / "action_schema_table.md", schema_rows)
        write_markdown(output_dir / "runtime_action_table.md", runtime_rows)
        write_markdown(output_dir / "correspondence_table.md", correspondence_rows)
        write_markdown(output_dir / "boundary_feedback_table.md", feedback_rows)

        raw = {
            "task": args.task,
            "config": args.config,
            "seed": args.seed,
            "active_arm": env.active_arm,
            "reference_lift": args.reference_lift,
            "raw_reference_gripper_pose": raw_reference_gripper_pose.tolist(),
            "reference_gripper_pose": reference_gripper_pose.tolist(),
            "schema_rows": schema_rows,
            "runtime_rows": runtime_rows,
            "correspondence_rows": correspondence_rows,
            "boundary_feedback_rows": feedback_rows,
            "visual_view": args.visual_view,
            "visual_artifacts": visuals.artifacts(),
        }
        (output_dir / "action_suite_results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

        print_summary(schema_rows, runtime_rows, correspondence_rows, feedback_rows, output_dir)
    finally:
        env.close()


def build_schema_rows() -> list[dict[str, object]]:
    rows = []
    for action_name in list_available_actions():
        spec = parse_action(action_name)
        rows.append({
            "action": action_name,
            "target": spec.target,
            "type": spec.type,
            "direction_or_mode": spec.direction or spec.mode or "-",
            "scale": spec.scale or "-",
            "expected_effect": expected_effect(spec),
            "schema_pass": True,
        })
    return rows


def run_runtime_action_table(env, reference_camera_pose, reference_gripper_pose, visuals) -> list[dict[str, object]]:
    rows = []
    for idx, action_name in enumerate(list_available_actions(), start=1):
        reset_camera(env, reference_camera_pose)
        reset_gripper(env, reference_gripper_pose)
        spec = parse_action(action_name)

        camera_before = camera_matrix(env)
        gripper_before = gripper_pose(env)
        gripper_open_before = gripper_is_open(env)
        frame_before = visuals.capture(env)

        result = env.step(action_name)
        env.done = False
        env.success = False

        camera_after = camera_matrix(env)
        gripper_after = gripper_pose(env)
        gripper_open_after = gripper_is_open(env)
        frame_after = visuals.capture(env)
        before_path, after_path = visuals.save_pair("runtime", idx, action_name, frame_before, frame_after)

        camera_delta = float(np.linalg.norm(camera_after[:3, 3] - camera_before[:3, 3]))
        camera_angle = rotation_angle_deg(camera_before[:3, :3], camera_after[:3, :3])
        gripper_delta_vec = gripper_after[:3] - gripper_before[:3]
        gripper_delta = float(np.linalg.norm(gripper_delta_vec))
        gripper_angle = quat_angle_deg(gripper_before[3:], gripper_after[3:])

        rows.append({
            "#": idx,
            "action": action_name,
            "valid": result["info"]["action_valid"],
            "planner": result["info"]["planner_success"],
            "camera_delta_m": round(camera_delta, 5),
            "camera_angle_deg": round(camera_angle, 3),
            "gripper_delta_m": round(gripper_delta, 5),
            "gripper_angle_deg": round(gripper_angle, 3),
            "gripper_open_before": gripper_open_before,
            "gripper_open_after": gripper_open_after,
            "pass": runtime_pass(spec, result["info"], camera_delta, camera_angle, gripper_delta, gripper_angle),
            "note": result["info"].get("last_error") or runtime_note(spec),
            "before_image": before_path,
            "after_image": after_path,
        })
    return rows


def run_correspondence_table(env, reference_camera_pose, reference_gripper_pose, visuals) -> list[dict[str, object]]:
    rows = []
    idx = 0
    for setup in CORRESPONDENCE_CAMERA_SETUPS:
        reset_camera(env, reference_camera_pose)
        if setup != "identity":
            env.step(setup)
            env.done = False
            env.success = False
        camera_frame = env.camera.frame()

        for action_name in CORRESPONDENCE_GRIPPER_ACTIONS:
            idx += 1
            reset_gripper(env, reference_gripper_pose)
            expected = env.camera.gripper_direction(parse_action(action_name).direction)
            before = gripper_pose(env)
            frame_before = visuals.capture(env)
            result = env.step(action_name)
            env.done = False
            env.success = False
            after = gripper_pose(env)
            frame_after = visuals.capture(env)
            label = f"{setup}__{action_name}"
            before_path, after_path = visuals.save_pair("correspondence", idx, label, frame_before, frame_after)
            measured = after[:3] - before[:3]
            measured_norm = float(np.linalg.norm(measured))
            expected_norm = float(np.linalg.norm(expected))
            cosine = safe_cosine(measured, expected)
            rows.append({
                "camera_setup": setup,
                "action": action_name,
                "valid": result["info"]["action_valid"],
                "planner": result["info"]["planner_success"],
                "expected_vector_xyz": fmt_vec(expected),
                "measured_delta_xyz": fmt_vec(measured),
                "measured_norm_m": round(measured_norm, 5),
                "cosine": round(cosine, 4),
                "pass": bool(result["info"]["planner_success"] and measured_norm > 0.005 and cosine > 0.92),
                "camera_forward_xyz": fmt_vec(camera_frame.forward),
                "camera_left_xyz": fmt_vec(camera_frame.left),
                "camera_up_xyz": fmt_vec(camera_frame.up),
                "before_image": before_path,
                "after_image": after_path,
            })
    return rows


def run_boundary_feedback_table(env, reference_camera_pose, raw_reference_gripper_pose, visuals) -> list[dict[str, object]]:
    rows = []
    for idx, action_name in enumerate(BOUNDARY_FEEDBACK_ACTIONS, start=1):
        reset_camera(env, reference_camera_pose)
        reset_gripper(env, raw_reference_gripper_pose)
        before = gripper_pose(env)
        frame_before = visuals.capture(env)
        result = env.step(action_name)
        env.done = False
        env.success = False
        after = gripper_pose(env)
        frame_after = visuals.capture(env)
        before_path, after_path = visuals.save_pair("boundary_feedback", idx, action_name, frame_before, frame_after)
        measured = after[:3] - before[:3]
        expected_feedback = bool(result["info"]["action_valid"] and not result["info"]["planner_success"])
        rows.append({
            "#": idx,
            "action": action_name,
            "valid": result["info"]["action_valid"],
            "planner": result["info"]["planner_success"],
            "feedback_preserved": expected_feedback,
            "measured_delta_m": round(float(np.linalg.norm(measured)), 5),
            "note": (
                "reachable-state-independent action, but current boundary pose is not reachable"
                if expected_feedback else result["info"].get("last_error")
            ),
            "before_image": before_path,
            "after_image": after_path,
        })
    return rows


def expected_effect(spec) -> str:
    return describe_action(spec.compact())


def runtime_pass(spec, info, camera_delta, camera_angle, gripper_delta, gripper_angle) -> bool:
    if not info["action_valid"] or not info["planner_success"]:
        return False
    if spec.target == "camera":
        return camera_delta > 0.001 or camera_angle > 0.5
    if spec.target == "gripper" and spec.type == "translate":
        return gripper_delta > 0.005
    if spec.target == "gripper" and spec.type == "rotate":
        return gripper_angle > 1.0
    if spec.target == "gripper" and spec.type in {"open", "close"}:
        return True
    return False


def runtime_note(spec) -> str:
    if spec.target == "gripper" and spec.type in {"open", "close"}:
        return "gripper state command; pose change is not required"
    if spec.target == "camera":
        return "camera pose should change; gripper pose should remain stable"
    if spec.target == "gripper":
        return "gripper pose should change; camera pose should remain stable"
    return "-"


def clone_camera_pose(env):
    pose = env.camera.get_camera().entity.get_pose()
    return type(pose)(pose.p.copy(), pose.q.copy())


def reset_camera(env, pose) -> None:
    with robotwin_cwd():
        env.camera.get_camera().entity.set_pose(deepcopy(pose))
        env.task._update_render()


def reset_gripper(env, pose) -> None:
    with robotwin_cwd():
        env.task.move(env.task.move_to_pose(env.active_arm, np.asarray(pose, dtype=np.float64).tolist()), save_freq=None)
        env.task.plan_success = True
        env.task._update_render()


def capture_active_rgb(env) -> np.ndarray:
    with robotwin_cwd():
        obs = env.task.get_obs()
    return obs["observation"][env.active_camera]["rgb"].copy()


def capture_active_rgb_and_camera(env) -> tuple[np.ndarray, object]:
    return capture_active_rgb(env), env.camera.get_camera()


def capture_observer_rgb_and_camera(env, view_mode: str, width: int, height: int) -> tuple[np.ndarray, object]:
    with robotwin_cwd():
        observer_camera = get_or_create_visual_camera(env, width, height)
        observer_camera.entity.set_pose(observer_pose(env, view_mode))
        env.task._update_render()
        observer_camera.take_picture()
        rgba = observer_camera.get_picture("Color")
    return (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8"), observer_camera


def get_or_create_visual_camera(env, width: int, height: int):
    size_key = (int(width), int(height))
    cameras = getattr(env.task, "_active_spatial_visual_cameras", None)
    if cameras is None:
        cameras = {}
        # Preserve a camera created by older code in this process, if present.
        legacy_camera = getattr(env.task, "_active_spatial_visual_camera", None)
        legacy_size = getattr(env.task, "_active_spatial_visual_camera_size", None)
        if legacy_camera is not None and legacy_size is not None:
            cameras[tuple(legacy_size)] = legacy_camera
        env.task._active_spatial_visual_cameras = cameras
    camera = cameras.get(size_key)
    if camera is not None:
        env.task._active_spatial_visual_camera = camera
        env.task._active_spatial_visual_camera_size = size_key
        return camera
    camera = env.task.scene.add_camera(
        name=f"active_spatial_visual_{width}x{height}",
        width=width,
        height=height,
        fovy=np.deg2rad(55),
        near=0.05,
        far=100,
    )
    cameras[size_key] = camera
    env.task._active_spatial_visual_camera = camera
    env.task._active_spatial_visual_camera_size = size_key
    return camera


def observer_pose(env, view_mode: str):
    if view_mode == "topdown":
        position = np.array([0.0, -0.12, 1.72], dtype=np.float64)
        target = np.array([0.0, -0.08, 0.80], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, -1.0, 0.0]))
    if view_mode == "gripper_follow":
        gripper = gripper_pose(env)[:3]
        side = 1.0 if env.active_arm in {"right", "both"} else -1.0
        position = gripper + np.array([0.38 * side, -0.42, 0.32], dtype=np.float64)
        target = gripper + np.array([0.0, 0.0, -0.02], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    if view_mode == "side":
        gripper = gripper_pose(env)[:3]
        side = 1.0 if env.active_arm in {"right", "both"} else -1.0
        position = gripper + np.array([0.58 * side, -0.02, 0.06], dtype=np.float64)
        target = gripper + np.array([0.0, 0.0, -0.02], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    if view_mode == "front_side_45":
        gripper = gripper_pose(env)[:3]
        side = 1.0 if env.active_arm in {"right", "both"} else -1.0
        position = gripper + np.array([0.44 * side, -0.44, 0.18], dtype=np.float64)
        target = gripper + np.array([0.0, 0.0, -0.02], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    if view_mode == "side_top_45":
        gripper = gripper_pose(env)[:3]
        side = 1.0 if env.active_arm in {"right", "both"} else -1.0
        position = gripper + np.array([0.48 * side, -0.02, 0.48], dtype=np.float64)
        target = gripper + np.array([0.0, 0.0, -0.02], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    if view_mode == "oblique_45":
        gripper = gripper_pose(env)[:3]
        side = 1.0 if env.active_arm in {"right", "both"} else -1.0
        position = gripper + np.array([0.42 * side, -0.42, 0.42], dtype=np.float64)
        target = gripper + np.array([0.0, 0.0, -0.02], dtype=np.float64)
        return look_at_pose(position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    raise ValueError(f"Unsupported observer visual view: {view_mode}")


def camera_matrix(env) -> np.ndarray:
    return env.camera.get_camera().entity.get_pose().to_transformation_matrix().copy()


def gripper_pose(env, arm: str | None = None) -> np.ndarray:
    resolved_arm = arm or env.active_arm
    with robotwin_cwd():
        if resolved_arm == "both":
            left = np.asarray(env.task.get_arm_pose("left"), dtype=np.float64)
            right = np.asarray(env.task.get_arm_pose("right"), dtype=np.float64)
            result = right.copy()
            result[:3] = (left[:3] + right[:3]) / 2.0
            return result
        return np.asarray(env.task.get_arm_pose(resolved_arm), dtype=np.float64)


def gripper_is_open(env) -> bool:
    if env.active_arm == "left":
        return bool(env.task.is_left_gripper_open())
    return bool(env.task.is_right_gripper_open())


def rotation_angle_deg(rot_before: np.ndarray, rot_after: np.ndarray) -> float:
    rel = rot_after @ rot_before.T
    cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def quat_angle_deg(q_before: np.ndarray, q_after: np.ndarray) -> float:
    q_before = np.asarray(q_before, dtype=np.float64)
    q_after = np.asarray(q_after, dtype=np.float64)
    dot = abs(float(np.dot(q_before / np.linalg.norm(q_before), q_after / np.linalg.norm(q_after))))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def safe_cosine(measured: np.ndarray, expected: np.ndarray) -> float:
    denom = np.linalg.norm(measured) * np.linalg.norm(expected)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(measured, expected) / denom)


def fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in np.asarray(vec, dtype=np.float64).tolist()) + "]"


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class VisualRecorder:
    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        view_mode: str = "gripper_follow",
        overlay_eepose: bool = True,
        visual_width: int = 1280,
        visual_height: int = 960,
    ):
        self.output_dir = output_dir
        self.enabled = enabled
        self.view_mode = view_mode
        self.overlay_eepose = overlay_eepose
        self.visual_width = visual_width
        self.visual_height = visual_height
        self.frames: dict[str, list[np.ndarray]] = {
            "runtime": [],
            "correspondence": [],
            "boundary_feedback": [],
        }
        self.video_paths: dict[str, str] = {}

    def capture(self, env) -> np.ndarray:
        if self.view_mode == "active":
            frame, camera = capture_active_rgb_and_camera(env)
        else:
            frame, camera = capture_observer_rgb_and_camera(env, self.view_mode, self.visual_width, self.visual_height)
        if self.overlay_eepose:
            return annotate_eepose(frame, camera, env)
        return frame

    def save_pair(
        self,
        group: str,
        idx: int,
        label: str,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
    ) -> tuple[str, str]:
        if not self.enabled:
            return "", ""
        group_dir = self.output_dir / "visuals" / group
        group_dir.mkdir(parents=True, exist_ok=True)
        safe_label = sanitize_filename(label)
        before_labeled = add_label(frame_before, f"{idx:03d} before {label}")
        after_labeled = add_label(frame_after, f"{idx:03d} after  {label}")
        before_path = group_dir / f"{idx:03d}_{safe_label}_before.png"
        after_path = group_dir / f"{idx:03d}_{safe_label}_after.png"
        imageio.imwrite(before_path, before_labeled)
        imageio.imwrite(after_path, after_labeled)
        self.frames[group].extend([before_labeled, after_labeled])
        return str(before_path.relative_to(self.output_dir)), str(after_path.relative_to(self.output_dir))

    def write_videos(self) -> None:
        if not self.enabled:
            return
        for group, frames in self.frames.items():
            if not frames:
                continue
            video_path = self.output_dir / f"{group}_sequence.mp4"
            write_h264_video(video_path, frames)
            self.video_paths[group] = str(video_path.relative_to(self.output_dir))

    def artifacts(self) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "view_mode": self.view_mode,
            "overlay_eepose": self.overlay_eepose,
            "visual_width": self.visual_width,
            "visual_height": self.visual_height,
            "videos": self.video_paths,
            "visual_root": "visuals",
        }


def annotate_eepose(frame: np.ndarray, camera, env) -> np.ndarray:
    control_pose = gripper_pose(env)
    pose = gripper_center_pose(env)
    xyz = pose[:3]
    control_xyz = control_pose[:3]
    quat = pose[3:]
    rot = t3d.quaternions.quat2mat(quat)
    rpy_deg = np.rad2deg(t3d.euler.quat2euler(quat))
    gripper_val = gripper_value(env)

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = load_font(24 if image.width >= 1200 else 20)

    center_uv = project_world_points(camera, xyz.reshape(1, 3))
    if center_uv is not None:
        u, v = center_uv[0]
        radius = 10 if image.width >= 1200 else 8
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 255, 0), outline=(0, 0, 0), width=3)
        draw_outlined_text(draw, (u + radius + 4, v - radius - 12), "GC", font=label_font, fill=(255, 255, 0))

    world_axes = [
        ("X", np.array([1.0, 0.0, 0.0]), (255, 60, 60)),
        ("Y", np.array([0.0, 1.0, 0.0]), (60, 255, 60)),
        ("Z", np.array([0.0, 0.0, 1.0]), (80, 150, 255)),
    ]
    local_axes = [
        ("rx", rot[:, 0], (255, 170, 0)),
        ("ry", rot[:, 1], (0, 220, 220)),
        ("rz", rot[:, 2], (255, 80, 255)),
    ]
    label_offsets = {
        "X": (14, -34),
        "Y": (14, 10),
        "Z": (14, -12),
        "rx": (-58, -28),
        "ry": (-58, 12),
        "rz": (14, 26),
    }

    for label, axis, color in world_axes:
        draw_projected_arrow(
            draw,
            camera,
            xyz,
            xyz + axis * 0.14,
            label,
            color,
            width=6,
            font=label_font,
            label_offset=label_offsets[label],
            image_size=image.size,
        )
    for label, axis, color in local_axes:
        draw_projected_arrow(
            draw,
            camera,
            xyz,
            xyz + axis * 0.10,
            label,
            color,
            width=4,
            font=label_font,
            label_offset=label_offsets[label],
            image_size=image.size,
        )

    text_lines = [
        f"finger center xyz: [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}] m",
        f"control eepose xyz: [{control_xyz[0]:+.3f}, {control_xyz[1]:+.3f}, {control_xyz[2]:+.3f}] m",
        f"rot rpy:    [{rpy_deg[0]:+.1f}, {rpy_deg[1]:+.1f}, {rpy_deg[2]:+.1f}] deg",
        f"quat:       [{quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f}]",
        f"gripper:    {gripper_val:.3f}  (0 closed, 1 open)",
        "world axes: X red, Y green, Z blue",
        "rot axes: rx orange, ry cyan, rz magenta",
    ]
    draw_text_panel(draw, text_lines, image.width, image.height)
    draw_gripper_bar(draw, gripper_val, image.width, image.height)
    draw_axis_legend(draw, image.width, image.height)
    return np.asarray(image)


def gripper_value(env) -> float:
    if env.active_arm == "left":
        return float(env.task.robot.get_left_gripper_val())
    return float(env.task.robot.get_right_gripper_val())


def gripper_center_pose(env, arm: str | None = None) -> np.ndarray:
    resolved_arm = arm or env.active_arm
    control_pose = gripper_pose(env, resolved_arm).copy()
    midpoint = gripper_finger_link_midpoint(env, resolved_arm)
    if midpoint is not None:
        control_pose[:3] = midpoint
    return control_pose


def gripper_finger_link_midpoint(env, arm: str | None = None) -> np.ndarray | None:
    robot = getattr(env.task, "robot", None)
    if robot is None:
        return None
    resolved_arm = arm or env.active_arm
    if resolved_arm == "both":
        left = gripper_finger_link_midpoint(env, "left")
        right = gripper_finger_link_midpoint(env, "right")
        if left is None or right is None:
            return None
        return (left + right) / 2.0
    if resolved_arm == "left":
        entity = getattr(robot, "left_entity", None)
        names = ("fl_link7", "fl_link8")
    else:
        entity = getattr(robot, "right_entity", None)
        names = ("fr_link7", "fr_link8")
    if entity is None:
        return None
    points = []
    for name in names:
        link = entity.find_link_by_name(name)
        if link is None:
            return None
        pose = getattr(link, "entity_pose", None)
        if pose is None and hasattr(link, "get_pose"):
            pose = link.get_pose()
        if pose is None:
            return None
        points.append(np.asarray(pose.p, dtype=np.float64))
    return (points[0] + points[1]) / 2.0


def project_world_points(camera, points: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    try:
        intrinsic = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64)
        extrinsic = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        cam_points = (extrinsic @ points_h.T).T[:, :3]
        valid = cam_points[:, 2] > 1e-6
        if not np.all(valid):
            return None
        pixels_h = (intrinsic @ cam_points.T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        if not np.all(np.isfinite(pixels)):
            return None
        return pixels
    except Exception:
        return None


def draw_projected_arrow(
    draw,
    camera,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    color,
    width: int,
    font,
    label_offset: tuple[int, int],
    image_size: tuple[int, int],
) -> None:
    pixels = project_world_points(camera, np.vstack([start, end]))
    if pixels is None:
        return
    start_uv, end_uv = pixels
    draw_arrow(draw, tuple(start_uv), tuple(end_uv), color=color, width=width)
    label_xy = (float(end_uv[0]) + label_offset[0], float(end_uv[1]) + label_offset[1])
    draw_outlined_text(draw, clamp_text_xy(draw, label_xy, label, font, image_size), label, font=font, fill=color)


def draw_arrow(draw, start, end, color, width: int = 2) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    vec = np.array([ex - sx, ey - sy], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return
    direction = vec / norm
    normal = np.array([-direction[1], direction[0]])
    head_len = max(12.0, float(width) * 3.0)
    head_w = max(6.0, float(width) * 1.6)
    p1 = np.array([ex, ey]) - direction * head_len + normal * head_w
    p2 = np.array([ex, ey]) - direction * head_len - normal * head_w
    draw.polygon([(ex, ey), tuple(p1), tuple(p2)], fill=color)


def draw_text_panel(draw, text_lines: list[str], width: int, height: int) -> None:
    font = load_font(24 if width >= 1200 else 20)
    line_h = 30 if width >= 1200 else 24
    panel_w = min(width - 24, 790 if width >= 1200 else 650)
    panel_h = line_h * len(text_lines) + 20
    x0, y0 = 12, height - panel_h - 12
    draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    for i, line in enumerate(text_lines):
        draw.text((x0 + 12, y0 + 10 + i * line_h), line, font=font, fill=(255, 255, 255))


def draw_gripper_bar(draw, gripper_val: float, width: int, height: int) -> None:
    font = load_font(22 if width >= 1200 else 20)
    x0, y0 = width - 236, 18
    bar_w, bar_h = 198, 24
    gripper_val = float(np.clip(gripper_val, 0.0, 1.0))
    draw.rectangle((x0 - 10, y0 - 10, x0 + bar_w + 10, y0 + 60), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.text((x0, y0), "gripper", font=font, fill=(255, 255, 255))
    draw.rectangle((x0, y0 + 32, x0 + bar_w, y0 + 32 + bar_h), outline=(255, 255, 255), fill=(20, 20, 20))
    draw.rectangle((x0, y0 + 32, x0 + int(bar_w * gripper_val), y0 + 32 + bar_h), fill=(255, 220, 40))


def draw_axis_legend(draw, width: int, height: int) -> None:
    font = load_font(20 if width >= 1200 else 18)
    line_h = 34 if width >= 1200 else 30
    x0, y0 = width - (318 if width >= 1200 else 260), 118 if width >= 1200 else 92
    lines = [
        ("world X", (255, 60, 60)),
        ("world Y", (60, 255, 60)),
        ("world Z", (80, 150, 255)),
        ("local rx", (255, 170, 0)),
        ("local ry", (0, 220, 220)),
        ("local rz", (255, 80, 255)),
    ]
    panel_w = 286 if width >= 1200 else 238
    panel_h = line_h * len(lines) + 16
    draw.rectangle((x0 - 10, y0 - 10, x0 + panel_w, y0 + panel_h), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    for i, (label, color) in enumerate(lines):
        y = y0 + i * line_h
        draw.line((x0, y + 12, x0 + 58, y + 12), fill=color, width=6)
        draw.text((x0 + 70, y), label, font=font, fill=(255, 255, 255))


def load_font(size: int):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_outlined_text(draw, xy, text: str, font, fill) -> None:
    x, y = xy
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def clamp_text_xy(draw, xy, text: str, font, image_size: tuple[int, int], pad: int = 8) -> tuple[float, float]:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    max_x = max(pad, image_size[0] - text_w - pad)
    max_y = max(pad, image_size[1] - text_h - pad)
    return (float(np.clip(x, pad, max_x)), float(np.clip(y, pad, max_y)))


def sanitize_filename(label: str) -> str:
    safe = []
    for ch in label:
        safe.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(safe)[:140]


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    font = load_font(28 if image.width >= 1200 else 22 if image.width >= 700 else 16)
    header_h = 64 if image.width >= 1200 else 48 if image.width >= 700 else 32
    canvas = Image.new("RGB", (image.width, image.height + header_h), (0, 0, 0))
    canvas.paste(image.convert("RGB"), (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 16 if header_h >= 64 else 11 if header_h >= 48 else 8), label[:140], font=font, fill=(255, 255, 255))
    return np.asarray(canvas)


def write_h264_video(path: Path, frames: list[np.ndarray], fps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def print_summary(schema_rows, runtime_rows, correspondence_rows, feedback_rows, output_dir: Path) -> None:
    runtime_pass_count = sum(bool(row["pass"]) for row in runtime_rows)
    corr_pass_count = sum(bool(row["pass"]) for row in correspondence_rows)
    feedback_count = sum(bool(row["feedback_preserved"]) for row in feedback_rows)
    print(f"schema actions: {len(schema_rows)} / {len(schema_rows)} valid")
    print(f"runtime actions: {runtime_pass_count} / {len(runtime_rows)} passed")
    print(f"correspondence checks: {corr_pass_count} / {len(correspondence_rows)} passed")
    print(f"boundary feedback checks: {feedback_count} / {len(feedback_rows)} preserved")
    print(f"tables: {output_dir}")


if __name__ == "__main__":
    main()
