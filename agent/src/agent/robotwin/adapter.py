from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

from .geometry import gripper_center_pose, gripper_geometry_snapshot
from .prompts import observation_prompt
from .validator import allowed_actions_for_model
from .visuals import annotate_model_frame, resize_to_fit


@dataclass(frozen=True)
class FrameRecord:
    step: int
    path: Path
    prompt: str
    model_view: str
    model_overlay: str = "none"
    model_debug_path: Path | None = None
    model_debug_overlay: str = "eepose"
    record_path: Path | None = None
    record_view: str = "gripper_follow"
    record_overlay: str = "eepose"
    action: str | None = None
    info: dict[str, Any] | None = None
    geometry: dict[str, Any] | None = None
    fixed_view_images: dict[str, Path] | None = None


class RoboTwinAdapter:
    def __init__(
        self,
        run_dir: Path,
        *,
        benchmark_dir: Path | None = None,
        task: str = "grasp_single_bottle",
        config: str = "demo_clean",
        seed: int = 0,
        active_arm: str | None = None,
        max_steps: int = 50,
        model_view: str = "active",
        record_view: str = "gripper_follow",
        initial_camera_view: str = "center_high",
        camera_policy: str = "fine",
        width: int = 1280,
        height: int = 960,
        model_overlay: str = "none",
        model_debug_overlay: str = "eepose",
        record_overlay: str = "eepose",
        fixed_views: tuple[str, ...] = (),
        fixed_view_width: int = 384,
        fixed_view_height: int = 288,
        fixed_view_jpeg_quality: int = 60,
    ):
        self.run_dir = run_dir
        self.benchmark_dir = benchmark_dir or default_benchmark_dir()
        self.task_name = task
        self.config = config
        self.seed = seed
        self.active_arm = active_arm
        self.max_steps = max_steps
        self.model_view = model_view
        self.record_view = record_view
        self.initial_camera_view = initial_camera_view
        self.camera_policy = camera_policy
        self.width = width
        self.height = height
        self.model_overlay = validate_overlay(model_overlay, "model_overlay")
        self.model_debug_overlay = validate_overlay(model_debug_overlay, "model_debug_overlay")
        self.record_overlay = validate_overlay(record_overlay, "record_overlay")
        self.fixed_views = tuple(dict.fromkeys(fixed_views))
        self.fixed_view_width = fixed_view_width
        self.fixed_view_height = fixed_view_height
        self.fixed_view_jpeg_quality = fixed_view_jpeg_quality
        if self.fixed_view_width < 64 or self.fixed_view_height < 64:
            raise ValueError("fixed view image dimensions must be at least 64px")
        if not 1 <= self.fixed_view_jpeg_quality <= 95:
            raise ValueError("fixed view JPEG quality must be in [1, 95]")
        self.model_frames_dir = run_dir / "model_frames"
        self.model_debug_frames_dir = run_dir / "model_debug_frames"
        self.record_frames_dir = run_dir / "record_frames"
        self.fixed_view_frames_dir = run_dir / "fixed_view_frames"
        self.model_frames_dir.mkdir(parents=True, exist_ok=True)
        if self.model_debug_overlay != "none":
            self.model_debug_frames_dir.mkdir(parents=True, exist_ok=True)
        self.record_frames_dir.mkdir(parents=True, exist_ok=True)
        if self.fixed_views:
            self.fixed_view_frames_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[FrameRecord] = []
        self.model_video_frames: list[Any] = []
        self.model_debug_video_frames: list[Any] = []
        self.record_video_frames: list[Any] = []
        self.env = None
        self.modules: dict[str, Any] = {}

    def reset(self) -> FrameRecord:
        self._load_modules()
        env_cls = self.modules["InteractiveRoboTwinEnv"]
        self.env = env_cls(
            task_name=self.task_name,
            config_name=self.config,
            active_arm=self.active_arm,
            max_steps=self.max_steps,
            output_dir=self.run_dir / "robotwin_env",
            save_images=False,
        )
        observation = self.env.reset(seed=self.seed)
        if self.model_view == "active" and self.initial_camera_view != "default":
            self._set_active_camera_initial_view(self.initial_camera_view)
            observation = self.env._make_observation()
        return self._record(observation, action=None, info={"reset": True, "step_count": 0})

    def step(self, action: str | dict[str, Any], *, label: str | None = None) -> tuple[FrameRecord, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("reset() must be called before step()")
        result = self.env.step(action)
        record = self._record(result["observation"], action=label or compact_action(action), info=result["info"])
        return record, result

    def record_external_result(
        self, result: dict[str, Any], *, label: str
    ) -> FrameRecord:
        """Record an atomic tool action that already updated the environment."""

        if self.env is None:
            raise RuntimeError("reset() must be called before recording a tool action")
        return self._record(result["observation"], action=label, info=result["info"])

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def available_actions(self) -> list[str]:
        self._load_modules()
        arm = getattr(self.env, "active_arm", self.active_arm)
        return list(self.modules["list_available_actions"](arm))

    def model_available_actions(self) -> list[str]:
        return allowed_actions_for_model(self.available_actions(), self.camera_policy)

    def action_space_description(self) -> str:
        self._load_modules()
        arm = getattr(self.env, "active_arm", self.active_arm)
        return str(self.modules["describe_action_space"](arm))

    def geometry(self) -> dict[str, Any] | None:
        if self.env is None:
            return None
        self._load_modules()
        return gripper_geometry_snapshot(self.env, self.modules)

    def image_for_step(self, value: Any | None) -> FrameRecord:
        if not self.records:
            raise ValueError("no frames are available")
        if value is None:
            return self.records[-1]
        step = parse_step(value)
        for record in self.records:
            if record.step == step:
                return record
        raise ValueError(f"historical image step {step} is not available")

    def write_videos(self) -> dict[str, Path]:
        self._load_modules()
        paths: dict[str, Path] = {}
        if self.model_video_frames:
            path = self.run_dir / "model_replay.mp4"
            self.modules["write_h264_video"](path, self.model_video_frames)
            paths["model_replay"] = path
        if self.model_debug_video_frames:
            path = self.run_dir / "model_debug_replay.mp4"
            self.modules["write_h264_video"](path, self.model_debug_video_frames)
            paths["model_debug_replay"] = path
        if self.record_video_frames:
            path = self.run_dir / "replay.mp4"
            self.modules["write_h264_video"](path, self.record_video_frames)
            paths["replay"] = path
        return paths

    def _record(self, observation: dict[str, Any], *, action: str | None, info: dict[str, Any] | None) -> FrameRecord:
        if self.env is None:
            raise RuntimeError("reset() must initialize env")
        imageio = self.modules["imageio"]
        step = int(info.get("step_count", 0)) if info else int(getattr(self.env, "step_count", 0))
        suffix = "reset" if action is None else sanitize(action)
        model_path = self.model_frames_dir / f"step_{step:03d}_{suffix}.png"
        model_debug_path = (
            self.model_debug_frames_dir / f"step_{step:03d}_{suffix}.png"
            if self.model_debug_overlay != "none"
            else None
        )
        record_path = self.record_frames_dir / f"step_{step:03d}_{suffix}.png"
        model_frame, model_debug_frame = self._capture_model_frames()
        fixed_view_frames = self._capture_fixed_view_frames()
        record_frame = self._capture_record_frame()
        geometry = self.geometry()
        imageio.imwrite(model_path, model_frame)
        if model_debug_path is not None and model_debug_frame is not None:
            imageio.imwrite(model_debug_path, model_debug_frame)
        imageio.imwrite(record_path, record_frame)
        fixed_view_images: dict[str, Path] = {}
        for view, frame in fixed_view_frames.items():
            fixed_path = self.fixed_view_frames_dir / f"step_{step:03d}_{sanitize(view)}.jpg"
            imageio.imwrite(fixed_path, frame, quality=self.fixed_view_jpeg_quality)
            fixed_view_images[view] = fixed_path
        self.model_video_frames.append(model_frame)
        if model_debug_frame is not None:
            self.model_debug_video_frames.append(model_debug_frame)
        self.record_video_frames.append(record_frame)
        prompt = observation_prompt(
            observation,
            step=step,
            model_view=self.model_view,
            model_overlay=self.model_overlay,
            record_view=self.record_view,
            record_overlay=self.record_overlay,
            initial_camera_view=self.initial_camera_view,
            camera_policy=self.camera_policy,
            geometry=geometry,
            fixed_view_images=fixed_view_images or None,
            available_actions=self.model_available_actions(),
            historical_steps=[record.step for record in self.records],
        )
        record = FrameRecord(
            step=step,
            path=model_path,
            prompt=prompt,
            model_view=self.model_view,
            model_overlay=self.model_overlay,
            model_debug_path=model_debug_path,
            model_debug_overlay=self.model_debug_overlay,
            record_path=record_path,
            record_view=self.record_view,
            record_overlay=self.record_overlay,
            action=action,
            info=info,
            geometry=geometry,
            fixed_view_images=fixed_view_images or None,
        )
        self.records.append(record)
        return record

    def _capture_fixed_view_frames(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("reset() must initialize env")
        return {
            view: resize_to_fit(
                capture_view_frame_and_camera(
                    self.env,
                    view,
                    self.fixed_view_width,
                    self.fixed_view_height,
                    self.modules,
                )[0],
                self.fixed_view_width,
                self.fixed_view_height,
            )
            for view in self.fixed_views
        }

    def _capture_model_frames(self):
        if self.env is None:
            raise RuntimeError("reset() must initialize env")
        frame, camera = capture_view_frame_and_camera(self.env, self.model_view, self.width, self.height, self.modules)
        clean_frame = resize_to_fit(frame, self.width, self.height)
        model_frame = (
            annotate_model_frame(frame, camera, self.env, self.width, self.height, self.modules)
            if self.model_overlay == "eepose"
            else clean_frame
        )
        model_debug_frame = (
            annotate_model_frame(frame, camera, self.env, self.width, self.height, self.modules)
            if self.model_debug_overlay == "eepose"
            else None
        )
        return model_frame, model_debug_frame

    def _capture_record_frame(self):
        if self.env is None:
            raise RuntimeError("reset() must initialize env")
        frame, camera = capture_view_frame_and_camera(
            self.env,
            self.record_view,
            self.width,
            self.height,
            self.modules,
        )
        if self.record_overlay == "eepose":
            return annotate_model_frame(frame, camera, self.env, self.width, self.height, self.modules)
        return resize_to_fit(frame, self.width, self.height)

    def _set_active_camera_initial_view(self, view: str) -> None:
        if self.env is None:
            raise RuntimeError("reset() must initialize env")
        pose = active_initial_camera_pose(self.env, view, self.modules)
        camera = self.env.camera.get_camera()
        camera.entity.set_pose(pose)
        self.env.task._update_render()

    def _load_modules(self) -> None:
        if self.modules:
            return
        benchmark_dir = self.benchmark_dir.resolve()
        scripts_dir = benchmark_dir / "scripts"
        for path in (str(benchmark_dir), str(scripts_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from active_spatial_benchmark import InteractiveRoboTwinEnv, describe_action_space, list_available_actions
        from active_spatial_benchmark.camera_control import look_at_pose
        from test_action_suite import (
            VisualRecorder,
            capture_active_rgb_and_camera,
            capture_observer_rgb_and_camera,
            clamp_text_xy,
            draw_arrow,
            draw_outlined_text,
            gripper_pose,
            load_font,
            observer_pose,
            project_world_points,
            write_h264_video,
        )
        import imageio.v2 as imageio

        self.modules = {
            "InteractiveRoboTwinEnv": InteractiveRoboTwinEnv,
            "describe_action_space": describe_action_space,
            "list_available_actions": list_available_actions,
            "look_at_pose": look_at_pose,
            "VisualRecorder": VisualRecorder,
            "capture_active_rgb_and_camera": capture_active_rgb_and_camera,
            "capture_observer_rgb_and_camera": capture_observer_rgb_and_camera,
            "clamp_text_xy": clamp_text_xy,
            "draw_arrow": draw_arrow,
            "draw_outlined_text": draw_outlined_text,
            "gripper_pose": gripper_pose,
            "load_font": load_font,
            "observer_pose": observer_pose,
            "project_world_points": project_world_points,
            "write_h264_video": write_h264_video,
            "imageio": imageio,
        }


def capture_view_frame_and_camera(env, view: str, width: int, height: int, modules: dict[str, Any]):
    if view == "active":
        return modules["capture_active_rgb_and_camera"](env)
    return modules["capture_observer_rgb_and_camera"](env, view, width, height)


def active_initial_camera_pose(env, view: str, modules: dict[str, Any]):
    import numpy as np

    if view == "default":
        return env.camera.get_camera().entity.get_pose()
    if view in {"topdown", "gripper_follow", "side", "front_side_45", "side_top_45", "oblique_45"}:
        return modules["observer_pose"](env, view)
    if view == "center_high":
        gripper = gripper_center_pose(env, modules)[:3]
        active_arm = getattr(env, "active_arm", "right")
        side = 0.0 if active_arm == "both" else (1.0 if active_arm == "right" else -1.0)
        position = np.array([0.0, -0.58, 1.48], dtype=np.float64)
        target = np.array([0.17 * side, gripper[1] - 0.02, max(0.86, gripper[2] - 0.05)], dtype=np.float64)
        return modules["look_at_pose"](position, target, up_hint=np.array([0.0, 0.0, 1.0]))
    if view == "workspace":
        frame = env.camera.frame()
        return modules["look_at_pose"](frame.position, np.array([0.0, -0.08, 0.86], dtype=np.float64))
    if view == "gripper":
        frame = env.camera.frame()
        return modules["look_at_pose"](frame.position, gripper_center_pose(env, modules)[:3])
    raise ValueError(f"unsupported initial camera view: {view}")


def compact_action(action: str | dict[str, Any]) -> str:
    if isinstance(action, str):
        return action
    if action.get("target") == "gripper" and action.get("type") == "move_world":
        axis = str(action.get("axis", ""))
        sign = "pos" if str(action.get("sign", "")) in {"+", "pos", "positive", "+1", "1"} else "neg"
        prefix = f"{action.get('arm')}_gripper" if action.get("arm") else "gripper"
        return f"{prefix}.world_{axis}_{sign}.{action.get('distance_mm')}mm"
    if action.get("target") == "gripper" and action.get("type") == "rotate":
        prefix = f"{action.get('arm')}_gripper" if action.get("arm") else "gripper"
        return f"{prefix}.{action.get('direction')}.{action.get('angle_deg')}deg"
    if action.get("target") == "dual_gripper":
        return f"dual_gripper.{action.get('type')}"
    return str(action)


def frame_to_json(record: FrameRecord, root: Path) -> dict[str, Any]:
    return {
        "step": record.step,
        "model_image": str(record.path.relative_to(root)),
        "model_view": record.model_view,
        "model_overlay": record.model_overlay,
        "model_debug_image": str(record.model_debug_path.relative_to(root)) if record.model_debug_path else None,
        "model_debug_overlay": record.model_debug_overlay,
        "record_image": str(record.record_path.relative_to(root)) if record.record_path else None,
        "record_view": record.record_view,
        "record_overlay": record.record_overlay,
        "action": record.action,
        "info": record.info,
        "geometry": record.geometry,
        "fixed_view_images": {
            view: str(path.relative_to(root))
            for view, path in (record.fixed_view_images or {}).items()
        },
    }


def parse_step(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"cannot parse step from {value!r}")
    return int(match.group(0))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80] or "action"


def validate_overlay(value: str, name: str) -> str:
    if value not in {"none", "eepose"}:
        raise ValueError(f"{name} must be one of: none, eepose")
    return value


def default_benchmark_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "active_spatial_benchmark_xyz"
