from __future__ import annotations

import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

from .actions import (
    ActionSpec,
    ActionValidationError,
    describe_action_space,
    gripper_rotation_axis_and_sign,
    gripper_rotation_angle_deg,
    gripper_translation_distance,
    list_available_actions,
    parse_action,
)
from .camera_control import CameraController
from .rotation_geometry import fixed_center_local_rotation_waypoints, fixed_center_rotation_waypoints


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
DUAL_ARM_TASKS = {
    "place_shoe",
    "handover_mic",
    "handover_horizontal_block",
    "handover_block",
    "handover_cube_to_target",
    "lift_pot",
}
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

_IMPORT_CWD = os.getcwd()
os.chdir(ROBOTWIN_ROOT)
try:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402
finally:
    os.chdir(_IMPORT_CWD)


@contextmanager
def robotwin_cwd():
    old_cwd = os.getcwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class InteractiveRoboTwinEnv:
    """Step-by-step active-perception wrapper for RoboTwin tasks."""

    def __init__(
        self,
        task_name: str = "grasp_single_bottle",
        config_name: str = "demo_clean",
        active_camera: str = "head_camera",
        active_arm: str | None = None,
        max_steps: int = 40,
        output_dir: str | Path | None = None,
        save_images: bool = True,
        gripper_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-0.55, -0.45, 0.74),
            (0.55, 0.35, 1.35),
        ),
    ):
        self.task_name = task_name
        self.config_name = config_name
        self.active_camera = active_camera
        self.requested_active_arm = active_arm
        self.max_steps = max_steps
        self.save_images = save_images
        self.output_dir = Path(output_dir) if output_dir is not None else PACKAGE_ROOT / "runs"
        self.gripper_bounds = np.asarray(gripper_bounds, dtype=np.float64)

        self.task = None
        self.camera = None
        self.active_arm = active_arm or "right"
        self.task_instruction = ""
        self.step_count = 0
        self.invalid_action_count = 0
        self.camera_action_count = 0
        self.action_history: list[str] = []
        self.last_action_valid = True
        self.last_planner_success = True
        self.last_error: str | None = None
        self.done = False
        self.success = False
        self.run_dir: Path | None = None

    def reset(
        self,
        task_name: str | None = None,
        seed: int = 0,
        config_name: str | None = None,
        active_arm: str | None = None,
    ) -> dict[str, Any]:
        self.close()
        self.task_name = task_name or self.task_name
        self.config_name = config_name or self.config_name
        self.requested_active_arm = active_arm or self.requested_active_arm
        self.step_count = 0
        self.invalid_action_count = 0
        self.camera_action_count = 0
        self.action_history = []
        self.last_action_valid = True
        self.last_planner_success = True
        self.last_error = None
        self.done = False
        self.success = False
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_dir / f"{self.task_name}_seed{seed}_{timestamp}"
        if self.save_images:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        with robotwin_cwd():
            self.task = self._instantiate_task(self.task_name)
            args = self._load_task_args(self.config_name)
            args["task_name"] = self.task_name
            args["save_data"] = False
            args["need_plan"] = True
            args["render_freq"] = 0
            args.setdefault("data_type", {})
            args["data_type"]["rgb"] = True
            args["data_type"]["endpose"] = False
            args["data_type"]["qpos"] = False
            args["data_type"]["depth"] = False
            args["data_type"]["pointcloud"] = False
            self.task.setup_demo(now_ep_num=0, seed=seed, **args)

        self.active_arm = self.requested_active_arm or self._select_active_arm()
        if self.active_arm not in {"left", "right", "both"}:
            raise ValueError("active_arm must be 'left', 'right', or 'both'")
        self.camera = CameraController(self.task, camera_name=self.active_camera)
        self.task_instruction = self._load_task_instruction(self.task_name)
        instruction_getter = getattr(self.task, "get_instruction", None)
        dynamic_instruction = instruction_getter() if callable(instruction_getter) else None
        if isinstance(dynamic_instruction, str) and dynamic_instruction.strip():
            self.task_instruction = dynamic_instruction.strip()
        return self._make_observation()

    def step(self, raw_action: str | dict[str, Any]) -> dict[str, Any]:
        if self.task is None or self.camera is None:
            raise RuntimeError("Environment must be reset before step().")
        if self.done:
            return {
                "observation": self._make_observation(),
                "reward": 1.0 if self.success else 0.0,
                "done": True,
                "info": self._info(),
            }

        self.step_count += 1
        action_valid = True
        planner_success = True
        error = None
        compact = "<invalid>"

        try:
            action = parse_action(raw_action)
            compact = action.compact()
            with robotwin_cwd():
                planner_success = self._apply_action(action)
            if action.target == "camera":
                self.camera_action_count += 1
        except (ActionValidationError, ValueError, KeyError, AssertionError) as exc:
            action_valid = False
            planner_success = False
            error = str(exc)
            self.invalid_action_count += 1
        except Exception as exc:  # Keep benchmark episodes alive after planner/runtime failures.
            planner_success = False
            error = f"{type(exc).__name__}: {exc}"

        if self.task is not None and getattr(self.task, "plan_success", True) is False:
            planner_success = False
            self.task.plan_success = True

        self.last_action_valid = action_valid
        self.last_planner_success = planner_success
        self.last_error = error
        self.action_history.append(compact)

        with robotwin_cwd():
            self.success = bool(self.task.check_success())
        self.done = self.success or self.step_count >= self.max_steps

        return {
            "observation": self._make_observation(),
            "reward": 1.0 if self.success else 0.0,
            "done": self.done,
            "info": self._info(),
        }

    def close(self) -> None:
        if self.task is None:
            return
        try:
            with robotwin_cwd():
                self.task.close_env(clear_cache=True)
        except Exception:
            pass
        self.task = None
        self.camera = None

    def available_actions(self) -> list[str]:
        return list_available_actions(self.active_arm)

    def _apply_action(self, action: ActionSpec) -> bool:
        if action.target == "camera":
            assert self.camera is not None
            self.camera.apply(action, active_arm=self.active_arm)
            self.task._update_render()
            return True
        if action.target == "gripper":
            return self._apply_gripper_action(action)
        if action.target == "dual_gripper":
            return self._apply_dual_gripper_action(action)
        raise ValueError(f"Unsupported action target: {action.target}")

    def _apply_gripper_action(self, action: ActionSpec) -> bool:
        arm = self._resolve_action_arm(action)
        if action.type == "open":
            result = self.task.move(self.task.open_gripper(arm), save_freq=None)
            return bool(result)
        if action.type == "close":
            result = self.task.move(self.task.close_gripper(arm), save_freq=None)
            return bool(result)
        if action.type == "translate":
            assert action.direction is not None
            return self._translate_gripper(action, arm)
        if action.type == "rotate":
            assert action.direction is not None
            return self._rotate_gripper(action, arm)
        raise ValueError(f"Unsupported gripper action type: {action.type}")

    def _apply_dual_gripper_action(self, action: ActionSpec) -> bool:
        if self.active_arm != "both":
            raise ActionValidationError("dual_gripper actions are available only in dual-arm mode")
        if action.type in {"open", "close"}:
            command = self.task.open_gripper if action.type == "open" else self.task.close_gripper
            result = self.task.move(command("left"), command("right"), save_freq=None)
            return bool(result)
        if action.type == "translate":
            assert action.left_action is not None and action.right_action is not None
            left_pose = self._translated_pose(action.left_action, "left")
            right_pose = self._translated_pose(action.right_action, "right")
            if not self._gripper_position_in_bounds(left_pose[:3]) or not self._gripper_position_in_bounds(right_pose[:3]):
                return False
            result = self.task.move(
                self.task.move_to_pose("left", left_pose.tolist()),
                self.task.move_to_pose("right", right_pose.tolist()),
                save_freq=None,
            )
            return bool(result)
        raise ValueError(f"Unsupported dual gripper action type: {action.type}")

    def _resolve_action_arm(self, action: ActionSpec) -> str:
        if self.active_arm == "both":
            if action.arm not in {"left", "right"}:
                raise ActionValidationError("Single-gripper action in dual-arm mode requires arm='left' or arm='right'")
            return action.arm
        if action.arm is not None and action.arm != self.active_arm:
            raise ActionValidationError(
                f"Action targets {action.arm} arm, but this episode controls only {self.active_arm} arm"
            )
        return self.active_arm

    def _translated_pose(self, action: ActionSpec, arm: str) -> np.ndarray:
        assert self.camera is not None
        assert action.direction is not None
        delta = self.camera.gripper_direction(action.direction) * gripper_translation_distance(action)
        pose = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64)
        pose[:3] = self._clamp_gripper_position(pose[:3] + delta)
        return pose

    def _translate_gripper(self, action: ActionSpec, arm: str) -> bool:
        pose = self._translated_pose(action, arm)
        result = self.task.move(self.task.move_to_pose(arm, pose.tolist()), save_freq=None)
        return bool(result)

    def _rotate_gripper(self, action: ActionSpec, arm: str) -> bool:
        pose = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64)
        assert action.direction is not None
        axis_name, signed_direction = gripper_rotation_axis_and_sign(action.direction)
        angle = np.deg2rad(gripper_rotation_angle_deg(action)) * signed_direction
        center = self._gripper_finger_center(arm)
        if action.rotation_frame == "local":
            target_poses = fixed_center_local_rotation_waypoints(pose, center, axis_name, angle)
        else:
            target_poses = fixed_center_rotation_waypoints(pose, center, axis_name, angle)
        if not all(self._gripper_position_in_bounds(target_pose[:3]) for target_pose in target_poses):
            return False

        for target_pose in target_poses:
            current_pose = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64)
            pre_rotate_pose = current_pose.copy()
            pre_rotate_pose[:3] = target_pose[:3]
            pre_move_result = self.task.move(self.task.move_to_pose(arm, pre_rotate_pose.tolist()), save_freq=None)
            if not pre_move_result:
                return False
            rotate_result = self.task.move(self.task.move_to_pose(arm, target_pose.tolist()), save_freq=None)
            if not rotate_result:
                return False
        return True

    def _gripper_finger_center(self, arm: str | None = None) -> np.ndarray:
        resolved_arm = arm or self.active_arm
        if resolved_arm == "both":
            return (self._gripper_finger_center("left") + self._gripper_finger_center("right")) / 2.0
        midpoint = self._gripper_finger_link_midpoint(resolved_arm)
        if midpoint is not None:
            return midpoint
        return np.asarray(self.task.get_arm_pose(resolved_arm)[:3], dtype=np.float64)

    def _gripper_finger_link_midpoint(self, arm: str) -> np.ndarray | None:
        robot = getattr(self.task, "robot", None)
        if robot is None:
            return None
        if arm == "left":
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

    def _make_observation(self) -> dict[str, Any]:
        with robotwin_cwd():
            obs = self.task.get_obs()
        camera_obs = obs["observation"][self.active_camera]
        rgb = camera_obs["rgb"]
        image_path = None
        if self.save_images and self.run_dir is not None:
            image_path = self.run_dir / f"step_{self.step_count:04d}.png"
            imageio.imwrite(image_path, rgb)

        return {
            "task": self.task_instruction,
            "action_space_description": describe_action_space(self.active_arm),
            "image": rgb,
            "image_path": str(image_path) if image_path is not None else None,
            "gripper_state": self._gripper_state(),
            "action_history": deepcopy(self.action_history),
            "available_actions": self.available_actions(),
        }

    def _gripper_state(self) -> dict[str, Any]:
        if self.active_arm == "both":
            return {
                "active_arm": "both",
                "left": self._single_gripper_state("left"),
                "right": self._single_gripper_state("right"),
                "last_action_valid": self.last_action_valid,
                "last_planner_success": self.last_planner_success,
            }
        state = self._single_gripper_state(self.active_arm)
        state.update({
            "active_arm": self.active_arm,
            "last_action_valid": self.last_action_valid,
            "last_planner_success": self.last_planner_success,
        })
        return state

    def _single_gripper_state(self, arm: str) -> dict[str, Any]:
        if arm == "left":
            is_open = bool(self.task.is_left_gripper_open())
            is_closed = bool(self.task.is_left_gripper_close())
        else:
            is_open = bool(self.task.is_right_gripper_open())
            is_closed = bool(self.task.is_right_gripper_close())
        control_pose = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64)
        finger_center = self._gripper_finger_center(arm)
        import transforms3d as t3d

        control_rpy_deg = np.rad2deg(t3d.euler.quat2euler(control_pose[3:]))
        return {
            "is_open": is_open,
            "is_closed": is_closed,
            "finger_center_xyz": [float(x) for x in finger_center],
            "control_eepose_xyz": [float(x) for x in control_pose[:3]],
            "control_eepose_rpy_deg": [float(x) for x in control_rpy_deg],
            "control_eepose_quat": [float(x) for x in control_pose[3:]],
        }

    def _info(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "planner_success": self.last_planner_success,
            "action_valid": self.last_action_valid,
            "last_error": self.last_error,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "invalid_action_count": self.invalid_action_count,
            "camera_action_count": self.camera_action_count,
            "active_arm": self.active_arm,
            "active_camera": self.active_camera,
            "run_dir": str(self.run_dir) if self.run_dir is not None else None,
        }

    def _clamp_gripper_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, self.gripper_bounds[0], self.gripper_bounds[1])

    def _gripper_position_in_bounds(self, position: np.ndarray) -> bool:
        position = np.asarray(position, dtype=np.float64)
        return bool(np.all(position >= self.gripper_bounds[0]) and np.all(position <= self.gripper_bounds[1]))

    def _select_active_arm(self) -> str:
        if self.task_name in DUAL_ARM_TASKS:
            return "both"
        if hasattr(self.task, "arm_tag"):
            return str(self.task.arm_tag)
        for attr_name in ("cup", "object", "can", "hammer"):
            actor = getattr(self.task, attr_name, None)
            if actor is None:
                continue
            try:
                return "right" if actor.get_pose().p[0] > 0 else "left"
            except Exception:
                continue
        return "right"

    def _instantiate_task(self, task_name: str):
        module = importlib.import_module(f"envs.{task_name}")
        try:
            task_cls = getattr(module, task_name)
        except AttributeError as exc:
            raise ValueError(f"No RoboTwin task class named {task_name!r}") from exc
        return task_cls()

    def _load_task_args(self, config_name: str) -> dict[str, Any]:
        config_path = Path(CONFIGS_PATH) / f"{config_name}.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        embodiment_type = args.get("embodiment")
        embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
        with embodiment_config_path.open("r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(name: str) -> str:
            robot_file = embodiment_types[name]["file_path"]
            if robot_file is None:
                raise ValueError(f"Missing embodiment file for {name!r}")
            return robot_file

        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("Embodiment config must contain either 1 or 3 entries.")

        args["left_embodiment_config"] = self._load_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = self._load_embodiment_config(args["right_robot_file"])
        return args

    def _load_embodiment_config(self, robot_file: str) -> dict[str, Any]:
        robot_config_file = ROBOTWIN_ROOT / robot_file / "config.yml"
        with robot_config_file.open("r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def _load_task_instruction(self, task_name: str) -> str:
        instruction_path = ROBOTWIN_ROOT / "description" / "task_instruction" / f"{task_name}.json"
        if not instruction_path.exists():
            return f"Complete the RoboTwin task: {task_name}."
        with instruction_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("full_description") or f"Complete the RoboTwin task: {task_name}."
