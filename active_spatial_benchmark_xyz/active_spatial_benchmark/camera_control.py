from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .actions import (
    ActionSpec,
    CAMERA_ROTATION_SCALES_DEG,
    CAMERA_TRANSLATION_SCALES,
)
from .camera_layout import DEFAULT_LAYOUT_VIEWS, CameraLayout


DEFAULT_CAMERA_BOUNDS = np.array(
    [
        [-0.85, -0.95, 0.78],
        [0.85, 0.35, 1.85],
    ]
)

# The discrete view catalogue as measured offsets from the finger center, with
# the sign of the x component mirrored for a left arm. This is the single source
# of truth for the layout: `view` moves the camera here, and callers that need a
# candidate pose before moving predict it with `camera_position_for_view`.
#
# The offsets have distinct magnitudes (0.68, 0.583, 0.641, 0.679, 0.727 m) and
# are not radial about a single point, so a camera position taken from this table
# is genuinely independent of the view ray -- unlike a position reconstructed as
# `centroid - direction * standoff`, which is collinear with the ray by
# construction and therefore adds no information to it.
DEFAULT_CAMERA_LAYOUT = CameraLayout.default()
VIEW_LAYOUT_OFFSETS_M: dict[str, tuple[float, float, float]] = {
    name: tuple(spec["offset_world_m"])
    for name, spec in DEFAULT_LAYOUT_VIEWS.items()
}
VIEW_LAYOUT_UP_HINTS: dict[str, tuple[float, float, float]] = {
    name: tuple(spec["up_hint_world"])
    for name, spec in DEFAULT_LAYOUT_VIEWS.items()
}


def camera_position_for_view(
    view: str | None,
    *,
    finger_center_world: np.ndarray,
    view_arm: str = "right",
    layout: CameraLayout | None = None,
) -> np.ndarray:
    """Predict where the camera lands for a named discrete view.

    Uses the declared layout table rather than a standoff along the view ray, so
    the resulting position carries information the ray does not. Unknown view
    names raise instead of falling back to a default pose.
    """

    if layout is not None:
        return np.asarray(
            np.asarray(finger_center_world, dtype=np.float64)
            + layout.offset_for_arm(view, view_arm=view_arm)
        )
    if view not in VIEW_LAYOUT_OFFSETS_M:
        raise ValueError(f"Unsupported camera view mode: {view}")
    offset = np.asarray(VIEW_LAYOUT_OFFSETS_M[view], dtype=np.float64).copy()
    if view_arm not in {"left", "right", "both"}:
        raise ValueError(f"Unsupported view arm: {view_arm}")
    if view_arm == "left":
        offset[0] = -offset[0]
    return np.asarray(finger_center_world, dtype=np.float64) + offset


def camera_pose_for_view(
    view: str | None,
    *,
    finger_center_world: np.ndarray,
    view_arm: str = "right",
    camera_bounds: np.ndarray = DEFAULT_CAMERA_BOUNDS,
    layout: CameraLayout | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounded layout position and the forward ray actually used.

    Candidate scoring and camera execution must use the same post-clamp pose.
    Returning both quantities from one helper prevents stale hand-written view
    directions from disagreeing with the camera that will acquire the frame.
    """

    target = np.asarray(finger_center_world, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("finger center must be a finite 3-vector")
    bounds = np.asarray(camera_bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("camera bounds must be a finite 2x3 array")
    if np.any(bounds[0] > bounds[1]):
        raise ValueError("camera lower bounds must not exceed upper bounds")
    raw_position = camera_position_for_view(
        view,
        finger_center_world=target,
        view_arm=view_arm,
        layout=layout,
    )
    bounded_position = np.clip(raw_position, bounds[0], bounds[1])
    return bounded_position, normalized(target - bounded_position)


@dataclass(frozen=True)
class CameraFrame:
    position: np.ndarray
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray


def normalized(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def normalize_camera_pose_perturbation(
    perturbation: Mapping[str, Any] | None,
) -> dict[str, list[float]] | None:
    """Validate the inference-only world-frame camera mounting perturbation."""

    if perturbation is None:
        return None
    if not isinstance(perturbation, Mapping):
        raise ValueError("camera pose perturbation must be a mapping")
    translation = np.asarray(
        perturbation.get("translation_world_m", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    rotation = np.asarray(
        perturbation.get("rotation_rpy_deg", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if translation.shape != (3,) or rotation.shape != (3,):
        raise ValueError(
            "camera pose perturbation translation and rotation must be 3-vectors"
        )
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise ValueError("camera pose perturbation must contain finite values")
    if np.linalg.norm(translation) > 0.25:
        raise ValueError("camera translation perturbation exceeds 0.25 m")
    if np.max(np.abs(rotation)) > 30.0:
        raise ValueError("camera rotation perturbation exceeds 30 degrees")
    return {
        "translation_world_m": translation.tolist(),
        "rotation_rpy_deg": rotation.tolist(),
    }


def perturb_camera_frame(
    position: np.ndarray,
    forward: np.ndarray,
    perturbation: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded world-frame mounting perturbation to a camera frame."""

    normalized_perturbation = normalize_camera_pose_perturbation(perturbation)
    position = np.asarray(position, dtype=np.float64)
    forward = normalized(np.asarray(forward, dtype=np.float64))
    if normalized_perturbation is None:
        return position.copy(), forward.copy()
    rotation = t3d.euler.euler2mat(
        *np.deg2rad(normalized_perturbation["rotation_rpy_deg"]), axes="sxyz"
    )
    translated = position + np.asarray(
        normalized_perturbation["translation_world_m"], dtype=np.float64
    )
    return translated, normalized(rotation @ forward)


def perturb_camera_pose(pose: sapien.Pose, perturbation: Mapping[str, Any] | None):
    """Return a SAPIEN pose after a world-frame mounting perturbation."""

    normalized_perturbation = normalize_camera_pose_perturbation(perturbation)
    if normalized_perturbation is None:
        return type(pose)(pose.p.copy(), pose.q.copy())
    delta_rotation = t3d.euler.euler2mat(
        *np.deg2rad(normalized_perturbation["rotation_rpy_deg"]), axes="sxyz"
    )
    matrix = pose.to_transformation_matrix().astype(np.float64)
    matrix[:3, :3] = delta_rotation @ matrix[:3, :3]
    matrix[:3, 3] += np.asarray(
        normalized_perturbation["translation_world_m"], dtype=np.float64
    )
    quaternion = t3d.quaternions.mat2quat(matrix[:3, :3])
    return type(pose)(matrix[:3, 3], quaternion)


def pose_from_frame(
    position: np.ndarray, forward: np.ndarray, left: np.ndarray
) -> sapien.Pose:
    forward = normalized(np.asarray(forward, dtype=np.float64))
    left = np.asarray(left, dtype=np.float64)
    left = left - np.dot(left, forward) * forward
    left = normalized(left)
    up = normalized(np.cross(forward, left))
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = np.asarray(position, dtype=np.float64)
    return sapien.Pose(mat44)


def look_at_pose(
    position: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
) -> sapien.Pose:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_hint = (
        np.array([0.0, 0.0, 1.0])
        if up_hint is None
        else np.asarray(up_hint, dtype=np.float64)
    )
    forward = normalized(target - position)
    left = np.cross(up_hint, forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([-1.0, 0.0, 0.0])
    return pose_from_frame(position, forward, left)


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    quat = t3d.quaternions.axangle2quat(normalized(axis), angle_rad)
    mat = t3d.quaternions.quat2mat(quat)
    return mat @ vec


class CameraController:
    def __init__(
        self,
        task_env,
        camera_name: str = "head_camera",
        workspace_center: tuple[float, float, float] = (0.0, -0.08, 0.86),
        camera_bounds: np.ndarray = DEFAULT_CAMERA_BOUNDS,
        camera_layout: CameraLayout | Mapping[str, Any] | None = None,
    ):
        self.task_env = task_env
        self.camera_name = camera_name
        self.workspace_center = np.asarray(workspace_center, dtype=np.float64)
        self.camera_bounds = np.asarray(camera_bounds, dtype=np.float64)
        self.camera_layout = (
            CameraLayout.from_mapping(camera_layout)
            if isinstance(camera_layout, Mapping)
            else camera_layout or DEFAULT_CAMERA_LAYOUT
        )

    def get_camera(self):
        cameras = self.task_env.cameras
        for camera, name in zip(cameras.static_camera_list, cameras.static_camera_name):
            if name == self.camera_name:
                return camera
        raise ValueError(f"Camera {self.camera_name!r} is not available.")

    def frame(self) -> CameraFrame:
        camera = self.get_camera()
        mat = camera.entity.get_pose().to_transformation_matrix()
        return CameraFrame(
            position=mat[:3, 3].astype(np.float64),
            forward=normalized(mat[:3, 0].astype(np.float64)),
            left=normalized(mat[:3, 1].astype(np.float64)),
            up=normalized(mat[:3, 2].astype(np.float64)),
        )

    def gripper_direction(self, direction: str) -> np.ndarray:
        if direction in {"image_left", "world_x_neg"}:
            return np.array([-1.0, 0.0, 0.0])
        if direction in {"image_right", "world_x_pos"}:
            return np.array([1.0, 0.0, 0.0])
        if direction in {"image_up", "world_y_pos"}:
            return np.array([0.0, 1.0, 0.0])
        if direction in {"image_down", "world_y_neg"}:
            return np.array([0.0, -1.0, 0.0])
        if direction in {"depth_forward", "world_z_neg"}:
            return np.array([0.0, 0.0, -1.0])
        if direction in {"depth_backward", "world_z_pos"}:
            return np.array([0.0, 0.0, 1.0])
        if direction == "lift_up":
            return np.array([0.0, 0.0, 1.0])
        if direction == "lift_down":
            return np.array([0.0, 0.0, -1.0])
        raise ValueError(f"Unsupported gripper direction: {direction}")

    def apply(self, action: ActionSpec, active_arm: str) -> None:
        if action.type == "look_at":
            if action.mode == "gripper":
                self.look_at_gripper(active_arm)
            elif action.mode in {"left_gripper", "right_gripper", "both_grippers"}:
                self.look_at_gripper(action.mode.split("_", 1)[0])
            elif action.mode == "workspace":
                self.look_at_workspace()
            else:
                raise ValueError(f"Unsupported look_at mode: {action.mode}")
            return
        if action.type == "view":
            self.view(active_arm, action.mode)
            return
        if action.type == "move":
            self._move_camera(action)
            return
        if action.type == "rotate":
            self._rotate_camera(action)
            return
        raise ValueError(f"Unsupported camera action type: {action.type}")

    def look_at_workspace(self) -> None:
        frame = self.frame()
        self._set_pose(look_at_pose(frame.position, self.workspace_center))

    def look_at_gripper(self, active_arm: str) -> None:
        gripper_xyz = self._finger_center(active_arm)
        frame = self.frame()
        self._set_pose(look_at_pose(frame.position, gripper_xyz))

    def view(self, active_arm: str, mode: str | None) -> None:
        view_arm, base_mode = self._view_target(active_arm, mode)
        target = self._finger_center(view_arm)
        layout = getattr(self, "camera_layout", None) or DEFAULT_CAMERA_LAYOUT
        position, _forward = camera_pose_for_view(
            base_mode,
            finger_center_world=target,
            view_arm=view_arm,
            camera_bounds=getattr(self, "camera_bounds", DEFAULT_CAMERA_BOUNDS),
            layout=layout,
        )
        up_hint = layout.view(base_mode).up_hint_world
        pose = look_at_pose(
            position,
            target,
            up_hint=np.asarray(up_hint, dtype=np.float64),
        )
        self._set_pose(pose)

    @staticmethod
    def _view_target(active_arm: str, mode: str | None) -> tuple[str, str | None]:
        if mode is None:
            return active_arm, mode
        for arm in ("left", "right", "both"):
            prefix = arm + "_"
            if mode.startswith(prefix):
                return arm, mode.removeprefix(prefix)
        return active_arm, mode

    def _finger_center(self, active_arm: str) -> np.ndarray:
        if active_arm == "both":
            return (self._finger_center("left") + self._finger_center("right")) / 2.0
        robot = getattr(self.task_env, "robot", None)
        if robot is not None:
            if active_arm == "left":
                entity = getattr(robot, "left_entity", None)
                names = ("fl_link7", "fl_link8")
            else:
                entity = getattr(robot, "right_entity", None)
                names = ("fr_link7", "fr_link8")
            if entity is not None:
                points = []
                for name in names:
                    link = entity.find_link_by_name(name)
                    if link is None:
                        break
                    pose = getattr(link, "entity_pose", None)
                    if pose is None and hasattr(link, "get_pose"):
                        pose = link.get_pose()
                    if pose is None:
                        break
                    points.append(np.asarray(pose.p, dtype=np.float64))
                if len(points) == 2:
                    return (points[0] + points[1]) / 2.0
        return np.asarray(self.task_env.get_arm_pose(active_arm)[:3], dtype=np.float64)

    def _move_camera(self, action: ActionSpec) -> None:
        frame = self.frame()
        direction = action.direction
        if direction == "zoom_in":
            delta = frame.forward * 0.08
        elif direction == "zoom_out":
            delta = -frame.forward * 0.08
        else:
            assert action.scale is not None
            step = CAMERA_TRANSLATION_SCALES[action.scale]
            if direction == "move_left":
                delta = frame.left * step
            elif direction == "move_right":
                delta = -frame.left * step
            elif direction == "move_up":
                delta = frame.up * step
            elif direction == "move_down":
                delta = -frame.up * step
            else:
                raise ValueError(f"Unsupported camera move direction: {direction}")
        position = self._clamp_position(frame.position + delta)
        self._set_pose(pose_from_frame(position, frame.forward, frame.left))

    def _rotate_camera(self, action: ActionSpec) -> None:
        assert action.scale is not None
        angle = np.deg2rad(CAMERA_ROTATION_SCALES_DEG[action.scale])
        frame = self.frame()
        if action.direction == "yaw_left":
            axis, signed_angle = np.array([0.0, 0.0, 1.0]), angle
        elif action.direction == "yaw_right":
            axis, signed_angle = np.array([0.0, 0.0, 1.0]), -angle
        elif action.direction == "pitch_up":
            axis, signed_angle = frame.left, angle
        elif action.direction == "pitch_down":
            axis, signed_angle = frame.left, -angle
        else:
            raise ValueError(f"Unsupported camera rotate direction: {action.direction}")
        forward = rotate_vector(frame.forward, axis, signed_angle)
        left = rotate_vector(frame.left, axis, signed_angle)
        self._set_pose(pose_from_frame(frame.position, forward, left))

    def _set_pose(self, pose: sapien.Pose) -> None:
        self.get_camera().entity.set_pose(pose)

    def _clamp_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, self.camera_bounds[0], self.camera_bounds[1])
