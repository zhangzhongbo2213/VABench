from __future__ import annotations

import numpy as np
import transforms3d as t3d


MAX_FIXED_CENTER_ROTATION_WAYPOINT_DEG = 10.0

WORLD_AXES = {
    "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def axis_delta_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("rotation axis vector must be nonzero")
    return t3d.quaternions.quat2mat(t3d.quaternions.axangle2quat(axis / norm, angle_rad))


def world_axis_delta_rotation(axis_name: str, angle_rad: float) -> np.ndarray:
    try:
        axis = WORLD_AXES[axis_name]
    except KeyError as exc:
        raise ValueError(f"Unknown world axis: {axis_name!r}") from exc
    return axis_delta_rotation(axis, angle_rad)


def local_axis_vector(control_pose: np.ndarray, axis_name: str) -> np.ndarray:
    pose = np.asarray(control_pose, dtype=np.float64)
    rot = t3d.quaternions.quat2mat(pose[3:])
    try:
        index = {"X": 0, "Y": 1, "Z": 2}[axis_name]
    except KeyError as exc:
        raise ValueError(f"Unknown local axis: {axis_name!r}") from exc
    return rot[:, index]


def compensated_control_pose_for_fixed_center(
    control_pose: np.ndarray,
    center_position: np.ndarray,
    delta_rot: np.ndarray,
) -> np.ndarray:
    """Return a control pose whose new orientation leaves the finger center fixed.

    `control_pose` is the low-level eepose. `center_position` is the visual
    gripper/finger center used by the overlay. The returned pose changes both
    the control position and orientation so that applying `delta_rot` around a
    fixed world axis keeps the center position approximately unchanged.
    """
    pose = np.asarray(control_pose, dtype=np.float64).copy()
    center = np.asarray(center_position, dtype=np.float64)
    delta = np.asarray(delta_rot, dtype=np.float64)
    rot = t3d.quaternions.quat2mat(pose[3:])
    offset_local = rot.T @ (center - pose[:3])
    new_rot = delta @ rot
    pose[:3] = center - new_rot @ offset_local
    pose[3:] = t3d.quaternions.mat2quat(new_rot)
    return pose


def fixed_center_rotation_waypoints(
    control_pose: np.ndarray,
    center_position: np.ndarray,
    axis_name: str | np.ndarray,
    angle_rad: float,
    max_waypoint_deg: float = MAX_FIXED_CENTER_ROTATION_WAYPOINT_DEG,
) -> list[np.ndarray]:
    if max_waypoint_deg <= 0:
        raise ValueError("max_waypoint_deg must be positive.")
    step_count = max(1, int(np.ceil(abs(np.rad2deg(angle_rad)) / max_waypoint_deg)))
    return [
        compensated_control_pose_for_fixed_center(
            control_pose,
            center_position,
            delta_rotation(axis_name, angle_rad * step_index / step_count),
        )
        for step_index in range(1, step_count + 1)
    ]


def fixed_center_local_rotation_waypoints(
    control_pose: np.ndarray,
    center_position: np.ndarray,
    axis_name: str,
    angle_rad: float,
    max_waypoint_deg: float = MAX_FIXED_CENTER_ROTATION_WAYPOINT_DEG,
) -> list[np.ndarray]:
    return fixed_center_rotation_waypoints(
        control_pose,
        center_position,
        local_axis_vector(control_pose, axis_name),
        angle_rad,
        max_waypoint_deg=max_waypoint_deg,
    )


def delta_rotation(axis_name: str | np.ndarray, angle_rad: float) -> np.ndarray:
    if isinstance(axis_name, str):
        return world_axis_delta_rotation(axis_name, angle_rad)
    return axis_delta_rotation(axis_name, angle_rad)
