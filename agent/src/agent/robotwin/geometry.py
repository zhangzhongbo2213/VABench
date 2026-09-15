from __future__ import annotations

from typing import Any


def gripper_geometry_snapshot(env, modules: dict[str, Any]) -> dict[str, Any]:
    if getattr(env, "active_arm", "right") == "both":
        return {
            "active_arm": "both",
            "left": _single_gripper_geometry_snapshot(env, modules, "left"),
            "right": _single_gripper_geometry_snapshot(env, modules, "right"),
        }
    return _single_gripper_geometry_snapshot(env, modules, getattr(env, "active_arm", "right"))


def _single_gripper_geometry_snapshot(env, modules: dict[str, Any], arm: str) -> dict[str, Any]:
    import numpy as np
    import transforms3d as t3d

    control_pose = np.asarray(modules["gripper_pose"](env, arm), dtype=np.float64)
    center_pose = gripper_center_pose(env, modules, arm)
    control_xyz = np.asarray(control_pose[:3], dtype=np.float64)
    center_xyz = np.asarray(center_pose[:3], dtype=np.float64)
    quat = np.asarray(control_pose[3:], dtype=np.float64)
    rot = t3d.quaternions.quat2mat(quat)
    finger_center_vector = center_xyz - control_xyz
    norm = float(np.linalg.norm(finger_center_vector))
    if norm < 1e-6:
        approach_dir = rot[:, 0]
        source = "local_rx_fallback"
    else:
        approach_dir = finger_center_vector / norm
        source = "finger_center_minus_eepose"
    axes = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    angles = {name: vector_angle_deg(approach_dir, axis) for name, axis in axes.items()}
    rpy = t3d.euler.quat2euler(quat, axes="sxyz")
    return {
        "control_eepose_xyz": rounded_list(control_xyz, 4),
        "finger_center_xyz": rounded_list(center_xyz, 4),
        "control_eepose_quat_wxyz": rounded_list(quat, 5),
        "control_eepose_rpy_deg": rounded_list(np.degrees(rpy), 2),
        "local_axes_world": {
            "rx": rounded_list(rot[:, 0], 4),
            "ry": rounded_list(rot[:, 1], 4),
            "rz": rounded_list(rot[:, 2], 4),
        },
        "grip_center_direction_world": rounded_list(approach_dir, 4),
        "grip_center_direction_source": source,
        "grip_center_length_m": round(norm, 4),
        "grip_center_angle_to_world_axes_deg": {name: round(angle, 1) for name, angle in angles.items()},
        "grip_center_angle_to_table_down_deg": round(angles["-Z"], 1),
        "topdown_posture_ok_30deg": bool(angles["-Z"] <= 30.0),
        "topdown_posture_ok_45deg": bool(angles["-Z"] <= 45.0),
    }

def gripper_center_pose(env, modules: dict[str, Any], arm: str | None = None):
    import numpy as np

    resolved_arm = arm or getattr(env, "active_arm", "right")
    control_pose = np.asarray(modules["gripper_pose"](env, resolved_arm), dtype=np.float64).copy()
    midpoint = gripper_finger_link_midpoint(env, resolved_arm)
    if midpoint is not None:
        control_pose[:3] = midpoint
    return control_pose


def gripper_finger_link_midpoint(env, arm: str | None = None):
    import numpy as np

    robot = getattr(getattr(env, "task", None), "robot", None)
    if robot is None:
        return None
    resolved_arm = arm or getattr(env, "active_arm", "right")
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


def vector_angle_deg(a, b) -> float:
    import math
    import numpy as np

    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-12:
        return 180.0
    cos_value = max(-1.0, min(1.0, float(np.dot(av, bv) / denom)))
    return math.degrees(math.acos(cos_value))


def rounded_list(values, ndigits: int) -> list[float]:
    return [round(float(value), ndigits) for value in values]
