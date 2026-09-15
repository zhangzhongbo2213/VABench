"""Convert sparse grasp frames into RoboTwin gripper target poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import transforms3d as t3d

from .grasp_candidates import GraspCandidate
from .spatial_graph import unit


@dataclass(frozen=True)
class RobotGraspPoses:
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    rotation_world: np.ndarray
    approach_axis_world: np.ndarray
    closing_axis_world: np.ndarray
    control_to_grasp_center_m: float
    pregrasp_distance_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pregrasp_pose": np.round(self.pregrasp_pose, 8).tolist(),
            "grasp_pose": np.round(self.grasp_pose, 8).tolist(),
            "rotation_world": np.round(self.rotation_world, 8).tolist(),
            "approach_axis_world": np.round(
                self.approach_axis_world, 8
            ).tolist(),
            "closing_axis_world": np.round(self.closing_axis_world, 8).tolist(),
            "control_to_grasp_center_m": self.control_to_grasp_center_m,
            "pregrasp_distance_m": self.pregrasp_distance_m,
        }


def candidate_to_robot_grasp_poses(
    candidate: GraspCandidate,
    *,
    control_to_grasp_center_m: float = 0.12,
    pregrasp_distance_m: float | None = None,
) -> RobotGraspPoses:
    """Map a candidate frame to RoboTwin's `[xyz, wxyz]` gripper poses.

    RoboTwin's gripper target local X points along the approach direction and
    local Y lies on the finger closing axis.  Its commanded control point is
    behind the desired grasp center along local X.
    """

    if control_to_grasp_center_m <= 0.0:
        raise ValueError("control_to_grasp_center_m must be positive")
    approach = unit(np.asarray(candidate.approach_axis_world, dtype=np.float64))
    closing_raw = np.asarray(candidate.closing_axis_world, dtype=np.float64)
    closing = closing_raw - float(np.dot(closing_raw, approach)) * approach
    if float(np.linalg.norm(closing)) <= 1e-8:
        raise ValueError("candidate closing axis must not be parallel to approach axis")
    closing = unit(closing)
    transverse = unit(np.cross(approach, closing))
    rotation = np.column_stack([approach, closing, transverse])
    if float(np.linalg.det(rotation)) < 0.999:
        raise ValueError("candidate axes did not produce a right-handed rotation")
    center = np.asarray(candidate.center_world_m, dtype=np.float64)
    if pregrasp_distance_m is None:
        candidate_pregrasp = np.asarray(
            candidate.pregrasp_center_world_m, dtype=np.float64
        )
        pregrasp_distance_m = float(
            np.dot(center - candidate_pregrasp, approach)
        )
    if pregrasp_distance_m < -1e-8:
        raise ValueError("candidate pregrasp center lies past the grasp center")
    pregrasp_distance_m = max(0.0, float(pregrasp_distance_m))
    grasp_position = center - approach * control_to_grasp_center_m
    pregrasp_position = grasp_position - approach * pregrasp_distance_m
    quaternion_wxyz = t3d.quaternions.mat2quat(rotation)
    grasp_pose = np.concatenate([grasp_position, quaternion_wxyz])
    pregrasp_pose = np.concatenate([pregrasp_position, quaternion_wxyz])
    return RobotGraspPoses(
        pregrasp_pose=pregrasp_pose,
        grasp_pose=grasp_pose,
        rotation_world=rotation,
        approach_axis_world=approach,
        closing_axis_world=closing,
        control_to_grasp_center_m=float(control_to_grasp_center_m),
        pregrasp_distance_m=float(pregrasp_distance_m),
    )
