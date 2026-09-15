from copy import deepcopy

import numpy as np
import transforms3d as t3d

from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class place_cube_in_bowl(SingleRightArmTask):
    """Single-right-arm scene for placing a cube inside a bowl."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        self.cube_half_size = 0.02

        bowl_pose = rand_pose(
            xlim=[0.10, 0.14],
            ylim=[0.00, 0.04],
            zlim=[0.741],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=3,
            convex=True,
            is_static=True,
        )

        cube_pose = rand_pose(
            xlim=[0.22, 0.27],
            ylim=[-0.17, -0.10],
            zlim=[0.741 + self.cube_half_size],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
            qpos=[1, 0, 0, 0],
        )
        self.cube = create_box(
            scene=self,
            pose=cube_pose,
            half_size=(self.cube_half_size,) * 3,
            color=(0.9, 0.1, 0.1),
            name="cube",
        )
        self.cube.set_mass(0.01)

        self.bowl_start_pose = np.asarray(self.bowl.get_pose().p, dtype=np.float64).copy()
        self.cube_start_pose = np.asarray(self.cube.get_pose().p, dtype=np.float64).copy()
        self.peak_cube_lift_mm = 0.0
        self.add_prohibit_area(self.cube, padding=0.07)
        self.add_prohibit_area(self.bowl, padding=0.05)
        self.delay(4)

    def play_once(self):
        self.move(self.grasp_actor(self.cube, arm_tag=self.arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self._update_peak_cube_lift()

        bowl_center = np.asarray(self.bowl.get_pose().p, dtype=np.float64)
        cube_center = np.asarray(self.cube.get_pose().p, dtype=np.float64)
        horizontal_displacement = bowl_center[:2] - cube_center[:2]
        transfer_position = np.asarray(
            self.get_arm_pose(self.arm_tag)[:3],
            dtype=np.float64,
        )
        transfer_position[:2] += horizontal_displacement
        self._move_right_with_reachable_world_yaw(transfer_position)

        cube_center = np.asarray(self.cube.get_pose().p, dtype=np.float64)
        target_cube_height = float(bowl_center[2] + 0.05)
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag,
                z=target_cube_height - float(cube_center[2]),
            )
        )
        self.move(self.open_gripper(arm_tag=self.arm_tag))
        self.delay(2)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.05))
        self.delay(2)

        self.info["info"] = {
            "{A}": "red cube",
            "{B}": "002_bowl/base3",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def _update_peak_cube_lift(self):
        lift_mm = float(
            (self.cube.get_pose().p[2] - self.cube_start_pose[2]) * 1000.0
        )
        self.peak_cube_lift_mm = max(self.peak_cube_lift_mm, lift_mm)

    def _move_right_with_reachable_world_yaw(self, target_position):
        if not self.plan_success:
            return False

        if self.need_plan:
            current_pose = np.asarray(self.get_arm_pose(self.arm_tag), dtype=np.float64)
            current_rotation = t3d.quaternions.quat2mat(current_pose[3:])
            target_poses = []
            for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
                yaw_rotation = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], angle)
                target_quaternion = t3d.quaternions.mat2quat(
                    yaw_rotation @ current_rotation
                )
                target_poses.append(
                    np.concatenate([target_position, target_quaternion]).tolist()
                )

            batch = self.robot.right_plan_multi_path(target_poses)
            successful = [
                index
                for index, status in enumerate(batch["status"])
                if status == "Success"
            ]
            if not successful:
                self.right_joint_path.append({"status": "Fail"})
                self.plan_success = False
                return False
            selected_index = min(
                successful,
                key=lambda index: len(batch["position"][index]),
            )
            right_result = self.robot.right_plan_path(target_poses[selected_index])
            if right_result["status"] != "Success":
                positions = np.asarray(batch["position"][selected_index])
                velocities = np.asarray(batch["velocity"][selected_index])
                moving_steps = np.flatnonzero(
                    np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-7
                )
                retained_steps = (
                    int(moving_steps[-1]) + 2 if len(moving_steps) else 1
                )
                right_result = {
                    "status": "Success",
                    "position": positions[:retained_steps],
                    "velocity": velocities[:retained_steps],
                }
            self.right_joint_path.append(deepcopy(right_result))
        else:
            right_result = deepcopy(self.right_joint_path[self.right_cnt])
            self.right_cnt += 1

        if right_result["status"] != "Success":
            self.plan_success = False
            return False
        self.take_dense_action(
            {
                "left_arm": None,
                "left_gripper": None,
                "right_arm": right_result,
                "right_gripper": None,
            }
        )
        return True

    def check_success(self):
        self._update_peak_cube_lift()
        cube_position = np.asarray(self.cube.get_pose().p, dtype=np.float64)
        bowl_position = np.asarray(self.bowl.get_pose().p, dtype=np.float64)

        horizontal_offset = float(np.linalg.norm(cube_position[:2] - bowl_position[:2]))
        relative_height = float(cube_position[2] - bowl_position[2])
        bowl_displacement = float(np.linalg.norm(bowl_position - self.bowl_start_pose))
        cube_contacts_bowl = self.check_actors_contact("cube", "002_bowl")

        return (
            self.peak_cube_lift_mm >= 50.0
            and horizontal_offset < 0.055
            and 0.015 < relative_height < 0.095
            and bowl_displacement < 0.005
            and cube_contacts_bowl
            and self.is_right_gripper_open()
        )
