import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class grasp_pen_leaning_cube(SingleRightArmTask):
    """Grasp a pen resting diagonally between the table and a support cube."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")

        yaw = np.random.uniform(-np.pi / 12, np.pi / 12)
        tilt = np.deg2rad(30.0)
        group_x = np.random.uniform(0.16, 0.22)
        lower_y = np.random.uniform(-0.24, -0.20)

        yaw_quat = t3d.euler.euler2quat(0.0, 0.0, yaw)
        tilt_quat = t3d.euler.euler2quat(tilt, 0.0, 0.0)
        pen_quat = t3d.quaternions.qmult(yaw_quat, tilt_quat)

        support_direction = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        cube_half_size = 0.04
        support_distance = 0.177
        cube_xy = np.array([group_x, lower_y]) + support_direction[:2] * support_distance

        self.support_cube = create_box(
            scene=self,
            pose=sapien.Pose(
                [cube_xy[0], cube_xy[1], 0.74 + cube_half_size],
                yaw_quat,
            ),
            half_size=(cube_half_size, cube_half_size, cube_half_size),
            color=(0.12, 0.35, 0.85),
            is_static=True,
            name="pen_support_cube",
        )

        self.pen_id = 0
        self.pen = rand_create_actor(
            self,
            xlim=[group_x],
            ylim=[lower_y],
            zlim=[0.756],
            modelname="058_markpen",
            rotate_rand=False,
            qpos=pen_quat,
            convex=True,
            model_id=self.pen_id,
        )
        self.pen.set_mass(0.01)

        self.add_prohibit_area(self.support_cube, padding=0.04)
        self.add_prohibit_area(self.pen, padding=0.08)
        self.delay(4)

        self.pen_start_height = float(self.pen.get_pose().p[2])

    def play_once(self):
        self.move(self.grasp_actor(self.pen, arm_tag=self.arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.info["info"] = {
            "{A}": f"058_markpen/base{self.pen_id}",
            "{B}": "blue support cube",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        pen_height = float(self.pen.get_pose().p[2])
        lifted = pen_height - self.pen_start_height > 0.05
        return lifted and self.is_right_gripper_close()
