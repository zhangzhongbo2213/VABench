import numpy as np

from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class place_cube_on_cube(SingleRightArmTask):
    """Pick up a red cube with the right arm and stack it on a green cube."""

    cube_half_size = 0.025
    target_half_height = cube_half_size
    required_lift_height = 0.05
    min_top_overlap = 0.002
    max_horizontal_step = 0.04
    max_horizontal_waypoints = 12

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        table_surface_z = 0.741 + self.table_z_bias

        stack_pose = rand_pose(
            xlim=[0.20, 0.26],
            ylim=[-0.14, -0.06],
            zlim=[0.741 + self.cube_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.stack_cube = create_box(
            scene=self,
            pose=stack_pose,
            half_size=(self.cube_half_size,) * 3,
            color=(0.90, 0.10, 0.10),
            name="stack_cube",
        )
        self.stack_cube.set_mass(0.01)

        base_pose = rand_pose(
            xlim=[0.10, 0.14],
            ylim=[0.00, 0.04],
            zlim=[0.741 + self.cube_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
        )
        self.base_cube = create_box(
            scene=self,
            pose=base_pose,
            half_size=(self.cube_half_size,) * 3,
            color=(0.08, 0.72, 0.28),
            name="base_cube",
        )
        self.base_cube.set_mass(0.05)

        self.add_prohibit_area(self.stack_cube, padding=0.08)
        self.add_prohibit_area(self.base_cube, padding=0.07)
        self.delay(4)

        self.stack_start_xyz = np.asarray(
            self.stack_cube.get_pose().p,
            dtype=np.float64,
        ).copy()
        self.base_start_xyz = np.asarray(
            self.base_cube.get_pose().p,
            dtype=np.float64,
        ).copy()
        self.stack_support_offset = float(self.stack_start_xyz[2] - table_surface_z)
        self.object_support_offset = self.stack_support_offset
        self.peak_object_lift_mm = 0.0
        self.peak_stack_lift_mm = 0.0
        self.was_lifted = False

    def play_once(self):
        self.move(
            self.grasp_actor(
                self.stack_cube,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.09,
            )
        )
        self.delay(2)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.delay(1)
        self._update_lift_state()

        base_xyz = np.asarray(self.base_cube.get_pose().p, dtype=np.float64)
        self._move_stack_cube_horizontally_to(base_xyz[:2])

        stack_xyz = np.asarray(self.stack_cube.get_pose().p, dtype=np.float64)
        base_xyz = np.asarray(self.base_cube.get_pose().p, dtype=np.float64)
        desired_stack_z = float(base_xyz[2] + 2.0 * self.cube_half_size)
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag,
                z=desired_stack_z - float(stack_xyz[2]),
            )
        )
        self.move(self.open_gripper(arm_tag=self.arm_tag))
        self.delay(3)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.05))
        self.delay(2)

        self.info["info"] = {
            "{A}": "red cube",
            "{B}": "green cube",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        self._update_lift_state()
        stack_xyz = np.asarray(self.stack_cube.get_pose().p, dtype=np.float64)
        base_xyz = np.asarray(self.base_cube.get_pose().p, dtype=np.float64)

        center_delta_xy = np.abs(stack_xyz[:2] - base_xyz[:2])
        top_overlap_xy = 2.0 * self.cube_half_size - center_delta_xy
        top_faces_overlap = bool(np.all(top_overlap_xy >= self.min_top_overlap))
        expected_height = float(base_xyz[2] + 2.0 * self.cube_half_size)
        supported_height = abs(float(stack_xyz[2]) - expected_height) <= 0.012
        base_undisturbed = bool(
            np.linalg.norm(base_xyz - self.base_start_xyz) <= 0.012
        )
        cubes_contact = self.check_actors_contact(
            self.stack_cube.get_name(),
            self.base_cube.get_name(),
        )
        return bool(
            self.was_lifted
            and top_faces_overlap
            and supported_height
            and base_undisturbed
            and cubes_contact
            and self.is_right_gripper_open()
        )

    def _update_lift_state(self):
        current_z = float(self.stack_cube.get_pose().p[2])
        lift_m = current_z - float(self.stack_start_xyz[2])
        self.peak_object_lift_mm = max(self.peak_object_lift_mm, lift_m * 1000.0)
        self.peak_stack_lift_mm = self.peak_object_lift_mm
        if lift_m >= self.required_lift_height:
            self.was_lifted = True

    def _move_stack_cube_horizontally_to(self, target_xy):
        target_xy = np.asarray(target_xy, dtype=np.float64)
        for _ in range(self.max_horizontal_waypoints):
            if not self.plan_success:
                return
            stack_xy = np.asarray(
                self.stack_cube.get_pose().p[:2],
                dtype=np.float64,
            )
            remaining = target_xy - stack_xy
            max_axis_distance = float(np.max(np.abs(remaining)))
            if max_axis_distance <= 0.002:
                return
            step = remaining * min(1.0, self.max_horizontal_step / max_axis_distance)
            self.move(
                self.move_by_displacement(
                    arm_tag=self.arm_tag,
                    x=float(step[0]),
                    y=float(step[1]),
                )
            )
            self.delay(1)

        stack_xy = np.asarray(
            self.stack_cube.get_pose().p[:2],
            dtype=np.float64,
        )
        if np.max(np.abs(target_xy - stack_xy)) > 0.002:
            self.plan_success = False
