import numpy as np

from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class SingleRightArmPlaceTargetTask(SingleRightArmTask):
    """Shared physical target-pad placement behavior for single-right-arm tasks."""

    target_color_name = "green target area"
    target_color = (0.08, 0.72, 0.28)
    target_half_height = 0.003
    required_lift_height = 0.05
    max_horizontal_step = 0.06
    max_horizontal_waypoints = 12

    def configure_place_target(
        self,
        *,
        actor,
        actor_description: str,
        target_half_size: tuple[float, float],
        target_xy_tolerance: tuple[float, float],
        pre_grasp_distance: float,
        target_xlim: tuple[float, float] = (0.03, 0.09),
        target_ylim: tuple[float, float] = (0.07, 0.13),
        source_padding: float = 0.08,
    ) -> None:
        target_pose = rand_pose(
            xlim=list(target_xlim),
            ylim=list(target_ylim),
            zlim=[0.741 + self.target_half_height],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.target_pad = create_box(
            scene=self,
            pose=target_pose,
            half_size=(*target_half_size, self.target_half_height),
            color=self.target_color,
            name="target_pad",
            is_static=True,
        )
        self.place_actor_object = actor
        self.place_actor_description = actor_description
        self.target_half_size = np.asarray(target_half_size, dtype=np.float64)
        self.target_xy_tolerance = np.asarray(target_xy_tolerance, dtype=np.float64)
        self.pre_grasp_distance = float(pre_grasp_distance)

        self.add_prohibit_area(actor, padding=source_padding)
        self.add_prohibit_area(self.target_pad, padding=0.02)
        self.delay(4)

        self.object_start_xyz = np.asarray(actor.get_pose().p, dtype=np.float64).copy()
        self.target_start_xyz = np.asarray(
            self.target_pad.get_pose().p,
            dtype=np.float64,
        ).copy()
        table_surface_z = 0.741 + self.table_z_bias
        self.object_support_offset = float(self.object_start_xyz[2] - table_surface_z)
        self.peak_object_lift_mm = 0.0
        self.was_lifted = False

    def play_once(self):
        actor = self.place_actor_object
        self.move(
            self.grasp_actor(
                actor,
                arm_tag=self.arm_tag,
                pre_grasp_dis=self.pre_grasp_distance,
            )
        )
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self._update_lift_state()

        target_xyz = np.asarray(self.target_pad.get_pose().p, dtype=np.float64)
        self._move_actor_horizontally_to(target_xyz[:2])

        actor_xyz = np.asarray(actor.get_pose().p, dtype=np.float64)
        target_top_z = float(target_xyz[2] + self.target_half_height)
        desired_actor_z = target_top_z + self.object_support_offset
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag,
                z=desired_actor_z - float(actor_xyz[2]),
            )
        )
        self.move(self.open_gripper(arm_tag=self.arm_tag))
        self.delay(2)
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.10))
        self.delay(1)

        self.info["info"] = {
            "{A}": self.place_actor_description,
            "{B}": self.target_color_name,
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        self._update_lift_state()
        actor_xyz = np.asarray(self.place_actor_object.get_pose().p, dtype=np.float64)
        target_xyz = np.asarray(self.target_pad.get_pose().p, dtype=np.float64)
        target_top_z = float(target_xyz[2] + self.target_half_height)
        desired_actor_z = target_top_z + self.object_support_offset

        centered = bool(
            np.all(np.abs(actor_xyz[:2] - target_xyz[:2]) <= self.target_xy_tolerance)
        )
        supported_height = abs(float(actor_xyz[2]) - desired_actor_z) <= 0.035
        target_undisturbed = bool(
            np.linalg.norm(target_xyz - self.target_start_xyz) <= 0.002
        )
        on_target = self.check_actors_contact(
            self.place_actor_object.get_name(),
            self.target_pad.get_name(),
        )
        return bool(
            self.was_lifted
            and centered
            and supported_height
            and target_undisturbed
            and on_target
            and self.is_right_gripper_open()
        )

    def _update_lift_state(self) -> None:
        current_z = float(self.place_actor_object.get_pose().p[2])
        lift_m = current_z - float(self.object_start_xyz[2])
        self.peak_object_lift_mm = max(self.peak_object_lift_mm, lift_m * 1000.0)
        if lift_m >= self.required_lift_height:
            self.was_lifted = True

    def _move_actor_horizontally_to(self, target_xy: np.ndarray) -> None:
        for _ in range(self.max_horizontal_waypoints):
            if not self.plan_success:
                return
            actor_xy = np.asarray(
                self.place_actor_object.get_pose().p[:2],
                dtype=np.float64,
            )
            remaining = np.asarray(target_xy, dtype=np.float64) - actor_xy
            max_axis_distance = float(np.max(np.abs(remaining)))
            if max_axis_distance <= 0.002:
                return
            scale = min(1.0, self.max_horizontal_step / max_axis_distance)
            step = remaining * scale
            self.move(
                self.move_by_displacement(
                    arm_tag=self.arm_tag,
                    x=float(step[0]),
                    y=float(step[1]),
                )
            )

        actor_xy = np.asarray(
            self.place_actor_object.get_pose().p[:2],
            dtype=np.float64,
        )
        if np.max(np.abs(np.asarray(target_xy, dtype=np.float64) - actor_xy)) > 0.002:
            self.plan_success = False
