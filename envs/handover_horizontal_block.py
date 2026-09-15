from ._dual_arm_task import DualArmTask
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *


class handover_horizontal_block(DualArmTask):
    VERTICAL_COS_THRESHOLD = float(np.cos(np.deg2rad(20.0)))

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.05, 0.0],
            zlim=[0.757],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
        )
        while abs(rand_pos.p[0]) < 0.15:
            rand_pos = rand_pose(
                xlim=[-0.2, 0.2],
                ylim=[-0.05, 0.0],
                zlim=[0.757],
                qpos=[0.707, 0.707, 0, 0],
                rotate_rand=False,
            )

        self.block = create_box(
            scene=self,
            pose=rand_pos,
            half_size=(0.015, 0.015, 0.1),
            color=(1, 0, 0),
            name="handover_horizontal_block",
            boxtype="long",
        )

        self.add_prohibit_area(self.block, padding=0.07)
        # Functional point 0 uses a local Rx(pi) frame. This target therefore
        # presents the block's long local-Z axis vertically in world space.
        self.handover_middle_pose = [0, -0.05, 0.9, 0, 1, 0, 0]
        self.grasp_arm_tag = ArmTag("right" if self.block.get_pose().p[0] > 0 else "left")
        self.handover_arm_tag = self.grasp_arm_tag.opposite
        self.block_start_height = float(self.block.get_pose().p[2])
        self.peak_object_lift_mm = 0.0
        self.giver_grasp_observed = False
        self.vertical_presentation_observed = False
        self.receiver_grasp_observed = False
        self.handover_completed = False

    def play_once(self):
        grasp_arm_tag = self.grasp_arm_tag
        handover_arm_tag = self.handover_arm_tag

        self.move(
            self.grasp_actor(
                self.block,
                arm_tag=grasp_arm_tag,
                contact_point_id=[3],
                pre_grasp_dis=0.1,
            )
        )
        self.move(
            self.move_by_displacement(
                grasp_arm_tag,
                z=0.12,
                quat=(
                    GRASP_DIRECTION_DIC["front_right"]
                    if grasp_arm_tag == "left"
                    else GRASP_DIRECTION_DIC["front_left"]
                ),
                move_axis="arm",
            )
        )

        self.move(
            self.place_actor(
                self.block,
                arm_tag=grasp_arm_tag,
                target_pose=self.handover_middle_pose,
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="free",
            )
        )
        self.move(
            self.grasp_actor(
                self.block,
                arm_tag=handover_arm_tag,
                contact_point_id=[4, 5, 6, 7],
                pre_grasp_dis=0.1,
            )
        )
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(
            self.move_by_displacement(grasp_arm_tag, z=0.07, move_axis="arm"),
            self.move_by_displacement(
                handover_arm_tag,
                x=0.05 if handover_arm_tag == "right" else -0.05,
            ),
        )

        self.info["info"] = {
            "{A}": "red horizontal block",
            "{a}": str(grasp_arm_tag),
            "{b}": str(handover_arm_tag),
        }
        return self.info

    def move(self, *args, **kwargs):
        result = super().move(*args, **kwargs)
        if hasattr(self, "block"):
            self._update_handover_state()
        return result

    def check_success(self):
        self._update_handover_state()
        block_pose = self.block.get_pose().p
        receiver_closed = (
            self.is_left_gripper_close()
            if self.handover_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        giver_open = (
            self.is_left_gripper_open()
            if self.grasp_arm_tag == "left"
            else self.is_right_gripper_open()
        )
        on_receiver_side = (
            block_pose[0] < 0
            if self.handover_arm_tag == "left"
            else block_pose[0] > 0
        )
        return (
            self.handover_completed
            and self.vertical_presentation_observed
            and receiver_closed
            and giver_open
            and block_pose[2] > 0.92
            and on_receiver_side
            and self._is_block_vertical()
        )

    def _update_handover_state(self):
        block_height = float(self.block.get_pose().p[2])
        self.peak_object_lift_mm = max(
            self.peak_object_lift_mm,
            (block_height - self.block_start_height) * 1000.0,
        )
        giver_contact = self._arm_has_block_contact(str(self.grasp_arm_tag))
        receiver_contact = self._arm_has_block_contact(str(self.handover_arm_tag))
        giver_closed = (
            self.is_left_gripper_close()
            if self.grasp_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        receiver_closed = (
            self.is_left_gripper_close()
            if self.handover_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        giver_open = (
            self.is_left_gripper_open()
            if self.grasp_arm_tag == "left"
            else self.is_right_gripper_open()
        )
        if giver_closed and giver_contact:
            self.giver_grasp_observed = True
        if (
            self.giver_grasp_observed
            and giver_closed
            and giver_contact
            and self.peak_object_lift_mm >= 50.0
            and self._is_block_vertical()
        ):
            self.vertical_presentation_observed = True
        if (
            self.vertical_presentation_observed
            and receiver_closed
            and receiver_contact
        ):
            self.receiver_grasp_observed = True
        if self.receiver_grasp_observed and giver_open and receiver_contact:
            self.handover_completed = True

    def _is_block_vertical(self):
        rotation = self.block.get_pose().to_transformation_matrix()[:3, :3]
        long_axis = rotation[:, 2]
        return abs(float(long_axis[2])) >= self.VERTICAL_COS_THRESHOLD

    def _arm_has_block_contact(self, arm):
        gripper_links = set(getattr(self.robot, f"{arm}_fix_gripper_name", []))
        for joint, _, _ in getattr(self.robot, f"{arm}_gripper", []):
            gripper_links.add(joint.child_link.get_name())
        for contact in self.scene.get_contacts():
            names = {
                contact.bodies[0].entity.name,
                contact.bodies[1].entity.name,
            }
            if self.block.get_name() in names and names.intersection(gripper_links):
                return True
        return False
