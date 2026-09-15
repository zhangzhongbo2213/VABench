from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class beat_block_hammer_right(SingleRightArmTask):
    """Active-perception right-arm tool-use benchmark with a hammer and block."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        hammer_pose = rand_pose(
            xlim=[0.02, 0.06],
            ylim=[-0.10, -0.04],
            zlim=[0.783],
            qpos=[0, 0, 0.995, 0.105],
            rotate_rand=False,
        )
        self.hammer = create_actor(
            scene=self,
            pose=hammer_pose,
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        block_pose = rand_pose(
            xlim=[0.14, 0.22],
            ylim=[0.06, 0.13],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.001)
        self.hammer_start_height = float(self.hammer.get_pose().p[2])
        self.hammer_lift_peak = self.hammer_start_height
        self.hammer_lifted_with_grasp = False
        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])
        self.delay(4)

    def play_once(self):
        self.move(
            self.grasp_actor(
                self.hammer,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.12,
                grasp_dis=0.01,
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, z=0.10, move_axis="world"))
        self._update_tool_progress()
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),
                arm_tag=self.arm_tag,
                functional_point_id=0,
                pre_dis=0.10,
                dis=0,
                is_open=False,
            )
        )
        self.info["info"] = {
            "{A}": "020_hammer/base0",
            "{B}": "red block",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        self._update_tool_progress()
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        aligned = np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps)
        contacted = self.check_actors_contact(self.hammer.get_name(), self.block.get_name())
        return self.hammer_lifted_with_grasp and self.is_right_gripper_close() and aligned and contacted

    def _update_tool_progress(self):
        hammer_height = float(self.hammer.get_pose().p[2])
        self.hammer_lift_peak = max(self.hammer_lift_peak, hammer_height)
        lifted = hammer_height - self.hammer_start_height > 0.04
        grasp_contact = bool(self.get_gripper_actor_contact_position(self.hammer.get_name()))
        if lifted and grasp_contact and self.is_right_gripper_close():
            self.hammer_lifted_with_grasp = True
