from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class grasp_single_bottle(SingleRightArmTask):
    """Single-right-arm bottle grasp scene for active spatial benchmark tests."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        self.bottle_id = 13
        self.bottle = rand_create_actor(
            self,
            xlim=[0.20, 0.28],
            ylim=[-0.16, -0.04],
            zlim=[0.785],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0, 0, 1, 0],
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.01)
        self.bottle_start_height = float(self.bottle.get_pose().p[2])
        self.add_prohibit_area(self.bottle, padding=0.08)
        self.delay(4)

    def play_once(self):
        self.move(self.grasp_actor(self.bottle, arm_tag=self.arm_tag, pre_grasp_dis=0.10))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        bottle_height = float(self.bottle.get_pose().p[2])
        lifted = bottle_height - self.bottle_start_height > 0.05
        gripper_closed = self.is_right_gripper_close()
        return lifted and gripper_closed
