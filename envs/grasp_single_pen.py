from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class grasp_single_pen(SingleRightArmTask):
    """Single-right-arm pen grasp scene for active spatial benchmark tests."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        self.pen_id = 0
        self.pen = rand_create_actor(
            self,
            xlim=[0.18, 0.26],
            ylim=[-0.20, -0.10],
            zlim=[0.758],
            modelname="058_markpen",
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 6],
            qpos=[1, 0, 0, 0],
            convex=True,
            model_id=self.pen_id,
        )
        self.pen.set_mass(0.01)
        self.pen_start_height = float(self.pen.get_pose().p[2])
        self.add_prohibit_area(self.pen, padding=0.08)
        self.delay(4)

    def play_once(self):
        self.move(self.grasp_actor(self.pen, arm_tag=self.arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.info["info"] = {
            "{A}": f"058_markpen/base{self.pen_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        pen_height = float(self.pen.get_pose().p[2])
        lifted = pen_height - self.pen_start_height > 0.05
        gripper_closed = self.is_right_gripper_close()
        return lifted and gripper_closed
