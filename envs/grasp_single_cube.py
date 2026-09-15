from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class grasp_single_cube(SingleRightArmTask):
    """Single-right-arm cube grasp scene for active spatial benchmark tests."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        self.cube_half_size = 0.025
        cube_pose = rand_pose(
            xlim=[0.18, 0.26],
            ylim=[-0.16, -0.06],
            zlim=[0.741 + self.cube_half_size],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
        )
        self.cube = create_box(
            scene=self,
            pose=cube_pose,
            half_size=(self.cube_half_size, self.cube_half_size, self.cube_half_size),
            color=(0.9, 0.1, 0.1),
            name="cube",
        )
        self.cube.set_mass(0.01)
        self.cube_start_height = float(self.cube.get_pose().p[2])
        self.add_prohibit_area(self.cube, padding=0.08)
        self.delay(4)

    def play_once(self):
        self.move(self.grasp_actor(self.cube, arm_tag=self.arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
        self.info["info"] = {
            "{A}": "red cube",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        cube_height = float(self.cube.get_pose().p[2])
        lifted = cube_height - self.cube_start_height > 0.05
        gripper_closed = self.is_right_gripper_close()
        return lifted and gripper_closed
