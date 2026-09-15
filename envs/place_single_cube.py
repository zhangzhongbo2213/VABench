from ._single_right_arm_place_target_task import SingleRightArmPlaceTargetTask
from .utils import *


class place_single_cube(SingleRightArmPlaceTargetTask):
    """Pick up a cube with the right arm and place it on a target pad."""

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
            half_size=(self.cube_half_size,) * 3,
            color=(0.9, 0.1, 0.1),
            name="cube",
        )
        self.cube.set_mass(0.01)
        self.configure_place_target(
            actor=self.cube,
            actor_description="red cube",
            target_half_size=(0.06, 0.06),
            target_xy_tolerance=(0.035, 0.035),
            pre_grasp_distance=0.09,
            target_xlim=(0.10, 0.14),
            target_ylim=(0.00, 0.04),
        )
