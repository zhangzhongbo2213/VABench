from ._single_right_arm_place_target_task import SingleRightArmPlaceTargetTask
from .utils import *


class place_single_bottle_upright(SingleRightArmPlaceTargetTask):
    """Pick up an upright bottle with the right arm and place it on a target pad."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        self.bottle_id = 13
        self.bottle = rand_create_actor(
            self,
            xlim=[0.20, 0.28],
            ylim=[-0.16, -0.04],
            zlim=[0.84],
            modelname="001_bottle",
            rotate_rand=False,
            qpos=[0.70710678, 0.70710678, 0.0, 0.0],
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.01)
        self.configure_place_target(
            actor=self.bottle,
            actor_description=f"001_bottle/base{self.bottle_id}",
            target_half_size=(0.065, 0.065),
            target_xy_tolerance=(0.04, 0.04),
            pre_grasp_distance=0.10,
        )
