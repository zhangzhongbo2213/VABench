from ._single_right_arm_task import SingleRightArmTask
from .utils import *


class click_bell_right(SingleRightArmTask):
    """Single-right-arm bell pressing task for active visual evaluation."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("right")
        bell_pose = rand_pose(
            xlim=[0.12, 0.26],
            ylim=[-0.18, -0.04],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.bell_id = int(np.random.choice([0, 1]))
        self.bell = create_actor(
            scene=self,
            pose=bell_pose,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.delay(4)

    def play_once(self):
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=self.arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.10,
                contact_point_id=0,
            )
        )
        self.move(self.move_by_displacement(self.arm_tag, z=-0.045))
        self.check_success()
        self.move(self.move_by_displacement(self.arm_tag, z=0.045))

        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if self.stage_success_tag:
            return True
        if not self.is_right_gripper_close():
            return False

        target_position = self.bell.get_contact_point(0)[:3]
        contact_positions = self.get_gripper_actor_contact_position(self.bell.get_name())
        for position in contact_positions:
            xy_aligned = np.all(np.abs(position[:2] - target_position[:2]) < 0.025)
            z_aligned = abs(position[2] - target_position[2]) < 0.03
            if xy_aligned and z_aligned:
                self.stage_success_tag = True
                return True
        return False
