import sapien

from ._base_task import Base_Task


class SingleRightArmTask(Base_Task):
    """RoboTwin base task with the unused left arm and camera stand hidden."""

    _REMOVED_ARM_LINK_NAMES = {
        "fl_base_link",
        "fl_link1",
        "fl_link2",
        "fl_link3",
        "fl_link4",
        "fl_link5",
        "fl_link6",
        "fl_link7",
        "fl_link8",
        "left_camera",
    }
    _HIDDEN_CAMERA_STAND_LINK_NAMES = {
        "camera_base_link",
        "camera_link1",
        "camera_link2",
    }

    def load_robot(self, **kwargs):
        super().load_robot(**kwargs)
        self.removed_arm_links = []

        for link in self.robot.left_entity.get_links():
            link_name = link.get_name()
            if link_name not in self._REMOVED_ARM_LINK_NAMES:
                continue
            self._hide_link_visuals(link)
            for shape in link.get_collision_shapes():
                shape.set_collision_groups([0, 0, 0, 0])
            self.removed_arm_links.append(link_name)

        self.hidden_camera_stand_links = []
        for link in self.robot.left_entity.get_links():
            link_name = link.get_name()
            if link_name not in self._HIDDEN_CAMERA_STAND_LINK_NAMES:
                continue
            self._hide_link_visuals(link)
            self.hidden_camera_stand_links.append(link_name)

    @staticmethod
    def _hide_link_visuals(link):
        for component in link.entity.components:
            if isinstance(component, sapien.render.RenderBodyComponent):
                component.visibility = 0.0
                component.disable()
