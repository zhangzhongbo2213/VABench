import sapien

from ._base_task import Base_Task


class DualArmTask(Base_Task):
    """RoboTwin dual-arm base task with the central camera stand hidden visually."""

    _HIDDEN_CAMERA_STAND_LINK_NAMES = {
        "camera_base_link",
        "camera_link1",
        "camera_link2",
    }

    def load_robot(self, **kwargs):
        super().load_robot(**kwargs)
        self.hidden_camera_stand_links = []
        for entity in (self.robot.left_entity, self.robot.right_entity):
            for link in entity.get_links():
                link_name = link.get_name()
                if link_name not in self._HIDDEN_CAMERA_STAND_LINK_NAMES:
                    continue
                self._hide_link_visuals(link)
                if link_name not in self.hidden_camera_stand_links:
                    self.hidden_camera_stand_links.append(link_name)

    @staticmethod
    def _hide_link_visuals(link):
        for component in link.entity.components:
            if isinstance(component, sapien.render.RenderBodyComponent):
                component.visibility = 0.0
                component.disable()
