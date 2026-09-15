from __future__ import annotations

import unittest

from agent.protocol import parse_decision
from agent.robotwin.validator import allowed_actions_for_model, is_discrete_gripper_motion, validate_decision


SAFE_TOPDOWN_GEOMETRY = {"grip_center_angle_to_table_down_deg": 12.0}
UNSAFE_SIDE_GEOMETRY = {"grip_center_angle_to_table_down_deg": 78.0, "grip_center_direction_world": [1.0, 0.0, 0.0]}


class ValidatorTest(unittest.TestCase):
    def test_discrete_action_must_be_available(self) -> None:
        decision = parse_decision('{"action":"camera.view_topdown"}')

        self.assertIsNone(
            validate_decision(
                decision,
                available_actions={"camera.view_topdown"},
                task_name="grasp_single_bottle",
                geometry=UNSAFE_SIDE_GEOMETRY,
            )
        )
        self.assertIn(
            "camera actions are disabled",
            validate_decision(
                decision,
                available_actions=set(),
                task_name="grasp_single_bottle",
                geometry=UNSAFE_SIDE_GEOMETRY,
            )
            or "",
        )

    def test_parameterized_action_range_is_checked(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.move_world","axis":"x","sign":"+","distance_mm":120,"reason":"too far"}'
        )

        self.assertIn(
            "[1, 100]",
            validate_decision(
                decision,
                available_actions=set(),
                task_name="upright_bottle",
                geometry=SAFE_TOPDOWN_GEOMETRY,
            )
            or "",
        )

    def test_discrete_gripper_motion_is_rejected(self) -> None:
        decision = parse_decision('{"action":"gripper.world_x_pos.small","reason":"old discrete move"}')

        self.assertIn(
            "numeric JSON",
            validate_decision(
                decision,
                available_actions={"gripper.world_x_pos.small"},
                task_name="any_task",
                geometry=SAFE_TOPDOWN_GEOMETRY,
            )
            or "",
        )

    def test_discrete_gripper_motion_detector(self) -> None:
        self.assertTrue(is_discrete_gripper_motion("gripper.world_x_pos.small"))
        self.assertTrue(is_discrete_gripper_motion("gripper.rotate_ry_cw.xlarge"))
        self.assertTrue(is_discrete_gripper_motion("gripper.depth_forward.medium"))
        self.assertTrue(is_discrete_gripper_motion("left_gripper.world_x_pos.small"))
        self.assertFalse(is_discrete_gripper_motion("gripper.open"))
        self.assertFalse(is_discrete_gripper_motion("camera.move_left.small"))
        self.assertFalse(is_discrete_gripper_motion("gripper.world_x_pos.20mm"))

    def test_dual_mode_requires_explicit_arm_for_single_gripper_numeric_action(self) -> None:
        missing = parse_decision(
            '{"action":"gripper.move_world","axis":"x","sign":"+","distance_mm":10}'
        )
        explicit = parse_decision(
            '{"action":"gripper.move_world","arm":"left","axis":"x","sign":"+","distance_mm":10}'
        )
        geometry = {"active_arm": "both", "left": {}, "right": {}}

        self.assertIn(
            "requires arm",
            validate_decision(missing, available_actions=set(), task_name="place_shoe", geometry=geometry) or "",
        )
        self.assertIsNone(
            validate_decision(explicit, available_actions=set(), task_name="place_shoe", geometry=geometry)
        )

    def test_generic_profile_does_not_block_task_specific_geometry(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.move_world","axis":"z","sign":"-","distance_mm":20,"reason":"descend"}'
        )

        self.assertIsNone(
            validate_decision(
                decision,
                available_actions=set(),
                task_name="grasp_single_bottle",
                geometry=UNSAFE_SIDE_GEOMETRY,
            )
        )

    def test_no_task_specific_geometry_gate_for_close(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.close","reason":"model judges visual evidence itself"}'
        )

        self.assertIsNone(
            validate_decision(
                decision,
                available_actions={"gripper.close"},
                task_name="grasp_single_bottle",
                geometry=UNSAFE_SIDE_GEOMETRY,
            )
        )

    def test_task_specific_local_constraint_profile_is_unsupported(self) -> None:
        decision = parse_decision('{"action":"gripper.close","reason":"close"}')

        self.assertIn(
            "unsupported local constraint profile",
            validate_decision(
                decision,
                available_actions={"gripper.close"},
                task_name="grasp_single_bottle",
                geometry=UNSAFE_SIDE_GEOMETRY,
                constraint_profile="horizontal_bottle_topdown",
            )
            or "",
        )

    def test_fine_camera_policy_removes_coarse_views(self) -> None:
        actions = [
            "camera.view_topdown",
            "camera.move_left.small",
            "camera.move_left.medium",
            "camera.look_at_gripper",
            "gripper.world_x_pos.small",
            "gripper.rotate_rx_ccw.large",
            "gripper.open",
        ]

        self.assertEqual(
            allowed_actions_for_model(actions, "fine"),
            ["camera.view_topdown", "camera.move_left.small", "gripper.open"],
        )

    def test_full_camera_policy_still_removes_discrete_gripper_motion(self) -> None:
        actions = [
            "camera.move_left.medium",
            "gripper.world_x_pos.small",
            "gripper.rotate_rx_ccw.large",
            "gripper.open",
        ]

        self.assertEqual(
            allowed_actions_for_model(actions, "full"),
            ["camera.move_left.medium", "gripper.open"],
        )


if __name__ == "__main__":
    unittest.main()
