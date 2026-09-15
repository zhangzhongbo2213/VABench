from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import select_safe_eval_seeds as selector  # noqa: E402


class HammerSafetyProfileTests(unittest.TestCase):
    def test_hammer_profile_requires_full_tool_contact(self) -> None:
        profile = selector.safety_profile("beat_block_hammer_right")

        self.assertEqual(profile.actor_attr, "hammer")
        self.assertEqual(profile.contact_actor_attr, "block")
        self.assertEqual(profile.verification_mode, "tool_contact")
        self.assertEqual(profile.required_gripper_state, "closed")
        self.assertEqual(
            profile.verification_method_override,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertGreaterEqual(profile.min_right_arm_plan_segments, 5)

        criteria = selector.selection_criteria(
            profile,
            profile.min_verified_lift_mm,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertIn("retained tool grasp", criteria)
        self.assertIn("tool target contact", criteria)

    def test_hammer_peak_lift_uses_tracked_peak_height(self) -> None:
        class Task:
            hammer_start_height = 0.80
            hammer_lift_peak = 0.905

        self.assertAlmostEqual(
            selector.task_peak_object_lift_mm(Task(), final_object_lift_mm=0.0),
            105.0,
        )


class RemainingTaskSafetyProfileTests(unittest.TestCase):
    def test_leaning_pen_uses_full_right_arm_expert(self) -> None:
        profile = selector.safety_profile("grasp_pen_leaning_cube")

        self.assertEqual(profile.actor_attr, "pen")
        self.assertEqual(profile.required_active_arm, "right")
        self.assertEqual(profile.arm_plan_policy, "right_only")
        self.assertEqual(
            profile.verification_method_override,
            selector.FULL_EXPERT_METHOD,
        )

    def test_shoe_selects_only_the_arm_on_the_object_side(self) -> None:
        profile = selector.safety_profile("place_shoe")

        left = selector.validate_plan_chains(
            profile,
            initial_xyz=np.array([-0.2, 0.0, 0.75]),
            right_waypoint_counts=[],
            left_waypoint_counts=[20, 15, 12, 8],
        )
        right = selector.validate_plan_chains(
            profile,
            initial_xyz=np.array([0.2, 0.0, 0.75]),
            right_waypoint_counts=[20, 15, 12, 8],
            left_waypoint_counts=[],
        )
        wrong_arm = selector.validate_plan_chains(
            profile,
            initial_xyz=np.array([-0.2, 0.0, 0.75]),
            right_waypoint_counts=[20, 15, 12, 8],
            left_waypoint_counts=[],
        )

        self.assertEqual(left["selected_arm"], "left")
        self.assertTrue(left["policy_ok"])
        self.assertEqual(right["selected_arm"], "right")
        self.assertTrue(right["policy_ok"])
        self.assertFalse(wrong_arm["policy_ok"])

    def test_handover_and_lift_require_both_plan_chains(self) -> None:
        for task in (
            "handover_mic",
            "handover_horizontal_block",
            "handover_block",
            "handover_cube_to_target",
            "lift_pot",
        ):
            profile = selector.safety_profile(task)
            right_complete = [10] * profile.min_right_arm_plan_segments
            left_complete = [10] * profile.min_left_arm_plan_segments
            complete = selector.validate_plan_chains(
                profile,
                initial_xyz=np.zeros(3),
                right_waypoint_counts=right_complete,
                left_waypoint_counts=left_complete,
            )
            incomplete = selector.validate_plan_chains(
                profile,
                initial_xyz=np.zeros(3),
                right_waypoint_counts=right_complete,
                left_waypoint_counts=left_complete[:-1],
            )

            self.assertTrue(complete["policy_ok"])
            self.assertFalse(incomplete["policy_ok"])

    def test_handover_mic_uses_microphone_profile(self) -> None:
        profile = selector.safety_profile("handover_mic")

        self.assertEqual(profile.actor_attr, "microphone")
        self.assertEqual(profile.method, "microphone_dual_arm_handover_v2")
        self.assertIn("microphone", profile.object_layout)
        self.assertEqual(profile.verification_mode, "dual_handover")
        self.assertEqual(profile.required_gripper_state, "handover")
        criteria = selector.selection_criteria(
            profile,
            profile.min_verified_lift_mm,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertIn("microphone lift", criteria)
        self.assertIn("vertical presentation", criteria)
        self.assertNotIn("horizontal block lift", criteria)

    def test_horizontal_block_uses_independent_vertical_handover_profile(self) -> None:
        profile = selector.safety_profile("handover_horizontal_block")

        self.assertEqual(profile.actor_attr, "block")
        self.assertEqual(
            profile.method,
            "horizontal_block_vertical_dual_arm_handover_v2",
        )
        self.assertIn("3x3x20 cm block", profile.object_layout)
        self.assertIn("presented vertically", profile.object_layout)
        self.assertEqual(profile.verification_mode, "dual_handover")
        criteria = selector.selection_criteria(
            profile,
            profile.min_verified_lift_mm,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertIn("horizontal block lift", criteria)
        self.assertIn("vertical presentation", criteria)
        self.assertNotIn("microphone lift", criteria)

    def test_handover_block_requires_transfer_place_and_both_open(self) -> None:
        profile = selector.safety_profile("handover_block")

        self.assertEqual(profile.actor_attr, "box")
        self.assertEqual(profile.container_attr, "target_box")
        self.assertEqual(profile.verification_mode, "handover_place")
        self.assertEqual(profile.required_gripper_state, "both_open")
        self.assertEqual(profile.required_active_arm, "both")
        self.assertEqual(
            profile.verification_method_override,
            selector.FULL_EXPERT_METHOD,
        )

        criteria = selector.selection_criteria(
            profile,
            profile.min_verified_lift_mm,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertIn("left-to-right block handover", criteria)
        self.assertIn("block on target pad", criteria)

    def test_cube_handover_uses_dynamic_roles_and_receiver_placement(self) -> None:
        profile = selector.safety_profile("handover_cube_to_target")

        self.assertEqual(profile.actor_attr, "cube")
        self.assertEqual(profile.container_attr, "target_pad")
        self.assertEqual(profile.verification_mode, "handover_place")
        self.assertEqual(profile.required_gripper_state, "both_open")
        self.assertEqual(profile.arm_plan_policy, "both")
        self.assertEqual(profile.required_active_arm, "both")
        self.assertEqual(profile.method, "dynamic_cube_two_stage_relay_place_v1")

        criteria = selector.selection_criteria(
            profile,
            profile.min_verified_lift_mm,
            selector.FULL_EXPERT_METHOD,
        )
        self.assertIn("giver from cube side", criteria)
        self.assertIn("receiver from target side", criteria)
        self.assertIn("middle-pad set-down", criteria)
        self.assertIn("separate receiver pickup", criteria)
        self.assertNotIn("left-to-right", criteria)

    def test_dual_gripper_terminal_states(self) -> None:
        class Task:
            grasp_arm_tag = "left"
            handover_arm_tag = "right"

        self.assertTrue(
            selector.gripper_state_matches(
                Task(),
                "handover",
                right_closed=True,
                right_open=False,
                left_closed=False,
                left_open=True,
            )
        )
        self.assertFalse(
            selector.gripper_state_matches(
                Task(),
                "handover",
                right_closed=False,
                right_open=True,
                left_closed=True,
                left_open=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
