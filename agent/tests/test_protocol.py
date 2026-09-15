from __future__ import annotations

import unittest

from agent.protocol import (
    decision_env_action,
    parse_decision,
    parameterized_action_error,
)


class ProtocolTest(unittest.TestCase):
    def test_parse_plain_action(self) -> None:
        decision = parse_decision('{"action":"camera.view_topdown","reason":"inspect xy"}')

        self.assertEqual(decision.kind, "action")
        self.assertEqual(decision.action, "camera.view_topdown")
        self.assertEqual(decision.reason, "inspect xy")

    def test_parse_tool_from_fenced_json(self) -> None:
        decision = parse_decision(
            '```json\n{"tool":"geometry.verify","args":{"check":"gripper_posture"},"reason":"need posture"}\n```'
        )

        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.tool_name, "geometry.verify")
        self.assertEqual(decision.tool_args, {"check": "gripper_posture"})

    def test_candidate_execution_tool_is_a_terminal_environment_decision(self) -> None:
        decision = parse_decision(
            '{"tool":"spatial.execute_grasp_candidate",'
            '"args":{"candidate_id":"grasp_candidate_000"},'
            '"reason":"execute authorized frame"}'
        )

        self.assertEqual(decision.kind, "candidate_execution")
        self.assertEqual(decision.tool_name, "spatial.execute_grasp_candidate")
        self.assertEqual(
            decision.tool_args, {"candidate_id": "grasp_candidate_000"}
        )

    def test_known_tool_mislabeled_as_action_is_normalized(self) -> None:
        decision = parse_decision(
            '{"action":"expert.frame","args":{"seed":0,"step":20,"view":"center_high"},'
            '"reason":"inspect middle frame"}'
        )

        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.tool_name, "expert.frame")
        self.assertEqual(
            decision.tool_args,
            {"seed": 0, "step": 20, "view": "center_high"},
        )
        self.assertEqual(decision.reason, "inspect middle frame")

    def test_parse_parameterized_world_move(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.move_world","axis":"y","sign":"+","distance_mm":12.5,"reason":"align gc"}'
        )

        self.assertEqual(decision.action, "gripper.world_y_pos.12.5mm")
        self.assertEqual(
            decision_env_action(decision),
            {"target": "gripper", "type": "move_world", "axis": "y", "sign": "+", "distance_mm": 12.5},
        )

    def test_parse_parameterized_rotation(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.rotate_world","axis":"ry","sign":"-","angle_deg":45,"reason":"point down"}'
        )

        self.assertEqual(decision.action, "gripper.rotate_ry_cw.45deg")
        self.assertEqual(
            decision_env_action(decision),
            {"target": "gripper", "type": "rotate", "direction": "rotate_ry_cw", "angle_deg": 45},
        )

    def test_parse_parameterized_local_rotation(self) -> None:
        decision = parse_decision(
            '{"action":"gripper.rotate_local","axis":"ry","sign":"-","angle_deg":45,"reason":"tilt in gripper frame"}'
        )

        self.assertEqual(decision.action, "gripper.local_rotate_ry_cw.45deg")
        self.assertEqual(
            decision_env_action(decision),
            {"target": "gripper", "type": "rotate_local", "axis": "ry", "sign": "-", "angle_deg": 45},
        )

    def test_parse_arm_specific_numeric_actions(self) -> None:
        move = parse_decision(
            '{"action":"gripper.move_world","arm":"left","axis":"x","sign":"+","distance_mm":15}'
        )
        rotate = parse_decision(
            '{"action":"gripper.rotate_local","arm":"right","axis":"rz","sign":"-","angle_deg":20}'
        )

        self.assertEqual(move.action, "left_gripper.world_x_pos.15mm")
        self.assertEqual(decision_env_action(move)["arm"], "left")
        self.assertEqual(rotate.action, "right_gripper.local_rotate_rz_cw.20deg")
        self.assertEqual(decision_env_action(rotate)["arm"], "right")

    def test_parse_synchronized_dual_move(self) -> None:
        decision = parse_decision(
            '{"action":"dual_gripper.move_world",'
            '"left":{"axis":"z","sign":"+","distance_mm":25},'
            '"right":{"axis":"z","sign":"+","distance_mm":25}}'
        )

        self.assertEqual(decision.action, "dual_gripper.move_world")
        self.assertEqual(decision_env_action(decision)["left"]["distance_mm"], 25)
        self.assertEqual(decision_env_action(decision)["right"]["sign"], "+")

    def test_parse_stop(self) -> None:
        decision = parse_decision('{"stop":true,"reason":"unsafe"}')

        self.assertEqual(decision.kind, "stop")
        self.assertEqual(decision.reason, "unsafe")

    def test_parameterized_range_errors(self) -> None:
        self.assertIsNone(parameterized_action_error("gripper.world_x_pos.20mm"))
        self.assertIsNone(parameterized_action_error("gripper.rotate_rx_ccw.90deg"))
        self.assertIsNone(parameterized_action_error("gripper.local_rotate_rx_ccw.90deg"))
        self.assertIn("[1, 100]", parameterized_action_error("gripper.world_x_pos.120mm") or "")
        self.assertIn("[1, 90]", parameterized_action_error("gripper.rotate_rx_ccw.120deg") or "")


if __name__ == "__main__":
    unittest.main()
