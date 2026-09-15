from __future__ import annotations

import unittest

from agent.harness import AgentHarness, ToolResult
from agent.protocol import Decision, parse_decision
from agent.session import Part


class FakeThread:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[list[Part] | str] = []

    def ask(self, content):
        self.prompts.append(content)
        if not self.replies:
            raise AssertionError("unexpected model call")
        return self.replies.pop(0)


class FakeLogger:
    def __init__(self):
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, event_type: str, data: dict[str, object] | None = None, **_kwargs) -> None:
        self.events.append((event_type, data or {}))


class HarnessTest(unittest.TestCase):
    def test_tool_result_stays_in_loop_until_action(self) -> None:
        thread = FakeThread(
            [
                '{"tool":"geometry.verify","args":{"check":"posture"},"reason":"need geometry"}',
                '{"action":"camera.view_topdown","reason":"inspect xy"}',
            ]
        )
        logger = FakeLogger()
        tool_calls: list[Decision] = []

        def handle_tool(decision: Decision) -> ToolResult:
            tool_calls.append(decision)
            return ToolResult([Part.text_part("geometry ok")], "tool_result", {"tool": decision.tool_name or ""})

        harness = AgentHarness(
            thread,
            logger,
            parse=parse_decision,
            kind=lambda decision: decision.kind,
            repair_prompt="repair",
        )

        decision = harness.run(
            [Part.text_part("initial")],
            tool_handlers={"tool": handle_tool},
            validate=lambda _decision: None,
            is_terminal=lambda decision: decision.kind in {"action", "stop"},
        )

        self.assertEqual(decision.action, "camera.view_topdown")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(logger.events[0][0], "tool_result")
        self.assertEqual(len(thread.prompts), 2)

    def test_json_repair_then_action(self) -> None:
        thread = FakeThread(["not json", '{"action":"gripper.open"}'])
        logger = FakeLogger()
        harness = AgentHarness(
            thread,
            logger,
            parse=parse_decision,
            kind=lambda decision: decision.kind,
            repair_prompt="repair",
        )

        decision = harness.run(
            [Part.text_part("initial")],
            tool_handlers={},
            validate=lambda _decision: None,
            is_terminal=lambda decision: decision.kind in {"action", "stop"},
        )

        self.assertEqual(decision.action, "gripper.open")
        self.assertEqual(logger.events[0][0], "model_json_error")


if __name__ == "__main__":
    unittest.main()
