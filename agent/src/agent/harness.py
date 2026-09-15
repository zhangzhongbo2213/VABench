from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .events import RunLogger
from .session import Part
from .thread import AgentThread


DecisionT = TypeVar("DecisionT")


@dataclass(frozen=True)
class ToolResult:
    parts: list[Part]
    event_type: str | None = None
    event_data: dict[str, object] = field(default_factory=dict)


ToolHandler = Callable[[DecisionT], ToolResult]
Validator = Callable[[DecisionT], str | None]


class AgentHarness(Generic[DecisionT]):
    def __init__(
        self,
        thread: AgentThread,
        logger: RunLogger,
        *,
        parse: Callable[[str], DecisionT],
        kind: Callable[[DecisionT], str],
        repair_prompt: str,
        max_repairs: int = 2,
        max_iterations: int = 20,
    ):
        self.thread = thread
        self.logger = logger
        self.parse = parse
        self.kind = kind
        self.repair_prompt = repair_prompt
        self.max_repairs = max_repairs
        self.max_iterations = max_iterations

    def run(
        self,
        initial_parts: list[Part],
        *,
        tool_handlers: Mapping[str, ToolHandler[DecisionT]],
        validate: Validator[DecisionT],
        is_terminal: Callable[[DecisionT], bool],
    ) -> DecisionT:
        content = list(initial_parts)
        repairs = 0
        iterations = 0
        while True:
            iterations += 1
            if iterations > self.max_iterations:
                raise RuntimeError(f"harness exceeded {self.max_iterations} model/tool iterations")
            reply = self.thread.ask(content)
            try:
                decision = self.parse(reply)
            except ValueError as exc:
                repairs += 1
                self.logger.event("model_json_error", {"error": str(exc), "retry": repairs})
                if repairs > self.max_repairs:
                    raise
                content = [Part.text_part(self.repair_prompt + "\n" + json_error_feedback(str(exc)))]
                continue

            decision_kind = self.kind(decision)
            handler = tool_handlers.get(decision_kind)
            if handler is not None:
                result = handler(decision)
                if result.event_type:
                    self.logger.event(result.event_type, result.event_data)
                content = result.parts
                continue

            if is_terminal(decision):
                error = validate(decision)
                if error:
                    repairs += 1
                    self.logger.event("decision_rejected", {"error": error, "retry": repairs})
                    if repairs > self.max_repairs:
                        raise RuntimeError(error)
                    content = [Part.text_part(self.repair_prompt + "\n" + error + "\nOutput a corrected JSON command.")]
                    continue
                return decision

            repairs += 1
            error = f"Unsupported decision kind: {decision_kind!r}"
            self.logger.event("decision_kind_error", {"kind": decision_kind, "retry": repairs})
            if repairs > self.max_repairs:
                raise RuntimeError(error)
            content = [Part.text_part(self.repair_prompt + "\n" + error)]


def json_error_feedback(error: str) -> str:
    return (
        f"Invalid JSON command: {error}. Output exactly one JSON object using action, tool, or stop."
    )
