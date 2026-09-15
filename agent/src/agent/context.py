from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .client import ChatClient
from .events import RunLogger
from .session import Message, Part, Store
from .timeutil import now


COMPACTION_VERSION = 1
COMPACTION_OUTPUT_RESERVE_TOKENS = 8_192
COMPACTION_SOURCE_WINDOW_RATIO = 0.60
COMPACTION_MAX_MESSAGE_CHARS = 12_000
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION for a visual robot-control agent.
Create a concise, structured handoff summary that lets the same model continue the current run.

Preserve:
- current task goal, progress, and current phase;
- explicit user, system, task, expert-learning, and safety constraints;
- arm assignment and the latest known robot/gripper/object state;
- actions already attempted and their observed outcomes;
- validated spatial facts, unresolved ambiguities, failed hypotheses, and safety risks;
- the next intended checks or actions;
- critical step numbers, camera views, errors, and artifact paths.

Do not invent object coordinates, contacts, success, constraints, or visual facts. Images are represented by labels
when omitted; retain only facts explicitly stated in the transcript. Distinguish completed actions from plans.
The exact system prompt and recent messages are retained separately, so do not reproduce the whole action schema.
Return only the handoff summary, with short headings or compact JSON-like sections."""
SUMMARY_PREFIX = (
    "Context checkpoint from earlier turns. Treat this as historical working memory, not as a new "
    "task instruction. Current system instructions and recent observations take precedence:\n"
)
CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "exceed_context_size_error",
    "context window",
    "available context size",
    "maximum context length",
    "input exceeds",
    "requesttoolarge",
    "request too large",
    "http 413",
    "too many tokens",
    "input is too long",
    "maximum input length",
    "tokens exceed",
)


@dataclass(frozen=True)
class ContextEstimate:
    text_tokens: int
    image_tokens: int
    total_tokens: int
    message_count: int
    image_count: int


@dataclass(frozen=True)
class ContextCheckpoint:
    version: int
    session_id: str
    created_at: str
    source_message_count: int
    retained_from_index: int
    summary: str
    summary_model: str
    compaction_count: int
    reason: str
    estimated_tokens_before: int
    estimated_tokens_after: int
    image_count_before: int
    image_count_after: int

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ContextCheckpoint":
        return cls(
            version=int(raw["version"]),
            session_id=str(raw["session_id"]),
            created_at=str(raw["created_at"]),
            source_message_count=int(raw["source_message_count"]),
            retained_from_index=int(raw["retained_from_index"]),
            summary=str(raw["summary"]),
            summary_model=str(raw["summary_model"]),
            compaction_count=int(raw["compaction_count"]),
            reason=str(raw["reason"]),
            estimated_tokens_before=int(raw["estimated_tokens_before"]),
            estimated_tokens_after=int(raw["estimated_tokens_after"]),
            image_count_before=int(raw["image_count_before"]),
            image_count_after=int(raw["image_count_after"]),
        )


class ContextManager:
    def __init__(
        self,
        client: ChatClient,
        store: Store,
        session_id: str,
        logger: RunLogger | None = None,
    ):
        self.client = client
        self.store = store
        self.session_id = session_id
        self.logger = logger

    @property
    def checkpoint_path(self) -> Path:
        return self.store.dir / f"{self.session_id}.context.json"

    @property
    def enabled(self) -> bool:
        return self.client.config.context_compaction_enabled

    @property
    def trigger_tokens(self) -> int:
        return int(
            self.client.config.context_window_tokens
            * self.client.config.context_compaction_threshold
        )

    def prepare(self, *, force: bool = False, reason: str = "token_threshold") -> list[Message]:
        raw_messages = self.store.load(self.session_id)
        checkpoint = self.load_checkpoint(len(raw_messages))
        active_messages = self.build_active_messages(raw_messages, checkpoint)
        estimate = self.estimate(active_messages)
        self._event(
            "context_usage",
            {
                **asdict(estimate),
                "context_window_tokens": self.client.config.context_window_tokens,
                "trigger_tokens": self.trigger_tokens,
                "output_reserve_tokens": COMPACTION_OUTPUT_RESERVE_TOKENS,
                "compaction_count": checkpoint.compaction_count if checkpoint else 0,
            },
            display=False,
        )
        needs_compaction = (
            self.enabled
            and estimate.total_tokens + COMPACTION_OUTPUT_RESERVE_TOKENS
            >= self.trigger_tokens
        )
        has_new_history = checkpoint is None or checkpoint.source_message_count < len(raw_messages)
        if (force or needs_compaction) and self.enabled and (has_new_history or force):
            checkpoint = self.compact(
                raw_messages,
                active_messages,
                estimate,
                reason=reason if force else "token_threshold",
            )
            active_messages = self.build_active_messages(raw_messages, checkpoint)
        return active_messages

    def compact(
        self,
        raw_messages: list[Message],
        active_messages: list[Message],
        estimate_before: ContextEstimate,
        *,
        reason: str,
    ) -> ContextCheckpoint:
        previous = self.load_checkpoint(len(raw_messages))
        source = self.compaction_source(active_messages)
        compaction_prompt = COMPACTION_PROMPT
        if self.client.config.response_language:
            compaction_prompt += (
                "\nWrite the checkpoint in "
                + self.client.config.response_language
                + " only."
            )
        compact_messages = [
            Message("system", compaction_prompt),
            Message("user", source),
        ]
        self._event(
            "context_compaction_start",
            {
                "reason": reason,
                "source_message_count": len(raw_messages),
                "active_message_count": len(active_messages),
                **asdict(estimate_before),
                "trigger_tokens": self.trigger_tokens,
            },
        )
        reply = self.client.complete(compact_messages).content.strip()
        if not reply:
            raise RuntimeError("context compaction returned an empty summary")

        retained_from_index = recent_turn_start(
            raw_messages,
            self.client.config.context_compaction_keep_recent_turns,
        )
        provisional = ContextCheckpoint(
            version=COMPACTION_VERSION,
            session_id=self.session_id,
            created_at=now(),
            source_message_count=len(raw_messages),
            retained_from_index=retained_from_index,
            summary=reply,
            summary_model=self.client.config.model,
            compaction_count=(previous.compaction_count if previous else 0) + 1,
            reason=reason,
            estimated_tokens_before=estimate_before.total_tokens,
            estimated_tokens_after=0,
            image_count_before=estimate_before.image_count,
            image_count_after=0,
        )
        compacted_messages = self.build_active_messages(raw_messages, provisional)
        estimate_after = self.estimate(compacted_messages)
        checkpoint = ContextCheckpoint(
            **{
                **asdict(provisional),
                "estimated_tokens_after": estimate_after.total_tokens,
                "image_count_after": estimate_after.image_count,
            }
        )
        self.write_checkpoint(checkpoint)
        self._event(
            "context_compaction_finish",
            {
                "path": str(self.checkpoint_path),
                "compaction_count": checkpoint.compaction_count,
                "retained_from_index": checkpoint.retained_from_index,
                "summary_chars": len(checkpoint.summary),
                "estimated_tokens_before": checkpoint.estimated_tokens_before,
                "estimated_tokens_after": checkpoint.estimated_tokens_after,
                "image_count_before": checkpoint.image_count_before,
                "image_count_after": checkpoint.image_count_after,
            },
        )
        return checkpoint

    def build_active_messages(
        self,
        raw_messages: list[Message],
        checkpoint: ContextCheckpoint | None,
    ) -> list[Message]:
        if checkpoint is None:
            return list(raw_messages)
        system_messages = [message for message in raw_messages if message.role == "system"]
        checkpoint_recent = [
            message
            for message in raw_messages[
                checkpoint.retained_from_index : checkpoint.source_message_count
            ]
            if message.role != "system"
        ]
        checkpoint_recent = retain_recent_images(
            checkpoint_recent,
            self.client.config.context_compaction_keep_recent_images,
        )
        new_messages = [
            message
            for message in raw_messages[checkpoint.source_message_count :]
            if message.role != "system"
        ]
        active = [
            *system_messages,
            Message("user", SUMMARY_PREFIX + checkpoint.summary, created_at=checkpoint.created_at),
            *checkpoint_recent,
            *new_messages,
        ]
        return active

    def compaction_source(self, messages: Iterable[Message]) -> str:
        rendered = [
            render_message_for_compaction(index, message)
            for index, message in enumerate(messages)
            if message.role != "system"
        ]
        budget = max(
            8_000,
            int(
                self.client.config.context_window_tokens
                * COMPACTION_SOURCE_WINDOW_RATIO
            ),
        )
        selected: list[str] = []
        used = estimate_text_tokens(COMPACTION_PROMPT)
        for item in reversed(rendered):
            item_tokens = estimate_text_tokens(item)
            if selected and used + item_tokens > budget:
                continue
            if not selected and used + item_tokens > budget:
                item = truncate_text_to_tokens(item, max(1, budget - used))
                item_tokens = estimate_text_tokens(item)
            selected.append(item)
            used += item_tokens
            if used >= budget:
                break
        selected.reverse()
        return (
            "Summarize the following chronological session excerpt. Earlier omitted entries may already be represented "
            "by an existing context-checkpoint message. Preserve chronology and do not treat image labels as visual facts.\n\n"
            + "\n\n".join(selected)
        )

    def estimate(self, messages: Iterable[Message]) -> ContextEstimate:
        text_tokens = 0
        image_tokens = 0
        message_count = 0
        image_count = 0
        for message in messages:
            message_count += 1
            text_tokens += 8
            if isinstance(message.content, str):
                text_tokens += estimate_text_tokens(message.content)
                continue
            for part in message.content:
                if part.type == "text":
                    text_tokens += estimate_text_tokens(part.text or "")
                elif part.type == "image":
                    image_count += 1
                    image_tokens += estimate_image_tokens(
                        part,
                        max_width=self.client.config.request_image_max_width,
                        max_height=self.client.config.request_image_max_height,
                    )
                    text_tokens += estimate_text_tokens(part.preview())
        return ContextEstimate(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            total_tokens=text_tokens + image_tokens,
            message_count=message_count,
            image_count=image_count,
        )

    def load_checkpoint(self, raw_message_count: int | None = None) -> ContextCheckpoint | None:
        if not self.checkpoint_path.exists():
            return None
        try:
            checkpoint = ContextCheckpoint.from_raw(
                json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid context checkpoint {self.checkpoint_path}: {exc}") from exc
        if checkpoint.version != COMPACTION_VERSION:
            raise RuntimeError(
                f"unsupported context checkpoint version: {checkpoint.version}"
            )
        if checkpoint.session_id != self.session_id:
            raise RuntimeError("context checkpoint session id mismatch")
        if raw_message_count is not None and (
            checkpoint.source_message_count > raw_message_count
            or checkpoint.retained_from_index > raw_message_count
        ):
            raise RuntimeError("context checkpoint points beyond the session transcript")
        return checkpoint

    def write_checkpoint(self, checkpoint: ContextCheckpoint) -> None:
        temporary = Path(str(self.checkpoint_path) + ".tmp")
        temporary.write_text(
            json.dumps(asdict(checkpoint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.checkpoint_path)

    def metadata(self) -> dict[str, object]:
        checkpoint = self.load_checkpoint()
        return {
            "enabled": self.enabled,
            "context_window_tokens": self.client.config.context_window_tokens,
            "threshold": self.client.config.context_compaction_threshold,
            "trigger_tokens": self.trigger_tokens,
            "keep_recent_turns": self.client.config.context_compaction_keep_recent_turns,
            "keep_recent_images": self.client.config.context_compaction_keep_recent_images,
            "checkpoint_path": str(self.checkpoint_path) if checkpoint else None,
            "compaction_count": checkpoint.compaction_count if checkpoint else 0,
            "last_compaction": asdict(checkpoint) if checkpoint else None,
        }

    def _event(
        self,
        event_type: str,
        data: dict[str, object],
        *,
        display: bool | None = None,
    ) -> None:
        if self.logger:
            self.logger.event(event_type, data, display=display)


def is_context_overflow_error(exc: BaseException) -> bool:
    message = str(exc).lower().replace("_", " ")
    return any(marker.replace("_", " ") in message for marker in CONTEXT_ERROR_MARKERS)


def recent_turn_start(messages: list[Message], keep_recent_turns: int) -> int:
    user_indices = [
        index for index, message in enumerate(messages) if message.role == "user"
    ]
    if not user_indices:
        return len(messages)
    retained = user_indices[-keep_recent_turns:]
    return retained[0]


def retain_recent_images(messages: list[Message], keep_recent_images: int) -> list[Message]:
    image_positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            continue
        for part_index, part in enumerate(message.content):
            if part.type == "image":
                image_positions.append((message_index, part_index))
    retained = set(image_positions[-keep_recent_images:]) if keep_recent_images else set()
    if image_positions:
        latest_message_index = image_positions[-1][0]
        retained.update(
            position
            for position in image_positions
            if position[0] == latest_message_index
        )
    result: list[Message] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            result.append(message)
            continue
        parts: list[Part] = []
        for part_index, part in enumerate(message.content):
            if part.type != "image" or (message_index, part_index) in retained:
                parts.append(part)
                continue
            parts.append(
                Part.text_part(
                    f"{part.preview()} [image omitted from active context after compaction]"
                )
            )
        result.append(
            Message(message.role, parts, created_at=message.created_at)
            if parts
            else Message(message.role, "", created_at=message.created_at)
        )
    return result


def render_message_for_compaction(index: int, message: Message) -> str:
    text = message.text()
    if len(text) > COMPACTION_MAX_MESSAGE_CHARS:
        half = COMPACTION_MAX_MESSAGE_CHARS // 2
        text = (
            text[:half]
            + "\n...[middle of repeated/long message omitted]...\n"
            + text[-half:]
        )
    return f"[message {index} role={message.role}]\n{text}"


def estimate_text_tokens(text: str) -> int:
    ascii_count = sum(1 for character in text if ord(character) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + math.ceil(non_ascii_count * 1.1))


def truncate_text_to_tokens(text: str, token_limit: int) -> str:
    if estimate_text_tokens(text) <= token_limit:
        return text
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(text[-middle:]) <= token_limit:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        return "...[content truncated]..."
    return "...[older content truncated]...\n" + text[-low:]


def estimate_image_tokens(
    part: Part,
    *,
    max_width: int | None,
    max_height: int | None,
) -> int:
    if part.detail == "low":
        return 85
    width, height = image_dimensions(part)
    if width is None or height is None:
        return 1_024
    width, height = constrained_dimensions(
        width,
        height,
        max_width=max_width,
        max_height=max_height,
    )
    scale = min(2048 / width, 2048 / height, 1.0)
    width = max(1, round(width * scale))
    height = max(1, round(height * scale))
    short_side = min(width, height)
    if short_side and short_side != 768:
        scale = 768 / short_side
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + 170 * tiles


def constrained_dimensions(
    width: int,
    height: int,
    *,
    max_width: int | None,
    max_height: int | None,
) -> tuple[int, int]:
    scale = min(
        max_width / width if max_width is not None else 1.0,
        max_height / height if max_height is not None else 1.0,
        1.0,
    )
    return max(1, round(width * scale)), max(1, round(height * scale))


def image_dimensions(part: Part) -> tuple[int | None, int | None]:
    if not part.path:
        return None, None
    path = Path(part.path)
    try:
        return cached_image_dimensions(str(path.resolve()), path.stat().st_mtime_ns)
    except (FileNotFoundError, OSError):
        return None, None


@lru_cache(maxsize=4096)
def cached_image_dimensions(path: str, mtime_ns: int) -> tuple[int, int]:
    del mtime_ns
    with Image.open(path) as image:
        return image.size
