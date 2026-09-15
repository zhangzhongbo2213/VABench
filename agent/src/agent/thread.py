from __future__ import annotations

from collections.abc import Iterable
import sys

from .client import ChatClient
from .context import ContextManager, is_context_overflow_error
from .events import RunLogger
from .session import Message, Part, Store


class AgentThread:
    def __init__(
        self,
        client: ChatClient,
        store: Store,
        session_id: str,
        logger: RunLogger | None = None,
        *,
        stream: bool = True,
    ):
        self.client = client
        self.store = store
        self.session_id = session_id
        self.logger = logger
        self.stream = stream
        self.context = ContextManager(client, store, session_id, logger)

    def ask(self, content: str | Iterable[Part]) -> str:
        if isinstance(content, str):
            message = Message("user", content)
            transcript = content
        else:
            parts = list(content)
            message = Message.with_parts("user", parts)
            transcript = "\n".join(part.preview() for part in parts)
        self.store.append(self.session_id, message)
        if self.logger:
            self.logger.transcript("user", transcript)
            self.logger.event("model_start", {"session": self.session_id})
        reply = self.complete()
        self.store.append(self.session_id, Message("assistant", reply))
        if self.logger:
            self.logger.transcript("assistant", reply)
            self.logger.event("model_finish", {"chars": len(reply)})
        return reply

    def complete(self) -> str:
        messages = self.context.prepare()
        if not self.stream:
            reply = self._complete_with_context_retry(messages)
            print(reply)
            return reply
        chunks: list[str] = []
        try:
            print("assistant> ", end="", flush=True)
            for delta in self.client.stream(messages):
                chunks.append(delta)
                print(delta, end="", flush=True)
            print()
            if chunks:
                return "".join(chunks)
        except Exception as exc:
            if self.logger:
                self.logger.event("model_stream_error", {"error": str(exc)})
            else:
                print(f"[model_stream_error] {exc}", file=sys.stderr)
            if is_context_overflow_error(exc):
                messages = self.context.prepare(
                    force=True,
                    reason="provider_context_error",
                )
        reply = self._complete_with_context_retry(messages)
        print(reply)
        return reply

    def _complete_with_context_retry(self, messages: list[Message]) -> str:
        try:
            return self.client.complete(messages).content
        except Exception as exc:
            if not is_context_overflow_error(exc) or not self.context.enabled:
                raise
            if self.logger:
                self.logger.event(
                    "context_overflow_retry",
                    {"error": str(exc), "session": self.session_id},
                )
            compacted = self.context.prepare(
                force=True,
                reason="provider_context_error",
            )
            return self.client.complete(compacted).content


def ensure_session(store: Store, session_id: str | None, system: str | None, title: str | None) -> str:
    sid = session_id or store.create(title)
    if session_id and not store.path(sid).exists():
        raise SystemExit(f"session not found: {sid}")
    if system and not any(message.role == "system" for message in store.load(sid)):
        store.append(sid, Message("system", system))
    return sid
