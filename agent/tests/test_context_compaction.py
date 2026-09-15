from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from agent.client import ChatClient
from agent.config import Config
from agent.context import (
    COMPACTION_PROMPT,
    ContextManager,
    estimate_text_tokens,
    is_context_overflow_error,
)
from agent.session import Message, Part, Store
from agent.thread import AgentThread


class ContextCompactionTest(unittest.TestCase):
    def config(self, root: Path, **overrides) -> Config:
        values = {
            "base_url": "https://provider.example/v1",
            "api_key": "sk-test",
            "model": "test-model",
            "state_dir": root,
            "context_window_tokens": 1_000,
            "context_compaction_threshold": 0.5,
            "context_compaction_keep_recent_turns": 2,
            "context_compaction_keep_recent_images": 1,
        }
        values.update(overrides)
        return Config(**values)

    def test_estimator_counts_non_ascii_more_conservatively(self) -> None:
        self.assertGreater(estimate_text_tokens("空间理解" * 10), 10)
        self.assertLess(estimate_text_tokens("a" * 100), 30)

    def test_compaction_preserves_raw_transcript_and_limits_active_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_paths = []
            for index in range(3):
                path = root / f"frame_{index}.png"
                Image.new("RGB", (960, 720), color=(index, 10, 20)).save(path)
                image_paths.append(path)
            store = Store(root)
            sid = store.create("context test")
            store.append(sid, Message("system", "authoritative system rules"))
            for index, image_path in enumerate(image_paths):
                store.append(
                    sid,
                    Message.with_parts(
                        "user",
                        [
                            Part.text_part("observation " + ("x" * 220)),
                            Part.image_part(image_path, label=f"step {index}"),
                        ],
                    ),
                )
                store.append(sid, Message("assistant", f'action {index} reason'))
            raw_before = store.path(sid).read_text(encoding="utf-8")
            calls: list[dict[str, object]] = []

            def transport(url, headers, payload, timeout):
                del url, headers, timeout
                calls.append(payload)
                return {"choices": [{"message": {"content": "structured checkpoint"}}]}

            client = ChatClient(self.config(root), transport=transport)
            manager = ContextManager(client, store, sid)
            active = manager.prepare()

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0]["messages"][0]["content"],
                COMPACTION_PROMPT,
            )
            self.assertEqual(store.path(sid).read_text(encoding="utf-8"), raw_before)
            self.assertTrue(manager.checkpoint_path.exists())
            checkpoint = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["summary_model"], "test-model")
            self.assertEqual(checkpoint["compaction_count"], 1)
            self.assertEqual(active[0].role, "system")
            self.assertIn("structured checkpoint", active[1].text())
            active_images = sum(
                1
                for message in active
                if isinstance(message.content, list)
                for part in message.content
                if part.type == "image"
            )
            self.assertEqual(active_images, 1)
            self.assertTrue(
                any("image omitted" in message.text() for message in active)
            )

            new_image = root / "frame_new.png"
            Image.new("RGB", (960, 720), color=(30, 40, 50)).save(new_image)
            store.append(
                sid,
                Message.with_parts(
                    "user",
                    [
                        Part.text_part("new unsummarized observation"),
                        Part.image_part(new_image, label="new step"),
                    ],
                ),
            )
            active_with_new_history = manager.build_active_messages(
                store.load(sid),
                manager.load_checkpoint(),
            )
            active_images = sum(
                1
                for message in active_with_new_history
                if isinstance(message.content, list)
                for part in message.content
                if part.type == "image"
            )
            self.assertEqual(active_images, 2)

    def test_compaction_inherits_response_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root)
            sid = store.create("language test")
            store.append(sid, Message("system", "system"))
            store.append(sid, Message("user", "x" * 2_000))
            calls: list[dict[str, object]] = []

            def transport(url, headers, payload, timeout):
                del url, headers, timeout
                calls.append(payload)
                return {"choices": [{"message": {"content": "checkpoint"}}]}

            client = ChatClient(
                self.config(root, response_language="English"),
                transport=transport,
            )
            ContextManager(client, store, sid).prepare()
            self.assertIn(
                "Write the checkpoint in English only.",
                calls[0]["messages"][0]["content"],
            )

    def test_agent_retries_context_error_after_forced_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root)
            sid = store.create("retry")
            store.append(sid, Message("system", "rules"))
            store.append(sid, Message("user", "short history"))
            calls: list[dict[str, object]] = []

            def transport(url, headers, payload, timeout):
                del url, headers, timeout
                calls.append(payload)
                first_message = payload["messages"][0]["content"]
                if first_message == COMPACTION_PROMPT:
                    return {
                        "choices": [
                            {"message": {"content": "forced checkpoint summary"}}
                        ]
                    }
                evaluation_calls = [
                    call
                    for call in calls
                    if call["messages"][0]["content"] != COMPACTION_PROMPT
                ]
                if len(evaluation_calls) == 1:
                    raise RuntimeError(
                        "HTTP 400: context_length_exceeded: maximum context length"
                    )
                return {"choices": [{"message": {"content": "recovered"}}]}

            client = ChatClient(
                self.config(
                    root,
                    context_window_tokens=100_000,
                    context_compaction_threshold=0.9,
                ),
                transport=transport,
            )
            thread = AgentThread(client, store, sid, stream=False)
            reply = thread.complete()

            self.assertEqual(reply, "recovered")
            self.assertEqual(len(calls), 3)
            self.assertTrue(thread.context.checkpoint_path.exists())
            self.assertEqual(len(store.load(sid)), 2)

    def test_context_error_detection_handles_413_and_named_code(self) -> None:
        self.assertTrue(
            is_context_overflow_error(
                RuntimeError("HTTP 413 RequestTooLarge from provider")
            )
        )
        self.assertTrue(
            is_context_overflow_error(
                RuntimeError('{"code":"context_length_exceeded"}')
            )
        )
        self.assertTrue(
            is_context_overflow_error(
                RuntimeError(
                    'HTTP 400: {"type":"exceed_context_size_error",'
                    '"message":"request exceeds the available context size"}'
                )
            )
        )
        self.assertFalse(is_context_overflow_error(RuntimeError("HTTP 429 quota")))

    def test_compaction_keeps_every_image_in_latest_multiframe_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = []
            for index in range(5):
                path = root / f"expert_{index}.png"
                Image.new("RGB", (64, 48), color=(index, 0, 0)).save(path)
                images.append(path)
            store = Store(root)
            sid = store.create("multiframe")
            store.append(sid, Message("system", "rules"))
            store.append(
                sid,
                Message.with_parts(
                    "user",
                    [
                        Part.text_part("inspect ordered expert frames"),
                        *[
                            Part.image_part(path, label=f"expert {index}")
                            for index, path in enumerate(images)
                        ],
                    ],
                ),
            )

            def transport(url, headers, payload, timeout):
                del url, headers, payload, timeout
                return {"choices": [{"message": {"content": "checkpoint"}}]}

            manager = ContextManager(
                ChatClient(
                    self.config(
                        root,
                        context_compaction_keep_recent_images=1,
                        context_window_tokens=100,
                        context_compaction_threshold=0.5,
                    ),
                    transport=transport,
                ),
                store,
                sid,
            )
            active = manager.prepare()
            active_images = [
                part
                for message in active
                if isinstance(message.content, list)
                for part in message.content
                if part.type == "image"
            ]

            self.assertEqual(len(active_images), 5)


if __name__ == "__main__":
    unittest.main()
