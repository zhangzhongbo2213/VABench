from __future__ import annotations

import base64
from http.client import RemoteDisconnected
from io import BytesIO
import json
from pathlib import Path
import signal
import ssl
import tempfile
import unittest
from unittest.mock import patch
from urllib import error as urllib_error

from PIL import Image

from agent.client import (
    ChatClient,
    ProviderHTTPError,
    content,
    delta_content,
    endpoint,
    headers,
    is_account_quota_exceeded,
    pause_on_transient_error,
    post,
    provider_stream_error,
    redacted_base_url,
    redact_secrets,
    stream_post,
)
from agent.config import Config
from agent.robotwin.expert import ExpertStore
from agent.session import Message, Part, Store


class ClientSessionExpertTest(unittest.TestCase):
    def test_client_fallback_is_sticky_until_next_run(self) -> None:
        calls: list[tuple[str, str]] = []

        def transport(url, headers_, payload, timeout):
            calls.append((url, str(payload["model"])))
            if url.startswith("https://coding.example") and len(calls) == 1:
                raise ProviderHTTPError(
                    429,
                    '{"error":{"code":"AccountQuotaExceeded",'
                    '"message":"reset at 1970-01-01 00:33:20 +0000 CST"}}',
                )
            label = "balance" if url.startswith("https://balance.example") else "plan"
            return {"choices": [{"message": {"content": label}}]}

        client = ChatClient(
            Config(
                base_url="https://coding.example/v3",
                api_key="sk-test",
                model="coding-model",
                state_dir=Path("/tmp/state"),
                quota_fallback_base_url="https://balance.example/v3",
                quota_fallback_model="balance-model",
            ),
            transport=transport,
        )
        with patch("agent.client.time.time", return_value=1000.0):
            first = client.complete([Message("user", "one")])
            second = client.complete([Message("user", "two")])
            third = client.complete([Message("user", "three")])
            client.begin_run()
            fourth = client.complete([Message("user", "four")])

        self.assertEqual(
            [first.content, second.content, third.content, fourth.content],
            ["balance", "balance", "balance", "plan"],
        )
        self.assertEqual(
            calls,
            [
                ("https://coding.example/v3/chat/completions", "coding-model"),
                ("https://balance.example/v3/chat/completions", "balance-model"),
                ("https://balance.example/v3/chat/completions", "balance-model"),
                ("https://balance.example/v3/chat/completions", "balance-model"),
                ("https://coding.example/v3/chat/completions", "coding-model"),
            ],
        )

    def test_fallback_records_route_event_and_request_counts(self) -> None:
        events: list[dict[str, object]] = []

        def transport(url, headers_, payload, timeout):
            del headers_, payload, timeout
            if url.startswith("https://plan.example"):
                raise ProviderHTTPError(
                    429,
                    '{"error":{"code":"USAGE_LIMIT_EXCEEDED",'
                    '"message":"five hour limit exceeded"}}',
                )
            return {"choices": [{"message": {"content": "ok"}}]}

        client = ChatClient(
            Config(
                base_url="https://plan.example/v1",
                api_key="plan-key",
                model="qwen3.8-max",
                state_dir=Path("/tmp/state"),
                quota_fallback_base_url="https://api.example/v1",
                quota_fallback_model="qwen3.8-max",
                quota_fallback_api_key="api-key",
            ),
            transport=transport,
        )
        client.set_route_event_handler(events.append)

        self.assertEqual(
            client.complete([Message("user", "hello")]).content,
            "ok",
        )

        routing = client.routing_metadata()
        self.assertTrue(routing["fallback_used"])
        self.assertEqual(routing["primary_request_count"], 1)
        self.assertEqual(routing["fallback_request_count"], 1)
        self.assertEqual(events[0]["reason"], "quota_exhausted")
        self.assertEqual(events[0]["provider_error_code"], "USAGE_LIMIT_EXCEEDED")
        self.assertNotIn("api_key", json.dumps(events[0]))

    def test_quota_classifier_excludes_rate_limits_and_request_size(self) -> None:
        self.assertTrue(
            is_account_quota_exceeded(
                429,
                '{"code":"AccountQuotaExceeded"}',
            )
        )
        self.assertTrue(
            is_account_quota_exceeded(
                403,
                '{"error":{"code":"DAILY_LIMIT_EXCEEDED"}}',
            )
        )
        self.assertTrue(
            is_account_quota_exceeded(
                429,
                '{"error":{"code":"insufficient_quota"}}',
            )
        )
        self.assertFalse(
            is_account_quota_exceeded(
                429,
                '{"error":{"code":"rate_limit_exceeded",'
                '"message":"too many requests"}}',
            )
        )
        self.assertFalse(
            is_account_quota_exceeded(413, "RequestTooLarge")
        )
        self.assertFalse(
            is_account_quota_exceeded(403, "invalid API key")
        )
        self.assertFalse(
            is_account_quota_exceeded(429, "rate quota exceeded")
        )

    def test_route_telemetry_removes_url_credentials_and_query(self) -> None:
        self.assertEqual(
            redacted_base_url("https://user:secret@example.com:8443/v1?key=secret"),
            "https://example.com:8443/v1",
        )
        self.assertEqual(
            redact_secrets("provider echoed sk-secret", "sk-secret"),
            "provider echoed ***",
        )

    def test_fallback_provider_error_is_not_retried_on_primary(self) -> None:
        calls: list[str] = []

        def transport(url, headers_, payload, timeout):
            del headers_, payload, timeout
            calls.append(url)
            if url.startswith("https://plan.example"):
                raise ProviderHTTPError(429, '{"code":"AccountQuotaExceeded"}')
            raise ProviderHTTPError(503, "fallback unavailable")

        client = ChatClient(
            Config(
                base_url="https://plan.example/v1",
                api_key="plan-key",
                model="qwen3.8-max",
                state_dir=Path("/tmp/state"),
                quota_fallback_base_url="https://api.example/v1",
                quota_fallback_model="qwen3.8-max",
                quota_fallback_api_key="api-key",
            ),
            transport=transport,
        )

        with self.assertRaisesRegex(ProviderHTTPError, "HTTP 503"):
            client.complete([Message("user", "hello")])
        self.assertEqual(
            calls,
            [
                "https://plan.example/v1/chat/completions",
                "https://api.example/v1/chat/completions",
            ],
        )

    def test_stream_falls_back_before_any_output(self) -> None:
        calls: list[str] = []

        def stream_transport(url, headers_, payload, timeout):
            calls.append(url)
            if url.startswith("https://coding.example"):
                raise ProviderHTTPError(
                    429,
                    '{"error":{"code":"AccountQuotaExceeded"}}',
                )
            return ["fallback output"]

        client = ChatClient(
            Config(
                base_url="https://coding.example/v3",
                api_key="sk-test",
                model="coding-model",
                state_dir=Path("/tmp/state"),
                quota_fallback_base_url="https://balance.example/v3",
                quota_fallback_model="balance-model",
            ),
            stream_transport=stream_transport,
        )

        self.assertEqual(
            list(client.stream([Message("user", "hello")])),
            ["fallback output"],
        )
        self.assertEqual(
            calls,
            [
                "https://coding.example/v3/chat/completions",
                "https://balance.example/v3/chat/completions",
            ],
        )

    def test_stream_quota_after_partial_output_arms_fallback_for_repair(self) -> None:
        calls: list[str] = []

        def stream_transport(url, headers_, payload, timeout):
            del headers_, payload, timeout
            calls.append(url)
            if url.startswith("https://coding.example"):
                def partial():
                    yield "partial"
                    raise ProviderHTTPError(
                        429,
                        '{"code":"USAGE_LIMIT_EXCEEDED"}',
                    )
                return partial()
            return ["fallback stream"]

        def transport(url, headers_, payload, timeout):
            del headers_, payload, timeout
            calls.append(url)
            return {"choices": [{"message": {"content": "repaired"}}]}

        client = ChatClient(
            Config(
                base_url="https://coding.example/v3",
                api_key="plan-key",
                model="qwen3.8-max",
                state_dir=Path("/tmp/state"),
                quota_fallback_base_url="https://api.example/v3",
                quota_fallback_model="qwen3.8-max",
                quota_fallback_api_key="api-key",
            ),
            transport=transport,
            stream_transport=stream_transport,
        )

        iterator = client.stream([Message("user", "hello")])
        self.assertEqual(next(iterator), "partial")
        with self.assertRaises(ProviderHTTPError):
            next(iterator)
        self.assertEqual(
            client.complete([Message("user", "hello")]).content,
            "repaired",
        )
        self.assertEqual(calls[-1], "https://api.example/v3/chat/completions")

    def test_stream_error_factory_recognizes_quota_code(self) -> None:
        exc = provider_stream_error({"code": "USAGE_LIMIT_EXCEEDED"})
        self.assertIsInstance(exc, ProviderHTTPError)
        self.assertEqual(exc.status, 429)

    def test_post_surfaces_quota_immediately_when_fallback_is_enabled(self) -> None:
        quota = urllib_error.HTTPError(
            "https://coding.example/v3/chat/completions",
            429,
            "Too Many Requests",
            None,
            BytesIO(b'{"error":{"code":"AccountQuotaExceeded"}}'),
        )
        with (
            patch(
                "agent.client.request.urlopen",
                side_effect=quota,
            ) as urlopen,
            patch("agent.client.time.sleep") as sleep,
        ):
            with self.assertRaises(ProviderHTTPError):
                post(
                    "https://coding.example/v3/chat/completions",
                    {},
                    {"model": "test"},
                    7.0,
                    max_attempts=6,
                    fail_fast_quota=True,
                )

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_transient_pause_is_opt_in_and_stops_current_process(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(pause_on_transient_error("HTTP 503"))

        with (
            patch.dict("os.environ", {"AGENT_PAUSE_ON_TRANSIENT_HTTP": "1"}),
            patch("agent.client.os.getpid", return_value=1234),
            patch("agent.client.os.kill") as kill,
        ):
            self.assertTrue(pause_on_transient_error("HTTP 503"))
        kill.assert_called_once_with(1234, signal.SIGSTOP)

    def test_post_retries_remote_disconnect(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        with (
            patch(
                "agent.client.request.urlopen",
                side_effect=[RemoteDisconnected("closed"), Response()],
            ) as urlopen,
            patch("agent.client.time.sleep") as sleep,
        ):
            result = post(
                "https://provider.example/v1/chat/completions",
                {},
                {"model": "test"},
                7.0,
                max_attempts=2,
                retry_backoff_seconds=15.0,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(15.0)

    def test_post_retries_ssl_eof(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        with (
            patch(
                "agent.client.request.urlopen",
                side_effect=[ssl.SSLEOFError(8, "unexpected EOF"), Response()],
            ) as urlopen,
            patch("agent.client.time.sleep"),
        ):
            result = post(
                "https://provider.example/v1/chat/completions",
                {},
                {"model": "test"},
                7.0,
                max_attempts=2,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(urlopen.call_count, 2)

    def test_stream_transport_retries_transient_error_before_output(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def __iter__(self):
                return iter(
                    [
                        b'data: {"type":"response.output_text.delta","delta":"ok"}\n',
                        b"data: [DONE]\n",
                    ]
                )

        transient = urllib_error.HTTPError(
            "https://provider.example/v1/responses",
            503,
            "Service Unavailable",
            None,
            BytesIO(b'{"error":"auth unavailable"}'),
        )
        with (
            patch(
                "agent.client.request.urlopen",
                side_effect=[transient, Response()],
            ),
            patch("agent.client.time.sleep") as sleep,
        ):
            chunks = list(
                stream_post(
                    "https://provider.example/v1/responses",
                    {},
                    {"model": "test"},
                    7.0,
                    max_attempts=2,
                    retry_backoff_seconds=15.0,
                )
            )

        self.assertEqual(chunks, ["ok"])
        sleep.assert_called_once_with(15.0)

    def test_client_builds_openai_compatible_payload(self) -> None:
        seen: dict[str, object] = {}

        def transport(url, headers_, payload, timeout):
            seen["url"] = url
            seen["headers"] = headers_
            seen["payload"] = payload
            seen["timeout"] = timeout
            return {"choices": [{"message": {"content": "ok"}}]}

        client = ChatClient(
            Config(
                base_url="https://provider.example/v1",
                api_key="sk-test",
                model="test-model",
                state_dir=Path("/tmp/state"),
                timeout=7.0,
                temperature=0.2,
                reasoning_effort="xhigh",
                max_tokens=1024,
            ),
            transport=transport,
        )
        reply = client.complete([Message("user", "hello")])

        self.assertEqual(reply.content, "ok")
        self.assertEqual(seen["url"], "https://provider.example/v1/chat/completions")
        self.assertEqual(
            seen["headers"],
            {
                "content-type": "application/json",
                "user-agent": "OpenAI/Python 1.0.0",
                "authorization": "Bearer sk-test",
            },
        )
        self.assertEqual(seen["payload"]["model"], "test-model")
        self.assertEqual(seen["payload"]["temperature"], 0.2)
        self.assertEqual(seen["payload"]["reasoning_effort"], "xhigh")
        self.assertEqual(seen["payload"]["max_tokens"], 1024)
        self.assertEqual(seen["timeout"], 7.0)

    def test_endpoint_and_headers_helpers(self) -> None:
        self.assertEqual(endpoint("https://x/v1"), "https://x/v1/chat/completions")
        self.assertEqual(endpoint("https://x/v1/chat/completions"), "https://x/v1/chat/completions")
        self.assertEqual(endpoint("https://x/v1", "responses"), "https://x/v1/responses")
        self.assertEqual(endpoint("https://x/v1/responses", "responses"), "https://x/v1/responses")
        self.assertEqual(
            endpoint("https://api.anthropic.com/v1", "anthropic_messages"),
            "https://api.anthropic.com/v1/messages",
        )
        self.assertEqual(
            headers(None),
            {
                "content-type": "application/json",
                "user-agent": "OpenAI/Python 1.0.0",
            },
        )
        self.assertEqual(
            headers(
                "sk-ant-test",
                "anthropic_messages",
                anthropic_version="2023-06-01",
            ),
            {
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": "sk-ant-test",
            },
        )
        self.assertEqual(
            headers(
                None,
                "anthropic_messages",
                auth_token="auth-token-test",
                anthropic_version="2023-06-01",
            ),
            {
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "authorization": "Bearer auth-token-test",
            },
        )

    def test_client_normalizes_anthropic_native_tool_use(self) -> None:
        client = ChatClient(
            Config(
                base_url="https://provider.example/v1",
                api_key=None,
                auth_token="auth-token-test",
                model="claude-opus-4-8",
                state_dir=Path("/tmp/state"),
                wire_api="anthropic_messages",
            ),
            transport=lambda *_: {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "expert.retrieve",
                        "input": {
                            "args": '{"mode":"trajectory"}',
                            "reason": "study trajectories",
                        },
                    }
                ],
            },
        )

        reply = client.complete([Message("user", "learn")])

        self.assertEqual(
            json.loads(reply.content),
            {
                "tool": "expert.retrieve",
                "args": {"mode": "trajectory"},
                "reason": "study trajectories",
            },
        )

    def test_stream_transport_normalizes_anthropic_native_tool_use(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def __iter__(self):
                events = [
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "gripper.move_world",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(
                                {
                                    "args": json.dumps(
                                        {
                                            "arm": "right",
                                            "axis": "z",
                                            "sign": "-",
                                            "distance_mm": 20,
                                        }
                                    ),
                                    "reason": "descend",
                                }
                            ),
                        },
                    },
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_stop"},
                ]
                return iter(
                    [
                        ("data: " + json.dumps(event) + "\n").encode("utf-8")
                        for event in events
                    ]
                )

        with patch("agent.client.request.urlopen", return_value=Response()):
            chunks = list(
                stream_post(
                    "https://provider.example/v1/messages",
                    {},
                    {"model": "claude-opus-4-8", "stream": True},
                    7.0,
                )
            )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            json.loads(chunks[0]),
            {
                "action": "gripper.move_world",
                "arm": "right",
                "axis": "z",
                "sign": "-",
                "distance_mm": 20,
                "reason": "descend",
            },
        )

    def test_client_builds_responses_payload_and_parses_reply(self) -> None:
        seen: dict[str, object] = {}

        def transport(url, headers_, payload, timeout):
            seen["url"] = url
            seen["payload"] = payload
            return {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    },
                ],
            }

        client = ChatClient(
            Config(
                base_url="https://provider.example/v1",
                api_key="sk-test",
                model="test-model",
                state_dir=Path("/tmp/state"),
                wire_api="responses",
                temperature=0.0,
                reasoning_effort=None,
                max_tokens=2048,
            ),
            transport=transport,
        )
        reply = client.complete(
            [Message("system", "rules"), Message("user", "hello"), Message("assistant", "prior")]
        )

        self.assertEqual(reply.content, "ok")
        self.assertEqual(seen["url"], "https://provider.example/v1/responses")
        self.assertEqual(seen["payload"]["max_output_tokens"], 2048)
        self.assertNotIn("reasoning", seen["payload"])
        self.assertEqual(
            seen["payload"]["input"],
            [
                {"role": "system", "content": [{"type": "input_text", "text": "rules"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "prior"}]},
            ],
        )

    def test_responses_payload_resizes_image_and_uses_input_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            Image.new("RGB", (1280, 960), color=(10, 20, 30)).save(image_path)
            client = ChatClient(
                Config(
                    base_url="https://provider.example/v1",
                    api_key="sk-test",
                    model="test-model",
                    state_dir=Path(tmp),
                    wire_api="responses",
                    reasoning_effort=None,
                    request_image_max_width=768,
                    request_image_max_height=576,
                    request_image_jpeg_quality=75,
                )
            )

            payload = client.payload(
                [Message.with_parts("user", [Part.text_part("look"), Part.image_part(image_path)])]
            )
            image_part = payload["input"][0]["content"][1]
            self.assertEqual(image_part["type"], "input_image")
            self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))
            transmitted = Image.open(
                BytesIO(base64.b64decode(image_part["image_url"].split(",", 1)[1]))
            )
            self.assertEqual(transmitted.size, (768, 576))

    def test_responses_content_and_stream_delta_helpers(self) -> None:
        raw = {
            "output": [
                {"type": "reasoning", "content": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one"},
                        {"type": "output_text", "text": " two"},
                    ],
                },
            ]
        }
        self.assertEqual(content(raw, "responses"), "one two")
        self.assertEqual(
            delta_content({"type": "response.output_text.delta", "delta": "piece"}),
            "piece",
        )

    def test_client_builds_anthropic_payload_and_parses_reply(self) -> None:
        seen: dict[str, object] = {}

        def transport(url, headers_, payload, timeout):
            seen.update(
                {
                    "url": url,
                    "headers": headers_,
                    "payload": payload,
                    "timeout": timeout,
                }
            )
            return {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": " two"},
                ],
                "stop_reason": "end_turn",
            }

        client = ChatClient(
            Config(
                base_url="https://api.anthropic.com/v1",
                api_key="sk-ant-test",
                model="claude-test",
                state_dir=Path("/tmp/state"),
                wire_api="anthropic_messages",
                anthropic_version="2023-06-01",
                temperature=0.0,
                reasoning_effort="xhigh",
                max_tokens=2048,
            ),
            transport=transport,
        )
        reply = client.complete(
            [
                Message("system", "first rule"),
                Message("system", "second rule"),
                Message("user", "hello"),
                Message("assistant", "prior"),
            ]
        )

        self.assertEqual(reply.content, "one two")
        self.assertEqual(seen["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(
            seen["headers"],
            {
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": "sk-ant-test",
            },
        )
        self.assertEqual(
            seen["payload"],
            {
                "model": "claude-test",
                "system": "first rule\n\nsecond rule",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "prior"},
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        )

    def test_client_prefers_anthropic_auth_token_for_messages(self) -> None:
        seen: dict[str, object] = {}

        def transport(url, headers_, payload, timeout):
            seen["headers"] = headers_
            return {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            }

        client = ChatClient(
            Config(
                base_url="https://gateway.example/v1",
                api_key="fallback-api-key",
                auth_token="preferred-auth-token",
                model="claude-test",
                state_dir=Path("/tmp/state"),
                wire_api="anthropic_messages",
            ),
            transport=transport,
        )

        self.assertEqual(client.complete([Message("user", "hello")]).content, "ok")
        self.assertEqual(
            seen["headers"],
            {
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "authorization": "Bearer preferred-auth-token",
            },
        )

    def test_anthropic_payload_resizes_image_and_uses_base64_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            Image.new("RGB", (1280, 960), color=(10, 20, 30)).save(image_path)
            client = ChatClient(
                Config(
                    base_url="https://api.anthropic.com/v1",
                    api_key="sk-ant-test",
                    model="claude-test",
                    state_dir=Path(tmp),
                    wire_api="anthropic_messages",
                    reasoning_effort=None,
                    request_image_max_width=768,
                    request_image_max_height=576,
                    request_image_jpeg_quality=75,
                )
            )

            payload = client.payload(
                [
                    Message.with_parts(
                        "user",
                        [Part.text_part("look"), Part.image_part(image_path)],
                    )
                ]
            )
            image_part = payload["messages"][0]["content"][1]
            self.assertEqual(image_part["type"], "image")
            self.assertEqual(image_part["source"]["type"], "base64")
            self.assertEqual(image_part["source"]["media_type"], "image/jpeg")
            transmitted = Image.open(
                BytesIO(base64.b64decode(image_part["source"]["data"]))
            )
            self.assertEqual(transmitted.size, (768, 576))

    def test_anthropic_stream_delta_helper(self) -> None:
        self.assertEqual(
            delta_content(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "piece"},
                }
            ),
            "piece",
        )

    def test_client_optionally_resizes_request_images_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            Image.new("RGB", (1280, 960), color=(10, 20, 30)).save(image_path)
            source_bytes = image_path.read_bytes()
            client = ChatClient(
                Config(
                    base_url="https://provider.example/v1",
                    api_key="sk-test",
                    model="test-model",
                    state_dir=Path(tmp),
                    request_image_max_width=960,
                    request_image_max_height=720,
                    request_image_jpeg_quality=80,
                )
            )

            payload = client.payload(
                [
                    Message.with_parts(
                        "user",
                        [Part.text_part("look"), Part.image_part(image_path)],
                    )
                ]
            )
            image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
            self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
            transmitted = Image.open(
                BytesIO(base64.b64decode(image_url.split(",", 1)[1]))
            )

            self.assertEqual(transmitted.size, (960, 720))
            self.assertEqual(image_path.read_bytes(), source_bytes)

    def test_client_uses_compact_jpeg_transport_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "frame.png"
            Image.new("RGB", (64, 48), color=(10, 20, 30)).save(image_path)
            client = ChatClient(
                Config(
                    base_url="https://provider.example/v1",
                    api_key="sk-test",
                    model="test-model",
                    state_dir=Path(tmp),
                )
            )

            payload = client.payload(
                [Message.with_parts("user", [Part.image_part(image_path)])]
            )
            image_url = payload["messages"][0]["content"][0]["image_url"]["url"]

            self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
            transmitted = Image.open(
                BytesIO(base64.b64decode(image_url.split(",", 1)[1]))
            )
            self.assertEqual(transmitted.size, (64, 48))

    def test_client_limits_request_to_most_recent_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            messages = []
            for index in range(10):
                image_path = root / f"frame_{index}.png"
                Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_path)
                messages.append(
                    Message.with_parts(
                        "user",
                        [
                            Part.text_part(f"step {index}"),
                            Part.image_part(image_path, label=f"frame {index}"),
                        ],
                    )
                )
            client = ChatClient(
                Config(
                    base_url="https://provider.example/v1",
                    api_key="sk-test",
                    model="test-model",
                    state_dir=root,
                    request_max_images=8,
                )
            )

            payload = client.payload(messages)
            content_parts = [
                part
                for message in payload["messages"]
                for part in message["content"]
            ]
            images = [part for part in content_parts if part["type"] == "image_url"]
            omitted = [
                part["text"]
                for part in content_parts
                if part["type"] == "text" and "image omitted" in part["text"]
            ]

            self.assertEqual(len(images), 8)
            self.assertEqual(len(omitted), 2)
            self.assertIn("frame 0", omitted[0])
            self.assertIn("frame 1", omitted[1])

    def test_client_image_limit_is_strict_with_multiframe_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_parts = []
            for index in range(10):
                image_path = root / f"expert_{index}.png"
                Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_path)
                image_parts.append(
                    Part.image_part(image_path, label=f"expert {index}")
                )
            client = ChatClient(
                Config(
                    base_url="https://provider.example/v1",
                    api_key="sk-test",
                    model="test-model",
                    state_dir=root,
                    request_max_images=8,
                )
            )

            payload = client.payload(
                [Message.with_parts("user", [Part.text_part("demo"), *image_parts])]
            )
            parts = payload["messages"][0]["content"]

            self.assertEqual(
                sum(part["type"] == "image_url" for part in parts),
                8,
            )
            self.assertEqual(
                sum(
                    part["type"] == "text" and "image omitted" in part["text"]
                    for part in parts
                ),
                2,
            )

    def test_store_preserves_multimodal_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "frame.png"
            image.write_bytes(b"fake")
            store = Store(root)
            sid = store.create("demo")

            store.append(sid, Message("system", "system prompt"))
            store.append(sid, Message.with_parts("user", [Part.text_part("look"), Part.image_part(image, label="frame")]))
            messages = store.load(sid)

            self.assertEqual([message.role for message in messages], ["system", "user"])
            self.assertEqual(messages[1].text(), "look\n[image: frame]")
            self.assertTrue(store.path(sid).exists())

    def test_expert_store_retrieves_manifest_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_0"
            frames = demo / "frames"
            frames.mkdir(parents=True)
            image = frames / "side.png"
            image.write_bytes(b"fake")
            (demo / "expert_demo.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "image_style": {"text_label": False, "eepose_overlay": False},
                        "phases": [
                            {
                                "phase": "final_grasp",
                                "step": 2,
                                "note": "ready to close",
                                "target_eepose": {"xyz": [1, 2, 3], "rpy_deg": [4, 5, 6]},
                                "state": {"eepose_xyz": [1.1, 2.1, 3.1], "gripper_value": 1.0},
                                "images": {"side": "frames/side.png"},
                            }
                        ],
                        "trajectory": "eepose_trajectory.jsonl",
                    }
                ),
                encoding="utf-8",
            )
            (demo / "eepose_trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "step": 0,
                        "phase": "initial",
                        "command": "reset",
                        "executed": True,
                        "target_eepose": {"xyz": [0, 0, 0], "rpy_deg": [0, 0, 0]},
                        "state": {"eepose_xyz": [0, 0, 0], "gripper_value": 1.0},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "step": 1,
                        "phase": "final_grasp",
                        "command": "move_to_eepose",
                        "executed": False,
                        "target_eepose": {"xyz": [1, 2, 3], "rpy_deg": [4, 5, 6]},
                        "state": {"eepose_xyz": [0, 0, 0], "gripper_value": 1.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            store = ExpertStore(root)
            frame = store.find({"phase": "final", "view": "side", "seed": 0})

            self.assertEqual(frame.demo_id, "seed_0")
            self.assertEqual(frame.phase, "final_grasp")
            self.assertEqual(frame.path, image)
            self.assertTrue(frame.clean_for_model)
            self.assertIn("ready to close", frame.caption)
            self.assertIn("target_eepose_xyz", frame.caption)
            self.assertEqual(store.summary()["trajectories"], 1)
            self.assertEqual(store.summary()["clean_frames"], 1)
            trajectory = store.trajectory_text({"mode": "trajectory", "seed": 0})
            self.assertIn("Expert trajectory seed_0", trajectory)
            self.assertIn("target_xyz=[1, 2, 3]", trajectory)

    def test_expert_store_treats_missing_image_style_as_not_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_0"
            frames = demo / "frames"
            frames.mkdir(parents=True)
            image = frames / "side.png"
            image.write_bytes(b"fake")
            (demo / "expert_demo.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "phases": [
                            {
                                "phase": "final_grasp",
                                "images": {"side": "frames/side.png"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = ExpertStore(root)
            frame = store.find({"phase": "final", "view": "side", "seed": 0})

            self.assertFalse(frame.clean_for_model)
            self.assertEqual(store.summary()["clean_frames"], 0)

    def test_expert_store_formats_observation_only_trajectory_without_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_0"
            frame_dir = demo / "frames" / "00000"
            frame_dir.mkdir(parents=True)
            image = frame_dir / "active.png"
            image.write_bytes(b"fake")
            (demo / "eepose_trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "step": 0,
                        "images": {"active": "frames/00000/active.png"},
                        "state": {
                            "eepose_xyz": [0.1, 0.2, 0.3],
                            "eepose_rpy_deg": [1.0, 2.0, 3.0],
                            "gripper_value": 1.0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (demo / "expert_demo.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "trajectory": "eepose_trajectory.jsonl",
                        "phases": [],
                        "image_style": {"text_label": False, "eepose_overlay": False},
                    }
                ),
                encoding="utf-8",
            )

            store = ExpertStore(root)
            text = store.trajectory_text({"seed": 0})
            summary = store.summary()
            frame = store.find({"seed": 0, "step": 0, "view": "active"})

            self.assertIn("observed_ee=[0.1, 0.2, 0.3]", text)
            self.assertIn("image_views=['active']", text)
            self.assertNotIn("command=", text)
            self.assertNotIn("target_xyz=", text)
            self.assertEqual(summary["seeds"], [0])
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["views"], ["active"])
            self.assertTrue(summary["observation_only"])
            self.assertEqual(frame.phase, "observation")
            self.assertEqual(frame.path, image)
            self.assertTrue(frame.clean_for_model)

    def test_expert_store_returns_full_observation_only_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_0"
            demo.mkdir()
            rows = [
                {"step": step, "images": {}, "state": {"eepose_xyz": [step, 0, 0], "gripper_value": 1.0}}
                for step in range(100)
            ]
            (demo / "eepose_trajectory.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (demo / "expert_demo.json").write_text(
                json.dumps({"seed": 0, "trajectory": "eepose_trajectory.jsonl", "phases": []}),
                encoding="utf-8",
            )

            text = ExpertStore(root).trajectory_text({"seed": 0, "max_rows": 3})

            self.assertIn("step=0 ", text)
            self.assertIn("step=50 ", text)
            self.assertIn("step=99 ", text)
            self.assertEqual(text.count("step="), 100)
            self.assertNotIn("additional rows omitted", text)

    def test_observation_trajectory_retrieval_does_not_select_special_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_4"
            demo.mkdir()
            values = [1.0] * 13 + [1.0, 0.75, 0.5, 0.25, 0.0] + [0.0] * 12
            rows = [
                {"step": step, "images": {}, "state": {"eepose_xyz": [step, 0, 0], "gripper_value": value}}
                for step, value in enumerate(values)
            ]
            (demo / "eepose_trajectory.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (demo / "expert_demo.json").write_text(
                json.dumps({"seed": 4, "trajectory": "eepose_trajectory.jsonl", "phases": []}),
                encoding="utf-8",
            )

            text = ExpertStore(root).trajectory_text({"max_rows": 10})

            for step in range(len(rows)):
                self.assertIn(f"step={step} ", text)
            self.assertEqual(text.count("step="), len(rows))

    def test_expert_store_finds_observation_frame_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "seed_0"
            frame_dir = demo / "frames" / "00007"
            frame_dir.mkdir(parents=True)
            image = frame_dir / "side.png"
            image.write_bytes(b"fake")
            (demo / "expert_demo.json").write_text(
                json.dumps(
                    {
                        "seed": 0,
                        "image_style": {"text_label": False, "eepose_overlay": False},
                        "phases": [
                            {
                                "step": 7,
                                "phase": "trajectory",
                                "images": {"side": "frames/00007/side.png"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            frame = ExpertStore(root).find({"seed": 0, "step": 7, "view": "side"})

            self.assertEqual(frame.path, image)

    def test_expert_store_loads_video_only_rgb_sequence(self) -> None:
        class FakeReader:
            def __iter__(self):
                return iter([object(), object(), object(), object()])

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "center_high_qpos_replay.mp4"
            video.write_bytes(b"fake-video")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "task": "grasp_single_bottle_upright",
                        "seed": 0,
                        "video": video.name,
                        "view": "center_high",
                        "frame_count": 4,
                        "stored_eepose": False,
                        "stored_qpos_trajectory": False,
                        "stored_actions": False,
                    }
                ),
                encoding="utf-8",
            )

            def write_fake_frame(path, _frame):
                Path(path).write_bytes(b"fake-png")

            with patch("agent.robotwin.expert.imageio.get_reader", return_value=FakeReader()), patch(
                "agent.robotwin.expert.imageio.imwrite", side_effect=write_fake_frame
            ):
                store = ExpertStore(root)

            summary = store.summary()
            trajectory = store.trajectory_text({"mode": "trajectory", "seed": 0})
            frames = store.find_many({"seed": 0, "steps": [0, 2, 3], "view": "center_high"})

            self.assertTrue(summary["observation_only"])
            self.assertTrue(summary["video_only"])
            self.assertEqual(summary["frames"], 4)
            self.assertEqual([frame.step for frame in frames], [0, 2, 3])
            self.assertIn("source=video_only_rgb", trajectory)
            self.assertNotIn("observed_ee", trajectory)
            self.assertNotIn("qpos", trajectory)
            self.assertNotIn("command=", trajectory)


if __name__ == "__main__":
    unittest.main()
