from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from http.client import RemoteDisconnected
import json
import os
import re
import signal
import socket
import ssl
import sys
import time
from typing import Callable, Iterable, Iterator
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from .config import Config, DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS
from .session import Message, Part


Transport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]
StreamTransport = Callable[[str, dict[str, str], dict[str, object], float], Iterable[str]]
RouteEventHandler = Callable[[dict[str, object]], None]
TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504, 520, 522, 524}
TRANSIENT_CONNECTION_ERRORS = (
    ConnectionResetError,
    RemoteDisconnected,
    ssl.SSLError,
    TimeoutError,
    socket.timeout,
)
QUOTA_RESET_PATTERN = re.compile(
    r"reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})",
    re.IGNORECASE,
)
QUOTA_EXHAUSTION_MARKERS = (
    "accountquotaexceeded",
    "quota exhausted",
    "insufficient_quota",
    "insufficient quota",
    "insufficient_balance",
    "insufficient balance",
    "usage_limit_exceeded",
    "usage limit exceeded",
    "daily_limit_exceeded",
    "daily limit exceeded",
    "daily usage limit",
    "weekly_limit_exceeded",
    "weekly limit exceeded",
    "monthly_limit_exceeded",
    "monthly limit exceeded",
    "five_hour_limit",
    "five hour limit",
    "5-hour limit",
    "5 hour limit",
    "plan limit exceeded",
)
EXPLICIT_QUOTA_CODE_MARKERS = (
    "accountquotaexceeded",
    "insufficient_quota",
    "insufficient_balance",
    "usage_limit_exceeded",
    "daily_limit_exceeded",
    "weekly_limit_exceeded",
    "monthly_limit_exceeded",
    "five_hour_limit",
)
RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "too many requests",
    "requests per minute",
    "tokens per minute",
    "concurrency limit",
    "concurrent request",
    "rpm limit",
    "tpm limit",
)


class ProviderHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status} from provider: {body}")


@dataclass(frozen=True)
class ProviderRoute:
    base_url: str
    model: str
    api_key: str | None


def pause_on_transient_error(detail: str) -> bool:
    value = os.environ.get("AGENT_PAUSE_ON_TRANSIENT_HTTP", "")
    if value.strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    print(
        f"INFRA pause: {detail}. Resume this process with SIGCONT (pid={os.getpid()}).",
        file=sys.stderr,
        flush=True,
    )
    os.kill(os.getpid(), signal.SIGSTOP)
    return True


@dataclass(frozen=True)
class Reply:
    content: str
    raw: dict[str, object]


class ChatClient:
    def __init__(
        self,
        config: Config,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
    ):
        self.config = config
        self._primary_route = ProviderRoute(
            config.base_url,
            config.model,
            config.api_key,
        )
        self._fallback_route = (
            ProviderRoute(
                config.quota_fallback_base_url,
                config.quota_fallback_model,
                config.quota_fallback_api_key or config.api_key,
            )
            if config.quota_fallback_base_url and config.quota_fallback_model
            else None
        )
        self._quota_fallback_until = 0.0
        self._quota_fallback_active = False
        self._route_event_handler: RouteEventHandler | None = None
        self._routing_events: list[dict[str, object]] = []
        self._logical_request_count = 0
        self._primary_request_count = 0
        self._fallback_request_count = 0
        self.transport = transport or (
            lambda url, headers_, payload, timeout: post(
                url,
                headers_,
                payload,
                timeout,
                max_attempts=config.transient_http_retry_attempts,
                retry_backoff_seconds=config.transient_http_retry_backoff_seconds,
                fail_fast_quota=self._fallback_route is not None,
            )
        )
        self.stream_transport = stream_transport or (
            lambda url, headers_, payload, timeout: stream_post(
                url,
                headers_,
                payload,
                timeout,
                max_attempts=config.transient_http_retry_attempts,
                retry_backoff_seconds=config.transient_http_retry_backoff_seconds,
                fail_fast_quota=self._fallback_route is not None,
            )
        )

    def complete(self, messages: Iterable[Message]) -> Reply:
        payload = self.payload(list(messages))
        request_index = self._next_request_index()
        route = self._selected_route()
        try:
            raw = self._complete_on_route(route, payload)
        except Exception as exc:
            if not self._can_fallback(route, exc):
                raise
            route = self._activate_quota_fallback(exc, request_index=request_index)
            raw = self._complete_on_route(route, payload)
        return Reply(content=content(raw, self.config.wire_api), raw=raw)

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        payload = self.payload(list(messages))
        payload["stream"] = True
        request_index = self._next_request_index()
        route = self._selected_route()
        while True:
            emitted = False
            try:
                for chunk in self._stream_on_route(
                    route,
                    payload,
                ):
                    emitted = True
                    yield chunk
            except Exception as exc:
                if not self._can_fallback(route, exc):
                    raise
                route = self._activate_quota_fallback(
                    exc,
                    request_index=request_index,
                )
                if emitted:
                    raise
                continue
            return

    def begin_run(self) -> None:
        """Start a new independent run and probe the primary route first."""
        self._quota_fallback_active = False
        self._quota_fallback_until = 0.0
        self._routing_events = []
        self._logical_request_count = 0
        self._primary_request_count = 0
        self._fallback_request_count = 0

    def set_route_event_handler(
        self,
        handler: RouteEventHandler | None,
    ) -> None:
        self._route_event_handler = handler

    def routing_metadata(self) -> dict[str, object]:
        fallback = self._fallback_route
        return {
            "primary": route_metadata(self._primary_route),
            "fallback": route_metadata(fallback) if fallback is not None else None,
            "fallback_configured": fallback is not None,
            "fallback_used": bool(self._routing_events),
            "fallback_active": self._quota_fallback_active,
            "active_route": "fallback" if self._quota_fallback_active else "primary",
            "switch_count": len(self._routing_events),
            "logical_request_count": self._logical_request_count,
            "primary_request_count": self._primary_request_count,
            "fallback_request_count": self._fallback_request_count,
            "fallback_until": (
                datetime.fromtimestamp(self._quota_fallback_until)
                .astimezone()
                .isoformat(timespec="seconds")
                if self._quota_fallback_until
                else None
            ),
            "sticky_for_run": True,
        }

    @property
    def routing_events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._routing_events)

    def _next_request_index(self) -> int:
        self._logical_request_count += 1
        return self._logical_request_count

    def _selected_route(self) -> ProviderRoute:
        if self._fallback_route is not None and self._quota_fallback_active:
            return self._fallback_route
        return self._primary_route

    def _complete_on_route(
        self,
        route: ProviderRoute,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self._record_route_request(route)
        routed_payload = dict(payload)
        routed_payload["model"] = route.model
        return self.transport(
            endpoint(route.base_url, self.config.wire_api),
            self._route_headers(route),
            routed_payload,
            self.config.timeout,
        )

    def _stream_on_route(
        self,
        route: ProviderRoute,
        payload: dict[str, object],
    ) -> Iterable[str]:
        self._record_route_request(route)
        routed_payload = dict(payload)
        routed_payload["model"] = route.model
        return self.stream_transport(
            endpoint(route.base_url, self.config.wire_api),
            self._route_headers(route),
            routed_payload,
            self.config.timeout,
        )

    def _route_headers(self, route: ProviderRoute) -> dict[str, str]:
        return headers(
            route.api_key,
            self.config.wire_api,
            auth_token=(
                self.config.auth_token if route is self._primary_route else None
            ),
            anthropic_version=self.config.anthropic_version,
        )

    def _can_fallback(self, route: ProviderRoute, exc: Exception) -> bool:
        return (
            route is self._primary_route
            and self._fallback_route is not None
            and is_account_quota_exceeded_error(exc)
        )

    def _activate_quota_fallback(
        self,
        exc: Exception,
        *,
        request_index: int,
    ) -> ProviderRoute:
        assert self._fallback_route is not None
        now = time.time()
        self._quota_fallback_until = quota_reset_epoch(
            str(exc),
            now=now,
            default_cooldown_seconds=self.config.quota_fallback_cooldown_seconds,
        )
        self._quota_fallback_active = True
        resume_at = datetime.fromtimestamp(self._quota_fallback_until).astimezone()
        error_info = quota_error_info(exc)
        error_info["detail"] = redact_secrets(
            str(error_info.get("detail") or ""),
            self._primary_route.api_key,
            self._fallback_route.api_key,
            self.config.auth_token,
        )
        event = {
            "from": "primary",
            "to": "fallback",
            "reason": "quota_exhausted",
            "request_index": request_index,
            "http_status": error_info.get("http_status"),
            "provider_error_code": error_info.get("provider_error_code"),
            "detail": error_info.get("detail"),
            "primary": route_metadata(self._primary_route),
            "fallback": route_metadata(self._fallback_route),
            "fallback_until": resume_at.isoformat(timespec="seconds"),
            "sticky_for_run": True,
        }
        self._routing_events.append(event)
        if self._route_event_handler is not None:
            self._route_event_handler(dict(event))
        print(
            "Primary provider quota exhausted; using the API fallback for the "
            "remainder of this run. A new run will probe the primary route first "
            f"(reported reset: {resume_at.isoformat(timespec='seconds')}).",
            file=sys.stderr,
            flush=True,
        )
        return self._fallback_route

    def _record_route_request(self, route: ProviderRoute) -> None:
        if route is self._fallback_route:
            self._fallback_request_count += 1
        else:
            self._primary_request_count += 1

    def payload(self, messages: Iterable[Message]) -> dict[str, object]:
        if self.config.wire_api == "responses":
            return self.responses_payload(messages)
        if self.config.wire_api == "anthropic_messages":
            return self.anthropic_payload(messages)
        return self.chat_completions_payload(messages)

    def chat_completions_payload(self, messages: Iterable[Message]) -> dict[str, object]:
        messages = limit_request_images(messages, self.config.request_max_images)
        result: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                message.api(
                    image_max_width=self.config.request_image_max_width,
                    image_max_height=self.config.request_image_max_height,
                    image_jpeg_quality=self.config.request_image_jpeg_quality,
                )
                for message in messages
            ],
        }
        if self.config.temperature is not None:
            result["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            result["reasoning_effort"] = self.config.reasoning_effort
        if self.config.max_tokens is not None:
            result["max_tokens"] = self.config.max_tokens
        return result

    def responses_payload(self, messages: Iterable[Message]) -> dict[str, object]:
        messages = limit_request_images(messages, self.config.request_max_images)
        result: dict[str, object] = {
            "model": self.config.model,
            "input": [
                message.responses_api(
                    image_max_width=self.config.request_image_max_width,
                    image_max_height=self.config.request_image_max_height,
                    image_jpeg_quality=self.config.request_image_jpeg_quality,
                )
                for message in messages
            ],
        }
        if self.config.temperature is not None:
            result["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            result["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.max_tokens is not None:
            result["max_output_tokens"] = self.config.max_tokens
        return result

    def anthropic_payload(self, messages: Iterable[Message]) -> dict[str, object]:
        messages = limit_request_images(messages, self.config.request_max_images)
        system_parts: list[str] = []
        conversation: list[dict[str, object]] = []
        for message in messages:
            if message.role == "system":
                text = message.text().strip()
                if text:
                    system_parts.append(text)
                continue
            conversation.append(
                message.anthropic_api(
                    image_max_width=self.config.request_image_max_width,
                    image_max_height=self.config.request_image_max_height,
                    image_jpeg_quality=self.config.request_image_jpeg_quality,
                )
            )
        result: dict[str, object] = {
            "model": self.config.model,
            "messages": conversation,
            "max_tokens": self.config.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system_parts:
            result["system"] = "\n\n".join(system_parts)
        if self.config.temperature is not None:
            result["temperature"] = self.config.temperature
        if self.config.anthropic_thinking is not None:
            thinking: dict[str, object] = {"type": self.config.anthropic_thinking}
            if (
                self.config.anthropic_thinking == "enabled"
                and self.config.anthropic_thinking_budget_tokens is not None
            ):
                thinking["budget_tokens"] = self.config.anthropic_thinking_budget_tokens
            result["thinking"] = thinking
        return result


def limit_request_images(
    messages: Iterable[Message],
    max_images: int | None,
) -> list[Message]:
    result = list(messages)
    if max_images is None:
        return result

    image_positions = [
        (message_index, part_index)
        for message_index, message in enumerate(result)
        if isinstance(message.content, list)
        for part_index, part in enumerate(message.content)
        if part.type == "image"
    ]
    retained = set(image_positions[-max_images:]) if max_images else set()
    if len(retained) == len(image_positions):
        return result

    limited: list[Message] = []
    for message_index, message in enumerate(result):
        if not isinstance(message.content, list):
            limited.append(message)
            continue
        parts = [
            part
            if part.type != "image" or (message_index, part_index) in retained
            else Part.text_part(
                f"{part.preview()} [image omitted from API request due to provider image limit]"
            )
            for part_index, part in enumerate(message.content)
        ]
        limited.append(
            Message(message.role, parts, created_at=message.created_at)
        )
    return limited


def endpoint(base_url: str, wire_api: str = "chat_completions") -> str:
    suffixes = {
        "anthropic_messages": "/messages",
        "chat_completions": "/chat/completions",
        "responses": "/responses",
    }
    try:
        suffix = suffixes[wire_api]
    except KeyError as exc:
        raise ValueError(f"unsupported wire API: {wire_api}") from exc
    return base_url if base_url.endswith(suffix) else base_url.rstrip("/") + suffix


def headers(
    api_key: str | None,
    wire_api: str = "chat_completions",
    *,
    auth_token: str | None = None,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
) -> dict[str, str]:
    result = {"content-type": "application/json"}
    if wire_api == "anthropic_messages":
        result["anthropic-version"] = anthropic_version
        if auth_token:
            result["authorization"] = f"Bearer {auth_token}"
        elif api_key:
            result["x-api-key"] = api_key
        return result
    result["user-agent"] = "OpenAI/Python 1.0.0"
    credential = auth_token or api_key
    if credential:
        result["authorization"] = f"Bearer {credential}"
    return result


def post(
    url: str,
    headers_: dict[str, str],
    payload: dict[str, object],
    timeout: float,
    *,
    max_attempts: int = 6,
    retry_backoff_seconds: float = 1.5,
    fail_fast_quota: bool = False,
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers_, method="POST")
    last_error: error.URLError | OSError | None = None
    for attempt in range(max_attempts):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if fail_fast_quota and is_account_quota_exceeded(exc.code, body):
                raise ProviderHTTPError(exc.code, body) from exc
            if exc.code in TRANSIENT_HTTP_STATUS and pause_on_transient_error(
                f"HTTP {exc.code} from provider: {body}"
            ):
                continue
            if exc.code in TRANSIENT_HTTP_STATUS and attempt < max_attempts - 1:
                time.sleep(retry_backoff_seconds * (attempt + 1))
                continue
            raise ProviderHTTPError(exc.code, body) from exc
        except error.URLError as exc:
            last_error = exc
            if pause_on_transient_error(f"provider request failed: {exc.reason}"):
                continue
            if attempt == max_attempts - 1:
                break
            time.sleep(retry_backoff_seconds * (attempt + 1))
        except TRANSIENT_CONNECTION_ERRORS as exc:
            last_error = exc
            if pause_on_transient_error(f"provider request failed: {exc}"):
                continue
            if attempt == max_attempts - 1:
                break
            time.sleep(retry_backoff_seconds * (attempt + 1))
    assert last_error is not None
    detail = last_error.reason if isinstance(last_error, error.URLError) else str(last_error)
    raise RuntimeError(f"provider request failed: {detail}") from last_error


def stream_post(
    url: str,
    headers_: dict[str, str],
    payload: dict[str, object],
    timeout: float,
    *,
    max_attempts: int = 6,
    retry_backoff_seconds: float = 1.5,
    fail_fast_quota: bool = False,
) -> Iterator[str]:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_attempts):
        req = request.Request(url, data=data, headers=headers_, method="POST")
        emitted = False
        anthropic_tool_blocks: dict[int, dict[str, object]] = {}
        try:
            with request.urlopen(req, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    value = line.removeprefix("data:").strip()
                    if value == "[DONE]":
                        return
                    try:
                        chunk = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("type") == "error":
                        raise provider_stream_error(chunk.get("error"))
                    tool_command = anthropic_stream_tool_command(
                        chunk,
                        anthropic_tool_blocks,
                    )
                    if tool_command:
                        emitted = True
                        yield tool_command
                    delta = delta_content(chunk)
                    if delta:
                        emitted = True
                        yield delta
                return
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (
                not emitted
                and fail_fast_quota
                and is_account_quota_exceeded(exc.code, body)
            ):
                raise ProviderHTTPError(exc.code, body) from exc
            if (
                not emitted
                and exc.code in TRANSIENT_HTTP_STATUS
                and pause_on_transient_error(f"HTTP {exc.code} from provider: {body}")
            ):
                continue
            if (
                not emitted
                and exc.code in TRANSIENT_HTTP_STATUS
                and attempt < max_attempts - 1
            ):
                time.sleep(retry_backoff_seconds * (attempt + 1))
                continue
            raise ProviderHTTPError(exc.code, body) from exc
        except error.URLError as exc:
            if not emitted and pause_on_transient_error(
                f"provider request failed: {exc.reason}"
            ):
                continue
            if not emitted and attempt < max_attempts - 1:
                time.sleep(retry_backoff_seconds * (attempt + 1))
                continue
            raise RuntimeError(f"provider request failed: {exc.reason}") from exc
        except TRANSIENT_CONNECTION_ERRORS as exc:
            if not emitted and pause_on_transient_error(
                f"provider request failed: {exc}"
            ):
                continue
            if not emitted and attempt < max_attempts - 1:
                time.sleep(retry_backoff_seconds * (attempt + 1))
                continue
            raise RuntimeError(f"provider request failed: {exc}") from exc


def route_metadata(route: ProviderRoute) -> dict[str, object]:
    return {
        "base_url": redacted_base_url(route.base_url),
        "model": route.model,
    }


def redacted_base_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def redact_secrets(value: str, *secrets: str | None) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "***")
    return result


def is_account_quota_exceeded(status: int, body: str) -> bool:
    if status not in {403, 429}:
        return False
    detail = body.lower()
    if any(marker in detail for marker in EXPLICIT_QUOTA_CODE_MARKERS):
        return True
    if any(marker in detail for marker in RATE_LIMIT_MARKERS):
        return False
    return any(marker in detail for marker in QUOTA_EXHAUSTION_MARKERS)


def is_account_quota_exceeded_error(exc: Exception) -> bool:
    if isinstance(exc, ProviderHTTPError):
        return is_account_quota_exceeded(exc.status, exc.body)
    detail = str(exc)
    match = re.search(r"HTTP\s+(403|429)\b", detail, re.IGNORECASE)
    return bool(match) and is_account_quota_exceeded(int(match.group(1)), detail)


def quota_error_info(exc: Exception) -> dict[str, object]:
    status = exc.status if isinstance(exc, ProviderHTTPError) else None
    body = exc.body if isinstance(exc, ProviderHTTPError) else str(exc)
    provider_error_code: str | None = None
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        error_value = value.get("error")
        candidates = [value]
        if isinstance(error_value, dict):
            candidates.insert(0, error_value)
        for candidate in candidates:
            for key in ("code", "type", "reason"):
                code = candidate.get(key)
                if isinstance(code, str) and code.strip():
                    provider_error_code = code.strip()
                    break
            if provider_error_code:
                break
    detail = re.sub(r"\s+", " ", body).strip()
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return {
        "http_status": status,
        "provider_error_code": provider_error_code,
        "detail": detail,
    }


def provider_stream_error(value: object) -> Exception:
    body = json.dumps(value, ensure_ascii=False)
    status = 500
    if isinstance(value, dict):
        raw_status = value.get("status") or value.get("status_code")
        raw_code = value.get("code")
        if isinstance(raw_status, int):
            status = raw_status
        elif isinstance(raw_code, int):
            status = raw_code
        elif is_account_quota_exceeded(429, body):
            status = 429
    return ProviderHTTPError(status, body)


def quota_reset_epoch(
    detail: str,
    *,
    now: float,
    default_cooldown_seconds: float,
) -> float:
    match = QUOTA_RESET_PATTERN.search(detail)
    if match:
        try:
            reset_at = datetime.strptime(
                match.group(1),
                "%Y-%m-%d %H:%M:%S %z",
            ).timestamp()
            if reset_at > now:
                return reset_at + 1.0
        except ValueError:
            pass
    return now + default_cooldown_seconds


def content(raw: dict[str, object], wire_api: str = "chat_completions") -> str:
    if wire_api == "responses":
        return responses_content(raw)
    if wire_api == "anthropic_messages":
        return anthropic_content(raw)
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("provider response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("provider choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("provider response missing message")
    value = message.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    raise RuntimeError("provider response missing text content")


def responses_content(raw: dict[str, object]) -> str:
    direct = raw.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = raw.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            value = item.get("content")
            if not isinstance(value, list):
                continue
            for part in value:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "".join(parts)
    status = raw.get("status")
    details = raw.get("incomplete_details")
    suffix = f" (status={status}, incomplete_details={details})" if status or details else ""
    raise RuntimeError(f"provider response missing output text{suffix}")


def anthropic_content(raw: dict[str, object]) -> str:
    value = raw.get("content")
    if isinstance(value, list):
        for part in value:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                return anthropic_tool_use_command(part)
        parts = [
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        text = "".join(part for part in parts if isinstance(part, str))
        if text:
            return text
    stop_reason = raw.get("stop_reason")
    raise RuntimeError(
        f"provider response missing Anthropic text content (stop_reason={stop_reason})"
    )


def anthropic_tool_use_command(part: dict[str, object]) -> str:
    name = part.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("provider Anthropic tool_use block is missing a tool name")
    raw_input = part.get("input")
    tool_input = raw_input if isinstance(raw_input, dict) else {}
    reason = tool_input.get("reason")
    raw_args = tool_input.get("args") if "args" in tool_input else None
    if isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"provider Anthropic tool_use args are not valid JSON: {exc.msg}"
            ) from exc
        args = parsed_args if isinstance(parsed_args, dict) else {"value": parsed_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    elif raw_args is not None:
        args = {"value": raw_args}
    else:
        args = {
            key: value
            for key, value in tool_input.items()
            if key != "reason"
        }

    if name == "stop":
        command: dict[str, object] = {"stop": True}
    elif name in {
        "camera.history",
        "expert.frame",
        "expert.learn",
        "expert.retrieve",
        "geometry.verify",
    }:
        command = {"tool": name, "args": args}
    else:
        command = {"action": name, **args}
    if isinstance(reason, str) and reason:
        command["reason"] = reason
    return json.dumps(command, ensure_ascii=False, separators=(",", ":"))


def anthropic_stream_tool_command(
    raw: dict[str, object],
    blocks: dict[int, dict[str, object]],
) -> str:
    event_type = raw.get("type")
    index = raw.get("index")
    if not isinstance(index, int):
        return ""
    if event_type == "content_block_start":
        block = raw.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            blocks[index] = {
                "name": block.get("name"),
                "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                "partial_json": "",
            }
        return ""
    if event_type == "content_block_delta" and index in blocks:
        delta = raw.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                blocks[index]["partial_json"] = (
                    str(blocks[index].get("partial_json", "")) + partial
                )
        return ""
    if event_type != "content_block_stop" or index not in blocks:
        return ""

    block = blocks.pop(index)
    partial_json = block.pop("partial_json", "")
    if isinstance(partial_json, str) and partial_json:
        try:
            parsed_input = json.loads(partial_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"provider Anthropic streamed tool_use input is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed_input, dict):
            raise RuntimeError("provider Anthropic streamed tool_use input is not an object")
        block["input"] = parsed_input
    return anthropic_tool_use_command(block)


def delta_content(raw: dict[str, object]) -> str:
    if raw.get("type") == "response.output_text.delta":
        value = raw.get("delta")
        return value if isinstance(value, str) else ""
    if raw.get("type") == "content_block_delta":
        delta = raw.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            value = delta.get("text")
            return value if isinstance(value, str) else ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    value = delta.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return ""
