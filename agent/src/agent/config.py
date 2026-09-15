from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MODEL = "openai/gpt-5.6-sol"
DEFAULT_WIRE_API = "chat_completions"
WIRE_APIS = {"anthropic_messages", "chat_completions", "responses"}
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TEMPERATURE: float | None = None
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_STREAM = True
DEFAULT_ANTHROPIC_THINKING = "enabled"
DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS = 1024
DEFAULT_MAX_STEPS = 100
DEFAULT_REQUEST_IMAGE_MAX_WIDTH = 1280
DEFAULT_REQUEST_IMAGE_MAX_HEIGHT = 960
DEFAULT_REQUEST_IMAGE_JPEG_QUALITY = 95
DEFAULT_REQUEST_MAX_IMAGES = 4
DEFAULT_TRANSIENT_HTTP_RETRY_ATTEMPTS = 6
DEFAULT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS = 1.5
DEFAULT_QUOTA_FALLBACK_COOLDOWN_SECONDS = 300.0
DEFAULT_CONTEXT_WINDOW_TOKENS = 262_144
DEFAULT_CONTEXT_COMPACTION_THRESHOLD = 0.80
DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_TURNS = 6
DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_IMAGES = 4


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str | None
    model: str
    state_dir: Path
    auth_token: str | None = None
    wire_api: str = DEFAULT_WIRE_API
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    timeout: float = 120.0
    temperature: float | None = DEFAULT_TEMPERATURE
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    stream: bool = DEFAULT_STREAM
    transient_http_retry_attempts: int = DEFAULT_TRANSIENT_HTTP_RETRY_ATTEMPTS
    transient_http_retry_backoff_seconds: float = (
        DEFAULT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS
    )
    quota_fallback_base_url: str | None = None
    quota_fallback_model: str | None = None
    quota_fallback_api_key: str | None = None
    quota_fallback_cooldown_seconds: float = DEFAULT_QUOTA_FALLBACK_COOLDOWN_SECONDS
    request_image_max_width: int | None = DEFAULT_REQUEST_IMAGE_MAX_WIDTH
    request_image_max_height: int | None = DEFAULT_REQUEST_IMAGE_MAX_HEIGHT
    request_image_jpeg_quality: int = DEFAULT_REQUEST_IMAGE_JPEG_QUALITY
    request_max_images: int | None = DEFAULT_REQUEST_MAX_IMAGES
    context_compaction_enabled: bool = True
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    context_compaction_threshold: float = DEFAULT_CONTEXT_COMPACTION_THRESHOLD
    context_compaction_keep_recent_turns: int = DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_TURNS
    context_compaction_keep_recent_images: int = DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_IMAGES
    response_language: str | None = None
    anthropic_thinking: str | None = DEFAULT_ANTHROPIC_THINKING
    anthropic_thinking_budget_tokens: int | None = DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS

    @property
    def redacted(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
            "auth_token": "***" if self.auth_token else None,
            "model": self.model,
            "wire_api": self.wire_api,
            "anthropic_version": self.anthropic_version,
            "state_dir": str(self.state_dir),
            "timeout": self.timeout,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "transient_http_retry_attempts": self.transient_http_retry_attempts,
            "transient_http_retry_backoff_seconds": (
                self.transient_http_retry_backoff_seconds
            ),
            "quota_fallback_base_url": self.quota_fallback_base_url,
            "quota_fallback_model": self.quota_fallback_model,
            "quota_fallback_api_key": "***" if self.quota_fallback_api_key else None,
            "quota_fallback_cooldown_seconds": self.quota_fallback_cooldown_seconds,
            "request_image_max_width": self.request_image_max_width,
            "request_image_max_height": self.request_image_max_height,
            "request_image_jpeg_quality": self.request_image_jpeg_quality,
            "request_max_images": self.request_max_images,
            "context_compaction_enabled": self.context_compaction_enabled,
            "context_window_tokens": self.context_window_tokens,
            "context_compaction_threshold": self.context_compaction_threshold,
            "context_compaction_keep_recent_turns": self.context_compaction_keep_recent_turns,
            "context_compaction_keep_recent_images": self.context_compaction_keep_recent_images,
            "response_language": self.response_language,
            "anthropic_thinking": self.anthropic_thinking,
            "anthropic_thinking_budget_tokens": self.anthropic_thinking_budget_tokens,
        }


def load(args) -> Config:
    anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL")
    anthropic_auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    argument_base_url = getattr(args, "base_url", None)
    agent_base_url = os.environ.get("AGENT_BASE_URL")
    base_url = (
        argument_base_url
        or agent_base_url
        or anthropic_base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    if (
        anthropic_base_url
        and not argument_base_url
        and not agent_base_url
        and not base_url.endswith(("/v1", "/v1/messages"))
    ):
        base_url += "/v1"
    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("AGENT_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    auth_token = (
        getattr(args, "auth_token", None)
        or os.environ.get("AGENT_AUTH_TOKEN")
        or anthropic_auth_token
    )
    model = (
        getattr(args, "model", None)
        or os.environ.get("AGENT_MODEL")
        or anthropic_model
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL
    )
    inferred_wire_api = (
        "anthropic_messages"
        if anthropic_base_url or anthropic_model or anthropic_auth_token
        else DEFAULT_WIRE_API
    )
    wire_api = (
        getattr(args, "wire_api", None)
        or os.environ.get("AGENT_WIRE_API")
        or inferred_wire_api
    ).strip().lower().replace("-", "_")
    if wire_api not in WIRE_APIS:
        raise ValueError(f"wire API must be one of: {', '.join(sorted(WIRE_APIS))}")
    anthropic_version = (
        getattr(args, "anthropic_version", None)
        or os.environ.get("AGENT_ANTHROPIC_VERSION")
        or DEFAULT_ANTHROPIC_VERSION
    ).strip()
    if not anthropic_version:
        raise ValueError("Anthropic API version cannot be empty")
    state_dir = Path(
        getattr(args, "state_dir", None)
        or os.environ.get("AGENT_STATE_DIR")
        or ".agent"
    )
    temperature_arg = getattr(args, "temperature", None)
    temperature_env = os.environ.get("AGENT_TEMPERATURE")
    if temperature_arg is not None:
        temperature = temperature_arg
    else:
        temperature_value = optional_string(
            None,
            temperature_env,
            default=str(DEFAULT_TEMPERATURE) if DEFAULT_TEMPERATURE is not None else None,
        )
        temperature = float(temperature_value) if temperature_value is not None else None
    reasoning_effort = optional_string(
        getattr(args, "reasoning_effort", None),
        os.environ.get("AGENT_REASONING_EFFORT"),
        default=DEFAULT_REASONING_EFFORT,
    )
    anthropic_thinking = optional_string(
        getattr(args, "anthropic_thinking", None),
        os.environ.get("AGENT_ANTHROPIC_THINKING"),
        default=DEFAULT_ANTHROPIC_THINKING,
    )
    anthropic_thinking_budget_tokens = optional_positive_int(
        getattr(args, "anthropic_thinking_budget_tokens", None),
        os.environ.get("AGENT_ANTHROPIC_THINKING_BUDGET_TOKENS"),
        name="Anthropic thinking budget tokens",
    )
    if anthropic_thinking_budget_tokens is None and anthropic_thinking == "enabled":
        anthropic_thinking_budget_tokens = DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS
    max_tokens = optional_positive_int(
        getattr(args, "max_tokens", None),
        os.environ.get("AGENT_MAX_TOKENS"),
        name="max tokens",
    ) or DEFAULT_MAX_TOKENS
    stream_value = os.environ.get("AGENT_STREAM")
    stream = (
        DEFAULT_STREAM
        if stream_value is None
        else parse_bool(stream_value, name="stream")
    )
    transient_http_retry_attempts = optional_positive_int(
        getattr(args, "transient_http_retry_attempts", None),
        os.environ.get("AGENT_TRANSIENT_HTTP_RETRY_ATTEMPTS"),
        name="transient HTTP retry attempts",
    ) or DEFAULT_TRANSIENT_HTTP_RETRY_ATTEMPTS
    retry_backoff_arg = getattr(
        args,
        "transient_http_retry_backoff_seconds",
        None,
    )
    retry_backoff_env = os.environ.get(
        "AGENT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS"
    )
    transient_http_retry_backoff_seconds = (
        retry_backoff_arg
        if retry_backoff_arg is not None
        else float(retry_backoff_env)
        if retry_backoff_env is not None
        else DEFAULT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS
    )
    if transient_http_retry_backoff_seconds <= 0:
        raise ValueError("transient HTTP retry backoff seconds must be positive")
    quota_fallback_base_url = optional_string(
        getattr(args, "quota_fallback_base_url", None),
        os.environ.get("AGENT_QUOTA_FALLBACK_BASE_URL"),
        default=None,
    )
    if quota_fallback_base_url:
        quota_fallback_base_url = quota_fallback_base_url.rstrip("/")
    quota_fallback_model = optional_string(
        getattr(args, "quota_fallback_model", None),
        os.environ.get("AGENT_QUOTA_FALLBACK_MODEL"),
        default=None,
    )
    quota_fallback_api_key = optional_string(
        getattr(args, "quota_fallback_api_key", None),
        os.environ.get("AGENT_QUOTA_FALLBACK_API_KEY"),
        default=None,
    )
    if bool(quota_fallback_base_url) != bool(quota_fallback_model):
        raise ValueError(
            "quota fallback base URL and model must be configured together"
        )
    quota_fallback_cooldown_arg = getattr(
        args,
        "quota_fallback_cooldown_seconds",
        None,
    )
    quota_fallback_cooldown_env = os.environ.get(
        "AGENT_QUOTA_FALLBACK_COOLDOWN_SECONDS"
    )
    quota_fallback_cooldown_seconds = (
        quota_fallback_cooldown_arg
        if quota_fallback_cooldown_arg is not None
        else float(quota_fallback_cooldown_env)
        if quota_fallback_cooldown_env is not None
        else DEFAULT_QUOTA_FALLBACK_COOLDOWN_SECONDS
    )
    if quota_fallback_cooldown_seconds <= 0:
        raise ValueError("quota fallback cooldown seconds must be positive")
    request_image_max_width = optional_positive_int(
        getattr(args, "request_image_max_width", None),
        os.environ.get("AGENT_REQUEST_IMAGE_MAX_WIDTH"),
        name="request image max width",
    ) or DEFAULT_REQUEST_IMAGE_MAX_WIDTH
    request_image_max_height = optional_positive_int(
        getattr(args, "request_image_max_height", None),
        os.environ.get("AGENT_REQUEST_IMAGE_MAX_HEIGHT"),
        name="request image max height",
    ) or DEFAULT_REQUEST_IMAGE_MAX_HEIGHT
    request_image_jpeg_quality = optional_positive_int(
        getattr(args, "request_image_jpeg_quality", None),
        os.environ.get("AGENT_REQUEST_IMAGE_JPEG_QUALITY"),
        name="request image JPEG quality",
    ) or DEFAULT_REQUEST_IMAGE_JPEG_QUALITY
    if request_image_jpeg_quality > 95:
        raise ValueError("request image JPEG quality must be between 1 and 95")
    request_max_images = optional_nonnegative_int(
        getattr(args, "request_max_images", None),
        os.environ.get("AGENT_REQUEST_MAX_IMAGES"),
        name="request max images",
    )
    if request_max_images is None:
        request_max_images = DEFAULT_REQUEST_MAX_IMAGES
    context_window_tokens = optional_positive_int(
        getattr(args, "context_window_tokens", None),
        os.environ.get("AGENT_CONTEXT_WINDOW_TOKENS"),
        name="context window tokens",
    ) or DEFAULT_CONTEXT_WINDOW_TOKENS
    threshold_arg = getattr(args, "context_compaction_threshold", None)
    threshold_env = os.environ.get("AGENT_CONTEXT_COMPACTION_THRESHOLD")
    context_compaction_threshold = (
        threshold_arg
        if threshold_arg is not None
        else float(threshold_env)
        if threshold_env is not None
        else DEFAULT_CONTEXT_COMPACTION_THRESHOLD
    )
    if not 0.1 <= context_compaction_threshold <= 0.95:
        raise ValueError("context compaction threshold must be between 0.1 and 0.95")
    context_compaction_keep_recent_turns = optional_positive_int(
        getattr(args, "context_compaction_keep_recent_turns", None),
        os.environ.get("AGENT_CONTEXT_COMPACTION_KEEP_RECENT_TURNS"),
        name="context compaction recent turns",
    ) or DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_TURNS
    context_compaction_keep_recent_images = optional_nonnegative_int(
        getattr(args, "context_compaction_keep_recent_images", None),
        os.environ.get("AGENT_CONTEXT_COMPACTION_KEEP_RECENT_IMAGES"),
        name="context compaction recent images",
    )
    if context_compaction_keep_recent_images is None:
        context_compaction_keep_recent_images = DEFAULT_CONTEXT_COMPACTION_KEEP_RECENT_IMAGES
    context_compaction_disabled_arg = bool(
        getattr(args, "no_context_compaction", False)
    )
    enabled_env = os.environ.get("AGENT_CONTEXT_COMPACTION_ENABLED")
    if context_compaction_disabled_arg:
        context_compaction_enabled = False
    elif enabled_env is not None:
        context_compaction_enabled = parse_bool(enabled_env, name="context compaction enabled")
    else:
        context_compaction_enabled = True
    response_language = optional_string(
        getattr(args, "response_language", None),
        os.environ.get("AGENT_RESPONSE_LANGUAGE"),
        default=None,
    )
    return Config(
        base_url=base_url,
        api_key=api_key,
        model=model,
        state_dir=state_dir,
        auth_token=auth_token,
        wire_api=wire_api,
        anthropic_version=anthropic_version,
        timeout=float(getattr(args, "timeout", 120.0)),
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        stream=stream,
        transient_http_retry_attempts=transient_http_retry_attempts,
        transient_http_retry_backoff_seconds=(
            transient_http_retry_backoff_seconds
        ),
        quota_fallback_base_url=quota_fallback_base_url,
        quota_fallback_model=quota_fallback_model,
        quota_fallback_api_key=quota_fallback_api_key,
        quota_fallback_cooldown_seconds=quota_fallback_cooldown_seconds,
        request_image_max_width=request_image_max_width,
        request_image_max_height=request_image_max_height,
        request_image_jpeg_quality=request_image_jpeg_quality,
        request_max_images=request_max_images,
        context_compaction_enabled=context_compaction_enabled,
        context_window_tokens=context_window_tokens,
        context_compaction_threshold=context_compaction_threshold,
        context_compaction_keep_recent_turns=context_compaction_keep_recent_turns,
        context_compaction_keep_recent_images=context_compaction_keep_recent_images,
        response_language=response_language,
        anthropic_thinking=anthropic_thinking,
        anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
    )


def optional_positive_int(
    argument: int | None,
    environment: str | None,
    *,
    name: str,
) -> int | None:
    value = argument if argument is not None else int(environment) if environment else None
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def optional_nonnegative_int(
    argument: int | None,
    environment: str | None,
    *,
    name: str,
) -> int | None:
    value = argument if argument is not None else int(environment) if environment else None
    if value is not None and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def optional_string(
    argument: str | None,
    environment: str | None,
    *,
    default: str | None,
) -> str | None:
    value = argument if argument is not None else environment
    if value is None:
        return default
    normalized = value.strip()
    if normalized.lower() in {"none", "null", "off"}:
        return None
    return normalized


def parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")
