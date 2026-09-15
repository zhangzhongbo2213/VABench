from __future__ import annotations

from dataclasses import dataclass, field
import base64
from functools import lru_cache
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4

from PIL import Image

from .timeutil import now


VALID_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Part:
    type: str
    text: str | None = None
    path: str | None = None
    url: str | None = None
    mime: str | None = None
    detail: str | None = None
    label: str | None = None

    @classmethod
    def text_part(cls, text: str) -> "Part":
        return cls(type="text", text=text)

    @classmethod
    def image_part(cls, path: str | Path, *, label: str | None = None, detail: str | None = None) -> "Part":
        return cls(type="image", path=str(path), label=label, detail=detail)

    @classmethod
    def from_raw(cls, raw: Any) -> "Part":
        if isinstance(raw, Part):
            return raw
        if isinstance(raw, str):
            return cls.text_part(raw)
        if not isinstance(raw, dict):
            raise ValueError(f"invalid message part: {raw!r}")
        return cls(
            type=str(raw["type"]),
            text=raw.get("text"),
            path=raw.get("path"),
            url=raw.get("url"),
            mime=raw.get("mime"),
            detail=raw.get("detail"),
            label=raw.get("label"),
        )

    def dump(self) -> dict[str, object]:
        result: dict[str, object] = {"type": self.type}
        for key in ("text", "path", "url", "mime", "detail", "label"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    def preview(self) -> str:
        if self.type == "text":
            return self.text or ""
        if self.type == "image":
            return f"[image: {self.label or self.path or self.url or 'image'}]"
        return f"[{self.type}]"

    def api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        if self.type == "image":
            image_url: dict[str, object] = {
                "url": self.data_url(
                    max_width=image_max_width,
                    max_height=image_max_height,
                    jpeg_quality=image_jpeg_quality,
                )
            }
            if self.detail:
                image_url["detail"] = self.detail
            return {"type": "image_url", "image_url": image_url}
        raise ValueError(f"unsupported part type: {self.type}")

    def responses_api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if self.type == "text":
            return {"type": "input_text", "text": self.text or ""}
        if self.type == "image":
            result: dict[str, object] = {
                "type": "input_image",
                "image_url": self.data_url(
                    max_width=image_max_width,
                    max_height=image_max_height,
                    jpeg_quality=image_jpeg_quality,
                ),
            }
            if self.detail:
                result["detail"] = self.detail
            return result
        raise ValueError(f"unsupported part type: {self.type}")

    def anthropic_api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        if self.type == "image":
            return {
                "type": "image",
                "source": anthropic_image_source(
                    self.data_url(
                        max_width=image_max_width,
                        max_height=image_max_height,
                        jpeg_quality=image_jpeg_quality,
                    )
                ),
            }
        raise ValueError(f"unsupported part type: {self.type}")

    def data_url(
        self,
        *,
        max_width: int | None = None,
        max_height: int | None = None,
        jpeg_quality: int = 80,
    ) -> str:
        if self.url:
            return self.url
        if not self.path:
            raise ValueError("image part has no path or url")
        path = Path(self.path).resolve()
        mime = self.mime or mimetypes.guess_type(path.name)[0] or "image/png"
        return local_image_data_url(
            str(path),
            path.stat().st_mtime_ns,
            mime,
            max_width,
            max_height,
            jpeg_quality,
        )


@dataclass(frozen=True)
class Message:
    role: str
    content: str | list[Part]
    created_at: str = field(default_factory=now)

    @classmethod
    def with_parts(cls, role: str, parts: Iterable[Part]) -> "Message":
        return cls(role=role, content=list(parts))

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.preview() for part in self.content).strip()

    def serialized_content(self) -> str | list[dict[str, object]]:
        if isinstance(self.content, str):
            return self.content
        return [part.dump() for part in self.content]

    def line(self) -> str:
        return json.dumps(
            {"role": self.role, "content": self.serialized_content(), "created_at": self.created_at},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        if self.role == "assistant":
            return {"role": self.role, "content": self.text()}
        return {
            "role": self.role,
            "content": [
                part.api(
                    image_max_width=image_max_width,
                    image_max_height=image_max_height,
                    image_jpeg_quality=image_jpeg_quality,
                )
                for part in self.content
            ],
        }

    def responses_api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if self.role == "assistant":
            return {
                "role": self.role,
                "content": [{"type": "output_text", "text": self.text()}],
            }
        if isinstance(self.content, str):
            return {
                "role": self.role,
                "content": [{"type": "input_text", "text": self.content}],
            }
        return {
            "role": self.role,
            "content": [
                part.responses_api(
                    image_max_width=image_max_width,
                    image_max_height=image_max_height,
                    image_jpeg_quality=image_jpeg_quality,
                )
                for part in self.content
            ],
        }

    def anthropic_api(
        self,
        *,
        image_max_width: int | None = None,
        image_max_height: int | None = None,
        image_jpeg_quality: int = 80,
    ) -> dict[str, object]:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported Anthropic message role: {self.role}")
        if self.role == "assistant" or isinstance(self.content, str):
            return {"role": self.role, "content": self.text()}
        return {
            "role": self.role,
            "content": [
                part.anthropic_api(
                    image_max_width=image_max_width,
                    image_max_height=image_max_height,
                    image_jpeg_quality=image_jpeg_quality,
                )
                for part in self.content
            ],
        }

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Message":
        content = raw["content"]
        if isinstance(content, list):
            parsed: str | list[Part] = [Part.from_raw(item) for item in content]
        else:
            parsed = str(content)
        return cls(role=str(raw["role"]), content=parsed, created_at=raw.get("created_at") or now())


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.dir = root / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)

    def create(self, title: str | None = None) -> str:
        sid = "ses_" + uuid4().hex[:16]
        self.meta(sid).write_text(
            json.dumps({"id": sid, "title": title or "new session", "created_at": now(), "updated_at": now()}, indent=2),
            encoding="utf-8",
        )
        self.path(sid).touch()
        return sid

    def append(self, sid: str, message: Message) -> None:
        self.check(sid)
        with self.path(sid).open("a", encoding="utf-8") as f:
            f.write(message.line() + "\n")
        self.touch(sid, message.text() if message.role == "user" else None)

    def load(self, sid: str) -> list[Message]:
        self.check(sid)
        result: list[Message] = []
        path = self.path(sid)
        if not path.exists():
            raise FileNotFoundError(f"session does not exist: {sid}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(Message.from_raw(json.loads(line)))
        return result

    def path(self, sid: str) -> Path:
        self.check(sid)
        return self.dir / f"{sid}.jsonl"

    def meta(self, sid: str) -> Path:
        self.check(sid)
        return self.dir / f"{sid}.meta.json"

    def touch(self, sid: str, title: str | None = None) -> None:
        path = self.meta(sid)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"id": sid, "created_at": now()}
        data["updated_at"] = now()
        if title and data.get("title") in {None, "", "new session"}:
            data["title"] = title[:80].replace("\n", " ")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def check(self, sid: str) -> None:
        if not VALID_SESSION_ID.match(sid):
            raise ValueError(f"invalid session id: {sid}")


@lru_cache(maxsize=2048)
def local_image_data_url(
    path_value: str,
    mtime_ns: int,
    mime: str,
    max_width: int | None,
    max_height: int | None,
    jpeg_quality: int,
) -> str:
    del mtime_ns
    path = Path(path_value)
    if max_width is None and max_height is None:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    with Image.open(path) as source:
        image = source.convert("RGB")
        target = (
            max_width if max_width is not None else image.width,
            max_height if max_height is not None else image.height,
        )
        image.thumbnail(target, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def anthropic_image_source(value: str) -> dict[str, object]:
    if value.startswith(("https://", "http://")):
        return {"type": "url", "url": value}
    if not value.startswith("data:") or "," not in value:
        raise ValueError("Anthropic image must be an HTTP(S) URL or base64 data URL")
    metadata, data = value.split(",", 1)
    fields = metadata.removeprefix("data:").split(";")
    media_type = fields[0]
    if not media_type.startswith("image/") or "base64" not in fields[1:]:
        raise ValueError("Anthropic image data URL must contain a base64 image")
    return {
        "type": "base64",
        "media_type": media_type,
        "data": data,
    }
