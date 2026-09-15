from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from uuid import uuid4

from .timeutil import now


@dataclass(frozen=True)
class Event:
    type: str
    data: dict[str, object]
    created_at: str


class RunLogger:
    def __init__(
        self,
        state_dir: Path,
        run_id: str | None = None,
        *,
        relative_parent: Path | str | None = None,
        display: bool = True,
    ):
        self.run_id = run_id or "run_" + uuid4().hex[:12]
        parent = Path(relative_parent) if relative_parent is not None else Path()
        if parent.is_absolute() or ".." in parent.parts:
            raise ValueError("run relative_parent must stay inside the runs directory")
        self.dir = state_dir / "runs" / parent / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.display = display
        self.event_path = self.dir / "events.jsonl"
        self.transcript_path = self.dir / "transcript.jsonl"

    def event(self, event_type: str, data: dict[str, object] | None = None, *, display: bool | None = None) -> None:
        event = Event(type=event_type, data=data or {}, created_at=now())
        with self.event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n")
        if self.display if display is None else display:
            suffix = f" {json.dumps(event.data, ensure_ascii=False)}" if event.data else ""
            print(f"[{event.type}]{suffix}", file=sys.stderr)

    def transcript(self, role: str, text: str) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": text, "created_at": now()}, ensure_ascii=False) + "\n")

    def artifact(self, label: str, path: Path) -> None:
        try:
            value = str(path.relative_to(self.dir))
        except ValueError:
            value = str(path)
        self.event("artifact", {"label": label, "path": value})
