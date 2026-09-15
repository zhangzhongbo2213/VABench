from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


ACTIVE_STATUS = "active"


@dataclass
class LearnedConstraintSet:
    path: Path
    task: str
    constraints: list[dict[str, Any]]
    preserved: list[str]
    revisions: list[dict[str, Any]]
    unresolved_conflicts: list[str]

    @classmethod
    def load_or_empty(cls, path: Path, *, task: str) -> "LearnedConstraintSet":
        if not path.exists():
            return cls(path=path, task=task, constraints=[], preserved=[], revisions=[], unresolved_conflicts=[])
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("learned constraint file must contain a JSON object")
        constraints = data.get("constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("learned constraint file constraints must be a list")
        return cls(
            path=path,
            task=str(data.get("task") or task),
            constraints=[item for item in constraints if isinstance(item, dict)],
            preserved=[str(item) for item in data.get("preserved", []) if str(item).strip()],
            revisions=[item for item in data.get("revisions", []) if isinstance(item, dict)],
            unresolved_conflicts=[str(item) for item in data.get("unresolved_conflicts", []) if str(item).strip()],
        )

    @property
    def active_constraints(self) -> list[dict[str, Any]]:
        return [item for item in self.constraints if item.get("status", ACTIVE_STATUS) == ACTIVE_STATUS and str(item.get("text", "")).strip()]

    def prompt_text(self) -> str:
        active = self.active_constraints
        if not active and not self.preserved:
            return ""
        lines = [
            "\nLearned correction constraints from previous evaluation rounds. Treat them as additive task experience.",
            "Preserve active learned constraints unless direct visual evidence shows a specific conflict; if a conflict exists, name it before revising.",
        ]
        if self.preserved:
            lines.append("Previously preserved validated constraints:")
            lines.extend(f"- {item}" for item in self.preserved[-12:])
        if active:
            lines.append("Active learned constraints:")
            for item in active:
                cid = str(item.get("id") or "learned_constraint")
                scope = str(item.get("scope") or "task")
                text = str(item.get("text") or "").strip()
                lines.append(f"- [{cid} scope={scope}] {text}")
        if self.unresolved_conflicts:
            lines.append("Unresolved correction conflicts requiring caution:")
            lines.extend(f"- {item}" for item in self.unresolved_conflicts[-8:])
        return "\n".join(lines) + "\n"

    def apply_update(self, update: dict[str, Any], *, source_run: str, diagnosis_report: Path | None = None) -> None:
        if not isinstance(update, dict):
            raise ValueError("constraint update must be a JSON object")
        for item in update.get("preserve", []):
            text = str(item).strip()
            if text and text not in self.preserved:
                self.preserved.append(text)
        for item in update.get("remove", []):
            cid = _item_id(item)
            if cid:
                self._mark_removed(cid, item, source_run=source_run, diagnosis_report=diagnosis_report)
        for item in update.get("revise", []):
            cid = _item_id(item)
            if cid:
                self._mark_removed(cid, item, source_run=source_run, diagnosis_report=diagnosis_report)
            text = _item_text(item)
            if text:
                self._append_constraint(item, text, source_run=source_run, diagnosis_report=diagnosis_report, revision_of=cid)
        for item in update.get("append", []):
            text = _item_text(item)
            if text:
                self._append_constraint(item, text, source_run=source_run, diagnosis_report=diagnosis_report)
        for item in update.get("unresolved_conflicts", []):
            text = str(item).strip()
            if text:
                self.unresolved_conflicts.append(text)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task,
            "constraints": self.constraints,
            "preserved": self.preserved,
            "revisions": self.revisions,
            "unresolved_conflicts": self.unresolved_conflicts,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_constraint(
        self,
        raw: Any,
        text: str,
        *,
        source_run: str,
        diagnosis_report: Path | None,
        revision_of: str | None = None,
    ) -> None:
        cid = _item_id(raw) or self._next_id()
        entry = {
            "id": _unique_id(cid, self.constraints),
            "scope": _item_scope(raw) or "task",
            "text": text,
            "rationale": _item_rationale(raw),
            "source_run": source_run,
            "diagnosis_report": str(diagnosis_report) if diagnosis_report else None,
            "revision_of": revision_of,
            "status": ACTIVE_STATUS,
        }
        self.constraints.append(entry)

    def _mark_removed(self, constraint_id: str, raw: Any, *, source_run: str, diagnosis_report: Path | None) -> None:
        for item in self.constraints:
            if item.get("id") == constraint_id and item.get("status", ACTIVE_STATUS) == ACTIVE_STATUS:
                item["status"] = "removed"
                item["removed_by_run"] = source_run
                item["removed_reason"] = _item_rationale(raw) or _item_text(raw)
                item["removed_diagnosis_report"] = str(diagnosis_report) if diagnosis_report else None
        self.revisions.append(
            {
                "id": constraint_id,
                "action": "remove_or_revise",
                "source_run": source_run,
                "reason": _item_rationale(raw) or _item_text(raw),
                "diagnosis_report": str(diagnosis_report) if diagnosis_report else None,
            }
        )

    def _next_id(self) -> str:
        return f"learned_{len(self.constraints) + 1:03d}"


def default_learned_constraint_file(state_dir: Path, *, task: str) -> Path:
    return state_dir / "runs" / "data" / "learned_constraints" / f"{sanitize_run_component(task)}.json"


def load_prompt_text(path: Path | None, *, task: str) -> str:
    if path is None or not path.exists():
        return ""
    return LearnedConstraintSet.load_or_empty(path, task=task).prompt_text()


def validate_constraint_update(update: dict[str, Any], *, allowed_scopes: set[str] | None = None) -> str | None:
    if not isinstance(update, dict):
        return "constraint update must be an object"
    for key in ("preserve", "append", "revise", "remove", "unresolved_conflicts"):
        if key in update and not isinstance(update[key], list):
            return f"constraint update field {key!r} must be a list"
    for key in ("append", "revise"):
        for item in update.get(key, []):
            if _item_text(item) == "":
                return f"constraint update field {key!r} contains an item without text"
            if allowed_scopes is not None:
                scope = _item_scope(item) or "task"
                if scope not in allowed_scopes:
                    return f"constraint update field {key!r} contains disallowed scope {scope!r}"
    return None


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text") or item.get("new_text") or "").strip()
    return ""


def _item_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or item.get("constraint_id") or "").strip()
    return ""


def _item_scope(item: Any) -> str:
    if isinstance(item, dict):
        scope = str(item.get("scope") or "").strip()
        if scope in {"general", "task", "run"}:
            return scope
    return ""


def _item_rationale(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("rationale") or item.get("reason") or "").strip()
    return ""


def _unique_id(base: str, constraints: list[dict[str, Any]]) -> str:
    existing = {str(item.get("id")) for item in constraints}
    if base not in existing:
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def sanitize_run_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "task"
