from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from agent.client import ChatClient
from agent.session import Message

from .learned_constraints import LearnedConstraintSet, validate_constraint_update


CONSTRAINT_SYNTHESIS_SYSTEM = """You are a RoboTwin constraint-maintenance agent.

You receive one failed evaluation run diagnosis and optional human feedback.
Convert them into a conservative JSON constraint update for future evaluation rounds.

Rules:
- Do not add hidden target-coordinate shortcuts.
- Preserve existing validated constraints unless the diagnosis cites direct evidence that they are wrong or harmful.
- Automatic correction is task-scoped only. Do not append, revise, or remove general/run-scope constraints.
- If a rule seems broadly useful, still encode it as scope "task" for the current task only.
- If a non-task constraint appears wrong or harmful, report it in unresolved_conflicts instead of editing it.
- Do not overfit to a single seed with object coordinates, exact pixels, or private simulator state.
- A correction may append, revise, or remove constraints, but removals/revisions must include rationale.
- Return exactly one JSON object with keys:
  preserve: list[str]
  append: list[{"id": str, "scope": "task", "text": str, "rationale": str}]
  revise: list[{"id": str, "scope": "task", "text": str, "rationale": str}]
  remove: list[{"id": str, "rationale": str}]
  unresolved_conflicts: list[str]
"""


def diagnose_with_script(
    client: ChatClient,
    run_dir: Path,
    output_dir: Path,
    *,
    max_turns: int = 18,
    display: bool = True,
) -> Path:
    script = Path(__file__).resolve().parents[3] / "scripts" / "diagnose_run_directory_agent.py"
    if not script.exists():
        raise FileNotFoundError(script)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AGENT_BASE_URL"] = client.config.base_url
    env["AGENT_MODEL"] = client.config.model
    if client.config.api_key:
        env["AGENT_API_KEY"] = client.config.api_key
    if client.config.auth_token:
        env["AGENT_AUTH_TOKEN"] = client.config.auth_token
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output_dir),
        "--max-turns",
        str(max_turns),
        "--timeout",
        str(client.config.timeout),
    ]
    stdout = None if display else subprocess.DEVNULL
    stderr = None if display else subprocess.DEVNULL
    subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=stderr)
    return output_dir / "report.md"


def synthesize_constraint_update(
    client: ChatClient,
    *,
    task: str,
    run_dir: Path,
    diagnosis_report: Path,
    current_constraints: LearnedConstraintSet,
    output_dir: Path,
    human_feedback: str = "",
    default_scope: str = "task",
    display: bool = True,
) -> Path:
    report_text = diagnosis_report.read_text(encoding="utf-8", errors="replace") if diagnosis_report.exists() else ""
    messages = [
        Message("system", CONSTRAINT_SYNTHESIS_SYSTEM),
        Message(
            "user",
            (
                f"Task: {task}\n"
                f"Run directory: {run_dir}\n"
                f"Default scope: {default_scope}\n\n"
                "Current learned constraint package:\n"
                + json.dumps(
                    {
                        "task": current_constraints.task,
                        "constraints": current_constraints.constraints,
                        "preserved": current_constraints.preserved,
                        "unresolved_conflicts": current_constraints.unresolved_conflicts,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n\nDiagnosis report:\n"
                + report_text[:24000]
                + "\n\nHuman correction protocol or feedback:\n"
                + (human_feedback.strip() or "(none; use the diagnosis only)")
            ),
        ),
    ]
    if display:
        print("[correction] synthesizing constraint update", flush=True)
    reply = client.complete(messages)
    update = parse_json_object(reply.content)
    update = task_scoped_update(update, current_constraints=current_constraints)
    error = validate_constraint_update(update, allowed_scopes={"task"})
    if error:
        raise ValueError(error + f": {reply.content[:1000]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    update_path = output_dir / "constraint_update.json"
    update_path.write_text(json.dumps(update, indent=2, ensure_ascii=False), encoding="utf-8")
    if display:
        print(f"[correction] constraint update: {update_path}", flush=True)
    return update_path


def task_scoped_update(update: dict[str, Any], *, current_constraints: LearnedConstraintSet) -> dict[str, Any]:
    """Return a correction update that cannot write outside task-scope constraints."""
    protected_ids = {
        str(item.get("id"))
        for item in current_constraints.constraints
        if str(item.get("scope") or "task") != "task"
    }
    sanitized: dict[str, Any] = {
        "preserve": [str(item) for item in update.get("preserve", [])],
        "append": [],
        "revise": [],
        "remove": [],
        "unresolved_conflicts": [str(item) for item in update.get("unresolved_conflicts", [])],
    }
    for key in ("append", "revise"):
        for item in update.get(key, []):
            if not isinstance(item, dict):
                sanitized[key].append(item)
                continue
            cid = str(item.get("id") or item.get("constraint_id") or "").strip()
            if key == "revise" and cid in protected_ids:
                sanitized["unresolved_conflicts"].append(
                    f"Correction attempted to revise non-task constraint {cid}; left unchanged."
                )
                continue
            scoped = dict(item)
            original_scope = str(scoped.get("scope") or "").strip()
            scoped["scope"] = "task"
            if original_scope and original_scope != "task":
                rationale = str(scoped.get("rationale") or scoped.get("reason") or "").strip()
                scoped["rationale"] = (
                    rationale
                    + (" " if rationale else "")
                    + f"Original requested scope {original_scope!r} was restricted to task scope."
                )
            sanitized[key].append(scoped)
    for item in update.get("remove", []):
        if isinstance(item, dict):
            cid = str(item.get("id") or item.get("constraint_id") or "").strip()
            if cid in protected_ids:
                sanitized["unresolved_conflicts"].append(
                    f"Correction attempted to remove non-task constraint {cid}; left unchanged."
                )
                continue
        sanitized["remove"].append(item)
    return sanitized


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    value, _ = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(value, dict):
        raise ValueError("model did not return a JSON object")
    return value
