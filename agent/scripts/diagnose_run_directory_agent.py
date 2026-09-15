from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.client import ChatClient
from agent.config import Config
from agent.session import Message, Part


SYSTEM_PROMPT = """You are a robotics run-diagnosis agent.

You are given a RoboTwin evaluation run directory. You must autonomously inspect
the directory, choose which logs and images matter, and diagnose why the run
failed. Do not assume the user has already selected the key frames for you.

Available tools are JSON-only:
- {"tool":"list_dir","args":{"path":".","max_depth":2}}
- {"tool":"read_text","args":{"path":"events.jsonl","start_line":1,"max_lines":120}}
- {"tool":"summarize_events","args":{"include_reasons":true}}
- {"tool":"inspect_images","args":{"paths":["model_frames/step_010_camera.view_side.png","model_debug_frames/step_010_camera.view_side.png"],"detail":"high"}}

Rules:
- Use only files inside the run directory.
- Prefer model_frames for what the evaluated model saw. In new runs model_frames are clean images with no drawn EE/GC labels, axes, or grip-center lines.
- Use model_debug_frames or record_frames only as human-debug evidence; never assume those overlays were visible to the evaluated model unless events explicitly say model_overlay was not none.
- Do not use hidden object coordinates or propose adding object-coordinate cheating.
- You may use robot eepose/finger-center data because it is part of the run observation, but treat it as robot-state data only, not target coordinates or contact proof.
- Select the key failure nodes yourself. Inspect enough images to justify the diagnosis.
- End with {"final":"..."} in Chinese. Include:
  A) main failure mode,
  B) whether the fix should be generic or task-specific,
  C) exact prompt/action-feedback wording to add,
  D) whether action space or visual feedback should change.
- Constraint updates must follow an evidence-based lifecycle. In the final report, explicitly separate:
  1) preserved constraints that were already correct,
  2) new constraints to append,
  3) existing constraints that should be revised or removed because evidence shows they are wrong, harmful, or in conflict,
  4) unresolved conflicts that require human review.
  Never silently replace or delete an earlier constraint. You may recommend revising/removing a bad constraint only when you cite direct run evidence.
Return only one JSON object per response."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Let a model autonomously diagnose a RoboTwin run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--base-url", default=os.environ.get("AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.5")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing diagnosis transcript instead of starting over.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    output_dir = Path(args.output).resolve() if args.output else run_dir / "diagnosis_agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.jsonl"
    report_path = output_dir / "report.md"

    client = ChatClient(
        Config(
            base_url=(args.base_url or "https://api.openai.com/v1").rstrip("/"),
            api_key=args.api_key,
            model=args.model,
            state_dir=output_dir,
            timeout=args.timeout,
            temperature=0,
        )
    )

    analysis_request_path = run_dir / "analysis_request.md"
    analysis_request = (
        analysis_request_path.read_text(encoding="utf-8", errors="replace")
        if analysis_request_path.is_file()
        else ""
    )
    used_tools: set[str] = set()
    if args.resume and transcript_path.is_file():
        messages = load_transcript(transcript_path)
        for message in messages:
            if message.role != "assistant":
                continue
            try:
                decision = parse_json(message.text())
            except ValueError:
                continue
            tool = decision.get("tool")
            if isinstance(tool, str) and tool:
                used_tools.add(tool)
        continuation = Message(
            "user",
            (
                "Continue from the evidence already inspected. Do not request more files or images "
                "unless strictly necessary. Return the required final Chinese diagnosis now as "
                'exactly one JSON object: {"final":"..."}.'
            ),
        )
        append_transcript(transcript_path, continuation)
        messages.append(continuation)
    else:
        messages = [
            Message("system", SYSTEM_PROMPT),
            Message(
                "user",
                (
                    "Diagnose this run directory autonomously.\n"
                    f"Run directory: {run_dir}\n"
                    "Start by exploring the directory. Then choose logs/images to inspect. "
                    "The evaluation was interrupted after repeated failures, so result.json may be absent."
                    + (
                        "\n\nAdditional analysis request supplied with this directory:\n"
                        + analysis_request
                        if analysis_request
                        else ""
                    )
                ),
            ),
        ]
        write_transcript(transcript_path, messages)

    final_text = ""
    for turn in range(args.max_turns):
        print(f"[diagnosis] model turn {turn}", flush=True)
        reply = client.complete(messages)
        print(f"[diagnosis] assistant: {reply.content[:1000]}", flush=True)
        assistant_msg = Message("assistant", reply.content)
        append_transcript(transcript_path, assistant_msg)
        messages.append(assistant_msg)

        decision = parse_json(reply.content)
        if "final" in decision:
            required = {"summarize_events", "inspect_images"}
            missing = sorted(required - used_tools)
            if missing:
                result = Message("user", f"Final rejected. You must call these tools before final diagnosis: {missing}")
                print(f"[diagnosis] tool_result for final_gate: {result.text()}", flush=True)
                append_transcript(transcript_path, result)
                messages.append(result)
                continue
            final_text = str(decision["final"])
            report_path.write_text(final_text + "\n", encoding="utf-8")
            print(f"[diagnosis] final report: {report_path}", flush=True)
            break

        tool = str(decision.get("tool", ""))
        tool_args = decision.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}
        try:
            result = run_tool(run_dir, tool, tool_args)
            used_tools.add(tool)
        except Exception as exc:  # Keep the model in the loop.
            result = Message("user", f"Tool error for {tool}: {type(exc).__name__}: {exc}")
        print(f"[diagnosis] tool_result for {tool}: {result.text()[:1000]}", flush=True)
        append_transcript(transcript_path, result)
        messages.append(result)

    if not final_text:
        fallback = "诊断 agent 未在最大轮数内给出 final。请查看 transcript.jsonl。"
        report_path.write_text(fallback + "\n", encoding="utf-8")
        print(f"[diagnosis] incomplete report: {report_path}", flush=True)


def parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant did not return JSON: {text[:500]}") from exc
    if not isinstance(value, dict):
        raise ValueError("assistant JSON is not an object")
    return value


def run_tool(run_dir: Path, tool: str, args: dict[str, Any]) -> Message:
    if tool == "list_dir":
        return Message("user", list_dir(run_dir, args))
    if tool == "read_text":
        return Message("user", read_text(run_dir, args))
    if tool == "summarize_events":
        return Message("user", summarize_events(run_dir, args))
    if tool == "inspect_images":
        return inspect_images(run_dir, args)
    raise ValueError(f"unknown tool: {tool}")


def safe_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes run dir: {rel}")
    return path


def list_dir(run_dir: Path, args: dict[str, Any]) -> str:
    base = safe_path(run_dir, str(args.get("path") or "."))
    max_depth = int(args.get("max_depth") or 2)
    if max_depth < 0 or max_depth > 5:
        max_depth = 2
    lines = [f"Directory listing for {base.relative_to(run_dir) if base != run_dir else '.'}:"]
    count = 0
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if len(rel.parts) > max_depth:
            continue
        count += 1
        if count > 300:
            lines.append("... listing truncated at 300 entries")
            break
        kind = "dir" if path.is_dir() else "file"
        size = path.stat().st_size if path.is_file() else 0
        lines.append(f"{kind:4s} {str(path.relative_to(run_dir))} {size} bytes")
    return "\n".join(lines)


def read_text(run_dir: Path, args: dict[str, Any]) -> str:
    path = safe_path(run_dir, str(args.get("path") or ""))
    start_line = max(1, int(args.get("start_line") or 1))
    max_lines = max(1, min(300, int(args.get("max_lines") or 120)))
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = text[start_line - 1 : start_line - 1 + max_lines]
    body = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start_line))
    if len(body) > 24000:
        body = body[:24000] + "\n... text truncated"
    return f"Text file {path.relative_to(run_dir)} lines {start_line}-{start_line + len(selected) - 1}:\n{body}"


def summarize_events(run_dir: Path, args: dict[str, Any]) -> str:
    include_reasons = bool(args.get("include_reasons", True))
    path = run_dir / "events.jsonl"
    if not path.exists():
        return "events.jsonl not found"
    lines = ["Event summary from events.jsonl:"]
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        typ = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if typ == "tool_result":
            lines.append(f"tool_result tool={data.get('tool')} mode={data.get('mode')}")
        elif typ == "action":
            reason = f" reason={data.get('reason')}" if include_reasons else ""
            lines.append(f"action step={data.get('step')} action={data.get('action')} payload={data.get('action_payload')}{reason}")
        elif typ == "env_step":
            geom = data.get("geometry") if isinstance(data.get("geometry"), dict) else {}
            lines.append(
                "env_step step={step} success={success} done={done} planner={planner} valid={valid} "
                "model_image={image} model_overlay={overlay} model_debug_image={debug_image} "
                "fc={fc} ee={ee} rpy={rpy}".format(
                    step=data.get("step"),
                    success=data.get("success"),
                    done=data.get("done"),
                    planner=data.get("planner_success"),
                    valid=data.get("action_valid"),
                    image=data.get("model_image"),
                    overlay=data.get("model_overlay"),
                    debug_image=data.get("model_debug_image"),
                    fc=geom.get("finger_center_xyz"),
                    ee=geom.get("control_eepose_xyz"),
                    rpy=geom.get("control_eepose_rpy_deg"),
                )
            )
        elif typ in {"eval_start", "eval_end", "model_error", "tool_error", "driver_error"}:
            lines.append(f"{typ} {json.dumps(data, ensure_ascii=False)}")
    body = "\n".join(lines)
    if len(body) > 30000:
        body = body[:30000] + "\n... event summary truncated"
    return body


def inspect_images(run_dir: Path, args: dict[str, Any]) -> Message:
    raw_paths = args.get("paths")
    if isinstance(raw_paths, str):
        paths = [raw_paths]
    elif isinstance(raw_paths, list):
        paths = [str(item) for item in raw_paths]
    else:
        raise ValueError("inspect_images requires args.paths")
    if not paths:
        raise ValueError("no image paths")
    if len(paths) > 6:
        paths = paths[:6]
    detail = str(args.get("detail") or "high")
    parts: list[Part] = [Part.text_part("Inspect these selected images from the run directory. Analyze visual evidence carefully.")]
    for rel in paths:
        path = safe_path(run_dir, rel)
        if not path.exists():
            raise FileNotFoundError(rel)
        parts.append(Part.text_part(f"Image path: {rel}"))
        parts.append(Part.image_part(path, label=rel, detail=detail))
    return Message.with_parts("user", parts)


def write_transcript(path: Path, messages: list[Message]) -> None:
    path.write_text("".join(message.line() + "\n" for message in messages), encoding="utf-8")


def append_transcript(path: Path, message: Message) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(message.line() + "\n")


def load_transcript(path: Path) -> list[Message]:
    messages: list[Message] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        messages.append(Message.from_raw(json.loads(line)))
    return messages


if __name__ == "__main__":
    main()
