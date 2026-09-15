from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any


INFRA_MARKERS = (
    "connection",
    "context length",
    "http 4",
    "http 5",
    "provider request failed",
    "remote end",
    "request too large",
    "ssl",
    "timed out",
    "timeout",
    "unexpected_eof",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact status monitor for one RoboTwin evaluation batch."
    )
    parser.add_argument("status_file")
    parser.add_argument(
        "--merge-status",
        action="append",
        default=[],
        metavar="STATUS_FILE",
        help=(
            "Merge another batch status into the primary batch. Later statuses "
            "replace earlier entries for the same seed."
        ),
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        default=[],
        metavar="SEED=RUN_DIR",
        help="Include a run outside the batch status file.",
    )
    parser.add_argument(
        "--manual-override-file",
        help="Apply audited per-task seed outcomes from a manual overrides JSON file.",
    )
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.refresh <= 0:
        raise ValueError("--refresh must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_extra_runs(values: list[str]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for value in values:
        seed_text, separator, run_dir = value.partition("=")
        if not separator or not run_dir:
            raise ValueError(f"invalid --extra-run value: {value}")
        seed = int(seed_text)
        result[seed] = {
            "seed": seed,
            "run_dir": str(Path(run_dir).expanduser().resolve()),
            "state": "completed",
        }
    return result


def run_result(evaluation: dict[str, Any]) -> dict[str, Any]:
    run_dir = evaluation.get("run_dir")
    if not isinstance(run_dir, str):
        return {}
    return read_json(Path(run_dir) / "result.json")


def read_events(run_dir: str | None) -> list[dict[str, Any]]:
    if not isinstance(run_dir, str):
        return []
    try:
        lines = (Path(run_dir) / "events.jsonl").read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def provider_label(
    evaluation: dict[str, Any],
    result: dict[str, Any],
    *,
    fallback_configured: bool,
) -> str:
    routing = result.get("provider_routing")
    if not isinstance(routing, dict):
        routing = evaluation.get("provider_routing")
    if isinstance(routing, dict) and routing.get("fallback_used"):
        return "API"
    if any(
        event.get("type") == "provider_fallback"
        for event in read_events(evaluation.get("run_dir"))
    ):
        return "API"
    return "PLAN" if fallback_configured else "PRIMARY"


def action_count(evaluation: dict[str, Any], result: dict[str, Any]) -> int:
    steps = result.get("steps")
    if isinstance(steps, list) and steps:
        numbered = [
            item.get("step")
            for item in steps
            if isinstance(item, dict) and isinstance(item.get("step"), int)
        ]
        if numbered:
            return max(numbered)

    run_dir = evaluation.get("run_dir")
    if not isinstance(run_dir, str):
        return 0
    events_path = Path(run_dir) / "events.jsonl"
    count = 0
    try:
        lines = events_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0
    latest_env_step: int | None = None
    latest_action_step: int | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("step"), int):
            continue
        if event.get("type") == "env_step":
            latest_env_step = max(latest_env_step or 0, int(data["step"]))
        elif event.get("type") == "action":
            latest_action_step = max(latest_action_step or 0, int(data["step"]))
            count += 1
    # env_step is the authoritative environment progress. During a provider
    # request there may be no new event yet, so retain the latest action count
    # as a useful lower bound instead of hiding progress behind a stale zero.
    return max(latest_env_step or 0, latest_action_step or 0, count)


def runtime_phase(evaluation: dict[str, Any]) -> str:
    """Return a short live phase so step=0 is not ambiguous."""
    events = read_events(evaluation.get("run_dir"))
    if events:
        latest = events[-1]
        event_type = str(latest.get("type") or "").lower()
        if event_type == "env_step":
            return "ENV"
        if event_type == "action":
            return "ACTION"
        if event_type in {"model_start", "context_usage"}:
            return "MODEL"
        if event_type in {"model_finish", "model_stream_error", "eval_error"}:
            return "API"
        if event_type == "eval_start":
            return "INIT"
    run_dir = evaluation.get("run_dir")
    if isinstance(run_dir, str):
        # Renderer/initialization failures can be written to the supervisor log
        # before the first structured eval event is emitted.
        try:
            log_text = "\n".join(
                p.read_text(encoding="utf-8", errors="ignore")[-4000:]
                for p in (Path(run_dir).parent.parent / "batches").glob(
                    f"*/seed_{evaluation.get('seed')}.log"
                )
            ).lower()
        except OSError:
            log_text = ""
        if "oidn error" in log_text or "out of memory" in log_text:
            return "OOM"
    return "INIT"


def is_infra_stop(result: dict[str, Any]) -> bool:
    reason = str(result.get("stop_reason") or "").lower()
    return any(marker in reason for marker in INFRA_MARKERS)


def classify(evaluation: dict[str, Any], result: dict[str, Any]) -> str:
    state = str(evaluation.get("state") or "waiting").lower()
    if state == "running":
        return "RUNNING"
    if state in {
        "process_failed",
        "interrupted",
        "terminated",
        "error",
        "infrastructure_error",
        "infra",
    }:
        return "INFRA"
    if state != "completed":
        return "WAITING"
    if is_infra_stop(result):
        return "INFRA"
    success = result.get("success", evaluation.get("success"))
    if success is True:
        return "SUCCESS"
    if success is False:
        return "FAIL"
    return "INFRA"


def load_manual_overrides(
    path: Path | None,
    *,
    task: str,
    model: str,
) -> dict[int, bool]:
    if path is None:
        return {}
    data = read_json(path)
    if data.get("model") != model:
        return {}
    return {
        int(item["seed"]): bool(item["success"])
        for item in data.get("overrides", [])
        if isinstance(item, dict)
        and item.get("task") == task
        and isinstance(item.get("seed"), int)
        and isinstance(item.get("success"), bool)
    }


def render(
    status_paths: list[Path],
    extra_runs: dict[int, dict[str, Any]],
    override_path: Path | None = None,
) -> str:
    statuses = [read_json(path) for path in status_paths]
    status = next((value for value in statuses if value), {})
    if not status:
        return f"Evaluation monitor | waiting for {status_paths[0]}"
    manual_overrides = load_manual_overrides(
        override_path,
        task=str(status.get("task") or ""),
        model=str(status.get("model") or ""),
    )
    fallback_configured = bool(
        isinstance(status.get("quota_fallback"), dict)
        and status["quota_fallback"].get("configured")
    )

    evaluations: dict[int, dict[str, Any]] = {}
    seeds: set[int] = set()
    for merged_status in statuses:
        seeds.update(int(seed) for seed in merged_status.get("seeds") or [])
        evaluations.update(
            {
                int(seed): value
                for seed, value in (merged_status.get("evaluations") or {}).items()
                if isinstance(value, dict)
            }
        )
    evaluations.update(extra_runs)
    seeds = sorted(seeds | set(extra_runs))

    rows: list[tuple[int, str, int, str, str]] = []
    counts = {
        "SUCCESS": 0,
        "FAIL": 0,
        "RUNNING": 0,
        "WAITING": 0,
        "INFRA": 0,
    }
    for seed in seeds:
        evaluation = evaluations.get(seed, {})
        result = run_result(evaluation)
        if seed in manual_overrides:
            state = "SUCCESS" if manual_overrides[seed] else "FAIL"
        else:
            state = classify(evaluation, result)
        steps = action_count(evaluation, result)
        provider = provider_label(
            evaluation,
            result,
            fallback_configured=fallback_configured,
        )
        phase = runtime_phase(evaluation) if state == "RUNNING" else "-"
        counts[state] += 1
        rows.append((seed, state, steps, provider, phase))

    lines = [
        f"Evaluation monitor | {datetime.now():%F %T}",
        (
            f"task={status.get('task', '-')} | model={status.get('model', '-')} "
            f"| max_steps={status.get('max_steps', '-')} | total={len(seeds)}"
        ),
        (
            f"SUCCESS {counts['SUCCESS']} | FAIL {counts['FAIL']} | "
            f"RUNNING {counts['RUNNING']} | WAITING {counts['WAITING']} | "
            f"INFRA {counts['INFRA']}"
        ),
        "",
        f"{'SEED':>5}  {'STATE':<8} {'STEPS':>9} {'PHASE':<6} {'PROVIDER':>8}",
        "-" * 47,
    ]
    lines.extend(
        f"{seed:>5}  {state:<8} {steps:>4}/{status.get('max_steps', '?')!s:<4} {phase:<6} {provider:>8}"
        for seed, state, steps, provider, phase in rows
    )

    active = [
        (seed, steps, provider, phase)
        for seed, state, steps, provider, phase in rows
        if state == "RUNNING"
    ]
    lines.extend(["", "RUNNING SEEDS"])
    if active:
        lines.extend(
            f"  seed={seed}  step={steps}/{status.get('max_steps', '?')}  phase={phase}  provider={provider}"
            for seed, steps, provider, phase in active
        )
    else:
        lines.append("  -")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    status_paths = [
        Path(path).expanduser().resolve()
        for path in [args.status_file, *args.merge_status]
    ]
    extra_runs = parse_extra_runs(args.extra_run)
    override_path = (
        Path(args.manual_override_file).expanduser().resolve()
        if args.manual_override_file
        else None
    )
    while True:
        # `watch` clears the terminal itself; emitting another escape sequence
        # through its pipe renders as literal `^[2J^[H` on some terminals.
        if not args.once and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(render(status_paths, extra_runs, override_path), flush=True)
        if args.once:
            return
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
