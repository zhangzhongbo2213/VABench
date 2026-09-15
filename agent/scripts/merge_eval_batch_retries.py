from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


INFRASTRUCTURE_STOP_PATTERNS = (
    re.compile(r"provider request failed", re.IGNORECASE),
    re.compile(r"HTTP\s+[45]\d\d\s+from provider", re.IGNORECASE),
    re.compile(r"usage[_ ]limit|daily[_ ]limit|rate[_ ]limit", re.IGNORECASE),
    re.compile(r"insufficient.*balance|quota", re.IGNORECASE),
    re.compile(r"connection (?:refused|reset)|network error", re.IGNORECASE),
    re.compile(r"timed? out|timeout", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge retried RoboTwin evaluation batches by seed, excluding provider "
            "and process failures from environment success rates."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--runs-dir",
        default=".agent/runs",
    )
    parser.add_argument(
        "--safe-seed-dir",
        default=".agent/runs/data/safe_eval_seed_sets",
    )
    parser.add_argument(
        "--output-dir",
        default=".agent/runs/data/merged_eval_results",
    )
    parser.add_argument(
        "--override-file",
        help="Optional audited manual seed overrides JSON.",
    )
    parser.add_argument("--tasks", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    safe_seed_dir = Path(args.safe_seed_dir).resolve()
    output_dir = Path(args.output_dir).resolve() / sanitize_name(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_sets = discover_safe_seed_sets(safe_seed_dir)
    manual_overrides = load_manual_overrides(
        Path(args.override_file).resolve() if args.override_file else None,
        model=args.model,
    )
    tasks = args.tasks or sorted(safe_sets)
    task_reports = [
        merge_task(
            task=task,
            model=args.model,
            runs_dir=runs_dir,
            safe_seed_data=safe_sets[task],
            manual_overrides=manual_overrides.get(task, {}),
        )
        for task in tasks
        if task in safe_sets
    ]
    report = {
        "format": "robotwin_merged_eval_retries_v1",
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merge_policy": {
            "unit": "safe evaluation seed",
            "selected_attempt": "latest valid attempt for each seed",
            "invalid_attempts": (
                "provider/API/network failures, nonzero process exits, missing results, "
                "or results without a boolean success value"
            ),
            "success_rate_denominator": "seeds with a valid selected attempt",
            "unresolved_safe_seeds_excluded_from_rate": True,
            "manual_overrides_are_explicit_and_auditable": True,
        },
        "override_file": str(Path(args.override_file).resolve()) if args.override_file else None,
        "tasks": task_reports,
    }
    json_path = output_dir / "merged_results.json"
    markdown_path = output_dir / "summary.md"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


def discover_safe_seed_sets(root: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task = data.get("task")
        seeds = data.get("selected_seeds")
        if not isinstance(task, str) or not data.get("complete"):
            continue
        if not isinstance(seeds, list) or len(seeds) < 20:
            continue
        candidates.setdefault(task, []).append((path, data))

    selected: dict[str, dict[str, Any]] = {}
    for task, values in candidates.items():
        path, data = max(
            values,
            key=lambda item: (item[0].stat().st_mtime_ns, len(item[1]["selected_seeds"])),
        )
        selected[task] = {
            **data,
            "source_file": str(path),
            "selected_seeds": [int(seed) for seed in data["selected_seeds"][:20]],
        }
    return selected


def merge_task(
    *,
    task: str,
    model: str,
    runs_dir: Path,
    safe_seed_data: dict[str, Any],
    manual_overrides: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    safe_seeds = safe_seed_data["selected_seeds"]
    attempts_by_seed: dict[int, list[dict[str, Any]]] = {
        seed: [] for seed in safe_seeds
    }
    batch_files = sorted((runs_dir / task / "batches").glob("*/batch_status.json"))
    included_batches: list[str] = []
    for batch_file in batch_files:
        batch = read_json(batch_file)
        batch_model = batch.get("evaluation_model") or batch.get("model")
        if batch_model != model:
            continue
        included_batches.append(str(batch_file))
        for key, evaluation in (batch.get("evaluations") or {}).items():
            try:
                seed = int(evaluation.get("seed", key))
            except (TypeError, ValueError):
                continue
            if seed not in attempts_by_seed:
                continue
            attempts_by_seed[seed].append(
                build_attempt(batch, batch_file, evaluation, seed)
            )

    seed_results = []
    for seed in safe_seeds:
        attempts = sorted(attempts_by_seed[seed], key=attempt_sort_key)
        valid_attempts = [attempt for attempt in attempts if attempt["valid"]]
        override = manual_overrides.get(seed)
        if override is not None:
            selected_attempt = manual_override_attempt(task, seed, override)
            attempts.append(selected_attempt)
            attempts.sort(key=attempt_sort_key)
            valid_attempts.append(selected_attempt)
        else:
            selected_attempt = valid_attempts[-1] if valid_attempts else None
        seed_results.append(
            {
                "seed": seed,
                "status": (
                    "success"
                    if selected_attempt and selected_attempt["success"]
                    else "failure"
                    if selected_attempt
                    else "unresolved"
                ),
                "selected_attempt": selected_attempt,
                "attempt_count": len(attempts),
                "valid_attempt_count": len(valid_attempts),
                "invalid_attempt_count": len(attempts) - len(valid_attempts),
                "attempts": attempts,
            }
        )

    valid_results = [item for item in seed_results if item["status"] != "unresolved"]
    successes = [item for item in valid_results if item["status"] == "success"]
    failures = [item for item in valid_results if item["status"] == "failure"]
    unresolved = [item for item in seed_results if item["status"] == "unresolved"]
    invalid_attempt_count = sum(item["invalid_attempt_count"] for item in seed_results)
    manual_override_count = sum(
        bool((item.get("selected_attempt") or {}).get("manual_override"))
        for item in seed_results
    )
    return {
        "task": task,
        "safe_seed_file": safe_seed_data["source_file"],
        "safe_seed_count": len(safe_seeds),
        "included_batch_count": len(included_batches),
        "included_batches": included_batches,
        "valid_evaluated_seed_count": len(valid_results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "unresolved_count": len(unresolved),
        "invalid_infrastructure_attempt_count": invalid_attempt_count,
        "manual_override_count": manual_override_count,
        "success_rate": (
            len(successes) / len(valid_results) if valid_results else None
        ),
        "success_seeds": [item["seed"] for item in successes],
        "failure_seeds": [item["seed"] for item in failures],
        "unresolved_seeds": [item["seed"] for item in unresolved],
        "seed_results": seed_results,
    }


def build_attempt(
    batch: dict[str, Any],
    batch_file: Path,
    evaluation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    run_dir_value = evaluation.get("run_dir")
    run_dir = Path(run_dir_value) if isinstance(run_dir_value, str) else None
    result_path = run_dir / "result.json" if run_dir else None
    result = read_json(result_path) if result_path and result_path.exists() else {}
    success = result.get("success", evaluation.get("success"))
    stop_reason = result.get("stop_reason")
    state = str(evaluation.get("state", "unknown"))
    returncode = evaluation.get("returncode")
    invalid_reason = infrastructure_failure_reason(
        state=state,
        returncode=returncode,
        result_exists=bool(result),
        success=success,
        stop_reason=stop_reason,
    )
    return {
        "seed": seed,
        "batch_id": batch.get("batch_id"),
        "batch_state": batch.get("state"),
        "batch_file": str(batch_file),
        "run_dir": str(run_dir) if run_dir else None,
        "result_file": str(result_path) if result_path and result_path.exists() else None,
        "evaluation_state": state,
        "returncode": returncode,
        "success": success if isinstance(success, bool) else None,
        "stop_reason": stop_reason,
        "finished_at": evaluation.get("finished_at") or batch.get("updated_at"),
        "valid": invalid_reason is None,
        "invalid_reason": invalid_reason,
    }


def load_manual_overrides(
    path: Path | None,
    *,
    model: str,
) -> dict[str, dict[int, dict[str, Any]]]:
    if path is None:
        return {}
    data = read_json(path)
    override_model = data.get("model")
    if override_model != model:
        raise ValueError(
            f"override model {override_model!r} does not match requested model {model!r}"
        )
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for item in data.get("overrides", []):
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        seed = item.get("seed")
        success = item.get("success")
        reason = item.get("reason")
        if not isinstance(task, str) or not isinstance(seed, int):
            raise ValueError(f"invalid manual override identity: {item!r}")
        if not isinstance(success, bool) or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"manual override requires boolean success and reason: {item!r}")
        if seed in result.setdefault(task, {}):
            raise ValueError(f"duplicate manual override for {task} seed {seed}")
        result[task][seed] = item
    return result


def manual_override_attempt(
    task: str,
    seed: int,
    override: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "batch_id": "manual_override",
        "batch_state": "manual_override",
        "batch_file": None,
        "run_dir": None,
        "result_file": None,
        "evaluation_state": "manual_override",
        "returncode": None,
        "success": override["success"],
        "stop_reason": override["reason"],
        "finished_at": override.get("created_at"),
        "valid": True,
        "invalid_reason": None,
        "manual_override": True,
        "provisional": bool(override.get("provisional", False)),
        "task": task,
    }


def infrastructure_failure_reason(
    *,
    state: str,
    returncode: Any,
    result_exists: bool,
    success: Any,
    stop_reason: Any,
) -> str | None:
    if state in {"process_failed", "superseded", "terminated"}:
        return f"evaluation_state={state}"
    if returncode not in {None, 0}:
        return f"returncode={returncode}"
    if not result_exists:
        return "missing_result"
    if not isinstance(success, bool):
        return "missing_boolean_success"
    if isinstance(stop_reason, str):
        for pattern in INFRASTRUCTURE_STOP_PATTERNS:
            if pattern.search(stop_reason):
                return f"infrastructure_stop_reason: {stop_reason}"
    return None


def attempt_sort_key(attempt: dict[str, Any]) -> tuple[datetime, str]:
    timestamp = attempt.get("finished_at")
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc), str(attempt.get("batch_id", ""))
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc), str(attempt.get("batch_id", ""))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Merged evaluation results: {report['model']}",
        "",
        "Each safe seed uses its latest valid attempt. Provider, quota, network, and process failures are excluded.",
        "",
        "| Task | Valid seeds | Success | Failure | Unresolved | Infra attempts | Manual | Success rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in report["tasks"]:
        rate = task["success_rate"]
        rate_text = f"{rate * 100:.1f}%" if rate is not None else "N/A"
        lines.append(
            f"| `{task['task']}` | {task['valid_evaluated_seed_count']}/{task['safe_seed_count']} "
            f"| {task['success_count']} | {task['failure_count']} | {task['unresolved_count']} "
            f"| {task['invalid_infrastructure_attempt_count']} | {task['manual_override_count']} "
            f"| {rate_text} |"
        )
    lines.append("")
    lines.append("Success rates exclude unresolved safe seeds; inspect `merged_results.json` for each selected attempt.")
    return "\n".join(lines) + "\n"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "model"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
