from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_parallel_video_eval_batch.py")
INFRASTRUCTURE_ERROR = re.compile(
    r"provider request failed|provider response missing text content|"
    r"HTTP\s+[45]\d\d|usage[_ ]limit|daily[_ ]limit|"
    r"rate[_ ]limit|insufficient.*balance|quota|connection (?:refused|reset)|"
    r"network error|timed?\s*out|timeout|CUDA out of memory|Traceback",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one task at a time across several GPUs. Each GPU runs its "
            "assigned seeds serially; infrastructure-invalid seeds are retried "
            "before the pipeline advances."
        )
    )
    parser.add_argument(
        "--task-spec",
        action="append",
        required=True,
        metavar="TASK:MAX_STEPS",
        help="Ordered task and step cap. Repeat for each task.",
    )
    parser.add_argument(
        "--seed-override",
        action="append",
        default=[],
        metavar="TASK:SEED,SEED,...",
        help=(
            "Evaluate only the listed safe seeds for a task. Repeat for multiple "
            "tasks; tasks without an override use their full 20-seed safe set."
        ),
    )
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--state-dir", default=".agent")
    parser.add_argument("--safe-seed-dir")
    parser.add_argument("--summary-root")
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument(
        "--cross-task",
        action="store_true",
        help=(
            "Schedule tasks from one global queue. Each GPU owns one batch worker "
            "and immediately takes the next task when its current task finishes."
        ),
    )
    parser.add_argument(
        "--fixed-views",
        nargs="+",
        choices=["topdown", "side", "front_side_45", "side_top_45", "oblique_45"],
        default=[],
    )
    parser.add_argument(
        "--camera-policy",
        choices=["fixed", "fine", "full"],
        default=None,
        help="Camera control mode; fine enables active model-selected view exploration.",
    )
    parser.add_argument(
        "--initial-camera-view",
        choices=[
            "default",
            "center_high",
            "gripper_follow",
            "topdown",
            "side",
            "front_side_45",
            "side_top_45",
            "oblique_45",
            "workspace",
            "gripper",
        ],
        default="center_high",
    )
    parser.add_argument("--fixed-view-width", type=int, default=384)
    parser.add_argument("--fixed-view-height", type=int, default=288)
    parser.add_argument("--fixed-view-jpeg-quality", type=int, default=60)
    parser.add_argument("--visual-width", type=int, default=1280)
    parser.add_argument("--visual-height", type=int, default=960)
    parser.add_argument("--model-debug-overlay", choices=["none", "eepose"], default="eepose")
    parser.add_argument("--record-overlay", choices=["none", "eepose"], default="eepose")
    parser.add_argument(
        "--experiment-mode",
        choices=["shared-summary", "model-specific-summary"],
        default="shared-summary",
    )
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stagger-seconds", type=float, default=10.0)
    parser.add_argument("--retry-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-infra-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must be unique")
    if not args.gpus:
        raise ValueError("at least one GPU is required")
    if args.max_parallel_per_gpu < 1:
        raise ValueError("--max-parallel-per-gpu must be positive")
    if args.camera_policy is None:
        args.camera_policy = "fixed" if args.fixed_views else "fine"
    if args.camera_policy == "fixed" and not args.fixed_views:
        raise ValueError("fixed camera policy requires at least one fixed view")
    if args.camera_policy != "fixed" and args.fixed_views:
        raise ValueError("--fixed-views can only be used with --camera-policy fixed")
    if args.request_timeout <= 0 or args.poll_seconds <= 0:
        raise ValueError("timeouts and polling intervals must be positive")
    if args.stagger_seconds < 0 or args.retry_cooldown_seconds < 0:
        raise ValueError("stagger and retry cooldown cannot be negative")
    if args.max_infra_retries < 0:
        raise ValueError("--max-infra-retries cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    pipeline_id = validate_leaf(args.pipeline_id)
    state_dir = Path(args.state_dir).resolve()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    safe_seed_dir = (
        Path(args.safe_seed_dir).resolve()
        if args.safe_seed_dir
        else state_dir / "runs" / "data" / "safe_eval_seed_sets"
    )
    summary_root = (
        Path(args.summary_root).resolve()
        if args.summary_root
        else state_dir / "runs" / "data" / "shared_summaries" / "gpt-5.6-sol"
    )
    task_specs = parse_task_specs(args.task_spec)
    seed_overrides = parse_seed_overrides(args.seed_override)
    safe_sets = discover_safe_seed_sets(safe_seed_dir)
    tasks = build_task_plan(
        task_specs,
        safe_sets,
        summary_root,
        args.gpus,
        seed_overrides=seed_overrides,
    )

    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"benchmark directory not found: {benchmark_dir}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"batch runner not found: {RUNNER}")

    pipeline_dir = state_dir / "runs" / "data" / "eval_pipelines" / pipeline_id
    if pipeline_dir.exists():
        raise FileExistsError(f"pipeline already exists: {pipeline_dir}")
    pipeline_dir.mkdir(parents=True)
    status_path = pipeline_dir / "pipeline_status.json"
    status: dict[str, Any] = {
        "format": "robotwin_sequential_gpu_eval_pipeline_v1",
        "pipeline_id": pipeline_id,
        "pipeline_dir": str(pipeline_dir),
        "supervisor_pid": os.getpid(),
        "created_at": now(),
        "updated_at": now(),
        "state": "planned" if args.dry_run else "initializing",
        "config": args.config,
        "benchmark_dir": str(benchmark_dir),
        "state_dir": str(state_dir),
        "safe_seed_dir": str(safe_seed_dir),
        "summary_root": str(summary_root),
        "seed_overrides": seed_overrides,
        "gpus": args.gpus,
        "max_parallel_per_gpu": args.max_parallel_per_gpu,
        "scheduling_mode": "cross-task" if args.cross_task else "task-barrier",
        "fixed_views": args.fixed_views,
        "camera_policy": args.camera_policy,
        "initial_camera_view": args.initial_camera_view,
        "fixed_view_transport": {
            "width": args.fixed_view_width,
            "height": args.fixed_view_height,
            "jpeg_quality": args.fixed_view_jpeg_quality,
        },
        "render_transport": {
            "width": args.visual_width,
            "height": args.visual_height,
            "model_debug_overlay": args.model_debug_overlay,
            "record_overlay": args.record_overlay,
        },
        "experiment_mode": args.experiment_mode,
        "request_timeout_seconds": args.request_timeout,
        "poll_seconds": args.poll_seconds,
        "stagger_seconds": args.stagger_seconds,
        "retry_cooldown_seconds": args.retry_cooldown_seconds,
        "max_infra_retries": args.max_infra_retries,
        "model": (
            os.environ.get("AGENT_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("OPENAI_MODEL")
        ),
        "wire_api": os.environ.get("AGENT_WIRE_API") or "chat_completions",
        "anthropic_version": os.environ.get("AGENT_ANTHROPIC_VERSION"),
        "request_image_transport": {
            "max_width": os.environ.get("AGENT_REQUEST_IMAGE_MAX_WIDTH"),
            "max_height": os.environ.get("AGENT_REQUEST_IMAGE_MAX_HEIGHT"),
            "jpeg_quality": os.environ.get("AGENT_REQUEST_IMAGE_JPEG_QUALITY"),
            "max_images": os.environ.get("AGENT_REQUEST_MAX_IMAGES"),
        },
        "current_task_index": None,
        "active_batches": [],
        "tasks": tasks,
    }
    refresh_status(status)
    write_status(status_path, status)
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    if not any(
        os.environ.get(name)
        for name in (
            "AGENT_API_KEY",
            "AGENT_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "OPENAI_API_KEY",
        )
    ):
        raise ValueError("a supported API key or Anthropic auth token is required")

    processes: list[subprocess.Popen[str]] = []
    stopping = False

    def stop_pipeline(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        status["state"] = "stopping"
        status["stop_signal"] = signum
        write_status(status_path, status)
        terminate_processes(processes, status)

    signal.signal(signal.SIGTERM, stop_pipeline)
    signal.signal(signal.SIGINT, stop_pipeline)

    try:
        status["state"] = "running"
        write_status(status_path, status)
        if args.cross_task:
            run_cross_task_scheduler(
                tasks=tasks,
                status=status,
                status_path=status_path,
                processes=processes,
                args=args,
                pipeline_id=pipeline_id,
                state_dir=state_dir,
                benchmark_dir=benchmark_dir,
                pipeline_dir=pipeline_dir,
                stopping=lambda: stopping,
            )
        else:
            for task_index, task in enumerate(tasks):
                if stopping:
                    break
                status["current_task_index"] = task_index
                task["state"] = "evaluating"
                task["started_at"] = now()
                write_status(status_path, status)

                initial_groups = split_round_robin(task["seeds"], args.gpus)
                initial_batches = [
                    batch_spec(
                        pipeline_id=pipeline_id,
                        task_index=task_index,
                        attempt=0,
                        gpu=gpu,
                        seeds=seeds,
                        task=task["task"],
                        state_dir=state_dir,
                    )
                    for gpu, seeds in initial_groups
                    if seeds
                ]
                launch_wave(
                    batches=initial_batches,
                    task=task,
                    status=status,
                    status_path=status_path,
                    processes=processes,
                    args=args,
                    state_dir=state_dir,
                    benchmark_dir=benchmark_dir,
                    pipeline_dir=pipeline_dir,
                    stopping=lambda: stopping,
                )

                retry_round = 0
                invalid = unresolved_infra_seeds(task)
                while invalid and retry_round < args.max_infra_retries and not stopping:
                    retry_round += 1
                    task["state"] = "infra_cooldown"
                    task["infra_retry_round"] = retry_round
                    task["infra_retry_seeds"] = invalid
                    write_status(status_path, status)
                    sleep_with_status(
                        args.retry_cooldown_seconds,
                        status_path,
                        status,
                        stopping=lambda: stopping,
                    )
                    if stopping:
                        break
                    retry_batch = batch_spec(
                        pipeline_id=pipeline_id,
                        task_index=task_index,
                        attempt=retry_round,
                        gpu=args.gpus[0],
                        seeds=invalid,
                        task=task["task"],
                        state_dir=state_dir,
                    )
                    task["state"] = "retrying_infra"
                    launch_wave(
                        batches=[retry_batch],
                        task=task,
                        status=status,
                        status_path=status_path,
                        processes=processes,
                        args=args,
                        state_dir=state_dir,
                        benchmark_dir=benchmark_dir,
                        pipeline_dir=pipeline_dir,
                        stopping=lambda: stopping,
                    )
                    invalid = unresolved_infra_seeds(task)

                if stopping:
                    break
                task["state"] = "completed" if not invalid else "completed_with_infra"
                task["finished_at"] = now()
                task["infra_retry_seeds"] = invalid
                refresh_status(status)
                write_status(status_path, status)

        status["state"] = "stopped" if stopping else "completed"
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        terminate_processes(processes, status)
        status["supervisor_pid"] = None
        status["current_task_index"] = None
        status["active_batches"] = []
        status["finished_at"] = now()
        refresh_status(status)
        write_status(status_path, status)


def run_cross_task_scheduler(
    *,
    tasks: list[dict[str, Any]],
    status: dict[str, Any],
    status_path: Path,
    processes: list[subprocess.Popen[str]],
    args: argparse.Namespace,
    pipeline_id: str,
    state_dir: Path,
    benchmark_dir: Path,
    pipeline_dir: Path,
    stopping: Any,
) -> None:
    """Run one batch worker per GPU and draw tasks from a global queue.

    A batch contains all seeds for one task and the child runner limits the
    number of simultaneous environments with ``--max-parallel``.  This keeps
    the existing per-GPU memory guard while removing the old task-wide barrier.
    """
    pending: list[dict[str, Any]] = [
        {
            "task_index": index,
            "seeds": list(task["seeds"]),
            "attempt": 0,
            "not_before": 0.0,
        }
        for index, task in enumerate(tasks)
    ]
    active: dict[int, dict[str, Any]] = {}
    retry_rounds: dict[int, int] = {index: 0 for index in range(len(tasks))}
    next_launch_at = 0.0
    status["current_task_index"] = None
    status["queue"] = {
        "pending_tasks": [task["task"] for task in tasks],
        "active_workers": {},
    }
    write_status(status_path, status)

    while pending or active:
        if stopping():
            break

        # Retire finished workers. Their batch status is authoritative for
        # per-seed results, including infrastructure-invalid classifications.
        for gpu, item in list(active.items()):
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            batch = item["batch"]
            batch["returncode"] = returncode
            batch["state"] = "completed" if returncode == 0 else "process_failed"
            batch["finished_at"] = now()
            log_file = getattr(process, "_pipeline_log_file", None)
            if log_file is not None and not log_file.closed:
                log_file.close()
            if process in processes:
                processes.remove(process)
            del active[gpu]

            task_index = int(item["task_index"])
            task = tasks[task_index]
            refresh_status(status)
            invalid = unresolved_infra_seeds(task)
            retry_round = retry_rounds[task_index]
            if invalid and retry_round < args.max_infra_retries:
                retry_round += 1
                retry_rounds[task_index] = retry_round
                task["state"] = "infra_cooldown"
                task["infra_retry_round"] = retry_round
                task["infra_retry_seeds"] = invalid
                pending.append(
                    {
                        "task_index": task_index,
                        "seeds": invalid,
                        "attempt": retry_round,
                        "not_before": time.monotonic() + args.retry_cooldown_seconds,
                    }
                )
            else:
                task["state"] = "completed" if not invalid else "completed_with_infra"
                task["finished_at"] = now()
                task["infra_retry_seeds"] = invalid

        # Fill every currently idle GPU from the global queue. A retry waiting
        # for cooldown does not block unrelated tasks behind it.
        current_time = time.monotonic()
        if current_time >= next_launch_at:
            for gpu in args.gpus:
                if gpu in active:
                    continue
                ready_index = next(
                    (
                        index
                        for index, job in enumerate(pending)
                        if float(job["not_before"]) <= current_time
                    ),
                    None,
                )
                if ready_index is None:
                    break
                job = pending.pop(ready_index)
                task_index = int(job["task_index"])
                task = tasks[task_index]
                if task.get("started_at") is None:
                    task["started_at"] = now()
                task["state"] = "evaluating"
                task["worker_gpu"] = gpu
                batch = batch_spec(
                    pipeline_id=pipeline_id,
                    task_index=task_index,
                    attempt=int(job["attempt"]),
                    gpu=gpu,
                    seeds=list(job["seeds"]),
                    task=task["task"],
                    state_dir=state_dir,
                    prefix="cross",
                )
                task["attempts"].append(batch)
                process = start_batch_process(
                    batch=batch,
                    task=task,
                    args=args,
                    state_dir=state_dir,
                    benchmark_dir=benchmark_dir,
                    pipeline_dir=pipeline_dir,
                )
                processes.append(process)
                active[gpu] = {
                    "task_index": task_index,
                    "batch": batch,
                    "process": process,
                }
                next_launch_at = time.monotonic() + args.stagger_seconds
                if args.stagger_seconds > 0:
                    break

        status["active_batches"] = [item["batch"] for item in active.values()]
        pending_task_names = [
            task_label(tasks, int(job["task_index"])) for job in pending
        ]
        active_worker_names = {
            str(gpu): task_label(tasks, int(item["task_index"]))
            for gpu, item in active.items()
        }
        status["queue"] = {
            "pending_tasks": pending_task_names,
            "active_workers": active_worker_names,
        }
        refresh_status(status)
        write_status(status_path, status)
        if pending or active:
            wait_seconds = args.poll_seconds
            if pending and not active:
                ready_at = min(float(job["not_before"]) for job in pending)
                wait_seconds = min(wait_seconds, max(0.05, ready_at - time.monotonic()))
            elif pending and next_launch_at > time.monotonic():
                wait_seconds = min(wait_seconds, max(0.05, next_launch_at - time.monotonic()))
            time.sleep(max(0.05, wait_seconds))

    status["active_batches"] = []
    status["queue"] = {
        "pending_tasks": [task_label(tasks, int(job["task_index"])) for job in pending],
        "active_workers": {},
    }
    refresh_status(status)
    write_status(status_path, status)


def task_label(tasks: list[dict[str, Any]], task_index: int) -> str:
    return str(tasks[task_index]["task"])


def parse_task_specs(values: list[str]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    seen: set[str] = set()
    for value in values:
        task, separator, max_steps_text = value.rpartition(":")
        if not separator or not task:
            raise ValueError(f"invalid task spec {value!r}; expected TASK:MAX_STEPS")
        try:
            max_steps = int(max_steps_text)
        except ValueError as exc:
            raise ValueError(f"invalid max steps in task spec {value!r}") from exc
        if max_steps < 1:
            raise ValueError(f"max steps must be positive in {value!r}")
        if task in seen:
            raise ValueError(f"duplicate task: {task}")
        seen.add(task)
        parsed.append((task, max_steps))
    return parsed


def parse_seed_overrides(values: list[str]) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for value in values:
        task, separator, seed_text = value.partition(":")
        if not separator or not task or not seed_text:
            raise ValueError(
                f"invalid seed override {value!r}; expected TASK:SEED,SEED,..."
            )
        if task in overrides:
            raise ValueError(f"duplicate seed override for task: {task}")
        try:
            seeds = [int(seed) for seed in seed_text.split(",")]
        except ValueError as exc:
            raise ValueError(f"invalid seed override {value!r}") from exc
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"seed override must contain unique seeds: {value!r}")
        overrides[task] = seeds
    return overrides


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
            key=lambda item: (
                item[0].stat().st_mtime_ns,
                len(item[1]["selected_seeds"]),
            ),
        )
        selected[task] = {
            "source_file": str(path),
            "selected_seeds": [int(seed) for seed in data["selected_seeds"][:20]],
        }
    return selected


def build_task_plan(
    task_specs: list[tuple[str, int]],
    safe_sets: dict[str, dict[str, Any]],
    summary_root: Path,
    gpus: list[int],
    *,
    seed_overrides: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    seed_overrides = seed_overrides or {}
    unknown_tasks = set(seed_overrides) - {task for task, _ in task_specs}
    if unknown_tasks:
        raise ValueError(
            "seed override provided for unknown task(s): "
            + ", ".join(sorted(unknown_tasks))
        )
    tasks = []
    for task, max_steps in task_specs:
        safe_set = safe_sets.get(task)
        if safe_set is None:
            raise FileNotFoundError(f"no complete 20-seed set found for {task}")
        summary_file = summary_root / task / "expert_learning.json"
        if not summary_file.is_file():
            raise FileNotFoundError(f"expert summary not found: {summary_file}")
        safe_seeds = safe_set["selected_seeds"]
        seeds = seed_overrides.get(task, safe_seeds)
        unsafe_seeds = sorted(set(seeds) - set(safe_seeds))
        if unsafe_seeds:
            raise ValueError(
                f"seed override for {task} contains seeds outside its safe set: "
                + ", ".join(str(seed) for seed in unsafe_seeds)
            )
        tasks.append(
            {
                "task": task,
                "max_steps": max_steps,
                "seeds": seeds,
                "safe_seed_file": safe_set["source_file"],
                "expert_learning_file": str(summary_file),
                "gpu_groups": [
                    {"gpu": gpu, "seeds": group}
                    for gpu, group in split_round_robin(seeds, gpus)
                ],
                "state": "queued",
                "attempts": [],
                "latest_results": {},
                "counts": empty_counts(len(seeds)),
            }
        )
    return tasks


def split_round_robin(seeds: list[int], gpus: list[int]) -> list[tuple[int, list[int]]]:
    groups = [(gpu, []) for gpu in gpus]
    for index, seed in enumerate(seeds):
        groups[index % len(groups)][1].append(seed)
    return groups


def batch_spec(
    *,
    pipeline_id: str,
    task_index: int,
    attempt: int,
    gpu: int,
    seeds: list[int],
    task: str,
    state_dir: Path,
    prefix: str = "seq4",
) -> dict[str, Any]:
    batch_id = f"{prefix}_{pipeline_id}_t{task_index + 1:02d}_a{attempt}_gpu{gpu}"
    status_path = (
        state_dir / "runs" / task / "batches" / batch_id / "batch_status.json"
    )
    return {
        "batch_id": batch_id,
        "attempt": attempt,
        "gpu": gpu,
        "seeds": seeds,
        "status_path": str(status_path),
        "state": "waiting",
        "pid": None,
    }


def start_batch_process(
    *,
    batch: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    state_dir: Path,
    benchmark_dir: Path,
    pipeline_dir: Path,
) -> subprocess.Popen[str]:
    command = build_batch_command(
        batch=batch,
        task=task,
        args=args,
        state_dir=state_dir,
        benchmark_dir=benchmark_dir,
    )
    log_path = pipeline_dir / f"{batch['batch_id']}.log"
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(batch["gpu"])
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    process._pipeline_log_file = log_file  # type: ignore[attr-defined]
    batch.update(
        {
            "pid": process.pid,
            "state": "running",
            "log": str(log_path),
            "started_at": now(),
        }
    )
    return process


def build_batch_command(
    *,
    batch: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    state_dir: Path,
    benchmark_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--task",
        task["task"],
        "--config",
        args.config,
        "--expert-learning-file",
        task["expert_learning_file"],
        "--experiment-mode",
        args.experiment_mode,
        "--seeds",
        *[str(seed) for seed in batch["seeds"]],
        "--batch-id",
        batch["batch_id"],
        "--benchmark-dir",
        str(benchmark_dir),
        "--state-dir",
        str(state_dir),
        "--max-steps",
        str(task["max_steps"]),
        "--max-parallel",
        str(args.max_parallel_per_gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--request-timeout",
        str(args.request_timeout),
        "--camera-policy",
        args.camera_policy,
        "--initial-camera-view",
        args.initial_camera_view,
        "--visual-width",
        str(args.visual_width),
        "--visual-height",
        str(args.visual_height),
        "--model-debug-overlay",
        args.model_debug_overlay,
        "--record-overlay",
        args.record_overlay,
    ]
    if args.fixed_views:
        command.extend(
            [
                "--fixed-views",
                *args.fixed_views,
                "--fixed-view-width",
                str(args.fixed_view_width),
                "--fixed-view-height",
                str(args.fixed_view_height),
                "--fixed-view-jpeg-quality",
                str(args.fixed_view_jpeg_quality),
            ]
        )
    return command


def launch_wave(
    *,
    batches: list[dict[str, Any]],
    task: dict[str, Any],
    status: dict[str, Any],
    status_path: Path,
    processes: list[subprocess.Popen[str]],
    args: argparse.Namespace,
    state_dir: Path,
    benchmark_dir: Path,
    pipeline_dir: Path,
    stopping: Any,
) -> None:
    task["attempts"].extend(batches)
    status["active_batches"] = batches
    write_status(status_path, status)
    for index, batch in enumerate(batches):
        if stopping():
            return
        command = [
            sys.executable,
            str(RUNNER),
            "--task",
            task["task"],
            "--config",
            args.config,
            "--expert-learning-file",
            task["expert_learning_file"],
            "--experiment-mode",
            args.experiment_mode,
            "--seeds",
            *[str(seed) for seed in batch["seeds"]],
            "--batch-id",
            batch["batch_id"],
            "--benchmark-dir",
            str(benchmark_dir),
            "--state-dir",
            str(state_dir),
            "--max-steps",
            str(task["max_steps"]),
            "--max-parallel",
            str(args.max_parallel_per_gpu),
            "--poll-seconds",
            str(args.poll_seconds),
            "--request-timeout",
            str(args.request_timeout),
            "--camera-policy",
            args.camera_policy,
            "--initial-camera-view",
            args.initial_camera_view,
            "--visual-width",
            str(args.visual_width),
            "--visual-height",
            str(args.visual_height),
            "--model-debug-overlay",
            args.model_debug_overlay,
            "--record-overlay",
            args.record_overlay,
        ]
        if args.fixed_views:
            command.extend(
                [
                    "--fixed-views",
                    *args.fixed_views,
                    "--fixed-view-width",
                    str(args.fixed_view_width),
                    "--fixed-view-height",
                    str(args.fixed_view_height),
                    "--fixed-view-jpeg-quality",
                    str(args.fixed_view_jpeg_quality),
                ]
            )
        log_path = pipeline_dir / f"{batch['batch_id']}.log"
        log_file = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(batch["gpu"])
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        process._pipeline_log_file = log_file  # type: ignore[attr-defined]
        processes.append(process)
        batch.update(
            {
                "pid": process.pid,
                "state": "running",
                "log": str(log_path),
                "started_at": now(),
            }
        )
        refresh_status(status)
        write_status(status_path, status)
        if index + 1 < len(batches):
            sleep_with_status(
                args.stagger_seconds,
                status_path,
                status,
                stopping=stopping,
            )

    while any(process.poll() is None for process in processes[-len(batches) :]):
        refresh_status(status)
        write_status(status_path, status)
        if stopping():
            return
        time.sleep(args.poll_seconds)

    for batch, process in zip(batches, processes[-len(batches) :], strict=True):
        batch["returncode"] = process.poll()
        batch["state"] = "completed" if process.returncode == 0 else "process_failed"
        batch["finished_at"] = now()
        log_file = getattr(process, "_pipeline_log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()
    refresh_status(status)
    write_status(status_path, status)
    status["active_batches"] = []
    processes[:] = [process for process in processes if process.poll() is None]


def refresh_status(status: dict[str, Any]) -> None:
    for task in status.get("tasks", []):
        latest = {str(seed): {"seed": seed, "category": "WAIT"} for seed in task["seeds"]}
        for batch in task.get("attempts", []):
            batch_status = read_json(Path(batch["status_path"]))
            evaluations = batch_status.get("evaluations") or {}
            pending = {int(seed) for seed in batch_status.get("pending_seeds") or []}
            for seed in batch["seeds"]:
                item = evaluations.get(str(seed))
                if isinstance(item, dict):
                    result = classify_evaluation(item)
                    result.update(
                        {
                            "seed": seed,
                            "gpu": batch["gpu"],
                            "batch_id": batch["batch_id"],
                            "run_dir": item.get("run_dir"),
                            "log": item.get("log"),
                            "steps": action_count(item),
                        }
                    )
                    latest[str(seed)] = result
                elif batch.get("state") in {"process_failed", "completed"}:
                    latest[str(seed)] = {
                        "seed": seed,
                        "category": "INFRA",
                        "gpu": batch["gpu"],
                        "batch_id": batch["batch_id"],
                    }
                elif seed in pending or not batch_status:
                    latest[str(seed)] = {
                        "seed": seed,
                        "category": "WAIT",
                        "gpu": batch["gpu"],
                        "batch_id": batch["batch_id"],
                    }
        task["latest_results"] = latest
        categories = [item["category"] for item in latest.values()]
        task["counts"] = {
            "total": len(task["seeds"]),
            "success": categories.count("SUCCESS"),
            "failure": categories.count("FAILED"),
            "infra": categories.count("INFRA"),
            "running": categories.count("RUN"),
            "waiting": categories.count("WAIT"),
        }
    status["updated_at"] = now()


def classify_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    state = str(evaluation.get("state") or "unknown")
    if state == "running":
        return {"category": "RUN"}
    run_dir = evaluation.get("run_dir")
    result = read_json(Path(run_dir) / "result.json") if isinstance(run_dir, str) else {}
    stop_reason = str(result.get("stop_reason") or "")
    if (
        state != "completed"
        or evaluation.get("returncode") not in {None, 0}
        or INFRASTRUCTURE_ERROR.search(stop_reason)
    ):
        return {"category": "INFRA", "stop_reason": stop_reason}
    success = result.get("success", evaluation.get("success"))
    if not isinstance(success, bool):
        return {"category": "INFRA", "stop_reason": stop_reason}
    return {
        "category": "SUCCESS" if success else "FAILED",
        "success": success,
        "stop_reason": stop_reason,
    }


def unresolved_infra_seeds(task: dict[str, Any]) -> list[int]:
    return [
        int(seed)
        for seed, item in task.get("latest_results", {}).items()
        if item.get("category") == "INFRA"
    ]


def action_count(evaluation: dict[str, Any]) -> int:
    run_dir = evaluation.get("run_dir")
    result = read_json(Path(run_dir) / "result.json") if isinstance(run_dir, str) else {}
    steps = result.get("steps")
    if isinstance(steps, list):
        return max(0, len(steps) - 1)
    log = evaluation.get("log")
    if not isinstance(log, str):
        return 0
    try:
        count = 0
        for line in Path(log).read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if not line.startswith("assistant>"):
                continue
            try:
                payload = json.loads(line.removeprefix("assistant>").strip())
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("action"), str):
                count += 1
        return count
    except OSError:
        return 0


def empty_counts(total: int) -> dict[str, int]:
    return {
        "total": total,
        "success": 0,
        "failure": 0,
        "infra": 0,
        "running": 0,
        "waiting": total,
    }


def sleep_with_status(
    seconds: float,
    status_path: Path,
    status: dict[str, Any],
    *,
    stopping: Any,
) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not stopping():
        refresh_status(status)
        write_status(status_path, status)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def terminate_processes(
    processes: list[subprocess.Popen[str]], status: dict[str, Any]
) -> None:
    for task in status.get("tasks", []):
        for batch in task.get("attempts", []):
            batch_status = read_json(Path(batch["status_path"]))
            for evaluation in (batch_status.get("evaluations") or {}).values():
                pid = evaluation.get("pid")
                if isinstance(pid, int):
                    terminate_group(pid)
    for process in processes:
        if process.poll() is None:
            terminate_group(process.pid)
        log_file = getattr(process, "_pipeline_log_file", None)
        if log_file is not None and not log_file.closed:
            log_file.close()


def terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def validate_leaf(value: str) -> str:
    candidate = value.strip()
    path = Path(candidate)
    if not candidate or path.is_absolute() or len(path.parts) != 1 or candidate in {".", ".."}:
        raise ValueError("pipeline-id must be a single directory name")
    return candidate


def write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
