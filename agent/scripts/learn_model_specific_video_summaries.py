from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent.robotwin.expert_learning import read_summary_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn isolated model-specific summaries from RGB-only expert videos."
    )
    parser.add_argument(
        "--task-demo",
        action="append",
        required=True,
        metavar="TASK:EXPERT_SEED:DEMO_DIR",
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--state-dir", default=".agent")
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--active-arm", choices=["left", "right", "both"], default="right")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.request_timeout <= 0 or args.retries < 0:
        raise ValueError("request timeout must be positive and retries nonnegative")
    return args


def parse_task_demo(value: str) -> tuple[str, int, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(f"invalid --task-demo {value!r}")
    return parts[0], int(parts[1]), Path(parts[2]).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = now()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model = (
        os.environ.get("AGENT_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if not model:
        raise ValueError("AGENT_MODEL or OPENAI_MODEL is required")
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

    benchmark_dir = Path(args.benchmark_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    summary_root = Path(args.summary_root).resolve()
    tasks = [parse_task_demo(value) for value in args.task_demo]
    work_dir = summary_root / "_learning"
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = summary_root / "learning_status.json"
    status: dict[str, Any] = {
        "format": "robotwin_model_specific_video_learning_v1",
        "model": model,
        "state": "running",
        "created_at": now(),
        "tasks": [
            {"task": task, "expert_seed": seed, "demo_dir": str(demo), "state": "waiting"}
            for task, seed, demo in tasks
        ],
    }
    write_status(status_path, status)

    try:
        for index, (task, expert_seed, demo_dir) in enumerate(tasks):
            if not demo_dir.is_dir():
                raise FileNotFoundError(f"expert demo directory not found: {demo_dir}")
            item = status["tasks"][index]
            log_path = work_dir / f"{index + 1:02d}_{task}.log"
            item.update({"state": "learning", "log": str(log_path)})
            write_status(status_path, status)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["AGENT_STATE_DIR"] = str(state_dir)

            returncode = 1
            source: Path | None = None
            for attempt in range(args.retries + 1):
                run_id = f"{args.run_prefix}_{index + 1:02d}_{task}_a{attempt + 1}"
                run_dir = state_dir / "runs" / task / f"seed_{expert_seed}" / run_id
                source = run_dir / "expert_learning.json"
                command = [
                    sys.executable,
                    "-m",
                    "agent",
                    "--timeout",
                    str(args.request_timeout),
                    "eval",
                    "robotwin",
                    "--task",
                    task,
                    "--config",
                    args.config,
                    "--seed",
                    str(expert_seed),
                    "--active-arm",
                    args.active_arm,
                    "--experiment-mode",
                    "model-specific-summary",
                    "--expert-demo-dir",
                    str(demo_dir),
                    "--run-id",
                    run_id,
                    "--benchmark-dir",
                    str(benchmark_dir),
                    "--learning-only",
                    "--quiet-events",
                ]
                item.update({"attempt": attempt + 1, "run_id": run_id})
                write_status(status_path, status)
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n=== attempt {attempt + 1} ===\n")
                    returncode = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    ).returncode
                if returncode == 0 and source.is_file():
                    break
            if source is None or returncode != 0 or not source.is_file():
                raise RuntimeError(
                    f"expert learning failed for {task}: returncode={returncode} file={source}"
                )
            summary_model = read_summary_model(source)
            if summary_model != model:
                raise RuntimeError(
                    f"summary model mismatch for {task}: {summary_model!r} != {model!r}"
                )
            destination = summary_root / task / "expert_learning.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            item.update(
                {
                    "state": "completed",
                    "summary_model": summary_model,
                    "source": str(source),
                    "destination": str(destination),
                }
            )
            write_status(status_path, status)
            print(f"learned {task}: {destination}", flush=True)
        status["state"] = "completed"
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished_at"] = now()
        write_status(status_path, status)


if __name__ == "__main__":
    main()
