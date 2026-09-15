from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a sequential multi-GPU RoboTwin evaluation pipeline."
    )
    parser.add_argument("--status", required=True)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.refresh <= 0:
        raise ValueError("--refresh must be positive")
    return args


def main() -> None:
    args = parse_args()
    status_path = Path(args.status).resolve()
    while True:
        output = render(status_path, args.refresh)
        if args.once:
            print(output)
            return
        print("\033[2J\033[H" + output, end="", flush=True)
        time.sleep(args.refresh)


def render(status_path: Path, refresh: float) -> str:
    status = read_json(status_path)
    if not status:
        return f"Waiting for pipeline status: {status_path}\n"
    tasks = status.get("tasks") or []
    current_index = status.get("current_task_index")
    current_task = (
        tasks[current_index].get("task")
        if isinstance(current_index, int) and 0 <= current_index < len(tasks)
        else "-"
    )
    lines = [
        f"RoboTwin sequential multi-GPU evaluation | {datetime.now().strftime('%F %T')}",
        f"pipeline={status.get('pipeline_id')} | state={status.get('state')} | "
        f"current={current_task} | refresh={refresh:g}s",
        f"model={status.get('model')} | scheduling={status.get('scheduling_mode', 'task-barrier')} | "
        f"parallel/GPU={status.get('max_parallel_per_gpu', 1)}",
        "",
    ]
    lines.extend(gpu_rows(status.get("gpus") or []))
    lines.extend(
        [
            "",
            f"{'TASK':31} {'STATE':20} {'VALID':>6} {'OK':>3} {'FAIL':>4} "
            f"{'INFRA':>5} {'RUN':>3} {'WAIT':>4} {'SR':>7}",
            "-" * 94,
        ]
    )
    for task in tasks:
        counts = task.get("counts") or {}
        success = int(counts.get("success") or 0)
        failure = int(counts.get("failure") or 0)
        valid = success + failure
        total = int(counts.get("total") or len(task.get("seeds") or []))
        success_rate = f"{100 * success / valid:.1f}%" if valid else "-"
        lines.append(
            f"{str(task.get('task')):31} {str(task.get('state')):20} "
            f"{f'{valid}/{total}':>6} {success:>3} {failure:>4} "
            f"{int(counts.get('infra') or 0):>5} "
            f"{int(counts.get('running') or 0):>3} "
            f"{int(counts.get('waiting') or 0):>4} {success_rate:>7}"
        )

    active = []
    for task in tasks:
        for item in (task.get("latest_results") or {}).values():
            if item.get("category") == "RUN":
                active.append(
                    (
                        int(item.get("gpu")),
                        task.get("task"),
                        int(item.get("seed")),
                        environment_step(item),
                    )
                )
    lines.extend(["", "Active workers (GPU | task | seed | step):"])
    if active:
        for gpu, task, seed, step in sorted(active):
            lines.append(f"  GPU {gpu} | {task:31} | seed={seed} step={step}")
    else:
        lines.append("  -")

    unresolved = []
    for task in tasks:
        infra = [
            int(seed)
            for seed, item in (task.get("latest_results") or {}).items()
            if item.get("category") == "INFRA"
        ]
        if infra:
            unresolved.append(f"  {task.get('task')}: {' '.join(map(str, sorted(infra)))}")
    lines.extend(["", "Latest infrastructure-invalid seeds:"])
    lines.extend(unresolved or ["  -"])
    return "\n".join(lines) + "\n"


def environment_step(item: dict[str, Any]) -> int:
    run_dir = item.get("run_dir")
    if isinstance(run_dir, str):
        largest = -1
        try:
            for path in (Path(run_dir) / "model_frames").glob("step_*.png"):
                match = re.match(r"step_(\d+)_", path.name)
                if match:
                    largest = max(largest, int(match.group(1)))
        except OSError:
            pass
        if largest >= 0:
            return largest
    return int(item.get("steps") or 0)


def gpu_rows(indices: list[int]) -> list[str]:
    wanted = set(indices)
    if not wanted:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ["GPU status unavailable"]
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or int(fields[0]) not in wanted:
            continue
        index, used, total, utilization = fields
        rows.append(f"GPU {index}: {used:>5}/{total} MiB | util {utilization:>3}%")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
