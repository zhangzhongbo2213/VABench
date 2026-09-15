"""Learn from the bundled demonstrations, then evaluate the main task suite."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agent" / "scripts"


def build_plan(args):
    suite = json.loads((ROOT / "configs/main_tasks.json").read_text())
    tasks = suite["tasks"]
    if args.tasks:
        unknown = set(args.tasks) - {t["task"] for t in tasks}
        if unknown:
            raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["task"] in args.tasks]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run-id must be a simple name, for example my-model-01")
    if not 1 <= args.num_seeds <= 20 or args.parallel < 1:
        raise ValueError("num-seeds must be 1..20 and parallel must be positive")
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        raise ValueError("GPU indices must be unique and nonnegative")
    run = args.output_dir.resolve() / args.run_id
    state = run / "state"
    summaries = args.summary_root.resolve() if args.summary_root else run / "summaries"
    commands = []
    if not args.summary_root:
        for arm in ("right", "both"):
            selected = [t for t in tasks if t["active_arm"] == arm]
            if not selected:
                continue
            command = [sys.executable, str(SCRIPTS / "learn_model_specific_video_summaries.py"),
                       "--benchmark-dir", str(ROOT / "active_spatial_benchmark_xyz"),
                       "--state-dir", str(state), "--summary-root", str(summaries),
                       "--run-prefix", f"{args.run_id}_{arm}", "--active-arm", arm,
                       "--config", args.config, "--request-timeout", str(args.timeout)]
            for task in selected:
                demo = ROOT / task["demo_dir"]
                if not (demo / "metadata.json").is_file():
                    raise FileNotFoundError(f"Missing demonstration: {demo}")
                command += ["--task-demo", f"{task['task']}:{task['expert_seed']}:{demo}"]
            commands.append(command)
    else:
        for task in tasks:
            summary = summaries / task["task"] / "expert_learning.json"
            if not summary.is_file():
                raise FileNotFoundError(f"Missing summary: {summary}")
    command = [sys.executable, str(SCRIPTS / "run_sequential_gpu_eval_pipeline.py"),
               "--pipeline-id", args.run_id, "--benchmark-dir", str(ROOT / "active_spatial_benchmark_xyz"),
               "--state-dir", str(state), "--summary-root", str(summaries),
               "--safe-seed-dir", str(ROOT / "configs/seeds"),
               "--config", args.config, "--gpus", *map(str, args.gpus),
               "--max-parallel-per-gpu", str(args.parallel), "--cross-task",
               "--experiment-mode", args.summary_mode, "--camera-policy", "fine",
               "--visual-width", "1280", "--visual-height", "960",
               "--request-timeout", str(args.timeout),
               "--max-infra-retries", str(args.infra_retries)]
    for task in tasks:
        command += ["--task-spec", f"{task['task']}:{task['max_steps']}",
                    "--seed-override", f"{task['task']}:" + ",".join(map(str, task["seeds"][:args.num_seeds]))]
    commands.append(command)
    return run, commands, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("ANTHROPIC_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL"))
    parser.add_argument("--wire-api", choices=("chat_completions", "anthropic_messages", "responses"), default=os.environ.get("AGENT_WIRE_API", "chat_completions"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--parallel", type=int, default=3, help="Environment processes per GPU")
    parser.add_argument("--num-seeds", type=int, default=20, help="Use the first N validated seeds per task")
    parser.add_argument("--config", default="vabench_eval")
    parser.add_argument("--summary-root", type=Path, help="Reuse TASK/expert_learning.json instead of learning")
    parser.add_argument("--summary-mode", choices=("model-specific-summary", "shared-summary"), default="model-specific-summary")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--infra-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating runs or calling a model")
    args = parser.parse_args()
    if not args.model or not args.base_url:
        parser.error("Set --model and --base-url (or AGENT_MODEL and AGENT_BASE_URL)")
    if args.timeout <= 0 or args.infra_retries < 0:
        parser.error("timeout must be positive and infra-retries nonnegative")
    if args.summary_mode == "shared-summary" and not args.summary_root:
        parser.error("shared-summary requires --summary-root")
    if not (ROOT / "task_config" / f"{args.config}.yml").is_file():
        parser.error("Task configuration does not exist")
    try:
        run, commands, tasks = build_plan(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"VA-Bench | {len(tasks)} tasks / {len(tasks) * args.num_seeds} episodes", flush=True)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    if run.exists():
        parser.error(f"Run already exists: {run}. Use a new --run-id.")
    env = os.environ.copy()
    env.update(AGENT_MODEL=args.model, AGENT_BASE_URL=args.base_url.rstrip("/"), AGENT_WIRE_API=args.wire_api)
    if args.wire_api == "anthropic_messages" and not env["AGENT_BASE_URL"].endswith(("/v1", "/v1/messages")):
        env["AGENT_BASE_URL"] += "/v1"
    # Endpoints serving local models may accept a placeholder API key.
    if not any(env.get(k) for k in ("AGENT_API_KEY", "AGENT_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")):
        env["AGENT_API_KEY"] = "EMPTY"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "agent/src"), str(ROOT / "active_spatial_benchmark_xyz"), str(ROOT)])
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run([sys.executable, str(ROOT / "script/check_setup.py")], cwd=ROOT, env=env, check=True)
    run.mkdir(parents=True)
    status_file = run / "launch_status.json"
    status = {"model": args.model, "wire_api": args.wire_api, "state": "starting", "tasks": [t["task"] for t in tasks], "episodes": len(tasks) * args.num_seeds}
    try:
        for i, command in enumerate(commands):
            status.update(state="evaluating" if i == len(commands) - 1 else "learning", stage=i + 1)
            status_file.write_text(json.dumps(status, indent=2) + "\n")
            print(f"Stage {i + 1}/{len(commands)}: {status['state']}", flush=True)
            stage_env = dict(env)
            if status["state"] == "learning":
                stage_env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
            subprocess.run(command, cwd=ROOT, env=stage_env, check=True)
        status["state"] = "completed"
    except BaseException:
        status["state"] = "interrupted_or_failed"
        raise
    finally:
        status_file.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
