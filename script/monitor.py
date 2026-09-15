"""Show summary learning and evaluation progress for a benchmark run."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent/scripts"))
from monitor_sequential_gpu_eval_pipeline import render


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run = args.output_dir / args.run_id
    pipeline = run / "state/runs/data/eval_pipelines" / args.run_id / "pipeline_status.json"
    while True:
        if pipeline.exists():
            text = render(pipeline, 5)
        else:
            launch = read(run / "launch_status.json")
            text = f"VA-Bench {args.run_id} | {launch.get('state', 'waiting')}\n"
            for task in launch.get("tasks", []):
                done = (run / "summaries" / task / "expert_learning.json").is_file()
                text += f"{task:34} {'summary ready' if done else 'waiting / learning'}\n"
        print(("" if args.once else "\033[2J\033[H") + text, flush=True)
        if args.once:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
