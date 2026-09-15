"""Check dependencies, assets, demonstrations and all 14 task imports."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Reset, render and step each environment on the selected GPU; no model API calls")
    parser.add_argument("--tasks", nargs="+", help="Limit smoke checks to these tasks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/setup_check")
    args = parser.parse_args()
    suite = json.loads((ROOT / "configs/main_tasks.json").read_text())
    tasks = suite["tasks"]
    errors = []
    for module in ("numpy", "yaml", "PIL", "imageio", "imageio_ffmpeg", "sapien", "mplib", "toppra", "torch", "transforms3d", "trimesh", "open3d", "cv2", "h5py", "gymnasium", "curobo"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing Python dependency: {module}")
    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg is not available on PATH")
    for rel in ("embodiments/aloha-agilex/config.yml", "objects/objaverse/list.json", "objects/cube/textured.obj"):
        if not (ROOT / "assets" / rel).is_file():
            errors.append(f"Missing asset: assets/{rel}")
    for obj in ("001_bottle", "002_bowl", "020_hammer", "041_shoe", "050_bell", "058_markpen", "060_kitchenpot"):
        if not (ROOT / "assets/objects" / obj).is_dir():
            errors.append(f"Missing object directory: {obj}")
    for task in tasks:
        name = task["task"]
        for path in (ROOT / "envs" / f"{name}.py", ROOT / "description/task_instruction" / f"{name}.json", ROOT / "configs/seeds" / f"{name}.json"):
            if not path.is_file():
                errors.append(f"Missing file: {path.relative_to(ROOT)}")
        demo = ROOT / task["demo_dir"]
        try:
            metadata = json.loads((demo / "metadata.json").read_text())
            if not (demo / metadata["video"]).is_file():
                errors.append(f"Missing demo video for {name}")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Invalid demo for {name}: {exc}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "agent/src"), str(ROOT / "active_spatial_benchmark_xyz"), str(ROOT)])
    code = f"""
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
assert torch.cuda.is_available(), 'CUDA GPU unavailable'
root = Path({str(ROOT)!r}).resolve()
for name in {['envs.' + t['task'] for t in tasks]!r}:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
from agent.robotwin.adapter import RoboTwinAdapter
import agent.robotwin.adapter as adapter_module
assert Path(adapter_module.__file__).resolve().is_relative_to(root), adapter_module.__file__
with TemporaryDirectory() as tmp:
    adapter = RoboTwinAdapter(Path(tmp), model_debug_overlay='none')
    adapter._load_modules()
    module = importlib.import_module('active_spatial_benchmark.env')
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
print('READY: 14 task environments and agent imported from this checkout')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr)
        raise SystemExit(1)
    print(result.stdout.strip())
    if args.smoke:
        unknown = set(args.tasks or []) - {t["task"] for t in tasks}
        if unknown:
            parser.error(f"Unknown tasks: {', '.join(sorted(unknown))}")
        selected = [t for t in tasks if not args.tasks or t["task"] in args.tasks]
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        for task in selected:
            name = task["task"]
            code = f"""
from pathlib import Path
import json
from PIL import Image
from agent.robotwin.adapter import RoboTwinAdapter
adapter = RoboTwinAdapter(Path({str(output / name)!r}), task={name!r}, config='vabench_eval', seed={task['seeds'][0]}, active_arm={task['active_arm']!r}, max_steps=2, model_debug_overlay='none', record_overlay='none')
try:
    frame = adapter.reset()
    assert Image.open(frame.path).size == (1280, 960), Image.open(frame.path).size
    _, result = adapter.step('camera.look_at_workspace')
    assert result['info']['action_valid'], result['info']
    assert result['info']['planner_success'], result['info']
    videos = adapter.write_videos()
    assert all(p.is_file() and p.stat().st_size for p in videos.values()), videos
    print('SMOKE_OK ' + {name!r})
finally:
    adapter.close()
"""
            print(f"Checking {name} ...", flush=True)
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            (output / f"{name}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                errors.append(name)
                print(f"FAILED: {name}; see {output / (name + '.log')}", flush=True)
            else:
                print(f"PASS: {name}", flush=True)
        if errors:
            raise SystemExit(1)
        print(f"SMOKE PASSED: {len(selected)} tasks")


if __name__ == "__main__":
    main()
