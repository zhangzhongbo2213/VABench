# VA-Bench

Vision-language models control a robot through RGB observations and structured
actions. This package includes the agent, active-camera interface, **11 single-arm
and 3 dual-arm task environments**, expert RGB videos, and **20 validated seeds
per task**. Generalization and long-horizon tasks are excluded.

## Setup

Linux, Python 3.10, an NVIDIA GPU, and a working RoboTwin environment with SAPIEN
3.0.0b1, mplib 0.2.1 and cuRobo are required. On a machine with an existing
RoboTwin environment and assets:

```bash
conda activate RoboTwin
cd VA-Bench
python -m pip install --no-deps --no-build-isolation -e ./agent -e ./active_spatial_benchmark_xyz
python script/configure_assets.py /path/to/RoboTwin/assets
python script/check_setup.py
```

For a new Python environment, install the dependencies first:

```bash
conda create -n va-bench python=3.10 -y
conda activate va-bench
bash script/install.sh
```

The installer needs Git, a CUDA toolkit compatible with PyTorch, and system
`ffmpeg`, `libgl1`, `libglib2.0-0`, `libvulkan1` and `vulkan-tools`.

For a checkout without `assets/`, download the
[main-task asset bundle](https://github.com/zhangzhongbo2213/VA-Bench-test/releases/tag/assets-v1)
using GitHub CLI (`gh`). This repository is private; authenticate with
`gh auth login` using an account with repository access, then run from `VA-Bench/`:

```bash
gh release download assets-v1 --repo zhangzhongbo2213/VA-Bench-test \
  --pattern 'va-bench-main-assets-v1.tar.gz*' --dir outputs/assets_download
(cd outputs/assets_download && sha256sum -c va-bench-main-assets-v1.tar.gz.sha256)
tar --keep-old-files -xzf outputs/assets_download/va-bench-main-assets-v1.tar.gz -C .
python script/check_setup.py
```

The bundle includes the robot and every object variant used by the 14 main
tasks with `vabench_eval` / `demo_clean`. `check_setup.py` verifies bundled asset
checksums. Existing full RoboTwin assets can still be linked as above.
To test actual rendering and stepping for all 14 environments without a model:

```bash
CUDA_VISIBLE_DEVICES=0 python script/check_setup.py --smoke
```

## Run evaluation

Set an OpenAI-compatible endpoint (a locally served vision model also works):

```bash
export AGENT_BASE_URL="http://localhost:8000/v1"
export AGENT_MODEL="your-model-name"
export AGENT_API_KEY="EMPTY"  # use the provider key for a hosted endpoint

# First test: learn a summary, then evaluate one validated seed.
python script/run_benchmark.py --run-id smoke \
  --tasks grasp_single_cube --num-seeds 1 --gpus 0 --parallel 1

# Full suite: 14 tasks × 20 seeds, three environments per selected GPU.
python script/run_benchmark.py --run-id full --gpus 0 1 --parallel 3

# In another terminal:
python script/monitor.py --run-id full
```

Use `--wire-api anthropic_messages` for Anthropic Messages, or `--wire-api responses`
for Responses. Each run first learns model-specific summaries from the bundled
videos, then evaluates the selected tasks. `--dry-run` prints the commands without
calling a model. Use a new `--run-id` for each experiment.

To reuse summaries, add `--summary-root /path/to/summaries`; each task needs
`TASK/expert_learning.json`. Add `--summary-mode shared-summary` when the summaries
come from another model. Model-specific mode validates the summary model name.

Defaults: active camera, actual RGB rendering at **1280×960**, JPEG 95, at most
4 images per request, 262144-token context, 4096 output tokens, streaming enabled,
temperature omitted. OpenAI reasoning defaults to `medium`; Anthropic thinking
defaults to `enabled` with a 1024-token budget. For models without these options:

```bash
export AGENT_REASONING_EFFORT=none
export AGENT_ANTHROPIC_THINKING=disabled
```

Task-specific horizons and the exact seed lists are in
[`configs/main_tasks.json`](configs/main_tasks.json). `--num-seeds N` takes the
first N validated seeds; the lists are not always contiguous. API/network failures
are classified as infrastructure errors and retried up to three times. Success
rate is `OK / (OK + FAIL)`; unresolved infrastructure errors are reported separately.

Results, summaries, videos and transcripts are under `outputs/RUN_ID/`.
`launch_status.json` tracks the stages; detailed evaluation status is under
`state/runs/data/eval_pipelines/RUN_ID/pipeline_status.json`.

## Tasks

Single arm: `grasp_single_cube`, `grasp_single_pen`, `grasp_single_bottle`,
`grasp_single_bottle_upright`, `grasp_pen_leaning_cube`, `click_bell_right`,
`beat_block_hammer_right`, `place_cube_in_bowl`, `place_single_cube`,
`place_single_bottle_upright`, `place_cube_on_cube`.

Dual arm: `lift_pot`, `place_shoe`, `handover_horizontal_block`.

Built on RoboTwin. See [LICENSE](LICENSE).
