from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent.config import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_ANTHROPIC_THINKING,
    DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS,
    DEFAULT_STREAM,
    DEFAULT_TRANSIENT_HTTP_RETRY_ATTEMPTS,
    DEFAULT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS,
)
from agent.robotwin.expert_learning import (  # noqa: E402
    EXPERIMENT_MODES,
    MODEL_SPECIFIC_SUMMARY_MODE,
    SHARED_SUMMARY_MODE,
    read_summary_model,
    validate_experiment_lineage as validate_model_lineage,
)
from agent.robotwin.spatial_tool import (  # noqa: E402
    DIRECT_FRAME_RESEARCH_TASKS,
)


INFRASTRUCTURE_ERROR = re.compile(
    r"provider request failed|provider response missing text content|HTTP\s+[45]\d\d|usage[_ ]limit|daily[_ ]limit|"
    r"rate[_ ]limit|insufficient.*balance|quota|connection (?:refused|reset)|"
    r"network error|timed?\s*out|timeout|CUDA out of memory|Traceback|"
    r"exceed_context_size_error|available context size",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn once from an RGB-only expert video, then evaluate seeds in parallel."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--active-arm", choices=["left", "right", "both"], default=None)
    expert_source = parser.add_mutually_exclusive_group(required=True)
    expert_source.add_argument("--expert-demo-dir")
    expert_source.add_argument(
        "--expert-learning-file",
        help="Reuse an existing expert_learning.json and skip the learning stage.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        help=(
            "shared-summary reuses one fixed summary across evaluation models; "
            "model-specific-summary learns a new summary with the evaluation model "
            "or reuses a prelearned summary from that same model. "
            "When omitted, the mode is inferred from the selected expert source."
        ),
    )
    parser.add_argument("--expert-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--state-dir", default=".agent")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--fixed-views",
        nargs="+",
        choices=["topdown", "side", "front_side_45", "side_top_45", "oblique_45"],
        default=[],
        help="Send all listed fixed observer views at every model turn.",
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
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume an existing batch, adopting live seed processes and pending seeds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-provider-request timeout in seconds for learning and evaluation.",
    )
    parser.add_argument(
        "--spatial-tool",
        action="store_true",
        help="Let the evaluation model call spatial.verify_pregrasp when it chooses.",
    )
    parser.add_argument(
        "--spatial-close-audit",
        action="store_true",
        help="Run a policy-invisible spatial audit before each close action.",
    )
    parser.add_argument(
        "--spatial-candidate-proposals",
        action="store_true",
        help="Let the model call spatial.propose_grasp_candidates.",
    )
    parser.add_argument("--spatial-checkpoint")
    parser.add_argument("--spatial-calibration")
    parser.add_argument("--spatial-view-ranker")
    parser.add_argument("--spatial-outcome-checkpoint")
    parser.add_argument("--spatial-candidate-ranker")
    parser.add_argument("--spatial-semantic-part-checkpoint")
    parser.add_argument("--spatial-direct-grasp-frame-checkpoint")
    parser.add_argument("--spatial-intent-embedding-store")
    parser.add_argument("--spatial-semantic-view-ranker")
    parser.add_argument(
        "--spatial-semantic-view-mode",
        choices=("analytic", "shadow"),
        default="analytic",
    )
    parser.add_argument("--spatial-dynamic-intent-encoder-python")
    parser.add_argument("--spatial-dynamic-intent-encoder-script")
    parser.add_argument("--spatial-dynamic-intent-encoder-snapshot")
    parser.add_argument("--spatial-max-additional-views", type=int, default=2)
    parser.add_argument("--spatial-required-confidence", type=float, default=0.75)
    parser.add_argument("--spatial-max-grasp-candidates", type=int, default=3)
    parser.add_argument(
        "--spatial-pen-radial-half-extent-m", type=float, default=0.01
    )
    args = parser.parse_args()
    if args.active_arm is None:
        args.active_arm = (
            "both"
            if args.task.lower()
            in {
                "place_shoe",
                "handover_mic",
                "handover_horizontal_block",
                "handover_block",
                "handover_cube_to_target",
                "lift_pot",
            }
            else "right"
        )
    if args.camera_policy is None:
        args.camera_policy = "fixed" if args.fixed_views else "fine"

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive")
    if args.camera_policy == "fixed" and not args.fixed_views:
        raise ValueError("fixed camera policy requires at least one fixed view")
    if args.camera_policy != "fixed" and args.fixed_views:
        raise ValueError("--fixed-views can only be used with --camera-policy fixed")
    if args.request_timeout <= 0:
        raise ValueError("request-timeout must be positive")
    if args.spatial_max_additional_views < 0:
        raise ValueError("spatial-max-additional-views must be >= 0")
    if not 0.0 < args.spatial_required_confidence <= 1.0:
        raise ValueError("spatial-required-confidence must be in (0, 1]")
    if args.spatial_max_grasp_candidates < 1:
        raise ValueError("spatial-max-grasp-candidates must be >= 1")
    if args.spatial_pen_radial_half_extent_m <= 0.0:
        raise ValueError("spatial-pen-radial-half-extent-m must be positive")
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
        raise ValueError(
            "AGENT_API_KEY, AGENT_AUTH_TOKEN, ANTHROPIC_API_KEY, "
            "ANTHROPIC_AUTH_TOKEN, or OPENAI_API_KEY is required"
        )

    task_dir_name = sanitize_component(args.task)
    batch_id = validate_leaf(args.batch_id)
    state_dir = Path(args.state_dir).resolve()
    expert_demo_dir = Path(args.expert_demo_dir).resolve() if args.expert_demo_dir else None
    reused_expert_learning_file = (
        Path(args.expert_learning_file).resolve() if args.expert_learning_file else None
    )
    spatial_active = (
        args.spatial_tool
        or args.spatial_close_audit
        or args.spatial_candidate_proposals
    )
    spatial_paths = {
        "checkpoint": args.spatial_checkpoint,
        "calibration": args.spatial_calibration,
        "view_ranker": args.spatial_view_ranker,
        "outcome_checkpoint": args.spatial_outcome_checkpoint,
    }
    if (
        args.spatial_semantic_part_checkpoint
        and args.spatial_direct_grasp_frame_checkpoint
    ):
        raise ValueError(
            "semantic part and direct grasp-frame checkpoints are mutually exclusive"
        )
    learned_paths = {
        "semantic_part_checkpoint": args.spatial_semantic_part_checkpoint,
        "direct_grasp_frame_checkpoint": args.spatial_direct_grasp_frame_checkpoint,
        "intent_embedding_store": args.spatial_intent_embedding_store,
    }
    perception_count = sum(
        bool(learned_paths[name])
        for name in ("semantic_part_checkpoint", "direct_grasp_frame_checkpoint")
    )
    learned_mode = perception_count == 1 and bool(
        learned_paths["intent_embedding_store"]
    )
    if any(learned_paths.values()) and not learned_mode:
        raise ValueError(
            "learned candidate mode requires one perception checkpoint and intent embedding store"
        )
    direct_mode = bool(args.spatial_direct_grasp_frame_checkpoint)
    dynamic_paths = {
        "dynamic_intent_encoder_python": args.spatial_dynamic_intent_encoder_python,
        "dynamic_intent_encoder_script": args.spatial_dynamic_intent_encoder_script,
        "dynamic_intent_encoder_snapshot": args.spatial_dynamic_intent_encoder_snapshot,
    }
    if any(dynamic_paths.values()) and not all(dynamic_paths.values()):
        raise ValueError("dynamic intent encoding requires python, script, and snapshot")
    if any(dynamic_paths.values()) and not direct_mode:
        raise ValueError("dynamic intent encoding requires direct grasp-frame mode")
    if args.spatial_semantic_view_ranker and not learned_mode:
        raise ValueError("semantic view ranker requires learned candidate mode")
    if (
        args.spatial_semantic_view_mode == "shadow"
        and not args.spatial_semantic_view_ranker
    ):
        raise ValueError("semantic view shadow mode requires a ranker checkpoint")
    if learned_mode and args.spatial_candidate_ranker:
        raise ValueError(
            "learned candidate mode and the legacy candidate ranker are mutually exclusive"
        )
    if spatial_active:
        supported_tasks = (
            DIRECT_FRAME_RESEARCH_TASKS
            if direct_mode and args.spatial_candidate_proposals
            else
            {"grasp_single_pen", "grasp_single_bottle", "grasp_single_cube"}
            if learned_mode and args.spatial_candidate_proposals
            else {"grasp_single_pen"}
        )
        if args.task.lower() not in supported_tasks:
            raise ValueError("the configured learned spatial tool does not support this task")
        legacy_required = (
            args.spatial_tool
            or args.spatial_close_audit
            or (args.spatial_candidate_proposals and not learned_mode)
        )
        missing = (
            [name for name, value in spatial_paths.items() if not value]
            if legacy_required
            else []
        )
        if missing:
            raise ValueError("spatial tool requires: " + ", ".join(sorted(missing)))
        for name, value in spatial_paths.items() if legacy_required else ():
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"spatial {name} not found: {path}")
            spatial_paths[name] = str(path)
        args.spatial_checkpoint = spatial_paths["checkpoint"]
        args.spatial_calibration = spatial_paths["calibration"]
        args.spatial_view_ranker = spatial_paths["view_ranker"]
        args.spatial_outcome_checkpoint = spatial_paths["outcome_checkpoint"]
        if learned_mode:
            for name, value in learned_paths.items():
                if value is None:
                    continue
                path = Path(value).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"spatial {name} not found: {path}")
                learned_paths[name] = str(path)
            args.spatial_semantic_part_checkpoint = learned_paths[
                "semantic_part_checkpoint"
            ]
            args.spatial_direct_grasp_frame_checkpoint = learned_paths[
                "direct_grasp_frame_checkpoint"
            ]
            args.spatial_intent_embedding_store = learned_paths[
                "intent_embedding_store"
            ]
            if args.spatial_semantic_view_ranker:
                semantic_view_ranker = Path(
                    args.spatial_semantic_view_ranker
                ).resolve()
                if not semantic_view_ranker.is_file():
                    raise FileNotFoundError(
                        f"spatial semantic view ranker not found: {semantic_view_ranker}"
                    )
                args.spatial_semantic_view_ranker = str(semantic_view_ranker)
            for name, value in dynamic_paths.items():
                if value is None:
                    continue
                path = Path(value).resolve()
                expected = path.is_dir() if name.endswith("snapshot") else path.is_file()
                if not expected:
                    raise FileNotFoundError(f"spatial {name} not found: {path}")
                dynamic_paths[name] = str(path)
            args.spatial_dynamic_intent_encoder_python = dynamic_paths[
                "dynamic_intent_encoder_python"
            ]
            args.spatial_dynamic_intent_encoder_script = dynamic_paths[
                "dynamic_intent_encoder_script"
            ]
            args.spatial_dynamic_intent_encoder_snapshot = dynamic_paths[
                "dynamic_intent_encoder_snapshot"
            ]
        if args.spatial_candidate_ranker:
            candidate_ranker = Path(args.spatial_candidate_ranker).resolve()
            if not candidate_ranker.is_file():
                raise FileNotFoundError(
                    f"spatial candidate_ranker not found: {candidate_ranker}"
                )
            args.spatial_candidate_ranker = str(candidate_ranker)
    experiment_mode = resolve_experiment_mode(
        args.experiment_mode,
        expert_demo_dir=expert_demo_dir,
        expert_learning_file=reused_expert_learning_file,
    )
    evaluation_model = (
        os.environ.get("AGENT_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL
    )
    if reused_expert_learning_file is not None and not reused_expert_learning_file.is_file():
        raise FileNotFoundError(
            f"expert learning file not found: {reused_expert_learning_file}"
        )
    summary_model = (
        read_summary_model(reused_expert_learning_file)
        if reused_expert_learning_file is not None
        else None
    )
    if reused_expert_learning_file is not None:
        validate_model_lineage(
            experiment_mode,
            summary_model=summary_model,
            evaluation_model=evaluation_model,
        )
    benchmark_dir = Path(args.benchmark_dir).resolve()
    batch_dir = state_dir / "runs" / task_dir_name / "batches" / batch_id
    if args.resume_existing:
        resume_existing_batch(
            args=args,
            task_dir_name=task_dir_name,
            state_dir=state_dir,
            batch_dir=batch_dir,
            benchmark_dir=benchmark_dir,
            expert_learning_file=reused_expert_learning_file,
            experiment_mode=experiment_mode,
            evaluation_model=evaluation_model,
        )
        return
    batch_dir.mkdir(parents=True, exist_ok=False)
    status_path = batch_dir / "batch_status.json"

    learning_run_id = f"learning_{batch_id}"
    evaluation_run_id = f"eval_{batch_id}"
    learning_run_dir = (
        state_dir / "runs" / task_dir_name / f"seed_{args.expert_seed}" / learning_run_id
    )
    expert_learning_file = (
        reused_expert_learning_file
        if reused_expert_learning_file is not None
        else learning_run_dir / "expert_learning.json"
    )

    status: dict[str, Any] = {
        "format": "robotwin_parallel_video_eval_batch_v3",
        "batch_id": batch_id,
        "task": args.task,
        "config": args.config,
        "active_arm": args.active_arm,
        "expert_seed": args.expert_seed,
        "expert_demo_dir": str(expert_demo_dir) if expert_demo_dir is not None else None,
        "reused_expert_learning_file": (
            str(reused_expert_learning_file)
            if reused_expert_learning_file is not None
            else None
        ),
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "max_parallel": args.max_parallel,
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
        "request_timeout_seconds": args.request_timeout,
        "experiment_mode": experiment_mode,
        "summary_model": summary_model,
        "evaluation_model": evaluation_model,
        "model": evaluation_model,
        "base_url": (
            os.environ.get("AGENT_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
        "quota_fallback": {
            "base_url": os.environ.get("AGENT_QUOTA_FALLBACK_BASE_URL"),
            "model": os.environ.get("AGENT_QUOTA_FALLBACK_MODEL"),
            "configured": bool(
                os.environ.get("AGENT_QUOTA_FALLBACK_BASE_URL")
                and os.environ.get("AGENT_QUOTA_FALLBACK_MODEL")
            ),
            "cooldown_seconds": os.environ.get(
                "AGENT_QUOTA_FALLBACK_COOLDOWN_SECONDS"
            ),
            "api_key_recorded": False,
        },
        "wire_api": os.environ.get("AGENT_WIRE_API")
        or (
            "anthropic_messages"
            if os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            else "chat_completions"
        ),
        "anthropic_version": os.environ.get("AGENT_ANTHROPIC_VERSION"),
        "temperature": os.environ.get("AGENT_TEMPERATURE"),
        "reasoning_effort": os.environ.get("AGENT_REASONING_EFFORT"),
        "max_tokens": os.environ.get("AGENT_MAX_TOKENS")
        or str(DEFAULT_MAX_TOKENS),
        "stream": (
            os.environ.get("AGENT_STREAM")
            if os.environ.get("AGENT_STREAM") is not None
            else str(DEFAULT_STREAM).lower()
        ),
        "anthropic_thinking": os.environ.get("AGENT_ANTHROPIC_THINKING")
        or DEFAULT_ANTHROPIC_THINKING,
        "anthropic_thinking_budget_tokens": os.environ.get(
            "AGENT_ANTHROPIC_THINKING_BUDGET_TOKENS"
        )
        or str(DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS),
        "response_language": os.environ.get("AGENT_RESPONSE_LANGUAGE"),
        "transient_http_retry": {
            "max_attempts": os.environ.get(
                "AGENT_TRANSIENT_HTTP_RETRY_ATTEMPTS"
            )
            or str(DEFAULT_TRANSIENT_HTTP_RETRY_ATTEMPTS),
            "backoff_seconds": os.environ.get(
                "AGENT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS"
            )
            or str(DEFAULT_TRANSIENT_HTTP_RETRY_BACKOFF_SECONDS),
        },
        "request_image_transport": {
            "max_width": os.environ.get("AGENT_REQUEST_IMAGE_MAX_WIDTH"),
            "max_height": os.environ.get("AGENT_REQUEST_IMAGE_MAX_HEIGHT"),
            "jpeg_quality": os.environ.get("AGENT_REQUEST_IMAGE_JPEG_QUALITY"),
            "max_images": os.environ.get("AGENT_REQUEST_MAX_IMAGES"),
        },
        "spatial_tool": {
            "enabled_for_model": args.spatial_tool,
            "close_shadow_audit": args.spatial_close_audit,
            "candidate_proposals": args.spatial_candidate_proposals,
            "checkpoint": args.spatial_checkpoint,
            "calibration": args.spatial_calibration,
            "view_ranker": args.spatial_view_ranker,
            "outcome_checkpoint": args.spatial_outcome_checkpoint,
            "candidate_ranker": args.spatial_candidate_ranker,
            "semantic_part_checkpoint": args.spatial_semantic_part_checkpoint,
            "direct_grasp_frame_checkpoint": args.spatial_direct_grasp_frame_checkpoint,
            "intent_embedding_store": args.spatial_intent_embedding_store,
            "semantic_view_ranker": args.spatial_semantic_view_ranker,
            "semantic_view_mode": args.spatial_semantic_view_mode,
            "dynamic_intent_encoder_python": args.spatial_dynamic_intent_encoder_python,
            "dynamic_intent_encoder_script": args.spatial_dynamic_intent_encoder_script,
            "dynamic_intent_encoder_snapshot": args.spatial_dynamic_intent_encoder_snapshot,
            "max_additional_views": args.spatial_max_additional_views,
            "required_confidence": args.spatial_required_confidence,
            "max_grasp_candidates": args.spatial_max_grasp_candidates,
            "pen_radial_half_extent_m": args.spatial_pen_radial_half_extent_m,
        },
        "api_key_recorded": False,
        "supervisor_pid": os.getpid(),
        "learned_constraint_file": None,
        "created_at": now(),
        "updated_at": now(),
        "state": "initializing",
        "learning": {
            "run_id": learning_run_id,
            "run_dir": str(learning_run_dir),
            "expert_learning_file": str(expert_learning_file),
            "log": str(batch_dir / "learning.log") if expert_demo_dir is not None else None,
            "pid": None,
            "returncode": 0 if reused_expert_learning_file is not None else None,
            "state": "reused" if reused_expert_learning_file is not None else "pending",
        },
        "pending_seeds": list(args.seeds),
        "evaluations": {},
    }
    write_status(status_path, status)

    env = child_environment(state_dir)
    if expert_demo_dir is not None:
        learning_command = eval_command(
            task=args.task,
            config=args.config,
            seed=args.expert_seed,
            active_arm=args.active_arm,
            benchmark_dir=benchmark_dir,
            run_id=learning_run_id,
            max_steps=args.max_steps,
            experiment_mode=experiment_mode,
            visual_width=args.visual_width,
            visual_height=args.visual_height,
            model_debug_overlay=args.model_debug_overlay,
            record_overlay=args.record_overlay,
            request_timeout=args.request_timeout,
            expert_demo_dir=expert_demo_dir,
            learning_only=True,
        )
        learning_log = Path(status["learning"]["log"])
        with learning_log.open("w", encoding="utf-8") as log_file:
            learning_process = subprocess.Popen(
                learning_command,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["state"] = "learning"
            status["learning"].update({"pid": learning_process.pid, "state": "running"})
            write_status(status_path, status)
            returncode = wait_for_process(
                learning_process, status_path, status, args.poll_seconds
            )

        status["learning"].update(
            {
                "returncode": returncode,
                "state": (
                    "completed"
                    if returncode == 0 and expert_learning_file.exists()
                    else "failed"
                ),
            }
        )
        if returncode != 0 or not expert_learning_file.exists():
            status["state"] = "learning_failed"
            status["finished_at"] = now()
            write_status(status_path, status)
            raise RuntimeError(
                f"expert learning failed returncode={returncode} "
                f"file_exists={expert_learning_file.exists()}"
            )
        summary_model = read_summary_model(expert_learning_file)
        try:
            validate_model_lineage(
                experiment_mode,
                summary_model=summary_model,
                evaluation_model=evaluation_model,
            )
        except ValueError:
            status["state"] = "learning_failed"
            status["summary_model"] = summary_model
            status["finished_at"] = now()
            write_status(status_path, status)
            raise
        status["summary_model"] = summary_model
        write_status(status_path, status)

    processes: dict[int, subprocess.Popen[str]] = {}
    log_files: dict[int, Any] = {}
    pending = list(args.seeds)
    status["state"] = "evaluating"
    try:
        while pending or processes:
            while pending and len(processes) < args.max_parallel:
                seed = pending.pop(0)
                run_dir = state_dir / "runs" / task_dir_name / f"seed_{seed}" / evaluation_run_id
                log_path = batch_dir / f"seed_{seed}.log"
                command = eval_command(
                    task=args.task,
                    config=args.config,
                    seed=seed,
                    active_arm=args.active_arm,
                    benchmark_dir=benchmark_dir,
                    run_id=evaluation_run_id,
                    max_steps=args.max_steps,
                    camera_policy=args.camera_policy,
                    initial_camera_view=args.initial_camera_view,
                    fixed_views=args.fixed_views,
                    fixed_view_width=args.fixed_view_width,
                    fixed_view_height=args.fixed_view_height,
                    fixed_view_jpeg_quality=args.fixed_view_jpeg_quality,
                    experiment_mode=experiment_mode,
                    visual_width=args.visual_width,
                    visual_height=args.visual_height,
                    model_debug_overlay=args.model_debug_overlay,
                    record_overlay=args.record_overlay,
                    request_timeout=args.request_timeout,
                    expert_learning_file=expert_learning_file,
                    spatial_tool=args.spatial_tool,
                    spatial_close_audit=args.spatial_close_audit,
                    spatial_candidate_proposals=args.spatial_candidate_proposals,
                    spatial_checkpoint=args.spatial_checkpoint,
                    spatial_calibration=args.spatial_calibration,
                    spatial_view_ranker=args.spatial_view_ranker,
                    spatial_outcome_checkpoint=args.spatial_outcome_checkpoint,
                    spatial_candidate_ranker=args.spatial_candidate_ranker,
                    spatial_semantic_part_checkpoint=args.spatial_semantic_part_checkpoint,
                    spatial_direct_grasp_frame_checkpoint=args.spatial_direct_grasp_frame_checkpoint,
                    spatial_intent_embedding_store=args.spatial_intent_embedding_store,
                    spatial_semantic_view_ranker=args.spatial_semantic_view_ranker,
                    spatial_semantic_view_mode=args.spatial_semantic_view_mode,
                    spatial_dynamic_intent_encoder_python=args.spatial_dynamic_intent_encoder_python,
                    spatial_dynamic_intent_encoder_script=args.spatial_dynamic_intent_encoder_script,
                    spatial_dynamic_intent_encoder_snapshot=args.spatial_dynamic_intent_encoder_snapshot,
                    spatial_max_additional_views=args.spatial_max_additional_views,
                    spatial_required_confidence=args.spatial_required_confidence,
                    spatial_max_grasp_candidates=args.spatial_max_grasp_candidates,
                    spatial_pen_radial_half_extent_m=args.spatial_pen_radial_half_extent_m,
                )
                log_file = log_path.open("w", encoding="utf-8")
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
                processes[seed] = process
                log_files[seed] = log_file
                status["evaluations"][str(seed)] = {
                    "seed": seed,
                    "run_id": evaluation_run_id,
                    "run_dir": str(run_dir),
                    "log": str(log_path),
                    "pid": process.pid,
                    "returncode": None,
                    "state": "running",
                    "success": None,
                }
            status["pending_seeds"] = list(pending)
            write_status(status_path, status)

            for seed, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_files[seed].close()
                run_dir = Path(status["evaluations"][str(seed)]["run_dir"])
                status["evaluations"][str(seed)].update(
                    evaluation_outcome(run_dir / "result.json", returncode)
                )
                del processes[seed]
            if pending or processes:
                time.sleep(args.poll_seconds)
    finally:
        for seed, log_file in log_files.items():
            if not log_file.closed:
                log_file.close()

    evaluations = list(status["evaluations"].values())
    status["state"] = "completed"
    status["supervisor_pid"] = None
    status["pending_seeds"] = []
    status["finished_at"] = now()
    status["summary"] = {
        "evaluation_count": len(evaluations),
        "process_completed": sum(item["returncode"] == 0 for item in evaluations),
        "process_failed": sum(item["returncode"] != 0 for item in evaluations),
        "infrastructure_failed": sum(
            item.get("state") == "infrastructure_error" for item in evaluations
        ),
        "environment_success": sum(item["success"] is True for item in evaluations),
        "environment_failure": sum(item["success"] is False for item in evaluations),
    }
    write_status(status_path, status)


def resume_existing_batch(
    *,
    args: argparse.Namespace,
    task_dir_name: str,
    state_dir: Path,
    batch_dir: Path,
    benchmark_dir: Path,
    expert_learning_file: Path | None,
    experiment_mode: str,
    evaluation_model: str,
) -> None:
    status_path = batch_dir / "batch_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"batch status not found: {status_path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid batch status: {status_path}") from exc
    if not isinstance(status, dict):
        raise ValueError(f"invalid batch status object: {status_path}")

    expected = {
        "task": args.task,
        "config": args.config,
        "active_arm": args.active_arm,
        "batch_id": args.batch_id,
        "experiment_mode": experiment_mode,
        "evaluation_model": evaluation_model,
    }
    for field, value in expected.items():
        recorded = status.get(field)
        if recorded != value:
            raise ValueError(
                f"cannot resume {args.batch_id}: {field}={recorded!r}, expected {value!r}"
            )
    if list(status.get("seeds") or []) != list(args.seeds):
        raise ValueError("cannot resume batch with a different seed list")
    if int(status.get("max_steps") or 0) != args.max_steps:
        raise ValueError("cannot resume batch with a different max_steps")

    recorded_learning = (
        (status.get("learning") or {}).get("expert_learning_file")
        or status.get("reused_expert_learning_file")
    )
    if not isinstance(recorded_learning, str):
        raise ValueError("existing batch does not record an expert learning file")
    recorded_learning_path = Path(recorded_learning).resolve()
    if expert_learning_file is None:
        expert_learning_file = recorded_learning_path
    if expert_learning_file.resolve() != recorded_learning_path:
        raise ValueError("cannot resume batch with a different expert learning file")
    if not expert_learning_file.is_file():
        raise FileNotFoundError(f"expert learning file not found: {expert_learning_file}")

    previous_supervisor = status.get("supervisor_pid")
    if (
        isinstance(previous_supervisor, int)
        and previous_supervisor != os.getpid()
        and process_matches(previous_supervisor, args.batch_id)
    ):
        raise RuntimeError(
            f"batch already has a live supervisor pid={previous_supervisor}"
        )

    evaluation_run_id = f"eval_{args.batch_id}"
    env = child_environment(state_dir)
    evaluations = status.setdefault("evaluations", {})
    processes: dict[int, subprocess.Popen[str]] = {}
    external_processes: dict[int, int] = {}
    log_files: dict[int, Any] = {}
    completed_seeds: set[int] = set()

    for seed in args.seeds:
        item = evaluations.get(str(seed))
        if not isinstance(item, dict):
            continue
        state = str(item.get("state") or "")
        pid = item.get("pid")
        if (
            state == "running"
            and isinstance(pid, int)
            and process_matches(pid, evaluation_run_id)
        ):
            external_processes[seed] = pid
            continue
        if state == "running":
            finish_orphaned_evaluation(item)
        completed_seeds.add(seed)

    pending = [seed for seed in args.seeds if seed not in completed_seeds and seed not in external_processes]
    status.update(
        {
            "state": "evaluating",
            "supervisor_pid": os.getpid(),
            "max_parallel": args.max_parallel,
            "request_timeout_seconds": args.request_timeout,
            "pending_seeds": list(pending),
            "resumed_at": now(),
        }
    )
    write_status(status_path, status)

    try:
        while pending or processes or external_processes:
            active_count = len(processes) + len(external_processes)
            while pending and active_count < args.max_parallel:
                seed = pending.pop(0)
                run_dir = (
                    state_dir
                    / "runs"
                    / task_dir_name
                    / f"seed_{seed}"
                    / evaluation_run_id
                )
                log_path = batch_dir / f"seed_{seed}.log"
                command = eval_command(
                    task=args.task,
                    config=args.config,
                    seed=seed,
                    active_arm=args.active_arm,
                    benchmark_dir=benchmark_dir,
                    run_id=evaluation_run_id,
                    max_steps=args.max_steps,
                    camera_policy=args.camera_policy,
                    initial_camera_view=args.initial_camera_view,
                    experiment_mode=experiment_mode,
                    visual_width=args.visual_width,
                    visual_height=args.visual_height,
                    model_debug_overlay=args.model_debug_overlay,
                    record_overlay=args.record_overlay,
                    request_timeout=args.request_timeout,
                    expert_learning_file=expert_learning_file,
                    spatial_tool=args.spatial_tool,
                    spatial_close_audit=args.spatial_close_audit,
                    spatial_candidate_proposals=args.spatial_candidate_proposals,
                    spatial_checkpoint=args.spatial_checkpoint,
                    spatial_calibration=args.spatial_calibration,
                    spatial_view_ranker=args.spatial_view_ranker,
                    spatial_outcome_checkpoint=args.spatial_outcome_checkpoint,
                    spatial_candidate_ranker=args.spatial_candidate_ranker,
                    spatial_semantic_part_checkpoint=args.spatial_semantic_part_checkpoint,
                    spatial_direct_grasp_frame_checkpoint=args.spatial_direct_grasp_frame_checkpoint,
                    spatial_intent_embedding_store=args.spatial_intent_embedding_store,
                    spatial_semantic_view_ranker=args.spatial_semantic_view_ranker,
                    spatial_semantic_view_mode=args.spatial_semantic_view_mode,
                    spatial_dynamic_intent_encoder_python=args.spatial_dynamic_intent_encoder_python,
                    spatial_dynamic_intent_encoder_script=args.spatial_dynamic_intent_encoder_script,
                    spatial_dynamic_intent_encoder_snapshot=args.spatial_dynamic_intent_encoder_snapshot,
                    spatial_max_additional_views=args.spatial_max_additional_views,
                    spatial_required_confidence=args.spatial_required_confidence,
                    spatial_max_grasp_candidates=args.spatial_max_grasp_candidates,
                    spatial_pen_radial_half_extent_m=args.spatial_pen_radial_half_extent_m,
                )
                log_file = log_path.open("w", encoding="utf-8")
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
                processes[seed] = process
                log_files[seed] = log_file
                evaluations[str(seed)] = {
                    "seed": seed,
                    "run_id": evaluation_run_id,
                    "run_dir": str(run_dir),
                    "log": str(log_path),
                    "pid": process.pid,
                    "returncode": None,
                    "state": "running",
                    "success": None,
                }
                active_count += 1

            status["pending_seeds"] = list(pending)
            write_status(status_path, status)

            for seed, pid in list(external_processes.items()):
                if process_matches(pid, evaluation_run_id):
                    continue
                finish_orphaned_evaluation(evaluations[str(seed)])
                del external_processes[seed]

            for seed, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_files[seed].close()
                item = evaluations[str(seed)]
                run_dir = Path(item["run_dir"])
                item.update(evaluation_outcome(run_dir / "result.json", returncode))
                del processes[seed]
            if pending or processes or external_processes:
                time.sleep(args.poll_seconds)
    finally:
        for log_file in log_files.values():
            if not log_file.closed:
                log_file.close()

    evaluation_values = list(evaluations.values())
    status["state"] = "completed"
    status["supervisor_pid"] = None
    status["pending_seeds"] = []
    status["finished_at"] = now()
    status["summary"] = {
        "evaluation_count": len(evaluation_values),
        "process_completed": sum(
            item.get("returncode") == 0 for item in evaluation_values
        ),
        "process_failed": sum(
            item.get("returncode") != 0 for item in evaluation_values
        ),
        "infrastructure_failed": sum(
            item.get("state") == "infrastructure_error"
            for item in evaluation_values
        ),
        "environment_success": sum(
            item.get("success") is True for item in evaluation_values
        ),
        "environment_failure": sum(
            item.get("success") is False for item in evaluation_values
        ),
    }
    write_status(status_path, status)


def finish_orphaned_evaluation(evaluation: dict[str, Any]) -> None:
    run_dir = evaluation.get("run_dir")
    result_path = Path(run_dir) / "result.json" if isinstance(run_dir, str) else None
    returncode = 0 if result_path is not None and result_path.is_file() else -1
    evaluation.update(evaluation_outcome(result_path, returncode))


def process_matches(pid: int, marker: str) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return True
    return marker.encode() in command


def eval_command(
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str,
    benchmark_dir: Path,
    run_id: str,
    max_steps: int,
    experiment_mode: str,
    camera_policy: str = "fine",
    initial_camera_view: str = "center_high",
    fixed_views: list[str] | tuple[str, ...] = (),
    fixed_view_width: int = 384,
    fixed_view_height: int = 288,
    fixed_view_jpeg_quality: int = 60,
    visual_width: int = 1280,
    visual_height: int = 960,
    model_debug_overlay: str = "eepose",
    record_overlay: str = "eepose",
    request_timeout: float = 120.0,
    expert_demo_dir: Path | None = None,
    expert_learning_file: Path | None = None,
    learning_only: bool = False,
    spatial_tool: bool = False,
    spatial_close_audit: bool = False,
    spatial_candidate_proposals: bool = False,
    spatial_checkpoint: str | Path | None = None,
    spatial_calibration: str | Path | None = None,
    spatial_view_ranker: str | Path | None = None,
    spatial_outcome_checkpoint: str | Path | None = None,
    spatial_candidate_ranker: str | Path | None = None,
    spatial_semantic_part_checkpoint: str | Path | None = None,
    spatial_direct_grasp_frame_checkpoint: str | Path | None = None,
    spatial_intent_embedding_store: str | Path | None = None,
    spatial_semantic_view_ranker: str | Path | None = None,
    spatial_semantic_view_mode: str = "analytic",
    spatial_dynamic_intent_encoder_python: str | Path | None = None,
    spatial_dynamic_intent_encoder_script: str | Path | None = None,
    spatial_dynamic_intent_encoder_snapshot: str | Path | None = None,
    spatial_max_additional_views: int = 2,
    spatial_required_confidence: float = 0.75,
    spatial_max_grasp_candidates: int = 3,
    spatial_pen_radial_half_extent_m: float = 0.01,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "agent",
        "--timeout",
        str(request_timeout),
        "eval",
        "robotwin",
        "--task",
        task,
        "--config",
        config,
        "--seed",
        str(seed),
        "--active-arm",
        active_arm,
        "--benchmark-dir",
        str(benchmark_dir),
        "--max-steps",
        str(max_steps),
        "--run-id",
        run_id,
        "--model-view",
        "active",
        "--record-view",
        "gripper_follow",
        "--initial-camera-view",
        initial_camera_view,
        "--camera-policy",
        camera_policy,
        "--visual-width",
        str(visual_width),
        "--visual-height",
        str(visual_height),
        "--model-overlay",
        "none",
        "--model-debug-overlay",
        model_debug_overlay,
        "--record-overlay",
        record_overlay,
        "--quiet-events",
        "--experiment-mode",
        experiment_mode,
    ]
    # Streaming is enabled by default for long visual turns; --no-stream remains
    # available through AGENT_STREAM=false for providers that need it.
    stream_value = os.environ.get("AGENT_STREAM")
    stream_enabled = (
        DEFAULT_STREAM
        if stream_value is None
        else stream_value.strip().lower() in {"1", "true", "yes", "on"}
    )
    if not stream_enabled:
        command.insert(command.index("--experiment-mode"), "--no-stream")
    if fixed_views:
        command.extend(
            [
                "--fixed-views",
                *fixed_views,
                "--fixed-view-width",
                str(fixed_view_width),
                "--fixed-view-height",
                str(fixed_view_height),
                "--fixed-view-jpeg-quality",
                str(fixed_view_jpeg_quality),
            ]
        )
    if expert_demo_dir is not None:
        command.extend(["--expert-demo-dir", str(expert_demo_dir)])
    if expert_learning_file is not None:
        command.extend(["--expert-learning-file", str(expert_learning_file)])
    if learning_only:
        command.append("--learning-only")
    if spatial_tool:
        command.append("--spatial-tool")
    if spatial_close_audit:
        command.append("--spatial-close-audit")
    if spatial_candidate_proposals:
        command.append("--spatial-candidate-proposals")
    if spatial_tool or spatial_close_audit or spatial_candidate_proposals:
        spatial_values = (
            ("--spatial-checkpoint", spatial_checkpoint),
            ("--spatial-calibration", spatial_calibration),
            ("--spatial-view-ranker", spatial_view_ranker),
            ("--spatial-outcome-checkpoint", spatial_outcome_checkpoint),
            ("--spatial-candidate-ranker", spatial_candidate_ranker),
            (
                "--spatial-semantic-part-checkpoint",
                spatial_semantic_part_checkpoint,
            ),
            (
                "--spatial-direct-grasp-frame-checkpoint",
                spatial_direct_grasp_frame_checkpoint,
            ),
            ("--spatial-intent-embedding-store", spatial_intent_embedding_store),
            ("--spatial-semantic-view-ranker", spatial_semantic_view_ranker),
            (
                "--spatial-dynamic-intent-encoder-python",
                spatial_dynamic_intent_encoder_python,
            ),
            (
                "--spatial-dynamic-intent-encoder-script",
                spatial_dynamic_intent_encoder_script,
            ),
            (
                "--spatial-dynamic-intent-encoder-snapshot",
                spatial_dynamic_intent_encoder_snapshot,
            ),
        )
        for flag, value in spatial_values:
            if value is not None:
                command.extend([flag, str(value)])
        command.extend(
            [
                "--spatial-semantic-view-mode",
                str(spatial_semantic_view_mode),
                "--spatial-max-additional-views",
                str(spatial_max_additional_views),
                "--spatial-required-confidence",
                str(spatial_required_confidence),
                "--spatial-max-grasp-candidates",
                str(spatial_max_grasp_candidates),
                "--spatial-pen-radial-half-extent-m",
                str(spatial_pen_radial_half_extent_m),
            ]
        )
    return command


def child_environment(state_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["AGENT_STATE_DIR"] = str(state_dir)
    return env


def wait_for_process(
    process: subprocess.Popen[str],
    status_path: Path,
    status: dict[str, Any],
    poll_seconds: float,
) -> int:
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        write_status(status_path, status)
        time.sleep(poll_seconds)


def read_success(result_path: Path) -> bool | None:
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return bool(result.get("success"))


def evaluation_outcome(
    result_path: Path | None,
    returncode: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if result_path is not None and result_path.is_file():
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            result = value

    stop_reason = str(result.get("stop_reason") or "")
    success = result.get("success")
    if returncode != 0 or not isinstance(success, bool):
        state = "process_failed"
        success = None
    elif INFRASTRUCTURE_ERROR.search(stop_reason):
        state = "infrastructure_error"
        success = None
    else:
        state = "completed"

    return {
        "returncode": returncode,
        "state": state,
        "success": success,
        "stop_reason": stop_reason or None,
        "provider_routing": result.get("provider_routing"),
        "finished_at": now(),
    }


def resolve_experiment_mode(
    requested: str | None,
    *,
    expert_demo_dir: Path | None,
    expert_learning_file: Path | None,
) -> str:
    inferred = (
        MODEL_SPECIFIC_SUMMARY_MODE
        if expert_demo_dir is not None
        else SHARED_SUMMARY_MODE
    )
    resolved = requested or inferred
    if resolved == SHARED_SUMMARY_MODE and expert_learning_file is None:
        raise ValueError("shared-summary requires --expert-learning-file")
    if (
        resolved == MODEL_SPECIFIC_SUMMARY_MODE
        and expert_demo_dir is None
        and expert_learning_file is None
    ):
        raise ValueError(
            "model-specific-summary requires --expert-demo-dir or "
            "--expert-learning-file"
        )
    return resolved


def validate_leaf(value: str) -> str:
    candidate = value.strip()
    path = Path(candidate)
    if not candidate or path.is_absolute() or len(path.parts) != 1 or candidate in {".", ".."}:
        raise ValueError("batch-id must be a single directory name")
    return candidate


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return sanitized or "task"


def write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
