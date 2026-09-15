from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import ChatClient
from .config import DEFAULT_MAX_STEPS, WIRE_APIS, load
from .robotwin import run_robotwin_eval, run_robotwin_loop
from .robotwin.expert_learning import EXPERIMENT_MODES
from .robotwin.spatial_tool import SpatialToolConfig
from .session import Store


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load(args)
    store = Store(config.state_dir)
    if args.command == "config":
        print(json.dumps(config.redacted, indent=2, ensure_ascii=False))
        return
    client = ChatClient(config)
    if args.command == "eval" and args.eval_command == "robotwin":
        run_dir = run_robotwin_eval(
            client,
            store,
            config.state_dir,
            task=args.task,
            config=args.config_name,
            seed=args.seed,
            active_arm=args.active_arm,
            max_steps=args.max_steps,
            benchmark_dir=Path(args.benchmark_dir) if args.benchmark_dir else None,
            session_id=args.session,
            run_id=args.run_id,
            model_view=args.model_view,
            record_view=args.record_view,
            initial_camera_view=args.initial_camera_view,
            camera_policy=args.camera_policy,
            width=args.visual_width,
            height=args.visual_height,
            model_overlay=args.model_overlay,
            model_debug_overlay=args.model_debug_overlay,
            record_overlay=args.record_overlay,
            fixed_views=tuple(args.fixed_views or ()),
            fixed_view_width=args.fixed_view_width,
            fixed_view_height=args.fixed_view_height,
            fixed_view_jpeg_quality=args.fixed_view_jpeg_quality,
            expert_demo_dir=Path(args.expert_demo_dir) if args.expert_demo_dir else None,
            expert_learning_file=Path(args.expert_learning_file) if args.expert_learning_file else None,
            experiment_mode=args.experiment_mode,
            learned_constraint_file=Path(args.learned_constraint_file) if args.learned_constraint_file else None,
            local_constraint_profile=args.local_constraint_profile,
            display=not args.quiet_events,
            stream=config.stream and not args.no_stream,
            learning_only=args.learning_only,
            spatial_tool_config=spatial_tool_config_from_args(args),
        )
        print(f"run: {run_dir}")
        return
    if args.command == "loop" and args.loop_command == "robotwin":
        result = run_robotwin_loop(
            client,
            store,
            config.state_dir,
            task=args.task,
            config=args.config_name,
            seed=args.seed,
            active_arm=args.active_arm,
            max_steps=args.max_steps,
            max_rounds=args.max_rounds,
            correction=args.correction,
            benchmark_dir=Path(args.benchmark_dir) if args.benchmark_dir else None,
            session_id=args.session,
            model_view=args.model_view,
            record_view=args.record_view,
            initial_camera_view=args.initial_camera_view,
            camera_policy=args.camera_policy,
            width=args.visual_width,
            height=args.visual_height,
            model_overlay=args.model_overlay,
            model_debug_overlay=args.model_debug_overlay,
            record_overlay=args.record_overlay,
            expert_demo_dir=Path(args.expert_demo_dir) if args.expert_demo_dir else None,
            expert_learning_file=Path(args.expert_learning_file) if args.expert_learning_file else None,
            learned_constraint_file=Path(args.learned_constraint_file) if args.learned_constraint_file else None,
            manual_feedback_file=Path(args.manual_feedback_file) if args.manual_feedback_file else None,
            diagnosis_max_turns=args.diagnosis_max_turns,
            local_constraint_profile=args.local_constraint_profile,
            display=not args.quiet_events,
            stream=config.stream and not args.no_stream,
            spatial_tool_config=spatial_tool_config_from_args(args),
        )
        print(f"loop: {result.loop_dir}")
        print(f"status: {result.status}")
        if result.manual_feedback_template:
            print(f"manual_feedback: {result.manual_feedback_template}")
        return
    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--auth-token")
    parser.add_argument("--model")
    parser.add_argument("--wire-api", choices=sorted(WIRE_APIS))
    parser.add_argument("--anthropic-version")
    parser.add_argument("--state-dir")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--transient-http-retry-attempts", type=int)
    parser.add_argument("--transient-http-retry-backoff-seconds", type=float)
    parser.add_argument("--quota-fallback-base-url")
    parser.add_argument("--quota-fallback-model")
    parser.add_argument("--quota-fallback-api-key")
    parser.add_argument("--quota-fallback-cooldown-seconds", type=float)
    parser.add_argument("--request-image-max-width", type=int)
    parser.add_argument("--request-image-max-height", type=int)
    parser.add_argument("--request-image-jpeg-quality", type=int)
    parser.add_argument("--request-max-images", type=int)
    parser.add_argument("--context-window-tokens", type=int)
    parser.add_argument("--context-compaction-threshold", type=float)
    parser.add_argument("--context-compaction-keep-recent-turns", type=int)
    parser.add_argument("--context-compaction-keep-recent-images", type=int)
    parser.add_argument("--no-context-compaction", action="store_true")
    parser.add_argument("--response-language")
    parser.add_argument(
        "--anthropic-thinking",
        choices=("enabled", "adaptive", "disabled"),
    )
    parser.add_argument("--anthropic-thinking-budget-tokens", type=int)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("config")

    evalp = sub.add_parser("eval")
    evalsub = evalp.add_subparsers(dest="eval_command")
    robotp = evalsub.add_parser("robotwin")
    add_robotwin_eval_arguments(robotp)

    loopp = sub.add_parser("loop")
    loopsub = loopp.add_subparsers(dest="loop_command")
    loop_robotp = loopsub.add_parser("robotwin")
    add_robotwin_eval_arguments(loop_robotp, include_run_id=False)
    loop_robotp.add_argument("--max-rounds", type=int, default=2)
    loop_robotp.add_argument("--correction", choices=["auto", "manual", "none"], default="auto")
    loop_robotp.add_argument("--manual-feedback-file")
    loop_robotp.add_argument("--diagnosis-max-turns", type=int, default=18)
    return parser


def add_robotwin_eval_arguments(robotp: argparse.ArgumentParser, *, include_run_id: bool = True) -> None:
    robotp.add_argument("--task", default="grasp_single_bottle")
    robotp.add_argument("--config", dest="config_name", default="demo_clean")
    robotp.add_argument("--seed", type=int, default=0)
    robotp.add_argument("--active-arm", choices=["left", "right", "both"], default=None)
    robotp.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    robotp.add_argument("--benchmark-dir")
    robotp.add_argument("--session")
    if include_run_id:
        robotp.add_argument("--run-id")
        robotp.add_argument("--experiment-mode", choices=EXPERIMENT_MODES)
        robotp.add_argument(
            "--learning-only",
            action="store_true",
            help="Stop after expert learning is saved, before executing any environment action.",
        )
    view_choices = ["active", "topdown", "gripper_follow", "side", "front_side_45", "side_top_45", "oblique_45"]
    robotp.add_argument("--model-view", choices=view_choices, default="active")
    robotp.add_argument("--record-view", choices=view_choices, default="gripper_follow")
    robotp.add_argument(
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
    robotp.add_argument("--camera-policy", choices=["fixed", "fine", "full"], default="fine")
    robotp.add_argument("--visual-width", type=int, default=1280)
    robotp.add_argument("--visual-height", type=int, default=960)
    robotp.add_argument("--model-overlay", choices=["none", "eepose"], default="none")
    robotp.add_argument("--model-debug-overlay", choices=["none", "eepose"], default="eepose")
    robotp.add_argument("--record-overlay", choices=["none", "eepose"], default="eepose")
    robotp.add_argument(
        "--fixed-views",
        nargs="+",
        choices=["topdown", "side", "front_side_45", "side_top_45", "oblique_45"],
        default=[],
    )
    robotp.add_argument("--fixed-view-width", type=int, default=384)
    robotp.add_argument("--fixed-view-height", type=int, default=288)
    robotp.add_argument("--fixed-view-jpeg-quality", type=int, default=60)
    robotp.add_argument("--expert-demo-dir")
    robotp.add_argument("--expert-learning-file")
    robotp.add_argument("--learned-constraint-file")
    robotp.add_argument(
        "--spatial-tool",
        "--enable-spatial-tool",
        dest="spatial_tool",
        action="store_true",
        help="Expose spatial.verify_pregrasp to the model as an optional tool.",
    )
    robotp.add_argument(
        "--spatial-close-audit",
        action="store_true",
        help="Run a policy-invisible spatial audit before each gripper close.",
    )
    robotp.add_argument(
        "--spatial-candidate-proposals",
        action="store_true",
        help=(
            "Expose spatial.propose_grasp_candidates, using expert semantic intent "
            "and the learned multi-view sparse object graph."
        ),
    )
    robotp.add_argument("--spatial-checkpoint")
    robotp.add_argument("--spatial-calibration")
    robotp.add_argument("--spatial-view-ranker")
    robotp.add_argument("--spatial-outcome-checkpoint")
    robotp.add_argument(
        "--spatial-candidate-ranker",
        help="Optional Phase 11 grasp-candidate ranker JSON checkpoint.",
    )
    robotp.add_argument(
        "--spatial-candidate-ranker-mode",
        choices=("shadow", "gated"),
        default="shadow",
        help="Keep the candidate ranker as evidence or allow an explicitly passed checkpoint to control ordering.",
    )
    robotp.add_argument(
        "--spatial-semantic-part-checkpoint",
        help="Optional guarded RGB-D+text semantic part-region checkpoint.",
    )
    robotp.add_argument(
        "--spatial-direct-grasp-frame-checkpoint",
        help="Optional guarded RGB-D+text direct grasp-frame checkpoint.",
    )
    robotp.add_argument(
        "--spatial-intent-embedding-store",
        help="Frozen text embedding sidecar matching the semantic part checkpoint.",
    )
    robotp.add_argument(
        "--spatial-semantic-view-ranker",
        help="Optional Phase 11 open-vocabulary event-view ranker JSON checkpoint.",
    )
    robotp.add_argument(
        "--spatial-semantic-view-mode",
        choices=("analytic", "shadow"),
        default="analytic",
        help="Keep semantic next-view ranker analytic-only or log it in shadow mode.",
    )
    robotp.add_argument("--spatial-dynamic-intent-encoder-python")
    robotp.add_argument("--spatial-dynamic-intent-encoder-script")
    robotp.add_argument("--spatial-dynamic-intent-encoder-snapshot")
    robotp.add_argument(
        "--spatial-appearance-encoder-mode",
        choices=("color", "smolvlm_shadow"),
        default="color",
    )
    robotp.add_argument("--spatial-appearance-encoder-python")
    robotp.add_argument("--spatial-appearance-encoder-script")
    robotp.add_argument("--spatial-appearance-encoder-snapshot")
    robotp.add_argument("--spatial-appearance-projection-checkpoint")
    robotp.add_argument("--spatial-max-additional-views", type=int, default=2)
    robotp.add_argument("--spatial-required-confidence", type=float, default=0.75)
    robotp.add_argument("--spatial-max-grasp-candidates", type=int, default=3)
    robotp.add_argument("--spatial-pen-radial-half-extent-m", type=float, default=0.01)
    robotp.add_argument("--local-constraint-profile", choices=["generic"], default="generic", help=argparse.SUPPRESS)
    robotp.add_argument("--no-stream", action="store_true")
    robotp.add_argument("--quiet-events", action="store_true")


def spatial_tool_config_from_args(args: argparse.Namespace) -> SpatialToolConfig:
    return SpatialToolConfig(
        enabled=bool(args.spatial_tool),
        close_audit=bool(args.spatial_close_audit),
        candidate_proposals=bool(args.spatial_candidate_proposals),
        checkpoint=Path(args.spatial_checkpoint) if args.spatial_checkpoint else None,
        calibration=Path(args.spatial_calibration) if args.spatial_calibration else None,
        view_ranker_checkpoint=(
            Path(args.spatial_view_ranker) if args.spatial_view_ranker else None
        ),
        outcome_checkpoint=(
            Path(args.spatial_outcome_checkpoint)
            if args.spatial_outcome_checkpoint
            else None
        ),
        candidate_ranker_checkpoint=(
            Path(args.spatial_candidate_ranker)
            if args.spatial_candidate_ranker
            else None
        ),
        candidate_ranker_mode=args.spatial_candidate_ranker_mode,
        semantic_part_checkpoint=(
            Path(args.spatial_semantic_part_checkpoint)
            if args.spatial_semantic_part_checkpoint
            else None
        ),
        direct_grasp_frame_checkpoint=(
            Path(args.spatial_direct_grasp_frame_checkpoint)
            if args.spatial_direct_grasp_frame_checkpoint
            else None
        ),
        intent_embedding_store=(
            Path(args.spatial_intent_embedding_store)
            if args.spatial_intent_embedding_store
            else None
        ),
        semantic_view_ranker_checkpoint=(
            Path(args.spatial_semantic_view_ranker)
            if args.spatial_semantic_view_ranker
            else None
        ),
        semantic_view_mode=args.spatial_semantic_view_mode,
        dynamic_intent_encoder_python=(
            Path(args.spatial_dynamic_intent_encoder_python)
            if args.spatial_dynamic_intent_encoder_python
            else None
        ),
        dynamic_intent_encoder_script=(
            Path(args.spatial_dynamic_intent_encoder_script)
            if args.spatial_dynamic_intent_encoder_script
            else None
        ),
        dynamic_intent_encoder_snapshot=(
            Path(args.spatial_dynamic_intent_encoder_snapshot)
            if args.spatial_dynamic_intent_encoder_snapshot
            else None
        ),
        appearance_encoder_mode=args.spatial_appearance_encoder_mode,
        appearance_encoder_python=(
            Path(args.spatial_appearance_encoder_python)
            if args.spatial_appearance_encoder_python
            else None
        ),
        appearance_encoder_script=(
            Path(args.spatial_appearance_encoder_script)
            if args.spatial_appearance_encoder_script
            else None
        ),
        appearance_encoder_snapshot=(
            Path(args.spatial_appearance_encoder_snapshot)
            if args.spatial_appearance_encoder_snapshot
            else None
        ),
        appearance_projection_checkpoint=(
            Path(args.spatial_appearance_projection_checkpoint)
            if args.spatial_appearance_projection_checkpoint
            else None
        ),
        max_additional_views=args.spatial_max_additional_views,
        required_confidence=args.spatial_required_confidence,
        max_grasp_candidates=args.spatial_max_grasp_candidates,
        pen_radial_half_extent_m=args.spatial_pen_radial_half_extent_m,
    )


if __name__ == "__main__":
    main()
