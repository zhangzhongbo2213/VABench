from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from agent.client import ChatClient
from agent.events import RunLogger
from agent.harness import AgentHarness, ToolResult
from agent.protocol import Decision, decision_env_action, decision_label, parse_decision
from agent.session import Part, Store
from agent.thread import AgentThread, ensure_session

from .adapter import FrameRecord, RoboTwinAdapter, frame_to_json
from .expert import ExpertStore
from .expert_learning import (
    ExpertLearningState,
    expert_learning_gate_error,
    validate_experiment_lineage,
    validate_expert_learning_args,
)
from .learned_constraints import load_prompt_text
from .prompts import system_prompt
from .spatial_tool import (
    EXECUTE_CANDIDATE_TOOL_NAME,
    GRASP_CANDIDATE_TOOL_NAME,
    SPATIAL_TOOL_NAME,
    VERIFY_CANDIDATE_TOOL_NAME,
    SpatialPregraspRuntime,
    SpatialToolConfig,
    candidate_executability_model_payload,
    grasp_outcome_model_payload,
    grasp_candidate_model_payload,
    is_gripper_close_action,
    model_payload as spatial_model_payload,
    spatial_tool_system_prompt,
)
from .validator import validate_decision


def run_robotwin_eval(
    client: ChatClient,
    store: Store,
    state_dir: Path,
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str | None,
    max_steps: int,
    benchmark_dir: Path | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    model_view: str = "active",
    record_view: str = "gripper_follow",
    initial_camera_view: str = "center_high",
    camera_policy: str = "fine",
    width: int = 1280,
    height: int = 960,
    model_overlay: str = "none",
    model_debug_overlay: str = "eepose",
    record_overlay: str = "eepose",
    fixed_views: tuple[str, ...] = (),
    fixed_view_width: int = 384,
    fixed_view_height: int = 288,
    fixed_view_jpeg_quality: int = 60,
    expert_demo_dir: Path | None = None,
    expert_learning_file: Path | None = None,
    experiment_mode: str | None = None,
    learned_constraint_file: Path | None = None,
    local_constraint_profile: str = "generic",
    display: bool = True,
    stream: bool = True,
    learning_only: bool = False,
    spatial_tool_config: SpatialToolConfig | None = None,
) -> Path:
    spatial_config = spatial_tool_config or SpatialToolConfig()
    spatial_config.validate(task=task)
    resolved_run_id = validate_run_id(run_id) if run_id is not None else default_robotwin_run_id(
        state_dir, task=task, seed=seed
    )
    logger = RunLogger(
        state_dir,
        resolved_run_id,
        relative_parent=robotwin_run_parent(task=task, seed=seed),
        display=display,
    )
    client.begin_run()
    provider_fallback_steps: list[int] = []
    provider_step = {"value": 0}

    def log_provider_fallback(data: dict[str, object]) -> None:
        step = provider_step["value"]
        provider_fallback_steps.append(step)
        logger.event("provider_fallback", {**data, "step": step})

    client.set_route_event_handler(log_provider_fallback)
    resolved_expert_demo_dir = expert_demo_dir or default_expert_demo_dir(state_dir, task=task, active_arm=active_arm)
    expert_store = ExpertStore(resolved_expert_demo_dir) if resolved_expert_demo_dir else None
    if learning_only and expert_store is None:
        raise ValueError("learning_only requires an expert demo directory")
    if expert_store:
        logger.event("expert_store_loaded", expert_store.summary())
    learned_constraints_text = load_prompt_text(learned_constraint_file, task=task)
    if learned_constraint_file is not None:
        logger.event(
            "learned_constraints_loaded",
            {"source": str(learned_constraint_file), "active": bool(learned_constraints_text.strip())},
        )
    expert_learning = ExpertLearningState(logger.dir / "expert_learning.json") if expert_store or expert_learning_file else None
    if expert_learning_file is not None:
        if expert_learning is None:
            raise ValueError("expert_learning_file requires expert learning state")
        expert_learning.load_from(expert_learning_file)
        if experiment_mode is not None:
            validate_experiment_lineage(
                experiment_mode,
                summary_model=expert_learning.summary_model,
                evaluation_model=client.config.model,
            )
        logger.event(
            "expert_learning_preloaded",
            {"source": str(expert_learning_file), "path": str(expert_learning.path.relative_to(logger.dir)), "summary": expert_learning.summary or {}},
        )
    adapter = RoboTwinAdapter(
        logger.dir,
        benchmark_dir=benchmark_dir,
        task=task,
        config=config,
        seed=seed,
        active_arm=active_arm,
        max_steps=max_steps,
        model_view=model_view,
        record_view=record_view,
        initial_camera_view=initial_camera_view,
        camera_policy=camera_policy,
        width=width,
        height=height,
        model_overlay=model_overlay,
        model_debug_overlay=model_debug_overlay,
        record_overlay=record_overlay,
        fixed_views=fixed_views,
        fixed_view_width=fixed_view_width,
        fixed_view_height=fixed_view_height,
        fixed_view_jpeg_quality=fixed_view_jpeg_quality,
    )
    spatial_runtime = (
        SpatialPregraspRuntime(
            spatial_config,
            benchmark_dir=adapter.benchmark_dir,
            run_dir=logger.dir,
            task=task,
        )
        if spatial_config.active
        else None
    )
    last_result: dict[str, Any] | None = None
    stop_reason: str | None = None
    sid = ""
    try:
        current = adapter.reset()
        provider_step["value"] = current.step
        system = system_prompt(
            adapter.action_space_description(),
            task=task,
            model_view=model_view,
            model_overlay=model_overlay,
            record_view=record_view,
            record_overlay=record_overlay,
            camera_policy=camera_policy,
            response_language=client.config.response_language,
        )
        system += spatial_tool_system_prompt(spatial_config)
        sid = ensure_session(store, session_id, system, f"robotwin {task} seed {seed}")
        thread = AgentThread(client, store, sid, logger, stream=stream)
        logger.event(
            "eval_start",
            {
                "type": "robotwin",
                "task": task,
                "seed": seed,
                "run_id": logger.run_id,
                "session": sid,
                "expert_demo_dir": str(resolved_expert_demo_dir) if resolved_expert_demo_dir else None,
                "expert_learning_file": str(expert_learning_file) if expert_learning_file else None,
                "experiment_mode": experiment_mode,
                "summary_model": expert_learning.summary_model if expert_learning else None,
                "evaluation_model": client.config.model,
                "provider_routing": client.routing_metadata(),
                "max_tokens": client.config.max_tokens,
                "response_language": client.config.response_language,
                "request_image_transport": {
                    "max_width": client.config.request_image_max_width,
                    "max_height": client.config.request_image_max_height,
                    "jpeg_quality": client.config.request_image_jpeg_quality,
                    "max_images": client.config.request_max_images,
                    "enabled": bool(
                        client.config.request_image_max_width
                        or client.config.request_image_max_height
                    ),
                },
                "context_compaction": thread.context.metadata(),
                "learned_constraint_file": str(learned_constraint_file) if learned_constraint_file else None,
                "model_view": model_view,
                "model_overlay": model_overlay,
                "model_debug_overlay": model_debug_overlay,
                "record_view": record_view,
                "record_overlay": record_overlay,
                "initial_camera_view": initial_camera_view,
                "camera_policy": camera_policy,
                "fixed_views": list(fixed_views),
                "fixed_view_transport": {
                    "width": fixed_view_width,
                    "height": fixed_view_height,
                    "jpeg_quality": fixed_view_jpeg_quality,
                },
                "local_constraint_profile": local_constraint_profile,
                "learning_only": learning_only,
                "spatial_tool": spatial_config.metadata(),
            },
        )
        done = False
        while not done and current.step < max_steps:
            provider_step["value"] = current.step
            decision = ask_for_decision(
                thread,
                logger,
                adapter,
                current,
                expert_store,
                expert_learning,
                local_constraint_profile=local_constraint_profile,
                learned_constraints_text=learned_constraints_text,
                learning_only=learning_only,
                spatial_runtime=(
                    spatial_runtime
                    if spatial_config.enabled or spatial_config.candidate_proposals
                    else None
                ),
            )
            if learning_only and expert_learning and expert_learning.learned:
                stop_reason = "learning_only_completed"
                logger.event(
                    "learning_only_completed",
                    {
                        "step": current.step,
                        "expert_learning": str(expert_learning.path.relative_to(logger.dir)),
                    },
                )
                break
            if decision.kind == "candidate_execution":
                if spatial_runtime is None:
                    stop_reason = "candidate execution requested while spatial runtime is disabled"
                    logger.event("eval_error", {"error": stop_reason})
                    break
                candidate_id = str((decision.tool_args or {}).get("candidate_id", ""))
                label = f"{EXECUTE_CANDIDATE_TOOL_NAME}.{candidate_id}"
                logger.event(
                    "action",
                    {
                        "step": current.step,
                        "action": label,
                        "action_payload": {
                            "target": "grasp_candidate",
                            "type": "execute_authorized",
                            "candidate_id": candidate_id,
                        },
                        "reason": decision.reason,
                        "pre_action_geometry": current.geometry,
                    },
                )
                try:
                    last_result = spatial_runtime.execute_grasp_candidate(
                        adapter,
                        tool_args=decision.tool_args,
                    )
                except Exception as exc:
                    stop_reason = f"candidate execution failed before completion: {exc}"
                    logger.event(
                        "candidate_execution_error",
                        {
                            "step": current.step,
                            "candidate_id": candidate_id,
                            "error": str(exc),
                        },
                    )
                    break
                current = adapter.record_external_result(last_result, label=label)
                info = last_result["info"]
                done = bool(last_result["done"])
                grasp_verification = last_result["candidate_execution"].get(
                    "grasp_verification"
                )
                logger.event(
                    "candidate_execution_result",
                    {
                        "step": info.get("step_count"),
                        "candidate_id": candidate_id,
                        "planner_success": info.get("planner_success"),
                        "task_success": info.get("success"),
                        "done": done,
                        "phases": last_result["candidate_execution"]["phases"],
                        "model_image": str(current.path.relative_to(logger.dir)),
                    },
                )
                if isinstance(grasp_verification, dict):
                    logger.event(
                        "grasp_outcome_verification",
                        {
                            "step": info.get("step_count"),
                            **grasp_outcome_model_payload(grasp_verification),
                        },
                    )
                logger.event(
                    "env_step",
                    {
                        "step": info.get("step_count"),
                        "done": done,
                        "success": info.get("success"),
                        "planner_success": info.get("planner_success"),
                        "action_valid": info.get("action_valid"),
                        "last_error": info.get("last_error"),
                        "model_image": str(current.path.relative_to(logger.dir)),
                        "model_overlay": current.model_overlay,
                        "model_debug_image": (
                            str(current.model_debug_path.relative_to(logger.dir))
                            if current.model_debug_path
                            else None
                        ),
                        "model_debug_overlay": current.model_debug_overlay,
                        "record_image": (
                            str(current.record_path.relative_to(logger.dir))
                            if current.record_path
                            else None
                        ),
                        "record_overlay": current.record_overlay,
                        "geometry": current.geometry,
                        "atomic_candidate_execution": True,
                        "candidate_id": candidate_id,
                    },
                )
                continue
            if decision.kind == "stop":
                stop_reason = decision.reason or "model stopped"
                logger.event("eval_stopped", {"step": current.step, "reason": stop_reason})
                break
            if not decision.action:
                stop_reason = decision.reason or "no action"
                logger.event("eval_error", {"error": stop_reason})
                break
            payload = decision_env_action(decision)
            label = decision_label(decision)
            if (
                spatial_runtime is not None
                and spatial_config.close_audit
                and is_gripper_close_action(decision.action)
            ):
                try:
                    audit = spatial_runtime.query(
                        adapter,
                        trigger="close_shadow",
                        step=current.step,
                    )
                    audit_final = audit["result"]
                    logger.event(
                        "spatial_close_shadow_result",
                        {
                            "step": current.step,
                            "proposed_action": label,
                            "policy_visible": False,
                            "query_id": audit["query_id"],
                            "query_dir": audit["query_dir"],
                            "verdict": audit_final.get("verdict"),
                            "confidence": audit_final.get("confidence"),
                            "relations": audit_final.get("relations"),
                            "missing_evidence": audit_final.get("missing_evidence"),
                            "observed_view_sequence": audit["observed_view_sequence"],
                            "world_frozen": audit["world_frozen"],
                            "world_fingerprint_delta": audit["world_fingerprint_delta"],
                            "action_will_execute_unchanged": True,
                        },
                    )
                except Exception as exc:
                    logger.event(
                        "spatial_close_shadow_error",
                        {
                            "step": current.step,
                            "proposed_action": label,
                            "policy_visible": False,
                            "error": str(exc),
                            "action_will_execute_unchanged": True,
                        },
                    )
            logger.event(
                "action",
                {
                    "step": current.step,
                    "action": label,
                    "action_payload": payload,
                    "reason": decision.reason,
                    "pre_action_geometry": current.geometry,
                },
            )
            current, last_result = adapter.step(payload, label=label)
            info = last_result["info"]
            done = bool(last_result["done"])
            logger.event(
                "env_step",
                {
                    "step": info.get("step_count"),
                    "done": done,
                    "success": info.get("success"),
                    "planner_success": info.get("planner_success"),
                    "action_valid": info.get("action_valid"),
                    "last_error": info.get("last_error"),
                    "model_image": str(current.path.relative_to(logger.dir)),
                    "model_overlay": current.model_overlay,
                    "model_debug_image": str(current.model_debug_path.relative_to(logger.dir))
                    if current.model_debug_path
                    else None,
                    "model_debug_overlay": current.model_debug_overlay,
                    "record_image": str(current.record_path.relative_to(logger.dir)) if current.record_path else None,
                    "record_overlay": current.record_overlay,
                    "geometry": current.geometry,
                },
            )

        for label, path in adapter.write_videos().items():
            logger.artifact(label, path)
        result = {
            "task": task,
            "seed": seed,
            "run_id": logger.run_id,
            "session": sid,
            "expert_demo_dir": str(resolved_expert_demo_dir) if resolved_expert_demo_dir else None,
            "expert_learning": str(expert_learning.path.relative_to(logger.dir)) if expert_learning and expert_learning.learned else None,
            "expert_learning_file": str(expert_learning_file) if expert_learning_file else None,
            "experiment_mode": experiment_mode,
            "summary_model": expert_learning.summary_model if expert_learning else None,
            "evaluation_model": client.config.model,
            "provider_routing": {
                **client.routing_metadata(),
                "fallback_steps": provider_fallback_steps,
                "first_fallback_step": (
                    provider_fallback_steps[0] if provider_fallback_steps else None
                ),
            },
            "max_tokens": client.config.max_tokens,
            "request_image_transport": {
                "max_width": client.config.request_image_max_width,
                "max_height": client.config.request_image_max_height,
                "jpeg_quality": client.config.request_image_jpeg_quality,
                "max_images": client.config.request_max_images,
                "enabled": bool(
                    client.config.request_image_max_width
                    or client.config.request_image_max_height
                ),
            },
            "context_compaction": thread.context.metadata(),
            "learning_only": learning_only,
            "learned_constraint_file": str(learned_constraint_file) if learned_constraint_file else None,
            "success": bool(last_result and last_result["info"].get("success")),
            "stop_reason": stop_reason,
            "steps": [frame_to_json(record, logger.dir) for record in adapter.records],
            "last_info": last_result["info"] if last_result else None,
            "spatial_tool": (
                spatial_runtime.summary()
                if spatial_runtime is not None
                else spatial_config.metadata()
            ),
        }
        result_path = logger.dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.artifact("result", result_path)
        logger.event("eval_finish", {"result": str(result_path), "success": result["success"]})
        return logger.dir
    finally:
        client.set_route_event_handler(None)
        adapter.close()


def default_robotwin_run_id(state_dir: Path, *, task: str, seed: int) -> str:
    seed_dir = state_dir / "runs" / robotwin_run_parent(task=task, seed=seed)
    prefix = "test"
    test_number = next_test_number(seed_dir, prefix)
    return f"{prefix}{test_number:03d}_{uuid4().hex[:8]}"


def robotwin_run_parent(*, task: str, seed: int) -> Path:
    return Path(sanitize_run_component(task)) / f"seed_{seed}"


def validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError("run_id must be a single directory name, not a path")
    return value


def next_test_number(runs_dir: Path, prefix: str) -> int:
    if not runs_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_")
    highest = 0
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def sanitize_run_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "task"


def default_expert_demo_dir(state_dir: Path, *, task: str, active_arm: str | None) -> Path | None:
    dual_arm_tasks = {
        "place_shoe",
        "handover_mic",
        "handover_horizontal_block",
        "handover_block",
        "handover_cube_to_target",
        "lift_pot",
    }
    arm = active_arm or ("both" if task.lower() in dual_arm_tasks else "right")
    candidates = []
    normalized_task = task.lower()
    if normalized_task == "put_everything_in_basket":
        composite = ensure_put_everything_composite_experts(state_dir)
        if composite is not None:
            return composite
    expert_source_task = {
        "grasp_single_bottle_generalization": "grasp_single_bottle",
        "grasp_single_bottle_upright_generalization": "grasp_single_bottle_upright",
        "place_single_bottle_upright_generalization": "place_single_bottle_upright",
        "grasp_single_cube_generalization": "grasp_single_cube",
        "place_single_cube_generalization": "place_single_cube",
        "place_cube_in_bowl_generalization": "place_cube_in_bowl",
        "place_cube_on_cube_generalization": "place_cube_on_cube",
    }.get(normalized_task, normalized_task)
    if expert_source_task == "grasp_single_bottle_upright":
        candidates.append(f"grasp_single_bottle_upright_{arm}_eepose_clean")
    elif expert_source_task == "grasp_single_bottle":
        candidates.append(f"grasp_single_bottle_{arm}_eepose_clean")
    if expert_source_task != normalized_task:
        candidates.append(f"{expert_source_task}_{arm}_eepose_clean")
    candidates.append(f"{task}_{arm}_eepose_clean")
    data_root = state_dir / "runs" / "data" / "expert_demos"
    for name in candidates:
        path = data_root / name
        if path.exists():
            return path
    return None


def ensure_put_everything_composite_experts(state_dir: Path) -> Path | None:
    data_root = state_dir / "runs" / "data"
    source_names = (
        "place_single_cube_seed1",
        "grasp_single_bottle_seed0",
        "grasp_single_bottle_upright_seed0",
        "grasp_single_pen_seed1",
    )
    source_root = data_root / "qpos_video_demos"
    sources = [source_root / name for name in source_names]
    if not all(path.exists() for path in sources):
        return None

    composite_root = data_root / "composite_expert_demos" / "put_everything_in_basket"
    composite_root.mkdir(parents=True, exist_ok=True)
    manifest = composite_root / "composite_sources.json"
    payload = {
        "format": "agent-composite-expert-v1",
        "task": "put_everything_in_basket",
        "description": (
            "Atomic RGB-only skills for compositional transfer. "
            "No full put-everything trajectory is included."
        ),
        "sources": [
            {
                "task": name.rsplit("_seed", 1)[0],
                "path": f"../../qpos_video_demos/{name}",
            }
            for name in source_names
        ],
    }
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != payload:
        manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    return composite_root


def expert_learning_instructions(store: ExpertStore) -> str:
    metadata = json.dumps(store.summary(), indent=2, ensure_ascii=False)
    common = (
        "\nExpert demos are configured for this run. Before any environment action, complete the expert-learning gate. "
        "First retrieve the available eepose trajectories with "
        '{"tool":"expert.retrieve","args":{"mode":"trajectory"},"reason":"study all available expert trajectories"}. '
        "Do not assume a seed or semantic phase that is absent from the metadata.\n"
        "Available expert metadata:\n"
        + metadata
        + "\n"
    )
    if store.observation_only:
        if store.video_only:
            if store.composite:
                common += (
                    "This is a composite set of independent RGB-only atomic-skill videos, not a demonstration of the current "
                    "five-object task. It contains no eepose, qpos, gripper values, action labels, object coordinates, contact "
                    "state, or semantic phase labels. Study every listed demo separately. For each demo, autonomously inspect "
                    "at least 3 distinct frames spanning its early, grasp/close, and completion portions. Select a demo explicitly "
                    'with {"demo":"DEMO_ID","steps":[K1,K2,...],"view":"..."}; up to 8 frames may be requested at once. '
                    "Transfer the demonstrated cube placement and bottle/pen grasp skills compositionally, but choose the current "
                    "object order, poses, motions, and basket drop locations from current visual evidence. In expert.learn, add an "
                    "object_strategies mapping that separately records cube, horizontal bottle, upright bottle, and pen experience, "
                    "plus a release_or_completion_rule for completing and counting all five placements. "
                )
            else:
                common += (
                    "This expert source is an RGB-only, uniformly thinned video. It contains no eepose, qpos, gripper values, "
                    "action labels, object coordinates, contact state, or semantic phase labels. Derive expert experience only "
                    "from the returned video frames. After trajectory retrieval, autonomously select frames spanning the early, "
                    "middle, grasp/close, and lifted portions of the sequence. Inspect frames with expert.frame using either "
                    '{"seed":N,"step":K,"view":"..."} or up to 8 frames at once with '
                    '{"seed":N,"steps":[K1,K2,...],"view":"..."}. At least 3 distinct video frames must be inspected before expert.learn. '
                )
        else:
            common += (
                "These are observation-only trajectories: they contain clean RGB frames plus observed eepose and gripper state, "
                "with no action labels, target eepose, object coordinates, or semantic grasp phases. After trajectory retrieval, "
                "study the complete uniformly sampled sequence and autonomously choose which returned seed/step images need visual "
                "inspection. Use expert.frame whenever you judge image evidence necessary; pass seed and step as integers and view "
                "as one of the metadata views. The harness does not select key frames or prescribe how many frames to inspect. "
                "Do not request final_grasp or close unless that phase is explicitly listed in metadata. "
            )
    else:
        common += "You may request a listed semantic phase/view frame when it helps ground the grasp position. "
    return common + (
        "Only after the required evidence is inspected, save structured task experience with "
        '{"tool":"expert.learn","args":{"grasp_object_part":"...","grasp_region":"...","grasp_height_or_depth":"...","finger_placement":"...","approach_strategy":"...","posture_or_rotation":"...","pre_close_checks":["..."],"test_lift_rule":"...","task_stage":"initial_grasp|handover_grasp|tool_grasp|...","post_grasp_goal":"...","functional_constraints":["preserve task-relevant function"],"required_arms":1,"semantic_ambiguity":0.0,"avoid_roles":["optional rejected functional parts"],"affordance_profile":{"part_shape":"unknown|slender_cylinder|broad_cylinder|box|handle|flat|irregular","symmetry_class":"unknown|continuous|two_fold|four_fold|asymmetric","centering_tolerance":"unknown|strict|moderate|permissive","vertical_tolerance":"unknown|strict|moderate|permissive","avoid_ends":true,"requires_bilateral_contact":true,"requires_dual_arm":false},"arm_assignment":"optional role evidence","coordination_sequence":["optional ordered multi-arm phases"],"synchronized_actions":["optional simultaneous actions"],"release_or_completion_rule":"optional transfer/place completion evidence","safety_notes":["..."]},"reason":"summarize expert trajectory constraints"}. '
        "Explicitly identify the demonstrated grasp position, intended grasp part, rejected regions, approach, posture, finger placement, task stage, post-grasp goal, and functional constraints. Do not invent metric coordinates. For dual-arm demonstrations, also save visible arm-role assignment, ordered coordination/transfer phases, actions that must be synchronized, and the release/completion rule in the optional fields. "
        "Use test_lift_rule as the legacy field name for a direct post-close completion lift, not a preliminary 20-30 mm test. "
        "Preserve non-conflicting posture, camera-check, depth, safety, and completion-lift constraints."
    )


def expert_trajectory_followup(store: ExpertStore) -> str:
    if store.video_only:
        if store.composite:
            return (
                "\nThis did not execute an environment action. This is the complete ordered frame index for every independent "
                "RGB-only atomic demo. No eepose, qpos, action, gripper value, object coordinate, or semantic phase is available. "
                "Use the returned demo IDs to inspect at least 3 actual frames from each demo before learning. Infer transferable "
                "per-object grasp/place skills; do not infer that the videos show a single combined trajectory or fixed object order."
            )
        return (
            "\nThis did not execute an environment action. This is the complete ordered frame index of an RGB-only expert "
            "video. No eepose, qpos, action, gripper value, or semantic phase is available. Select actual returned steps and "
            "inspect at least 3 distinct frames with expert.frame before learning. Infer grasp location, posture, insertion, "
            "closure, lift, and any visible multi-arm role/coordination sequence only from changes across the video."
        )
    if store.observation_only:
        return (
            "\nThis did not execute an environment action. This is the complete uniformly sampled observation sequence; no "
            "semantic phase or key frame has been selected for you. Autonomously choose actual returned seed/step values and "
            "use expert.frame whenever image evidence is needed. Ground the object part, grasp region, height/depth, and physical "
            "finger placement in the sequence without expecting additional local hints."
        )
    return (
        "\nThis did not execute an environment action. Before any environment action, call expert.learn with a structured "
        "summary derived from the demonstrated final grasp/close waypoints: object part, body region, height/depth, finger "
        "placement, approach, posture, pre-close checks, and direct completion lift. Request a listed clean phase/view frame "
        "if the grasp position is visually unclear."
    )


def ask_for_decision(
    thread: AgentThread,
    logger: RunLogger,
    adapter: RoboTwinAdapter,
    current: FrameRecord,
    expert_store: ExpertStore | None,
    expert_learning: ExpertLearningState | None,
    local_constraint_profile: str = "generic",
    learned_constraints_text: str = "",
    learning_only: bool = False,
    spatial_runtime: SpatialPregraspRuntime | None = None,
) -> Decision:
    isolated_composite_learning = bool(
        learning_only and expert_store is not None and expert_store.composite
    )
    prompt_base = (
        "Expert-learning-only phase for a compositional task. No evaluation "
        "episode inventory, appearance, pose, object order, first target, or "
        "current-scene visual observation is provided. Study every independent "
        "expert video and save only transferable per-object skills plus a generic "
        "completion rule for the five objects specified by each future episode."
        if isolated_composite_learning
        else current.prompt
    )
    if learned_constraints_text.strip():
        prompt_base += learned_constraints_text
    pending_grasp_outcome = None
    if spatial_runtime is not None:
        consume_outcome = getattr(
            spatial_runtime, "consume_pending_grasp_outcome", None
        )
        if callable(consume_outcome):
            pending_grasp_outcome = consume_outcome()
    if pending_grasp_outcome is not None:
        outcome_payload = grasp_outcome_model_payload(
            pending_grasp_outcome["result"]
        )
        prompt_base += (
            "\nPost-execution spatial.verify_grasp_outcome result:\n"
            + json.dumps(outcome_payload, indent=2, ensure_ascii=False)
            + "\nUse this RGB-D diagnostic result instead of inferring grasp success "
            "from gripper closure alone. The attached diagnostic views were captured "
            "after the lift."
        )
    if expert_store is not None and not (expert_learning and expert_learning.learned):
        prompt_base += expert_learning_instructions(expert_store)
    elif expert_learning is not None and expert_learning.learned:
        prompt_base += (
            "\nPrelearned expert constraints are already saved for this run. "
            "Use them as additive task experience before acting. You may use expert.retrieve or expert.frame for reference if the current image is ambiguous, but the expert-learning gate is already complete."
        )
    if expert_learning is not None:
        prompt_base += expert_learning.prompt_text()
    available = set(adapter.model_available_actions())

    def history_tool(decision: Decision) -> ToolResult:
        try:
            record = adapter.image_for_step((decision.tool_args or {}).get("step"))
        except ValueError as exc:
            return ToolResult([Part.text_part(prompt_base + f"\nHistorical image request failed: {exc}")], "tool_error", {"tool": decision.tool_name or "", "error": str(exc)})
        return ToolResult(
            [
                Part.text_part(prompt_base + "\nHistorical image attached. This did not execute an environment action."),
                Part.image_part(record.path, label=f"history step {record.step}"),
            ],
            "tool_result",
            {
                "tool": decision.tool_name or "",
                "served_step": record.step,
                "model_image": str(record.path.relative_to(adapter.run_dir)),
                "reason": decision.reason,
            },
        )

    def geometry_tool(decision: Decision) -> ToolResult:
        geometry = adapter.geometry() or {}
        return ToolResult(
            [
                Part.text_part(
                    prompt_base
                    + "\ngeometry.verify result:\n"
                    + json.dumps(geometry, indent=2, ensure_ascii=False)
                    + "\nThis did not execute an environment action."
                )
            ],
            "tool_result",
            {"tool": decision.tool_name or "", "geometry": geometry, "reason": decision.reason},
        )

    def spatial_tool(decision: Decision) -> ToolResult:
        if spatial_runtime is None:
            return ToolResult(
                [Part.text_part(prompt_base + "\nThe learned spatial tool is disabled for this run.")],
                "tool_error",
                {"tool": decision.tool_name or "", "error": "spatial tool disabled"},
            )
        tool_name = decision.tool_name or SPATIAL_TOOL_NAME
        try:
            if tool_name == GRASP_CANDIDATE_TOOL_NAME:
                record = spatial_runtime.propose_grasp_candidates(
                    adapter,
                    step=current.step,
                    tool_args=decision.tool_args,
                    expert_summary=(
                        expert_learning.summary if expert_learning is not None else None
                    ),
                )
                payload = grasp_candidate_model_payload(record)
            else:
                record = spatial_runtime.query(
                    adapter,
                    trigger="model",
                    step=current.step,
                    tool_args=decision.tool_args,
                )
                payload = spatial_model_payload(record)
        except Exception as exc:
            logger.event(
                "spatial_tool_error",
                {
                    "step": current.step,
                    "tool": decision.tool_name or "",
                    "reason": decision.reason,
                    "error": str(exc),
                },
            )
            direct_grounding_retry = bool(
                tool_name == GRASP_CANDIDATE_TOOL_NAME
                and spatial_runtime.config.direct_grasp_frame_checkpoint is not None
                and not any(
                    record.get("query_tool") == GRASP_CANDIDATE_TOOL_NAME
                    for record in spatial_runtime.records
                )
            )
            followup = (
                " The required initial candidate query is not complete. Re-inspect the "
                "current RGB, correct visual_grounding so the target box tightly covers the "
                "named object, and call spatial.propose_grasp_candidates again; no camera or "
                "robot action is allowed yet."
                if direct_grounding_retry
                else " Decide from the remaining evidence."
            )
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + f"\n{tool_name} failed: {exc}."
                        + followup
                    )
                ],
                "tool_error",
                {"tool": decision.tool_name or "", "error": str(exc)},
            )
        logger.event(
            "spatial_tool_result",
            {
                "step": current.step,
                "tool": decision.tool_name or "",
                "reason": decision.reason,
                "query_id": record["query_id"],
                "query_dir": record["query_dir"],
                "verdict": payload["verdict"],
                "confidence": payload["confidence"],
                "relations": payload.get("relations"),
                "missing_evidence": payload.get("missing_evidence"),
                "recommended_candidate_id": payload.get("recommended_candidate_id"),
                "view_assessment": payload.get("view_assessment"),
                "observed_view_sequence": record["observed_view_sequence"],
                "policy_visible": True,
                "world_frozen": record["world_frozen"],
                "world_fingerprint_delta": record["world_fingerprint_delta"],
            },
        )
        image_parts = [
            Part.image_part(
                Path(path),
                label=f"spatial diagnostic view {index + 1}: {view}",
            )
            for index, (path, view) in enumerate(
                zip(record["evidence_images"], record["observed_view_sequence"])
            )
        ]
        return ToolResult(
            [
                Part.text_part(
                    prompt_base
                    + f"\n{tool_name} result:\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False)
                    + "\nDiagnostic RGB views are attached in observation order. This did not "
                    "execute an environment action. Use the uncertainty and missing-evidence fields; "
                    "do not treat an uncertain result as success."
                ),
                *image_parts,
            ],
            "tool_result",
            {
                "tool": decision.tool_name or "",
                "query_id": record["query_id"],
                "query_dir": record["query_dir"],
                "verdict": payload["verdict"],
                "confidence": payload["confidence"],
                "observed_view_sequence": record["observed_view_sequence"],
                "reason": decision.reason,
            },
        )

    def candidate_authorization_tool(decision: Decision) -> ToolResult:
        if spatial_runtime is None:
            return ToolResult(
                [Part.text_part(prompt_base + "\nThe spatial runtime is disabled.")],
                "tool_error",
                {"tool": decision.tool_name or "", "error": "spatial runtime disabled"},
            )
        try:
            record = spatial_runtime.verify_candidate_executability(
                adapter,
                step=current.step,
                tool_args=decision.tool_args,
            )
            payload = candidate_executability_model_payload(record)
        except Exception as exc:
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + f"\n{VERIFY_CANDIDATE_TOOL_NAME} failed: {exc}. "
                        "Do not execute or manually approximate this candidate."
                    )
                ],
                "tool_error",
                {"tool": decision.tool_name or "", "error": str(exc)},
            )
        return ToolResult(
            [
                Part.text_part(
                    prompt_base
                    + f"\n{VERIFY_CANDIDATE_TOOL_NAME} result:\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False)
                    + "\nThis dry-run did not execute an environment action. If authorization "
                    "is true, call spatial.execute_grasp_candidate with the same candidate_id; "
                    "do not issue manual gripper motion."
                )
            ],
            "candidate_executability_result",
            {
                "tool": VERIFY_CANDIDATE_TOOL_NAME,
                "query_id": record["query_id"],
                "query_dir": record["query_dir"],
                "candidate_id": record["candidate_id"],
                "execution_authorized": payload["execution_authorized"],
                "failed_hard_constraints": payload["failed_hard_constraints"],
                "world_frozen": payload["world_frozen"],
                "reason": decision.reason,
            },
        )

    def expert_tool(decision: Decision) -> ToolResult:
        if expert_store is None:
            return ToolResult([Part.text_part(prompt_base + "\nNo expert store is configured. Output an action JSON.")])
        args = decision.tool_args or {}
        if (decision.tool_name or "") == "expert.learn":
            if expert_learning is None:
                return ToolResult([Part.text_part(prompt_base + "\nNo expert-learning state is configured. Output an action JSON.")])
            error = validate_expert_learning_args(
                args,
                trajectory_retrieved=expert_learning.trajectory_retrieved,
                observation_only=expert_store.observation_only,
                retrieved_frame_views=len(expert_learning.retrieved_frame_views),
                video_only=expert_store.video_only,
                retrieved_frame_count=len(expert_learning.retrieved_frames),
                required_video_demos=(
                    tuple(expert_store.demo_ids)
                    if expert_store.composite
                    else ()
                ),
                retrieved_frame_counts_by_demo=(
                    expert_learning.retrieved_frame_counts_by_demo
                ),
                forbid_episode_inventory_counts=(
                    adapter.task_name.lower() == "put_everything_in_basket"
                ),
            )
            if error:
                return ToolResult(
                    [Part.text_part(prompt_base + "\n" + error + "\nCorrect this by retrieving the trajectory first or filling all required learning fields.")],
                    "tool_error",
                    {"tool": "expert.learn", "error": error, "reason": decision.reason},
                )
            save_expert_learning_for_thread(expert_learning, args, thread)
            rel_path = expert_learning.path.relative_to(logger.dir)
            saved_summary = expert_learning.summary or {}
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + "\nExpert learning saved to "
                        + str(rel_path)
                        + ". Use these saved constraints for every later action in this run. Now output one action JSON or use a camera/geometry tool if evidence is still ambiguous.\n"
                        + json.dumps(saved_summary, indent=2, ensure_ascii=False)
                    )
                ],
                "expert_learning_saved",
                {
                    "tool": "expert.learn",
                    "path": str(rel_path),
                    "summary": saved_summary,
                    "summary_model": expert_learning.summary_model,
                    "reason": decision.reason,
                },
            )
        mode = str(args.get("mode", "")).lower()
        phase = str(args.get("phase", "")).lower()
        if mode == "trajectory" or phase in {"trajectory", "eepose_trajectory"}:
            try:
                trajectory = expert_store.trajectory_text(args)
            except ValueError as exc:
                return ToolResult(
                    [
                        Part.text_part(
                            prompt_base
                            + f"\nExpert trajectory lookup failed: {exc}. "
                            + "Use the available expert metadata below and retry expert.retrieve before attempting expert.learn:\n"
                            + json.dumps(expert_store.summary(), indent=2, ensure_ascii=False)
                        )
                    ],
                    "tool_error",
                    {"tool": decision.tool_name or "", "error": str(exc)},
                )
            if expert_learning is not None:
                expert_learning.trajectory_retrieved = True
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + ("\nExpert video frame sequence:\n" if expert_store.video_only else "\nExpert eepose trajectory:\n")
                        + trajectory
                        + expert_trajectory_followup(expert_store)
                    )
                ],
                "tool_result",
                {"tool": decision.tool_name or "", "mode": "trajectory", "reason": decision.reason},
            )
        try:
            frames = expert_store.find_many(args)
        except ValueError as exc:
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + f"\nExpert frame lookup failed: {exc}. "
                        + "For observation-only data, use an actual seed and step returned by expert.retrieve; do not invent a semantic phase. "
                        + "Available expert metadata:\n"
                        + json.dumps(expert_store.summary(), indent=2, ensure_ascii=False)
                    )
                ],
                "tool_error",
                {"tool": decision.tool_name or "", "error": str(exc)},
            )
        if adapter.model_overlay == "none" and any(not frame.clean_for_model for frame in frames):
            frame = frames[0]
            return ToolResult(
                [
                    Part.text_part(
                        prompt_base
                        + "\n"
                        + frame.caption
                        + "\nExpert frame image omitted because this run uses clean model images and the stored expert frame may contain debug overlays or labels. "
                        + "Use expert trajectory text for the demonstrated eepose sequence and derive the grasp-position constraint from final_grasp/close waypoints; use the current clean image for visual contact evidence."
                    )
                ],
                "tool_result",
                {
                    "tool": decision.tool_name or "",
                    "demo": frame.demo_id,
                    "seed": frame.seed,
                    "phase": frame.phase,
                    "view": frame.view,
                    "image_omitted": True,
                    "reason": decision.reason,
                },
            )
        if expert_learning is not None:
            for frame in frames:
                expert_learning.mark_frame_retrieved(
                    demo=frame.demo_id,
                    seed=frame.seed,
                    step=frame.step,
                    view=frame.view,
                )
        frame_parts = [
            Part.image_part(frame.path, label=f"expert {frame.demo_id} {frame.phase} {frame.view} step {frame.step}")
            for frame in frames
        ]
        return ToolResult(
            [
                Part.text_part(
                    prompt_base
                    + "\n"
                    + "\n".join(frame.caption for frame in frames)
                    + "\nExpert frame(s) attached. This did not execute an environment action. Compare the ordered visible states and infer the demonstrated grasp position, posture, insertion depth, closure, and lift without assuming hidden robot or object state."
                ),
                *frame_parts,
            ],
            "tool_result",
            {
                "tool": decision.tool_name or "",
                "demo": frames[0].demo_id,
                "seed": frames[0].seed,
                "phase": frames[0].phase,
                "steps": [frame.step for frame in frames],
                "view": frames[0].view,
                "clean_for_model": all(frame.clean_for_model for frame in frames),
                "reason": decision.reason,
            },
        )

    def dispatch_tool(decision: Decision) -> ToolResult:
        name = decision.tool_name or ""
        if name in {"camera.history", "history.image", "image.history"}:
            return history_tool(decision)
        if name in {"geometry.verify", "robot.geometry"}:
            return geometry_tool(decision)
        if name in {SPATIAL_TOOL_NAME, GRASP_CANDIDATE_TOOL_NAME}:
            return spatial_tool(decision)
        if name == VERIFY_CANDIDATE_TOOL_NAME:
            return candidate_authorization_tool(decision)
        if name in {"expert.retrieve", "expert.frame", "expert.learn"}:
            return expert_tool(decision)
        return ToolResult([Part.text_part(prompt_base + f"\nUnknown tool {name!r}. Use geometry.verify, camera.history, spatial.propose_grasp_candidates, spatial.verify_pregrasp, expert.retrieve, expert.learn, action, or stop.")], "tool_error", {"tool": name})

    harness = AgentHarness(
        thread,
        logger,
        parse=parse_decision,
        kind=lambda decision: decision.kind,
        repair_prompt=prompt_base,
        max_repairs=2,
    )

    def validate_model_decision(decision: Decision) -> str | None:
        if decision.kind == "candidate_execution":
            if spatial_runtime is None:
                return "spatial candidate execution is disabled"
            error = spatial_runtime.candidate_execution_error(
                str((decision.tool_args or {}).get("candidate_id", "")),
                adapter=adapter,
            )
            if error:
                return error
            return expert_learning_gate_error(
                expert_store is not None,
                bool(expert_learning and expert_learning.learned),
            )
        error = validate_decision(
            decision,
            available_actions=available,
            task_name=adapter.task_name,
            geometry=adapter.geometry(),
            constraint_profile=local_constraint_profile,
        )
        if error:
            return error
        error = initial_direct_candidate_gate_error(decision, spatial_runtime)
        if error:
            return error
        return expert_learning_gate_error(
            expert_store is not None,
            bool(expert_learning and expert_learning.learned),
        )

    try:
        initial_parts = [Part.text_part(prompt_base)]
        if not isolated_composite_learning:
            if current.fixed_view_images:
                initial_parts.extend(
                    Part.image_part(path, label=f"fixed {view} step {current.step}")
                    for view, path in current.fixed_view_images.items()
                )
            else:
                initial_parts.append(
                    Part.image_part(
                        current.path,
                        label=f"current {current.model_view} step {current.step}",
                    )
                )
        if pending_grasp_outcome is not None:
            initial_parts.extend(
                Part.image_part(
                    Path(path),
                    label=f"post-lift grasp diagnostic view {index + 1}",
                )
                for index, path in enumerate(
                    pending_grasp_outcome.get("evidence_images", ())
                )
                if Path(path).is_file()
            )
        return harness.run(
            initial_parts,
            tool_handlers={"tool": dispatch_tool},
            validate=validate_model_decision,
            is_terminal=lambda decision: decision.kind
            in {"action", "stop", "candidate_execution"},
        )
    except (RuntimeError, ValueError) as exc:
        logger.event("eval_error", {"error": str(exc)})
        return Decision(kind="stop", reason=str(exc))


def initial_direct_candidate_gate_error(
    decision: Decision,
    spatial_runtime: SpatialPregraspRuntime | None,
) -> str | None:
    """Keep direct-frame robot control inside the candidate tool sequence."""

    if decision.kind != "action" or spatial_runtime is None:
        return None
    config = spatial_runtime.config
    if not config.candidate_proposals or config.direct_grasp_frame_checkpoint is None:
        return None
    if getattr(spatial_runtime, "executed_candidate_ids", set()):
        return None
    candidate_records = [
        record
        for record in spatial_runtime.records
        if record.get("query_tool") == GRASP_CANDIDATE_TOOL_NAME
        and record.get("policy_visible") is True
    ]
    if not candidate_records:
        return (
            "Initial direct grasp-frame candidate query is required before any camera or robot "
            "environment action. The proposed action was not executed. Inspect the current RGB and "
            "output exactly one tool call using "
            '{"tool":"spatial.propose_grasp_candidates","args":{"target":"visible target",'
            '"preferred_roles":["functional grasp region"],"visual_grounding":{'
            '"view":"current","target_box_normalized_xyxy":[x0,y0,x1,y1],'
            '"grasp_point_normalized_uv":[u,v],"confidence":0.0}},"reason":"..."}. '
            "Coordinates must be numeric values in [0,1] from the current RGB with top-left origin; "
            "omit grasp_point_normalized_uv if the exact grasp point is not visually reliable."
        )
    latest_candidate_result = candidate_records[-1].get("result", {})
    if latest_candidate_result.get("visual_evidence_sufficient") is not True:
        assessment = latest_candidate_result.get("view_assessment") or {}
        return (
            "The spatial tool did not obtain sufficient visual evidence for its grasp "
            "candidates, so camera or robot environment actions remain blocked and the "
            "candidate must not be verified or executed. The proposed action was not "
            "executed. Inspect the unresolved visual reasons and stop, or issue a new "
            "spatial.propose_grasp_candidates call with corrected visual grounding. "
            f"Unresolved reasons: {assessment.get('reasons', [])}; "
            f"stop reason: {assessment.get('stop_reason', 'unknown')}."
        )
    authorizations = getattr(spatial_runtime, "candidate_authorizations", {})
    if authorizations:
        candidate_id = next(reversed(authorizations))
        return (
            "The candidate is authorized, but manual camera or gripper environment actions remain "
            "blocked until the atomic candidate action is used. The proposed action was not "
            "executed. Output exactly "
            f'{{"tool":"{EXECUTE_CANDIDATE_TOOL_NAME}","args":'
            f'{{"candidate_id":"{candidate_id}"}},"reason":"execute the authorized 3D frame"}}.'
        )
    candidate_id = str(
        candidate_records[-1].get("result", {}).get("recommended_candidate_id", "")
    )
    return (
        "A direct grasp candidate exists but is not authorized. Manual camera or gripper "
        "environment actions are blocked and the proposed action was not executed. Output exactly "
        f'{{"tool":"{VERIFY_CANDIDATE_TOOL_NAME}","args":'
        f'{{"candidate_id":"{candidate_id}"}},"reason":"dry-run reachability and collision checks"}}.'
    )


def save_expert_learning_for_thread(
    expert_learning: ExpertLearningState,
    summary: dict[str, Any],
    thread: AgentThread,
) -> None:
    expert_learning.save(summary, summary_model=thread.client.config.model)
