"""Inference-time planning and guarded execution for sparse grasp candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .appearance_runtime import encode_candidate_appearance
from .env import robotwin_cwd
from .grasp_candidates import GraspCandidate
from .grasp_execution import candidate_to_robot_grasp_poses
from .rgbd_approach_corridor import analyze_rgbd_approach_corridor
from .rgbd_grasp_outcome import (
    build_target_appearance_signature,
    verify_postgrasp_outcome,
)


FORBIDDEN_PROVENANCE_TOKENS = ("oracle", "simulator_truth", "training_only")


def candidate_preexecution_context(
    env: Any,
    candidate_value: Mapping[str, Any] | GraspCandidate,
    *,
    arm: str | None = None,
    rgbd_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compute deployment-available context without moving the robot.

    Data generation may pass an oracle-generated candidate frame, but every
    returned measurement is produced by the same workspace bounds, motion
    planner, robot state, and RGB-D corridor analysis available at inference.
    Candidate provenance and simulator contact outcomes are never features.
    """

    candidate = (
        candidate_value
        if isinstance(candidate_value, GraspCandidate)
        else GraspCandidate.from_mapping(candidate_value)
    )
    selected_arm = _selected_arm(env, arm)
    poses = candidate_to_robot_grasp_poses(candidate)
    state_before = runtime_world_state_digest(env)
    workspace_valid = _poses_in_workspace(env, poses)
    corridor = analyze_rgbd_approach_corridor(candidate, rgbd_evidence)
    candidate_center = np.asarray(candidate.center_world_m, dtype=np.float64)
    pregrasp_center = np.asarray(candidate.pregrasp_center_world_m, dtype=np.float64)
    grasp_control = np.asarray(poses.grasp_pose[:3], dtype=np.float64)
    pregrasp_control = np.asarray(poses.pregrasp_pose[:3], dtype=np.float64)
    support_z = corridor.get("support_plane_z_m")
    support_z = float(support_z) if support_z is not None else None
    workspace_normalized = _workspace_normalized_position(env, grasp_control)
    current_gripper = _current_gripper_world_position(env, selected_arm)
    planner = (
        _plan_candidate_paths(env, poses, selected_arm)
        if workspace_valid
        else {
            "arm": selected_arm,
            "pregrasp": {"status": "workspace_fail", "waypoint_count": 0},
            "grasp": {"status": "not_run", "waypoint_count": 0},
        }
    )
    pregrasp_collision = planner["pregrasp"].get("endpoint_collision", {})
    grasp_collision = planner["grasp"].get("endpoint_collision", {})
    state_after = runtime_world_state_digest(env)
    if state_before != state_after:
        raise RuntimeError("candidate dry-run planning changed the world state")
    return {
        "schema_version": "spatial.candidate_preexecution_context.v3",
        "workspace_bounds_pass": bool(workspace_valid),
        "pregrasp_reachable": planner["pregrasp"]["status"] == "Success",
        "grasp_reachable": planner["grasp"]["status"] == "Success",
        "pregrasp_waypoint_count": int(planner["pregrasp"]["waypoint_count"]),
        "grasp_waypoint_count": int(planner["grasp"]["waypoint_count"]),
        "corridor_clear": bool(corridor["clear"]),
        "corridor_coverage_sufficient": bool(corridor["coverage_sufficient"]),
        "corridor_evidence_view_count": int(corridor["evidence_view_count"]),
        "corridor_local_voxel_count": int(corridor["local_voxel_count"]),
        "corridor_obstacle_voxel_count": int(corridor["obstacle_voxel_count"]),
        "corridor_minimum_non_target_clearance_m": corridor[
            "minimum_non_target_clearance_m"
        ],
        "candidate_center_world_m": _rounded(candidate_center),
        "pregrasp_center_world_m": _rounded(pregrasp_center),
        "grasp_control_world_m": _rounded(grasp_control),
        "pregrasp_control_world_m": _rounded(pregrasp_control),
        "support_plane_z_m": support_z,
        "candidate_center_support_clearance_m": _support_clearance(
            candidate_center, support_z
        ),
        "grasp_control_support_clearance_m": _support_clearance(
            grasp_control, support_z
        ),
        "pregrasp_control_support_clearance_m": _support_clearance(
            pregrasp_control, support_z
        ),
        "grasp_control_workspace_normalized": _rounded(workspace_normalized),
        "current_gripper_world_m": (
            _rounded(current_gripper) if current_gripper is not None else None
        ),
        "current_gripper_to_pregrasp_distance_m": (
            float(np.linalg.norm(current_gripper - pregrasp_control))
            if current_gripper is not None
            else None
        ),
        "planner_collision_check_available": bool(
            pregrasp_collision.get("available")
            and grasp_collision.get("available")
        ),
        "pregrasp_self_collision_count": int(
            pregrasp_collision.get("self_collision_count", 0)
        ),
        "pregrasp_environment_collision_count": int(
            pregrasp_collision.get("environment_collision_count", 0)
        ),
        "pregrasp_support_collision_count": int(
            pregrasp_collision.get("support_collision_count", 0)
        ),
        "grasp_self_collision_count": int(
            grasp_collision.get("self_collision_count", 0)
        ),
        "grasp_environment_collision_count": int(
            grasp_collision.get("environment_collision_count", 0)
        ),
        "grasp_support_collision_count": int(
            grasp_collision.get("support_collision_count", 0)
        ),
        "planner_self_collision_free": bool(
            pregrasp_collision.get("available")
            and grasp_collision.get("available")
            and pregrasp_collision.get("self_collision_count", 0) == 0
            and grasp_collision.get("self_collision_count", 0) == 0
        ),
        "planner_support_collision_free": bool(
            pregrasp_collision.get("available")
            and grasp_collision.get("available")
            and pregrasp_collision.get("support_collision_count", 0) == 0
            and grasp_collision.get("support_collision_count", 0) == 0
        ),
        "world_state_unchanged": True,
        "source": "inference_visible_robot_state_planner_and_multiview_rgbd",
        "access": "inference_visible_derived",
    }


def verify_candidate_executability(
    env: Any,
    candidate_value: Mapping[str, Any] | GraspCandidate,
    *,
    arm: str | None = None,
    rgbd_evidence: Sequence[Mapping[str, Any]] = (),
    appearance_encoder_config: Mapping[str, Any] | None = None,
    appearance_output_dir: str | Path | None = None,
    appearance_text_embedding: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Plan to a candidate without executing or reading target-object truth."""

    candidate = _validated_candidate(candidate_value)
    selected_arm = _selected_arm(env, arm)
    poses = candidate_to_robot_grasp_poses(candidate)
    state_before = runtime_world_state_digest(env)
    checks: dict[str, dict[str, Any]] = {
        "inference_visible_candidate": _check(
            True,
            source="candidate_access_and_provenance",
            hard_constraint=True,
        ),
        "frame_geometry_valid": _check(
            _finite_grasp_poses(poses),
            source="deterministic_grasp_frame_conversion",
            hard_constraint=True,
        ),
        "workspace_bounds": _check(
            _poses_in_workspace(env, poses),
            source="interactive_env_gripper_bounds",
            hard_constraint=True,
            measurement={
                "pregrasp_control_xyz_m": _rounded(poses.pregrasp_pose[:3]),
                "grasp_control_xyz_m": _rounded(poses.grasp_pose[:3]),
                "bounds_m": np.round(np.asarray(env.gripper_bounds), 7).tolist(),
            },
        ),
    }
    approach_corridor = analyze_rgbd_approach_corridor(
        candidate, rgbd_evidence
    )
    checks["rgbd_final_approach_corridor_clear"] = _check(
        bool(approach_corridor["clear"]),
        source="inference_visible_multiview_rgbd",
        hard_constraint=True,
        measurement=approach_corridor,
    )
    target_signature = build_target_appearance_signature(
        candidate,
        rgbd_evidence,
        support_z_m=approach_corridor.get("support_plane_z_m"),
    )
    if appearance_encoder_config is not None and appearance_output_dir is not None:
        try:
            target_signature["frozen_appearance_shadow"] = encode_candidate_appearance(
                candidate,
                rgbd_evidence,
                center_world_m=candidate.center_world_m,
                output_dir=appearance_output_dir,
                encoder=appearance_encoder_config,
                stage="pre_execution",
                minimum_world_z_m=(
                    float(approach_corridor["support_plane_z_m"]) + 0.006
                    if approach_corridor.get("support_plane_z_m") is not None
                    else None
                ),
                text_embedding=appearance_text_embedding,
            )
        except Exception as exc:
            target_signature["frozen_appearance_shadow"] = {
                "available": False,
                "reason": "shadow_encoder_exception",
                "error_type": type(exc).__name__,
                "controls_authorization": False,
                "access": "inference_visible_server_side",
            }
    checks["target_appearance_trackable"] = _check(
        bool(target_signature.get("available")),
        source="candidate_local_inference_visible_rgbd",
        hard_constraint=False,
        measurement={
            "available": bool(target_signature.get("available")),
            "discriminative": bool(target_signature.get("discriminative")),
            "sample_count": int(target_signature.get("sample_count", 0)),
            "reason": target_signature.get("reason"),
        },
    )
    plan_summary: dict[str, Any] = {
        "arm": selected_arm,
        "pregrasp": {"status": "not_run", "waypoint_count": 0},
        "grasp": {"status": "not_run", "waypoint_count": 0},
    }
    if all(
        not row["hard_constraint"] or row["state"] == "pass"
        for row in checks.values()
    ):
        plan_summary = _plan_candidate_paths(env, poses, selected_arm)
    pregrasp_ok = plan_summary["pregrasp"]["status"] == "Success"
    grasp_ok = plan_summary["grasp"]["status"] == "Success"
    checks["pregrasp_reachable"] = _check(
        pregrasp_ok,
        source="robot_motion_planner_dry_run",
        hard_constraint=True,
        measurement=plan_summary["pregrasp"],
    )
    checks["grasp_reachable"] = _check(
        grasp_ok,
        source="robot_motion_planner_dry_run",
        hard_constraint=True,
        measurement=plan_summary["grasp"],
    )
    collision_scope = _planner_collision_scope(
        env, selected_arm, approach_corridor=approach_corridor
    )
    checks["collision_checked"] = _check(
        pregrasp_ok and grasp_ok and bool(approach_corridor["clear"]),
        source="robot_motion_planner_and_inference_visible_multiview_rgbd",
        hard_constraint=True,
        measurement=collision_scope,
    )
    checks["predicted_execution_success"] = {
        "state": "unknown",
        "probability": None,
        "hard_constraint": False,
        "source": "not_evaluated",
        "measurement": {},
    }
    state_after = runtime_world_state_digest(env)
    world_frozen = state_before == state_after
    checks["world_state_unchanged"] = _check(
        world_frozen,
        source="runtime_state_digest",
        hard_constraint=True,
    )
    hard_failures = [
        name
        for name, row in checks.items()
        if row["hard_constraint"] and row["state"] != "pass"
    ]
    authorized = not hard_failures
    candidate_hash = grasp_candidate_digest(candidate)
    authorization_id = _authorization_digest(
        candidate_hash=candidate_hash,
        world_state_digest=state_after,
        arm=selected_arm,
    )
    executability_graph = _build_executability_graph(
        candidate=candidate,
        arm=selected_arm,
        approach_corridor=approach_corridor,
        pregrasp_reachable=pregrasp_ok,
        grasp_reachable=grasp_ok,
        authorized=authorized,
    )
    return {
        "schema_version": "spatial.grasp_candidate_executability.v2",
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate_hash,
        "world_state_digest": state_after,
        "arm": selected_arm,
        "checks": checks,
        "planner": plan_summary,
        "execution_authorized": authorized,
        "authorization_id": authorization_id if authorized else None,
        "failed_hard_constraints": hard_failures,
        "predicted_execution_success": None,
        "world_frozen": world_frozen,
        "collision_scope": collision_scope,
        "approach_corridor": approach_corridor,
        "target_appearance_signature": target_signature,
        "executability_graph": executability_graph,
        "access": "inference_visible",
    }


def execute_authorized_candidate(
    env: Any,
    candidate_value: Mapping[str, Any] | GraspCandidate,
    authorization: Mapping[str, Any],
    *,
    lift_m: float = 0.12,
    post_verification_output_dir: str | Path | None = None,
    appearance_encoder_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically execute one previously planned candidate by candidate ID."""

    if lift_m <= 0.05 or lift_m > 0.25:
        raise ValueError("lift_m must be in (0.05, 0.25]")
    candidate = _validated_candidate(candidate_value)
    if authorization.get("candidate_id") != candidate.candidate_id:
        raise ValueError("authorization candidate_id does not match candidate")
    if authorization.get("candidate_digest") != grasp_candidate_digest(candidate):
        raise ValueError("candidate changed after authorization")
    if not bool(authorization.get("execution_authorized")):
        raise ValueError("candidate is not authorized for execution")
    current_state = runtime_world_state_digest(env)
    if authorization.get("world_state_digest") != current_state:
        raise ValueError("world state changed after candidate authorization")
    arm = _selected_arm(env, str(authorization.get("arm", "")))
    expected_authorization_id = _authorization_digest(
        candidate_hash=grasp_candidate_digest(candidate),
        world_state_digest=current_state,
        arm=arm,
    )
    if authorization.get("authorization_id") != expected_authorization_id:
        raise ValueError("candidate authorization token is invalid")

    poses = candidate_to_robot_grasp_poses(candidate)
    from envs.utils.action import Action

    actions = (
        ("open", Action(arm, "open", target_gripper_pos=1.0)),
        ("pregrasp", Action(arm, "move", target_pose=poses.pregrasp_pose)),
        (
            "grasp",
            Action(
                arm,
                "move",
                target_pose=poses.grasp_pose,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
        ),
        ("close", Action(arm, "close", target_gripper_pos=0.0)),
    )
    rows = []
    planner_success = True
    env.task.plan_success = True
    for phase, action in actions:
        if not planner_success:
            rows.append(
                {"phase": phase, "executed": False, "planner_success": None}
            )
            continue
        with robotwin_cwd():
            ok = bool(env.task.move((arm, [action]), save_freq=None))
            env.task._update_render()
        planner_success = planner_success and ok
        rows.append({"phase": phase, "executed": True, "planner_success": ok})

    lift_ok = False
    if planner_success:
        with robotwin_cwd():
            lift_plan = env.task.move_by_displacement(arm_tag=arm, z=float(lift_m))
            lift_ok = bool(env.task.move(lift_plan, save_freq=None))
            env.task._update_render()
        planner_success = planner_success and lift_ok
    rows.append(
        {
            "phase": "lift",
            "executed": bool(rows[-1]["planner_success"]),
            "planner_success": lift_ok if rows[-1]["planner_success"] else None,
            "distance_m": float(lift_m),
        }
    )

    grasp_verification = None
    if planner_success and post_verification_output_dir is not None:
        try:
            grasp_verification = verify_postgrasp_outcome(
                env,
                candidate,
                authorization.get("target_appearance_signature"),
                arm=arm,
                output_dir=post_verification_output_dir,
                appearance_encoder_config=appearance_encoder_config,
            )
        except Exception as exc:
            grasp_verification = {
                "schema_version": "spatial.rgbd_grasp_outcome.v1",
                "query": "verify_grasp_outcome",
                "candidate_id": candidate.candidate_id,
                "verdict": "unobservable",
                "confidence": 0.0,
                "missing_evidence": ["post_execution_rgbd_verification_failed"],
                "recommended_action": {
                    "action": "stop_for_review",
                    "reason": "post_execution_verification_error",
                },
                "error_type": type(exc).__name__,
                "access": "inference_visible",
            }

    env.step_count += 1
    action_label = f"spatial.execute_grasp_candidate.{candidate.candidate_id}"
    env.action_history.append(action_label)
    env.last_action_valid = True
    env.last_planner_success = planner_success
    env.last_error = None if planner_success else "candidate execution planner failed"
    with robotwin_cwd():
        env.success = bool(env.task.check_success())
    env.done = env.success or env.step_count >= env.max_steps
    result = {
        "schema_version": "spatial.grasp_candidate_execution.v1",
        "candidate_id": candidate.candidate_id,
        "authorization_id": authorization["authorization_id"],
        "arm": arm,
        "action": action_label,
        "phases": rows,
        "planner_success": planner_success,
        "grasp_verification": grasp_verification,
        "task_success_evaluation_only": env.success,
        "task_success_model_visible": False,
        "world_state_digest_after": runtime_world_state_digest(env),
        "access": "mixed",
    }
    return {
        "observation": env._make_observation(),
        "reward": 1.0 if env.success else 0.0,
        "done": env.done,
        "info": env._info(),
        "candidate_execution": result,
    }


def runtime_world_state_digest(env: Any) -> str:
    """Hash physical state for authorization invalidation without exposing poses."""

    with robotwin_cwd():
        actors = list(env.task.scene.get_all_actors())
    actor_rows = []
    for index, actor in enumerate(actors):
        pose = actor.get_pose()
        actor_rows.append(
            {
                "index": index,
                "name": str(actor.get_name()),
                "pose": _rounded(np.concatenate([pose.p, pose.q]), digits=9),
            }
        )
    robot = env.task.robot
    payload = {
        "actors": actor_rows,
        "left_qpos": _rounded(robot.left_entity.get_qpos(), digits=9),
        "right_qpos": _rounded(robot.right_entity.get_qpos(), digits=9),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def grasp_candidate_digest(candidate: GraspCandidate) -> str:
    return hashlib.sha256(
        json.dumps(
            candidate.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _validated_candidate(
    candidate_value: Mapping[str, Any] | GraspCandidate,
) -> GraspCandidate:
    candidate = (
        candidate_value
        if isinstance(candidate_value, GraspCandidate)
        else GraspCandidate.from_mapping(candidate_value)
    )
    if candidate.access != "inference_visible":
        raise ValueError("candidate must have inference_visible access")
    serialized = json.dumps(candidate.as_dict(), sort_keys=True).lower()
    forbidden = [token for token in FORBIDDEN_PROVENANCE_TOKENS if token in serialized]
    if forbidden:
        raise ValueError(
            "candidate contains forbidden provenance token(s): " + ", ".join(forbidden)
        )
    if not candidate.eligible:
        raise ValueError(
            "candidate has failed hard constraints: "
            + ", ".join(candidate.failed_constraints)
        )
    return candidate


def _selected_arm(env: Any, requested: str | None) -> str:
    value = str(requested or getattr(env, "active_arm", "")).lower()
    if value not in {"left", "right"}:
        raise ValueError("candidate runtime currently requires one explicit arm")
    return value


def _finite_grasp_poses(poses: Any) -> bool:
    return bool(
        np.isfinite(poses.pregrasp_pose).all()
        and np.isfinite(poses.grasp_pose).all()
        and np.linalg.det(poses.rotation_world) > 0.999
    )


def _poses_in_workspace(env: Any, poses: Any) -> bool:
    bounds = np.asarray(env.gripper_bounds, dtype=np.float64)
    return bool(
        np.all(poses.pregrasp_pose[:3] >= bounds[0])
        and np.all(poses.pregrasp_pose[:3] <= bounds[1])
        and np.all(poses.grasp_pose[:3] >= bounds[0])
        and np.all(poses.grasp_pose[:3] <= bounds[1])
    )


def _workspace_normalized_position(env: Any, position: np.ndarray) -> np.ndarray:
    bounds = np.asarray(env.gripper_bounds, dtype=np.float64)
    span = np.maximum(bounds[1] - bounds[0], 1e-9)
    return (np.asarray(position, dtype=np.float64) - bounds[0]) / span


def _current_gripper_world_position(env: Any, arm: str) -> np.ndarray | None:
    getter = getattr(env.task.robot, f"get_{arm}_ee_pose", None)
    if not callable(getter):
        return None
    value = np.asarray(getter(), dtype=np.float64)
    if value.shape[0] < 3 or not np.isfinite(value[:3]).all():
        return None
    return value[:3].copy()


def _support_clearance(position: np.ndarray, support_z: float | None) -> float | None:
    if support_z is None:
        return None
    return float(np.asarray(position, dtype=np.float64)[2] - support_z)


def _plan_candidate_paths(env: Any, poses: Any, arm: str) -> dict[str, Any]:
    robot = env.task.robot
    plan = getattr(robot, f"{arm}_plan_path")
    entity = getattr(robot, f"{arm}_entity")
    current_qpos = np.asarray(entity.get_qpos()).copy()
    pregrasp = plan(poses.pregrasp_pose, last_qpos=current_qpos)
    pregrasp_row = _plan_row(pregrasp)
    if pregrasp_row["status"] != "Success":
        return {
            "arm": arm,
            "pregrasp": pregrasp_row,
            "grasp": {"status": "not_run", "waypoint_count": 0},
        }
    next_qpos = _qpos_after_plan(robot, arm, current_qpos, pregrasp)
    pregrasp_row["endpoint_collision"] = _mplib_endpoint_collision_summary(
        robot, arm, next_qpos
    )
    grasp = plan(
        poses.grasp_pose,
        constraint_pose=[1, 1, 1, 0, 0, 0],
        last_qpos=next_qpos,
    )
    grasp_row = _plan_row(grasp)
    if grasp_row["status"] == "Success":
        grasp_qpos = _qpos_after_plan(robot, arm, next_qpos, grasp)
        grasp_row["endpoint_collision"] = _mplib_endpoint_collision_summary(
            robot, arm, grasp_qpos
        )
    return {
        "arm": arm,
        "pregrasp": pregrasp_row,
        "grasp": grasp_row,
    }


def _mplib_endpoint_collision_summary(
    robot: Any, arm: str, qpos: np.ndarray
) -> dict[str, Any]:
    wrapper = getattr(robot, f"{arm}_mplib_planner", None)
    planner = getattr(wrapper, "planner", None)
    if planner is None or not all(
        callable(getattr(planner, name, None))
        for name in ("check_for_self_collision", "check_for_env_collision")
    ):
        return {
            "available": False,
            "self_collision_count": 0,
            "environment_collision_count": 0,
            "support_collision_count": 0,
        }
    try:
        self_collisions = list(planner.check_for_self_collision(qpos))
        environment_collisions = list(planner.check_for_env_collision(qpos))
    except Exception as exc:
        return {
            "available": False,
            "self_collision_count": 0,
            "environment_collision_count": 0,
            "support_collision_count": 0,
            "error_type": type(exc).__name__,
        }
    support_collisions = sum(
        _collision_mentions_support(collision) for collision in environment_collisions
    )
    return {
        "available": True,
        "self_collision_count": len(self_collisions),
        "environment_collision_count": len(environment_collisions),
        "support_collision_count": int(support_collisions),
        "source": "mplib_dry_run_endpoint_collision",
        "access": "inference_visible_robot_model_and_static_workspace",
    }


def _collision_mentions_support(collision: Any) -> bool:
    names = (
        getattr(collision, "link_name1", ""),
        getattr(collision, "link_name2", ""),
        getattr(collision, "object_name1", ""),
        getattr(collision, "object_name2", ""),
    )
    normalized = " ".join(str(name).casefold() for name in names)
    return any(token in normalized for token in ("table", "support_surface"))


def _qpos_after_plan(
    robot: Any,
    arm: str,
    current_qpos: np.ndarray,
    plan_result: Mapping[str, Any],
) -> np.ndarray:
    final = np.asarray(plan_result["position"][-1], dtype=current_qpos.dtype)
    if final.shape == current_qpos.shape:
        return final.copy()
    planner = getattr(robot, f"{arm}_planner", None)
    active_names = list(getattr(planner, "active_joints_name", ()))
    all_names = list(getattr(planner, "all_joints", ()))
    indices = [all_names.index(name) for name in active_names if name in all_names]
    if len(indices) != len(final):
        raise RuntimeError(
            "planner result cannot be mapped back to the full articulation qpos"
        )
    result = current_qpos.copy()
    result[indices] = final
    return result


def _plan_row(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    position = result.get("position")
    return {
        "status": str(result.get("status", "Fail")),
        "waypoint_count": int(len(position)) if position is not None else 0,
    }


def _planner_collision_scope(
    env: Any,
    arm: str,
    *,
    approach_corridor: Mapping[str, Any],
) -> dict[str, Any]:
    robot = env.task.robot
    planner = getattr(robot, f"{arm}_planner", None)
    planner_class = type(planner).__name__ if planner is not None else "unknown"
    return {
        "planner_class": planner_class,
        "coverage": (
            "robot_self_and_planner_configured_static_world_plus_observed_"
            "rgbd_final_approach_corridor"
        ),
        "dynamic_rgbd_obstacle_sweep": True,
        "dynamic_rgbd_obstacle_sweep_clear": bool(
            approach_corridor.get("clear")
        ),
        "dynamic_rgbd_evidence_view_count": int(
            approach_corridor.get("evidence_view_count", 0)
        ),
        "target_contact_prediction": False,
        "limitation": (
            "RGB-D coverage is limited to observed geometry around the final "
            "pregrasp-to-grasp center segment. It does not claim hidden-surface, "
            "full-gripper-mesh, current-to-pregrasp dynamic clearance, or grasp success."
        ),
    }


def _check(
    passed: bool,
    *,
    source: str,
    hard_constraint: bool,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "state": "pass" if passed else "fail",
        "probability": 1.0 if passed else 0.0,
        "hard_constraint": hard_constraint,
        "source": source,
        "measurement": dict(measurement or {}),
    }


def _authorization_digest(
    *, candidate_hash: str, world_state_digest: str, arm: str
) -> str:
    return hashlib.sha256(
        f"{candidate_hash}:{world_state_digest}:{arm}".encode("ascii")
    ).hexdigest()


def _build_executability_graph(
    *,
    candidate: GraspCandidate,
    arm: str,
    approach_corridor: Mapping[str, Any],
    pregrasp_reachable: bool,
    grasp_reachable: bool,
    authorized: bool,
) -> dict[str, Any]:
    corridor_id = f"{candidate.candidate_id}.final_approach_corridor"
    arm_id = f"robot.{arm}_arm"
    return {
        "schema_version": "spatial.grasp_candidate_executability_graph.v1",
        "nodes": [
            {
                "id": candidate.candidate_id,
                "node_type": "grasp_candidate",
                "semantic_role": candidate.semantic_role,
                "access": "inference_visible",
            },
            {
                "id": corridor_id,
                "node_type": "approach_corridor",
                "attributes": dict(approach_corridor.get("segment", {})),
                "access": "inference_visible",
            },
            {
                "id": arm_id,
                "node_type": "robot_arm",
                "access": "inference_visible",
            },
            {
                "id": "scene.observed_geometry",
                "node_type": "observed_geometry",
                "attributes": {
                    "evidence_view_count": approach_corridor.get(
                        "evidence_view_count"
                    ),
                    "coverage": approach_corridor.get("coverage"),
                },
                "access": "inference_visible",
            },
            {
                "id": "robot.atomic_grasp_execution",
                "node_type": "atomic_action",
                "access": "inference_visible",
            },
        ],
        "edges": [
            {
                "source": candidate.candidate_id,
                "target": corridor_id,
                "relation": "has_final_approach",
                "state": "pass",
                "probability": 1.0,
                "source_type": "deterministic_grasp_frame_conversion",
            },
            {
                "source": candidate.candidate_id,
                "target": arm_id,
                "relation": "planner_reachable_by",
                "state": (
                    "pass" if pregrasp_reachable and grasp_reachable else "fail"
                ),
                "probability": (
                    1.0 if pregrasp_reachable and grasp_reachable else 0.0
                ),
                "source_type": "robot_motion_planner_dry_run",
            },
            {
                "source": corridor_id,
                "target": "scene.observed_geometry",
                "relation": "observed_clear_of",
                "state": "pass" if approach_corridor.get("clear") else "fail",
                "probability": 1.0 if approach_corridor.get("clear") else 0.0,
                "source_type": "inference_visible_multiview_rgbd",
                "measurement": {
                    "obstacle_voxel_count": approach_corridor.get(
                        "obstacle_voxel_count"
                    ),
                    "minimum_non_target_clearance_m": approach_corridor.get(
                        "minimum_non_target_clearance_m"
                    ),
                    "coverage": approach_corridor.get("coverage"),
                },
            },
            {
                "source": candidate.candidate_id,
                "target": "robot.atomic_grasp_execution",
                "relation": "authorized_for",
                "state": "pass" if authorized else "fail",
                "probability": 1.0 if authorized else 0.0,
                "source_type": "runtime_hard_constraint_gate",
            },
        ],
        "access": "inference_visible",
    }


def _rounded(value: Any, *, digits: int = 7) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), digits).tolist()
