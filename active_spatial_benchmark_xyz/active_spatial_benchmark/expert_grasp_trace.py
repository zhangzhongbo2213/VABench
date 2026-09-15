"""Automatic grasp-frame labels extracted from successful expert trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import transforms3d as t3d

from .grasp_event_manifest import ResolvedGraspEvent


EXPERT_GRASP_EVENT_SAMPLE_SCHEMA = "spatial.expert_grasp_event_sample.v5"
SUPPORTED_EXPERT_GRASP_EVENT_SAMPLE_SCHEMAS = {
    "spatial.expert_grasp_event_sample.v4",
    EXPERT_GRASP_EVENT_SAMPLE_SCHEMA,
}
FORBIDDEN_INFERENCE_KEYS = {
    "actor_entity_ids",
    "actor_segmentation",
    "contact_points",
    "expert_grasp_frame",
    "physics_contacts",
    "target_mask",
    "task_success",
}


@dataclass(frozen=True)
class CommandedGraspFrame:
    """Metric frame implied by the expert's final pre-close move command."""

    center_world_m: np.ndarray
    control_point_world_m: np.ndarray
    approach_axis_world: np.ndarray
    closing_axis_world: np.ndarray
    orientation_world_wxyz: np.ndarray
    control_to_grasp_center_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_world_m": _rounded(self.center_world_m),
            "control_point_world_m": _rounded(self.control_point_world_m),
            "approach_axis_world": _rounded(self.approach_axis_world),
            "closing_axis_world": _rounded(self.closing_axis_world),
            "orientation_world_wxyz": _rounded(self.orientation_world_wxyz),
            "control_to_grasp_center_m": float(self.control_to_grasp_center_m),
        }


def commanded_pose_to_grasp_frame(
    pose_world: Sequence[float],
    *,
    control_to_grasp_center_m: float = 0.12,
) -> CommandedGraspFrame:
    """Convert RoboTwin's `[xyz, wxyz]` command into a grasp-center frame."""

    pose = np.asarray(pose_world, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("commanded grasp pose must contain seven finite values")
    if control_to_grasp_center_m <= 0.0:
        raise ValueError("control_to_grasp_center_m must be positive")
    quaternion = pose[3:]
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-9:
        raise ValueError("commanded grasp quaternion must be non-zero")
    quaternion = quaternion / quaternion_norm
    rotation = t3d.quaternions.quat2mat(quaternion)
    approach = _unit(rotation[:, 0])
    closing = _unit(rotation[:, 1])
    center = pose[:3] + approach * float(control_to_grasp_center_m)
    return CommandedGraspFrame(
        center_world_m=center,
        control_point_world_m=pose[:3].copy(),
        approach_axis_world=approach,
        closing_axis_world=closing,
        orientation_world_wxyz=quaternion,
        control_to_grasp_center_m=float(control_to_grasp_center_m),
    )


class ExpertGraspTraceRecorder:
    """Intercept expert `move` calls and label every registered close event.

    RoboTwin commonly sends pregrasp, grasp and close actions in one `move`
    call. The recorder splits only batches containing a true close, preserving
    action order while exposing the physically reached state immediately before
    closure. Camera evidence is supplied by an optional callback.
    """

    def __init__(
        self,
        task: Any,
        events: Sequence[ResolvedGraspEvent],
        *,
        on_preapproach: Callable[[ResolvedGraspEvent, Mapping[str, Any]], Any]
        | None = None,
        on_before_close: Callable[
            [ResolvedGraspEvent, Mapping[str, Any], Callable[..., Any]],
            Mapping[str, Any] | None,
        ]
        | None = None,
        on_preclose: Callable[[ResolvedGraspEvent, Mapping[str, Any]], Any]
        | None = None,
        close_threshold: float = 0.2,
        control_to_grasp_center_m: float = 0.12,
    ) -> None:
        if not events:
            raise ValueError("at least one resolved grasp event is required")
        if not 0.0 <= close_threshold < 1.0:
            raise ValueError("close_threshold must be in [0, 1)")
        self.task = task
        self.events = tuple(events)
        self.on_preapproach = on_preapproach
        self.on_before_close = on_before_close
        self.on_preclose = on_preclose
        self.close_threshold = float(close_threshold)
        self.control_to_grasp_center_m = float(control_to_grasp_center_m)
        self.traces: list[dict[str, Any]] = []
        self.unmatched_closes: list[dict[str, Any]] = []
        self._used_event_ids: set[str] = set()
        self._last_move_pose_by_arm: dict[str, np.ndarray] = {}
        self._action_index = 0
        self._original_move: Callable[..., Any] | None = None

    @property
    def unresolved_event_ids(self) -> tuple[str, ...]:
        return tuple(
            event.event_id
            for event in self.events
            if event.event_id not in self._used_event_ids
        )

    def install(self) -> None:
        if self._original_move is not None:
            raise RuntimeError("expert grasp trace recorder is already installed")
        self._original_move = self.task.move
        self.task.move = self._recording_move

    def uninstall(self) -> None:
        if self._original_move is None:
            return
        self.task.move = self._original_move
        self._original_move = None

    def run_play_once(self) -> Any:
        self.install()
        try:
            return self.task.play_once()
        finally:
            self.uninstall()

    def execute_original_move(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a probe move without recursively recording another event."""

        if self._original_move is None:
            raise RuntimeError("expert grasp trace recorder is not installed")
        return self._original_move(*args, **kwargs)

    def _recording_move(
        self,
        actions_by_arm1: tuple[Any, list[Any]],
        actions_by_arm2: tuple[Any, list[Any]] | None = None,
        save_freq: int | None = -1,
    ) -> Any:
        if self._original_move is None:
            raise RuntimeError("expert grasp trace recorder is not installed")
        groups = [actions_by_arm1, actions_by_arm2]
        if not any(
            self._is_true_close(action)
            for group in groups
            if group is not None
            for action in group[1]
        ):
            result = self._original_move(
                actions_by_arm1,
                actions_by_arm2,
                save_freq=save_freq,
            )
            if result:
                self._remember_successful_moves(groups)
            self._action_index += max(
                (len(group[1]) for group in groups if group is not None),
                default=0,
            )
            return result

        reservations = self._reserve_preapproach_events(groups)
        columns = _action_columns(actions_by_arm1, actions_by_arm2)
        overall_result = True
        for column in columns:
            closers = [
                action
                for group in column
                if group is not None
                for action in group[1]
                if self._is_true_close(action)
            ]
            pending = [
                self._begin_close(action, reservations=reservations)
                for action in closers
            ]
            present = [group for group in column if group is not None]
            if len(present) == 2:
                result = self._original_move(
                    present[0], present[1], save_freq=save_freq
                )
            elif present:
                result = self._original_move(present[0], save_freq=save_freq)
            else:
                result = True
            overall_result = overall_result and bool(result)
            if result:
                self._remember_successful_moves(present)
            for attempt in pending:
                if attempt is not None:
                    self._finish_close(attempt, planner_success=bool(result))
            self._action_index += 1
            if not result:
                break
        return overall_result

    def _is_true_close(self, action: Any) -> bool:
        return bool(
            getattr(action, "action", None) == "gripper"
            and getattr(action, "target_gripper_pos", 1.0) is not None
            and float(action.target_gripper_pos) <= self.close_threshold
        )

    def _remember_successful_moves(
        self, groups: Sequence[tuple[Any, list[Any]] | None]
    ) -> None:
        for group in groups:
            if group is None:
                continue
            for action in group[1]:
                if getattr(action, "action", None) != "move":
                    continue
                pose = np.asarray(action.target_pose, dtype=np.float64)
                if pose.shape == (7,) and np.all(np.isfinite(pose)):
                    self._last_move_pose_by_arm[str(action.arm_tag)] = pose.copy()

    def _reserve_preapproach_events(
        self, groups: Sequence[tuple[Any, list[Any]] | None]
    ) -> dict[str, dict[str, Any]]:
        reservations: dict[str, dict[str, Any]] = {}
        for group in groups:
            if group is None:
                continue
            actions = list(group[1])
            close_indices = [
                index
                for index, action in enumerate(actions)
                if self._is_true_close(action)
            ]
            if not close_indices:
                continue
            close_index = close_indices[-1]
            move_actions = [
                action
                for action in actions[:close_index]
                if getattr(action, "action", None) == "move"
            ]
            if not move_actions:
                continue
            action = actions[close_index]
            arm = str(action.arm_tag)
            commanded_pose = np.asarray(
                move_actions[-1].target_pose, dtype=np.float64
            )
            frame = commanded_pose_to_grasp_frame(
                commanded_pose,
                control_to_grasp_center_m=self.control_to_grasp_center_m,
            )
            event = self._select_event(arm, frame.center_world_m)
            if event is None:
                continue
            if len(move_actions) >= 2:
                pregrasp_position = np.asarray(
                    move_actions[-2].target_pose[:3], dtype=np.float64
                )
                grasp_position = commanded_pose[:3]
                expert_pregrasp_distance = abs(
                    float(
                        np.dot(
                            grasp_position - pregrasp_position,
                            frame.approach_axis_world,
                        )
                    )
                )
            else:
                expert_pregrasp_distance = 0.0
            preapproach = {
                "action_index": self._action_index,
                "commanded_grasp_frame": frame.as_dict(),
                "target_pose_world": _actor_pose(event.actor),
                "target_body_poses_world": _actor_body_poses(event.actor),
                "gripper_pose_world": _finite_list(self.task.get_arm_pose(arm)),
                "expert_pregrasp_distance_m": round(
                    expert_pregrasp_distance, 8
                ),
            }
            evidence = (
                self.on_preapproach(event, preapproach)
                if self.on_preapproach
                else None
            )
            reservations[arm] = {
                "event": event,
                "preapproach": preapproach,
                "preapproach_evidence": evidence,
                "frame": frame,
            }
        return reservations

    def _begin_close(
        self,
        action: Any,
        *,
        reservations: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        arm = str(action.arm_tag)
        reservation = reservations.pop(arm, None)
        if reservation is not None:
            event = reservation["event"]
            frame = reservation["frame"]
            preapproach = reservation["preapproach"]
            preapproach_evidence = reservation["preapproach_evidence"]
        else:
            event = None
            frame = None
            preapproach = None
            preapproach_evidence = None
        commanded_pose = self._last_move_pose_by_arm.get(arm)
        if commanded_pose is None:
            self.unmatched_closes.append(
                {
                    "action_index": self._action_index,
                    "arm": arm,
                    "reason": "no_preceding_move_pose",
                }
            )
            return None
        if frame is None:
            frame = commanded_pose_to_grasp_frame(
                commanded_pose,
                control_to_grasp_center_m=self.control_to_grasp_center_m,
            )
        if event is None:
            event = self._select_event(arm, frame.center_world_m)
        if event is None:
            self.unmatched_closes.append(
                {
                    "action_index": self._action_index,
                    "arm": arm,
                    "reason": "no_unresolved_registered_event",
                }
            )
            return None
        if event.event_id not in self._used_event_ids:
            self._used_event_ids.add(event.event_id)
        if preapproach is None:
            preapproach = {
                "action_index": self._action_index,
                "commanded_grasp_frame": frame.as_dict(),
                "target_pose_world": _actor_pose(event.actor),
                "target_body_poses_world": _actor_body_poses(event.actor),
                "gripper_pose_world": _finite_list(self.task.get_arm_pose(arm)),
            }
        preparation = None
        if self.on_before_close is not None:
            preparation_value = self.on_before_close(
                event,
                {
                    "action_index": self._action_index,
                    "preapproach": preapproach,
                    "expert_commanded_grasp_frame": frame.as_dict(),
                },
                self.execute_original_move,
            )
            if preparation_value is not None:
                if not isinstance(preparation_value, Mapping):
                    raise TypeError("on_before_close must return a mapping or None")
                preparation = dict(preparation_value)
                override_pose = preparation.get("commanded_pose_world")
                if override_pose is not None:
                    frame = commanded_pose_to_grasp_frame(
                        override_pose,
                        control_to_grasp_center_m=self.control_to_grasp_center_m,
                    )
        preclose = {
            "action_index": self._action_index,
            "commanded_grasp_frame": frame.as_dict(),
            "target_pose_world": _actor_pose(event.actor),
            "target_body_poses_world": _actor_body_poses(event.actor),
            "gripper_pose_world": _finite_list(self.task.get_arm_pose(arm)),
            "non_target_gripper_contact_audit": non_target_gripper_contact_audit(
                self.task, event.actor, arm
            ),
        }
        preclose_evidence = (
            self.on_preclose(event, preclose) if self.on_preclose else None
        )
        return {
            "event": event,
            "preapproach": preapproach,
            "preclose": preclose,
            "preparation": preparation,
            "evidence": {
                "preapproach": preapproach_evidence,
                "preclose": preclose_evidence,
            },
            "frame": frame,
        }

    def _select_event(
        self, arm: str, grasp_center_world_m: np.ndarray
    ) -> ResolvedGraspEvent | None:
        candidates = [
            event
            for event in self.events
            if event.event_id not in self._used_event_ids and event.arm == arm
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda event: float(
                np.linalg.norm(
                    np.asarray(event.actor.get_pose().p, dtype=np.float64)
                    - grasp_center_world_m
                )
            ),
        )

    def _finish_close(
        self, attempt: Mapping[str, Any], *, planner_success: bool
    ) -> None:
        event = attempt["event"]
        frame: CommandedGraspFrame = attempt["frame"]
        contacts = extract_actor_gripper_contacts(
            self.task,
            event.actor,
            event.arm,
            closing_axis_world=frame.closing_axis_world,
            reference_center_world_m=frame.center_world_m,
            preclose_target_body_poses_world=attempt["preapproach"][
                "target_body_poses_world"
            ],
        )
        frame_label = build_expert_grasp_frame_label(frame, contacts)
        trace = {
            "event_id": event.event_id,
            "task": event.task,
            "actor_name": event.actor_name,
            "arm": event.arm,
            "task_stage": event.task_stage,
            "post_grasp_goal": event.post_grasp_goal,
            "coordination_group": event.coordination_group,
            "action_index": attempt["preclose"]["action_index"],
            "planner_success": bool(planner_success)
            and bool((attempt.get("preparation") or {}).get("planner_success", True)),
            "preapproach": attempt["preapproach"],
            "preclose": attempt["preclose"],
            "preparation": attempt.get("preparation"),
            "postclose": {
                "target_pose_world": _actor_pose(event.actor),
                "gripper_pose_world": _finite_list(
                    self.task.get_arm_pose(event.arm)
                ),
                "contacts": contacts,
            },
            "expert_grasp_frame": frame_label,
            "evidence": attempt.get("evidence"),
            "access": "oracle/training_only",
        }
        self.traces.append(trace)


def extract_actor_gripper_contacts(
    task: Any,
    actor: Any,
    arm: str,
    *,
    closing_axis_world: Sequence[float],
    reference_center_world_m: Sequence[float],
    preclose_target_body_poses_world: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Read training-only contacts and canonicalize them along closing axis."""

    target_names = _actor_body_names(actor)
    finger_names = _finger_link_names(task, arm)
    closing_axis = _unit(np.asarray(closing_axis_world, dtype=np.float64))
    center = np.asarray(reference_center_world_m, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    points_by_finger: dict[str, list[np.ndarray]] = {}
    postclose_body_poses = _actor_body_poses(actor)
    aligned_point_count = 0
    for contact in task.scene.get_contacts():
        body_names = [str(body.entity.name) for body in contact.bodies]
        target_matches = target_names.intersection(body_names)
        finger_matches = finger_names.intersection(body_names)
        if not target_matches or not finger_matches:
            continue
        finger_name = sorted(finger_matches)[0]
        target_name = sorted(target_matches)[0]
        for point in contact.points:
            position = np.asarray(point.position, dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            preclose_pose = (
                preclose_target_body_poses_world or {}
            ).get(target_name)
            postclose_pose = postclose_body_poses.get(target_name)
            if preclose_pose is not None and postclose_pose is not None:
                aligned_position = _transform_point_between_body_poses(
                    position,
                    source_pose_world=postclose_pose,
                    target_pose_world=preclose_pose,
                )
                aligned_point_count += 1
            else:
                aligned_position = position
            points_by_finger.setdefault(finger_name, []).append(aligned_position)
            rows.append(
                {
                    "target_body": target_name,
                    "finger_body": finger_name,
                    "position_world_m_after_close": _rounded(position),
                    "position_world_m_preclose_aligned": _rounded(
                        aligned_position
                    ),
                    "impulse_norm": round(
                        float(np.linalg.norm(point.impulse)), 8
                    ),
                    "separation_m": round(float(point.separation), 9),
                }
            )
    centroids = [
        (name, np.mean(np.stack(points), axis=0), np.stack(points))
        for name, points in points_by_finger.items()
        if points
    ]
    centroids.sort(key=lambda item: float(np.dot(item[1] - center, closing_axis)))
    left = centroids[0][1] if centroids else None
    right = centroids[-1][1] if len(centroids) >= 2 else None
    bilateral = len(centroids) >= 2
    return {
        "target_body_names": sorted(target_names),
        "finger_body_names": sorted(finger_names),
        "raw_contact_points": rows,
        "preclose_alignment_applied": bool(rows)
        and aligned_point_count == len(rows),
        "contacting_finger_count": len(centroids),
        "bilateral_contact": bilateral,
        "negative_closing_contact_world_m": (
            _rounded(left) if left is not None else None
        ),
        "positive_closing_contact_world_m": (
            _rounded(right) if right is not None else None
        ),
        "opening_width_m": (
            round(float(np.linalg.norm(right - left)), 8)
            if left is not None and right is not None
            else None
        ),
        "source": "simulator_contacts_aligned_to_preclose_target_pose",
        "access": "oracle/training_only",
    }


def build_expert_grasp_frame_label(
    commanded: CommandedGraspFrame,
    contacts: Mapping[str, Any],
) -> dict[str, Any]:
    left_value = contacts.get("negative_closing_contact_world_m")
    right_value = contacts.get("positive_closing_contact_world_m")
    left = np.asarray(left_value, dtype=np.float64) if left_value is not None else None
    right = (
        np.asarray(right_value, dtype=np.float64) if right_value is not None else None
    )
    valid_contacts = bool(contacts.get("bilateral_contact"))
    variance = 1e-6 if valid_contacts else 4e-6
    return {
        "center_world_m": _rounded(commanded.center_world_m),
        "left_contact_world_m": _rounded(left) if left is not None else None,
        "right_contact_world_m": _rounded(right) if right is not None else None,
        "approach_axis_world": _rounded(commanded.approach_axis_world),
        "closing_axis_world": _rounded(commanded.closing_axis_world),
        "opening_width_m": contacts.get("opening_width_m"),
        "position_covariance_m2": np.diag([variance] * 3).tolist(),
        "orientation_covariance_rad2": np.diag([1e-4] * 3).tolist(),
        "bilateral_contact_observed": valid_contacts,
        "valid_for_contact_supervision": valid_contacts,
        "center_source": "successful_expert_commanded_grasp_frame",
        "axis_source": "successful_expert_commanded_pose",
        "point_annotation": "automatic_from_successful_trajectory",
        "access": "oracle/training_only",
    }


def build_expert_grasp_event_sample(
    *,
    task: str,
    config: str,
    seed: int,
    trace: Mapping[str, Any],
    task_success: bool,
    candidate_perturbations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_sets = trace.get("evidence")
    if not isinstance(evidence_sets, Mapping):
        raise ValueError("trace evidence sets must be a mapping")
    evidence = evidence_sets.get("preapproach")
    preclose_evidence = evidence_sets.get("preclose")
    if not isinstance(evidence, Mapping):
        raise ValueError("trace requires preapproach evidence")
    if not isinstance(preclose_evidence, Mapping):
        raise ValueError("trace requires preclose evidence")
    inference_observations = evidence.get("inference_observations")
    training_observations = evidence.get("training_observations")
    if not isinstance(inference_observations, list) or not inference_observations:
        raise ValueError("trace requires inference-visible RGB-D observations")
    if not isinstance(training_observations, list) or not training_observations:
        raise ValueError("trace requires training-only observation labels")
    candidates = [dict(candidate) for candidate in candidate_perturbations]
    if not candidates:
        raise ValueError("trace requires automatic candidate perturbations")
    sample = {
        "schema_version": EXPERT_GRASP_EVENT_SAMPLE_SCHEMA,
        "sample_id": f"{task}_seed{int(seed)}/{trace['event_id']}",
        "task": task,
        "config": config,
        "seed": int(seed),
        "world_state_version": int(trace["preapproach"]["action_index"]) + 1,
        "inference_visible": {
            "query": {
                "type": "propose_grasp_candidates",
                "target": trace["actor_name"],
                "task_goal": task,
                "task_stage": trace["task_stage"],
                "post_grasp_goal": trace["post_grasp_goal"],
                "preferred_roles": [trace["task_stage"]],
                "required_arms": 2 if trace.get("coordination_group") else 1,
                "active_arm": trace["arm"],
                "coordination_group": trace.get("coordination_group"),
                "source": "task_instruction_or_vlm_at_deployment",
                "confidence": 1.0,
            },
            "observations": inference_observations,
            "observation_phase": "preapproach_before_robot_grasp_motion",
        },
        "training_only": {
            "access": "oracle/training_only",
            "event": {
                "event_id": trace["event_id"],
                "actor_name": trace["actor_name"],
                "arm": trace["arm"],
                "action_index": trace["action_index"],
            },
            "expert_grasp_frame": trace["expert_grasp_frame"],
            "physics_contacts": trace["postclose"]["contacts"],
            "preclose_state": trace["preclose"],
            "preapproach_state": trace["preapproach"],
            "postclose_state": {
                key: value
                for key, value in trace["postclose"].items()
                if key != "contacts"
            },
            "observations": training_observations,
            "observation_sets": {
                "preapproach": {
                    "inference_observations": inference_observations,
                    "training_observations": training_observations,
                },
                "preclose": {
                    "inference_observations": preclose_evidence.get(
                        "inference_observations", []
                    ),
                    "training_observations": preclose_evidence.get(
                        "training_observations", []
                    ),
                },
            },
            "view_supervision": evidence.get("view_supervision", []),
            "candidate_perturbations": candidates,
            "outcome": {
                "planner_success": bool(trace["planner_success"]),
                "task_success": bool(task_success),
            },
            "label_generation": {
                "point_annotation": "automatic_from_successful_trajectory",
                "manual_metric_point_labels": False,
                "simulator_contacts_are_training_only": True,
            },
        },
    }
    errors = validate_expert_grasp_event_sample(sample)
    if errors:
        raise ValueError("invalid expert grasp event sample: " + "; ".join(errors))
    return sample


def validate_expert_grasp_event_sample(sample: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    schema_version = sample.get("schema_version")
    if schema_version not in SUPPORTED_EXPERT_GRASP_EVENT_SAMPLE_SCHEMAS:
        errors.append("unsupported schema_version")
    inference = sample.get("inference_visible")
    training = sample.get("training_only")
    if not isinstance(inference, Mapping):
        errors.append("inference_visible must be a mapping")
        return errors
    if not isinstance(training, Mapping):
        errors.append("training_only must be a mapping")
        return errors
    leaked = sorted(_recursive_keys(inference) & FORBIDDEN_INFERENCE_KEYS)
    if leaked:
        errors.append("training-only fields leaked into inference_visible: " + ", ".join(leaked))
    if training.get("access") != "oracle/training_only":
        errors.append("training_only.access must be oracle/training_only")
    frame = training.get("expert_grasp_frame")
    if not isinstance(frame, Mapping):
        errors.append("training_only.expert_grasp_frame must be a mapping")
    elif frame.get("point_annotation") != "automatic_from_successful_trajectory":
        errors.append("expert frame must be automatically labelled")
    observations = inference.get("observations")
    if not isinstance(observations, list) or not observations:
        errors.append("inference_visible.observations must be non-empty")
    if inference.get("observation_phase") != "preapproach_before_robot_grasp_motion":
        errors.append("inference observations must come from preapproach")
    query = inference.get("query")
    if schema_version == EXPERT_GRASP_EVENT_SAMPLE_SCHEMA:
        if not isinstance(query, Mapping):
            errors.append("v5 inference_visible.query must be a mapping")
        else:
            for name in ("target", "task_goal", "task_stage", "post_grasp_goal"):
                if not isinstance(query.get(name), str) or not query[name].strip():
                    errors.append(f"v5 query.{name} must be non-empty")
            preferred_roles = query.get("preferred_roles")
            if not isinstance(preferred_roles, list) or not preferred_roles:
                errors.append("v5 query.preferred_roles must be non-empty")
            if query.get("required_arms") not in (1, 2):
                errors.append("v5 query.required_arms must be 1 or 2")
    observation_sets = training.get("observation_sets")
    if not isinstance(observation_sets, Mapping):
        errors.append("training_only.observation_sets must be a mapping")
    else:
        for phase in ("preapproach", "preclose"):
            phase_value = observation_sets.get(phase)
            if not isinstance(phase_value, Mapping):
                errors.append(f"observation_sets.{phase} must be a mapping")
                continue
            if not phase_value.get("inference_observations"):
                errors.append(
                    f"observation_sets.{phase}.inference_observations must be non-empty"
                )
    preclose_state = training.get("preclose_state")
    if not isinstance(preclose_state, Mapping) or not isinstance(
        preclose_state.get("non_target_gripper_contact_audit"), Mapping
    ):
        errors.append("preclose_state requires expert contact baseline audit")
    candidates = training.get("candidate_perturbations")
    if not isinstance(candidates, list) or not candidates:
        errors.append("training_only.candidate_perturbations must be non-empty")
    else:
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                errors.append("candidate perturbations must be mappings")
                continue
            if candidate.get("access") != "oracle/training_only":
                errors.append("candidate perturbations must remain training-only")
            if candidate.get("predicted_execution_success") is not None:
                errors.append("unprobed candidate cannot claim execution success")
    return errors


def _action_columns(
    first: tuple[Any, list[Any]], second: tuple[Any, list[Any]] | None
) -> list[tuple[tuple[Any, list[Any]] | None, tuple[Any, list[Any]] | None]]:
    groups = [first, second]
    length = max(len(group[1]) for group in groups if group is not None)
    result = []
    for index in range(length):
        column = []
        for group in groups:
            if group is None or index >= len(group[1]):
                column.append(None)
            else:
                column.append((group[0], [group[1][index]]))
        result.append((column[0], column[1]))
    return result


def _actor_body_names(actor: Any) -> set[str]:
    names = {str(actor.get_name())}
    raw_actor = getattr(actor, "actor", actor)
    get_links = getattr(raw_actor, "get_links", None)
    if callable(get_links):
        names.update(str(link.get_name()) for link in get_links())
    return {name for name in names if name}


def actor_body_names(actor: Any) -> set[str]:
    """Return simulator body names belonging to one rigid/articulated target."""

    return _actor_body_names(actor)


def _actor_body_poses(actor: Any) -> dict[str, list[float]]:
    raw_actor = getattr(actor, "actor", actor)
    get_links = getattr(raw_actor, "get_links", None)
    if callable(get_links):
        return {
            str(link.get_name()): _pose_value(link.get_pose())
            for link in get_links()
        }
    return {str(actor.get_name()): _pose_value(actor.get_pose())}


def _pose_value(pose: Any) -> list[float]:
    return _finite_list(np.concatenate([pose.p, pose.q]))


def _transform_point_between_body_poses(
    point_world: np.ndarray,
    *,
    source_pose_world: Sequence[float],
    target_pose_world: Sequence[float],
) -> np.ndarray:
    source = np.asarray(source_pose_world, dtype=np.float64)
    target = np.asarray(target_pose_world, dtype=np.float64)
    if source.shape != (7,) or target.shape != (7,):
        raise ValueError("body poses must contain [xyz, wxyz]")
    source_rotation = t3d.quaternions.quat2mat(_unit(source[3:]))
    target_rotation = t3d.quaternions.quat2mat(_unit(target[3:]))
    point_local = source_rotation.T @ (point_world - source[:3])
    return target[:3] + target_rotation @ point_local


def _finger_link_names(task: Any, arm: str) -> set[str]:
    robot = task.robot
    names: set[str] = set()
    for value in getattr(robot, f"{arm}_gripper", ()):
        joint = value[0] if isinstance(value, (tuple, list)) else value
        child = getattr(joint, "child_link", None)
        if child is not None:
            names.add(str(child.get_name()))
    names.update(str(value) for value in getattr(robot, f"{arm}_fix_gripper_name", ()))
    if not names:
        raise RuntimeError(f"could not resolve {arm} gripper collision links")
    return names


def finger_link_names(task: Any, arm: str) -> set[str]:
    """Return collision-link names for one gripper."""

    return _finger_link_names(task, arm)


def non_target_gripper_contact_audit(
    task: Any, target_actor: Any, arm: str
) -> dict[str, Any]:
    """Measure gripper contacts excluding the intended target body."""

    target_names = actor_body_names(target_actor)
    finger_names = finger_link_names(task, arm)
    rows = []
    impulses = []
    separations = []
    for contact in task.scene.get_contacts():
        names = {str(body.entity.name) for body in contact.bodies}
        if not names.intersection(finger_names) or names.intersection(target_names):
            continue
        other_names = names - finger_names
        if not other_names:
            continue
        point_impulses = [
            float(np.linalg.norm(point.impulse)) for point in contact.points
        ]
        point_separations = [float(point.separation) for point in contact.points]
        impulses.extend(point_impulses)
        separations.extend(point_separations)
        rows.append(
            {
                "pair": sorted(names),
                "point_count": len(contact.points),
                "max_impulse": max(point_impulses, default=0.0),
                "minimum_separation_m": min(point_separations, default=0.0),
            }
        )
    return {
        "non_target_contact": bool(rows),
        "max_non_target_contact_impulse": max(impulses, default=None),
        "minimum_non_target_separation_m": min(separations, default=None),
        "contacts": rows,
    }


def _actor_pose(actor: Any) -> list[float]:
    pose = actor.get_pose()
    return _finite_list(np.concatenate([pose.p, pose.q]))


def _finite_list(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("trajectory state contains non-finite values")
    return [round(float(item), 8) for item in array.reshape(-1)]


def _rounded(value: np.ndarray | Sequence[float]) -> list[float]:
    return [round(float(item), 8) for item in np.asarray(value).reshape(-1)]


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key) for key in value}
        for item in value.values():
            result.update(_recursive_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_recursive_keys(item))
        return result
    return set()
