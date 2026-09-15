from __future__ import annotations

from typing import Any

from agent.protocol import Decision, is_parameterized_action, parameterized_action_error


GENERIC_CONSTRAINT_PROFILE = "generic"


def validate_decision(
    decision: Decision,
    *,
    available_actions: set[str],
    task_name: str,
    geometry: dict[str, Any] | None,
    constraint_profile: str = GENERIC_CONSTRAINT_PROFILE,
) -> str | None:
    if decision.kind == "stop":
        return None
    if decision.kind != "action" or not decision.action:
        return "output must be an action or stop after tool use"
    if is_discrete_gripper_motion(decision.action):
        return "gripper movement and rotation must use numeric JSON commands, not gripper .small/.medium/.large/.xlarge scale actions"
    if decision.action not in available_actions:
        if not any(action.startswith("camera.") for action in available_actions) and decision.action.startswith("camera."):
            return "camera actions are disabled in fixed-camera evaluation"
        if not is_parameterized_action(decision.action):
            return f"invalid action {decision.action!r}; choose from available actions or supported parameterized commands"
        error = parameterized_action_error(decision.action)
        if error:
            return error
    params = decision.action_params or {}
    if (
        (geometry or {}).get("active_arm") == "both"
        and params.get("target") == "gripper"
        and params.get("arm") not in {"left", "right"}
    ):
        return "single-gripper numeric action in dual-arm mode requires arm='left' or arm='right'"
    if constraint_profile != GENERIC_CONSTRAINT_PROFILE:
        return f"unsupported local constraint profile: {constraint_profile!r}"
    return None


def allowed_actions_for_model(actions: list[str], camera_policy: str) -> list[str]:
    base_actions = [action for action in actions if not is_discrete_gripper_motion(action)]
    if camera_policy == "full":
        return base_actions
    if camera_policy == "fixed":
        return [action for action in base_actions if not action.startswith("camera.")]
    if camera_policy != "fine":
        raise ValueError(f"unsupported camera policy: {camera_policy}")
    result: list[str] = []
    for action in base_actions:
        if not action.startswith("camera."):
            result.append(action)
            continue
        if action.startswith("camera.look_at_") or action.endswith(".medium"):
            continue
        result.append(action)
    return result


def is_discrete_gripper_motion(action: str) -> bool:
    parts = action.split(".")
    if len(parts) != 3 or parts[0] not in {"gripper", "left_gripper", "right_gripper"}:
        return False
    direction, scale = parts[1], parts[2]
    if scale not in {"small", "medium", "large", "xlarge"}:
        return False
    return direction.startswith(("world_", "image_", "depth_", "lift_", "rotate_"))
