from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
PARAM_WORLD_MOVE_RE = re.compile(r"^(?:(left|right)_)?gripper\.world_([xyz])_(pos|neg)\.([0-9]+(?:\.[0-9]+)?)mm$")
PARAM_ROTATE_RE = re.compile(r"^(?:(left|right)_)?gripper\.(local_)?rotate_r([xyz])_(ccw|cw)\.([0-9]+(?:\.[0-9]+)?)deg$")
KNOWN_TOOL_NAMES = {
    "camera.history",
    "expert.frame",
    "expert.learn",
    "expert.retrieve",
    "geometry.verify",
    "spatial.verify_pregrasp",
    "spatial.verify_candidate_executability",
}
TERMINAL_ACTION_TOOL_NAMES = {"spatial.execute_grasp_candidate"}


@dataclass(frozen=True)
class Decision:
    kind: str
    action: str | None = None
    action_params: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    reason: str = ""


def parse_decision(text: str) -> Decision:
    data = parse_json_object(text)
    reason = str(data.get("reason", ""))
    if data.get("stop") is True:
        return Decision(kind="stop", reason=reason)

    tool = data.get("tool", data.get("tool_name"))
    if isinstance(tool, str) and tool.strip():
        tool_name = tool.strip()
        args = data.get("args", data.get("tool_args", {}))
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("tool args must be an object")
        return Decision(
            kind=(
                "candidate_execution"
                if tool_name in TERMINAL_ACTION_TOOL_NAMES
                else "tool"
            ),
            tool_name=tool_name,
            tool_args=args,
            reason=reason,
        )

    action = data.get("action")
    if isinstance(action, str) and action.strip():
        name = action.strip()
        if name in KNOWN_TOOL_NAMES:
            args = data.get("args", data.get("tool_args", {}))
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ValueError("tool args must be an object")
            return Decision(
                kind="tool",
                tool_name=name,
                tool_args=args,
                reason=reason,
            )
        if name == "gripper.move_world":
            return parse_move_world(data, reason)
        if name == "gripper.rotate_world":
            return parse_rotate(data, reason, frame="world")
        if name == "gripper.rotate_local":
            return parse_rotate(data, reason, frame="local")
        if name == "dual_gripper.move_world":
            return parse_dual_move_world(data, reason)
        return Decision(kind="action", action=name, reason=reason)

    raise ValueError('JSON must contain "action", "tool", or true "stop"')


def parse_move_world(data: dict[str, Any], reason: str) -> Decision:
    arm = normalize_arm(data.get("arm"))
    axis = str(data.get("axis", "")).lower()
    sign_raw = str(data.get("sign", "")).lower()
    sign_value, sign_name = normalize_sign(sign_raw, positive="pos", negative="neg")
    distance = data.get("distance_mm")
    try:
        distance_float = float(distance)
    except (TypeError, ValueError):
        distance_float = None
    if axis in {"x", "y", "z"} and sign_name in {"pos", "neg"} and distance_float is not None:
        prefix = f"{arm}_gripper" if arm else "gripper"
        action = f"{prefix}.world_{axis}_{sign_name}.{format_number(distance_float)}mm"
    else:
        action = "gripper.move_world"
    action_params = {
        "target": "gripper",
        "type": "move_world",
        "axis": axis,
        "sign": sign_value,
        "distance_mm": distance,
    }
    if arm is not None:
        action_params["arm"] = arm
    return Decision(
        kind="action",
        action=action,
        action_params=action_params,
        reason=reason,
    )


def parse_rotate(data: dict[str, Any], reason: str, *, frame: str) -> Decision:
    arm = normalize_arm(data.get("arm"))
    axis = str(data.get("axis", "")).lower()
    if axis in {"x", "y", "z"}:
        axis = "r" + axis
    sign_raw = str(data.get("sign", "")).lower()
    sign_value, sign_name = normalize_sign(sign_raw, positive="ccw", negative="cw")
    angle = data.get("angle_deg")
    try:
        angle_float = float(angle)
    except (TypeError, ValueError):
        angle_float = None
    if axis in {"rx", "ry", "rz"} and sign_name in {"ccw", "cw"} and angle_float is not None:
        prefix = "local_rotate" if frame == "local" else "rotate"
        gripper_prefix = f"{arm}_gripper" if arm else "gripper"
        action = f"{gripper_prefix}.{prefix}_{axis}_{sign_name}.{format_number(angle_float)}deg"
    else:
        action = f"gripper.rotate_{frame}"
    if frame == "local":
        action_params = {
            "target": "gripper",
            "type": "rotate_local",
            "axis": axis,
            "sign": sign_value,
            "angle_deg": angle,
        }
    else:
        action_params = {
            "target": "gripper",
            "type": "rotate",
            "direction": f"rotate_{axis}_{sign_name}",
            "angle_deg": angle,
        }
    if arm is not None:
        action_params["arm"] = arm
    return Decision(
        kind="action",
        action=action,
        action_params=action_params,
        reason=reason,
    )


def parse_dual_move_world(data: dict[str, Any], reason: str) -> Decision:
    left, left_valid = normalize_dual_move_side(data.get("left"))
    right, right_valid = normalize_dual_move_side(data.get("right"))
    action = "dual_gripper.move_world" if left_valid and right_valid else "dual_gripper.move_world.invalid"
    return Decision(
        kind="action",
        action=action,
        action_params={
            "target": "dual_gripper",
            "type": "move_world",
            "left": left,
            "right": right,
        },
        reason=reason,
    )


def normalize_dual_move_side(value: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, dict):
        return {}, False
    axis = str(value.get("axis", "")).lower()
    sign_value, sign_name = normalize_sign(str(value.get("sign", "")).lower(), positive="pos", negative="neg")
    distance = value.get("distance_mm")
    try:
        distance_float = float(distance)
    except (TypeError, ValueError):
        distance_float = None
    valid = (
        axis in {"x", "y", "z"}
        and sign_name in {"pos", "neg"}
        and distance_float is not None
        and 1.0 <= distance_float <= 100.0
    )
    return {"axis": axis, "sign": sign_value, "distance_mm": distance}, valid


def normalize_arm(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    arm = str(value).lower()
    return arm if arm in {"left", "right"} else str(value)


def normalize_sign(value: str, *, positive: str, negative: str) -> tuple[str, str]:
    if value in {"+", "pos", "positive", "ccw", "+1", "1"}:
        return "+", positive
    if value in {"-", "neg", "negative", "cw", "-1"}:
        return "-", negative
    return value, value


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    match = JSON_BLOCK.search(value)
    if match:
        value = match.group(1).strip()
    elif not value.startswith("{"):
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("model output JSON must be an object")
    return data


def decision_env_action(decision: Decision) -> str | dict[str, Any]:
    if decision.action_params is not None:
        return dict(decision.action_params)
    if not decision.action:
        raise ValueError("decision has no action")
    return decision.action


def decision_label(decision: Decision) -> str:
    if decision.action:
        return decision.action
    return compact_action(decision_env_action(decision))


def compact_action(action: str | dict[str, Any]) -> str:
    if isinstance(action, str):
        return action
    if action.get("target") == "gripper" and action.get("type") == "move_world":
        axis = str(action.get("axis", "")).lower()
        sign = str(action.get("sign", ""))
        sign_name = "pos" if sign in {"+", "pos", "positive", "+1", "1"} else "neg"
        prefix = f"{action.get('arm')}_gripper" if action.get("arm") else "gripper"
        return f"{prefix}.world_{axis}_{sign_name}.{format_number(float(action.get('distance_mm', 0.0)))}mm"
    if action.get("target") == "gripper" and action.get("type") in {"rotate", "rotate_local"}:
        if action.get("type") == "rotate_local":
            axis = str(action.get("axis", ""))
            if axis in {"x", "y", "z"}:
                axis = "r" + axis
            sign_raw = str(action.get("sign", ""))
            sign_name = "ccw" if sign_raw in {"+", "pos", "positive", "ccw", "+1", "1"} else "cw"
            prefix = f"{action.get('arm')}_gripper" if action.get("arm") else "gripper"
            return f"{prefix}.local_rotate_{axis}_{sign_name}.{format_number(float(action.get('angle_deg', 0.0)))}deg"
        direction = str(action.get("direction", ""))
        prefix = f"{action.get('arm')}_gripper" if action.get("arm") else "gripper"
        return f"{prefix}.{direction}.{format_number(float(action.get('angle_deg', 0.0)))}deg"
    if action.get("target") == "dual_gripper" and action.get("type") == "move_world":
        return "dual_gripper.move_world"
    return json.dumps(action, sort_keys=True)


def is_parameterized_action(action: str | None) -> bool:
    if not action:
        return False
    return PARAM_WORLD_MOVE_RE.match(action) is not None or PARAM_ROTATE_RE.match(action) is not None


def parameterized_action_error(action: str | None) -> str | None:
    if not action:
        return "missing action"
    move = PARAM_WORLD_MOVE_RE.match(action)
    if move:
        distance_mm = float(move.group(4))
        if 1.0 <= distance_mm <= 100.0:
            return None
        return f"parameterized gripper translation {distance_mm:g} mm is outside [1, 100] mm"
    rotate = PARAM_ROTATE_RE.match(action)
    if rotate:
        angle_deg = float(rotate.group(5))
        if 1.0 <= angle_deg <= 90.0:
            return None
        return f"parameterized gripper rotation {angle_deg:g} deg is outside [1, 90] deg"
    return f"unsupported parameterized action syntax: {action!r}"


def is_world_z_neg(decision: Decision) -> bool:
    if decision.action and re.match(r"^(?:(?:left|right)_)?gripper\.world_z_neg\.", decision.action):
        return True
    params = decision.action_params or {}
    return (
        params.get("target") == "gripper"
        and params.get("type") == "move_world"
        and str(params.get("axis", "")).lower() == "z"
        and str(params.get("sign", "")) in {"-", "neg", "negative", "-1"}
    )


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")
