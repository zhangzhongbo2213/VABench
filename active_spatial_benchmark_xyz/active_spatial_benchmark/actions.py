from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


GRIPPER_TRANSLATION_SCALES = {
    "small": 0.015,
    "medium": 0.04,
    "large": 0.10,
}

GRIPPER_LIFT_SCALES = {
    "small": 0.015,
    "medium": 0.04,
}

GRIPPER_ROTATION_SCALES_DEG = {
    "small": 8.0,
    "medium": 20.0,
    "large": 45.0,
    "xlarge": 90.0,
}

CAMERA_TRANSLATION_SCALES = {
    "small": 0.05,
    "medium": 0.12,
}

CAMERA_ROTATION_SCALES_DEG = {
    "small": 10.0,
    "medium": 25.0,
}

GRIPPER_TRANSLATE_DIRECTIONS = {
    "image_left",
    "image_right",
    "image_up",
    "image_down",
    "depth_forward",
    "depth_backward",
    "lift_up",
    "lift_down",
}

GRIPPER_WORLD_TRANSLATE_DIRECTIONS = {
    "world_x_neg",
    "world_x_pos",
    "world_y_neg",
    "world_y_pos",
    "world_z_neg",
    "world_z_pos",
}

GRIPPER_ALL_TRANSLATE_DIRECTIONS = GRIPPER_TRANSLATE_DIRECTIONS | GRIPPER_WORLD_TRANSLATE_DIRECTIONS

GRIPPER_ROTATE_DIRECTIONS = {
    "rotate_rx_cw",
    "rotate_rx_ccw",
    "rotate_ry_cw",
    "rotate_ry_ccw",
    "rotate_rz_cw",
    "rotate_rz_ccw",
}

GRIPPER_LEGACY_ROTATE_DIRECTIONS = {
    "rotate_cw",
    "rotate_ccw",
}

CAMERA_MOVE_DIRECTIONS = {
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "zoom_in",
    "zoom_out",
}

CAMERA_ROTATE_DIRECTIONS = {
    "yaw_left",
    "yaw_right",
    "pitch_up",
    "pitch_down",
}

ARM_NAMES = {"left", "right"}

CAMERA_LOOK_AT_MODES = {
    "gripper",
    "left_gripper",
    "right_gripper",
    "both_grippers",
    "workspace",
}

CAMERA_BASE_VIEW_MODES = {
    "front_side_45",
    "oblique_45",
    "side",
    "side_top_45",
    "topdown",
}

CAMERA_VIEW_MODES = CAMERA_BASE_VIEW_MODES | {
    f"{arm}_{mode}"
    for arm in ("left", "right", "both")
    for mode in CAMERA_BASE_VIEW_MODES
}


class ActionValidationError(ValueError):
    pass


ACTION_SPACE_DESCRIPTION = """Action space semantics for this XYZ benchmark variant:
- The active observation contains an RGB image and the listed discrete actions. The model should choose one action per step.
- Gripper translation actions use fixed world XYZ axes, independent of the active camera pose. Prefer the explicit world-coordinate action names: world_x_neg=-world X, world_x_pos=+world X, world_y_neg=-world Y, world_y_pos=+world Y, world_z_neg=-world Z, world_z_pos=+world Z.
- Do not choose a gripper.world_x_* or gripper.world_y_* translation as the first action from an oblique/center_high view. Use camera.view_topdown first to establish the XY relation. Every world_x/world_y translation reason should include direction evidence such as "direction: target is +world Y from GC" or "direction: retreat is -world X". If direction is uncertain or only perspective-inferred, inspect with camera.view_topdown instead. Large XY translations require fresh topdown evidence after the latest gripper pose change.
- Historical names like image_left or depth_forward remain accepted for backward compatibility, but benchmark agents should use the world_* names.
- Gripper translation mapping: image_left=-world X, image_right=+world X, image_up=+world Y, image_down=-world Y, depth_forward=-world Z, depth_backward=+world Z, lift_up=+world Z, lift_down=-world Z.
- Gripper translation scales: world_x_*/world_y_* and legacy planar moves support small=0.015 m, medium=0.040 m, large=0.100 m. world_z_*, lift_up, and lift_down support small=0.015 m and medium=0.040 m.
- Gripper rotation actions rotate the gripper orientation around fixed world axes, independent of the active camera pose and independent of the current gripper orientation. rotate_rx_* rotates around world X/red, rotate_ry_* rotates around world Y/green, and rotate_rz_* rotates around world Z/blue. The orange/cyan/magenta local rx/ry/rz arrows still show the current gripper orientation, but they are diagnostic overlays, not the rotation-action axes.
- Gripper rotation sign convention: rotate_{axis}_ccw is a positive right-hand-rule rotation around the selected fixed world axis; rotate_{axis}_cw is a negative right-hand-rule rotation around the selected fixed world axis. small=8 deg, medium=20 deg, large=45 deg, and xlarge=90 deg.
- Backward-compatible gripper.rotate_ccw and gripper.rotate_cw are aliases for gripper.rotate_rx_ccw and gripper.rotate_rx_cw.
- A gripper rotation is intended to keep the visual gripper/finger center (yellow GC) xyz fixed while changing orientation. The environment computes compensated control eepose waypoints from the fixed GC, moves the control eepose to each compensated position, then rotates toward that waypoint. Therefore control eepose xyz may move during rotate actions even when GC is intended to stay fixed. The exact executed pose can still be limited by the robot planner. Rotation is not collision-free: the gripper fingers, palm, and wrist still sweep through nearby space. Before any rotate action, confirm safe clearance from the bottle and table; if the gripper is close to, overlapping, touching, or occluded by the bottle/table, first create safety space with camera inspection, world_z_pos, or a small/medium horizontal world_* move away.
- For bottle grasping, do not lower or close based on center overlap alone. Before gripper.world_z_neg.* as a downward grasp approach or gripper.close, use current/top/side views to confirm grasp posture: the jaw/closing axis between the two fingers, approximately local ry/cyan for this Aloha gripper, should be perpendicular to the bottle long axis so the fingers straddle the bottle diameter. The final pre-close camera order should be topdown, then side, then one subtle side-derived camera action; do not switch back to topdown before closing. In the final side-derived view, the yellow finger-center/GC point must lie on the middle of the bottle body, not above/below it, left/right of it, or on the neck/cap/end. If not, choose gripper.rotate_rx_* (world X), gripper.rotate_ry_* (world Y), gripper.rotate_rz_* (world Z), or world_* corrective translation before lowering or closing.
- Treat the latest gripper.rotate_* as a rotate-lock for the final approach. For bottle grasping, every action reason should explicitly assess whether the gripper jaw/closing axis is perpendicular to the bottle long axis. Do not use gripper.world_z_neg.* or gripper.close unless the latest evidence positively confirms that perpendicular relationship. If the jaw axis is parallel, not perpendicular, uncertain, or occluded, use camera inspection or gripper.rotate_* correction instead of lowering or closing. rotate_rz_* alone is an in-plane/yaw adjustment and is not enough for vertical descent alignment after a downward world_z_neg approach. After rotate-lock, use at most one final world_* translation before gripper.close; if another translation is needed, rotate-lock again.
- gripper.open and gripper.close only command the gripper aperture; a TCP pose change is not required.
- Camera move actions use the current camera frame: move_left/move_right translate along camera left/right, move_up/move_down translate along camera up/down, zoom_in/zoom_out translate along camera forward/backward.
- Camera rotation actions change only the camera view: yaw_left/yaw_right rotate around world Z; pitch_up/pitch_down rotate around the current camera left axis. small=10 deg and medium=25 deg.
- camera.look_at_gripper and camera.look_at_workspace keep the camera position and reorient it toward the active gripper or workspace center.
- Fixed diagnostic camera views are centered on the active gripper/finger region. camera.view_topdown looks down from above; camera.view_side looks horizontally from the active-arm side; camera.view_front_side_45 is halfway between front and side in the horizontal plane; camera.view_side_top_45 is halfway between side and top; camera.view_oblique_45 is a diagonal elevated view. Use the intermediate views to resolve projection overlap or depth ambiguity instead of relying only on topdown and side.
- If eepose overlays are shown, red/green/blue arrows are fixed world X/Y/Z and are the axes used by gripper.rotate_rx/ry/rz actions. Orange/cyan/magenta arrows are current gripper local rx/ry/rz diagnostic orientation axes, the yellow dot is the visual gripper/finger center, and the text panel shows xyz, rpy, quaternion, and gripper aperture.
"""

DUAL_ARM_ACTION_SPACE_DESCRIPTION = """
Dual-arm mode:
- Both robot arms are controllable. Every single-gripper numeric command must include arm="left" or arm="right". Compact aperture commands are left_gripper.open/close and right_gripper.open/close. There is no hidden selected-arm state.
- Single-arm numeric translation: {"action":"gripper.move_world","arm":"left|right","axis":"x|y|z","sign":"+|-","distance_mm":N}. Single-arm numeric local rotation uses the same arm field with action gripper.rotate_local.
- Synchronized translation: {"action":"dual_gripper.move_world","left":{"axis":"x|y|z","sign":"+|-","distance_mm":N},"right":{"axis":"x|y|z","sign":"+|-","distance_mm":N}}. Both target poses are planned and executed in the same RoboTwin move call.
- dual_gripper.open and dual_gripper.close command both apertures in the same RoboTwin move call. Use synchronized translation when task success depends on coordinated motion, such as lifting an object with both hands; do not approximate it with alternating single-arm lifts.
- Generic camera.view_* actions center the two-arm midpoint in dual-arm mode. Explicit camera.view_left_*, camera.view_right_*, and camera.view_both_* variants select the observation target without changing robot control. camera.look_at_left_gripper, camera.look_at_right_gripper, and camera.look_at_both_grippers reorient the current camera in place.
- "left" and "right" name the robot arms, not image-left/image-right. Choose an arm from visible reach, collision clearance, task role, and current robot geometry. Camera motion never changes either arm's world XYZ controls.
"""


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class ActionSpec:
    target: str
    type: str
    direction: str | None = None
    scale: str | None = None
    distance_m: float | None = None
    angle_deg: float | None = None
    mode: str | None = None
    rotation_frame: str | None = None
    arm: str | None = None
    left_action: ActionSpec | None = None
    right_action: ActionSpec | None = None
    raw: Any = None

    def compact(self) -> str:
        gripper_prefix = f"{self.arm}_gripper" if self.arm else "gripper"
        if self.target == "gripper" and self.type in {"open", "close"}:
            return f"{gripper_prefix}.{self.type}"
        if self.target == "dual_gripper" and self.type in {"open", "close"}:
            return f"dual_gripper.{self.type}"
        if self.target == "dual_gripper" and self.type == "translate":
            assert self.left_action is not None and self.right_action is not None
            return f"dual_gripper.move_world[{self.left_action.compact()},{self.right_action.compact()}]"
        if self.target == "camera" and self.type == "look_at":
            return f"camera.look_at_{self.mode}"
        if self.target == "camera" and self.type == "view":
            return f"camera.view_{self.mode}"
        if self.target == "gripper" and self.distance_m is not None:
            return f"{gripper_prefix}.{self.direction}.{format_number(self.distance_m * 1000.0)}mm"
        if self.target == "gripper" and self.angle_deg is not None:
            if self.rotation_frame == "local":
                return f"{gripper_prefix}.local_{self.direction}.{format_number(self.angle_deg)}deg"
            return f"{gripper_prefix}.{self.direction}.{format_number(self.angle_deg)}deg"
        if self.target == "gripper" and self.scale is not None:
            return f"{gripper_prefix}.{self.direction}.{self.scale}"
        if self.scale is None:
            return f"{self.target}.{self.direction}"
        return f"{self.target}.{self.direction}.{self.scale}"


def list_available_actions(active_arm: str | None = None) -> list[str]:
    actions: list[str] = []
    gripper_prefixes = ["left_gripper", "right_gripper"] if active_arm == "both" else ["gripper"]
    for prefix in gripper_prefixes:
        for direction in sorted(GRIPPER_ALL_TRANSLATE_DIRECTIONS):
            scales = gripper_translation_scales(direction)
            actions.extend(f"{prefix}.{direction}.{scale}" for scale in scales)
        for direction in sorted(GRIPPER_ROTATE_DIRECTIONS):
            actions.extend(f"{prefix}.{direction}.{scale}" for scale in GRIPPER_ROTATION_SCALES_DEG)
        actions.extend([f"{prefix}.open", f"{prefix}.close"])
    if active_arm == "both":
        actions.extend(["dual_gripper.move_world", "dual_gripper.open", "dual_gripper.close"])

    for direction in sorted(CAMERA_MOVE_DIRECTIONS):
        if direction.startswith("zoom_"):
            actions.append(f"camera.{direction}")
        else:
            actions.extend(f"camera.{direction}.{scale}" for scale in CAMERA_TRANSLATION_SCALES)
    for direction in sorted(CAMERA_ROTATE_DIRECTIONS):
        actions.extend(f"camera.{direction}.{scale}" for scale in CAMERA_ROTATION_SCALES_DEG)
    look_at_modes = CAMERA_LOOK_AT_MODES if active_arm == "both" else {"gripper", "workspace"}
    view_modes = CAMERA_VIEW_MODES if active_arm == "both" else CAMERA_BASE_VIEW_MODES
    actions.extend(f"camera.look_at_{mode}" for mode in sorted(look_at_modes))
    actions.extend(f"camera.view_{mode}" for mode in sorted(view_modes))
    return actions


def describe_action_space(active_arm: str | None = None) -> str:
    if active_arm == "both":
        return ACTION_SPACE_DESCRIPTION + DUAL_ARM_ACTION_SPACE_DESCRIPTION
    return ACTION_SPACE_DESCRIPTION


def describe_action(raw: str | Mapping[str, Any]) -> str:
    action = parse_action(raw)
    if action.target == "gripper" and action.type == "translate":
        assert action.direction is not None
        scale = gripper_translation_distance(action)
        direction_effects = {
            "image_left": "-world X",
            "image_right": "+world X",
            "image_up": "+world Y",
            "image_down": "-world Y",
            "depth_forward": "-world Z",
            "depth_backward": "+world Z",
            "lift_up": "+world Z",
            "lift_down": "-world Z",
            "world_x_neg": "-world X",
            "world_x_pos": "+world X",
            "world_y_neg": "-world Y",
            "world_y_pos": "+world Y",
            "world_z_neg": "-world Z",
            "world_z_pos": "+world Z",
        }
        return f"Move TCP {scale:.3f} m along {direction_effects[action.direction]}."
    if action.target == "gripper" and action.type == "rotate":
        assert action.direction is not None
        angle = gripper_rotation_angle_deg(action)
        axis, signed_direction = gripper_rotation_axis_and_sign(action.direction)
        sign = "+" if signed_direction > 0 else "-"
        if action.rotation_frame == "local":
            local_axis = f"local r{axis.lower()}"
            return f"Rotate gripper {sign}{angle:.1f} deg around current gripper {local_axis}, using compensated control eepose waypoints to keep the yellow GC/finger-center approximately fixed."
        effect = gripper_rotation_effect(axis, signed_direction)
        return f"Rotate gripper {sign}{angle:.1f} deg around fixed world {axis} ({effect}), using compensated control eepose waypoints to keep the yellow GC/finger-center approximately fixed."
    if action.target == "gripper" and action.type == "open":
        return f"Open the {action.arm or 'active'} gripper; TCP pose change is not required."
    if action.target == "gripper" and action.type == "close":
        return f"Close the {action.arm or 'active'} gripper; TCP pose change is not required."
    if action.target == "dual_gripper" and action.type in {"open", "close"}:
        return f"{action.type.title()} both grippers synchronously."
    if action.target == "dual_gripper" and action.type == "translate":
        return "Move both grippers to their independently specified world-axis targets in one synchronized RoboTwin move call."
    if action.target == "camera" and action.type == "move":
        assert action.direction is not None
        if action.direction == "zoom_in":
            return "Move camera 0.080 m forward along the current camera view direction."
        if action.direction == "zoom_out":
            return "Move camera 0.080 m backward against the current camera view direction."
        assert action.scale is not None
        scale = CAMERA_TRANSLATION_SCALES[action.scale]
        effects = {
            "move_left": "camera left",
            "move_right": "camera right",
            "move_up": "camera up",
            "move_down": "camera down",
        }
        return f"Move camera {scale:.3f} m along {effects[action.direction]}."
    if action.target == "camera" and action.type == "rotate":
        assert action.direction is not None and action.scale is not None
        angle = CAMERA_ROTATION_SCALES_DEG[action.scale]
        if action.direction.startswith("yaw_"):
            axis = "world Z"
        else:
            axis = "current camera left axis"
        return f"Rotate camera view {angle:.1f} deg for {action.direction} around {axis}."
    if action.target == "camera" and action.type == "look_at":
        return f"Reorient camera toward {action.mode}; camera position is unchanged."
    if action.target == "camera" and action.type == "view":
        return f"Move camera to a fixed {action.mode} diagnostic viewpoint centered on the active gripper/finger region."
    return "Unknown action effect."


def parse_action(raw: str | Mapping[str, Any]) -> ActionSpec:
    if isinstance(raw, str):
        return _parse_compact(raw)
    if isinstance(raw, Mapping):
        return _parse_mapping(raw)
    raise ActionValidationError(f"Action must be a string or mapping, got {type(raw).__name__}.")


def _parse_compact(raw: str) -> ActionSpec:
    parts = raw.strip().split(".")
    if len(parts) < 2:
        raise ActionValidationError(f"Invalid compact action: {raw!r}")
    target = parts[0]
    name = parts[1]

    arm = None
    if target in {"left_gripper", "right_gripper"}:
        arm = target.removesuffix("_gripper")
        target = "gripper"

    if target == "gripper":
        if name in {"open", "close"} and len(parts) == 2:
            return ActionSpec(target="gripper", type=name, arm=arm, raw=raw)
        if len(parts) != 3:
            raise ActionValidationError(f"Gripper action needs 2 or 3 parts: {raw!r}")
        direction, scale = name, parts[2]
        if direction in GRIPPER_ALL_TRANSLATE_DIRECTIONS:
            if scale.endswith("mm"):
                return _with_arm(_validate_gripper_translate_distance(direction, scale[:-2], "distance_mm", raw), arm)
            return _with_arm(_validate_gripper_translate(direction, scale, raw), arm)
        if direction.startswith("local_rotate_"):
            local_direction = direction.removeprefix("local_")
            if scale.endswith("deg"):
                return _with_arm(_validate_gripper_rotate_angle(local_direction, scale[:-3], raw, rotation_frame="local"), arm)
            raise ActionValidationError(f"Compact local gripper rotation requires an explicit deg value: {raw!r}")
        if direction in GRIPPER_ROTATE_DIRECTIONS or direction in GRIPPER_LEGACY_ROTATE_DIRECTIONS:
            if scale.endswith("deg"):
                return _with_arm(_validate_gripper_rotate_angle(direction, scale[:-3], raw), arm)
            return _with_arm(_validate_gripper_rotate(direction, scale, raw), arm)
        raise ActionValidationError(f"Unknown gripper direction: {direction!r}")

    if target == "dual_gripper":
        if name in {"open", "close"} and len(parts) == 2:
            return ActionSpec(target="dual_gripper", type=name, raw=raw)
        raise ActionValidationError("dual_gripper compact actions support only open and close; use mapping JSON for synchronized movement")

    if target == "camera":
        if name.startswith("look_at_") and len(parts) == 2:
            mode = name.removeprefix("look_at_")
            return _validate_camera_look_at(mode, raw)
        if name.startswith("view_") and len(parts) == 2:
            mode = name.removeprefix("view_")
            return _validate_camera_view(mode, raw)
        if name in {"zoom_in", "zoom_out"} and len(parts) == 2:
            return ActionSpec(target="camera", type="move", direction=name, raw=raw)
        if len(parts) != 3:
            raise ActionValidationError(f"Camera action needs 2 or 3 parts: {raw!r}")
        direction, scale = name, parts[2]
        if direction in CAMERA_MOVE_DIRECTIONS:
            return _validate_camera_move(direction, scale, raw)
        if direction in CAMERA_ROTATE_DIRECTIONS:
            return _validate_camera_rotate(direction, scale, raw)
        raise ActionValidationError(f"Unknown camera direction: {direction!r}")

    raise ActionValidationError(f"Unknown action target: {target!r}")


def _parse_mapping(raw: Mapping[str, Any]) -> ActionSpec:
    target = str(raw.get("target", ""))
    action_type = str(raw.get("type", ""))

    if target == "gripper":
        arm = _validate_arm(raw.get("arm"), required=False)
        if action_type in {"open", "close"}:
            return ActionSpec(target="gripper", type=action_type, arm=arm, raw=dict(raw))
        direction = str(raw.get("direction", ""))
        scale = str(raw.get("scale", ""))
        if action_type == "move_world":
            direction = move_world_direction(raw)
            return _with_arm(_validate_gripper_translate_distance(direction, raw.get("distance_mm"), "distance_mm", dict(raw)), arm)
        if action_type == "translate":
            if "distance_m" in raw:
                return _with_arm(_validate_gripper_translate_distance(direction, raw.get("distance_m"), "distance_m", dict(raw)), arm)
            if "distance_mm" in raw:
                return _with_arm(_validate_gripper_translate_distance(direction, raw.get("distance_mm"), "distance_mm", dict(raw)), arm)
            return _with_arm(_validate_gripper_translate(direction, scale, dict(raw)), arm)
        if action_type == "rotate_local":
            direction = rotate_local_direction(raw)
            return _with_arm(_validate_gripper_rotate_angle(direction, raw.get("angle_deg"), dict(raw), rotation_frame="local"), arm)
        if action_type == "rotate":
            if "angle_deg" in raw:
                return _with_arm(_validate_gripper_rotate_angle(direction, raw.get("angle_deg"), dict(raw)), arm)
            return _with_arm(_validate_gripper_rotate(direction, scale, dict(raw)), arm)
        raise ActionValidationError(f"Unknown gripper action type: {action_type!r}")

    if target == "dual_gripper":
        if action_type in {"open", "close"}:
            return ActionSpec(target="dual_gripper", type=action_type, raw=dict(raw))
        if action_type == "move_world":
            left = _parse_dual_move_side(raw.get("left"), "left")
            right = _parse_dual_move_side(raw.get("right"), "right")
            return ActionSpec(
                target="dual_gripper",
                type="translate",
                left_action=left,
                right_action=right,
                raw=dict(raw),
            )
        raise ActionValidationError(f"Unknown dual_gripper action type: {action_type!r}")

    if target == "camera":
        if action_type == "look_at":
            return _validate_camera_look_at(str(raw.get("mode", "")), dict(raw))
        if action_type == "view":
            return _validate_camera_view(str(raw.get("mode", "")), dict(raw))
        direction = str(raw.get("direction", ""))
        scale = raw.get("scale")
        if action_type == "move":
            return _validate_camera_move(direction, None if scale is None else str(scale), dict(raw))
        if action_type == "rotate":
            return _validate_camera_rotate(direction, str(scale), dict(raw))
        raise ActionValidationError(f"Unknown camera action type: {action_type!r}")

    raise ActionValidationError(f"Unknown action target: {target!r}")


def _with_arm(action: ActionSpec, arm: str | None) -> ActionSpec:
    return replace(action, arm=arm)


def _validate_arm(value: Any, *, required: bool) -> str | None:
    if value in {None, ""}:
        if required:
            raise ActionValidationError("Dual-arm gripper action requires arm='left' or arm='right'")
        return None
    arm = str(value).lower()
    if arm not in ARM_NAMES:
        raise ActionValidationError(f"Invalid arm {value!r}; expected 'left' or 'right'")
    return arm


def _parse_dual_move_side(value: Any, arm: str) -> ActionSpec:
    if not isinstance(value, Mapping):
        raise ActionValidationError(f"dual_gripper.move_world requires a {arm} movement object")
    side = dict(value)
    side.update({"target": "gripper", "type": "move_world", "arm": arm})
    action = _parse_mapping(side)
    if action.type != "translate":
        raise ActionValidationError(f"dual_gripper {arm} command must be a world translation")
    return action


def _validate_gripper_translate(direction: str, scale: str, raw: Any) -> ActionSpec:
    if direction not in GRIPPER_ALL_TRANSLATE_DIRECTIONS:
        raise ActionValidationError(f"Unknown gripper translation direction: {direction!r}")
    valid_scales = gripper_translation_scales(direction)
    if scale not in valid_scales:
        raise ActionValidationError(f"Invalid scale {scale!r} for {direction!r}")
    return ActionSpec(target="gripper", type="translate", direction=direction, scale=scale, raw=raw)


def _validate_gripper_translate_distance(direction: str, value: Any, unit: str, raw: Any) -> ActionSpec:
    if direction not in GRIPPER_ALL_TRANSLATE_DIRECTIONS:
        raise ActionValidationError(f"Unknown gripper translation direction: {direction!r}")
    try:
        distance = float(value)
    except (TypeError, ValueError):
        raise ActionValidationError(f"Invalid gripper translation distance: {value!r}") from None
    distance_m = distance / 1000.0 if unit == "distance_mm" else distance
    if not 0.001 <= distance_m <= 0.100:
        raise ActionValidationError("Parameterized gripper translation distance must be between 0.001 and 0.100 m")
    return ActionSpec(target="gripper", type="translate", direction=direction, distance_m=distance_m, raw=raw)


def gripper_translation_distance(action: ActionSpec) -> float:
    if action.distance_m is not None:
        return action.distance_m
    assert action.direction is not None and action.scale is not None
    return gripper_translation_scales(action.direction)[action.scale]


def gripper_translation_scales(direction: str) -> dict[str, float]:
    if direction.startswith("lift_") or direction.startswith("world_z_"):
        return GRIPPER_LIFT_SCALES
    return GRIPPER_TRANSLATION_SCALES


def _validate_gripper_rotate(direction: str, scale: str, raw: Any) -> ActionSpec:
    if direction not in GRIPPER_ROTATE_DIRECTIONS and direction not in GRIPPER_LEGACY_ROTATE_DIRECTIONS:
        raise ActionValidationError(f"Unknown gripper rotation direction: {direction!r}")
    if scale not in GRIPPER_ROTATION_SCALES_DEG:
        raise ActionValidationError(f"Invalid gripper rotation scale: {scale!r}")
    return ActionSpec(target="gripper", type="rotate", direction=direction, scale=scale, raw=raw)


def _validate_gripper_rotate_angle(direction: str, value: Any, raw: Any, *, rotation_frame: str | None = None) -> ActionSpec:
    if direction not in GRIPPER_ROTATE_DIRECTIONS and direction not in GRIPPER_LEGACY_ROTATE_DIRECTIONS:
        raise ActionValidationError(f"Unknown gripper rotation direction: {direction!r}")
    try:
        angle_deg = float(value)
    except (TypeError, ValueError):
        raise ActionValidationError(f"Invalid gripper rotation angle: {value!r}") from None
    if not 1.0 <= angle_deg <= 90.0:
        raise ActionValidationError("Parameterized gripper rotation angle must be between 1 and 90 deg")
    return ActionSpec(target="gripper", type="rotate", direction=direction, angle_deg=angle_deg, rotation_frame=rotation_frame, raw=raw)


def gripper_rotation_angle_deg(action: ActionSpec) -> float:
    if action.angle_deg is not None:
        return action.angle_deg
    assert action.scale is not None
    return GRIPPER_ROTATION_SCALES_DEG[action.scale]


def move_world_direction(raw: Mapping[str, Any]) -> str:
    axis = str(raw.get("axis", "")).lower()
    sign = str(raw.get("sign", "")).lower()
    if sign in {"+", "pos", "positive", "+1", "1"}:
        sign_name = "pos"
    elif sign in {"-", "neg", "negative", "-1"}:
        sign_name = "neg"
    else:
        sign_name = sign
    return f"world_{axis}_{sign_name}"


def rotate_local_direction(raw: Mapping[str, Any]) -> str:
    axis = str(raw.get("axis", "")).lower()
    sign = str(raw.get("sign", "")).lower()
    if sign in {"+", "pos", "positive", "ccw", "+1", "1"}:
        sign_name = "ccw"
    elif sign in {"-", "neg", "negative", "cw", "-1"}:
        sign_name = "cw"
    else:
        sign_name = sign
    return f"rotate_{axis}_{sign_name}"


def gripper_rotation_axis_and_sign(direction: str) -> tuple[str, int]:
    aliases = {
        "rotate_cw": "rotate_rx_cw",
        "rotate_ccw": "rotate_rx_ccw",
    }
    direction = aliases.get(direction, direction)
    parts = direction.split("_")
    if len(parts) != 3 or parts[0] != "rotate" or parts[1] not in {"rx", "ry", "rz"} or parts[2] not in {"cw", "ccw"}:
        raise ActionValidationError(f"Unknown gripper rotation direction: {direction!r}")
    axis = {"rx": "X", "ry": "Y", "rz": "Z"}[parts[1]]
    return axis, 1 if parts[2] == "ccw" else -1


def gripper_rotation_effect(axis: str, signed_direction: int) -> str:
    sign = "positive" if signed_direction > 0 else "negative"
    return f"{sign} right-hand-rule rotation around world {axis}"


def _validate_camera_move(direction: str, scale: str | None, raw: Any) -> ActionSpec:
    if direction not in CAMERA_MOVE_DIRECTIONS:
        raise ActionValidationError(f"Unknown camera move direction: {direction!r}")
    if direction.startswith("zoom_"):
        if scale not in {None, ""}:
            raise ActionValidationError(f"Zoom action does not take scale: {direction!r}")
        return ActionSpec(target="camera", type="move", direction=direction, raw=raw)
    if scale not in CAMERA_TRANSLATION_SCALES:
        raise ActionValidationError(f"Invalid camera move scale: {scale!r}")
    return ActionSpec(target="camera", type="move", direction=direction, scale=scale, raw=raw)


def _validate_camera_rotate(direction: str, scale: str, raw: Any) -> ActionSpec:
    if direction not in CAMERA_ROTATE_DIRECTIONS:
        raise ActionValidationError(f"Unknown camera rotation direction: {direction!r}")
    if scale not in CAMERA_ROTATION_SCALES_DEG:
        raise ActionValidationError(f"Invalid camera rotation scale: {scale!r}")
    return ActionSpec(target="camera", type="rotate", direction=direction, scale=scale, raw=raw)


def _validate_camera_look_at(mode: str, raw: Any) -> ActionSpec:
    if mode not in CAMERA_LOOK_AT_MODES:
        raise ActionValidationError(f"Unknown camera look_at mode: {mode!r}")
    return ActionSpec(target="camera", type="look_at", mode=mode, raw=raw)


def _validate_camera_view(mode: str, raw: Any) -> ActionSpec:
    if mode not in CAMERA_VIEW_MODES:
        raise ActionValidationError(f"Unknown camera view mode: {mode!r}")
    return ActionSpec(target="camera", type="view", mode=mode, raw=raw)
