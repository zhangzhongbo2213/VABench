from __future__ import annotations

from typing import Any

from .geometry import gripper_center_pose


def annotate_model_frame(frame, camera, env, width: int, height: int, modules: dict[str, Any]):
    import numpy as np
    from PIL import Image, ImageDraw
    import transforms3d as t3d

    raw = Image.fromarray(frame).convert("RGB")
    scene = Image.fromarray(resize_to_fit(frame, width, height)).convert("RGB")
    scale_x = scene.width / raw.width
    scale_y = scene.height / raw.height
    draw = ImageDraw.Draw(scene)
    font = modules["load_font"](22)

    arms = ("left", "right") if getattr(env, "active_arm", "right") == "both" else (getattr(env, "active_arm", "right"),)
    for arm in arms:
        control_pose = np.asarray(modules["gripper_pose"](env, arm), dtype=np.float64)
        center_pose = gripper_center_pose(env, modules, arm)
        control_xyz = control_pose[:3]
        center_xyz = center_pose[:3]
        rot = t3d.quaternions.quat2mat(control_pose[3:])
        prefix = arm[0].upper() + "-" if len(arms) == 2 else ""

        ee_uv = modules["project_world_points"](camera, control_xyz.reshape(1, 3))
        gc_uv = modules["project_world_points"](camera, center_xyz.reshape(1, 3))
        if ee_uv is not None and gc_uv is not None:
            ee = scaled_uv(ee_uv[0], scale_x, scale_y)
            gc = scaled_uv(gc_uv[0], scale_x, scale_y)
            draw.line((*ee, *gc), fill=(255, 255, 255), width=3)
            draw_marker(draw, ee, prefix + "EE", (255, 255, 255), font, modules, scene.size)
            draw_marker(draw, gc, prefix + "GC", (255, 255, 0), font, modules, scene.size)

        redline = center_xyz - control_xyz
        norm = float(np.linalg.norm(redline))
        redline_dir = rot[:, 0] if norm < 1e-6 else redline / norm
        draw_projected_arrow_scaled(
            draw,
            camera,
            center_xyz,
            center_xyz + redline_dir * 0.28,
            prefix + "grip center",
            (255, 0, 0),
            8,
            font,
            (12, 12),
            scene.size,
            scale_x,
            scale_y,
            modules,
        )

        for label, axis, color, offset in [
            ("rx", rot[:, 0], (255, 170, 0), (-46, -24)),
            ("ry", rot[:, 1], (0, 220, 220), (-46, 8)),
            ("rz", rot[:, 2], (255, 80, 255), (10, 20)),
        ]:
            draw_projected_arrow_scaled(
                draw,
                camera,
                control_xyz,
                control_xyz + axis * 0.085,
                prefix + label,
                color,
                4,
                font,
                offset,
                scene.size,
                scale_x,
                scale_y,
                modules,
            )

    return np.asarray(scene)


def draw_marker(draw, uv, label: str, color, font, modules: dict[str, Any], image_size: tuple[int, int]) -> None:
    u, v = uv
    radius = 9
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(0, 0, 0), width=3)
    modules["draw_outlined_text"](
        draw,
        modules["clamp_text_xy"](draw, (u + radius + 5, v - radius - 10), label, font, image_size),
        label,
        font=font,
        fill=color,
    )


def draw_projected_arrow_scaled(
    draw,
    camera,
    start,
    end,
    label: str,
    color,
    width: int,
    font,
    label_offset: tuple[int, int],
    image_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
    modules: dict[str, Any],
) -> None:
    import numpy as np

    pixels = modules["project_world_points"](camera, np.vstack([start, end]))
    if pixels is None:
        return
    start_uv = scaled_uv(pixels[0], scale_x, scale_y)
    end_uv = scaled_uv(pixels[1], scale_x, scale_y)
    modules["draw_arrow"](draw, start_uv, end_uv, color=color, width=width)
    modules["draw_outlined_text"](
        draw,
        modules["clamp_text_xy"](draw, (end_uv[0] + label_offset[0], end_uv[1] + label_offset[1]), label, font, image_size),
        label,
        font=font,
        fill=color,
    )


def resize_to_fit(frame, target_width: int, target_height: int):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(frame).convert("RGB")
    if target_width <= 0 or target_height <= 0:
        return np.asarray(image)
    scale = min(target_width / image.width, target_height / image.height)
    if scale <= 0:
        return np.asarray(image)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if new_size != image.size:
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image = image.resize(new_size, resampling)
    return np.asarray(image)


def scaled_uv(uv, scale_x: float, scale_y: float) -> tuple[float, float]:
    return float(uv[0]) * scale_x, float(uv[1]) * scale_y
