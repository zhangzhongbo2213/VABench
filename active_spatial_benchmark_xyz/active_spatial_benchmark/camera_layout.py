"""Configurable camera layouts independent of view-name model features.

The runtime controller may still use the historical five-view catalogue, but
the catalogue itself is now an explicit data contract. A layout can be loaded
from JSON, validated, mirrored for the active arm, bounded against the camera
workspace, and split into holdout view sets for layout-generalization tests.
View names are metadata only; pose rows carry the geometry used by scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CAMERA_LAYOUT_SCHEMA = "spatial.camera_layout.v1"
DEFAULT_CAMERA_LAYOUT_ID = "robotwin_default_five_view_v1"
DEFAULT_LAYOUT_VIEWS = {
    "topdown": {"offset_world_m": (0.0, 0.0, 0.68), "up_hint_world": (0.0, 1.0, 0.0)},
    "side": {"offset_world_m": (0.58, -0.02, 0.06), "up_hint_world": (0.0, 0.0, 1.0)},
    "front_side_45": {"offset_world_m": (0.44, -0.44, 0.18), "up_hint_world": (0.0, 0.0, 1.0)},
    "side_top_45": {"offset_world_m": (0.48, -0.02, 0.48), "up_hint_world": (0.0, 0.0, 1.0)},
    "oblique_45": {"offset_world_m": (0.42, -0.42, 0.42), "up_hint_world": (0.0, 0.0, 1.0)},
}


@dataclass(frozen=True)
class CameraLayoutView:
    offset_world_m: tuple[float, float, float]
    up_hint_world: tuple[float, float, float]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        offset = _vector3(self.offset_world_m, "offset_world_m")
        up_hint = _vector3(self.up_hint_world, "up_hint_world")
        if float(np.linalg.norm(offset)) <= 1e-9:
            raise ValueError("offset_world_m must be nonzero")
        if float(np.linalg.norm(up_hint)) <= 1e-9:
            raise ValueError("up_hint_world must be nonzero")
        tags = _string_tuple(self.tags)
        object.__setattr__(self, "offset_world_m", tuple(offset.tolist()))
        object.__setattr__(self, "up_hint_world", tuple(up_hint.tolist()))
        object.__setattr__(self, "tags", tags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_world_m": list(self.offset_world_m),
            "up_hint_world": list(self.up_hint_world),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class CameraLayout:
    layout_id: str
    views: Mapping[str, CameraLayoutView]
    mirror_x_for_left_arm: bool = True
    schema_version: str = CAMERA_LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMERA_LAYOUT_SCHEMA:
            raise ValueError("unsupported camera layout schema")
        layout_id = str(self.layout_id).strip()
        if not layout_id or "/" in layout_id or "\\" in layout_id:
            raise ValueError("layout_id must be a path-safe non-empty string")
        object.__setattr__(self, "layout_id", layout_id)
        normalized: dict[str, CameraLayoutView] = {}
        for name, view in dict(self.views).items():
            view_name = str(name).strip()
            if not view_name or "/" in view_name or "\\" in view_name:
                raise ValueError("camera view names must be path-safe")
            if not isinstance(view, CameraLayoutView):
                raise ValueError("camera layout views must be CameraLayoutView values")
            normalized[view_name] = view
        if not normalized:
            raise ValueError("camera layout requires at least one view")
        object.__setattr__(self, "views", normalized)

    @classmethod
    def default(cls) -> "CameraLayout":
        return cls(
            layout_id=DEFAULT_CAMERA_LAYOUT_ID,
            views={
                name: CameraLayoutView(
                    offset_world_m=tuple(spec["offset_world_m"]),
                    up_hint_world=tuple(spec["up_hint_world"]),
                )
                for name, spec in DEFAULT_LAYOUT_VIEWS.items()
            },
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraLayout":
        if value.get("schema_version") != CAMERA_LAYOUT_SCHEMA:
            raise ValueError("unsupported camera layout schema")
        raw_views = value.get("views")
        if not isinstance(raw_views, Mapping):
            raise ValueError("camera layout views must be an object")
        views = {}
        for name, raw in raw_views.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"camera layout view {name!r} must be an object")
            views[str(name)] = CameraLayoutView(
                offset_world_m=_tuple3(raw.get("offset_world_m"), "offset_world_m"),
                up_hint_world=_tuple3(raw.get("up_hint_world"), "up_hint_world"),
                tags=_string_tuple(raw.get("tags", ())),
            )
        return cls(
            layout_id=str(value.get("layout_id", "")),
            views=views,
            mirror_x_for_left_arm=bool(value.get("mirror_x_for_left_arm", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraLayout":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("camera layout JSON must contain an object")
        return cls.from_mapping(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layout_id": self.layout_id,
            "mirror_x_for_left_arm": bool(self.mirror_x_for_left_arm),
            "views": {name: view.as_dict() for name, view in self.views.items()},
        }

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(self.views)

    def view(self, name: str) -> CameraLayoutView:
        try:
            return self.views[str(name)]
        except KeyError as exc:
            raise ValueError(f"unsupported camera view mode: {name}") from exc

    def offset_for_arm(self, name: str, *, view_arm: str = "right") -> np.ndarray:
        if view_arm not in {"left", "right", "both"}:
            raise ValueError(f"unsupported camera view arm: {view_arm}")
        offset = np.asarray(self.view(name).offset_world_m, dtype=np.float64)
        if view_arm == "left" and self.mirror_x_for_left_arm:
            offset = offset.copy()
            offset[0] *= -1.0
        return offset

    def pose_for_view(
        self,
        name: str,
        *,
        finger_center_world: Sequence[float],
        view_arm: str = "right",
        camera_bounds: Sequence[Sequence[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = _vector3(finger_center_world, "finger center")
        position = center + self.offset_for_arm(name, view_arm=view_arm)
        if camera_bounds is not None:
            bounds = np.asarray(camera_bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError("camera bounds must be a finite 2x3 array")
            if np.any(bounds[0] > bounds[1]):
                raise ValueError("camera lower bounds must not exceed upper bounds")
            position = np.clip(position, bounds[0], bounds[1])
        return position, _unit(center - position, "camera forward direction")

    def split(self, *, holdout_views: Sequence[str]) -> dict[str, Any]:
        holdout = tuple(str(name) for name in holdout_views)
        if len(set(holdout)) != len(holdout):
            raise ValueError("holdout views must be unique")
        unknown = sorted(set(holdout) - set(self.views))
        if unknown:
            raise ValueError(f"holdout contains unknown views: {unknown}")
        holdout_set = set(holdout)
        return {
            "layout_id": self.layout_id,
            "train_views": [name for name in self.views if name not in holdout_set],
            "holdout_views": list(holdout),
            "split_rule": "view_pose_holdout_not_view_name_feature_training",
        }

    def candidate_rows(
        self,
        *,
        finger_center_world: Sequence[float],
        view_arm: str = "right",
        camera_bounds: Sequence[Sequence[float]] | None = None,
        view_names: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        names = self.view_names if view_names is None else tuple(view_names)
        rows = []
        for name in names:
            position, direction = self.pose_for_view(
                name,
                finger_center_world=finger_center_world,
                view_arm=view_arm,
                camera_bounds=camera_bounds,
            )
            rows.append(
                {
                    "view": name,
                    "camera_position_world": position.tolist(),
                    "view_direction_world": direction.tolist(),
                    "camera_position_source": "declared_camera_layout_after_bounds_clamp",
                    "camera_pose_source": self.layout_id,
                    "camera_reachable": True,
                    "camera_safe": True,
                }
            )
        return rows


def _tuple3(value: Any, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if len(values) != 3 or not all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite numbers")
    if np.linalg.norm(values) <= 1e-9 and name == "up_hint_world":
        raise ValueError("up_hint_world must be nonzero")
    return values


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence):
        raise ValueError("camera layout tags must be a sequence")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(set(result)) != len(result):
        raise ValueError("camera layout tags must be unique")
    return result


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    result = _vector3(value, name)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return result / norm
