"""Offline-only task-state interventions for targeted data collection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sapien
from scipy.spatial.transform import Rotation


STATE_INTERVENTION_SCHEMA = "spatial.state_intervention.v1"


@dataclass(frozen=True)
class StateIntervention:
    mode: str
    translation_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_role: str = "grasp_target"
    keep_support_height: bool = True
    schema_version: str = STATE_INTERVENTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATE_INTERVENTION_SCHEMA:
            raise ValueError("unsupported state intervention schema")
        if self.mode not in {"target_translation", "target_rotation", "target_translation_rotation"}:
            raise ValueError(f"unsupported intervention mode: {self.mode}")
        for name in ("translation_world_m", "rotation_rpy_deg"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
        if self.keep_support_height and abs(float(self.translation_world_m[2])) > 1e-9:
            raise ValueError("support-preserving intervention cannot translate in z")
        translation_nonzero = bool(np.linalg.norm(self.translation_world_m) > 1e-9)
        rotation_nonzero = bool(np.linalg.norm(self.rotation_rpy_deg) > 1e-9)
        if self.mode == "target_translation" and rotation_nonzero:
            raise ValueError("target_translation cannot include rotation")
        if self.mode == "target_rotation" and translation_nonzero:
            raise ValueError("target_rotation cannot include translation")
        if self.mode == "target_translation_rotation" and not (
            translation_nonzero or rotation_nonzero
        ):
            raise ValueError(
                "target_translation_rotation must include translation or rotation"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StateIntervention":
        if value.get("schema_version") != STATE_INTERVENTION_SCHEMA:
            raise ValueError("unsupported state intervention schema")
        return cls(
            mode=str(value.get("mode", "")),
            translation_world_m=_vector3(value.get("translation_world_m", (0, 0, 0)), "translation_world_m"),
            rotation_rpy_deg=_vector3(value.get("rotation_rpy_deg", (0, 0, 0)), "rotation_rpy_deg"),
            target_role=str(value.get("target_role", "grasp_target")),
            keep_support_height=bool(value.get("keep_support_height", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "StateIntervention":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("state intervention JSON must contain an object")
        return cls.from_mapping(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "translation_world_m": list(self.translation_world_m),
            "rotation_rpy_deg": list(self.rotation_rpy_deg),
            "target_role": self.target_role,
            "keep_support_height": bool(self.keep_support_height),
        }

    @property
    def intervention_id(self) -> str:
        digest = hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"{self.mode}_{digest}"


def apply_state_intervention(actor: Any, intervention: StateIntervention) -> dict[str, Any]:
    if intervention.target_role != "grasp_target":
        raise ValueError(f"unsupported target role: {intervention.target_role}")
    target = actor.actor if hasattr(actor, "actor") else actor
    if not hasattr(target, "get_pose") or not hasattr(target, "set_pose"):
        raise TypeError("intervention target must expose get_pose/set_pose")
    before = target.get_pose()
    before_p = np.asarray(before.p, dtype=np.float64)
    before_q = np.asarray(before.q, dtype=np.float64)
    after_p = before_p + np.asarray(intervention.translation_world_m, dtype=np.float64)
    if intervention.keep_support_height:
        after_p[2] = before_p[2]
    old_rotation = Rotation.from_quat(_wxyz_to_xyzw(before_q))
    delta_rotation = Rotation.from_euler("xyz", intervention.rotation_rpy_deg, degrees=True)
    after_q = _xyzw_to_wxyz((delta_rotation * old_rotation).as_quat())
    target.set_pose(sapien.Pose(after_p.tolist(), after_q.tolist()))
    return {
        "schema_version": STATE_INTERVENTION_SCHEMA,
        "intervention_id": intervention.intervention_id,
        "mode": intervention.mode,
        "access": "offline_collection_only",
        "before_pose": np.concatenate([before_p, before_q]).tolist(),
        "after_pose": np.concatenate([after_p, after_q]).tolist(),
        "translation_world_m": list(intervention.translation_world_m),
        "rotation_rpy_deg": list(intervention.rotation_rpy_deg),
        "inference_visible": False,
        "runtime_policy_access": False,
    }


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in result)


def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite 4-vector")
    return np.asarray([value[1], value[2], value[3], value[0]], dtype=np.float64)


def _xyzw_to_wxyz(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("quaternion must be a finite 4-vector")
    return np.asarray([result[3], result[0], result[1], result[2]], dtype=np.float64)
