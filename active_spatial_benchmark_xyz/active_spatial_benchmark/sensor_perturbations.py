"""Deterministic inference-only RGB-D calibration/noise perturbations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np


def normalize_sensor_perturbation(
    perturbation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if perturbation is None:
        return None
    if not isinstance(perturbation, Mapping):
        raise ValueError("sensor perturbation must be a mapping")
    focal_scale = _vector(
        perturbation.get("focal_scale_xy", (1.0, 1.0)), "focal_scale_xy"
    )
    principal_offset = _vector(
        perturbation.get("principal_point_offset_px", (0.0, 0.0)),
        "principal_point_offset_px",
    )
    depth_std = float(perturbation.get("depth_gaussian_std_m", 0.0))
    dropout = float(perturbation.get("depth_dropout_probability", 0.0))
    quantization = float(perturbation.get("depth_quantization_m", 0.0))
    seed = perturbation.get("random_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sensor perturbation random_seed must be an integer")
    if np.max(np.abs(focal_scale - 1.0)) > 0.1 or np.any(focal_scale <= 0.0):
        raise ValueError("sensor focal scale must remain within 10 percent")
    if np.max(np.abs(principal_offset)) > 32.0:
        raise ValueError("principal point perturbation exceeds 32 pixels")
    if not 0.0 <= depth_std <= 0.02:
        raise ValueError("depth Gaussian noise must be in [0, 0.02] m")
    if not 0.0 <= dropout <= 0.25:
        raise ValueError("depth dropout probability must be in [0, 0.25]")
    if not 0.0 <= quantization <= 0.01:
        raise ValueError("depth quantization must be in [0, 0.01] m")
    return {
        "focal_scale_xy": focal_scale.tolist(),
        "principal_point_offset_px": principal_offset.tolist(),
        "depth_gaussian_std_m": depth_std,
        "depth_dropout_probability": dropout,
        "depth_quantization_m": quantization,
        "random_seed": seed,
    }


def apply_sensor_perturbation(
    capture: Mapping[str, Any],
    perturbation: Mapping[str, Any] | None,
    *,
    frame_id: int,
) -> dict[str, Any]:
    """Return a perturbed capture without modifying simulator-owned arrays."""

    config = normalize_sensor_perturbation(perturbation)
    result = deepcopy(dict(capture))
    if config is None:
        return result
    intrinsic = np.asarray(capture["intrinsic_cv"], dtype=np.float64).copy()
    intrinsic[0, 0] *= config["focal_scale_xy"][0]
    intrinsic[1, 1] *= config["focal_scale_xy"][1]
    intrinsic[0, 2] += config["principal_point_offset_px"][0]
    intrinsic[1, 2] += config["principal_point_offset_px"][1]
    depth_m = np.asarray(capture["depth_mm"], dtype=np.float64).copy() / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    rng = np.random.default_rng(config["random_seed"] + int(frame_id) * 104729)
    if config["depth_gaussian_std_m"] > 0.0:
        depth_m[valid] += rng.normal(
            0.0, config["depth_gaussian_std_m"], int(np.count_nonzero(valid))
        )
    if config["depth_quantization_m"] > 0.0:
        depth_m[valid] = (
            np.round(depth_m[valid] / config["depth_quantization_m"])
            * config["depth_quantization_m"]
        )
    dropped = np.zeros(depth_m.shape, dtype=bool)
    if config["depth_dropout_probability"] > 0.0:
        dropped = valid & (
            rng.random(depth_m.shape) < config["depth_dropout_probability"]
        )
        depth_m[dropped] = 0.0
    depth_m[valid & ~dropped] = np.maximum(depth_m[valid & ~dropped], 1e-6)
    result["intrinsic_cv"] = intrinsic
    result["depth_mm"] = depth_m.astype(np.float32) * 1000.0
    result["sensor_perturbation"] = config
    result["sensor_perturbation_stats"] = {
        "valid_depth_pixel_count_before": int(np.count_nonzero(valid)),
        "dropped_depth_pixel_count": int(np.count_nonzero(dropped)),
        "frame_id": int(frame_id),
    }
    return result


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2-vector")
    return result
