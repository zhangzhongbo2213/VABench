"""Task-conditioned sparse 3D grasp-candidate graphs.

The module is deliberately independent from RoboTwin actors.  Training code may
construct :class:`OrientedObjectGeometry` from simulator truth, while inference
code can construct the same schema from learned RGB-D estimates.  Provenance is
kept on every region and candidate so oracle geometry cannot silently become a
model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .spatial_graph import obb_support_radius, unit


GRASP_CANDIDATE_GRAPH_SCHEMA = "spatial.grasp_candidate_graph.v2"
GRASP_INTENT_SCHEMA = "spatial.grasp_intent.v2"
PART_SHAPES = {
    "unknown",
    "slender_cylinder",
    "broad_cylinder",
    "box",
    "handle",
    "flat",
    "irregular",
}
SYMMETRY_CLASSES = {"unknown", "continuous", "two_fold", "four_fold", "asymmetric"}
TOLERANCE_LEVELS = {"unknown", "strict", "moderate", "permissive"}


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_tuple(
    value: Any, *, field_name: str, required: bool = False
) -> tuple[str, ...]:
    if value is None:
        values: Sequence[Any] = ()
    elif isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ValueError(f"{field_name} must be a string or a sequence of strings")
    result = tuple(str(item).strip() for item in values if str(item).strip())
    if required and not result:
        raise ValueError(f"{field_name} must contain at least one role")
    return result


def _vector3(value: Any, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must be a finite 3-vector")
    return vector


def _covariance3(value: Any, *, field_name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{field_name} must be a finite 3x3 matrix")
    if not np.allclose(covariance, covariance.T, atol=1e-9):
        raise ValueError(f"{field_name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
        raise ValueError(f"{field_name} must be positive semidefinite")
    return covariance


def _round_vector(value: np.ndarray) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), 7).tolist()


def _round_matrix(value: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(value, dtype=np.float64), 9).tolist()


@dataclass(frozen=True)
class GraspAffordanceProfile:
    part_shape: str = "unknown"
    symmetry_class: str = "unknown"
    centering_tolerance: str = "unknown"
    vertical_tolerance: str = "unknown"
    avoid_ends: bool | None = None
    requires_bilateral_contact: bool | None = None
    requires_dual_arm: bool | None = None

    def __post_init__(self) -> None:
        if self.part_shape not in PART_SHAPES:
            raise ValueError(f"unsupported grasp part_shape {self.part_shape!r}")
        if self.symmetry_class not in SYMMETRY_CLASSES:
            raise ValueError(
                f"unsupported grasp symmetry_class {self.symmetry_class!r}"
            )
        for name in ("centering_tolerance", "vertical_tolerance"):
            value = getattr(self, name)
            if value not in TOLERANCE_LEVELS:
                raise ValueError(f"unsupported grasp {name} {value!r}")
        for name in (
            "avoid_ends",
            "requires_bilateral_contact",
            "requires_dual_arm",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"grasp affordance {name} must be bool or null")

    def as_dict(self) -> dict[str, Any]:
        return {
            "part_shape": self.part_shape,
            "symmetry_class": self.symmetry_class,
            "centering_tolerance": self.centering_tolerance,
            "vertical_tolerance": self.vertical_tolerance,
            "avoid_ends": self.avoid_ends,
            "requires_bilateral_contact": self.requires_bilateral_contact,
            "requires_dual_arm": self.requires_dual_arm,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GraspAffordanceProfile":
        values = dict(value or {})
        return cls(
            part_shape=str(values.get("part_shape", "unknown")),
            symmetry_class=str(values.get("symmetry_class", "unknown")),
            centering_tolerance=str(values.get("centering_tolerance", "unknown")),
            vertical_tolerance=str(values.get("vertical_tolerance", "unknown")),
            avoid_ends=values.get("avoid_ends"),
            requires_bilateral_contact=values.get("requires_bilateral_contact"),
            requires_dual_arm=values.get("requires_dual_arm"),
        )


@dataclass(frozen=True)
class GraspIntent:
    """Semantic task constraints supplied by a VLM or expert summary."""

    target: str
    task_goal: str
    preferred_roles: tuple[str, ...]
    task_stage: str = "initial_grasp"
    post_grasp_goal: str = "unspecified"
    functional_constraints: tuple[str, ...] = ()
    required_arms: int | None = None
    semantic_ambiguity: float | None = None
    affordance_profile: GraspAffordanceProfile = field(
        default_factory=GraspAffordanceProfile
    )
    avoided_roles: tuple[str, ...] = ()
    contact_pattern: str = "opposed_sides"
    approach_relation: str = "unspecified"
    closing_axis_relation: str = "unspecified"
    grasp_depth_rule: str = "unspecified"
    completion_rule: str = "unspecified"
    active_arm: str = "right"
    natural_language_constraints: tuple[str, ...] = ()
    source: str = "vlm"
    source_model: str | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _text(self.target, field_name="target"))
        object.__setattr__(
            self, "task_goal", _text(self.task_goal, field_name="task_goal")
        )
        object.__setattr__(
            self, "task_stage", _text(self.task_stage, field_name="task_stage")
        )
        object.__setattr__(
            self,
            "post_grasp_goal",
            _text(self.post_grasp_goal, field_name="post_grasp_goal"),
        )
        object.__setattr__(
            self,
            "preferred_roles",
            _string_tuple(
                self.preferred_roles, field_name="preferred_roles", required=True
            ),
        )
        object.__setattr__(
            self,
            "functional_constraints",
            _string_tuple(
                self.functional_constraints, field_name="functional_constraints"
            ),
        )
        if self.required_arms is not None and (
            isinstance(self.required_arms, bool)
            or int(self.required_arms) not in (1, 2)
        ):
            raise ValueError("required_arms must be 1, 2, or null")
        if self.required_arms is not None:
            object.__setattr__(self, "required_arms", int(self.required_arms))
        if (
            self.semantic_ambiguity is not None
            and not 0.0 <= float(self.semantic_ambiguity) <= 1.0
        ):
            raise ValueError("semantic_ambiguity must be in [0, 1] or null")
        if self.semantic_ambiguity is not None:
            object.__setattr__(
                self, "semantic_ambiguity", float(self.semantic_ambiguity)
            )
        object.__setattr__(
            self,
            "avoided_roles",
            _string_tuple(self.avoided_roles, field_name="avoided_roles"),
        )
        if isinstance(self.affordance_profile, Mapping):
            object.__setattr__(
                self,
                "affordance_profile",
                GraspAffordanceProfile.from_mapping(self.affordance_profile),
            )
        elif not isinstance(self.affordance_profile, GraspAffordanceProfile):
            raise ValueError(
                "affordance_profile must be a mapping or GraspAffordanceProfile"
            )
        object.__setattr__(
            self,
            "natural_language_constraints",
            _string_tuple(
                self.natural_language_constraints,
                field_name="natural_language_constraints",
            ),
        )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRASP_INTENT_SCHEMA,
            "target": self.target,
            "task_goal": self.task_goal,
            "task_stage": self.task_stage,
            "post_grasp_goal": self.post_grasp_goal,
            "functional_constraints": list(self.functional_constraints),
            "required_arms": self.required_arms,
            "semantic_ambiguity": (
                round(float(self.semantic_ambiguity), 6)
                if self.semantic_ambiguity is not None
                else None
            ),
            "preferred_roles": list(self.preferred_roles),
            "affordance_profile": self.affordance_profile.as_dict(),
            "avoided_roles": list(self.avoided_roles),
            "contact_pattern": self.contact_pattern,
            "approach_relation": self.approach_relation,
            "closing_axis_relation": self.closing_axis_relation,
            "grasp_depth_rule": self.grasp_depth_rule,
            "completion_rule": self.completion_rule,
            "active_arm": self.active_arm,
            "natural_language_constraints": list(self.natural_language_constraints),
            "source": self.source,
            "source_model": self.source_model,
            "confidence": round(float(self.confidence), 6),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraspIntent":
        return cls(
            target=value.get("target", "target object"),
            task_goal=value.get("task_goal", "grasp"),
            task_stage=value.get("task_stage", "initial_grasp"),
            post_grasp_goal=value.get("post_grasp_goal", "unspecified"),
            functional_constraints=_string_tuple(
                value.get("functional_constraints"),
                field_name="functional_constraints",
            ),
            required_arms=value.get("required_arms"),
            semantic_ambiguity=value.get("semantic_ambiguity"),
            preferred_roles=_string_tuple(
                value.get("preferred_roles", value.get("preferred_role")),
                field_name="preferred_roles",
                required=True,
            ),
            affordance_profile=GraspAffordanceProfile.from_mapping(
                value.get("affordance_profile")
            ),
            avoided_roles=_string_tuple(
                value.get("avoided_roles", value.get("avoid_roles")),
                field_name="avoided_roles",
            ),
            contact_pattern=str(value.get("contact_pattern", "opposed_sides")),
            approach_relation=str(value.get("approach_relation", "unspecified")),
            closing_axis_relation=str(
                value.get("closing_axis_relation", "unspecified")
            ),
            grasp_depth_rule=str(value.get("grasp_depth_rule", "unspecified")),
            completion_rule=str(value.get("completion_rule", "unspecified")),
            active_arm=str(value.get("active_arm", "right")),
            natural_language_constraints=_string_tuple(
                value.get("natural_language_constraints"),
                field_name="natural_language_constraints",
            ),
            source=str(value.get("source", "vlm")),
            source_model=(
                str(value["source_model"]).strip()
                if value.get("source_model")
                else None
            ),
            confidence=float(value.get("confidence", 1.0)),
        )


def compile_expert_summary_to_intent(
    summary: Mapping[str, Any],
    *,
    target: str,
    task_goal: str = "grasp",
) -> GraspIntent:
    """Compile existing model-specific expert learning into a spatial intent.

    The compiler preserves free-text constraints instead of pretending to infer
    exact metric geometry from prose.  Metric grounding remains the spatial
    tool's responsibility.
    """

    required = (
        "grasp_object_part",
        "grasp_region",
        "grasp_height_or_depth",
        "finger_placement",
        "approach_strategy",
        "posture_or_rotation",
        "pre_close_checks",
        "test_lift_rule",
    )
    missing = [key for key in required if not summary.get(key)]
    if missing:
        raise ValueError("expert summary is missing fields: " + ", ".join(missing))
    checks = _string_tuple(
        summary["pre_close_checks"], field_name="pre_close_checks", required=True
    )
    safety = _string_tuple(summary.get("safety_notes"), field_name="safety_notes")
    functional = _string_tuple(
        summary.get("functional_constraints"),
        field_name="functional_constraints",
    )
    constraints = (
        _text(summary["grasp_region"], field_name="grasp_region"),
        _text(summary["finger_placement"], field_name="finger_placement"),
        _text(summary["posture_or_rotation"], field_name="posture_or_rotation"),
        *checks,
        *safety,
        *functional,
    )
    profile = GraspAffordanceProfile.from_mapping(summary.get("affordance_profile"))
    required_arms = summary.get("required_arms")
    if required_arms is None and profile.requires_dual_arm is not None:
        required_arms = 2 if profile.requires_dual_arm else 1
    return GraspIntent(
        target=target,
        task_goal=task_goal,
        task_stage=str(summary.get("task_stage") or "initial_grasp"),
        post_grasp_goal=str(
            summary.get("post_grasp_goal")
            or summary.get("release_or_completion_rule")
            or summary["test_lift_rule"]
        ),
        functional_constraints=functional,
        required_arms=required_arms,
        semantic_ambiguity=summary.get("semantic_ambiguity"),
        preferred_roles=(
            _text(summary["grasp_object_part"], field_name="grasp_object_part"),
        ),
        affordance_profile=profile,
        avoided_roles=_string_tuple(
            summary.get("avoid_roles"), field_name="avoid_roles"
        ),
        contact_pattern="opposed_sides",
        approach_relation=_text(
            summary["approach_strategy"], field_name="approach_strategy"
        ),
        closing_axis_relation=_text(
            summary["finger_placement"], field_name="finger_placement"
        ),
        grasp_depth_rule=_text(
            summary["grasp_height_or_depth"], field_name="grasp_height_or_depth"
        ),
        completion_rule=_text(summary["test_lift_rule"], field_name="test_lift_rule"),
        active_arm=str(summary.get("active_arm", "right")),
        natural_language_constraints=constraints,
        source="expert_summary",
        source_model=(
            str(summary["summary_model"]).strip()
            if summary.get("summary_model")
            else None
        ),
        confidence=float(summary.get("intent_confidence", 1.0)),
    )


@dataclass(frozen=True)
class OrientedObjectGeometry:
    object_id: str
    center_world_m: np.ndarray
    rotation_world: np.ndarray
    half_extents_m: np.ndarray
    support_z_m: float | None = None
    source: str = "oracle_object_obb"
    access: str = "oracle/training_only"

    def __post_init__(self) -> None:
        center = _vector3(self.center_world_m, field_name="center_world_m")
        rotation = np.asarray(self.rotation_world, dtype=np.float64)
        extents = _vector3(self.half_extents_m, field_name="half_extents_m")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_world must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation_world must be orthonormal")
        if np.any(extents <= 0.0):
            raise ValueError("half_extents_m must be positive")
        object.__setattr__(
            self, "object_id", _text(self.object_id, field_name="object_id")
        )
        object.__setattr__(self, "center_world_m", center)
        object.__setattr__(self, "rotation_world", rotation)
        object.__setattr__(self, "half_extents_m", extents)


@dataclass(frozen=True)
class SemanticGraspRegion:
    region_id: str
    role: str
    center_world_m: np.ndarray
    longitudinal_axis_world: np.ndarray
    half_length_m: float
    position_covariance_m2: np.ndarray
    confidence: float
    source: str
    access: str = "inference_visible"
    radial_half_extent_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "region_id", _text(self.region_id, field_name="region_id")
        )
        object.__setattr__(self, "role", _text(self.role, field_name="role"))
        object.__setattr__(
            self,
            "center_world_m",
            _vector3(self.center_world_m, field_name="center_world_m"),
        )
        object.__setattr__(
            self,
            "longitudinal_axis_world",
            unit(
                _vector3(
                    self.longitudinal_axis_world, field_name="longitudinal_axis_world"
                )
            ),
        )
        object.__setattr__(
            self,
            "position_covariance_m2",
            _covariance3(
                self.position_covariance_m2, field_name="position_covariance_m2"
            ),
        )
        if float(self.half_length_m) <= 0.0:
            raise ValueError("half_length_m must be positive")
        if (
            self.radial_half_extent_m is not None
            and float(self.radial_half_extent_m) <= 0.0
        ):
            raise ValueError("radial_half_extent_m must be positive when provided")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "role": self.role,
            "position_mean_world_m": _round_vector(self.center_world_m),
            "position_covariance_m2": _round_matrix(self.position_covariance_m2),
            "longitudinal_axis_world": _round_vector(self.longitudinal_axis_world),
            "half_length_m": round(float(self.half_length_m), 7),
            "radial_half_extent_m": (
                round(float(self.radial_half_extent_m), 7)
                if self.radial_half_extent_m is not None
                else None
            ),
            "confidence": round(float(self.confidence), 6),
            "source": self.source,
            "access": self.access,
        }


@dataclass(frozen=True)
class CandidateCheck:
    name: str
    probability: float | None
    hard_constraint: bool
    source: str
    measurement: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> str:
        if self.probability is None:
            return "unknown"
        return "pass" if self.probability >= 0.5 else "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "probability": (
                round(float(self.probability), 6)
                if self.probability is not None
                else None
            ),
            "hard_constraint": self.hard_constraint,
            "source": self.source,
            "measurement": dict(self.measurement),
        }


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    target_id: str
    region_id: str
    semantic_role: str
    center_world_m: np.ndarray
    left_contact_world_m: np.ndarray
    right_contact_world_m: np.ndarray
    approach_axis_world: np.ndarray
    closing_axis_world: np.ndarray
    pregrasp_center_world_m: np.ndarray
    opening_width_m: float
    generation_parameters: Mapping[str, Any]
    position_covariance_m2: np.ndarray
    orientation_covariance_rad2: np.ndarray
    checks: tuple[CandidateCheck, ...]
    score: float
    score_confidence: float
    source: str
    access: str

    @property
    def eligible(self) -> bool:
        return not any(
            check.hard_constraint
            and check.probability is not None
            and check.probability < 0.5
            for check in self.checks
        )

    @property
    def failed_constraints(self) -> list[str]:
        return [check.name for check in self.checks if check.state == "fail"]

    @property
    def unknown_constraints(self) -> list[str]:
        return [check.name for check in self.checks if check.state == "unknown"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "target_id": self.target_id,
            "region_id": self.region_id,
            "semantic_role": self.semantic_role,
            "frame": {
                "center_world_m": _round_vector(self.center_world_m),
                "left_contact_world_m": _round_vector(self.left_contact_world_m),
                "right_contact_world_m": _round_vector(self.right_contact_world_m),
                "approach_axis_world": _round_vector(self.approach_axis_world),
                "closing_axis_world": _round_vector(self.closing_axis_world),
                "pregrasp_center_world_m": _round_vector(self.pregrasp_center_world_m),
                "opening_width_m": round(float(self.opening_width_m), 7),
            },
            "generation_parameters": dict(self.generation_parameters),
            "position_covariance_m2": _round_matrix(self.position_covariance_m2),
            "orientation_covariance_rad2": _round_matrix(
                self.orientation_covariance_rad2
            ),
            "checks": {check.name: check.as_dict() for check in self.checks},
            "eligible": self.eligible,
            "failed_constraints": self.failed_constraints,
            "unknown_constraints": self.unknown_constraints,
            "score": round(float(self.score), 6),
            "score_confidence": round(float(self.score_confidence), 6),
            "predicted_execution_success": next(
                (
                    round(float(check.probability), 6)
                    for check in self.checks
                    if check.name == "predicted_execution_success"
                    and check.probability is not None
                ),
                None,
            ),
            "source": self.source,
            "access": self.access,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraspCandidate":
        frame = value.get("frame", {})
        checks_value = value.get("checks", {})
        checks = tuple(
            CandidateCheck(
                name=str(name),
                probability=(
                    float(check["probability"])
                    if check.get("probability") is not None
                    else None
                ),
                hard_constraint=bool(check.get("hard_constraint", False)),
                source=str(check.get("source", "unknown")),
                measurement=dict(check.get("measurement", {})),
            )
            for name, check in checks_value.items()
        )
        return cls(
            candidate_id=_text(value.get("id"), field_name="candidate id"),
            target_id=_text(value.get("target_id"), field_name="candidate target id"),
            region_id=_text(value.get("region_id"), field_name="candidate region id"),
            semantic_role=_text(
                value.get("semantic_role"), field_name="candidate semantic role"
            ),
            center_world_m=_vector3(
                frame.get("center_world_m"), field_name="candidate center"
            ),
            left_contact_world_m=_vector3(
                frame.get("left_contact_world_m"), field_name="left contact"
            ),
            right_contact_world_m=_vector3(
                frame.get("right_contact_world_m"), field_name="right contact"
            ),
            approach_axis_world=_vector3(
                frame.get("approach_axis_world"), field_name="approach axis"
            ),
            closing_axis_world=_vector3(
                frame.get("closing_axis_world"), field_name="closing axis"
            ),
            pregrasp_center_world_m=_vector3(
                frame.get("pregrasp_center_world_m"), field_name="pregrasp center"
            ),
            opening_width_m=float(frame.get("opening_width_m")),
            generation_parameters=dict(value.get("generation_parameters", {})),
            position_covariance_m2=_covariance3(
                value.get("position_covariance_m2"),
                field_name="candidate position covariance",
            ),
            orientation_covariance_rad2=_covariance3(
                value.get("orientation_covariance_rad2"),
                field_name="candidate orientation covariance",
            ),
            checks=checks,
            score=float(value.get("score", 0.0)),
            score_confidence=float(value.get("score_confidence", 0.0)),
            source=str(value.get("source", "unknown")),
            access=str(value.get("access", "unknown")),
        )


@dataclass(frozen=True)
class CandidateGenerationConfig:
    along_axis_fractions: tuple[float, ...] = (0.0, -0.35, 0.35)
    vertical_offsets_m: tuple[float, ...] = (0.0, 0.008, 0.015)
    contact_clearance_m: float = 0.006
    pregrasp_distance_m: float = 0.09
    min_opening_width_m: float = 0.01
    max_opening_width_m: float = 0.09
    minimum_support_clearance_m: float = 0.018
    max_candidates: int | None = 5


CandidateEvaluator = Callable[[Mapping[str, Any]], float | bool | None]


def scale_aware_probe_generation_config(
    geometry: OrientedObjectGeometry,
    *,
    max_candidates: int | None = None,
) -> CandidateGenerationConfig:
    """Build a within-task boundary sweep scaled by observed object height."""

    vertical_radius = obb_support_radius(
        geometry.rotation_world,
        geometry.half_extents_m,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    medium_vertical = max(0.04, 1.6 * vertical_radius)
    severe_vertical = max(0.08, 3.2 * vertical_radius)
    return CandidateGenerationConfig(
        along_axis_fractions=(0.0, -0.35, 0.35, -1.1, 1.1, -1.8, 1.8),
        vertical_offsets_m=(
            0.0,
            0.015,
            round(float(medium_vertical), 7),
            round(float(severe_vertical), 7),
        ),
        max_candidates=max_candidates,
    )


def semantic_region_from_obb(
    intent: GraspIntent,
    geometry: OrientedObjectGeometry,
    *,
    confidence: float = 1.0,
) -> SemanticGraspRegion:
    major_index = int(np.argmax(geometry.half_extents_m))
    major_axis = geometry.rotation_world[:, major_index]
    variance = np.maximum(geometry.half_extents_m * 0.04, 0.0015) ** 2
    covariance = geometry.rotation_world @ np.diag(variance) @ geometry.rotation_world.T
    return SemanticGraspRegion(
        region_id=f"{geometry.object_id}.region.preferred_0",
        role=intent.preferred_roles[0],
        center_world_m=geometry.center_world_m,
        longitudinal_axis_world=major_axis,
        half_length_m=float(0.65 * geometry.half_extents_m[major_index]),
        position_covariance_m2=covariance,
        confidence=min(float(intent.confidence), float(confidence)),
        source=geometry.source,
        access=geometry.access,
    )


def geometry_and_region_from_sparse_graph(
    intent: GraspIntent,
    graph: Mapping[str, Any],
    *,
    object_id: str,
    radial_half_extent_m: float,
    region_confidence: float | None = None,
) -> tuple[OrientedObjectGeometry, SemanticGraspRegion]:
    """Lift learned object center/axis nodes into the candidate-generator schema.

    The current pen perception network does not estimate object thickness.  The
    caller must therefore provide a named radial prior; provenance in the output
    makes that approximation visible to the VLM and evaluation logs.
    """

    if radial_half_extent_m <= 0.0:
        raise ValueError("radial_half_extent_m must be positive")
    nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    required = ("object.center", "object.axis_start", "object.axis_end")
    missing = [node_id for node_id in required if node_id not in nodes]
    if missing:
        raise ValueError("sparse graph is missing object nodes: " + ", ".join(missing))
    center_node = nodes["object.center"]
    center = _vector3(
        center_node.get("position_mean_world_m"), field_name="object.center"
    )
    axis_start = _vector3(
        nodes["object.axis_start"].get("position_mean_world_m"),
        field_name="object.axis_start",
    )
    axis_end = _vector3(
        nodes["object.axis_end"].get("position_mean_world_m"),
        field_name="object.axis_end",
    )
    axis_delta = axis_end - axis_start
    half_length = 0.5 * float(np.linalg.norm(axis_delta))
    if half_length <= 1e-5:
        raise ValueError("sparse object axis endpoints are degenerate")
    longitudinal = unit(axis_delta)
    axes = graph.get("query_axes", {})
    closing_hint = np.asarray(
        axes.get("closing_axis_world", [1.0, 0.0, 0.0]), dtype=np.float64
    )
    closing = closing_hint - float(np.dot(closing_hint, longitudinal)) * longitudinal
    if float(np.linalg.norm(closing)) <= 1e-6:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(reference, longitudinal))) > 0.9:
            reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        closing = np.cross(reference, longitudinal)
    closing = unit(closing)
    transverse = unit(np.cross(closing, longitudinal))
    rotation = np.column_stack([closing, longitudinal, transverse])
    source = "learned_sparse_graph+radial_extent_prior"
    geometry = OrientedObjectGeometry(
        object_id=object_id,
        center_world_m=center,
        rotation_world=rotation,
        half_extents_m=np.array(
            [radial_half_extent_m, half_length, radial_half_extent_m],
            dtype=np.float64,
        ),
        support_z_m=None,
        source=source,
        access="inference_visible",
    )
    covariance = _covariance3(
        center_node.get("position_covariance_m2", np.eye(3) * 0.0004),
        field_name="object.center covariance",
    )
    if region_confidence is None:
        visibility = float(center_node.get("visibility", 0.5))
        observation_count = int(center_node.get("observation_count", 1))
        region_confidence = float(
            np.clip(
                0.5 * visibility + 0.5 * min(1.0, observation_count / 2.0), 0.0, 1.0
            )
        )
    region = SemanticGraspRegion(
        region_id=f"{object_id}.region.preferred_0",
        role=intent.preferred_roles[0],
        center_world_m=center,
        longitudinal_axis_world=longitudinal,
        half_length_m=0.65 * half_length,
        position_covariance_m2=covariance,
        confidence=min(float(intent.confidence), float(region_confidence)),
        source=source,
        access="inference_visible",
        radial_half_extent_m=radial_half_extent_m,
    )
    return geometry, region


def semantic_regions_from_sparse_graph(
    intent: GraspIntent,
    graph: Mapping[str, Any],
    *,
    object_id: str,
) -> list[SemanticGraspRegion]:
    """Read VLM-role-matched local part geometry from an inference graph.

    This is the integration boundary for handle/body/neck region detectors.  It
    does not infer part geometry from free text and rejects oracle graph access.
    """

    if graph.get("access") != "inference_visible":
        raise ValueError(
            "semantic part regions must come from an inference-visible graph"
        )
    preferred = {role.casefold() for role in intent.preferred_roles}
    regions: list[SemanticGraspRegion] = []
    for node in graph.get("nodes", ()):
        if node.get("node_type") != "semantic_part_region":
            continue
        if str(node.get("entity_id", object_id)) != object_id:
            continue
        role = _text(node.get("semantic_type"), field_name="semantic part role")
        if role.casefold() not in preferred:
            continue
        access = str(node.get("access", graph.get("access")))
        if access != "inference_visible":
            raise ValueError("semantic part region nodes must be inference-visible")
        covariance = _covariance3(
            node.get("position_covariance_m2", np.eye(3) * 0.0004),
            field_name="semantic part covariance",
        )
        regions.append(
            SemanticGraspRegion(
                region_id=_text(node.get("id"), field_name="semantic part id"),
                role=role,
                center_world_m=_vector3(
                    node.get("position_mean_world_m"),
                    field_name="semantic part center",
                ),
                longitudinal_axis_world=_vector3(
                    node.get("longitudinal_axis_world"),
                    field_name="semantic part longitudinal axis",
                ),
                half_length_m=float(node.get("half_length_m")),
                radial_half_extent_m=float(node.get("radial_half_extent_m")),
                position_covariance_m2=covariance,
                confidence=float(node.get("confidence", 0.5)),
                source=str(node.get("source", "learned_semantic_part_region")),
                access=access,
            )
        )
    if not regions:
        raise ValueError(
            "sparse graph has no inference-visible semantic part region matching "
            + ", ".join(intent.preferred_roles)
        )
    return regions


def geometry_from_semantic_regions(
    regions: Sequence[SemanticGraspRegion],
    *,
    object_id: str,
    approach_axis_world: Iterable[float] = (0.0, 0.0, -1.0),
) -> OrientedObjectGeometry:
    """Create an inference-visible local geometry proxy without an object OBB."""

    if not regions:
        raise ValueError("at least one semantic region is required")
    if any(region.access != "inference_visible" for region in regions):
        raise ValueError("local geometry proxy requires inference-visible regions")
    longitudinal = unit(
        np.mean([region.longitudinal_axis_world for region in regions], axis=0)
    )
    approach = unit(_vector3(approach_axis_world, field_name="approach_axis_world"))
    closing = np.cross(approach, longitudinal)
    if float(np.linalg.norm(closing)) <= 1e-6:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, longitudinal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        closing = np.cross(reference, longitudinal)
    closing = unit(closing)
    transverse = unit(np.cross(closing, longitudinal))
    radial = max(float(region.radial_half_extent_m or 0.01) for region in regions)
    center = np.mean([region.center_world_m for region in regions], axis=0)
    projected_spread = max(
        abs(float(np.dot(region.center_world_m - center, longitudinal)))
        + float(region.half_length_m)
        for region in regions
    )
    return OrientedObjectGeometry(
        object_id=object_id,
        center_world_m=center,
        rotation_world=np.column_stack([closing, longitudinal, transverse]),
        half_extents_m=np.array([radial, projected_spread, radial]),
        support_z_m=None,
        source="derived_from_learned_semantic_part_regions",
        access="inference_visible",
    )


def fuse_semantic_part_region_nodes(
    observations: Sequence[Mapping[str, Any]],
    *,
    maximum_mahalanobis_squared: float = 11.345,
    maximum_axis_angle_deg: float = 45.0,
    conflict_covariance_inflation: float = 0.5,
) -> dict[str, Any]:
    """Fuse same-role region observations while preserving evidence provenance."""

    if not observations:
        raise ValueError("semantic region fusion requires at least one observation")
    first = observations[0]
    identity = (
        str(first.get("id")),
        str(first.get("entity_id")),
        str(first.get("semantic_type")),
    )
    for observation in observations:
        current = (
            str(observation.get("id")),
            str(observation.get("entity_id")),
            str(observation.get("semantic_type")),
        )
        if current != identity:
            raise ValueError(
                "semantic region observations must share id, entity and role"
            )
        if observation.get("access") != "inference_visible":
            raise ValueError(
                "semantic region fusion rejects non-inference-visible evidence"
            )
    means = [
        _vector3(value.get("position_mean_world_m"), field_name="region center")
        for value in observations
    ]
    covariances = [
        _covariance3(
            value.get("position_covariance_m2"), field_name="region covariance"
        )
        for value in observations
    ]
    axes = [
        unit(_vector3(value.get("longitudinal_axis_world"), field_name="region axis"))
        for value in observations
    ]
    accepted_indices = [0]
    axis_accepted_indices = [0]
    rejection_by_index: dict[int, dict[str, Any]] = {}
    axis_rejection_by_index: dict[int, dict[str, Any]] = {}
    running_mean = means[0]
    running_covariance = covariances[0]
    running_axis = axes[0]
    for index in range(1, len(observations)):
        difference = means[index] - running_mean
        innovation = running_covariance + covariances[index]
        mahalanobis_squared = float(
            difference.T @ np.linalg.pinv(innovation) @ difference
        )
        axis_cosine = float(np.clip(abs(np.dot(axes[index], running_axis)), 0.0, 1.0))
        axis_angle_deg = math.degrees(math.acos(axis_cosine))
        center_conflict = mahalanobis_squared > maximum_mahalanobis_squared
        axis_conflict = axis_angle_deg > maximum_axis_angle_deg
        if center_conflict:
            reasons = ["center_mahalanobis_conflict"]
            if axis_conflict:
                reasons.append("axis_angle_conflict")
            rejection_by_index[index] = {
                "reasons": reasons,
                "mahalanobis_squared": round(mahalanobis_squared, 6),
                "axis_angle_deg_sign_invariant": round(axis_angle_deg, 6),
            }
            continue
        accepted_indices.append(index)
        running_information = np.linalg.pinv(running_covariance)
        new_information = np.linalg.pinv(covariances[index])
        running_covariance = np.linalg.pinv(running_information + new_information)
        running_mean = running_covariance @ (
            running_information @ running_mean + new_information @ means[index]
        )
        if axis_conflict:
            axis_rejection_by_index[index] = {
                "reasons": ["axis_angle_conflict"],
                "mahalanobis_squared": round(mahalanobis_squared, 6),
                "axis_angle_deg_sign_invariant": round(axis_angle_deg, 6),
            }
        else:
            axis_accepted_indices.append(index)
            aligned_axis = (
                axes[index]
                if np.dot(axes[index], running_axis) >= 0.0
                else -axes[index]
            )
            running_axis = unit(running_axis + aligned_axis)

    accepted_observations = [observations[index] for index in accepted_indices]
    accepted_means = [means[index] for index in accepted_indices]
    accepted_covariances = [covariances[index] for index in accepted_indices]
    information = [np.linalg.pinv(covariance) for covariance in accepted_covariances]
    fused_covariance = np.linalg.pinv(np.sum(information, axis=0))
    fused_mean = fused_covariance @ np.sum(
        [matrix @ mean for matrix, mean in zip(information, accepted_means)], axis=0
    )
    if rejection_by_index:
        fused_covariance *= 1.0 + conflict_covariance_inflation * len(
            rejection_by_index
        )
    weights = np.asarray(
        [
            max(1e-6, float(value.get("confidence", 0.5)))
            * max(1e-6, float(value.get("visibility", 0.5)))
            / max(float(np.trace(covariance)), 1e-9)
            for value, covariance in zip(accepted_observations, accepted_covariances)
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    axis_observations = [observations[index] for index in axis_accepted_indices]
    axis_covariances = [covariances[index] for index in axis_accepted_indices]
    axis_weights = np.asarray(
        [
            max(1e-6, float(value.get("confidence", 0.5)))
            * max(1e-6, float(value.get("visibility", 0.5)))
            / max(float(np.trace(covariance)), 1e-9)
            for value, covariance in zip(axis_observations, axis_covariances)
        ],
        dtype=np.float64,
    )
    axis_weights /= axis_weights.sum()
    reference_axis = unit(
        _vector3(
            axis_observations[0].get("longitudinal_axis_world"),
            field_name="region axis",
        )
    )
    aligned_axes = []
    for observation in axis_observations:
        axis = unit(
            _vector3(
                observation.get("longitudinal_axis_world"),
                field_name="region axis",
            )
        )
        aligned_axes.append(axis if np.dot(axis, reference_axis) >= 0.0 else -axis)
    fused_axis = unit(np.sum(np.asarray(aligned_axes) * axis_weights[:, None], axis=0))
    half_length = float(
        np.sum(
            weights
            * np.asarray(
                [float(value.get("half_length_m")) for value in accepted_observations]
            )
        )
    )
    radial_extent = float(
        np.sum(
            weights
            * np.asarray(
                [
                    float(value.get("radial_half_extent_m"))
                    for value in accepted_observations
                ]
            )
        )
    )
    confidence = 1.0 - float(
        np.prod(
            [
                1.0 - float(value.get("confidence", 0.0))
                for value in accepted_observations
            ]
        )
    )
    confidence *= 0.7 ** len(rejection_by_index)
    confidence *= 0.85 ** len(axis_rejection_by_index)
    evidence = []
    for index, value in enumerate(observations):
        rejection = rejection_by_index.get(index)
        axis_rejection = axis_rejection_by_index.get(index)
        row = {
            "view": value.get("view"),
            "frame_id": value.get("frame_id"),
            "confidence": value.get("confidence"),
            "visibility": value.get("visibility"),
            "source": value.get("source"),
            "camera_position_world_m": value.get("camera_position_world_m"),
            "view_direction_world": value.get("view_direction_world"),
            "camera_pose_source": value.get("camera_pose_source"),
            "selection_reason": value.get("selection_reason"),
            "fusion_state": (
                "rejected"
                if rejection
                else "partially_accepted"
                if axis_rejection
                else "accepted"
            ),
        }
        if rejection:
            row.update(rejection)
            row["accepted_components"] = []
        elif axis_rejection:
            row.update(axis_rejection)
            row["accepted_components"] = ["position", "scale"]
            row["rejected_components"] = ["longitudinal_axis"]
        else:
            row["accepted_components"] = [
                "position",
                "scale",
                "longitudinal_axis",
            ]
        evidence.append(row)
    return {
        "id": identity[0],
        "node_type": "semantic_part_region",
        "entity_id": identity[1],
        "semantic_type": identity[2],
        "position_mean_world_m": _round_vector(fused_mean),
        "position_covariance_m2": _round_matrix(fused_covariance),
        "longitudinal_axis_world": _round_vector(fused_axis),
        "half_length_m": round(half_length, 7),
        "radial_half_extent_m": round(radial_extent, 7),
        "confidence": round(confidence, 6),
        "visibility": round(
            float(
                np.sum(
                    weights
                    * np.asarray(
                        [
                            float(value.get("visibility", 0.0))
                            for value in accepted_observations
                        ]
                    )
                )
            ),
            6,
        ),
        "observation_count": len(accepted_observations),
        "total_observation_count": len(observations),
        "rejected_observation_count": len(rejection_by_index),
        "axis_rejected_observation_count": len(axis_rejection_by_index),
        "fusion_gate": {
            "maximum_mahalanobis_squared": maximum_mahalanobis_squared,
            "maximum_axis_angle_deg": maximum_axis_angle_deg,
            "conflict_covariance_inflation": conflict_covariance_inflation,
        },
        "evidence": evidence,
        "source": "multiview_information_fusion",
        "access": "inference_visible",
    }


def generate_obb_grasp_candidates(
    intent: GraspIntent,
    geometry: OrientedObjectGeometry,
    *,
    regions: Iterable[SemanticGraspRegion] | None = None,
    approach_axis_world: Iterable[float] = (0.0, 0.0, -1.0),
    config: CandidateGenerationConfig = CandidateGenerationConfig(),
    reachability_evaluator: CandidateEvaluator | None = None,
    collision_free_evaluator: CandidateEvaluator | None = None,
    execution_success_evaluator: CandidateEvaluator | None = None,
) -> list[GraspCandidate]:
    """Generate analytic antipodal candidates on a semantic OBB region.

    Optional evaluator callbacks are the integration boundary for simulator IK,
    collision checks and replayed close/lift outcomes.  Missing evaluators remain
    unknown and are never silently converted to successful checks.
    """

    approach = unit(_vector3(approach_axis_world, field_name="approach_axis_world"))
    resolved_regions = list(regions or (semantic_region_from_obb(intent, geometry),))
    if not resolved_regions:
        raise ValueError("at least one semantic grasp region is required")
    raw_candidates: list[GraspCandidate] = []
    for region in resolved_regions:
        closing_axes = _candidate_closing_axes(
            geometry=geometry,
            longitudinal_axis=region.longitudinal_axis_world,
            approach_axis=approach,
        )
        for axis_index, closing in enumerate(closing_axes):
            radius = (
                float(region.radial_half_extent_m)
                if region.radial_half_extent_m is not None
                else obb_support_radius(
                    geometry.rotation_world, geometry.half_extents_m, closing
                )
            )
            opening_width = 2.0 * (radius + config.contact_clearance_m)
            for fraction in config.along_axis_fractions:
                for vertical_offset_m in config.vertical_offsets_m:
                    center = (
                        region.center_world_m
                        + region.longitudinal_axis_world
                        * float(fraction)
                        * float(region.half_length_m)
                        - approach * float(vertical_offset_m)
                    )
                    left_contact = center - closing * radius
                    right_contact = center + closing * radius
                    pregrasp = center - approach * config.pregrasp_distance_m
                    generation_parameters = {
                        "closing_axis_index": axis_index,
                        "along_region_fraction": round(float(fraction), 6),
                        "vertical_offset_m": round(float(vertical_offset_m), 7),
                    }
                    payload = {
                        "target_id": geometry.object_id,
                        "region_id": region.region_id,
                        "center_world_m": center,
                        "left_contact_world_m": left_contact,
                        "right_contact_world_m": right_contact,
                        "approach_axis_world": approach,
                        "closing_axis_world": closing,
                        "opening_width_m": opening_width,
                        "generation_parameters": generation_parameters,
                    }
                    checks = _candidate_checks(
                        intent=intent,
                        region=region,
                        geometry=geometry,
                        center=center,
                        opening_width=opening_width,
                        offset_fraction=float(fraction),
                        vertical_offset_m=float(vertical_offset_m),
                        config=config,
                        payload=payload,
                        reachability_evaluator=reachability_evaluator,
                        collision_free_evaluator=collision_free_evaluator,
                        execution_success_evaluator=execution_success_evaluator,
                    )
                    score, score_confidence = _candidate_score(checks)
                    candidate_index = len(raw_candidates)
                    raw_candidates.append(
                        GraspCandidate(
                            candidate_id=f"grasp_candidate_{candidate_index:03d}",
                            target_id=geometry.object_id,
                            region_id=region.region_id,
                            semantic_role=region.role,
                            center_world_m=center,
                            left_contact_world_m=left_contact,
                            right_contact_world_m=right_contact,
                            approach_axis_world=approach,
                            closing_axis_world=closing,
                            pregrasp_center_world_m=pregrasp,
                            opening_width_m=opening_width,
                            generation_parameters=generation_parameters,
                            position_covariance_m2=region.position_covariance_m2,
                            orientation_covariance_rad2=np.eye(3, dtype=np.float64)
                            * math.radians(5.0) ** 2,
                            checks=checks,
                            score=score,
                            score_confidence=score_confidence,
                            source=(
                                "analytic_semantic_region_antipodal"
                                if region.radial_half_extent_m is not None
                                else "analytic_obb_antipodal"
                            ),
                            access=geometry.access,
                        )
                    )
    ranked = sorted(
        raw_candidates,
        key=lambda item: (item.eligible, item.score, item.score_confidence),
        reverse=True,
    )
    if config.max_candidates is None:
        return ranked
    if config.max_candidates < 1:
        raise ValueError("max_candidates must be >= 1 or None")
    return ranked[: config.max_candidates]


def select_task_native_probe_suite(
    candidates: Sequence[GraspCandidate],
    *,
    max_candidates: int = 5,
) -> list[GraspCandidate]:
    """Select a diverse, deterministic execution-probe suite.

    Online inference should keep using the score-ranked candidates returned by
    ``generate_obb_grasp_candidates``.  Dataset collection needs a different
    policy: it must exercise one perturbation at a time so that high-scoring,
    nearly identical candidates do not crowd out useful within-task negatives.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    ranked = list(candidates)
    if not ranked:
        return []

    def parameters(candidate: GraspCandidate) -> tuple[int, float, float]:
        values = candidate.generation_parameters
        return (
            int(values.get("closing_axis_index", 0)),
            float(values.get("along_region_fraction", 0.0)),
            float(values.get("vertical_offset_m", 0.0)),
        )

    def pure_nominal(candidate: GraspCandidate) -> bool:
        axis, along, vertical = parameters(candidate)
        return axis == 0 and abs(along) <= 1e-9 and abs(vertical) <= 1e-9

    def pure_along(candidate: GraspCandidate, sign: int) -> bool:
        axis, along, vertical = parameters(candidate)
        return axis == 0 and along * sign > 1e-9 and abs(vertical) <= 1e-9

    def pure_vertical(candidate: GraspCandidate) -> bool:
        axis, along, vertical = parameters(candidate)
        return axis == 0 and abs(along) <= 1e-9 and abs(vertical) > 1e-9

    def pure_orientation(candidate: GraspCandidate) -> bool:
        axis, along, vertical = parameters(candidate)
        return axis != 0 and abs(along) <= 1e-9 and abs(vertical) <= 1e-9

    selected: list[GraspCandidate] = []

    def add_best(
        predicate: Callable[[GraspCandidate], bool],
        *,
        prefer_larger_perturbation: bool = False,
    ) -> None:
        matches = [candidate for candidate in ranked if predicate(candidate)]
        if not matches:
            return
        if prefer_larger_perturbation:
            matches.sort(
                key=lambda candidate: (
                    abs(parameters(candidate)[1]),
                    abs(parameters(candidate)[2]),
                    -parameters(candidate)[0],
                    candidate.score,
                ),
                reverse=True,
            )
        candidate = matches[0]
        if candidate.candidate_id not in {item.candidate_id for item in selected}:
            selected.append(candidate)

    add_best(pure_nominal)
    add_best(
        lambda candidate: pure_along(candidate, -1), prefer_larger_perturbation=True
    )
    add_best(
        lambda candidate: pure_along(candidate, 1), prefer_larger_perturbation=True
    )
    add_best(pure_vertical, prefer_larger_perturbation=True)
    add_best(pure_orientation, prefer_larger_perturbation=True)
    add_best(lambda candidate: pure_along(candidate, -1))
    add_best(lambda candidate: pure_along(candidate, 1))
    add_best(pure_vertical)

    # Fill larger requested suites without repeating an identical perturbation.
    seen_parameters = {parameters(candidate) for candidate in selected}
    for candidate in ranked:
        if len(selected) >= max_candidates:
            break
        key = parameters(candidate)
        if key in seen_parameters:
            continue
        selected.append(candidate)
        seen_parameters.add(key)
    return selected[:max_candidates]


def _candidate_closing_axes(
    *,
    geometry: OrientedObjectGeometry,
    longitudinal_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    cross_axis = np.cross(approach_axis, longitudinal_axis)
    if float(np.linalg.norm(cross_axis)) > 1e-6:
        axes.append(unit(cross_axis))
    for index in np.argsort(geometry.half_extents_m):
        axis = geometry.rotation_world[:, int(index)]
        if abs(float(np.dot(axis, approach_axis))) > 0.85:
            continue
        if any(abs(float(np.dot(axis, existing))) > 0.95 for existing in axes):
            continue
        axes.append(unit(axis))
    if not axes:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, approach_axis))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axes.append(unit(np.cross(approach_axis, reference)))
    return axes


def _candidate_checks(
    *,
    intent: GraspIntent,
    region: SemanticGraspRegion,
    geometry: OrientedObjectGeometry,
    center: np.ndarray,
    opening_width: float,
    offset_fraction: float,
    vertical_offset_m: float,
    config: CandidateGenerationConfig,
    payload: Mapping[str, Any],
    reachability_evaluator: CandidateEvaluator | None,
    collision_free_evaluator: CandidateEvaluator | None,
    execution_success_evaluator: CandidateEvaluator | None,
) -> tuple[CandidateCheck, ...]:
    semantic_probability = (
        region.confidence if region.role in intent.preferred_roles else 0.0
    )
    along_probability = float(np.clip(1.0 - abs(offset_fraction), 0.0, 1.0))
    vertical_probability = float(
        np.clip(1.0 - abs(vertical_offset_m) / 0.025, 0.0, 1.0)
    )
    task_probability = along_probability * vertical_probability
    width_probability = float(
        config.min_opening_width_m <= opening_width <= config.max_opening_width_m
    )
    if geometry.support_z_m is None:
        support_probability = None
        support_measurement: dict[str, Any] = {"clearance_m": None}
    else:
        support_clearance = float(center[2] - geometry.support_z_m)
        support_probability = _sigmoid(
            (support_clearance - config.minimum_support_clearance_m) / 0.003
        )
        support_measurement = {"clearance_m": round(support_clearance, 7)}
    reachability = _evaluate(reachability_evaluator, payload)
    collision_free = _evaluate(collision_free_evaluator, payload)
    execution_success = _evaluate(execution_success_evaluator, payload)
    return (
        CandidateCheck(
            "semantic_match",
            semantic_probability,
            True,
            region.source,
            {"role": region.role, "preferred_roles": list(intent.preferred_roles)},
        ),
        CandidateCheck(
            "antipodal_geometry",
            1.0,
            True,
            geometry.source,
            {"contact_pattern": intent.contact_pattern},
        ),
        CandidateCheck(
            "task_compatibility",
            task_probability,
            False,
            "intent_region_centrality",
            {
                "normalized_region_offset": round(offset_fraction, 6),
                "vertical_offset_m": round(vertical_offset_m, 7),
            },
        ),
        CandidateCheck(
            "opening_width_feasible",
            width_probability,
            True,
            "gripper_limits",
            {
                "opening_width_m": round(float(opening_width), 7),
                "allowed_range_m": [
                    config.min_opening_width_m,
                    config.max_opening_width_m,
                ],
            },
        ),
        CandidateCheck(
            "support_clearance",
            support_probability,
            False,
            (
                "object_center_clearance_proxy"
                if geometry.support_z_m is not None
                else "not_observed"
            ),
            support_measurement,
        ),
        CandidateCheck(
            "reachable",
            reachability,
            True,
            "simulator_ik" if reachability_evaluator else "not_evaluated",
        ),
        CandidateCheck(
            "collision_free",
            collision_free,
            True,
            "simulator_collision_query"
            if collision_free_evaluator
            else "not_evaluated",
        ),
        CandidateCheck(
            "predicted_execution_success",
            execution_success,
            False,
            "execution_model" if execution_success_evaluator else "not_evaluated",
        ),
    )


def _evaluate(
    evaluator: CandidateEvaluator | None, payload: Mapping[str, Any]
) -> float | None:
    if evaluator is None:
        return None
    value = evaluator(payload)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("candidate evaluator probabilities must be in [0, 1]")
    return probability


def _candidate_score(checks: Sequence[CandidateCheck]) -> tuple[float, float]:
    weights = {
        "semantic_match": 1.5,
        "antipodal_geometry": 1.0,
        "task_compatibility": 1.0,
        "opening_width_feasible": 1.0,
        "support_clearance": 0.8,
        "reachable": 1.2,
        "collision_free": 1.2,
        "predicted_execution_success": 1.5,
    }
    total_weight = float(sum(weights.values()))
    known = [
        (check, weights[check.name])
        for check in checks
        if check.probability is not None
    ]
    known_weight = float(sum(weight for _, weight in known))
    if known_weight == 0.0:
        return 0.0, 0.0
    score = (
        sum(float(check.probability) * weight for check, weight in known) / known_weight
    )
    hard_failure = any(
        check.hard_constraint
        and check.probability is not None
        and check.probability < 0.5
        for check in checks
    )
    if hard_failure:
        score *= 0.25
    return float(score), float(known_weight / total_weight)


def candidates_equivalent_under_current_checks(
    first: GraspCandidate,
    second: GraspCandidate,
    *,
    maximum_center_distance_m: float = 0.003,
    maximum_opening_difference_m: float = 0.003,
    maximum_closing_axis_difference_deg: float = 8.0,
    maximum_approach_axis_difference_deg: float = 8.0,
) -> bool:
    """Return whether current evidence has no task-relevant way to separate frames."""

    closing_similarity = abs(
        float(np.dot(first.closing_axis_world, second.closing_axis_world))
    )
    approach_similarity = float(
        np.dot(first.approach_axis_world, second.approach_axis_world)
    )
    closing_difference_deg = math.degrees(
        math.acos(float(np.clip(closing_similarity, -1.0, 1.0)))
    )
    approach_difference_deg = math.degrees(
        math.acos(float(np.clip(approach_similarity, -1.0, 1.0)))
    )
    orientation_requires_disambiguation = bool(
        first.generation_parameters.get("orientation_requires_disambiguation")
        or second.generation_parameters.get("orientation_requires_disambiguation")
    )
    return bool(
        first.region_id == second.region_id
        and first.semantic_role == second.semantic_role
        and np.linalg.norm(first.center_world_m - second.center_world_m)
        <= maximum_center_distance_m
        and abs(first.opening_width_m - second.opening_width_m)
        <= maximum_opening_difference_m
        and (
            not orientation_requires_disambiguation
            or (
                closing_difference_deg <= maximum_closing_axis_difference_deg
                and approach_difference_deg <= maximum_approach_axis_difference_deg
            )
        )
        and first.eligible == second.eligible
        and first.failed_constraints == second.failed_constraints
    )


def candidate_equivalence_classes(
    candidates: Sequence[GraspCandidate],
) -> list[list[str]]:
    """Group candidate IDs that are equivalent under the currently evaluated checks."""

    groups: list[list[GraspCandidate]] = []
    for candidate in candidates:
        for group in groups:
            if candidates_equivalent_under_current_checks(candidate, group[0]):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return [[candidate.candidate_id for candidate in group] for group in groups]


def build_grasp_candidate_graph(
    intent: GraspIntent,
    geometry: OrientedObjectGeometry,
    regions: Sequence[SemanticGraspRegion],
    candidates: Sequence[GraspCandidate],
    *,
    world_state_version: int,
    top_k: int = 3,
    view_assessment: Mapping[str, Any] | None = None,
    candidate_ranking: Mapping[str, Any] | None = None,
    view_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Serialize candidate frames as sparse points and typed lines."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    nodes: list[dict[str, Any]] = [
        {
            "id": "task.goal",
            "node_type": "task_goal",
            "semantic_type": intent.task_goal,
            "attributes": {
                "task_stage": intent.task_stage,
                "post_grasp_goal": intent.post_grasp_goal,
                "functional_constraints": list(intent.functional_constraints),
                "required_arms": intent.required_arms,
            },
            "source": intent.source,
            "access": "inference_visible",
        },
        {
            "id": "intent.grasp",
            "node_type": "grasp_intent",
            "semantic_type": intent.task_goal,
            "attributes": intent.as_dict(),
            "source": intent.source,
            "access": "inference_visible",
        },
        {
            "id": geometry.object_id,
            "node_type": "entity",
            "semantic_type": "target_object",
            "position_mean_world_m": _round_vector(geometry.center_world_m),
            "orientation_world": np.round(geometry.rotation_world, 7).tolist(),
            "half_extents_m": _round_vector(geometry.half_extents_m),
            "source": geometry.source,
            "access": geometry.access,
        },
    ]
    edges: list[dict[str, Any]] = [
        _graph_edge(
            "intent_interprets_task_goal",
            "intent.grasp",
            "task.goal",
            "interprets_goal",
            intent.confidence,
            intent.source,
        ),
        _graph_edge(
            "intent_targets_object",
            "intent.grasp",
            geometry.object_id,
            "targets",
            intent.confidence,
            intent.source,
        ),
    ]
    region_by_id = {region.region_id: region for region in regions}
    selection_scores = dict((candidate_ranking or {}).get("scores", {}))
    selection_source = (candidate_ranking or {}).get("source")
    for region in regions:
        nodes.append(
            {
                "id": region.region_id,
                "node_type": "semantic_region",
                "semantic_type": region.role,
                **{
                    key: value
                    for key, value in region.as_dict().items()
                    if key not in {"id", "role"}
                },
            }
        )
        edges.append(
            _graph_edge(
                f"intent_targets_{region.region_id}",
                "intent.grasp",
                region.region_id,
                "prefers_region",
                region.confidence,
                region.source,
            )
        )
    for evidence_index, evidence in enumerate(view_evidence):
        frame_id = evidence.get("frame_id", evidence_index)
        evidence_id = f"view_evidence.{frame_id}"
        confidence = float(evidence.get("confidence", 0.0))
        visibility = float(evidence.get("visibility", 0.0))
        fusion_state = str(evidence.get("fusion_state", "observed"))
        accepted_probability = (
            0.0
            if fusion_state == "rejected"
            else float(np.clip(confidence * visibility, 0.0, 1.0))
        )
        nodes.append(
            {
                "id": evidence_id,
                "node_type": "view_evidence",
                "semantic_type": str(evidence.get("view", "unknown_view")),
                "attributes": {
                    "frame_id": frame_id,
                    "fusion_state": fusion_state,
                    "accepted_components": list(
                        evidence.get("accepted_components", ())
                    ),
                    "confidence": round(confidence, 6),
                    "visibility": round(visibility, 6),
                    "camera_position_world_m": evidence.get("camera_position_world_m"),
                    "view_direction_world": evidence.get("view_direction_world"),
                    "camera_pose_source": evidence.get("camera_pose_source"),
                    "selection_reason": evidence.get("selection_reason"),
                },
                "source": str(evidence.get("source", "observation_manager")),
                "access": "inference_visible",
            }
        )
        for region in regions:
            edges.append(
                _graph_edge(
                    f"{evidence_id}.observes.{region.region_id}",
                    evidence_id,
                    region.region_id,
                    "observes_region",
                    accepted_probability,
                    str(evidence.get("source", "observation_manager")),
                    {"fusion_state": fusion_state},
                )
            )
    for candidate in candidates:
        center_id = f"{candidate.candidate_id}.center"
        left_id = f"{candidate.candidate_id}.left_contact"
        right_id = f"{candidate.candidate_id}.right_contact"
        pregrasp_id = f"{candidate.candidate_id}.pregrasp"
        node_common = {
            "candidate_id": candidate.candidate_id,
            "position_covariance_m2": _round_matrix(candidate.position_covariance_m2),
            "source": candidate.source,
            "access": candidate.access,
        }
        nodes.extend(
            [
                {
                    "id": candidate.candidate_id,
                    "node_type": "grasp_frame_candidate",
                    "semantic_type": candidate.semantic_role,
                    "position_mean_world_m": _round_vector(candidate.center_world_m),
                    "attributes": {
                        "approach_axis_world": _round_vector(
                            candidate.approach_axis_world
                        ),
                        "closing_axis_world": _round_vector(
                            candidate.closing_axis_world
                        ),
                        "opening_width_m": round(candidate.opening_width_m, 7),
                        "score": round(candidate.score, 6),
                        "score_confidence": round(candidate.score_confidence, 6),
                        "eligible": candidate.eligible,
                        "failed_constraints": candidate.failed_constraints,
                        "unknown_constraints": candidate.unknown_constraints,
                    },
                    "orientation_covariance_rad2": _round_matrix(
                        candidate.orientation_covariance_rad2
                    ),
                    **node_common,
                },
                {
                    "id": center_id,
                    "node_type": "grasp_frame_point",
                    "semantic_type": "grasp_center",
                    "position_mean_world_m": _round_vector(candidate.center_world_m),
                    "attributes": {
                        "semantic_role": candidate.semantic_role,
                        "opening_width_m": round(candidate.opening_width_m, 7),
                        "approach_axis_world": _round_vector(
                            candidate.approach_axis_world
                        ),
                        "closing_axis_world": _round_vector(
                            candidate.closing_axis_world
                        ),
                        "score": round(candidate.score, 6),
                        "score_confidence": round(candidate.score_confidence, 6),
                        "eligible": candidate.eligible,
                        "selection_utility": selection_scores.get(
                            candidate.candidate_id
                        ),
                        "selection_utility_source": selection_source,
                    },
                    **node_common,
                },
                {
                    "id": left_id,
                    "node_type": "grasp_frame_point",
                    "semantic_type": "left_contact",
                    "position_mean_world_m": _round_vector(
                        candidate.left_contact_world_m
                    ),
                    **node_common,
                },
                {
                    "id": right_id,
                    "node_type": "grasp_frame_point",
                    "semantic_type": "right_contact",
                    "position_mean_world_m": _round_vector(
                        candidate.right_contact_world_m
                    ),
                    **node_common,
                },
                {
                    "id": pregrasp_id,
                    "node_type": "grasp_frame_point",
                    "semantic_type": "pregrasp_center",
                    "position_mean_world_m": _round_vector(
                        candidate.pregrasp_center_world_m
                    ),
                    **node_common,
                },
            ]
        )
        region = region_by_id[candidate.region_id]
        edges.extend(
            [
                _graph_edge(
                    f"{candidate.candidate_id}.supports_task_goal",
                    candidate.candidate_id,
                    "task.goal",
                    "supports_task_goal",
                    _check_probability(candidate, "task_compatibility"),
                    _check_source(candidate, "task_compatibility"),
                ),
                _graph_edge(
                    f"{candidate.candidate_id}.frame_targets_region",
                    candidate.candidate_id,
                    region.region_id,
                    "targets_region",
                    _check_probability(candidate, "semantic_match"),
                    region.source,
                ),
                _graph_edge(
                    f"{candidate.candidate_id}.targets_region",
                    center_id,
                    region.region_id,
                    "targets_region",
                    _check_probability(candidate, "semantic_match"),
                    region.source,
                ),
                _graph_edge(
                    f"{candidate.candidate_id}.closing_span",
                    left_id,
                    right_id,
                    "closing_span",
                    _check_probability(candidate, "antipodal_geometry"),
                    candidate.source,
                    {"opening_width_m": round(candidate.opening_width_m, 7)},
                ),
                _graph_edge(
                    f"{candidate.candidate_id}.approach_path",
                    pregrasp_id,
                    center_id,
                    "approach_path",
                    _check_probability(candidate, "collision_free"),
                    _check_source(candidate, "collision_free"),
                ),
                *[
                    _graph_edge(
                        f"{point_id}.component_of",
                        point_id,
                        candidate.candidate_id,
                        "component_of_frame",
                        1.0,
                        candidate.source,
                    )
                    for point_id in (center_id, left_id, right_id, pregrasp_id)
                ],
            ]
        )
        for check in candidate.checks:
            edges.append(
                _graph_edge(
                    f"{candidate.candidate_id}.{check.name}",
                    center_id,
                    geometry.object_id,
                    check.name,
                    check.probability,
                    check.source,
                    check.measurement,
                )
            )
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            if not candidates_equivalent_under_current_checks(first, second):
                continue
            edges.append(
                _graph_edge(
                    f"{first.candidate_id}.equivalent_to.{second.candidate_id}",
                    first.candidate_id,
                    second.candidate_id,
                    "equivalent_grasp_under_current_checks",
                    1.0,
                    "current_candidate_checks",
                )
            )
    ranked = sorted(
        candidates,
        key=lambda item: float(selection_scores.get(item.candidate_id, item.score)),
        reverse=True,
    )
    top_candidates = []
    for candidate in ranked[:top_k]:
        payload = candidate.as_dict()
        payload["selection_utility"] = selection_scores.get(candidate.candidate_id)
        payload["selection_utility_source"] = selection_source
        top_candidates.append(payload)
    return {
        "schema_version": GRASP_CANDIDATE_GRAPH_SCHEMA,
        "access": geometry.access,
        "query": "propose_grasp_candidates",
        "world_state_version": int(world_state_version),
        "coordinate_frame": "world",
        "units": {"position": "m", "angle": "rad"},
        "intent": intent.as_dict(),
        "nodes": nodes,
        "edges": edges,
        "candidate_count": len(candidates),
        "candidate_equivalence_classes": candidate_equivalence_classes(candidates),
        "top_candidates": top_candidates,
        "candidate_ranking": dict(candidate_ranking or {}),
        "view_assessment": dict(view_assessment or {}),
    }


def assess_candidate_view_sufficiency(
    candidates: Sequence[GraspCandidate],
    *,
    observed_view_directions_world: Iterable[Iterable[float]] = (),
    candidate_views: Iterable[Mapping[str, Any]] = (),
    minimum_views: int = 2,
    minimum_score_margin: float = 0.08,
    maximum_position_std_m: float = 0.015,
    candidate_selection_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Explain whether current views distinguish the leading grasp candidates."""

    ranked = [candidate for candidate in candidates if candidate.eligible]
    selection_scores = dict(candidate_selection_scores or {})
    ranked.sort(
        key=lambda item: float(selection_scores.get(item.candidate_id, item.score)),
        reverse=True,
    )
    observed = [
        unit(_vector3(value, field_name="observed_view_direction"))
        for value in observed_view_directions_world
    ]
    reasons: list[str] = []
    if not ranked:
        return {
            "sufficient": False,
            "visual_evidence_sufficient": False,
            "execution_ready": False,
            "reasons": ["no_eligible_grasp_candidate"],
            "top_candidate_id": None,
            "score_margin": None,
            "top_candidates_equivalent_under_current_checks": False,
            "unresolved_nonvisual_checks": [],
            "recommended_view": None,
            "candidate_view_scores": [],
        }
    top = ranked[0]
    top_utility = float(selection_scores.get(top.candidate_id, top.score))
    margin = (
        top_utility
        - float(selection_scores.get(ranked[1].candidate_id, ranked[1].score))
        if len(ranked) > 1
        else 1.0
    )
    top_candidates_equivalent = False
    if len(ranked) > 1:
        second = ranked[1]
        top_candidates_equivalent = candidates_equivalent_under_current_checks(
            top, second
        )
    position_std = float(
        math.sqrt(
            max(0.0, float(np.max(np.linalg.eigvalsh(top.position_covariance_m2))))
        )
    )
    if len(observed) < minimum_views:
        reasons.append("insufficient_independent_views")
    if margin < minimum_score_margin and not top_candidates_equivalent:
        reasons.append("top_candidates_not_separated")
    if position_std > maximum_position_std_m:
        reasons.append("top_candidate_position_uncertain")
    unresolved_nonvisual_checks = list(top.unknown_constraints)

    if len(ranked) > 1:
        discrimination = ranked[1].center_world_m - top.center_world_m
        if float(np.linalg.norm(discrimination)) < 1e-6:
            discrimination = ranked[1].closing_axis_world - top.closing_axis_world
    else:
        discrimination = top.closing_axis_world
    if float(np.linalg.norm(discrimination)) < 1e-6:
        discrimination = top.closing_axis_world
    discrimination = unit(discrimination)

    rows = []
    for view in candidate_views:
        name = _text(view.get("view"), field_name="candidate view name")
        direction = unit(
            _vector3(
                view.get("view_direction_world"), field_name="view_direction_world"
            )
        )
        projection_discrimination = 1.0 - abs(float(np.dot(direction, discrimination)))
        closing_observability = 1.0 - abs(
            float(np.dot(direction, top.closing_axis_world))
        )
        approach_observability = 1.0 - abs(
            float(np.dot(direction, top.approach_axis_world))
        )
        novelty = (
            1.0
            if not observed
            else 1.0
            - max(abs(float(np.dot(direction, previous))) for previous in observed)
        )
        move_cost = float(view.get("move_cost", 0.0))
        utility = (
            0.40 * projection_discrimination
            + 0.25 * closing_observability
            + 0.20 * approach_observability
            + 0.15 * novelty
            - 0.15 * move_cost
        )
        rows.append(
            {
                "view": name,
                "utility": round(float(utility), 6),
                "reason": {
                    "separates_top_candidates": round(projection_discrimination, 6),
                    "observes_closing_span": round(closing_observability, 6),
                    "observes_approach_depth": round(approach_observability, 6),
                    "view_novelty": round(novelty, 6),
                    "move_cost": round(move_cost, 6),
                },
            }
        )
    rows.sort(key=lambda item: item["utility"], reverse=True)
    sufficient = not reasons
    execution_ready = sufficient and not unresolved_nonvisual_checks
    return {
        "sufficient": sufficient,
        "visual_evidence_sufficient": sufficient,
        "execution_ready": execution_ready,
        "reasons": reasons,
        "top_candidate_id": top.candidate_id,
        "score_margin": round(float(margin), 6),
        "candidate_score_source": (
            "selection_utility" if selection_scores else "analytic_candidate_score"
        ),
        "top_candidate_position_std_m": round(position_std, 7),
        "top_candidates_equivalent_under_current_checks": top_candidates_equivalent,
        "observed_view_count": len(observed),
        "unresolved_nonvisual_checks": unresolved_nonvisual_checks,
        "recommended_view": None if sufficient or not rows else rows[0]["view"],
        "candidate_view_scores": rows,
    }


def _graph_edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    probability: float | None,
    evidence_source: str,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "state": (
            "unknown"
            if probability is None
            else ("true" if probability >= 0.5 else "false")
        ),
        "probability": round(float(probability), 6)
        if probability is not None
        else None,
        "uncertainty": (
            1.0 if probability is None else round(1.0 - abs(2.0 * probability - 1.0), 6)
        ),
        "measurement": dict(measurement or {}),
        "evidence_source": evidence_source,
    }


def _check_probability(candidate: GraspCandidate, name: str) -> float | None:
    return next(check.probability for check in candidate.checks if check.name == name)


def _check_source(candidate: GraspCandidate, name: str) -> str:
    return next(check.source for check in candidate.checks if check.name == name)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -40.0, 40.0))))
