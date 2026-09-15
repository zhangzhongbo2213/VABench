"""Phase 1.5 observation fusion and active-query state.

The module deliberately has no simulator dependency.  Phase 1.5 uses an
oracle-backed sensor adapter to create :class:`ObservationGraph` instances,
but the belief update below only consumes those observations.  A learned RGB-D
adapter can therefore replace the oracle adapter without changing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .spatial_graph import score_candidate_view


EPS = 1e-9


@dataclass(frozen=True)
class ViewSensorModel:
    """Effective keypoint/depth noise used by the analytic view model.

    These values describe the complete measurement pipeline, not nominal
    simulator precision. They must be calibrated from held-out prediction
    error before analytic utilities are treated as deployment evidence.
    """

    keypoint_std_px: float
    depth_std_m: float
    minimum_lateral_std_m: float = 1e-5
    minimum_depth_std_m: float = 1e-5

    def __post_init__(self) -> None:
        values = (
            self.keypoint_std_px,
            self.depth_std_m,
            self.minimum_lateral_std_m,
            self.minimum_depth_std_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "analytic view sensor noise must be finite and strictly positive"
            )


def point_measurement_information(
    *,
    view_direction_world: Iterable[float],
    range_m: float,
    focal_length_px: float,
    sensor_model: ViewSensorModel,
    measurement_probability: float = 1.0,
) -> np.ndarray:
    """Return a 3x3 Fisher-style point information contribution.

    ``measurement_probability`` is an analytic visibility/valid-depth factor.
    It keeps L1 geometry separate from the L2 visibility model and must never
    be filled with post-capture or oracle candidate evidence at inference.
    """

    if not np.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("view range must be finite and positive")
    if not np.isfinite(focal_length_px) or focal_length_px <= 0.0:
        raise ValueError("focal length must be finite and positive")
    if (
        not np.isfinite(measurement_probability)
        or not 0.0 <= measurement_probability <= 1.0
    ):
        raise ValueError("measurement probability must be in [0, 1]")
    direction = _strict_unit(
        np.asarray(tuple(view_direction_world), dtype=np.float64),
        "view direction",
    )
    lateral_std = max(
        range_m / focal_length_px * sensor_model.keypoint_std_px,
        sensor_model.minimum_lateral_std_m,
    )
    depth_std = max(
        sensor_model.depth_std_m,
        sensor_model.minimum_depth_std_m,
    )
    axial = np.outer(direction, direction)
    lateral = np.eye(3, dtype=np.float64) - axial
    information = measurement_probability * (
        lateral / lateral_std**2 + axial / depth_std**2
    )
    return _symmetric_positive_semidefinite(information)


def posterior_covariance_from_information(
    prior_covariance: np.ndarray,
    measurement_information: Iterable[np.ndarray],
) -> np.ndarray:
    """Fuse a prior covariance with one or more independent measurements."""

    prior = _strict_positive_definite(prior_covariance, "prior covariance")
    precision = np.linalg.inv(prior)
    for value in measurement_information:
        precision += _symmetric_positive_semidefinite(value)
    return _strict_positive_definite(np.linalg.inv(precision), "posterior covariance")


def directional_variance(
    covariance: np.ndarray,
    direction: Iterable[float],
) -> float:
    value = _strict_positive_definite(covariance, "covariance")
    axis = _strict_unit(np.asarray(tuple(direction), dtype=np.float64), "direction")
    return float(axis @ value @ axis)


ANALYTIC_FISHER_FEATURE_NAMES = (
    "fisher_prior_std_closing_m",
    "fisher_prior_std_lateral_m",
    "fisher_prior_std_approach_m",
    "fisher_posterior_std_closing_m",
    "fisher_posterior_std_lateral_m",
    "fisher_posterior_std_approach_m",
    "fisher_std_reduction_closing_m",
    "fisher_std_reduction_lateral_m",
    "fisher_std_reduction_approach_m",
    "fisher_log_volume_reduction_nats",
    "fisher_worst_axis_std_reduction_m",
)


def analytic_view_information_features(
    *,
    prior_covariance: np.ndarray,
    candidate_view_direction_world: Iterable[float],
    task_axes_world: Iterable[Iterable[float]],
    range_m: float,
    focal_length_px: float,
    sensor_model: ViewSensorModel,
    measurement_probability: float = 1.0,
) -> dict[str, float]:
    """L1 analytic view utility from a Fisher information update.

    The returned features depend only on belief-side quantities: the current
    node covariance, the candidate ray, and a calibrated sensor model. They
    carry no view label, candidate index, or simulator target pose, so the same
    feature vector is defined for an arbitrary number of cameras.

    ``task_axes_world`` must be the three uncertainty-frame axes (closing,
    lateral, approach) from ``view_pose_features.uncertainty_frame``.
    """

    axes = [
        _strict_unit(np.asarray(tuple(axis), dtype=np.float64), f"task axis {index}")
        for index, axis in enumerate(task_axes_world)
    ]
    if len(axes) != 3:
        raise ValueError("analytic view features need exactly three task axes")
    prior = _strict_positive_definite(prior_covariance, "prior covariance")
    information = point_measurement_information(
        view_direction_world=candidate_view_direction_world,
        range_m=range_m,
        focal_length_px=focal_length_px,
        sensor_model=sensor_model,
        measurement_probability=measurement_probability,
    )
    posterior = posterior_covariance_from_information(prior, (information,))
    prior_std = [float(np.sqrt(directional_variance(prior, axis))) for axis in axes]
    posterior_std = [
        float(np.sqrt(directional_variance(posterior, axis))) for axis in axes
    ]
    reductions = [
        float(max(0.0, before - after))
        for before, after in zip(prior_std, posterior_std)
    ]
    prior_sign, prior_logdet = np.linalg.slogdet(prior)
    posterior_sign, posterior_logdet = np.linalg.slogdet(posterior)
    if prior_sign <= 0.0 or posterior_sign <= 0.0:
        raise ValueError("covariance determinant must be positive")
    labels = ("closing", "lateral", "approach")
    features: dict[str, float] = {}
    for label, value in zip(labels, prior_std):
        features[f"fisher_prior_std_{label}_m"] = value
    for label, value in zip(labels, posterior_std):
        features[f"fisher_posterior_std_{label}_m"] = value
    for label, value in zip(labels, reductions):
        features[f"fisher_std_reduction_{label}_m"] = value
    features["fisher_log_volume_reduction_nats"] = float(
        max(0.0, 0.5 * (prior_logdet - posterior_logdet))
    )
    features["fisher_worst_axis_std_reduction_m"] = reductions[
        int(np.argmax(prior_std))
    ]
    return features


def depth_dominates_lateral_noise(
    *,
    sensor_model: ViewSensorModel,
    range_m: float,
    focal_length_px: float,
) -> bool:
    """Return whether axial depth noise exceeds lateral pixel noise.

    This describes the orientation of the measurement anisotropy; it is not a
    validity check for Fisher view selection. When lateral noise is larger than
    depth noise the preferred view geometry reverses, but the measurement is
    still anisotropic and view direction still carries information.
    """

    lateral_std, depth_std = measurement_noise_std_m(
        sensor_model=sensor_model,
        range_m=range_m,
        focal_length_px=focal_length_px,
    )
    return bool(depth_std > lateral_std)


def measurement_noise_std_m(
    *,
    sensor_model: ViewSensorModel,
    range_m: float,
    focal_length_px: float,
) -> tuple[float, float]:
    """Return effective lateral and axial standard deviations in metres."""

    if not np.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("view range must be finite and positive")
    if not np.isfinite(focal_length_px) or focal_length_px <= 0.0:
        raise ValueError("focal length must be finite and positive")
    lateral_std = max(
        range_m / focal_length_px * sensor_model.keypoint_std_px,
        sensor_model.minimum_lateral_std_m,
    )
    depth_std = max(sensor_model.depth_std_m, sensor_model.minimum_depth_std_m)
    return float(lateral_std), float(depth_std)


def measurement_noise_anisotropy_ratio(
    *,
    sensor_model: ViewSensorModel,
    range_m: float,
    focal_length_px: float,
) -> float:
    """Return ``max(sigma_lat, sigma_depth) / min(...)`` (always >= 1)."""

    lateral_std, depth_std = measurement_noise_std_m(
        sensor_model=sensor_model,
        range_m=range_m,
        focal_length_px=focal_length_px,
    )
    return float(max(lateral_std, depth_std) / min(lateral_std, depth_std))


def is_nearly_isotropic_noise(
    *,
    sensor_model: ViewSensorModel,
    range_m: float,
    focal_length_px: float,
    relative_tolerance: float = 0.05,
) -> bool:
    """Return whether camera-ray orientation has negligible noise anisotropy."""

    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("isotropy tolerance must be finite and nonnegative")
    ratio = measurement_noise_anisotropy_ratio(
        sensor_model=sensor_model,
        range_m=range_m,
        focal_length_px=focal_length_px,
    )
    return bool(ratio <= 1.0 + tolerance)


def propagate_relation_covariance(
    node_covariance: np.ndarray,
    relation_jacobian: np.ndarray,
) -> np.ndarray:
    """Delta-method propagation for smooth relation measurements only.

    Thresholded ``between`` predicates and min-distance clearance require a
    smooth surrogate, sigma points, or Monte Carlo rather than this function.
    """

    covariance = _strict_positive_definite(node_covariance, "node covariance")
    jacobian = np.asarray(relation_jacobian, dtype=np.float64)
    if jacobian.ndim == 1:
        jacobian = jacobian[None, :]
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != covariance.shape[0]
        or not np.all(np.isfinite(jacobian))
    ):
        raise ValueError("relation Jacobian must align with node covariance")
    return _symmetric_positive_semidefinite(jacobian @ covariance @ jacobian.T)


@dataclass(frozen=True)
class ObservationGraph:
    """Evidence produced by one actually acquired camera frame."""

    frame_id: int
    view: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    query_axes: dict[str, list[float]]
    source: str = "inference_observation"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationGraph":
        return cls(
            frame_id=int(value["frame_id"]),
            view=str(value["view"]),
            nodes=[dict(node) for node in value.get("nodes", [])],
            edges=[dict(edge) for edge in value.get("edges", [])],
            query_axes={
                key: list(axis) for key, axis in value.get("query_axes", {}).items()
            },
            source=str(value.get("source", "inference_observation")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "phase1.5.observation_graph.v1",
            "access": self.source,
            "frame_id": self.frame_id,
            "view": self.view,
            "nodes": self.nodes,
            "edges": self.edges,
            "query_axes": self.query_axes,
        }


@dataclass
class _NodeState:
    node_id: str
    semantic_type: str
    mean: np.ndarray
    covariance: np.ndarray
    source_frames: list[int] = field(default_factory=list)
    views: list[str] = field(default_factory=list)
    observations: int = 0
    visibility: float = 0.0
    source: str = "observation"


@dataclass
class _EdgeState:
    edge_id: str
    source: str
    target: str
    relation: str
    alpha: float = 1.0
    beta: float = 1.0
    evidence_weight: float = 0.0
    evidence_frames: list[int] = field(default_factory=list)
    evidence_views: list[str] = field(default_factory=list)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    valid_for_world_state: int = 1

    @property
    def probability(self) -> float:
        return self.alpha / max(self.alpha + self.beta, EPS)

    @property
    def epistemic_uncertainty(self) -> float:
        return float(np.clip(1.0 / (1.0 + self.evidence_weight), 0.0, 1.0))

    @property
    def posterior_std(self) -> float:
        total = max(self.alpha + self.beta, EPS)
        return float(
            np.sqrt(self.probability * (1.0 - self.probability) / (total + 1.0))
        )


class SpatialBelief:
    """Sparse world belief updated only by acquired observation graphs."""

    def __init__(
        self,
        *,
        query: str,
        world_state_version: int,
        required_edge_ids: Iterable[str],
        query_axes: Mapping[str, Iterable[float]],
        deterministic_nodes: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.query = query
        self.world_state_version = int(world_state_version)
        self.required_edge_ids = tuple(required_edge_ids)
        self.query_axes = {
            key: [float(x) for x in value] for key, value in query_axes.items()
        }
        self._nodes: dict[str, _NodeState] = {}
        self._edges: dict[str, _EdgeState] = {}
        self.history: list[dict[str, Any]] = []
        for node in deterministic_nodes:
            self._merge_node(
                dict(node), frame_id=0, view="kinematics", deterministic=True
            )

    def update(
        self, observation: ObservationGraph | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fuse one frame and return the resulting public belief graph."""

        if not isinstance(observation, ObservationGraph):
            observation = ObservationGraph.from_dict(observation)
        self._merge_query_axes(observation.query_axes)
        for node in observation.nodes:
            self._merge_node(
                node,
                frame_id=observation.frame_id,
                view=observation.view,
                deterministic=False,
            )
        for edge in observation.edges:
            self._merge_edge(edge, frame_id=observation.frame_id, view=observation.view)
        self.history.append(
            {
                "frame_id": observation.frame_id,
                "view": observation.view,
                "node_count": len(observation.nodes),
                "edge_count": len(observation.edges),
            }
        )
        return self.to_graph()

    def replace_derived_edges(
        self,
        edges: Iterable[Mapping[str, Any]],
        *,
        frame_id: int,
        view: str,
    ) -> dict[str, Any]:
        """Replace relation probabilities after recomputing them from fused nodes.

        Evidence weight and provenance still accumulate across acquired views,
        but the probability is not averaged with stale single-view geometry.
        """

        for edge in edges:
            edge_id = str(edge["id"])
            evidence_weight = float(np.clip(edge.get("evidence_weight", 0.0), 0.0, 1.0))
            if evidence_weight <= 1e-6:
                continue
            probability = float(np.clip(edge.get("probability", 0.5), 0.0, 1.0))
            state = self._edges.get(edge_id)
            if state is None:
                state = _EdgeState(
                    edge_id=edge_id,
                    source=str(edge.get("source", "")),
                    target=str(edge.get("target", "")),
                    relation=str(edge.get("relation", "unknown")),
                    valid_for_world_state=int(
                        edge.get("valid_for_world_state", self.world_state_version)
                    ),
                )
                self._edges[edge_id] = state
            added_weight = 8.0 * evidence_weight
            total_weight = state.evidence_weight + added_weight
            state.alpha = 1.0 + probability * total_weight
            state.beta = 1.0 + (1.0 - probability) * total_weight
            state.evidence_weight = total_weight
            state.evidence_frames.append(int(frame_id))
            state.evidence_views.append(str(view))
            if edge.get("measurement"):
                state.measurements.append(dict(edge["measurement"]))
        return self.to_graph()

    def _merge_query_axes(self, axes: Mapping[str, Iterable[float]]) -> None:
        for key, value in axes.items():
            candidate = np.asarray(list(value), dtype=np.float64)
            norm = float(np.linalg.norm(candidate))
            if norm < EPS:
                continue
            candidate /= norm
            if key not in self.query_axes:
                self.query_axes[key] = candidate.tolist()
                continue
            existing = np.asarray(self.query_axes[key], dtype=np.float64)
            existing_norm = float(np.linalg.norm(existing))
            if existing_norm < EPS:
                self.query_axes[key] = candidate.tolist()
                continue
            existing /= existing_norm
            # Axes are unoriented: ``a`` and ``-a`` describe the same line.
            if float(np.dot(existing, candidate)) < 0.0:
                candidate = -candidate
            fused = existing + candidate
            fused_norm = float(np.linalg.norm(fused))
            if fused_norm >= EPS:
                self.query_axes[key] = (fused / fused_norm).tolist()

    def edge_uncertainty(self) -> dict[str, float]:
        return {
            edge_id: self._edges.get(
                edge_id, _EdgeState(edge_id, "", "", "")
            ).epistemic_uncertainty
            for edge_id in self.required_edge_ids
        }

    def edge_probabilities(self) -> dict[str, float]:
        return {
            edge_id: self._edges[edge_id].probability
            for edge_id in self.required_edge_ids
            if edge_id in self._edges
        }

    def to_graph(self) -> dict[str, Any]:
        nodes = []
        for state in self._nodes.values():
            nodes.append(
                {
                    "id": state.node_id,
                    "semantic_type": state.semantic_type,
                    "position_mean_world_m": _vec(state.mean),
                    "position_covariance_m2": state.covariance.round(9).tolist(),
                    "visibility": round(float(state.visibility), 6),
                    "observation_count": state.observations,
                    "source_frames": list(state.source_frames),
                    "source_views": list(state.views),
                    "source": state.source,
                    "access": "inference_visible",
                    "valid_for_world_state": self.world_state_version,
                }
            )
        edges = []
        for edge_id in self.required_edge_ids:
            state = self._edges.get(edge_id)
            if state is None:
                edges.append(
                    {
                        "id": edge_id,
                        "state": "unknown",
                        "probability": 0.5,
                        "uncertainty": 1.0,
                        "epistemic_uncertainty": 1.0,
                        "evidence_weight": 0.0,
                        "evidence_frames": [],
                        "evidence_views": [],
                        "access": "inference_visible",
                    }
                )
                continue
            probability = state.probability
            state_name = (
                "true"
                if probability >= 0.8
                else "false"
                if probability <= 0.2
                else "unknown"
            )
            edges.append(
                {
                    "id": edge_id,
                    "source": state.source,
                    "target": state.target,
                    "relation": state.relation,
                    "state": state_name,
                    "probability": round(float(probability), 6),
                    "uncertainty": round(
                        float(max(state.epistemic_uncertainty, state.posterior_std)), 6
                    ),
                    "epistemic_uncertainty": round(state.epistemic_uncertainty, 6),
                    "posterior_std": round(state.posterior_std, 6),
                    "evidence_weight": round(state.evidence_weight, 6),
                    "evidence_frames": list(state.evidence_frames),
                    "evidence_views": list(state.evidence_views),
                    "measurement": state.measurements[-1] if state.measurements else {},
                    "access": "inference_visible",
                    "valid_for_world_state": state.valid_for_world_state,
                }
            )
        return {
            "schema_version": "phase1.5.spatial_belief.v1",
            "access": "inference_visible",
            "query": self.query,
            "world_state_version": self.world_state_version,
            "coordinate_frame": "world",
            "nodes": nodes,
            "edges": edges,
            "query_axes": self.query_axes,
            "observation_history": list(self.history),
        }

    def pregrasp_gate(
        self,
        *,
        required_confidence: float = 0.86,
        minimum_evidence_views: int = 2,
        relation_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Gate closure only when every required pre-grasp relation is supported."""

        thresholds = relation_thresholds or {}

        def threshold(edge_id: str, kind: str, fallback: float) -> float:
            value = thresholds.get(edge_id, {}).get(kind, fallback)
            return float(np.clip(value, 0.0, 1.0))

        missing_edges = [
            edge_id
            for edge_id in self.required_edge_ids
            if edge_id not in self._edges or self._edges[edge_id].evidence_weight <= 0.0
        ]
        if missing_edges:
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "reason": "Required pre-grasp relations have no observation evidence.",
                "action": "acquire_view",
                "missing_relations": missing_edges,
            }
        edge_states = {
            edge_id: self._edges[edge_id] for edge_id in self.required_edge_ids
        }
        distinct_views = set()
        for edge in edge_states.values():
            distinct_views.update(edge.evidence_views)
        failed = [
            edge_id
            for edge_id, edge in edge_states.items()
            if edge.probability <= threshold(edge_id, "negative", 0.2)
            and edge.epistemic_uncertainty <= 0.35
        ]
        if failed and len(distinct_views) >= minimum_evidence_views:
            confidence = max(
                1.0 - edge_states[edge_id].probability for edge_id in failed
            )
            return {
                "verdict": "adjust",
                "confidence": round(confidence, 6),
                "reason": "One or more required pre-grasp relations are confidently violated.",
                "action": "adjust_gripper_pose",
                "failed_relations": failed,
            }
        confidences = {
            edge_id: float(
                np.clip(
                    edge.probability * (1.0 - 0.35 * edge.epistemic_uncertainty),
                    0.0,
                    1.0,
                )
            )
            for edge_id, edge in edge_states.items()
        }
        confidence = min(confidences.values())
        if len(distinct_views) < minimum_evidence_views:
            return {
                "verdict": "uncertain",
                "confidence": round(confidence, 6),
                "reason": f"At least {minimum_evidence_views} complementary views are required.",
                "action": "acquire_view",
                "missing_relations": list(self.required_edge_ids),
            }
        low_confidence = [
            edge_id
            for edge_id, edge in edge_states.items()
            if edge.probability < threshold(edge_id, "positive", required_confidence)
            or confidences[edge_id]
            < threshold(edge_id, "positive", required_confidence)
        ]
        if not low_confidence:
            return {
                "verdict": "execute",
                "confidence": round(confidence, 6),
                "reason": "All required pre-grasp relations have sufficient multi-view evidence.",
                "action": "allow_gripper_close",
            }
        return {
            "verdict": "uncertain",
            "confidence": round(confidence, 6),
            "reason": "Pre-grasp relations are plausible but evidence is below the action threshold.",
            "action": "acquire_view",
            "missing_relations": low_confidence,
        }

    def _merge_node(
        self,
        node: Mapping[str, Any],
        *,
        frame_id: int,
        view: str,
        deterministic: bool,
    ) -> None:
        node_id = str(node["id"])
        mean = np.asarray(node["position_mean_world_m"], dtype=np.float64)
        covariance = np.asarray(
            node.get("position_covariance_m2", np.eye(3) * 1e-4), dtype=np.float64
        )
        covariance = _regularize_covariance(covariance)
        visibility = float(np.clip(node.get("visibility", 1.0), 0.0, 1.0))
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = _NodeState(
                node_id=node_id,
                semantic_type=str(node.get("semantic_type", "unknown")),
                mean=mean,
                covariance=covariance,
                source_frames=[] if deterministic else [frame_id],
                views=[] if deterministic else [view],
                observations=0 if deterministic else 1,
                visibility=visibility,
                source=str(node.get("source", "observation")),
            )
            return
        precision = np.linalg.inv(existing.covariance) + np.linalg.inv(covariance)
        fused_covariance = np.linalg.inv(precision)
        fused_mean = fused_covariance @ (
            np.linalg.inv(existing.covariance) @ existing.mean
            + np.linalg.inv(covariance) @ mean
        )
        existing.mean = fused_mean
        existing.covariance = _regularize_covariance(fused_covariance)
        if not deterministic:
            existing.source_frames.append(frame_id)
            existing.views.append(view)
            existing.observations += 1
        existing.visibility = max(existing.visibility, visibility)

    def _merge_edge(self, edge: Mapping[str, Any], *, frame_id: int, view: str) -> None:
        edge_id = str(edge["id"])
        evidence_weight = float(
            np.clip(edge.get("evidence_weight", edge.get("confidence", 0.0)), 0.0, 1.0)
        )
        if evidence_weight <= 1e-6:
            return
        probability = float(np.clip(edge.get("probability", 0.5), 0.0, 1.0))
        state = self._edges.get(edge_id)
        if state is None:
            state = _EdgeState(
                edge_id=edge_id,
                source=str(edge.get("source", "")),
                target=str(edge.get("target", "")),
                relation=str(edge.get("relation", "unknown")),
                valid_for_world_state=int(
                    edge.get("valid_for_world_state", self.world_state_version)
                ),
            )
            self._edges[edge_id] = state
        weight = 8.0 * evidence_weight
        state.alpha += probability * weight
        state.beta += (1.0 - probability) * weight
        state.evidence_weight += weight
        state.evidence_frames.append(frame_id)
        state.evidence_views.append(view)
        if edge.get("measurement"):
            state.measurements.append(dict(edge["measurement"]))


def score_belief_views(
    *,
    belief: SpatialBelief,
    current_view_direction_world: np.ndarray,
    current_camera_position_world: np.ndarray,
    candidates: Iterable[Mapping[str, Any]],
    visited_views: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Score poses without rendering their images or reading segmentation."""

    visited = set(visited_views)
    closing_axis = np.asarray(
        belief.query_axes.get("closing_axis_world", [1.0, 0.0, 0.0]), dtype=np.float64
    )
    rows = []
    for candidate in candidates:
        view = str(candidate["view"])
        if view in visited:
            continue
        position = np.asarray(candidate["camera_position_world"], dtype=np.float64)
        direction = np.asarray(candidate["view_direction_world"], dtype=np.float64)
        position_delta = float(np.linalg.norm(position - current_camera_position_world))
        direction_delta = float(
            np.arccos(
                np.clip(
                    np.dot(
                        _unit(direction),
                        _unit(
                            np.asarray(current_view_direction_world, dtype=np.float64)
                        ),
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
        move_cost = float(
            np.clip(
                0.7 * position_delta / 0.8 + 0.3 * direction_delta / np.pi, 0.0, 1.0
            )
        )
        predicted = score_candidate_view(
            view_direction_world=direction,
            current_view_direction_world=current_view_direction_world,
            closing_axis_world=closing_axis,
            edge_uncertainty=belief.edge_uncertainty(),
            framing_score=float(candidate.get("framing_score", 1.0)),
            move_cost=move_cost,
            relation_weights=candidate.get("relation_weights"),
        )
        rows.append(
            {
                "view": view,
                "predicted": predicted,
                "camera_position_world_m": _vec(position),
                "view_direction_world": _vec(direction),
                "visited": False,
            }
        )
    return sorted(rows, key=lambda row: row["predicted"]["utility"], reverse=True)


def _regularize_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _strict_positive_definite(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    result = (result + result.T) / 2.0
    if float(np.min(np.linalg.eigvalsh(result))) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return result


def _symmetric_positive_semidefinite(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("information matrix must be finite and square")
    result = (result + result.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    if float(np.min(eigenvalues)) < -1e-8:
        raise ValueError("information matrix must be positive semidefinite")
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _strict_unit(value: np.ndarray, name: str) -> np.ndarray:
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < EPS:
        return np.zeros_like(value)
    return value / norm


def _vec(value: Any) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(value, dtype=np.float64).reshape(-1)]
