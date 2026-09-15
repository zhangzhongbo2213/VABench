"""Sparse candidate-set graphs for VLM-conditioned open-vocabulary grasping."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .grasp_candidates import (
    GraspCandidate,
    GraspIntent,
    candidate_equivalence_classes,
    candidates_equivalent_under_current_checks,
)


OPEN_VOCAB_GRASP_GRAPH_SCHEMA = "spatial.grasp_candidate_graph.v3"
VLM_GRASP_SUMMARY_SCHEMA = "spatial.vlm_grasp_candidate_summary.v1"


def build_open_vocab_grasp_candidate_graph(
    intent: GraspIntent,
    candidates: Sequence[GraspCandidate],
    *,
    world_state_version: int,
    view_assessment: Mapping[str, Any] | None = None,
    candidate_ranking: Mapping[str, Any] | None = None,
    view_evidence: Sequence[Mapping[str, Any]] = (),
    visual_grounding: Mapping[str, Any] | None = None,
    fusion_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact task/part/candidate graph without dense scene geometry."""

    if not candidates:
        raise ValueError("open-vocabulary grasp graph requires candidates")
    ordered = _ranked_candidates(candidates, candidate_ranking)
    assessment = dict(view_assessment or {})
    ranking = dict(candidate_ranking or {})
    target_ids = {candidate.target_id for candidate in ordered}
    if len(target_ids) != 1:
        raise ValueError("one candidate set must refer to exactly one target")
    target_id = next(iter(target_ids))
    graph_nodes: list[dict[str, Any]] = [
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
            "id": target_id,
            "node_type": "target_object_hypothesis",
            "semantic_type": intent.target,
            "attributes": {
                "description": intent.target,
                "semantic_ambiguity": intent.semantic_ambiguity,
            },
            "source": intent.source,
            "access": "inference_visible",
        },
        {
            "id": "grasp.candidate_set",
            "node_type": "grasp_candidate_set",
            "semantic_type": "task_conditioned_grasp_frames",
            "attributes": _candidate_set_attributes(ordered, assessment, ranking),
            "source": str(ranking.get("source", "inference_candidate_hypotheses")),
            "access": "inference_visible",
        },
    ]
    graph_edges: list[dict[str, Any]] = [
        _edge(
            "intent.requests_set",
            "intent.grasp",
            "grasp.candidate_set",
            "requests_grasp_candidates",
            intent.confidence,
            intent.source,
        ),
        _edge(
            "set.targets_object",
            "grasp.candidate_set",
            target_id,
            "targets_object",
            intent.confidence,
            intent.source,
        ),
        _edge(
            "set.supports_goal",
            "grasp.candidate_set",
            "task.goal",
            "supports_task_goal",
            None,
            "candidate_checks",
        ),
    ]

    region_ids: set[str] = set()
    for candidate in ordered:
        if candidate.region_id not in region_ids:
            region_ids.add(candidate.region_id)
            graph_nodes.append(
                {
                    "id": candidate.region_id,
                    "node_type": "semantic_part_hypothesis",
                    "semantic_type": candidate.semantic_role,
                    "entity_id": target_id,
                    "attributes": {
                        "role": candidate.semantic_role,
                        "preferred": candidate.semantic_role in intent.preferred_roles,
                        "avoided": candidate.semantic_role in intent.avoided_roles,
                    },
                    "source": intent.source,
                    "access": "inference_visible",
                }
            )
            graph_edges.append(
                _edge(
                    f"{candidate.region_id}.part_of",
                    candidate.region_id,
                    target_id,
                    "part_of",
                    intent.confidence,
                    intent.source,
                )
            )
        _append_candidate_subgraph(graph_nodes, graph_edges, candidate)

    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            equivalent = candidates_equivalent_under_current_checks(first, second)
            graph_edges.append(
                _edge(
                    f"{first.candidate_id}.alternative.{second.candidate_id}",
                    first.candidate_id,
                    second.candidate_id,
                    (
                        "equivalent_grasp_under_current_checks"
                        if equivalent
                        else "alternative_grasp_hypothesis"
                    ),
                    1.0,
                    "current_candidate_checks",
                    _candidate_difference(first, second),
                )
            )

    view_node_ids: list[str] = []
    for index, evidence in enumerate(view_evidence):
        view = str(evidence.get("view", f"view_{index}"))
        frame = evidence.get("frame_observation")
        if not isinstance(frame, Mapping):
            frame = (
                evidence.get("frame")
                if isinstance(evidence.get("frame"), Mapping)
                else {}
            )
        node_id = f"view.{index:03d}.{_safe_id(view)}"
        view_node_ids.append(node_id)
        graph_nodes.append(
            {
                "id": node_id,
                "node_type": "view_evidence",
                "semantic_type": view,
                "attributes": {
                    "frame_id": evidence.get("frame_id", index),
                    "confidence": frame.get("confidence"),
                    "visibility": frame.get("visibility"),
                    "fusion_status": evidence.get("fusion_status", "accepted"),
                    "camera_position_world_m": evidence.get("camera_position_world_m"),
                    "view_direction_world": evidence.get("view_direction_world"),
                    "camera_pose_source": evidence.get("camera_pose_source"),
                    "selection_reason": evidence.get("selection_reason"),
                },
                "source": str(frame.get("source", "runtime_rgbd")),
                "access": "inference_visible",
            }
        )
        graph_edges.append(
            _edge(
                f"{node_id}.observes_set",
                node_id,
                "grasp.candidate_set",
                "observes_candidate_set",
                _optional_probability(frame.get("confidence")),
                str(frame.get("source", "runtime_rgbd")),
            )
        )

    fusion_summary = _multiview_fusion_summary(fusion_diagnostics)
    if fusion_summary is not None:
        fusion_node_id = "fusion.multiview_consensus"
        graph_nodes.append(
            {
                "id": fusion_node_id,
                "node_type": "multiview_fusion_evidence",
                "semantic_type": fusion_summary["status"],
                "attributes": fusion_summary,
                "source": str(fusion_summary["method"]),
                "access": "inference_visible",
            }
        )
        graph_edges.append(
            _edge(
                "fusion.supports_candidate_set",
                fusion_node_id,
                "grasp.candidate_set",
                "supports_candidate_geometry",
                None,
                str(fusion_summary["method"]),
            )
        )
        contribution_factors = list(fusion_summary["view_contribution_factors"])
        for index, node_id in enumerate(view_node_ids):
            factor = (
                contribution_factors[index]
                if index < len(contribution_factors)
                else None
            )
            graph_edges.append(
                _edge(
                    f"{node_id}.contributes_to_fusion",
                    node_id,
                    fusion_node_id,
                    "contributes_to_multiview_fusion",
                    _optional_probability(factor),
                    str(fusion_summary["method"]),
                )
            )

    if visual_grounding is not None:
        graph_nodes.append(
            {
                "id": "observation.vlm_visual_grounding",
                "node_type": "visual_grounding_anchor",
                "semantic_type": str(visual_grounding["kind"]),
                "position_mean_world_m": list(visual_grounding["world_point_m"]),
                "position_covariance_m2": visual_grounding["position_covariance_m2"],
                "attributes": {
                    "pixel_uv": visual_grounding["pixel_uv"],
                    "normalized_pixel_uv": visual_grounding["normalized_pixel_uv"],
                    "target_description": visual_grounding.get("target_description"),
                    "confidence": visual_grounding["confidence"],
                },
                "source": str(visual_grounding["source"]),
                "access": "inference_visible",
            }
        )
        graph_edges.append(
            _edge(
                "grounding.anchors_target",
                "observation.vlm_visual_grounding",
                target_id,
                "anchors_target_hypothesis",
                float(visual_grounding["confidence"]),
                str(visual_grounding["source"]),
            )
        )

    summary = build_vlm_grasp_candidate_summary(
        intent,
        ordered,
        view_assessment=assessment,
        candidate_ranking=ranking,
        fusion_diagnostics=fusion_diagnostics,
    )
    graph = {
        "schema_version": OPEN_VOCAB_GRASP_GRAPH_SCHEMA,
        "access": "inference_visible",
        "query": "propose_grasp_candidates",
        "world_state_version": int(world_state_version),
        "coordinate_frame": "world",
        "units": {"position": "m", "angle": "rad"},
        "intent": intent.as_dict(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "candidate_count": len(ordered),
        "candidate_equivalence_classes": candidate_equivalence_classes(ordered),
        "top_candidates": [candidate.as_dict() for candidate in ordered],
        "candidate_ranking": ranking,
        "view_assessment": assessment,
        "multiview_fusion": fusion_summary,
        "vlm_summary": summary,
    }
    errors = validate_open_vocab_grasp_candidate_graph(graph)
    if errors:
        raise ValueError("invalid open-vocabulary grasp graph: " + "; ".join(errors))
    return graph


def build_vlm_grasp_candidate_summary(
    intent: GraspIntent,
    candidates: Sequence[GraspCandidate],
    *,
    view_assessment: Mapping[str, Any] | None = None,
    candidate_ranking: Mapping[str, Any] | None = None,
    fusion_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a metric-light explanation suitable for a VLM tool response."""

    ordered = _ranked_candidates(candidates, candidate_ranking)
    assessment = dict(view_assessment or {})
    ranking = dict(candidate_ranking or {})
    scores = _selection_scores(ordered, ranking)
    probabilities = _softmax_scores(scores)
    normalized_entropy = _normalized_entropy(probabilities)
    candidate_rows = []
    for candidate, probability in zip(ordered, probabilities):
        position_std_mm = 1000.0 * _maximum_std(candidate.position_covariance_m2)
        orientation_std_deg = math.degrees(
            _maximum_std(candidate.orientation_covariance_rad2)
        )
        predicted_success = next(
            (
                check.probability
                for check in candidate.checks
                if check.name == "predicted_execution_success"
            ),
            None,
        )
        if not candidate.eligible:
            status = "rejected_by_known_constraint"
        elif candidate.unknown_constraints:
            status = "needs_executability_check"
        else:
            status = "eligible_under_current_checks"
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "semantic_role": candidate.semantic_role,
                "status": status,
                "selection_utility": round(scores[len(candidate_rows)], 6),
                "relative_selection_mass": round(probability, 6),
                "calibrated_success_probability": (
                    round(float(predicted_success), 6)
                    if predicted_success is not None
                    else None
                ),
                "position_std_mm": round(position_std_mm, 2),
                "orientation_std_deg": round(orientation_std_deg, 2),
                "opening_width_mm": round(1000.0 * candidate.opening_width_m, 2),
                "failed_constraints": candidate.failed_constraints,
                "unresolved_checks": candidate.unknown_constraints,
                "hypothesis_kind": candidate.generation_parameters.get("type"),
            }
        )
    recommended_view = assessment.get("recommended_view")
    ranked_views = assessment.get("candidate_view_scores") or []
    view_reason = None
    if recommended_view is not None:
        view_row = next(
            (row for row in ranked_views if row.get("view") == recommended_view),
            None,
        )
        if isinstance(view_row, Mapping):
            reason = view_row.get("reason")
            if isinstance(reason, Mapping) and reason:
                view_reason = max(reason, key=lambda key: float(reason[key]))
    visual_sufficient = bool(assessment.get("visual_evidence_sufficient", False))
    execution_ready = bool(assessment.get("execution_ready", False))
    if recommended_view is not None:
        observation_action = "acquire_view"
        observation_reason = view_reason
    elif visual_sufficient:
        observation_action = "stop"
        observation_reason = "visual_evidence_sufficient"
    else:
        observation_action = "stop"
        observation_reason = assessment.get(
            "stop_reason", "no_remaining_candidate_view"
        )
    return {
        "schema_version": VLM_GRASP_SUMMARY_SCHEMA,
        "target": intent.target,
        "task_goal": intent.task_goal,
        "preferred_roles": list(intent.preferred_roles),
        "avoided_roles": list(intent.avoided_roles),
        "candidate_count": len(ordered),
        "eligible_candidate_count": sum(candidate.eligible for candidate in ordered),
        "recommended_candidate_id": ordered[0].candidate_id if ordered else None,
        "candidates": candidate_rows,
        "ambiguity": {
            "candidate_identity_entropy": round(normalized_entropy, 6),
            "candidate_distribution_calibrated": False,
            "top1_top2_margin": assessment.get("score_margin"),
            "top_candidates_equivalent_under_current_checks": assessment.get(
                "top_candidates_equivalent_under_current_checks"
            ),
            "semantic_ambiguity": intent.semantic_ambiguity,
            "visual_evidence_sufficient": assessment.get("visual_evidence_sufficient"),
            "multiview_consensus": _multiview_fusion_summary(fusion_diagnostics),
        },
        "next_observation": {
            "action": observation_action,
            "view": recommended_view,
            "reason": observation_reason,
            "recommended_view_if_budget_extended": assessment.get(
                "recommended_view_if_budget_extended"
            ),
            "unresolved_visual_reasons": list(assessment.get("reasons", ())),
        },
        "execution": {
            "ready": execution_ready,
            "blocked_by_visual_evidence": not visual_sufficient,
            "required_next_tool": (
                "spatial.verify_candidate_executability"
                if visual_sufficient and not execution_ready
                else None
            ),
            "unresolved_nonvisual_checks": list(
                assessment.get("unresolved_nonvisual_checks", ())
            ),
        },
    }


def _multiview_fusion_summary(
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    views = [str(value) for value in diagnostics.get("frame_views", ())]
    residuals = [float(value) for value in diagnostics.get("point_residuals_m", ())]
    factors = [float(value) for value in diagnostics.get("weight_factors", ())]
    residual_by_view_mm = {
        view: round(1000.0 * residuals[index], 2)
        for index, view in enumerate(views)
        if index < len(residuals)
    }
    return {
        "method": str(diagnostics.get("method", "unknown_multiview_fusion")),
        "status": str(diagnostics.get("consensus_status", "unknown")),
        "selected_views": [
            str(value) for value in diagnostics.get("selected_consensus_views", ())
        ],
        "suppressed_views": [
            str(value) for value in diagnostics.get("rejected_views", ())
        ],
        "view_contribution_factors": [round(value, 6) for value in factors],
        "point_residual_by_view_mm": residual_by_view_mm,
        "consistency_cutoff_mm": round(
            1000.0 * float(diagnostics.get("robust_cutoff_m", 0.0)), 2
        ),
        "same_size_competitor_weight_ratio": diagnostics.get(
            "same_size_competitor_weight_ratio"
        ),
        "access": "inference_visible",
    }


def validate_open_vocab_grasp_candidate_graph(graph: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if graph.get("schema_version") != OPEN_VOCAB_GRASP_GRAPH_SCHEMA:
        errors.append("unsupported schema_version")
    if graph.get("access") != "inference_visible":
        errors.append("graph must be inference_visible")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return [*errors, "nodes and edges must be lists"]
    node_ids = [node.get("id") for node in nodes if isinstance(node, Mapping)]
    if len(node_ids) != len(nodes) or len(set(node_ids)) != len(node_ids):
        errors.append("node IDs must be present and unique")
    node_types = {
        str(node.get("node_type")) for node in nodes if isinstance(node, Mapping)
    }
    required = {
        "task_goal",
        "grasp_intent",
        "target_object_hypothesis",
        "semantic_part_hypothesis",
        "grasp_candidate_set",
        "grasp_frame_candidate",
        "grasp_frame_point",
    }
    missing = sorted(required - node_types)
    if missing:
        errors.append("missing node types: " + ", ".join(missing))
    known = set(node_ids)
    for edge in edges:
        if not isinstance(edge, Mapping):
            errors.append("edges must be mappings")
            continue
        if edge.get("source") not in known or edge.get("target") not in known:
            errors.append("edge references an unknown node")
    candidate_nodes = [
        node for node in nodes if node.get("node_type") == "grasp_frame_candidate"
    ]
    if int(graph.get("candidate_count", -1)) != len(candidate_nodes):
        errors.append("candidate_count does not match candidate nodes")
    summary = graph.get("vlm_summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version") != VLM_GRASP_SUMMARY_SCHEMA
    ):
        errors.append("graph requires a VLM candidate summary")
    if "oracle/training_only" in str(graph):
        errors.append("training-only evidence leaked into graph")
    return errors


def _append_candidate_subgraph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    candidate: GraspCandidate,
) -> None:
    candidate_id = candidate.candidate_id
    nodes.append(
        {
            "id": candidate_id,
            "node_type": "grasp_frame_candidate",
            "semantic_type": candidate.semantic_role,
            "entity_id": candidate.target_id,
            "attributes": {
                "approach_axis_world": np.round(
                    candidate.approach_axis_world, 7
                ).tolist(),
                "closing_axis_world": np.round(
                    candidate.closing_axis_world, 7
                ).tolist(),
                "opening_width_m": round(candidate.opening_width_m, 7),
                "position_covariance_m2": np.round(
                    candidate.position_covariance_m2, 9
                ).tolist(),
                "orientation_covariance_rad2": np.round(
                    candidate.orientation_covariance_rad2, 9
                ).tolist(),
                "checks": {check.name: check.as_dict() for check in candidate.checks},
                "eligible": candidate.eligible,
                "score": round(candidate.score, 6),
                "score_confidence": round(candidate.score_confidence, 6),
                "generation_parameters": dict(candidate.generation_parameters),
            },
            "source": candidate.source,
            "access": candidate.access,
        }
    )
    edges.extend(
        [
            _edge(
                f"set.contains.{candidate_id}",
                "grasp.candidate_set",
                candidate_id,
                "contains_candidate",
                1.0,
                candidate.source,
            ),
            _edge(
                f"{candidate_id}.targets_part",
                candidate_id,
                candidate.region_id,
                "targets_semantic_part",
                _check_probability(candidate, "semantic_match"),
                _check_source(candidate, "semantic_match"),
            ),
            _edge(
                f"{candidate_id}.supports_goal",
                candidate_id,
                "task.goal",
                "supports_task_goal",
                None,
                "candidate_checks",
            ),
        ]
    )
    point_specs = (
        ("center", "grasp_center", candidate.center_world_m),
        ("left_contact", "left_contact", candidate.left_contact_world_m),
        ("right_contact", "right_contact", candidate.right_contact_world_m),
        ("pregrasp", "pregrasp_center", candidate.pregrasp_center_world_m),
    )
    for suffix, semantic_type, position in point_specs:
        node_id = f"{candidate_id}.{suffix}"
        nodes.append(
            {
                "id": node_id,
                "node_type": "grasp_frame_point",
                "semantic_type": semantic_type,
                "entity_id": candidate.target_id,
                "position_mean_world_m": np.round(position, 7).tolist(),
                "position_covariance_m2": np.round(
                    candidate.position_covariance_m2, 9
                ).tolist(),
                "source": candidate.source,
                "access": candidate.access,
            }
        )
        edges.append(
            _edge(
                f"{node_id}.component",
                node_id,
                candidate_id,
                "component_of_frame",
                1.0,
                candidate.source,
            )
        )
    edges.extend(
        [
            _edge(
                f"{candidate_id}.closing_span",
                f"{candidate_id}.left_contact",
                f"{candidate_id}.right_contact",
                "closing_span",
                _check_probability(candidate, "contact_width_consistency"),
                _check_source(candidate, "contact_width_consistency"),
                {"opening_width_m": round(candidate.opening_width_m, 7)},
            ),
            _edge(
                f"{candidate_id}.approach_path",
                f"{candidate_id}.pregrasp",
                f"{candidate_id}.center",
                "approach_path",
                _check_probability(candidate, "collision_free"),
                _check_source(candidate, "collision_free"),
            ),
        ]
    )


def _candidate_set_attributes(
    candidates: Sequence[GraspCandidate],
    assessment: Mapping[str, Any],
    ranking: Mapping[str, Any],
) -> dict[str, Any]:
    scores = _selection_scores(candidates, ranking)
    probabilities = _softmax_scores(scores)
    return {
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(candidate.eligible for candidate in candidates),
        "recommended_candidate_id": candidates[0].candidate_id,
        "candidate_identity_entropy": round(_normalized_entropy(probabilities), 6),
        "candidate_distribution_calibrated": False,
        "equivalence_classes": candidate_equivalence_classes(candidates),
        "visual_evidence_sufficient": assessment.get("visual_evidence_sufficient"),
        "execution_ready": assessment.get("execution_ready"),
        "recommended_view": assessment.get("recommended_view"),
    }


def _ranked_candidates(
    candidates: Sequence[GraspCandidate],
    ranking: Mapping[str, Any] | None,
) -> list[GraspCandidate]:
    scores = dict((ranking or {}).get("scores") or {})
    return sorted(
        candidates,
        key=lambda candidate: float(
            scores.get(candidate.candidate_id, candidate.score)
        ),
        reverse=True,
    )


def _selection_scores(
    candidates: Sequence[GraspCandidate], ranking: Mapping[str, Any]
) -> list[float]:
    scores = dict(ranking.get("scores") or {})
    return [
        float(scores.get(candidate.candidate_id, candidate.score))
        for candidate in candidates
    ]


def _softmax_scores(
    scores: Sequence[float], *, temperature: float = 0.1
) -> list[float]:
    if not scores:
        return []
    values = np.asarray(scores, dtype=np.float64) / temperature
    values -= float(np.max(values))
    values = np.exp(values)
    values /= float(np.sum(values))
    return values.tolist()


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    values = np.asarray(probabilities, dtype=np.float64)
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-12))))
    return entropy / math.log(len(values))


def _maximum_std(covariance: np.ndarray) -> float:
    return math.sqrt(
        max(
            0.0,
            float(np.max(np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64)))),
        )
    )


def _candidate_difference(
    first: GraspCandidate, second: GraspCandidate
) -> dict[str, Any]:
    closing_dot = abs(
        float(np.dot(first.closing_axis_world, second.closing_axis_world))
    )
    approach_dot = float(np.dot(first.approach_axis_world, second.approach_axis_world))
    return {
        "center_distance_m": round(
            float(np.linalg.norm(first.center_world_m - second.center_world_m)), 7
        ),
        "closing_axis_difference_deg": round(
            math.degrees(math.acos(np.clip(closing_dot, -1.0, 1.0))), 4
        ),
        "approach_axis_difference_deg": round(
            math.degrees(math.acos(np.clip(approach_dot, -1.0, 1.0))), 4
        ),
        "opening_width_difference_m": round(
            abs(first.opening_width_m - second.opening_width_m), 7
        ),
    }


def _check_probability(candidate: GraspCandidate, name: str) -> float | None:
    return next(
        (check.probability for check in candidate.checks if check.name == name), None
    )


def _check_source(candidate: GraspCandidate, name: str) -> str:
    return next(
        (check.source for check in candidate.checks if check.name == name),
        "not_evaluated",
    )


def _optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    probability = float(value)
    return probability if 0.0 <= probability <= 1.0 else None


def _edge(
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
        "probability": round(float(probability), 6)
        if probability is not None
        else None,
        "uncertainty": round(1.0 - float(probability), 6)
        if probability is not None
        else None,
        "measurement": dict(measurement or {}),
        "evidence_source": evidence_source,
        "access": "inference_visible",
    }


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)
