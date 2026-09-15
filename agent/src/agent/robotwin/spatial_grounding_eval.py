from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DIRECT_CANDIDATE_TOOL = "spatial.propose_grasp_candidates"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_luna_spatial_grounding_run(
    run_dir: Path,
    *,
    candidate_execution_result: Path | None = None,
    expected_target_xy_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate the VLM-to-spatial-tool chain without conflating policy success."""

    run_dir = run_dir.resolve()
    events = read_jsonl(run_dir / "events.jsonl")
    indexed_events = list(enumerate(events))
    spatial_events = [
        (index, event)
        for index, event in indexed_events
        if event.get("type") == "spatial_tool_result"
        and event.get("data", {}).get("tool") == DIRECT_CANDIDATE_TOOL
    ]
    if not spatial_events:
        raise ValueError(f"No {DIRECT_CANDIDATE_TOOL!r} result in {run_dir / 'events.jsonl'}")

    tool_index, tool_event = spatial_events[0]
    tool_data = dict(tool_event.get("data") or {})
    query_dir = run_dir / str(tool_data["query_dir"])
    query_result_path = query_dir / "query_result.json"
    query = read_json(query_result_path)

    action_events = [
        (index, event)
        for index, event in indexed_events
        if event.get("type") == "action"
    ]
    first_action = action_events[0] if action_events else None
    prequery_actions = [row for row in action_events if row[0] < tool_index]
    initial_gate_rejections = [
        (index, event)
        for index, event in indexed_events
        if index < tool_index
        and event.get("type") == "decision_rejected"
        and "Initial direct grasp-frame candidate query"
        in str(event.get("data", {}).get("error", ""))
    ]
    authorization_events = [
        (index, event)
        for index, event in indexed_events
        if event.get("type") == "candidate_executability_result"
        and event.get("data", {}).get("execution_authorized") is True
        and (first_action is None or index < first_action[0])
    ]

    grounding = dict(
        (query.get("geometry_provenance") or {}).get("visual_grounding") or {}
    )
    graph = dict(query.get("candidate_graph") or {})
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    grounding_nodes = [
        node for node in nodes if node.get("node_type") == "visual_grounding_anchor"
    ]
    grounding_edges = [
        edge for edge in edges if edge.get("relation") == "anchors_grasp_frame"
    ]
    top_candidates = list(query.get("top_candidates") or [])
    top_candidate = top_candidates[0] if top_candidates else {}

    support_height = grounding.get("anchor_height_above_support_m")
    support_check_applicable = support_height is not None
    support_foreground_pass = (
        float(support_height) >= 0.006 if support_check_applicable else None
    )
    candidate_before_first_action = (
        first_action is None or tool_index < first_action[0]
    ) and not prequery_actions
    world_frozen = bool(query.get("world_frozen"))
    world_delta = float(query.get("world_fingerprint_delta") or 0.0)
    observed_views = list(query.get("observed_view_sequence") or [])
    provenance = dict(query.get("geometry_provenance") or {})

    target_xy_error_m = None
    if expected_target_xy_m is not None:
        if len(expected_target_xy_m) != 2:
            raise ValueError("expected_target_xy_m must contain exactly two values")
        center = (top_candidate.get("frame") or {}).get("center_world_m")
        if not isinstance(center, Sequence) or len(center) < 2:
            raise ValueError("Top candidate does not contain center_world_m")
        target_xy_error_m = math.hypot(
            float(center[0]) - float(expected_target_xy_m[0]),
            float(center[1]) - float(expected_target_xy_m[1]),
        )

    execution_probe = _execution_probe_summary(candidate_execution_result)
    model_action = dict(first_action[1].get("data") or {}) if first_action else None
    execution_ready = bool(query.get("execution_ready"))
    atomic_candidate_action = bool(
        model_action
        and str(model_action.get("action", "")).startswith(
            "spatial.execute_grasp_candidate."
        )
    )
    online_authorized = bool(authorization_events)
    action_without_tool_authorization = bool(model_action) and not (
        online_authorized and atomic_candidate_action
    )
    eval_finish = next(
        (
            dict(event.get("data") or {})
            for event in reversed(events)
            if event.get("type") == "eval_finish"
        ),
        {},
    )

    protocol_pass = candidate_before_first_action
    grounding_pass = bool(grounding) and bool(grounding_nodes) and bool(grounding_edges)
    if support_check_applicable:
        grounding_pass = grounding_pass and bool(support_foreground_pass)
    graph_pass = bool(top_candidate) and all(
        semantic_type in {str(node.get("semantic_type")) for node in nodes}
        for semantic_type in ("grasp_center", "left_contact", "right_contact")
    )
    observation_pass = (
        len(observed_views) >= 2 and world_frozen and abs(world_delta) <= 1e-9
    )
    no_oracle_geometry = provenance.get("oracle_object_geometry_used") is False
    tool_chain_pass = all(
        (protocol_pass, grounding_pass, graph_pass, observation_pass, no_oracle_geometry)
    )

    return {
        "schema_version": "spatial.luna_grounding_run_evaluation.v1",
        "run_dir": str(run_dir),
        "query_result": str(query_result_path),
        "stage_verdicts": {
            "initial_tool_protocol_pass": protocol_pass,
            "vlm_visual_grounding_pass": grounding_pass,
            "active_multiview_observation_pass": observation_pass,
            "sparse_grasp_graph_pass": graph_pass,
            "no_oracle_object_geometry": no_oracle_geometry,
            "vlm_to_spatial_tool_chain_pass": tool_chain_pass,
            "offline_execution_probe_pass": execution_probe.get(
                "task_native_safe_success"
            ),
            "online_candidate_authorization_pass": (
                online_authorized if authorization_events else None
            ),
            "online_policy_task_success": eval_finish.get("success"),
        },
        "protocol": {
            "initial_action_rejected_before_tool": bool(initial_gate_rejections),
            "initial_gate_rejection_count": len(initial_gate_rejections),
            "prequery_environment_action_count": len(prequery_actions),
            "candidate_tool_before_first_environment_action": candidate_before_first_action,
            "candidate_tool_event_index": tool_index,
            "first_environment_action_event_index": (
                first_action[0] if first_action else None
            ),
        },
        "visual_grounding": {
            "source": grounding.get("source"),
            "kind": grounding.get("kind"),
            "target_description": grounding.get("target_description"),
            "target_box_normalized_xyxy": grounding.get(
                "target_box_normalized_xyxy"
            ),
            "requested_grasp_point_normalized_uv": grounding.get(
                "grasp_point_normalized_uv"
            ),
            "resolved_normalized_pixel_uv": grounding.get("normalized_pixel_uv"),
            "world_anchor_m": grounding.get("world_point_m"),
            "confidence": grounding.get("confidence"),
            "anchor_recovered_from_target_box": grounding.get(
                "anchor_recovered_from_target_box"
            ),
            "support_plane_z_m": grounding.get("support_plane_z_m"),
            "anchor_height_above_support_m": support_height,
            "support_foreground_check_pass": support_foreground_pass,
            "target_xy_error_m": target_xy_error_m,
            "expected_target_xy_m": (
                [float(value) for value in expected_target_xy_m]
                if expected_target_xy_m is not None
                else None
            ),
        },
        "active_observation": {
            "observed_view_sequence": observed_views,
            "view_count": len(observed_views),
            "world_frozen": world_frozen,
            "world_fingerprint_delta": world_delta,
        },
        "sparse_graph": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": sorted({str(node.get("node_type")) for node in nodes}),
            "relations": sorted({str(edge.get("relation")) for edge in edges}),
            "has_visual_grounding_anchor": bool(grounding_nodes),
            "has_anchors_grasp_frame_edge": bool(grounding_edges),
            "recommended_candidate_id": query.get("recommended_candidate_id"),
            "candidate_center_world_m": (top_candidate.get("frame") or {}).get(
                "center_world_m"
            ),
        },
        "authorization": {
            "candidate_query_execution_ready": execution_ready,
            "online_execution_authorized": online_authorized,
            "authorization_event_count": len(authorization_events),
            "recommended_action": query.get("recommended_action"),
            "unknown_constraints": top_candidate.get("unknown_constraints"),
            "first_model_action": model_action,
            "first_action_is_atomic_candidate_execution": atomic_candidate_action,
            "action_without_tool_authorization": action_without_tool_authorization,
        },
        "offline_execution_probe": execution_probe,
        "online_evaluation": {
            "task_success": eval_finish.get("success"),
            "result": eval_finish.get("result"),
        },
        "interpretation": {
            "tool_chain_and_policy_are_separate": True,
            "summary": _summary_text(
                tool_chain_pass=tool_chain_pass,
                execution_probe=execution_probe,
                action_without_tool_authorization=action_without_tool_authorization,
                online_success=eval_finish.get("success"),
            ),
        },
    }


def _execution_probe_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "execution_success": None,
            "task_native_safe_success": None,
            "object_height_change_m": None,
        }
    payload = read_json(path.resolve())
    result = dict(payload.get("result") or {})
    return {
        "available": True,
        "path": str(path.resolve()),
        "candidate_id": result.get("candidate_id"),
        "ik_reachable": result.get("ik_reachable"),
        "bilateral_contact": result.get("bilateral_contact"),
        "lifted_from_support": result.get("lifted_from_support"),
        "execution_success": result.get("execution_success"),
        "task_native_safe_success": result.get("task_native_safe_success"),
        "object_height_change_m": result.get("object_height_change_m"),
        "collision_within_expert_baseline": result.get(
            "collision_within_expert_baseline"
        ),
    }


def _summary_text(
    *,
    tool_chain_pass: bool,
    execution_probe: Mapping[str, Any],
    action_without_tool_authorization: bool,
    online_success: Any,
) -> str:
    parts = [
        "VLM-to-spatial-tool chain passed."
        if tool_chain_pass
        else "VLM-to-spatial-tool chain did not pass all checks."
    ]
    if execution_probe.get("available"):
        parts.append(
            "The independently replayed candidate was task-native safe."
            if execution_probe.get("task_native_safe_success")
            else "The independently replayed candidate was not task-native safe."
        )
    if action_without_tool_authorization:
        parts.append(
            "The model nevertheless issued an environment action while the tool marked "
            "the candidate execution_ready=false."
        )
    if online_success is False:
        parts.append("The online policy episode did not complete the task.")
    return " ".join(parts)
