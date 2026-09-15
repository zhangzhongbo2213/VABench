"""Stateful learned multi-view tool for the ``verify_pregrasp`` query."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .active_belief import ObservationGraph, SpatialBelief
from .grasp_outcome import extract_outcome_features, load_outcome_checkpoint
from .learned_pregrasp import (
    PREGRASP_EDGE_IDS,
    PregraspObservationNet,
    derived_edges_from_belief_graph,
    predict_observation_graph_from_inputs,
    prepare_inference_inputs,
)
from .learned_view_selector import load_view_ranker, rerank_candidate_views
from .pregrasp_graph import score_pregrasp_candidate_views


@dataclass(frozen=True)
class PregraspQueryState:
    belief: SpatialBelief | None
    visited_views: tuple[str, ...]
    query_context: dict[str, float]


class LearnedPregraspTool:
    """Own model inference, graph fusion, action gate, and next-view choice."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        calibration: str | Path | Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        required_confidence: float = 0.75,
        minimum_evidence_views: int = 2,
        view_ranker_checkpoint: str | Path | None = None,
        outcome_checkpoint: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        value = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        self.model = PregraspObservationNet(
            pretrained_path=None,
            freeze_stem=value["model_config"]["freeze_stem"],
        )
        self.model.load_state_dict(value["model_state"])
        self.model.to(self.device)
        self.model.eval()
        if isinstance(calibration, (str, Path)):
            calibration = json.loads(Path(calibration).read_text(encoding="utf-8"))
        self.relation_thresholds = dict((calibration or {}).get("relation_thresholds", {}))
        self.required_confidence = float(required_confidence)
        self.minimum_evidence_views = int(minimum_evidence_views)
        self.view_ranker = (
            load_view_ranker(view_ranker_checkpoint, device=self.device)
            if view_ranker_checkpoint is not None
            else None
        )
        if outcome_checkpoint is None:
            self.outcome_model = None
            self.outcome_config: dict[str, Any] = {}
        else:
            self.outcome_model, self.outcome_config = load_outcome_checkpoint(
                str(outcome_checkpoint),
                device=self.device,
            )
        self.belief: SpatialBelief | None = None
        self.visited_views: list[str] = []
        self.query_context: dict[str, float] = {}

    def start_query(
        self,
        *,
        robot_kinematics: Mapping[str, Any],
        world_state_version: int,
        query_context: Mapping[str, float] | None = None,
    ) -> None:
        axes = robot_kinematics["query_axes"]
        self.belief = SpatialBelief(
            query="verify_pregrasp",
            world_state_version=int(world_state_version),
            required_edge_ids=PREGRASP_EDGE_IDS,
            query_axes={
                "closing_axis_world": axes["closing_axis_world"],
                "support_normal_world": axes["support_normal_world"],
            },
            deterministic_nodes=robot_kinematics["nodes"],
        )
        self.visited_views = []
        self.query_context = {
            str(key): float(value) for key, value in (query_context or {}).items()
        }

    def snapshot_query_state(self) -> PregraspQueryState:
        """Return an isolated state for training-only counterfactual observations."""

        return PregraspQueryState(
            belief=deepcopy(self.belief),
            visited_views=tuple(self.visited_views),
            query_context=dict(self.query_context),
        )

    def restore_query_state(self, state: PregraspQueryState) -> None:
        self.belief = deepcopy(state.belief)
        self.visited_views = list(state.visited_views)
        self.query_context = dict(state.query_context)

    def observe(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        camera: Mapping[str, Any],
        robot_kinematics: Mapping[str, Any],
        view: str,
        frame_id: int,
        candidates: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        if self.belief is None:
            self.start_query(
                robot_kinematics=robot_kinematics,
                world_state_version=int(robot_kinematics["world_state_version"]),
            )
        assert self.belief is not None
        rgbd, parsed_camera, kinematics = prepare_inference_inputs(
            rgb,
            depth_m,
            camera,
            robot_kinematics,
        )
        observation, decoded = predict_observation_graph_from_inputs(
            self.model,
            rgbd=rgbd,
            camera=parsed_camera,
            kinematics=kinematics,
            robot_kinematics=robot_kinematics,
            frame_id=frame_id,
            view=view,
            world_state_version=self.belief.world_state_version,
            device=self.device,
        )
        node_observation = ObservationGraph(
            frame_id=observation.frame_id,
            view=observation.view,
            nodes=observation.nodes,
            edges=[],
            query_axes=observation.query_axes,
            source=observation.source,
        )
        graph = self.belief.update(node_observation)
        camera_pose = np.asarray(parsed_camera["camera_pose_world"], dtype=np.float64)
        view_direction = camera_pose[:3, 0]
        edges, object_axis = derived_edges_from_belief_graph(
            graph,
            view_direction_world=view_direction,
        )
        if object_axis is not None:
            self.belief.query_axes["object_axis_world"] = object_axis
        if edges:
            graph = self.belief.replace_derived_edges(edges, frame_id=frame_id, view=view)
        self.visited_views.append(str(view))
        gate = self.belief.pregrasp_gate(
            required_confidence=self.required_confidence,
            minimum_evidence_views=self.minimum_evidence_views,
            relation_thresholds=self.relation_thresholds,
        )
        outcome = self._predict_grasp_outcome(gate=gate, graph=graph)
        ranking = []
        if gate["verdict"] == "uncertain":
            ranking = score_pregrasp_candidate_views(
                edge_uncertainty=self.belief.edge_uncertainty(),
                query_axes=self.belief.query_axes,
                current_view_direction_world=view_direction,
                current_camera_position_world=camera_pose[:3, 3],
                candidates=candidates,
                visited_views=self.visited_views,
            )
            if self.view_ranker is not None:
                ranking = rerank_candidate_views(
                    self.view_ranker,
                    ranking,
                    edge_uncertainty=self.belief.edge_uncertainty(),
                    device=self.device,
                    state_context=self.view_state_context(
                        gate=gate,
                        outcome=outcome,
                        remaining_candidate_count=len(ranking),
                    ),
                )
        return tool_result(
            gate=gate,
            graph=graph,
            observation=observation,
            decoded=decoded,
            ranking=ranking,
            outcome=outcome,
        )

    def view_state_context(
        self,
        *,
        gate: Mapping[str, Any],
        outcome: Mapping[str, Any] | None,
        remaining_candidate_count: int,
    ) -> dict[str, float]:
        recovery_cycle = float(self.query_context.get("recovery_cycle", 0.0))
        cumulative_recovery = float(
            self.query_context.get("cumulative_recovery_translation_m", 0.0)
        )
        return {
            "recovery_cycle_fraction": float(np.clip(recovery_cycle / 4.0, 0.0, 1.0)),
            "cumulative_recovery_fraction": float(
                np.clip(cumulative_recovery / 0.18, 0.0, 1.0)
            ),
            "observed_view_fraction": float(np.clip(len(self.visited_views) / 6.0, 0.0, 1.0)),
            "remaining_view_fraction": float(
                np.clip(remaining_candidate_count / 5.0, 0.0, 1.0)
            ),
            "gate_confidence": float(np.clip(gate.get("confidence", 0.0), 0.0, 1.0)),
            "outcome_probability": float(
                np.clip((outcome or {}).get("probability", 0.5), 0.0, 1.0)
            ),
            "outcome_uncertainty": float(
                np.clip((outcome or {}).get("uncertainty", 1.0), 0.0, 1.0)
            ),
            "post_recovery": float(recovery_cycle > 0.0),
        }

    def _predict_grasp_outcome(
        self,
        *,
        gate: Mapping[str, Any],
        graph: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if self.outcome_model is None:
            return None
        relations = graph_relations(graph)
        features = extract_outcome_features(
            {**gate, "relations": relations},
            observed_view_count=len(self.visited_views),
        )
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.outcome_model(tensor))[0].item())
        threshold = float(self.outcome_config.get("decision_threshold", 0.5))
        return {
            "relation": "grasp_success_if_execute",
            "probability": round(probability, 6),
            "uncertainty": round(1.0 - abs(2.0 * probability - 1.0), 6),
            "decision_threshold": threshold,
            "supports_execute": probability >= threshold,
            "evidence_views": list(self.visited_views),
            "advisory_only": True,
        }


def tool_result(
    *,
    gate: Mapping[str, Any],
    graph: Mapping[str, Any],
    observation: ObservationGraph,
    decoded: Mapping[str, Any],
    ranking: list[dict[str, Any]],
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relations = graph_relations(graph)
    if outcome is not None:
        relations["grasp_success_if_execute"] = dict(outcome)
    missing = list(gate.get("missing_relations", gate.get("failed_relations", [])))
    if gate["verdict"] == "uncertain" and ranking:
        recommended = {"tool": "camera.select_view", "view": ranking[0]["view"]}
        stop_reason = "additional_evidence_required"
    elif gate["verdict"] == "execute":
        recommended = {"tool": "robot.allow_gripper_close"}
        stop_reason = "confidence_reached"
    elif gate["verdict"] == "adjust":
        recommended = {"tool": "robot.adjust_gripper_pose", "failed_relations": missing}
        stop_reason = "confident_violation"
    else:
        recommended = {"tool": "stop"}
        stop_reason = "no_unvisited_candidate_view"
    return {
        "schema_version": "spatial.verify_pregrasp.tool_result.v1",
        "query": "verify_pregrasp",
        "verdict": gate["verdict"],
        "confidence": gate["confidence"],
        "relations": relations,
        "grasp_success_if_execute": dict(outcome) if outcome is not None else None,
        "evidence_frames": [item["frame_id"] for item in graph["observation_history"]],
        "evidence_views": [item["view"] for item in graph["observation_history"]],
        "missing_evidence": missing,
        "recommended_action": recommended,
        "stop_reason": stop_reason,
        "observation": observation.as_dict(),
        "decoded_keypoints": decoded["keypoints"],
        "candidate_view_scores": ranking,
        "belief_graph": dict(graph),
    }


def graph_relations(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(edge["id"]): {
            "state": edge.get("state", "unknown"),
            "probability": edge.get("probability", 0.5),
            "uncertainty": edge.get("uncertainty", 1.0),
            "measurement": edge.get("measurement", {}),
            "evidence_views": edge.get("evidence_views", []),
        }
        for edge in graph["edges"]
    }
