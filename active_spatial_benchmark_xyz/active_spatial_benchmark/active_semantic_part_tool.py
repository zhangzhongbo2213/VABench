"""Reusable active multi-view semantic-part grasp candidate tool."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import torch

from .analytic_view_control import rank_runtime_candidates_with_analytic_control
from .camera_control import DEFAULT_CAMERA_BOUNDS, DEFAULT_CAMERA_LAYOUT, camera_pose_for_view
from .camera_layout import CameraLayout
from .env import robotwin_cwd
from .expert_event_view_dataset import RELATION_NAMES
from .expert_event_view_ranker import (
    PairwiseLinearEventViewRanker,
    event_view_feature_vector,
    score_runtime_candidate_views,
)
from .grasp_candidate_neural_ranker import IntentEmbeddingStore
from .grasp_candidates import (
    CandidateGenerationConfig,
    GraspIntent,
    assess_candidate_view_sufficiency,
    build_grasp_candidate_graph,
    fuse_semantic_part_region_nodes,
    generate_obb_grasp_candidates,
    geometry_from_semantic_regions,
    semantic_regions_from_sparse_graph,
)
from .semantic_part_region_dataset import prepare_semantic_part_rgbd
from .semantic_part_region_model import (
    SemanticPartRegionNet,
    predict_semantic_part_region_node,
)
from .safe_candidate_view_ranker import (
    LEGACY_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    PREVIOUS_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    SafeCandidateViewRanker,
)
from .selective_risk_controller import SelectiveRiskController
from .view_visibility_model import attach_visibility_priors_to_candidates
from .view_value_of_information import rank_views_by_value_of_information


CANDIDATE_VIEW_DIRECTIONS = {
    "topdown": [0.0, 0.0, -1.0],
    "side": [-1.0, 0.0, 0.0],
    "front_side_45": [-1.0, 1.0, 0.0],
    "side_top_45": [-1.0, 0.0, -1.0],
    "oblique_45": [-1.0, 1.0, -1.0],
}
# Declared layout constant, not a measurement: converts a catalogue view
# direction into a camera position so range features are defined.
SEMANTIC_PART_CAMERA_STANDOFF_M = 0.6


class ActiveSemanticPartCandidateTool:
    def __init__(
        self,
        checkpoint: str | Path,
        intent_embeddings: str | Path,
        *,
        device: str | torch.device | None = None,
        image_size: tuple[int, int] = (240, 320),
        view_ranker_checkpoint: str | Path | None = None,
        view_selection_mode: str = "analytic",
        selective_risk_controller_checkpoint: str | Path | None = None,
        selective_risk_controller_domain: str | None = None,
        camera_layout: CameraLayout | Mapping[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint).resolve()
        self.embedding_path = Path(intent_embeddings).resolve()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = image_size
        if view_selection_mode not in {"analytic", "shadow"}:
            raise ValueError("view_selection_mode must be analytic or shadow")
        self.view_selection_mode = view_selection_mode
        if isinstance(camera_layout, Mapping):
            camera_layout = CameraLayout.from_mapping(camera_layout)
        if camera_layout is not None and not isinstance(camera_layout, CameraLayout):
            raise TypeError("camera_layout must be a CameraLayout or mapping")
        self.camera_layout = camera_layout
        self.view_ranker_path = (
            Path(view_ranker_checkpoint).resolve()
            if view_ranker_checkpoint is not None
            else None
        )
        self.selective_risk_controller_path = (
            Path(selective_risk_controller_checkpoint).resolve()
            if selective_risk_controller_checkpoint is not None
            else None
        )
        self.selective_risk_controller_domain = (
            str(selective_risk_controller_domain).strip()
            if selective_risk_controller_domain is not None
            else None
        )
        if self.selective_risk_controller_path is not None:
            if not self.selective_risk_controller_domain:
                raise ValueError(
                    "selective-risk controller requires an explicit declared domain"
                )
            if not self.selective_risk_controller_path.is_file():
                raise FileNotFoundError(self.selective_risk_controller_path)
        self.selective_risk_controller = None
        if self.selective_risk_controller_path is not None:
            self.selective_risk_controller = SelectiveRiskController.from_dict(
                json.loads(
                    self.selective_risk_controller_path.read_text(encoding="utf-8")
                )
            )
        if view_selection_mode == "shadow" and self.view_ranker_path is None:
            raise ValueError("shadow view selection requires a view-ranker checkpoint")
        self.view_ranker = None
        self.view_ranker_supported_views: set[str] = set()
        self.view_ranker_metadata: dict[str, Any] = {}
        if self.view_ranker_path is not None:
            ranker_payload = json.loads(
                self.view_ranker_path.read_text(encoding="utf-8")
            )
            if ranker_payload.get("schema_version") in {
                SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
                PREVIOUS_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
                LEGACY_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
            }:
                self.view_ranker = SafeCandidateViewRanker.from_dict(ranker_payload)
            else:
                self.view_ranker = PairwiseLinearEventViewRanker.from_dict(ranker_payload)
            self.view_ranker_metadata = dict(
                ranker_payload.get("training_metadata", {})
            )
            if not self.view_ranker.feature_config.supports_unseen_view_labels:
                self.view_ranker_supported_views = set(
                    str(value)
                    for value in self.view_ranker_metadata.get("trained_views", ())
                )
        if self.selective_risk_controller is not None:
            if not isinstance(self.view_ranker, SafeCandidateViewRanker):
                raise ValueError(
                    "selective-risk controller requires a SafeCandidateViewRanker"
                )
            if self.selective_risk_controller.feature_mode != (
                self.view_ranker.feature_config.mode
            ) or tuple(self.selective_risk_controller.feature_names or ()) != tuple(
                self.view_ranker.feature_names
            ):
                raise ValueError(
                    "selective-risk controller and view ranker feature contracts differ"
                )
        payload = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            payload.get("schema_version")
            != "phase11.semantic_part_region_checkpoint.v1"
        ):
            raise ValueError("unsupported semantic part region checkpoint")
        self.embeddings = IntentEmbeddingStore.load(self.embedding_path)
        if payload.get("encoder_id") != self.embeddings.encoder_id:
            raise ValueError("semantic part checkpoint and embedding encoder_id differ")
        config = payload["model_config"]
        self.model = SemanticPartRegionNet(
            text_embedding_dim=int(config["text_embedding_dim"]),
            feature_dim=int(config["feature_dim"]),
            freeze_stem=bool(config["freeze_stem"]),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()

    def propose(
        self,
        env: Any,
        intent: GraspIntent,
        *,
        output_dir: str | Path,
        world_state_version: int,
        max_additional_views: int = 2,
        max_candidates: int = 3,
    ) -> dict[str, Any]:
        if max_additional_views < 0 or max_candidates < 1:
            raise ValueError("invalid active semantic candidate budget")
        embedding = torch.from_numpy(self.embeddings.for_intent(intent.as_dict()))
        output_dir = Path(output_dir).resolve()
        views_dir = output_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        initial_camera_pose = deepcopy(env.camera.get_camera().entity.get_pose())
        target = _target_actor(env, intent.target)
        target_pose = target.get_pose()
        target_before = np.concatenate([target_pose.p, target_pose.q]).astype(
            np.float64
        )
        observed_nodes = []
        observed_directions = []
        visited = []
        history = []
        next_view = "current"
        final_candidates = []
        final_assessment: dict[str, Any] = {}
        final_graph: dict[str, Any] = {}
        layout = _runtime_camera_layout(env, self.camera_layout)
        camera_controller = getattr(env, "camera", None)
        previous_camera_layout = getattr(camera_controller, "camera_layout", None)
        if self.camera_layout is not None and camera_controller is not None:
            camera_controller.camera_layout = self.camera_layout
        try:
            for observation_index in range(max_additional_views + 1):
                capture = _capture_view(env, next_view, initial_camera_pose)
                visited.append(next_view)
                observed_directions.append(capture["view_direction_world"])
                image_path = views_dir / f"{observation_index:02d}_{next_view}.png"
                imageio.imwrite(image_path, capture["rgb"])
                rgbd = prepare_semantic_part_rgbd(
                    capture["rgb"],
                    capture["depth_mm"] / 1000.0,
                    self.image_size,
                )
                node = predict_semantic_part_region_node(
                    self.model,
                    rgbd,
                    embedding,
                    {
                        "intrinsic_cv": capture["intrinsic_cv"],
                        "extrinsic_cv": capture["extrinsic_cv"],
                    },
                    original_image_size=capture["rgb"].shape[:2],
                    region_id=f"object.{intent.target}.region.preferred_0",
                    entity_id=f"object.{intent.target}",
                    semantic_role=intent.preferred_roles[0],
                    device=self.device,
                )
                node["view"] = next_view
                node["frame_id"] = observation_index
                node["camera_position_world_m"] = capture["camera_position_world_m"]
                node["view_direction_world"] = capture["view_direction_world"]
                node["camera_pose_source"] = capture["camera_position_source"]
                node["selection_reason"] = _view_selection_reason(
                    next_view, final_assessment, observation_index
                )
                observed_nodes.append(node)
                fused = fuse_semantic_part_region_nodes(observed_nodes)
                belief = {
                    "schema_version": "spatial.semantic_part_multiview_belief.v1",
                    "access": "inference_visible",
                    "nodes": [fused],
                    "edges": [],
                }
                regions = semantic_regions_from_sparse_graph(
                    intent, belief, object_id=f"object.{intent.target}"
                )
                geometry = geometry_from_semantic_regions(
                    regions, object_id=f"object.{intent.target}"
                )
                final_candidates = generate_obb_grasp_candidates(
                    intent,
                    geometry,
                    regions=regions,
                    config=CandidateGenerationConfig(max_candidates=max_candidates),
                )
                remaining = _with_layout_camera_positions(
                    env,
                    [
                        {
                            "view": name,
                            "view_direction_world": _layout_view_direction(
                                layout, name, view_arm=str(env.active_arm)
                            ),
                            "move_cost": 0.2,
                        }
                        for name in layout.view_names
                        if name not in visited
                    ],
                    layout=layout,
                )
                final_assessment = assess_candidate_view_sufficiency(
                    final_candidates,
                    observed_view_directions_world=observed_directions,
                    candidate_views=remaining,
                )
                shadow_ranking = self._shadow_view_ranking(
                    intent,
                    fused=fused,
                    candidates=final_candidates,
                    observed_directions=observed_directions,
                    candidate_views=remaining,
                    analytic_recommended_view=final_assessment.get("recommended_view"),
                )
                if shadow_ranking is not None:
                    final_assessment["learned_view_ranker_shadow"] = shadow_ranking
                final_graph = build_grasp_candidate_graph(
                    intent,
                    geometry,
                    regions,
                    final_candidates,
                    world_state_version=world_state_version,
                    top_k=max_candidates,
                    view_assessment=final_assessment,
                    view_evidence=fused.get("evidence", ()),
                )
                history.append(
                    {
                        "frame_id": observation_index,
                        "view": next_view,
                        "image": str(image_path.relative_to(output_dir)),
                        "camera_position_world_m": capture["camera_position_world_m"],
                        "view_direction_world": capture["view_direction_world"],
                        "camera_pose_source": capture["camera_position_source"],
                        "selection_reason": node["selection_reason"],
                        "region_observation": node,
                        "fused_region": fused,
                        "view_assessment": final_assessment,
                    }
                )
                recommended = final_assessment.get("recommended_view")
                if observation_index >= max_additional_views or not recommended:
                    break
                next_view = str(recommended)
        finally:
            with robotwin_cwd():
                env.camera.get_camera().entity.set_pose(initial_camera_pose)
                env.task._update_render()
            if self.camera_layout is not None and camera_controller is not None:
                camera_controller.camera_layout = previous_camera_layout
        target_pose_after = target.get_pose()
        target_after = np.concatenate(
            [target_pose_after.p, target_pose_after.q]
        ).astype(np.float64)
        world_delta = float(np.max(np.abs(target_after - target_before)))
        top = final_candidates[0] if final_candidates else None
        visual_evidence_sufficient = bool(
            final_assessment.get("visual_evidence_sufficient", False)
        )
        execution_ready = bool(final_assessment.get("execution_ready", False))
        result = {
            "schema_version": "spatial.active_semantic_grasp_candidates.v1",
            "query": "propose_grasp_candidates",
            "verdict": (
                "ready"
                if top is not None and execution_ready
                else "uncertain"
                if top is not None
                else "no_candidate"
            ),
            "confidence": (
                round(float(top.score * top.score_confidence), 6)
                if top is not None
                else 0.0
            ),
            "intent": intent.as_dict(),
            "top_candidates": [candidate.as_dict() for candidate in final_candidates],
            "recommended_candidate_id": top.candidate_id if top is not None else None,
            "view_assessment": final_assessment,
            "visual_evidence_sufficient": visual_evidence_sufficient,
            "execution_ready": execution_ready,
            "recommended_action": (
                {
                    "tool": "camera.select_view",
                    "view": final_assessment.get("recommended_view"),
                }
                if final_assessment.get("recommended_view")
                else {
                    "tool": "spatial.refine_candidate_executability",
                    "candidate_id": top.candidate_id,
                    "missing_checks": top.unknown_constraints,
                }
                if top is not None and top.unknown_constraints
                else {
                    "tool": "robot.plan_to_grasp_candidate",
                    "candidate_id": top.candidate_id,
                }
                if top is not None
                else {"tool": "stop"}
            ),
            "candidate_graph": final_graph,
            "camera_layout": layout.as_dict(),
            "observed_view_sequence": visited,
            "evidence_views": visited,
            "evidence_frames": list(range(len(visited))),
            "history": history,
            "world_frozen": world_delta < 1e-6,
            "world_fingerprint_delta": world_delta,
            "geometry_provenance": {
                "semantic_region": "learned_rgbd_text_conditioned_multiview_belief",
                "candidate_geometry": "derived_from_learned_semantic_part_regions",
                "intent_embedding_encoder": self.embeddings.encoder_id,
                "oracle_object_geometry_used": False,
                "ik": "not_evaluated",
                "collision": "not_evaluated",
                "execution_success": "not_evaluated",
                "view_selection": (
                    "legacy_relation_sufficiency_with_ranker_and_analytic_control_shadow"
                    if self.view_selection_mode == "shadow"
                    else "legacy_relation_sufficiency"
                ),
                "camera_layout_id": layout.layout_id,
            },
        }
        (output_dir / "query_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def _shadow_view_ranking(
        self,
        intent: GraspIntent,
        *,
        fused: dict[str, Any],
        candidates: list[Any],
        observed_directions: list[list[float]],
        candidate_views: list[dict[str, Any]],
        analytic_recommended_view: str | None,
    ) -> dict[str, Any] | None:
        if (
            self.view_selection_mode != "shadow"
            or self.view_ranker is None
            or not candidates
        ):
            return None
        supported = [
            row
            for row in candidate_views
            if not self.view_ranker_supported_views
            or str(row.get("view")) in self.view_ranker_supported_views
        ]
        top = candidates[0]
        uncertainty = _learned_relation_uncertainty(fused, top)
        analytic_control: dict[str, Any]
        calibration = getattr(self.view_ranker, "analytic_fisher_context", None)
        if not candidate_views:
            analytic_control = {
                "schema_version": "spatial.analytic_view_control_policy.v1",
                "status": "no_remaining_candidate_view",
                "selected_view": None,
                "ranked_candidates": [],
            }
        elif calibration is None:
            analytic_control = {
                "schema_version": "spatial.analytic_view_control_policy.v1",
                "status": "blocked_missing_sensor_calibration",
                "selected_view": None,
                "ranked_candidates": [],
                "reason": (
                    "checkpoint_has_no_calibrated_analytic_fisher_context; "
                    "sensor noise is not guessed at runtime"
                ),
            }
        else:
            try:
                left_contact = getattr(top, "left_contact_world_m", None)
                right_contact = getattr(top, "right_contact_world_m", None)
                if left_contact is not None and right_contact is not None:
                    contact_separation_m = float(
                        np.linalg.norm(
                            np.asarray(right_contact, dtype=np.float64)
                            - np.asarray(left_contact, dtype=np.float64)
                        )
                    )
                    analytic_candidate_views = attach_visibility_priors_to_candidates(
                        candidate_views,
                        target_center_world=np.asarray(top.center_world_m).tolist(),
                        closing_axis_world=top.closing_axis_world,
                        approach_axis_world=top.approach_axis_world,
                        focal_length_px=calibration.focal_length_px,
                        contact_separation_m=contact_separation_m,
                        position_covariance_m2=np.asarray(
                            top.position_covariance_m2
                        ).tolist(),
                    )
                else:
                    analytic_candidate_views = candidate_views
                analytic_control = rank_runtime_candidates_with_analytic_control(
                    prior_covariance_m2=np.asarray(top.position_covariance_m2).tolist(),
                    closing_axis_world=top.closing_axis_world,
                    approach_axis_world=top.approach_axis_world,
                    relation_uncertainty=uncertainty,
                    sensor_model=calibration.sensor_model,
                    focal_length_px=calibration.focal_length_px,
                    default_range_m=calibration.camera_standoff_m,
                    belief_centroid_world=np.asarray(top.center_world_m).tolist(),
                    candidates=analytic_candidate_views,
                )
            except ValueError as error:
                analytic_control = {
                    "schema_version": "spatial.analytic_view_control_policy.v1",
                    "status": "blocked_invalid_analytic_inputs",
                    "selected_view": None,
                    "ranked_candidates": [],
                    "reason": str(error),
                }
        analytic_control["voi_shadow"] = rank_views_by_value_of_information(
            candidates=analytic_control.get("ranked_candidates", []),
            relation_uncertainty=uncertainty,
        )
        rows = (
            score_runtime_candidate_views(
                self.view_ranker,
                intent.as_dict(),
                current_relation_uncertainty=uncertainty,
                closing_axis_world=top.closing_axis_world,
                approach_axis_world=top.approach_axis_world,
                observed_view_directions_world=observed_directions,
                candidate_views=supported,
                belief_centroid_world=np.asarray(top.center_world_m).tolist(),
                prior_covariance_m2=np.asarray(top.position_covariance_m2).tolist(),
                camera_standoff_m=SEMANTIC_PART_CAMERA_STANDOFF_M,
            )
            if supported
            else []
        )
        selected = str(rows[0]["view"]) if rows else None
        safe_policy = isinstance(self.view_ranker, SafeCandidateViewRanker)
        policy = None
        if safe_policy and rows:
            policy_task = str(getattr(intent, "task_goal", "")).strip() or None
            policy_features = np.stack(
                [
                    event_view_feature_vector(
                        intent.as_dict(),
                        row["view"],
                        row["input_features"],
                        feature_config=self.view_ranker.feature_config,
                    )
                    for row in rows
                ]
            )
            policy = self.view_ranker.policy_decision(
                policy_features,
                [str(row["view"]) for row in rows],
                task=policy_task,
            )
            if self.selective_risk_controller is not None:
                policy = self.selective_risk_controller.policy_decision(
                    self.view_ranker,
                    policy_features,
                    [str(row["view"]) for row in rows],
                    domain=self.selective_risk_controller_domain,
                    task=policy_task,
                )
            selected = policy.get("view") or policy.get("ranked_view")
        return {
            "mode": "shadow",
            "control_policy": (
                "learned_safe_candidate_view_ranker_with_selective_risk_veto_evaluation_only"
                if safe_policy and self.selective_risk_controller is not None
                else "learned_safe_candidate_view_ranker_evaluation_only"
                if safe_policy
                else "legacy_analytic_sufficiency"
            ),
            "checkpoint": str(self.view_ranker_path),
            "target_scope": self.view_ranker_metadata.get("target_scope"),
            "training_sample_schemas": self.view_ranker_metadata.get(
                "sample_schemas", []
            ),
            "runtime_feature_adapter": (
                "semantic_region_and_candidate_frame_pose_conditioned_v2"
            ),
            "feature_mode": self.view_ranker.feature_config.mode,
            "camera_standoff_m": SEMANTIC_PART_CAMERA_STANDOFF_M,
            # Per-candidate provenance is recorded on each ranked row; this is
            # only the fallback used when the layout position is unavailable.
            "camera_position_fallback": "reconstructed_from_declared_standoff",
            "belief_centroid_source": "learned_candidate_center_world_m",
            "current_relation_uncertainty": uncertainty,
            "ranked_views": rows,
            "selected_view": selected,
            "safe_candidate_view_policy": policy,
            "analytic_control": analytic_control,
            "analytic_control_recommended_view": analytic_control.get("selected_view"),
            "analytic_recommended_view": analytic_recommended_view,
            "agreement": (
                selected == analytic_recommended_view
                if selected is not None and analytic_recommended_view is not None
                else None
            ),
            "reason": (
                None if supported else "no_remaining_view_seen_during_ranker_training"
            ),
            "deployment_gate": (
                "shadow_until_realized_ranker_cross_seed_metrics_and_closed_loop_execution_pass"
                if self.view_ranker_metadata.get("target_scope")
                == "cross_fitted_realized_learned_grasp_frame_error_reduction"
                else "shadow_until_multi_seed_realized_learned_graph_error_reduction_passes"
            ),
        }


def _capture_view(env: Any, view: str, initial_pose: Any) -> dict[str, Any]:
    if view == "current":
        with robotwin_cwd():
            env.camera.get_camera().entity.set_pose(deepcopy(initial_pose))
            env.task._update_render()
    else:
        with robotwin_cwd():
            env.camera.view(env.active_arm, view)
            env.task._update_render()
    camera = env.camera.get_camera()
    with robotwin_cwd():
        camera.take_picture()
        rgba = camera.get_picture("Color")
        position = camera.get_picture("Position")
    pose = camera.entity.get_pose().to_transformation_matrix().astype(np.float64)
    return {
        "view": view,
        "rgb": (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8),
        "depth_mm": (-position[..., 2] * 1000.0).astype(np.float32),
        "intrinsic_cv": np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64),
        "extrinsic_cv": np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64),
        "view_direction_world": _unit(pose[:3, 0]).tolist(),
        # Measured camera translation, not a standoff reconstruction. A
        # reconstructed position is collinear with the view ray by
        # construction, so range and position features carry no information
        # beyond the ray until the real extrinsic is recorded here.
        "camera_position_world_m": pose[:3, 3].tolist(),
        "camera_position_source": "measured_camera_extrinsic",
    }


def _with_layout_camera_positions(
    env: Any,
    candidate_views: list[dict[str, Any]],
    *,
    layout: CameraLayout | None = None,
) -> list[dict[str, Any]]:
    """Predict each candidate's real camera position from the layout table.

    The declared view catalogue is not radial about a single point, so these
    positions are independent of the view ray. When the finger center cannot be
    read the rows pass through unchanged and the scorer labels its standoff
    reconstruction degenerate rather than presenting it as a measurement.
    """

    controller = getattr(env, "camera", None)
    finger_center = getattr(controller, "_finger_center", None)
    if finger_center is None:
        return candidate_views
    try:
        center = np.asarray(finger_center(env.active_arm), dtype=np.float64)
    except Exception:  # pragma: no cover - simulator state is not always readable
        return candidate_views
    result = []
    for row in candidate_views:
        merged = dict(row)
        try:
            position, direction = camera_pose_for_view(
                str(row.get("view")),
                finger_center_world=center,
                view_arm=str(env.active_arm),
                camera_bounds=getattr(controller, "camera_bounds", None)
                if getattr(controller, "camera_bounds", None) is not None
                else DEFAULT_CAMERA_BOUNDS,
                layout=layout or _runtime_camera_layout(env, None),
            )
            merged["camera_position_world"] = position.tolist()
            merged["view_direction_world"] = direction.tolist()
            merged["camera_position_source"] = "declared_layout_after_bounds_clamp"
            merged["camera_pose_source"] = (
                layout or _runtime_camera_layout(env, None)
            ).layout_id
            merged["view_direction_source"] = "layout_look_at_finger_center"
            merged["camera_reachable"] = True
            merged["camera_safe"] = True
            merged["camera_safety_source"] = "declared_camera_bounds"
        except ValueError:
            pass
        result.append(merged)
    return result


def _view_selection_reason(
    view: str,
    assessment: dict[str, Any],
    observation_index: int,
) -> str:
    if observation_index == 0:
        return "initial_observation"
    for row in assessment.get("candidate_view_scores", ()):
        if str(row.get("view")) == str(view):
            reason = row.get("reason")
            if isinstance(reason, dict) and reason:
                return str(max(reason, key=lambda key: float(reason[key])))
    return "selected_by_previous_active_view_assessment"


def _runtime_camera_layout(
    env: Any, configured: CameraLayout | None
) -> CameraLayout:
    if configured is not None:
        return configured
    camera_layout = getattr(getattr(env, "camera", None), "camera_layout", None)
    return camera_layout if isinstance(camera_layout, CameraLayout) else DEFAULT_CAMERA_LAYOUT


def _layout_view_direction(
    layout: CameraLayout, view: str, *, view_arm: str
) -> list[float]:
    offset = layout.offset_for_arm(view, view_arm=view_arm)
    norm = float(np.linalg.norm(offset))
    if norm <= 1e-9:
        raise ValueError(f"camera layout view {view!r} has a zero offset")
    return (-offset / norm).tolist()


def _target_actor(env: Any, target: str) -> Any:
    aliases = {
        "pen": ("pen",),
        "cube": ("cube",),
        "bottle": ("bottle",),
    }
    for attribute in aliases.get(target.lower(), (target.lower(),)):
        actor = getattr(env.task, attribute, None)
        if actor is not None:
            return actor
    raise ValueError(f"active semantic tool cannot resolve target actor {target!r}")


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("camera view direction is degenerate")
    return value / norm


def _learned_relation_uncertainty(
    fused_region: dict[str, Any], candidate: Any
) -> dict[str, float]:
    position_covariance = np.asarray(
        fused_region.get("position_covariance_m2", np.eye(3) * 0.0009),
        dtype=np.float64,
    )
    if position_covariance.shape != (3, 3):
        raise ValueError("learned semantic region covariance must be 3x3")
    position_std = float(
        np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(position_covariance)))))
    )
    orientation_covariance = np.asarray(
        candidate.orientation_covariance_rad2, dtype=np.float64
    )
    orientation_std = float(
        np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(orientation_covariance)))))
    )
    position_uncertainty = float(position_std / (position_std + 0.03))
    orientation_scale = float(np.deg2rad(20.0))
    orientation_uncertainty = float(
        orientation_std / (orientation_std + orientation_scale)
    )
    confidence_uncertainty = float(
        np.clip(1.0 - float(fused_region.get("confidence", 0.0)), 0.0, 1.0)
    )
    contact_uncertainty = max(position_uncertainty, confidence_uncertainty)
    axis_uncertainty = max(orientation_uncertainty, confidence_uncertainty)
    result = {
        "grasp_center_location": position_uncertainty,
        "left_contact": contact_uncertainty,
        "right_contact": contact_uncertainty,
        "opening_width": contact_uncertainty,
        "approach_axis": axis_uncertainty,
        "closing_axis": axis_uncertainty,
    }
    if set(result) != set(RELATION_NAMES):
        raise RuntimeError("learned relation uncertainty schema mismatch")
    return {key: round(float(value), 8) for key, value in result.items()}
