"""Active multi-view runtime tool for learned task-conditioned grasp frames."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

from .analytic_view_control import rank_runtime_candidates_with_analytic_control
from .camera_control import (
    DEFAULT_CAMERA_BOUNDS,
    DEFAULT_CAMERA_LAYOUT,
    VIEW_LAYOUT_OFFSETS_M,
    camera_pose_for_view,
    normalize_camera_pose_perturbation,
    perturb_camera_frame,
    perturb_camera_pose,
)
from .camera_layout import CameraLayout
from .env import robotwin_cwd
from .expert_event_view_ranker import (
    PairwiseLinearEventViewRanker,
    event_view_feature_vector,
    score_runtime_candidate_views,
)
from .expert_grasp_frame_inference import (
    build_grasp_frame_graph,
    fuse_expert_grasp_frames,
    predict_expert_grasp_frame,
)
from .expert_grasp_frame_model import ExpertGraspFrameNet
from .grasp_candidate_neural_ranker import (
    IntentEmbeddingStore,
    affordance_profile_feature_vector,
    candidate_base_feature_vector,
    canonical_intent_text,
    load_neural_ranker_checkpoint,
)
from .grasp_candidates import (
    CandidateCheck,
    GraspCandidate,
    GraspIntent,
    assess_candidate_view_sufficiency,
)
from .grasp_candidate_safety_model import (
    FactorizedCandidateOutcomeModel,
    score_inference_candidate_set_shadow,
)
from .open_vocab_grasp_graph import build_open_vocab_grasp_candidate_graph
from .realized_event_view_dataset import relation_uncertainty_from_grasp_frame_graph
from .relation_view_ranker import (
    RELATION_VIEW_RANKER_SCHEMA,
    RelationDecomposedViewRanker,
)
from .safe_candidate_view_ranker import (
    LEGACY_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    PREVIOUS_SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    SAFE_CANDIDATE_VIEW_RANKER_SCHEMA,
    SafeCandidateViewRanker,
)
from .selective_risk_controller import SelectiveRiskController
from .semantic_part_region_dataset import prepare_semantic_part_rgbd
from .sensor_perturbations import (
    apply_sensor_perturbation,
    normalize_sensor_perturbation,
)
from .view_visibility_model import attach_visibility_priors_to_candidates
from .view_value_of_information import rank_views_by_value_of_information


DIRECT_FRAME_VIEW_DIRECTIONS = {
    view: (
        -np.asarray(offset, dtype=np.float64)
        / np.linalg.norm(np.asarray(offset, dtype=np.float64))
    ).tolist()
    for view, offset in VIEW_LAYOUT_OFFSETS_M.items()
}
# Declared layout constant for the current discrete view catalogue, not a
# measurement. It converts a view direction into a camera position so that
# range-dependent features are defined. Recorded next to every ranking.
DIRECT_FRAME_CAMERA_STANDOFF_M = 0.6


@dataclass(frozen=True)
class VisualGroundingHint:
    """A coarse VLM observation anchor expressed in the current RGB image."""

    target_box_normalized_xyxy: tuple[float, float, float, float] | None = None
    grasp_point_normalized_uv: tuple[float, float] | None = None
    confidence: float = 1.0
    source: str = "vlm"
    target_description: str | None = None
    view: str = "current"

    def __post_init__(self) -> None:
        if self.view != "current":
            raise ValueError(
                "visual grounding currently supports only the current view"
            )
        if (
            self.target_box_normalized_xyxy is None
            and self.grasp_point_normalized_uv is None
        ):
            raise ValueError("visual grounding requires a target box or grasp point")
        if self.target_box_normalized_xyxy is not None:
            box = tuple(float(value) for value in self.target_box_normalized_xyxy)
            if len(box) != 4 or not all(0.0 <= value <= 1.0 for value in box):
                raise ValueError("target box must contain four normalized coordinates")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("target box must have positive width and height")
            object.__setattr__(self, "target_box_normalized_xyxy", box)
        if self.grasp_point_normalized_uv is not None:
            point = tuple(float(value) for value in self.grasp_point_normalized_uv)
            if len(point) != 2 or not all(0.0 <= value <= 1.0 for value in point):
                raise ValueError("grasp point must contain two normalized coordinates")
            object.__setattr__(self, "grasp_point_normalized_uv", point)
            if self.target_box_normalized_xyxy is not None:
                box = self.target_box_normalized_xyxy
                if not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
                    raise ValueError("grasp point must lie inside the target box")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("visual grounding confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "source", str(self.source).strip() or "vlm")
        if self.target_description is not None:
            description = str(self.target_description).strip()
            object.__setattr__(self, "target_description", description or None)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "VisualGroundingHint | None":
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("visual_grounding must be an object")
        return cls(
            target_box_normalized_xyxy=(
                tuple(value["target_box_normalized_xyxy"])
                if value.get("target_box_normalized_xyxy") is not None
                else None
            ),
            grasp_point_normalized_uv=(
                tuple(value["grasp_point_normalized_uv"])
                if value.get("grasp_point_normalized_uv") is not None
                else None
            ),
            confidence=float(value.get("confidence", 1.0)),
            source=str(value.get("source", "vlm")),
            target_description=(
                str(value["target_description"])
                if value.get("target_description") is not None
                else None
            ),
            view=str(value.get("view", "current")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "coordinate_space": "normalized_0_1_top_left_origin",
            "target_box_normalized_xyxy": (
                list(self.target_box_normalized_xyxy)
                if self.target_box_normalized_xyxy is not None
                else None
            ),
            "grasp_point_normalized_uv": (
                list(self.grasp_point_normalized_uv)
                if self.grasp_point_normalized_uv is not None
                else None
            ),
            "confidence": round(self.confidence, 6),
            "source": self.source,
            "target_description": self.target_description,
        }


class ActiveExpertGraspFrameTool:
    def __init__(
        self,
        checkpoint: str | Path,
        intent_embeddings: str | Path,
        *,
        view_ranker_checkpoint: str | Path | None = None,
        selective_risk_controller_checkpoint: str | Path | None = None,
        selective_risk_controller_domain: str | None = None,
        candidate_safety_checkpoint: str | Path | None = None,
        candidate_safety_threshold: float = 0.9,
        candidate_ranker_checkpoint: str | Path | None = None,
        view_selection_mode: str = "analytic",
        camera_pose_perturbation: Mapping[str, Any] | None = None,
        sensor_perturbation: Mapping[str, Any] | None = None,
        camera_layout: CameraLayout | Mapping[str, Any] | None = None,
        dynamic_encoder_python: str | Path | None = None,
        dynamic_encoder_script: str | Path | None = None,
        dynamic_encoder_snapshot: str | Path | None = None,
        device: str | torch.device | None = None,
        image_size: tuple[int, int] = (240, 320),
    ) -> None:
        if view_selection_mode not in {
            "analytic",
            "shadow",
            "learned_evaluation",
            "forced_evaluation",
        }:
            raise ValueError(
                "direct grasp-frame view mode must be analytic, shadow, "
                "learned_evaluation, or forced_evaluation"
            )
        self.checkpoint_path = Path(checkpoint).resolve()
        self.embedding_path = Path(intent_embeddings).resolve()
        self.embeddings = IntentEmbeddingStore.load(self.embedding_path)
        self.view_selection_mode = view_selection_mode
        if isinstance(camera_layout, Mapping):
            camera_layout = CameraLayout.from_mapping(camera_layout)
        if camera_layout is not None and not isinstance(camera_layout, CameraLayout):
            raise TypeError("camera_layout must be a CameraLayout or mapping")
        self.camera_layout = camera_layout
        self.camera_pose_perturbation = normalize_camera_pose_perturbation(
            camera_pose_perturbation
        )
        self.sensor_perturbation = normalize_sensor_perturbation(sensor_perturbation)
        if not 0.0 < candidate_safety_threshold < 1.0:
            raise ValueError("candidate safety threshold must be in (0, 1)")
        self.candidate_safety_threshold = float(candidate_safety_threshold)
        self.candidate_safety_path = (
            Path(candidate_safety_checkpoint).resolve()
            if candidate_safety_checkpoint is not None
            else None
        )
        self.candidate_safety_model = None
        if self.candidate_safety_path is not None:
            if not self.candidate_safety_path.is_file():
                raise FileNotFoundError(self.candidate_safety_path)
            safety_payload = json.loads(
                self.candidate_safety_path.read_text(encoding="utf-8")
            )
            self.candidate_safety_model = FactorizedCandidateOutcomeModel.from_dict(
                safety_payload
            )
        self.candidate_ranker_path = (
            Path(candidate_ranker_checkpoint).resolve()
            if candidate_ranker_checkpoint is not None
            else None
        )
        self.candidate_ranker = None
        self.candidate_ranker_payload: dict[str, Any] = {}
        if self.candidate_ranker_path is not None:
            if not self.candidate_ranker_path.is_file():
                raise FileNotFoundError(self.candidate_ranker_path)
            (
                self.candidate_ranker,
                self.candidate_ranker_payload,
            ) = load_neural_ranker_checkpoint(self.candidate_ranker_path, device="cpu")
            if (
                self.candidate_ranker_payload.get("encoder_id")
                != self.embeddings.encoder_id
            ):
                raise ValueError("candidate ranker and embedding encoder_id differ")
        self.dynamic_encoder_python = (
            Path(dynamic_encoder_python).resolve()
            if dynamic_encoder_python is not None
            else None
        )
        self.dynamic_encoder_script = (
            Path(dynamic_encoder_script).resolve()
            if dynamic_encoder_script is not None
            else None
        )
        self.dynamic_encoder_snapshot = (
            Path(dynamic_encoder_snapshot).resolve()
            if dynamic_encoder_snapshot is not None
            else None
        )
        dynamic_values = (
            self.dynamic_encoder_python,
            self.dynamic_encoder_script,
            self.dynamic_encoder_snapshot,
        )
        if any(value is not None for value in dynamic_values) and not all(
            value is not None for value in dynamic_values
        ):
            raise ValueError(
                "dynamic intent encoding requires python, script, and snapshot"
            )
        for path in dynamic_values:
            if path is not None and not path.exists():
                raise FileNotFoundError(path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = tuple(int(value) for value in image_size)
        payload = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload.get("schema_version") != "phase11.expert_grasp_frame_checkpoint.v1":
            raise ValueError("unsupported expert grasp-frame checkpoint")
        if payload.get("encoder_id") != self.embeddings.encoder_id:
            raise ValueError("grasp-frame checkpoint and embedding encoder_id differ")
        config = payload["model_config"]
        self.model = ExpertGraspFrameNet(
            text_embedding_dim=int(config["text_embedding_dim"]),
            feature_dim=int(config["feature_dim"]),
            freeze_stem=bool(config["freeze_stem"]),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()
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
        if (
            view_selection_mode in {"shadow", "learned_evaluation"}
            and self.view_ranker_path is None
        ):
            raise ValueError(
                "direct grasp-frame shadow/learned-evaluation mode requires a view ranker"
            )
        self.view_ranker = None
        self.view_ranker_metadata: dict[str, Any] = {}
        self.view_ranker_supported_views: set[str] = set()
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
            elif ranker_payload.get("schema_version") == RELATION_VIEW_RANKER_SCHEMA:
                self.view_ranker = RelationDecomposedViewRanker.from_dict(
                    ranker_payload
                )
            else:
                self.view_ranker = PairwiseLinearEventViewRanker.from_dict(
                    ranker_payload
                )
            self.view_ranker_metadata = dict(
                ranker_payload.get("training_metadata", {})
            )
            if not self.view_ranker.feature_config.supports_unseen_view_labels:
                self.view_ranker_supported_views = set(
                    self.view_ranker_metadata.get("trained_views", ())
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

    def propose(
        self,
        env: Any,
        intent: GraspIntent,
        *,
        output_dir: str | Path,
        world_state_version: int,
        max_additional_views: int = 2,
        visual_grounding: Mapping[str, Any] | VisualGroundingHint | None = None,
        camera_pose_perturbation: Mapping[str, Any] | None = None,
        sensor_perturbation: Mapping[str, Any] | None = None,
        forced_additional_views: Sequence[str] | None = None,
        calibration_task: str | None = None,
    ) -> dict[str, Any]:
        if max_additional_views < 0:
            raise ValueError("max_additional_views must be non-negative")
        layout = _runtime_camera_layout(env, self.camera_layout)
        forced_views = _normalize_forced_additional_views(
            forced_additional_views,
            view_selection_mode=self.view_selection_mode,
            max_additional_views=max_additional_views,
            available_views=layout.view_names,
        )
        output_dir = Path(output_dir).resolve()
        runtime_perturbation = normalize_camera_pose_perturbation(
            camera_pose_perturbation
            if camera_pose_perturbation is not None
            else self.camera_pose_perturbation
        )
        runtime_sensor_perturbation = normalize_sensor_perturbation(
            sensor_perturbation
            if sensor_perturbation is not None
            else self.sensor_perturbation
        )
        policy_task = (
            str(calibration_task).strip()
            if calibration_task is not None and str(calibration_task).strip()
            else intent.task_goal
        )
        views_dir = output_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        embedding, embedding_source = self._embedding_for_intent(
            intent, cache_dir=output_dir / "intent_embedding_cache"
        )
        embedding_tensor = torch.from_numpy(embedding)
        grounding_hint = (
            visual_grounding
            if isinstance(visual_grounding, VisualGroundingHint)
            else VisualGroundingHint.from_mapping(visual_grounding)
        )
        resolved_grounding: dict[str, Any] | None = None
        initial_camera_pose = deepcopy(env.camera.get_camera().entity.get_pose())
        world_before = _world_fingerprint(env)
        frames = []
        observed_directions = []
        visited = []
        history = []
        next_view = "current"
        final_frame: dict[str, Any] | None = None
        final_execution_frame: dict[str, Any] | None = None
        final_candidate: GraspCandidate | None = None
        final_candidates: list[GraspCandidate] = []
        final_assessment: dict[str, Any] = {}
        camera_controller = getattr(env, "camera", None)
        previous_camera_layout = getattr(camera_controller, "camera_layout", None)
        if self.camera_layout is not None and camera_controller is not None:
            camera_controller.camera_layout = self.camera_layout
        try:
            for observation_index in range(max_additional_views + 1):
                capture = _capture_view(
                    env,
                    next_view,
                    initial_camera_pose,
                    pose_perturbation=runtime_perturbation,
                )
                capture = apply_sensor_perturbation(
                    capture,
                    runtime_sensor_perturbation,
                    frame_id=observation_index,
                )
                selection_reason = _view_selection_reason(
                    next_view, final_assessment, observation_index
                )
                if observation_index == 0 and grounding_hint is not None:
                    resolved_grounding = resolve_visual_grounding_hint(
                        capture, grounding_hint
                    )
                visited.append(next_view)
                observed_directions.append(capture["view_direction_world"])
                image_path = views_dir / f"{observation_index:02d}_{next_view}.png"
                depth_path = (
                    views_dir / f"{observation_index:02d}_{next_view}_depth_m.npy"
                )
                camera_path = (
                    views_dir / f"{observation_index:02d}_{next_view}_camera.json"
                )
                imageio.imwrite(image_path, capture["rgb"])
                np.save(
                    depth_path,
                    np.asarray(capture["depth_mm"], dtype=np.float32) / 1000.0,
                    allow_pickle=False,
                )
                camera_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "spatial.inference_camera_calibration.v1",
                            "view": next_view,
                            "frame_id": observation_index,
                            "intrinsic_cv": np.asarray(
                                capture["intrinsic_cv"], dtype=np.float64
                            ).tolist(),
                            "extrinsic_cv": np.asarray(
                                capture["extrinsic_cv"], dtype=np.float64
                            ).tolist(),
                            "view_direction_world": capture["view_direction_world"],
                            "camera_position_world_m": capture[
                                "camera_position_world_m"
                            ],
                            "camera_position_source": capture["camera_position_source"],
                            "sensor_perturbation": runtime_sensor_perturbation,
                            "sensor_perturbation_stats": capture.get(
                                "sensor_perturbation_stats"
                            ),
                            "access": "inference_visible",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rgbd = prepare_semantic_part_rgbd(
                    capture["rgb"],
                    capture["depth_mm"] / 1000.0,
                    self.image_size,
                )
                observed_frame = predict_expert_grasp_frame(
                    self.model,
                    rgbd,
                    embedding_tensor,
                    {
                        "intrinsic_cv": capture["intrinsic_cv"],
                        "extrinsic_cv": capture["extrinsic_cv"],
                    },
                    original_image_size=capture["rgb"].shape[:2],
                    frame_id="grasp_candidate_000",
                    target=f"object.{intent.target}",
                    semantic_role=intent.preferred_roles[0],
                    view=next_view,
                    device=self.device,
                )
                if resolved_grounding is not None:
                    observed_frame = align_frame_to_visual_grounding(
                        observed_frame, resolved_grounding
                    )
                for evidence in observed_frame["evidence"]:
                    evidence["frame_id"] = observation_index
                frames.append(observed_frame)
                final_frame = fuse_expert_grasp_frames(frames)
                if resolved_grounding is not None:
                    final_frame[
                        "visual_grounding_alignment"
                    ] = summarize_visual_grounding_alignments(
                        frames, resolved_grounding
                    )
                    final_frame[
                        "source"
                    ] = f"{final_frame['source']}+vlm_visual_grounding"
                final_execution_frame = refine_frame_support_clearance(
                    final_frame, capture
                )
                final_candidate = candidate_from_learned_frame(
                    intent, final_execution_frame
                )
                final_candidates = generate_local_candidate_hypotheses(final_candidate)
                remaining = _with_layout_camera_positions(
                    env,
                    [
                        {
                            "view": view,
                            "view_direction_world": _layout_view_direction(
                                layout, view, view_arm=str(env.active_arm)
                            ),
                        }
                        for view in layout.view_names
                        if view not in visited
                    ],
                    layout=layout,
                    pose_perturbation=runtime_perturbation,
                )
                for candidate_view in remaining:
                    candidate_view["move_cost"] = _view_move_cost(
                        observed_directions[-1],
                        candidate_view["view_direction_world"],
                    )
                final_assessment = assess_candidate_view_sufficiency(
                    final_candidates,
                    observed_view_directions_world=observed_directions,
                    candidate_views=remaining,
                    candidate_selection_scores={
                        candidate.candidate_id: candidate.score
                        for candidate in final_candidates
                    },
                )
                consistency = next(
                    check
                    for check in final_candidate.checks
                    if check.name == "contact_width_consistency"
                )
                if consistency.probability is None or consistency.probability < 0.5:
                    final_assessment["sufficient"] = False
                    final_assessment["visual_evidence_sufficient"] = False
                    final_assessment["execution_ready"] = False
                    final_assessment["reasons"] = list(
                        dict.fromkeys(
                            [
                                *final_assessment.get("reasons", ()),
                                "frame_contact_width_inconsistent",
                            ]
                        )
                    )
                    if final_assessment.get(
                        "recommended_view"
                    ) is None and final_assessment.get("candidate_view_scores"):
                        final_assessment["recommended_view"] = final_assessment[
                            "candidate_view_scores"
                        ][0]["view"]
                shadow = self._shadow_ranking(
                    intent,
                    final_frame=final_execution_frame,
                    observed_directions=observed_directions,
                    candidate_views=remaining,
                    analytic_recommended_view=final_assessment.get("recommended_view"),
                )
                if shadow is not None:
                    final_assessment["learned_view_ranker_shadow"] = shadow
                    if self.view_selection_mode == "learned_evaluation":
                        final_assessment[
                            "analytic_recommended_view"
                        ] = final_assessment.get("recommended_view")
                        final_assessment["recommended_view"] = shadow.get(
                            "selected_view"
                        )
                        final_assessment[
                            "view_control_policy"
                        ] = "learned_relation_ranker_evaluation_only"
                if self.view_selection_mode == "forced_evaluation":
                    final_assessment[
                        "analytic_recommended_view"
                    ] = final_assessment.get("recommended_view")
                    forced_next_view = (
                        forced_views[observation_index]
                        if observation_index < len(forced_views)
                        else None
                    )
                    final_assessment["recommended_view"] = forced_next_view
                    final_assessment["forced_selected_view"] = forced_next_view
                    final_assessment["forced_additional_views"] = list(forced_views)
                    final_assessment[
                        "view_control_policy"
                    ] = "forced_sequence_counterfactual_evaluation_only"
                history.append(
                    {
                        "frame_id": observation_index,
                        "view": next_view,
                        "image": str(image_path.relative_to(output_dir)),
                        "depth_m": str(depth_path.relative_to(output_dir)),
                        "camera": str(camera_path.relative_to(output_dir)),
                        "rgbd_access": "inference_visible_server_side",
                        "camera_position_world_m": capture["camera_position_world_m"],
                        "view_direction_world": capture["view_direction_world"],
                        "camera_pose_source": capture["camera_position_source"],
                        "selection_reason": selection_reason,
                        "frame_observation": observed_frame,
                        "fused_frame": final_frame,
                        "execution_frame": final_execution_frame,
                        "view_assessment": deepcopy(final_assessment),
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
        if (
            final_frame is None
            or final_execution_frame is None
            or final_candidate is None
            or not final_candidates
        ):
            raise RuntimeError("active grasp-frame tool produced no observation")
        world_after = _world_fingerprint(env)
        world_delta = _fingerprint_delta(world_before, world_after)
        if final_assessment.get("visual_evidence_sufficient"):
            final_assessment["stop_reason"] = "visual_evidence_sufficient"
        elif len(visited) >= max_additional_views + 1:
            final_assessment["stop_reason"] = "view_budget_exhausted"
            if final_assessment.get("recommended_view") is not None:
                final_assessment[
                    "recommended_view_if_budget_extended"
                ] = final_assessment["recommended_view"]
                final_assessment["recommended_view"] = None
        else:
            final_assessment["stop_reason"] = "no_remaining_candidate_view"
        candidate_ranking = {
            "source": "direct_frame_local_uncertainty_hypotheses",
            "calibrated_probability": False,
            "scores": {
                candidate.candidate_id: round(candidate.score, 6)
                for candidate in final_candidates
            },
            "candidate_pool_count": len(final_candidates),
        }
        candidate_ranker_shadow = self._score_candidate_ranker_shadow(
            intent,
            embedding,
            final_candidates,
            analytic_candidate_id=final_candidate.candidate_id,
        )
        if candidate_ranker_shadow is not None:
            candidate_ranking["learned_shadow"] = {
                "source": candidate_ranker_shadow["source"],
                "selected_candidate_id": candidate_ranker_shadow[
                    "selected_candidate_id"
                ],
                "analytic_selected_candidate_id": final_candidate.candidate_id,
                "agreement": candidate_ranker_shadow["agreement"],
            }
        graph = build_open_vocab_grasp_candidate_graph(
            intent,
            final_candidates,
            world_state_version=world_state_version,
            view_assessment=final_assessment,
            candidate_ranking=candidate_ranking,
            view_evidence=history,
            visual_grounding=resolved_grounding,
            fusion_diagnostics=final_frame.get("fusion_diagnostics"),
        )
        candidate_safety_shadow = None
        if self.candidate_safety_model is not None:
            candidate_safety_shadow = score_inference_candidate_set_shadow(
                self.candidate_safety_model,
                intent=intent.as_dict(),
                candidates=[candidate.as_dict() for candidate in final_candidates],
                candidate_graph=graph,
                view_assessment=final_assessment,
                authorization_threshold=self.candidate_safety_threshold,
            )
        visual_sufficient = bool(
            final_assessment.get("visual_evidence_sufficient", False)
        )
        execution_ready = bool(final_assessment.get("execution_ready", False))
        evaluation_only = self.view_selection_mode in {
            "learned_evaluation",
            "forced_evaluation",
        }
        if evaluation_only:
            execution_ready = False
            final_assessment["execution_ready"] = False
            final_assessment["view_control_evaluation_only"] = True
            final_assessment["view_control_evaluation_mode"] = self.view_selection_mode
        result = {
            "schema_version": "spatial.active_expert_grasp_frame_candidates.v2",
            "query": "propose_grasp_candidates",
            "verdict": "uncertain" if final_candidate else "no_candidate",
            "confidence": round(
                float(final_candidate.score * final_candidate.score_confidence), 6
            ),
            "intent": intent.as_dict(),
            "calibration_task": policy_task,
            "top_candidates": [candidate.as_dict() for candidate in final_candidates],
            "recommended_candidate_id": final_candidate.candidate_id,
            "candidate_ranking": candidate_ranking,
            "candidate_graph": graph,
            "camera_pose_perturbation": runtime_perturbation,
            "sensor_perturbation": runtime_sensor_perturbation,
            "camera_layout": layout.as_dict(),
            "vlm_candidate_summary": graph["vlm_summary"],
            "view_assessment": final_assessment,
            "multiview_fusion": graph.get("multiview_fusion"),
            "candidate_safety_shadow": candidate_safety_shadow,
            "candidate_ranker_shadow": candidate_ranker_shadow,
            "visual_evidence_sufficient": visual_sufficient,
            "execution_ready": execution_ready,
            "recommended_action": (
                {
                    "tool": "stop",
                    "reason": (
                        "learned_view_control_evaluation_only"
                        if self.view_selection_mode == "learned_evaluation"
                        else "forced_view_control_evaluation_only"
                    ),
                }
                if evaluation_only
                else {
                    "tool": "spatial.verify_candidate_executability",
                    "reason": "direct_frame_candidate_requires_nonexecuting_planner_authorization",
                    "candidate_id": final_candidate.candidate_id,
                    "missing_checks": final_candidate.unknown_constraints,
                }
                if visual_sufficient
                else {
                    "tool": "stop",
                    "reason": final_assessment["stop_reason"],
                    "unresolved_visual_reasons": final_assessment.get("reasons", []),
                }
            ),
            "observed_view_sequence": visited,
            "evidence_views": visited,
            "evidence_frames": list(range(len(visited))),
            "history": history,
            "world_frozen": world_delta < 1e-6,
            "world_fingerprint_delta": world_delta,
            "geometry_provenance": {
                "grasp_frame": "learned_rgbd_frozen_intent_direct_frame",
                "support_clearance_refinement": final_execution_frame.get(
                    "support_clearance_refinement"
                ),
                "intent_embedding": embedding_source,
                "visual_grounding": resolved_grounding,
                "oracle_object_geometry_used": False,
                "pointcloud_used": bool(
                    final_execution_frame.get("support_clearance_refinement", {}).get(
                        "applied", False
                    )
                ),
                "ik": "not_evaluated",
                "collision": "not_evaluated",
                "execution_success": "not_evaluated",
                "view_selection": (
                    "legacy_relation_sufficiency_with_ranker_and_analytic_control_shadow"
                    if self.view_selection_mode == "shadow"
                    else (
                        "learned_relation_ranker_evaluation_only"
                        if self.view_selection_mode == "learned_evaluation"
                        else (
                            "forced_sequence_counterfactual_evaluation_only"
                            if self.view_selection_mode == "forced_evaluation"
                            else "legacy_relation_sufficiency"
                        )
                    )
                ),
                "camera_layout_id": layout.layout_id,
            },
            "server_side_auxiliary": {
                "intent_embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                "intent_embedding_source": embedding_source,
                "access": "inference_visible_server_side",
            },
            "deployment_gate": (
                "blocked_until_direct_frame_closed_loop_execution_and_generalization_pass"
            ),
        }
        (output_dir / "query_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def _score_candidate_ranker_shadow(
        self,
        intent: GraspIntent,
        embedding: np.ndarray,
        candidates: Sequence[GraspCandidate],
        *,
        analytic_candidate_id: str,
    ) -> dict[str, Any] | None:
        """Score candidates with an optional learned ranker without changing control."""

        if self.candidate_ranker is None:
            return None
        intent_mapping = intent.as_dict()
        candidate_rows = [candidate.as_dict() for candidate in candidates]
        feature_rows = np.stack(
            [
                candidate_base_feature_vector(intent_mapping, candidate)
                for candidate in candidate_rows
            ]
        ).astype(np.float32)
        affordance = affordance_profile_feature_vector(
            intent_mapping.get("affordance_profile")
        ).astype(np.float32)
        with torch.no_grad():
            scores = (
                self.candidate_ranker(
                    torch.from_numpy(feature_rows),
                    torch.from_numpy(np.asarray(embedding, dtype=np.float32)),
                    torch.from_numpy(affordance),
                )
                .cpu()
                .numpy()
            )
        order = np.argsort(scores)[::-1]
        selected = candidates[int(order[0])]
        return {
            "schema_version": "spatial.neural_candidate_ranker_shadow.v1",
            "source": "frozen_text_two_tower_hybrid_ranker",
            "checkpoint": str(self.candidate_ranker_path),
            "encoder_id": self.candidate_ranker_payload.get("encoder_id"),
            "selected_candidate_id": selected.candidate_id,
            "agreement": bool(selected.candidate_id == analytic_candidate_id),
            "candidate_count": len(candidates),
            "ranked_candidates": [
                {
                    "candidate_id": candidates[int(index)].candidate_id,
                    "score": round(float(scores[int(index)]), 8),
                    "rank": rank,
                }
                for rank, index in enumerate(order, start=1)
            ],
            "execution_authorization": "disabled_shadow_only",
        }

    def _embedding_for_intent(
        self, intent: GraspIntent, *, cache_dir: Path
    ) -> tuple[np.ndarray, str]:
        try:
            return (
                self.embeddings.for_intent(intent.as_dict()),
                "precomputed_exact_text",
            )
        except ValueError:
            if self.dynamic_encoder_python is None:
                raise ValueError(
                    "no exact frozen intent embedding exists and dynamic local encoding is disabled"
                )
        text = canonical_intent_text(intent.as_dict())
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"{digest}.embedding.json"
        if not output_path.is_file():
            request_path = cache_dir / f"{digest}.request.json"
            request = {
                "schema_version": "spatial.intent_embedding_requests.v1",
                "request_count": 1,
                "unique_text_count": 1,
                "unique_texts": [{"text_sha256": digest, "text": text}],
                "requests": [
                    {
                        "sample_id": f"runtime/{digest}",
                        "task": intent.task_goal,
                        "text": text,
                        "text_sha256": digest,
                    }
                ],
            }
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            completed = subprocess.run(
                [
                    str(self.dynamic_encoder_python),
                    str(self.dynamic_encoder_script),
                    "--requests",
                    str(request_path),
                    "--output",
                    str(output_path),
                    "--snapshot",
                    str(self.dynamic_encoder_snapshot),
                    "--batch-size",
                    "1",
                    "--device",
                    str(self.device),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "dynamic frozen intent encoding failed: "
                    + (completed.stderr or completed.stdout)[-2000:]
                )
        store = IntentEmbeddingStore.load(output_path)
        if store.encoder_id != self.embeddings.encoder_id:
            raise ValueError(
                "dynamic intent encoder differs from grasp-frame checkpoint"
            )
        return store.for_intent(intent.as_dict()), "dynamic_local_frozen_text_encoder"

    def _shadow_ranking(
        self,
        intent: GraspIntent,
        *,
        final_frame: Mapping[str, Any],
        observed_directions: list[list[float]],
        candidate_views: list[dict[str, Any]],
        analytic_recommended_view: str | None,
    ) -> dict[str, Any] | None:
        if (
            self.view_selection_mode not in {"shadow", "learned_evaluation"}
            or self.view_ranker is None
        ):
            return None
        graph = build_grasp_frame_graph(final_frame)
        uncertainty = relation_uncertainty_from_grasp_frame_graph(graph)
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
                contact_separation_m = float(
                    final_frame.get(
                        "contact_separation_m",
                        np.linalg.norm(
                            np.asarray(final_frame["right_contact_world_m"], dtype=np.float64)
                            - np.asarray(final_frame["left_contact_world_m"], dtype=np.float64)
                        ),
                    )
                )
                analytic_candidate_views = attach_visibility_priors_to_candidates(
                    candidate_views,
                    target_center_world=final_frame["center_world_m"],
                    closing_axis_world=final_frame["closing_axis_world"],
                    approach_axis_world=final_frame["approach_axis_world"],
                    focal_length_px=calibration.focal_length_px,
                    contact_separation_m=contact_separation_m,
                    position_covariance_m2=final_frame["point_covariance_m2"][
                        "grasp_center"
                    ],
                )
                analytic_control = rank_runtime_candidates_with_analytic_control(
                    prior_covariance_m2=final_frame["point_covariance_m2"][
                        "grasp_center"
                    ],
                    closing_axis_world=final_frame["closing_axis_world"],
                    approach_axis_world=final_frame["approach_axis_world"],
                    relation_uncertainty=uncertainty,
                    sensor_model=calibration.sensor_model,
                    focal_length_px=calibration.focal_length_px,
                    default_range_m=calibration.camera_standoff_m,
                    belief_centroid_world=final_frame["center_world_m"],
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
        supported = [
            row
            for row in candidate_views
            if not self.view_ranker_supported_views
            or row["view"] in self.view_ranker_supported_views
        ]
        if supported and isinstance(self.view_ranker, RelationDecomposedViewRanker):
            rows = self.view_ranker.score_runtime(
                intent.as_dict(),
                current_relation_uncertainty=uncertainty,
                closing_axis_world=final_frame["closing_axis_world"],
                approach_axis_world=final_frame["approach_axis_world"],
                observed_view_directions_world=observed_directions,
                candidate_views=supported,
                belief_centroid_world=final_frame["center_world_m"],
                prior_covariance_m2=final_frame["point_covariance_m2"]["grasp_center"],
                camera_standoff_m=DIRECT_FRAME_CAMERA_STANDOFF_M,
            )
        elif supported:
            rows = score_runtime_candidate_views(
                self.view_ranker,
                intent.as_dict(),
                current_relation_uncertainty=uncertainty,
                closing_axis_world=final_frame["closing_axis_world"],
                approach_axis_world=final_frame["approach_axis_world"],
                observed_view_directions_world=observed_directions,
                candidate_views=supported,
                belief_centroid_world=final_frame["center_world_m"],
                prior_covariance_m2=final_frame["point_covariance_m2"]["grasp_center"],
                camera_standoff_m=DIRECT_FRAME_CAMERA_STANDOFF_M,
            )
        else:
            rows = []
        safe_policy = isinstance(self.view_ranker, SafeCandidateViewRanker)
        ranked_view = rows[0]["view"] if rows else None
        if safe_policy:
            policy_task = str(getattr(intent, "task_goal", "")).strip() or None
            if rows:
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
                safe_probability = policy["predicted_safe_candidate_probability"]
                threshold = policy["acquisition_threshold"]
                selected = policy["view"]
                policy_action = policy["action"]
                policy_reason = policy["reason"]
                selected_ood_score = policy.get("ood_score")
                selected_ood_threshold = policy.get("ood_threshold")
            else:
                safe_probability = None
                threshold = self.view_ranker.acquisition_threshold
                selected = None
                policy_action = "stop_for_review"
                policy_reason = "no_candidate_views"
                selected_ood_score = None
                selected_ood_threshold = self.view_ranker.ood_threshold
        else:
            selected = ranked_view
            safe_probability = None
            threshold = None
            policy_action = "acquire_view" if selected is not None else "stop"
            policy_reason = "relative_view_ranker"
        return {
            "mode": self.view_selection_mode,
            "control_policy": (
                "learned_safe_candidate_view_ranker_with_selective_risk_veto_evaluation_only"
                if self.view_selection_mode == "learned_evaluation"
                and safe_policy
                and self.selective_risk_controller is not None
                else "learned_safe_candidate_view_ranker_evaluation_only"
                if self.view_selection_mode == "learned_evaluation" and safe_policy
                else "learned_relation_ranker_evaluation_only"
                if self.view_selection_mode == "learned_evaluation"
                else "legacy_analytic_sufficiency"
            ),
            "checkpoint": str(self.view_ranker_path),
            "target_scope": self.view_ranker_metadata.get("target_scope"),
            "ranker_model_type": type(self.view_ranker).__name__,
            "relation_decomposed_explanation": isinstance(
                self.view_ranker, RelationDecomposedViewRanker
            ),
            "runtime_feature_adapter": "direct_grasp_frame_pose_conditioned_v2",
            "feature_mode": self.view_ranker.feature_config.mode,
            "camera_standoff_m": DIRECT_FRAME_CAMERA_STANDOFF_M,
            "camera_position_fallback": "reconstructed_from_declared_standoff",
            "camera_position_source": (
                "per_candidate_layout_pose"
                if any(
                    row.get("camera_position_world") is not None for row in supported
                )
                else "reconstructed_from_declared_standoff"
            ),
            "belief_centroid_source": "learned_grasp_frame_center_world_m",
            "current_relation_uncertainty": uncertainty,
            "ranked_views": rows,
            "selected_view": selected,
            "ranked_view_before_abstention": ranked_view,
            "safe_candidate_view_policy": (
                {
                    "action": policy_action,
                    "reason": policy_reason,
                    "predicted_safe_candidate_probability": safe_probability,
                    "acquisition_threshold": threshold,
                    "ood_score": selected_ood_score,
                    "ood_threshold": selected_ood_threshold,
                    "selective_risk_controller": (
                        {
                            "checkpoint": str(self.selective_risk_controller_path),
                            "domain": self.selective_risk_controller_domain,
                            "enabled": bool(
                                self.selective_risk_controller is not None
                                and self.selective_risk_controller.enabled
                            ),
                            "risk_threshold": (
                                self.selective_risk_controller.risk_threshold
                                if self.selective_risk_controller is not None
                                else None
                            ),
                            "evidence": policy.get("selective_risk"),
                        }
                        if self.selective_risk_controller is not None
                        else None
                    ),
                    "execution_authorization": "disabled_shadow_only",
                }
                if safe_policy
                else None
            ),
            "analytic_control": analytic_control,
            "analytic_control_recommended_view": analytic_control.get("selected_view"),
            "analytic_recommended_view": analytic_recommended_view,
            "agreement": (
                selected == analytic_recommended_view
                if selected is not None and analytic_recommended_view is not None
                else None
            ),
            "deployment_gate": (
                "shadow_until_realized_ranker_cross_seed_metrics_and_closed_loop_execution_pass"
            ),
        }


def candidate_from_learned_frame(
    intent: GraspIntent,
    frame: Mapping[str, Any],
    *,
    pregrasp_distance_m: float = 0.09,
) -> GraspCandidate:
    center = np.asarray(frame["center_world_m"], dtype=np.float64)
    left = np.asarray(frame["left_contact_world_m"], dtype=np.float64)
    right = np.asarray(frame["right_contact_world_m"], dtype=np.float64)
    approach = _unit(frame["approach_axis_world"])
    closing = _unit(frame["closing_axis_world"])
    opening = float(frame["opening_width_m"])
    separation = float(np.linalg.norm(right - left))
    consistency = math.exp(-abs(separation - opening) / 0.015)
    width_feasible = float(0.01 <= opening <= 0.09)
    confidence = float(frame["confidence"])
    grounding = frame.get("visual_grounding_alignment")
    grounding_checks = (
        (
            CandidateCheck(
                "vlm_visual_grounding",
                float(grounding["confidence"]),
                True,
                str(grounding["source"]),
                {
                    "kind": grounding["kind"],
                    "translation_m": grounding.get(
                        "translation_m", grounding.get("mean_translation_m")
                    ),
                    "target_description": grounding.get("target_description"),
                },
            ),
        )
        if grounding is not None
        else ()
    )
    checks = (
        CandidateCheck(
            "semantic_match",
            float(intent.confidence),
            True,
            intent.source,
            {"role": frame["semantic_role"]},
        ),
        CandidateCheck(
            "visual_frame_confidence",
            confidence,
            True,
            str(frame["source"]),
            {"view_count": len({row.get("view") for row in frame.get("evidence", ())})},
        ),
        CandidateCheck(
            "contact_width_consistency",
            consistency,
            False,
            "learned_contact_pair_vs_width_head",
            {
                "contact_separation_m": separation,
                "opening_width_m": opening,
            },
        ),
        CandidateCheck(
            "opening_width_feasible",
            width_feasible,
            True,
            "gripper_limits",
            {"allowed_range_m": [0.01, 0.09]},
        ),
        CandidateCheck("reachable", None, True, "not_evaluated"),
        CandidateCheck("collision_free", None, True, "not_evaluated"),
        CandidateCheck("predicted_execution_success", None, False, "not_evaluated"),
    ) + grounding_checks
    known = [check.probability for check in checks if check.probability is not None]
    score = float(np.mean(known)) if known else 0.0
    return GraspCandidate(
        candidate_id="grasp_candidate_000",
        target_id=str(frame["target"]),
        region_id=f"{frame['target']}.region.{_safe_role(str(frame['semantic_role']))}",
        semantic_role=str(frame["semantic_role"]),
        center_world_m=center,
        left_contact_world_m=left,
        right_contact_world_m=right,
        approach_axis_world=approach,
        closing_axis_world=closing,
        pregrasp_center_world_m=center - approach * pregrasp_distance_m,
        opening_width_m=opening,
        generation_parameters={
            "type": "direct_learned_grasp_frame",
            "pregrasp_distance_m": pregrasp_distance_m,
        },
        position_covariance_m2=np.asarray(
            frame["point_covariance_m2"]["grasp_center"], dtype=np.float64
        ),
        orientation_covariance_rad2=np.asarray(
            frame["orientation_covariance_rad2"], dtype=np.float64
        ),
        checks=checks,
        score=score,
        score_confidence=len(known) / len(checks),
        source=str(frame["source"]),
        access="inference_visible",
    )


def generate_local_candidate_hypotheses(
    nominal: GraspCandidate,
    *,
    max_candidates: int = 5,
    orientation_offset_deg: float = 15.0,
) -> list[GraspCandidate]:
    """Expand one direct prediction into local, inference-visible alternatives.

    The alternatives represent unresolved frame hypotheses, not labelled failures
    or calibrated execution probabilities. Their score gap grows as the learned
    covariance shrinks, allowing the active-view gate to stop once the nominal
    frame is sufficiently separated from nearby alternatives.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if orientation_offset_deg <= 0.0:
        raise ValueError("orientation_offset_deg must be positive")
    position_std_m = math.sqrt(
        max(
            0.0,
            float(np.max(np.linalg.eigvalsh(nominal.position_covariance_m2))),
        )
    )
    orientation_std_rad = math.sqrt(
        max(
            0.0,
            float(np.max(np.linalg.eigvalsh(nominal.orientation_covariance_rad2))),
        )
    )
    longitudinal = _unit(
        np.cross(nominal.approach_axis_world, nominal.closing_axis_world)
    )
    translation_m = float(
        np.clip(
            max(0.25 * nominal.opening_width_m, 0.75 * position_std_m),
            0.004,
            0.020,
        )
    )
    position_penalty = min(
        0.18,
        0.04 * (translation_m / max(position_std_m, 0.003)) ** 2,
    )
    orientation_offset_rad = math.radians(float(orientation_offset_deg))
    orientation_penalty = min(
        0.18,
        0.04
        * (orientation_offset_rad / max(orientation_std_rad, math.radians(3.0))) ** 2,
    )
    hypotheses = [
        replace(
            nominal,
            generation_parameters={
                **dict(nominal.generation_parameters),
                "type": "direct_frame_nominal_hypothesis",
                "hypothesis_role": "nominal",
                "candidate_distribution_calibrated": False,
            },
        )
    ]
    for sign, suffix in ((-1.0, "negative"), (1.0, "positive")):
        delta = sign * translation_m * longitudinal
        hypotheses.append(
            replace(
                nominal,
                candidate_id=f"{nominal.candidate_id}.longitudinal_{suffix}",
                center_world_m=nominal.center_world_m + delta,
                left_contact_world_m=nominal.left_contact_world_m + delta,
                right_contact_world_m=nominal.right_contact_world_m + delta,
                pregrasp_center_world_m=nominal.pregrasp_center_world_m + delta,
                generation_parameters={
                    **dict(nominal.generation_parameters),
                    "type": "direct_frame_local_position_hypothesis",
                    "hypothesis_role": f"longitudinal_{suffix}",
                    "offset_m": round(sign * translation_m, 7),
                    "offset_axis": "longitudinal",
                    "candidate_distribution_calibrated": False,
                },
                score=max(0.0, nominal.score - position_penalty),
                source=f"{nominal.source}+covariance_local_hypothesis",
            )
        )
    contact_midpoint = (
        nominal.left_contact_world_m + nominal.right_contact_world_m
    ) / 2.0
    left_offset = nominal.left_contact_world_m - contact_midpoint
    right_offset = nominal.right_contact_world_m - contact_midpoint
    for sign, suffix in ((-1.0, "negative"), (1.0, "positive")):
        angle_rad = sign * orientation_offset_rad
        rotation = _axis_angle_rotation(nominal.approach_axis_world, angle_rad)
        closing = _unit(rotation @ nominal.closing_axis_world)
        hypotheses.append(
            replace(
                nominal,
                candidate_id=f"{nominal.candidate_id}.closing_yaw_{suffix}",
                left_contact_world_m=contact_midpoint + rotation @ left_offset,
                right_contact_world_m=contact_midpoint + rotation @ right_offset,
                closing_axis_world=closing,
                generation_parameters={
                    **dict(nominal.generation_parameters),
                    "type": "direct_frame_local_orientation_hypothesis",
                    "hypothesis_role": f"closing_yaw_{suffix}",
                    "orientation_offset_deg": sign * orientation_offset_deg,
                    "rotation_axis": "approach",
                    "orientation_requires_disambiguation": True,
                    "candidate_distribution_calibrated": False,
                },
                score=max(0.0, nominal.score - orientation_penalty),
                source=f"{nominal.source}+covariance_local_hypothesis",
            )
        )
    hypotheses.sort(key=lambda candidate: candidate.score, reverse=True)
    return hypotheses[:max_candidates]


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64) * math.cos(angle_rad)
        + (1.0 - math.cos(angle_rad)) * np.outer(axis, axis)
        + math.sin(angle_rad) * skew
    )


def refine_frame_support_clearance(
    frame: Mapping[str, Any],
    capture: Mapping[str, Any],
    *,
    radial_margin_m: float = 0.003,
    absolute_minimum_clearance_m: float = 0.018,
) -> dict[str, Any]:
    result = deepcopy(dict(frame))
    center = np.asarray(frame["center_world_m"], dtype=np.float64)
    support_z, point_count = estimate_local_support_plane_z(
        capture, center_world_m=center
    )
    opening = float(frame["opening_width_m"])
    required_clearance = max(
        absolute_minimum_clearance_m,
        0.5 * opening + radial_margin_m,
    )
    adjustment = (
        max(0.0, support_z + required_clearance - float(center[2]))
        if support_z is not None
        else 0.0
    )
    if adjustment > 0.0:
        for field in (
            "center_world_m",
            "left_contact_world_m",
            "right_contact_world_m",
        ):
            value = np.asarray(result[field], dtype=np.float64)
            value[2] += adjustment
            result[field] = np.round(value, 7).tolist()
    approach = _unit(frame["approach_axis_world"])
    support_clearance = float(center[2] - support_z) if support_z is not None else None
    regularize_approach = bool(
        support_clearance is not None
        and support_clearance < 0.08
        and float(approach[2]) < -0.85
    )
    approach_correction_deg = 0.0
    if regularize_approach:
        downward = np.asarray([0.0, 0.0, -1.0])
        approach_correction_deg = math.degrees(
            math.acos(float(np.clip(np.dot(approach, downward), -1.0, 1.0)))
        )
        result["approach_axis_world"] = downward.tolist()
        closing = np.asarray(result["closing_axis_world"], dtype=np.float64)
        closing = closing - float(np.dot(closing, downward)) * downward
        result["closing_axis_world"] = np.round(_unit(closing), 7).tolist()
    result["support_clearance_refinement"] = {
        "applied": bool(adjustment > 0.0),
        "support_plane_z_m": (
            round(float(support_z), 7) if support_z is not None else None
        ),
        "local_depth_point_count": int(point_count),
        "required_center_clearance_m": round(required_clearance, 7),
        "center_height_adjustment_m": round(adjustment, 7),
        "approach_regularized_to_support_normal": regularize_approach,
        "approach_correction_deg": round(approach_correction_deg, 6),
        "source": "local_rgbd_horizontal_support_plane",
        "access": "inference_visible",
    }
    return result


def estimate_local_support_plane_z(
    capture: Mapping[str, Any],
    *,
    center_world_m: np.ndarray,
    crop_radius_px: int = 96,
    stride: int = 3,
) -> tuple[float | None, int]:
    depth_m = np.asarray(capture["depth_mm"], dtype=np.float64) / 1000.0
    intrinsic = np.asarray(capture["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(capture["extrinsic_cv"], dtype=np.float64)
    center_camera = extrinsic @ np.concatenate([center_world_m, [1.0]])
    if center_camera[2] <= 1e-6:
        return None, 0
    center_pixel_h = intrinsic @ center_camera[:3]
    center_pixel = center_pixel_h[:2] / center_pixel_h[2]
    height, width = depth_m.shape
    x0 = max(0, int(center_pixel[0]) - crop_radius_px)
    x1 = min(width, int(center_pixel[0]) + crop_radius_px + 1)
    y0 = max(0, int(center_pixel[1]) - crop_radius_px)
    y1 = min(height, int(center_pixel[1]) + crop_radius_px + 1)
    ys, xs = np.mgrid[y0:y1:stride, x0:x1:stride]
    depths = depth_m[ys, xs]
    valid = np.isfinite(depths) & (depths > 0.2) & (depths < 1.6)
    if int(valid.sum()) < 30:
        return None, int(valid.sum())
    pixels = np.stack(
        [xs[valid] * depths[valid], ys[valid] * depths[valid], depths[valid]],
        axis=-1,
    )
    camera_points = np.linalg.solve(intrinsic, pixels.T).T
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_points = (rotation.T @ (camera_points - translation).T).T
    horizontal_radius = np.linalg.norm(world_points[:, :2] - center_world_m[:2], axis=1)
    candidates = world_points[
        (horizontal_radius <= 0.16)
        & (world_points[:, 2] >= center_world_m[2] - 0.09)
        & (world_points[:, 2] <= center_world_m[2] - 0.003)
    ]
    if len(candidates) < 30:
        return None, int(len(candidates))
    bin_width = 0.002
    indices = np.floor(candidates[:, 2] / bin_width).astype(np.int64)
    values, counts = np.unique(indices, return_counts=True)
    mode = values[int(np.argmax(counts))]
    in_mode = candidates[np.abs(indices - mode) <= 1, 2]
    return float(np.median(in_mode)), int(len(candidates))


def build_runtime_grasp_frame_graph(
    intent: GraspIntent,
    frame: Mapping[str, Any],
    candidate: GraspCandidate,
    *,
    world_state_version: int,
    visual_grounding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    graph = build_grasp_frame_graph(frame)
    graph["schema_version"] = "spatial.active_task_conditioned_grasp_frame_graph.v1"
    graph["world_state_version"] = int(world_state_version)
    graph["nodes"].insert(
        0,
        {
            "id": "intent.grasp",
            "node_type": "grasp_intent",
            "semantic_type": intent.task_goal,
            "attributes": intent.as_dict(),
            "source": intent.source,
            "access": "inference_visible",
        },
    )
    graph["edges"].insert(
        0,
        {
            "source": "intent.grasp",
            "target": frame["frame_id"],
            "relation": "requests_grasp_frame",
            "probability": intent.confidence,
            "access": "inference_visible",
        },
    )
    if visual_grounding is not None:
        graph["nodes"].insert(
            1,
            {
                "id": "observation.vlm_visual_grounding",
                "node_type": "visual_grounding_anchor",
                "semantic_type": visual_grounding["kind"],
                "position_mean_world": visual_grounding["world_point_m"],
                "position_covariance": visual_grounding["position_covariance_m2"],
                "attributes": {
                    "pixel_uv": visual_grounding["pixel_uv"],
                    "normalized_pixel_uv": visual_grounding["normalized_pixel_uv"],
                    "target_description": visual_grounding.get("target_description"),
                    "confidence": visual_grounding["confidence"],
                },
                "source": visual_grounding["source"],
                "access": "inference_visible",
            },
        )
        graph["edges"].insert(
            1,
            {
                "source": "observation.vlm_visual_grounding",
                "target": frame["frame_id"],
                "relation": "anchors_grasp_frame",
                "probability": visual_grounding["confidence"],
                "measurement": frame.get("visual_grounding_alignment"),
                "access": "inference_visible",
            },
        )
    graph["candidate_checks"] = {
        check.name: check.as_dict() for check in candidate.checks
    }
    return graph


def resolve_visual_grounding_hint(
    capture: Mapping[str, Any], hint: VisualGroundingHint
) -> dict[str, Any]:
    """Lift a VLM pixel hint into metric world space using only runtime RGB-D."""

    depth_m = np.asarray(capture["depth_mm"], dtype=np.float64) / 1000.0
    height, width = depth_m.shape
    requested_grasp_point = (
        np.asarray(hint.grasp_point_normalized_uv, dtype=np.float64)
        if hint.grasp_point_normalized_uv is not None
        else None
    )
    box_foreground = None
    if hint.target_box_normalized_xyxy is not None:
        box = np.asarray(hint.target_box_normalized_xyxy, dtype=np.float64)
        box_foreground = _target_box_foreground(depth_m, box)
    if requested_grasp_point is not None:
        normalized = requested_grasp_point.copy()
        kind = "grasp_point"
    elif box_foreground is not None:
        normalized = np.asarray(box_foreground["normalized_pixel_uv"], dtype=np.float64)
        kind = "target_box_foreground"
    else:
        normalized = np.asarray(
            [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5],
            dtype=np.float64,
        )
        kind = "target_box_center"
    pixel = np.asarray(
        [normalized[0] * (width - 1), normalized[1] * (height - 1)],
        dtype=np.float64,
    )
    radius = max(3, int(round(min(height, width) * 0.0125)))
    center_x = int(round(pixel[0]))
    center_y = int(round(pixel[1]))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    local_depths = depth_m[y0:y1, x0:x1]
    valid = local_depths[
        np.isfinite(local_depths) & (local_depths > 0.2) & (local_depths < 1.6)
    ]
    if len(valid) < 5:
        raise ValueError("visual grounding point has insufficient valid depth")
    depth = float(np.median(valid))
    depth_sigma = max(float(np.median(np.abs(valid - depth))) * 1.4826, 0.002)
    point_consistent_with_box = None
    recovered_from_box = False
    requested_point_depth = depth if requested_grasp_point is not None else None
    if requested_grasp_point is not None and box_foreground is not None:
        foreground_depth = float(box_foreground["depth_m"])
        tolerance = max(
            0.02,
            3.0 * float(box_foreground["depth_sigma_m"]),
        )
        point_consistent_with_box = abs(depth - foreground_depth) <= tolerance
        if not point_consistent_with_box:
            recovered_from_box = True
            kind = "target_box_foreground"
            normalized = np.asarray(
                box_foreground["normalized_pixel_uv"], dtype=np.float64
            )
            pixel = np.asarray(box_foreground["pixel_uv"], dtype=np.float64)
            depth = foreground_depth
            depth_sigma = float(box_foreground["depth_sigma_m"])
    intrinsic = np.asarray(capture["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(capture["extrinsic_cv"], dtype=np.float64)
    camera_point = np.linalg.solve(
        intrinsic, np.asarray([pixel[0] * depth, pixel[1] * depth, depth])
    )
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_point = rotation.T @ (camera_point - translation)
    support_z, support_point_count = estimate_scene_support_plane_z(capture)
    optical_axis_world = rotation.T @ np.asarray([0.0, 0.0, 1.0])
    anchor_height_above_support = (
        float(world_point[2] - support_z) if support_z is not None else None
    )
    if (
        support_z is not None
        and optical_axis_world[2] < -0.35
        and anchor_height_above_support is not None
        and anchor_height_above_support < 0.006
        and box_foreground is not None
    ):
        foreground_pixel = np.asarray(box_foreground["pixel_uv"], dtype=np.float64)
        foreground_depth = float(box_foreground["depth_m"])
        foreground_camera = np.linalg.solve(
            intrinsic,
            np.asarray(
                [
                    foreground_pixel[0] * foreground_depth,
                    foreground_pixel[1] * foreground_depth,
                    foreground_depth,
                ]
            ),
        )
        foreground_world = rotation.T @ (foreground_camera - translation)
        foreground_height = float(foreground_world[2] - support_z)
        if foreground_height >= 0.006:
            recovered_from_box = True
            point_consistent_with_box = False
            kind = "target_box_foreground"
            pixel = foreground_pixel
            normalized = np.asarray(
                box_foreground["normalized_pixel_uv"], dtype=np.float64
            )
            depth = foreground_depth
            depth_sigma = float(box_foreground["depth_sigma_m"])
            camera_point = foreground_camera
            world_point = foreground_world
            anchor_height_above_support = foreground_height
        else:
            raise ValueError(
                "visual grounding target box contains no foreground above the support "
                f"plane: anchor_height_m={anchor_height_above_support:.4f}, "
                f"box_foreground_height_m={foreground_height:.4f}; revise the box from "
                "the current RGB"
            )
    pixel_sigma = max(2.0, 0.5 * radius)
    if recovered_from_box and hint.target_box_normalized_xyxy is not None:
        box = hint.target_box_normalized_xyxy
        pixel_sigma = max(
            pixel_sigma,
            0.12 * max((box[2] - box[0]) * width, (box[3] - box[1]) * height),
        )
    camera_covariance = np.diag(
        [
            (depth * pixel_sigma / intrinsic[0, 0]) ** 2,
            (depth * pixel_sigma / intrinsic[1, 1]) ** 2,
            depth_sigma**2,
        ]
    )
    world_covariance = rotation.T @ camera_covariance @ rotation
    effective_confidence = hint.confidence * (0.75 if recovered_from_box else 1.0)
    return {
        **hint.as_dict(),
        "confidence": round(effective_confidence, 6),
        "kind": kind,
        "pixel_uv": np.round(pixel, 4).tolist(),
        "normalized_pixel_uv": np.round(normalized, 6).tolist(),
        "world_point_m": np.round(world_point, 7).tolist(),
        "position_covariance_m2": np.round(world_covariance, 9).tolist(),
        "depth_m": round(depth, 7),
        "depth_sample_count": int(len(valid)),
        "requested_grasp_point_pixel_uv": (
            np.round(
                [
                    requested_grasp_point[0] * (width - 1),
                    requested_grasp_point[1] * (height - 1),
                ],
                4,
            ).tolist()
            if requested_grasp_point is not None
            else None
        ),
        "requested_grasp_point_depth_m": (
            round(float(requested_point_depth), 7)
            if requested_point_depth is not None
            else None
        ),
        "target_box_foreground_depth_m": (
            round(float(box_foreground["depth_m"]), 7)
            if box_foreground is not None
            else None
        ),
        "target_box_foreground_point_count": (
            int(box_foreground["point_count"]) if box_foreground is not None else None
        ),
        "grasp_point_consistent_with_target_box": point_consistent_with_box,
        "anchor_recovered_from_target_box": recovered_from_box,
        "support_plane_z_m": (
            round(float(support_z), 7) if support_z is not None else None
        ),
        "support_plane_point_count": int(support_point_count),
        "anchor_height_above_support_m": (
            round(float(anchor_height_above_support), 7)
            if anchor_height_above_support is not None
            else None
        ),
        "access": "inference_visible",
    }


def _target_box_foreground(
    depth_m: np.ndarray,
    normalized_box: np.ndarray,
) -> dict[str, Any] | None:
    """Find a stable closest-depth foreground cluster inside a VLM target box."""

    height, width = depth_m.shape
    x0 = max(0, min(width - 1, int(math.floor(normalized_box[0] * (width - 1)))))
    y0 = max(0, min(height - 1, int(math.floor(normalized_box[1] * (height - 1)))))
    x1 = max(x0 + 1, min(width, int(math.ceil(normalized_box[2] * (width - 1))) + 1))
    y1 = max(y0 + 1, min(height, int(math.ceil(normalized_box[3] * (height - 1))) + 1))
    crop = depth_m[y0:y1, x0:x1]
    valid_mask = np.isfinite(crop) & (crop > 0.2) & (crop < 1.6)
    valid_depths = crop[valid_mask]
    if len(valid_depths) < 20:
        return None
    near_depth = float(np.quantile(valid_depths, 0.12))
    near_subset = valid_depths[valid_depths <= np.quantile(valid_depths, 0.35)]
    near_median = float(np.median(near_subset))
    near_sigma = max(
        float(np.median(np.abs(near_subset - near_median))) * 1.4826,
        0.002,
    )
    foreground_mask = valid_mask & (crop <= near_depth + max(0.012, 3.0 * near_sigma))
    minimum_count = max(9, int(math.ceil(crop.size * 0.015)))
    if int(foreground_mask.sum()) < minimum_count:
        return None
    foreground_y, foreground_x = np.nonzero(foreground_mask)
    foreground_depths = crop[foreground_mask]
    pixel = np.asarray(
        [
            x0 + float(np.median(foreground_x)),
            y0 + float(np.median(foreground_y)),
        ],
        dtype=np.float64,
    )
    depth = float(np.median(foreground_depths))
    depth_sigma = max(
        float(np.median(np.abs(foreground_depths - depth))) * 1.4826,
        0.002,
    )
    return {
        "pixel_uv": pixel.tolist(),
        "normalized_pixel_uv": [
            pixel[0] / max(width - 1, 1),
            pixel[1] / max(height - 1, 1),
        ],
        "depth_m": depth,
        "depth_sigma_m": depth_sigma,
        "point_count": int(len(foreground_depths)),
    }


def estimate_scene_support_plane_z(
    capture: Mapping[str, Any],
    *,
    stride: int = 5,
) -> tuple[float | None, int]:
    """Estimate the dominant horizontal support plane from runtime depth."""

    depth_m = np.asarray(capture["depth_mm"], dtype=np.float64) / 1000.0
    intrinsic = np.asarray(capture["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(capture["extrinsic_cv"], dtype=np.float64)
    ys, xs = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    depths = depth_m[ys, xs]
    valid = np.isfinite(depths) & (depths > 0.2) & (depths < 1.6)
    if int(valid.sum()) < 100:
        return None, int(valid.sum())
    pixels = np.stack(
        [xs[valid] * depths[valid], ys[valid] * depths[valid], depths[valid]],
        axis=-1,
    )
    camera_points = np.linalg.solve(intrinsic, pixels.T).T
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    world_points = (rotation.T @ (camera_points - translation).T).T
    heights = world_points[:, 2]
    heights = heights[np.isfinite(heights) & (heights >= 0.5) & (heights <= 1.0)]
    if len(heights) < 100:
        return None, int(len(heights))
    bin_width = 0.002
    bins = np.floor(heights / bin_width).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[int(np.argmax(counts))]
    in_mode = heights[np.abs(bins - mode) <= 1]
    if len(in_mode) < 50:
        return None, int(len(heights))
    return float(np.median(in_mode)), int(len(in_mode))


def align_frame_to_visual_grounding(
    frame: Mapping[str, Any], grounding: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate a learned local frame onto a VLM-selected object or grasp part."""

    result = deepcopy(dict(frame))
    center = np.asarray(frame["center_world_m"], dtype=np.float64)
    anchor = np.asarray(grounding["world_point_m"], dtype=np.float64)
    translation = anchor - center
    if grounding["kind"] == "target_box_center":
        translation[2] = 0.0
    else:
        translation[2] = float(np.clip(translation[2], -0.05, 0.05))
    distance = float(np.linalg.norm(translation))
    if distance > 0.35:
        raise ValueError(
            f"visual grounding correction {distance:.3f} m exceeds the 0.35 m limit"
        )
    for field in ("center_world_m", "left_contact_world_m", "right_contact_world_m"):
        point = np.asarray(frame[field], dtype=np.float64) + translation
        result[field] = np.round(point, 7).tolist()
    grounding_covariance = np.asarray(
        grounding["position_covariance_m2"], dtype=np.float64
    )
    for name in ("grasp_center", "left_contact", "right_contact"):
        original = np.asarray(frame["point_covariance_m2"][name], dtype=np.float64)
        result["point_covariance_m2"][name] = np.round(
            original + grounding_covariance, 9
        ).tolist()
    result["visual_grounding_alignment"] = {
        "kind": grounding["kind"],
        "source": grounding["source"],
        "confidence": grounding["confidence"],
        "target_description": grounding.get("target_description"),
        "anchor_world_m": grounding["world_point_m"],
        "predicted_center_before_alignment_world_m": np.round(center, 7).tolist(),
        "translation_m": np.round(translation, 7).tolist(),
        "translation_norm_m": round(distance, 7),
        "z_policy": (
            "bounded_grasp_point_alignment"
            if grounding["kind"] == "grasp_point"
            else "preserve_learned_frame_height"
        ),
        "access": "inference_visible",
    }
    result["source"] = f"{frame['source']}+vlm_visual_grounding"
    return result


def summarize_visual_grounding_alignments(
    frames: list[Mapping[str, Any]], grounding: Mapping[str, Any]
) -> dict[str, Any]:
    alignments = [
        dict(frame["visual_grounding_alignment"])
        for frame in frames
        if frame.get("visual_grounding_alignment") is not None
    ]
    if not alignments:
        raise ValueError("grounded frame fusion has no alignment evidence")
    translations = np.asarray(
        [row["translation_m"] for row in alignments], dtype=np.float64
    )
    return {
        "kind": grounding["kind"],
        "source": grounding["source"],
        "confidence": grounding["confidence"],
        "target_description": grounding.get("target_description"),
        "anchor_world_m": grounding["world_point_m"],
        "per_view_translation_m": np.round(translations, 7).tolist(),
        "mean_translation_m": np.round(translations.mean(axis=0), 7).tolist(),
        "maximum_translation_norm_m": round(
            float(np.linalg.norm(translations, axis=1).max()), 7
        ),
        "aligned_view_count": len(alignments),
        "z_policy": alignments[-1]["z_policy"],
        "access": "inference_visible",
    }


def _capture_view(
    env: Any,
    view: str,
    initial_pose: Any,
    *,
    pose_perturbation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with robotwin_cwd():
        if view == "current":
            pose = perturb_camera_pose(initial_pose, pose_perturbation)
            env.camera.get_camera().entity.set_pose(pose)
        else:
            env.camera.view(env.active_arm, view)
            if pose_perturbation is not None:
                pose = perturb_camera_pose(
                    env.camera.get_camera().entity.get_pose(), pose_perturbation
                )
                env.camera.get_camera().entity.set_pose(pose)
        env.task._update_render()
        camera = env.camera.get_camera()
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
        # Measured camera translation, not a standoff reconstruction. A position
        # rebuilt as centroid - direction * standoff is collinear with the view
        # ray, so its alignment features duplicate the ray features exactly.
        "camera_position_world_m": pose[:3, 3].tolist(),
        "camera_position_source": "measured_camera_extrinsic",
    }


def _with_layout_camera_positions(
    env: Any,
    candidate_views: list[dict[str, Any]],
    *,
    layout: CameraLayout | None = None,
    pose_perturbation: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict each candidate's real camera position from the layout table.

    The position comes from the declared view catalogue, which is not radial
    about a single point, so range and position-alignment features are not a
    restatement of the view ray. If the finger center cannot be read the rows
    are returned untouched: the scorer then falls back to the standoff
    reconstruction and labels it degenerate rather than pretending otherwise.
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
                str(row["view"]),
                finger_center_world=center,
                view_arm=str(env.active_arm),
                camera_bounds=getattr(controller, "camera_bounds", None)
                if getattr(controller, "camera_bounds", None) is not None
                else DEFAULT_CAMERA_BOUNDS,
                layout=layout or _runtime_camera_layout(env, None),
            )
            position, direction = perturb_camera_frame(
                position, direction, pose_perturbation
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
            if pose_perturbation is not None:
                merged["camera_pose_perturbation"] = dict(pose_perturbation)
        except ValueError:
            # A view outside the declared catalogue keeps its direction only.
            pass
        result.append(merged)
    return result


def _view_selection_reason(
    view: str,
    assessment: Mapping[str, Any],
    observation_index: int,
) -> str:
    if observation_index == 0:
        return "initial_observation"
    if (
        assessment.get("view_control_policy")
        == "forced_sequence_counterfactual_evaluation_only"
        and assessment.get("forced_selected_view") == view
    ):
        return "forced_counterfactual_evaluation_sequence"
    for row in assessment.get("candidate_view_scores", ()):
        if str(row.get("view")) == str(view):
            reason = row.get("reason")
            if isinstance(reason, Mapping) and reason:
                return str(max(reason, key=lambda key: float(reason[key])))
    return "selected_by_previous_active_view_assessment"


def _normalize_forced_additional_views(
    views: Sequence[str] | None,
    *,
    view_selection_mode: str,
    max_additional_views: int,
    available_views: Sequence[str] = tuple(DIRECT_FRAME_VIEW_DIRECTIONS),
) -> tuple[str, ...]:
    if view_selection_mode != "forced_evaluation":
        if views:
            raise ValueError(
                "forced additional views require forced_evaluation view mode"
            )
        return ()
    if views is None:
        raise ValueError("forced_evaluation requires forced additional views")
    normalized = tuple(str(view) for view in views)
    if len(normalized) > max_additional_views:
        raise ValueError("forced view sequence exceeds the additional-view budget")
    if len(set(normalized)) != len(normalized):
        raise ValueError("forced additional views must be unique")
    invalid = [
        view
        for view in normalized
        if view == "current" or view not in set(available_views)
    ]
    if invalid:
        raise ValueError(f"unsupported forced additional views: {invalid}")
    return normalized


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


def _world_fingerprint(env: Any) -> dict[str, np.ndarray]:
    result = {}
    with robotwin_cwd():
        actors = env.task.scene.get_all_actors()
    for index, actor in enumerate(actors):
        pose = actor.get_pose()
        result[f"{index}:{actor.get_name()}"] = np.concatenate([pose.p, pose.q]).astype(
            np.float64
        )
    return result


def _fingerprint_delta(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]
) -> float:
    if set(before) != set(after):
        return float("inf")
    return max(
        (float(np.max(np.abs(before[key] - after[key]))) for key in before),
        default=0.0,
    )


def _view_move_cost(left: Any, right: Any) -> float:
    dot = float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0))
    return math.acos(dot) / math.pi


def _unit(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or norm <= 1e-9:
        raise ValueError("active grasp-frame geometry requires a nonzero 3-vector")
    return result / norm


def _safe_role(value: str) -> str:
    return "_".join(value.lower().split()) or "preferred"
