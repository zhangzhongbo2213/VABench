from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from agent.cli import build_parser, spatial_tool_config_from_args
from agent.protocol import Decision
from agent.robotwin.adapter import FrameRecord
from agent.robotwin.eval import (
    ask_for_decision,
    initial_direct_candidate_gate_error,
)
from agent.robotwin.spatial_tool import (
    EXECUTE_CANDIDATE_TOOL_NAME,
    GRASP_CANDIDATE_TOOL_NAME,
    VERIFY_CANDIDATE_TOOL_NAME,
    SpatialPregraspRuntime,
    SpatialToolConfig,
    candidate_executability_model_payload,
    grasp_outcome_model_payload,
    grasp_candidate_model_payload,
    is_gripper_close_action,
    model_payload,
    spatial_tool_system_prompt,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.env = object()


class SpatialToolTest(unittest.TestCase):
    def test_candidate_rgbd_loader_uses_server_side_legacy_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_dir = root / "spatial_tool" / "query_0001"
            query_dir.mkdir(parents=True)
            image_path = query_dir / "current.png"
            depth_path = query_dir / "current_depth_m.npy"
            camera_path = query_dir / "current_camera.json"
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
            np.save(depth_path, np.ones((3, 4), dtype=np.float32))
            camera_path.write_text(
                json.dumps(
                    {
                        "intrinsic_cv": np.eye(3).tolist(),
                        "extrinsic_cv": np.eye(4).tolist(),
                        "access": "inference_visible",
                    }
                ),
                encoding="utf-8",
            )
            runtime = SpatialPregraspRuntime.__new__(SpatialPregraspRuntime)
            runtime.run_dir = root

            evidence = runtime._load_candidate_rgbd_evidence(
                {
                    "query_dir": "spatial_tool/query_0001",
                    "result": {},
                    "server_side_history": [
                        {
                            "frame_id": 0,
                            "view": "current",
                            "image": str(image_path),
                            "depth_m": str(depth_path),
                            "camera": str(camera_path),
                        }
                    ],
                }
            )

            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["depth_m"].shape, (3, 4))
            self.assertEqual(evidence[0]["access"], "inference_visible_server_side")

    def test_agent_routes_proposal_verification_and_terminal_execution(self) -> None:
        class Thread:
            def __init__(self):
                self.replies = iter(
                    [
                        json.dumps(
                            {
                                "tool": GRASP_CANDIDATE_TOOL_NAME,
                                "args": {"target": "cube"},
                            }
                        ),
                        json.dumps(
                            {
                                "tool": VERIFY_CANDIDATE_TOOL_NAME,
                                "args": {"candidate_id": "grasp_candidate_000"},
                            }
                        ),
                        json.dumps(
                            {
                                "tool": EXECUTE_CANDIDATE_TOOL_NAME,
                                "args": {"candidate_id": "grasp_candidate_000"},
                            }
                        ),
                    ]
                )

            def ask(self, _parts):
                return next(self.replies)

        class Logger:
            def __init__(self):
                self.events = []

            def event(self, event_type, data):
                self.events.append((event_type, data))

        class Adapter:
            task_name = "grasp_single_cube"

            def __init__(self, root):
                self.run_dir = root
                self.env = object()

            def model_available_actions(self):
                return ["camera.view_topdown", "gripper.open"]

            def geometry(self):
                return {"active_arm": "right"}

        class Runtime:
            def __init__(self, root):
                self.config = SpatialToolConfig(
                    candidate_proposals=True,
                    direct_grasp_frame_checkpoint=Path("frame.pt"),
                    intent_embedding_store=Path("intent.json"),
                )
                self.root = root
                self.records = []
                self.candidate_authorizations = {}
                self.executed_candidate_ids = set()

            def propose_grasp_candidates(self, *_args, **_kwargs):
                record = {
                    "query_id": 1,
                    "query_tool": GRASP_CANDIDATE_TOOL_NAME,
                    "query_dir": "spatial_tool/query_0001",
                    "policy_visible": True,
                    "result": {
                        "verdict": "uncertain",
                        "confidence": 0.7,
                        "intent": {},
                        "top_candidates": [{"id": "grasp_candidate_000"}],
                        "recommended_candidate_id": "grasp_candidate_000",
                        "candidate_ranking": {},
                        "vlm_candidate_summary": {
                            "schema_version": "spatial.vlm_grasp_candidate_summary.v1",
                            "candidate_count": 5,
                        },
                        "view_assessment": {},
                        "visual_evidence_sufficient": True,
                        "execution_ready": False,
                        "recommended_action": {"tool": VERIFY_CANDIDATE_TOOL_NAME},
                        "evidence_frames": [0, 1],
                        "evidence_views": ["current", "topdown"],
                        "candidate_graph": {},
                        "geometry_provenance": {},
                    },
                    "observed_view_sequence": ["current", "topdown"],
                    "evidence_images": [
                        str(self.root / "current.png"),
                        str(self.root / "topdown.png"),
                    ],
                    "world_frozen": True,
                    "world_fingerprint_delta": 0.0,
                }
                self.records.append(record)
                return record

            def verify_candidate_executability(self, *_args, **_kwargs):
                record = {
                    "query_id": 2,
                    "query_dir": "spatial_tool/query_0002",
                    "candidate_id": "grasp_candidate_000",
                    "result": {
                        "candidate_id": "grasp_candidate_000",
                        "checks": {},
                        "execution_authorized": True,
                        "failed_hard_constraints": [],
                        "predicted_execution_success": None,
                        "world_frozen": True,
                        "collision_scope": {},
                        "recommended_action": {"tool": EXECUTE_CANDIDATE_TOOL_NAME},
                    },
                }
                self.candidate_authorizations["grasp_candidate_000"] = record
                return record

            def candidate_execution_error(self, candidate_id, **_kwargs):
                return (
                    None
                    if candidate_id in self.candidate_authorizations
                    else "not authorized"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_path = root / "current.png"
            current_path.touch()
            runtime = Runtime(root)
            logger = Logger()
            decision = ask_for_decision(
                Thread(),
                logger,
                Adapter(root),
                FrameRecord(
                    step=0,
                    path=current_path,
                    prompt="current observation",
                    model_view="active",
                ),
                None,
                None,
                spatial_runtime=runtime,
            )

        self.assertEqual(decision.kind, "candidate_execution")
        self.assertEqual(decision.tool_args, {"candidate_id": "grasp_candidate_000"})
        self.assertEqual(len(runtime.records), 1)
        self.assertIn("grasp_candidate_000", runtime.candidate_authorizations)
        self.assertTrue(
            any(event == "candidate_executability_result" for event, _ in logger.events)
        )

    def test_direct_frame_actions_are_gated_through_candidate_execution(self) -> None:
        runtime = object.__new__(SpatialPregraspRuntime)
        runtime.config = SpatialToolConfig(
            candidate_proposals=True,
            direct_grasp_frame_checkpoint=Path("frame.pt"),
            intent_embedding_store=Path("intent.json"),
        )
        runtime.records = []
        runtime.candidate_authorizations = {}
        runtime.executed_candidate_ids = set()
        decision = Decision(kind="action", action="camera.view_topdown")

        error = initial_direct_candidate_gate_error(decision, runtime)

        self.assertIn("was not executed", error or "")
        self.assertIn("target_box_normalized_xyxy", error or "")
        runtime.records.append(
            {
                "query_tool": GRASP_CANDIDATE_TOOL_NAME,
                "policy_visible": True,
                "result": {
                    "recommended_candidate_id": "grasp_candidate_000",
                    "visual_evidence_sufficient": True,
                },
            }
        )
        error = initial_direct_candidate_gate_error(decision, runtime)
        self.assertIn(VERIFY_CANDIDATE_TOOL_NAME, error or "")
        runtime.candidate_authorizations["grasp_candidate_000"] = {
            "result": {"execution_authorized": True}
        }
        error = initial_direct_candidate_gate_error(decision, runtime)
        self.assertIn(EXECUTE_CANDIDATE_TOOL_NAME, error or "")
        runtime.candidate_authorizations.clear()
        runtime.executed_candidate_ids.add("grasp_candidate_000")
        self.assertIsNone(initial_direct_candidate_gate_error(decision, runtime))

    def test_direct_frame_action_gate_blocks_a_visually_ambiguous_candidate(
        self,
    ) -> None:
        runtime = object.__new__(SpatialPregraspRuntime)
        runtime.config = SpatialToolConfig(
            candidate_proposals=True,
            direct_grasp_frame_checkpoint=Path("frame.pt"),
            intent_embedding_store=Path("intent.json"),
        )
        runtime.records = [
            {
                "query_tool": GRASP_CANDIDATE_TOOL_NAME,
                "policy_visible": True,
                "result": {
                    "recommended_candidate_id": "grasp_candidate_000",
                    "visual_evidence_sufficient": False,
                    "view_assessment": {
                        "reasons": ["top_candidates_not_separated"],
                        "stop_reason": "view_budget_exhausted",
                    },
                },
            }
        ]
        runtime.candidate_authorizations = {}
        runtime.executed_candidate_ids = set()

        error = initial_direct_candidate_gate_error(
            Decision(kind="action", action="gripper.close"), runtime
        )

        self.assertIn("must not be verified or executed", error or "")
        self.assertIn("top_candidates_not_separated", error or "")

    def test_executability_payload_hides_server_authorization_token(self) -> None:
        payload = candidate_executability_model_payload(
            {
                "result": {
                    "candidate_id": "grasp_candidate_000",
                    "execution_authorized": True,
                    "authorization_id": "server-secret-token",
                    "world_state_digest": "server-world-state",
                    "checks": {},
                    "failed_hard_constraints": [],
                    "world_frozen": True,
                }
            }
        )

        self.assertTrue(payload["execution_authorized"])
        self.assertNotIn("authorization_id", payload)
        self.assertNotIn("world_state_digest", payload)

    def test_candidate_rgbd_artifacts_stay_server_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_dir = root / "spatial_tool" / "query_0001"
            views_dir = query_dir / "views"
            views_dir.mkdir(parents=True)
            np.save(
                views_dir / "00_current_depth_m.npy",
                np.full((4, 5), 0.8, dtype=np.float32),
                allow_pickle=False,
            )
            (views_dir / "00_current_camera.json").write_text(
                json.dumps(
                    {
                        "view": "current",
                        "intrinsic_cv": np.eye(3).tolist(),
                        "extrinsic_cv": np.eye(4)[:3].tolist(),
                        "access": "inference_visible",
                    }
                ),
                encoding="utf-8",
            )
            runtime = object.__new__(SpatialPregraspRuntime)
            runtime.run_dir = root
            record = {
                "query_dir": "spatial_tool/query_0001",
                "result": {
                    "history": [
                        {
                            "frame_id": 0,
                            "view": "current",
                            "depth_m": "views/00_current_depth_m.npy",
                            "camera": "views/00_current_camera.json",
                        }
                    ]
                },
            }

            evidence = runtime._load_candidate_rgbd_evidence(record)
            model_visible = grasp_candidate_model_payload(record)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["depth_m"].shape, (4, 5))
        self.assertNotIn("history", model_visible)
        self.assertNotIn("depth_m", json.dumps(model_visible))

    def test_grasp_outcome_payload_hides_artifact_paths_and_task_truth(self) -> None:
        payload = grasp_outcome_model_payload(
            {
                "candidate_id": "grasp_candidate_000",
                "verdict": "true",
                "confidence": 0.9,
                "relations": {"grasp_retained_after_lift": {"state": "pass"}},
                "measurements": {"height_above_support_m": 0.12},
                "history": [{"image": "private/post.png"}],
                "task_success_evaluation_only": True,
                "frozen_appearance_shadow": {
                    "available": True,
                    "pre_post_cosine": -0.2,
                    "aggregate_descriptor": [0.1, 0.2],
                    "controls_verdict": False,
                    "reason": "shadow_only",
                },
            }
        )

        self.assertEqual(payload["verdict"], "true")
        serialized = json.dumps(payload)
        self.assertNotIn("private/post.png", serialized)
        self.assertNotIn("task_success", serialized)
        self.assertNotIn("aggregate_descriptor", serialized)
        self.assertEqual(payload["frozen_appearance_shadow"]["pre_post_cosine"], -0.2)

    def test_postgrasp_evidence_is_consumed_once(self) -> None:
        runtime = object.__new__(SpatialPregraspRuntime)
        runtime._pending_grasp_outcome = {
            "result": {"verdict": "true"},
            "evidence_images": ["one.png"],
        }

        first = runtime.consume_pending_grasp_outcome()
        second = runtime.consume_pending_grasp_outcome()

        self.assertEqual(first["result"]["verdict"], "true")
        self.assertIsNone(second)

    def test_next_model_turn_receives_postgrasp_payload_and_image(self) -> None:
        class Thread:
            def __init__(self):
                self.parts = None

            def ask(self, parts):
                self.parts = parts
                return '{"stop":true,"reason":"verified"}'

        class Logger:
            def event(self, *_args):
                return None

        class Adapter:
            task_name = "grasp_single_cube_generalization"

            def model_available_actions(self):
                return ["gripper.open"]

            def geometry(self):
                return {"active_arm": "right"}

        class Runtime:
            config = SpatialToolConfig(
                candidate_proposals=True,
                direct_grasp_frame_checkpoint=Path("frame.pt"),
                intent_embedding_store=Path("intent.json"),
            )

            def __init__(self, image_path):
                self.pending = {
                    "result": {
                        "candidate_id": "grasp_candidate_000",
                        "verdict": "true",
                        "confidence": 0.9,
                    },
                    "evidence_images": [str(image_path)],
                }

            def consume_pending_grasp_outcome(self):
                value, self.pending = self.pending, None
                return value

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_path = root / "current.png"
            diagnostic_path = root / "post.png"
            current_path.touch()
            diagnostic_path.touch()
            thread = Thread()

            decision = ask_for_decision(
                thread,
                Logger(),
                Adapter(),
                FrameRecord(
                    step=1,
                    path=current_path,
                    prompt="current observation",
                    model_view="active",
                ),
                None,
                None,
                spatial_runtime=Runtime(diagnostic_path),
            )

        self.assertEqual(decision.kind, "stop")
        text_parts = [part.text or "" for part in thread.parts if part.type == "text"]
        image_parts = [part for part in thread.parts if part.type == "image"]
        self.assertIn("spatial.verify_grasp_outcome", "\n".join(text_parts))
        self.assertEqual(len(image_parts), 2)
        self.assertEqual(image_parts[-1].label, "post-lift grasp diagnostic view 1")

    def test_cli_builds_direct_grasp_frame_config(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "robotwin",
                "--task",
                "grasp_single_cube",
                "--spatial-candidate-proposals",
                "--spatial-direct-grasp-frame-checkpoint",
                "frame.pt",
                "--spatial-intent-embedding-store",
                "embeddings.json",
                "--spatial-dynamic-intent-encoder-python",
                "pi0-python",
                "--spatial-dynamic-intent-encoder-script",
                "encode.py",
                "--spatial-dynamic-intent-encoder-snapshot",
                "snapshot",
                "--spatial-appearance-encoder-mode",
                "smolvlm_shadow",
                "--spatial-appearance-encoder-python",
                "pi0-python",
                "--spatial-appearance-encoder-script",
                "encode-appearance.py",
                "--spatial-appearance-encoder-snapshot",
                "vision-snapshot",
                "--spatial-appearance-projection-checkpoint",
                "appearance.pt",
            ]
        )

        config = spatial_tool_config_from_args(args)

        self.assertEqual(config.direct_grasp_frame_checkpoint, Path("frame.pt"))
        self.assertEqual(config.dynamic_intent_encoder_python, Path("pi0-python"))
        self.assertEqual(config.dynamic_intent_encoder_script, Path("encode.py"))
        self.assertEqual(config.appearance_encoder_mode, "smolvlm_shadow")
        self.assertEqual(config.appearance_projection_checkpoint, Path("appearance.pt"))
        self.assertEqual(config.dynamic_intent_encoder_snapshot, Path("snapshot"))

    def test_cli_builds_optional_tool_and_shadow_audit_config(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "robotwin",
                "--task",
                "grasp_single_pen",
                "--spatial-tool",
                "--spatial-close-audit",
                "--spatial-candidate-proposals",
                "--spatial-checkpoint",
                "perception.pt",
                "--spatial-calibration",
                "calibration.json",
                "--spatial-view-ranker",
                "ranker.pt",
                "--spatial-outcome-checkpoint",
                "outcome.pt",
                "--spatial-candidate-ranker",
                "candidate_ranker.json",
                "--spatial-semantic-part-checkpoint",
                "semantic_part.pt",
                "--spatial-intent-embedding-store",
                "intent_embeddings.json",
                "--spatial-semantic-view-ranker",
                "semantic_view_ranker.json",
                "--spatial-semantic-view-mode",
                "shadow",
                "--spatial-max-grasp-candidates",
                "4",
            ]
        )

        config = spatial_tool_config_from_args(args)

        self.assertTrue(config.enabled)
        self.assertTrue(config.close_audit)
        self.assertTrue(config.candidate_proposals)
        self.assertEqual(config.max_additional_views, 2)
        self.assertEqual(config.max_grasp_candidates, 4)
        self.assertEqual(config.checkpoint, Path("perception.pt"))
        self.assertEqual(
            config.candidate_ranker_checkpoint, Path("candidate_ranker.json")
        )
        self.assertEqual(config.semantic_part_checkpoint, Path("semantic_part.pt"))
        self.assertEqual(config.intent_embedding_store, Path("intent_embeddings.json"))
        self.assertEqual(
            config.semantic_view_ranker_checkpoint,
            Path("semantic_view_ranker.json"),
        )
        self.assertEqual(config.semantic_view_mode, "shadow")

    def test_prompt_makes_tool_optional_and_shadow_policy_invisible(self) -> None:
        prompt = spatial_tool_system_prompt(
            SpatialToolConfig(
                enabled=True,
                close_audit=True,
                candidate_proposals=True,
                max_additional_views=2,
            )
        )

        self.assertIn("You decide whether and when", prompt)
        self.assertIn("not a mandatory gate", prompt)
        self.assertIn("policy-invisible shadow audit", prompt)
        self.assertIn(GRASP_CANDIDATE_TOOL_NAME, prompt)
        self.assertIn("semantic grasp region", prompt)
        self.assertIn("target_box_normalized_xyxy", prompt)
        self.assertIn("origin at the top-left", prompt)
        self.assertTrue(SpatialToolConfig(candidate_proposals=True).active)

    def test_direct_grasp_frame_config_supports_dynamic_local_intent_encoding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "frame.pt"
            embeddings = root / "embeddings.json"
            ranker = root / "ranker.json"
            encoder_python = root / "python"
            encoder_script = root / "encode.py"
            snapshot = root / "snapshot"
            appearance_script = root / "encode_appearance.py"
            appearance_snapshot = root / "appearance_snapshot"
            appearance_projection = root / "appearance.pt"
            for path in (
                checkpoint,
                embeddings,
                ranker,
                encoder_python,
                encoder_script,
                appearance_script,
                appearance_projection,
            ):
                path.touch()
            snapshot.mkdir()
            appearance_snapshot.mkdir()
            config = SpatialToolConfig(
                candidate_proposals=True,
                direct_grasp_frame_checkpoint=checkpoint,
                intent_embedding_store=embeddings,
                semantic_view_ranker_checkpoint=ranker,
                semantic_view_mode="shadow",
                dynamic_intent_encoder_python=encoder_python,
                dynamic_intent_encoder_script=encoder_script,
                dynamic_intent_encoder_snapshot=snapshot,
                appearance_encoder_mode="smolvlm_shadow",
                appearance_encoder_python=encoder_python,
                appearance_encoder_script=appearance_script,
                appearance_encoder_snapshot=appearance_snapshot,
                appearance_projection_checkpoint=appearance_projection,
            )

            config.validate(task="grasp_single_cube_generalization")

            with self.assertRaisesRegex(ValueError, "does not support task"):
                config.validate(task="lift_pot")

            prompt = spatial_tool_system_prompt(config)
            self.assertIn("requires one initial", prompt)
            self.assertIn("before any standalone camera action", prompt)
            self.assertIn("target name alone is insufficient", prompt)
            self.assertIn(VERIFY_CANDIDATE_TOOL_NAME, prompt)
            self.assertIn(EXECUTE_CANDIDATE_TOOL_NAME, prompt)
            self.assertIn("later verification calls remain optional", prompt)
            self.assertIn("never changes authorization", prompt)

    def test_direct_and_semantic_perception_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            SpatialToolConfig(
                candidate_proposals=True,
                semantic_part_checkpoint=Path("semantic.pt"),
                direct_grasp_frame_checkpoint=Path("frame.pt"),
                intent_embedding_store=Path("embedding.json"),
            ).validate(task="grasp_single_cube")

    def test_candidate_payload_hides_semantic_view_shadow_audit(self) -> None:
        record = {
            "result": {
                "view_assessment": {
                    "recommended_view": "topdown",
                    "learned_view_ranker_shadow": {"selected_view": "side"},
                },
                "server_side_auxiliary": {"intent_embedding": [0.1, 0.2]},
            },
            "world_frozen": True,
            "world_fingerprint_delta": 0.0,
        }

        payload = grasp_candidate_model_payload(record)

        self.assertEqual(payload["view_assessment"]["recommended_view"], "topdown")
        self.assertNotIn("learned_view_ranker_shadow", payload["view_assessment"])
        self.assertNotIn("server_side_auxiliary", payload)
        self.assertNotIn("intent_embedding", json.dumps(payload))

    def test_candidate_payload_hides_blocked_candidate_ranker_shadow(self) -> None:
        payload = grasp_candidate_model_payload(
            {
                "result": {
                    "candidate_ranking": {
                        "source": "analytic_candidate_score",
                        "control_mode": "shadow",
                        "deployment_status": "blocked",
                        "checkpoint": "/server/private/checkpoint.json",
                        "shadow_ranked_candidates": [
                            {
                                "candidate_id": "grasp_candidate_000",
                                "predicted_safe_probability": 1.0,
                            }
                        ],
                    },
                    "view_assessment": {},
                }
            }
        )

        ranking = payload["candidate_ranking"]
        self.assertNotIn("shadow_ranked_candidates", ranking)
        self.assertNotIn("checkpoint", ranking)
        self.assertTrue(ranking["policy_invisible_shadow_audit_recorded"])

    def test_close_detection_accepts_single_and_dual_gripper(self) -> None:
        self.assertTrue(is_gripper_close_action("gripper.close"))
        self.assertTrue(is_gripper_close_action("right_gripper.close"))
        self.assertTrue(is_gripper_close_action("dual_gripper.close"))
        self.assertFalse(is_gripper_close_action("gripper.open"))

    def test_runtime_records_model_visibility_and_clamps_view_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / name for name in ("p.pt", "c.json", "v.pt", "o.pt")]
            for path in files:
                path.touch()
            integration_files = (
                root / "active_spatial_benchmark" / "pregrasp_tool.py",
                root / "active_spatial_benchmark" / "grasp_candidates.py",
                root / "active_spatial_benchmark" / "grasp_candidate_ranker.py",
                root / "scripts" / "evaluate_phase8_recovery_loop.py",
                root / "scripts" / "run_phase1_spatial_graph_demo.py",
            )
            for path in integration_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            config = SpatialToolConfig(
                enabled=True,
                close_audit=True,
                checkpoint=files[0],
                calibration=files[1],
                view_ranker_checkpoint=files[2],
                outcome_checkpoint=files[3],
                max_additional_views=2,
            )
            runtime = SpatialPregraspRuntime(
                config,
                benchmark_dir=root,
                run_dir=root / "run",
                task="grasp_single_pen",
            )
            calls: list[dict[str, object]] = []

            def fake_diagnose(**kwargs):
                calls.append(kwargs)
                output_dir = kwargs["output_dir"]
                output_dir.mkdir(parents=True, exist_ok=True)
                image = output_dir / "view.png"
                image.touch()
                final = {
                    "verdict": "uncertain",
                    "confidence": 0.6,
                    "relations": {"object_between_fingers": 0.7},
                    "grasp_success_if_execute": 0.5,
                    "evidence_frames": [1],
                    "evidence_views": ["current"],
                    "missing_evidence": ["right_contact"],
                    "recommended_action": "camera.view_side_top_45",
                    "stop_reason": "view_budget_exhausted",
                    "candidate_view_scores": [],
                    "belief_graph": {"nodes": [], "edges": []},
                }
                return {
                    "final_result": final,
                    "history": [{"image": str(image)}],
                    "observed_view_sequence": ["current"],
                    "world_frozen": True,
                    "world_fingerprint_delta": 0.0,
                }

            runtime._tool = object()
            runtime._diagnose = fake_diagnose
            runtime._clone_camera_pose = lambda env: "camera-pose"

            model_record = runtime.query(
                FakeAdapter(),
                trigger="model",
                step=7,
                tool_args={"target": "pen", "max_additional_views": 99},
            )
            shadow_record = runtime.query(FakeAdapter(), trigger="close_shadow", step=7)

            self.assertTrue(model_record["policy_visible"])
            self.assertFalse(shadow_record["policy_visible"])
            self.assertEqual(calls[0]["max_additional_views"], 2)
            self.assertEqual(model_payload(model_record)["verdict"], "uncertain")
            self.assertTrue(
                (
                    root / "run" / model_record["query_dir"] / "query_result.json"
                ).is_file()
            )

    def test_candidate_proposal_compiles_expert_intent_and_sparse_rgbd_graph(
        self,
    ) -> None:
        workspace = Path(__file__).resolve().parents[2]
        benchmark_dir = workspace / "active_spatial_benchmark_xyz"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / name for name in ("p.pt", "c.json", "v.pt", "o.pt")]
            for path in files:
                path.touch()
            runtime = SpatialPregraspRuntime(
                SpatialToolConfig(
                    enabled=True,
                    candidate_proposals=True,
                    checkpoint=files[0],
                    calibration=files[1],
                    view_ranker_checkpoint=files[2],
                    outcome_checkpoint=files[3],
                    max_grasp_candidates=3,
                ),
                benchmark_dir=benchmark_dir,
                run_dir=root / "run",
                task="grasp_single_pen",
            )

            def fake_diagnose(**kwargs):
                output_dir = kwargs["output_dir"]
                output_dir.mkdir(parents=True, exist_ok=True)
                image = output_dir / "view.png"
                image.touch()
                belief_graph = {
                    "access": "inference_visible",
                    "query_axes": {"closing_axis_world": [1.0, 0.0, 0.0]},
                    "nodes": [
                        {
                            "id": "object.center",
                            "position_mean_world_m": [0.2, -0.15, 0.8],
                            "position_covariance_m2": [
                                [0.0001, 0.0, 0.0],
                                [0.0, 0.0004, 0.0],
                                [0.0, 0.0, 0.0001],
                            ],
                            "visibility": 0.9,
                            "observation_count": 2,
                        },
                        {
                            "id": "object.axis_start",
                            "position_mean_world_m": [0.2, -0.25, 0.8],
                        },
                        {
                            "id": "object.axis_end",
                            "position_mean_world_m": [0.2, -0.05, 0.8],
                        },
                    ],
                    "edges": [],
                }
                final = {
                    "verdict": "uncertain",
                    "confidence": 0.6,
                    "evidence_frames": [1, 2],
                    "evidence_views": ["current", "side"],
                    "candidate_view_scores": [],
                    "belief_graph": belief_graph,
                }
                return {
                    "final_result": final,
                    "history": [{"image": str(image)}],
                    "observed_view_sequence": ["current", "side"],
                    "world_frozen": True,
                    "world_fingerprint_delta": 0.0,
                }

            runtime._tool = object()
            runtime._diagnose = fake_diagnose
            runtime._clone_camera_pose = lambda env: "camera-pose"
            record = runtime.propose_grasp_candidates(
                FakeAdapter(),
                step=4,
                tool_args={"target": "pen", "max_candidates": 2},
                expert_summary={
                    "grasp_object_part": "pen barrel/body",
                    "grasp_region": "central barrel",
                    "grasp_height_or_depth": "deep between pads",
                    "finger_placement": "opposed sides",
                    "approach_strategy": "top down",
                    "posture_or_rotation": "transverse to barrel",
                    "pre_close_checks": ["check both sides"],
                    "test_lift_rule": "close then lift",
                    "summary_model": "gpt-5.6-luna",
                },
            )

            payload = grasp_candidate_model_payload(record)
            self.assertEqual(payload["query"], GRASP_CANDIDATE_TOOL_NAME)
            self.assertEqual(payload["intent"]["source_model"], "gpt-5.6-luna")
            self.assertEqual(len(payload["top_candidates"]), 2)
            self.assertEqual(payload["candidate_graph"]["access"], "inference_visible")
            self.assertEqual(
                payload["vlm_candidate_summary"]["schema_version"],
                "spatial.vlm_grasp_candidate_summary.v1",
            )
            self.assertEqual(
                payload["geometry_provenance"]["center_and_axis"],
                "learned_multi_view_rgbd_sparse_graph",
            )
            self.assertIn(
                "reachable", payload["top_candidates"][0]["unknown_constraints"]
            )
            self.assertEqual(
                payload["candidate_ranking"]["source"], "analytic_candidate_score"
            )

    def test_optional_candidate_ranker_reorders_a_broad_candidate_pool(self) -> None:
        from active_spatial_benchmark.grasp_candidate_ranker import (
            CANDIDATE_FEATURE_NAMES,
            RANKER_SCHEMA,
        )

        workspace = Path(__file__).resolve().parents[2]
        benchmark_dir = workspace / "active_spatial_benchmark_xyz"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / name for name in ("p.pt", "c.json", "v.pt", "o.pt")]
            for path in files:
                path.touch()
            weights = [0.0] * len(CANDIDATE_FEATURE_NAMES)
            weights[CANDIDATE_FEATURE_NAMES.index("vertical_offset_m")] = 1000.0
            checkpoint = root / "candidate_ranker.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": RANKER_SCHEMA,
                        "model_type": "standardized_pairwise_linear_ranker",
                        "feature_names": list(CANDIDATE_FEATURE_NAMES),
                        "feature_mean": [0.0] * len(CANDIDATE_FEATURE_NAMES),
                        "feature_scale": [1.0] * len(CANDIDATE_FEATURE_NAMES),
                        "weights": weights,
                        "training_metadata": {"deployment_gate_status": "passed"},
                    }
                ),
                encoding="utf-8",
            )
            runtime = SpatialPregraspRuntime(
                SpatialToolConfig(
                    enabled=True,
                    candidate_proposals=True,
                    checkpoint=files[0],
                    calibration=files[1],
                    view_ranker_checkpoint=files[2],
                    outcome_checkpoint=files[3],
                    candidate_ranker_checkpoint=checkpoint,
                    candidate_ranker_mode="gated",
                ),
                benchmark_dir=benchmark_dir,
                run_dir=root / "run",
                task="grasp_single_pen",
            )
            module = runtime._load_candidate_module()
            intent = module.GraspIntent(
                target="pen",
                task_goal="pick_up",
                preferred_roles=("barrel",),
            )
            geometry = module.OrientedObjectGeometry(
                object_id="object.pen",
                center_world_m=np.array([0.2, -0.15, 0.8]),
                rotation_world=np.eye(3),
                half_extents_m=np.array([0.01, 0.1, 0.01]),
                source="learned_sparse_graph",
                access="inference_visible",
            )
            candidates = module.generate_obb_grasp_candidates(
                intent,
                geometry,
                config=module.CandidateGenerationConfig(
                    max_candidates=None,
                    along_axis_fractions=(0.0,),
                    vertical_offsets_m=(0.0, 0.04),
                ),
            )

            selected, ranking = runtime._rank_candidates(
                intent, candidates, max_candidates=1
            )

            self.assertEqual(
                selected[0].generation_parameters["vertical_offset_m"], 0.04
            )
            self.assertEqual(
                ranking["source"], "phase11_pairwise_linear_pen_ranker_gated"
            )
            self.assertFalse(ranking["calibrated_probability"])

    def test_context_aware_candidate_ranker_stays_shadow_when_context_is_missing(
        self,
    ) -> None:
        from active_spatial_benchmark.grasp_candidate_ranker import (
            CANDIDATE_FEATURE_NAMES,
            RANKER_SCHEMA,
        )

        workspace = Path(__file__).resolve().parents[2]
        benchmark_dir = workspace / "active_spatial_benchmark_xyz"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "candidate_ranker.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": RANKER_SCHEMA,
                        "model_type": "standardized_pairwise_linear_ranker",
                        "feature_names": list(CANDIDATE_FEATURE_NAMES),
                        "feature_mean": [0.0] * len(CANDIDATE_FEATURE_NAMES),
                        "feature_scale": [1.0] * len(CANDIDATE_FEATURE_NAMES),
                        "weights": [0.0] * len(CANDIDATE_FEATURE_NAMES),
                        "training_metadata": {
                            "deployment_gate_status": "blocked",
                            "requires_preexecution_context": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtime = SpatialPregraspRuntime.__new__(SpatialPregraspRuntime)
            runtime.config = SpatialToolConfig(
                candidate_ranker_checkpoint=checkpoint,
                candidate_ranker_mode="shadow",
            )
            runtime.benchmark_dir = benchmark_dir
            runtime._candidate_module = None
            runtime._candidate_ranker = None
            runtime._candidate_ranker_payload = None
            module = runtime._load_candidate_module()
            intent = module.GraspIntent(
                target="pen", task_goal="pick_up", preferred_roles=("barrel",)
            )
            geometry = module.OrientedObjectGeometry(
                object_id="object.pen",
                center_world_m=np.array([0.2, -0.15, 0.8]),
                rotation_world=np.eye(3),
                half_extents_m=np.array([0.01, 0.1, 0.01]),
                source="learned_sparse_graph",
                access="inference_visible",
            )
            candidates = module.generate_obb_grasp_candidates(
                intent,
                geometry,
                config=module.CandidateGenerationConfig(max_candidates=2),
            )

            selected, ranking = runtime._rank_candidates(
                intent, candidates, max_candidates=1
            )

            self.assertEqual(selected[0].candidate_id, candidates[0].candidate_id)
            self.assertEqual(
                ranking["deployment_status"],
                "blocked_missing_preexecution_context",
            )
            self.assertEqual(ranking["shadow_ranked_candidates"], [])

    def test_active_tool_rejects_untrained_task(self) -> None:
        config = SpatialToolConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, "does not support"):
            config.validate(task="grasp_single_bottle")

    def test_semantic_candidate_mode_allows_guarded_cube_without_pen_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic = root / "semantic.pt"
            embeddings = root / "embeddings.json"
            semantic.touch()
            embeddings.touch()
            config = SpatialToolConfig(
                candidate_proposals=True,
                semantic_part_checkpoint=semantic,
                intent_embedding_store=embeddings,
            )

            config.validate(task="grasp_single_cube")

            incomplete = SpatialToolConfig(
                candidate_proposals=True,
                semantic_part_checkpoint=semantic,
            )
            with self.assertRaisesRegex(ValueError, "requires"):
                incomplete.validate(task="grasp_single_cube")

            missing_ranker = SpatialToolConfig(
                candidate_proposals=True,
                semantic_part_checkpoint=semantic,
                intent_embedding_store=embeddings,
                semantic_view_mode="shadow",
            )
            with self.assertRaisesRegex(ValueError, "shadow mode requires"):
                missing_ranker.validate(task="grasp_single_cube")


if __name__ == "__main__":
    unittest.main()
