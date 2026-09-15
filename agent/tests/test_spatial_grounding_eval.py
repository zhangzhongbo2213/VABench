from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.robotwin.spatial_grounding_eval import (
    evaluate_luna_spatial_grounding_run,
)


class SpatialGroundingRunEvaluationTest(unittest.TestCase):
    def test_recognizes_authorized_atomic_candidate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_dir = root / "spatial_tool" / "query_0001"
            query_dir.mkdir(parents=True)
            events = [
                {
                    "type": "spatial_tool_result",
                    "data": {
                        "tool": "spatial.propose_grasp_candidates",
                        "query_dir": "spatial_tool/query_0001",
                    },
                },
                {
                    "type": "candidate_executability_result",
                    "data": {
                        "candidate_id": "grasp_candidate_000",
                        "execution_authorized": True,
                    },
                },
                {
                    "type": "action",
                    "data": {
                        "action": (
                            "spatial.execute_grasp_candidate.grasp_candidate_000"
                        )
                    },
                },
                {"type": "eval_finish", "data": {"success": True}},
            ]
            (root / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            query = {
                "world_frozen": True,
                "world_fingerprint_delta": 0.0,
                "observed_view_sequence": ["current", "topdown"],
                "geometry_provenance": {
                    "oracle_object_geometry_used": False,
                    "visual_grounding": {
                        "source": "vlm",
                        "kind": "target_box_foreground",
                        "anchor_height_above_support_m": 0.01,
                    },
                },
                "candidate_graph": {
                    "nodes": [
                        {"node_type": "visual_grounding_anchor"},
                        {"semantic_type": "grasp_center"},
                        {"semantic_type": "left_contact"},
                        {"semantic_type": "right_contact"},
                    ],
                    "edges": [{"relation": "anchors_grasp_frame"}],
                },
                "top_candidates": [
                    {
                        "frame": {"center_world_m": [0.2, -0.1, 0.77]},
                        "unknown_constraints": [],
                    }
                ],
                "execution_ready": False,
            }
            (query_dir / "query_result.json").write_text(
                json.dumps(query), encoding="utf-8"
            )

            report = evaluate_luna_spatial_grounding_run(root)

            self.assertTrue(
                report["authorization"]["online_execution_authorized"]
            )
            self.assertTrue(
                report["authorization"][
                    "first_action_is_atomic_candidate_execution"
                ]
            )
            self.assertFalse(
                report["authorization"]["action_without_tool_authorization"]
            )

    def test_separates_tool_chain_execution_probe_and_online_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_dir = root / "spatial_tool" / "query_0001"
            query_dir.mkdir(parents=True)
            events = [
                {
                    "type": "decision_rejected",
                    "data": {
                        "error": "Initial direct grasp-frame candidate query is required"
                    },
                },
                {
                    "type": "spatial_tool_result",
                    "data": {
                        "tool": "spatial.propose_grasp_candidates",
                        "query_dir": "spatial_tool/query_0001",
                    },
                },
                {
                    "type": "action",
                    "data": {"action": "gripper.world_z_neg.20mm"},
                },
                {"type": "eval_finish", "data": {"success": False}},
            ]
            (root / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            query = {
                "world_frozen": True,
                "world_fingerprint_delta": 0.0,
                "observed_view_sequence": ["current", "topdown"],
                "geometry_provenance": {
                    "oracle_object_geometry_used": False,
                    "visual_grounding": {
                        "source": "vlm",
                        "kind": "target_box_foreground",
                        "target_box_normalized_xyxy": [0.7, 0.6, 0.9, 0.8],
                        "normalized_pixel_uv": [0.8, 0.7],
                        "world_point_m": [0.2, -0.1, 0.75],
                        "anchor_height_above_support_m": 0.01,
                    },
                },
                "candidate_graph": {
                    "nodes": [
                        {"node_type": "visual_grounding_anchor"},
                        {"semantic_type": "grasp_center"},
                        {"semantic_type": "left_contact"},
                        {"semantic_type": "right_contact"},
                    ],
                    "edges": [{"relation": "anchors_grasp_frame"}],
                },
                "top_candidates": [
                    {
                        "frame": {"center_world_m": [0.21, -0.11, 0.77]},
                        "unknown_constraints": ["reachable", "collision_free"],
                    }
                ],
                "recommended_candidate_id": "grasp_candidate_000",
                "execution_ready": False,
                "recommended_action": {"tool": "stop"},
            }
            (query_dir / "query_result.json").write_text(
                json.dumps(query), encoding="utf-8"
            )
            execution_path = root / "execution.json"
            execution_path.write_text(
                json.dumps(
                    {
                        "result": {
                            "candidate_id": "grasp_candidate_000",
                            "execution_success": True,
                            "task_native_safe_success": True,
                            "object_height_change_m": 0.12,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_luna_spatial_grounding_run(
                root,
                candidate_execution_result=execution_path,
                expected_target_xy_m=(0.2, -0.1),
            )

            self.assertTrue(
                report["stage_verdicts"]["vlm_to_spatial_tool_chain_pass"]
            )
            self.assertTrue(
                report["stage_verdicts"]["offline_execution_probe_pass"]
            )
            self.assertFalse(report["stage_verdicts"]["online_policy_task_success"])
            self.assertTrue(
                report["authorization"]["action_without_tool_authorization"]
            )
            self.assertAlmostEqual(
                report["visual_grounding"]["target_xy_error_m"],
                2**0.5 * 0.01,
            )

    def test_rejects_runs_without_candidate_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "events.jsonl").write_text(
                json.dumps({"type": "eval_finish", "data": {}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "No 'spatial.propose"):
                evaluate_luna_spatial_grounding_run(root)


if __name__ == "__main__":
    unittest.main()
