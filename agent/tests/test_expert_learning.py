from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent.robotwin.eval import save_expert_learning_for_thread
from agent.robotwin.expert_learning import (
    ExpertLearningState,
    expert_learning_gate_error,
    read_summary_model,
    validate_expert_learning_args,
)


VALID_LEARNING = {
    "grasp_object_part": "upright bottle body",
    "grasp_region": "middle label/body region",
    "grasp_height_or_depth": "mid body height",
    "finger_placement": "two finger pads close on opposite sides of the body",
    "approach_strategy": "align horizontally, insert the body between fingers, close, then test lift",
    "posture_or_rotation": "keep side grasp posture unless image evidence requires correction",
    "pre_close_checks": ["bottle body is visibly between both physical fingers", "side/depth view confirms body height"],
    "test_lift_rule": "lift 20-30 mm and verify the bottle moves with the gripper",
}


class ExpertLearningTest(unittest.TestCase):
    def test_learn_requires_trajectory_first(self) -> None:
        error = validate_expert_learning_args(VALID_LEARNING, trajectory_retrieved=False)

        self.assertIn("blocked until expert.retrieve", error or "")

    def test_learn_requires_structured_fields(self) -> None:
        invalid = dict(VALID_LEARNING)
        invalid.pop("finger_placement")

        error = validate_expert_learning_args(invalid, trajectory_retrieved=True)

        self.assertIn("finger_placement", error or "")

    def test_optional_affordance_profile_is_strictly_structured(self) -> None:
        valid = {
            **VALID_LEARNING,
            "affordance_profile": {
                "part_shape": "broad_cylinder",
                "symmetry_class": "continuous",
                "centering_tolerance": "strict",
                "vertical_tolerance": "moderate",
                "avoid_ends": True,
                "requires_bilateral_contact": True,
                "requires_dual_arm": False,
            },
        }

        self.assertIsNone(
            validate_expert_learning_args(valid, trajectory_retrieved=True)
        )
        invalid = json.loads(json.dumps(valid))
        invalid["affordance_profile"]["part_shape"] = "bottle"
        error = validate_expert_learning_args(invalid, trajectory_retrieved=True)
        self.assertIn("part_shape must be one of", error or "")

    def test_optional_open_vocab_intent_fields_are_strictly_structured(self) -> None:
        valid = {
            **VALID_LEARNING,
            "task_stage": "initial_grasp",
            "post_grasp_goal": "lift while preserving the functional end",
            "functional_constraints": ["do not cover the functional end"],
            "required_arms": 1,
            "semantic_ambiguity": 0.2,
        }

        self.assertIsNone(
            validate_expert_learning_args(valid, trajectory_retrieved=True)
        )
        for field, value, message in (
            ("functional_constraints", [""], "functional_constraints"),
            ("required_arms", 3, "required_arms"),
            ("semantic_ambiguity", 1.2, "semantic_ambiguity"),
        ):
            invalid = {**valid, field: value}
            error = validate_expert_learning_args(
                invalid, trajectory_retrieved=True
            )
            self.assertIn(message, error or "")

    def test_observation_only_learning_has_no_local_frame_count_gate(self) -> None:
        no_frames = validate_expert_learning_args(
            VALID_LEARNING,
            trajectory_retrieved=True,
            observation_only=True,
            retrieved_frame_views=0,
        )
        one_view = validate_expert_learning_args(
            VALID_LEARNING,
            trajectory_retrieved=True,
            observation_only=True,
            retrieved_frame_views=1,
        )

        self.assertIsNone(no_frames)
        self.assertIsNone(one_view)

    def test_video_only_learning_requires_three_inspected_frames(self) -> None:
        blocked = validate_expert_learning_args(
            VALID_LEARNING,
            trajectory_retrieved=True,
            observation_only=True,
            video_only=True,
            retrieved_frame_count=2,
        )
        allowed = validate_expert_learning_args(
            VALID_LEARNING,
            trajectory_retrieved=True,
            observation_only=True,
            video_only=True,
            retrieved_frame_count=3,
        )

        self.assertIn("at least 3 distinct video frames", blocked or "")
        self.assertIsNone(allowed)

    def test_state_tracks_distinct_expert_frame_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ExpertLearningState(Path(tmp) / "expert_learning.json")

            state.mark_frame_retrieved(seed=2, step=15, view="side")
            state.mark_frame_retrieved(seed=2, step=15, view="side")
            state.mark_frame_retrieved(seed=2, step=15, view="topdown")

            self.assertEqual(state.retrieved_frame_views, {"side", "topdown"})
            self.assertEqual(len(state.retrieved_frames), 2)

    def test_state_saves_summary_and_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ExpertLearningState(Path(tmp) / "expert_learning.json")

            state.save(VALID_LEARNING)

            self.assertTrue(state.learned)
            self.assertEqual(json.loads(state.path.read_text(encoding="utf-8"))["grasp_region"], "middle label/body region")
            self.assertIn("Saved expert-learning constraints", state.prompt_text())
            self.assertIn("middle label/body region", state.prompt_text())

    def test_state_records_summary_model_without_prompting_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ExpertLearningState(Path(tmp) / "expert_learning.json")

            state.save(VALID_LEARNING, summary_model="gpt-5.6-sol")

            self.assertEqual(state.summary_model, "gpt-5.6-sol")
            self.assertEqual(read_summary_model(state.path), "gpt-5.6-sol")
            self.assertNotIn("summary_model", state.prompt_text())

    def test_eval_saves_learning_with_current_thread_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ExpertLearningState(Path(tmp) / "expert_learning.json")
            thread = SimpleNamespace(
                client=SimpleNamespace(
                    config=SimpleNamespace(model="gpt-5.6-sol"),
                )
            )

            save_expert_learning_for_thread(state, VALID_LEARNING, thread)

            self.assertEqual(read_summary_model(state.path), "gpt-5.6-sol")

    def test_state_rejects_conflicting_summary_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ExpertLearningState(Path(tmp) / "expert_learning.json")
            summary = {**VALID_LEARNING, "summary_model": "qwen3.7-plus"}

            with self.assertRaisesRegex(ValueError, "conflicts with the configured model"):
                state.save(summary, summary_model="gpt-5.6-sol")

    def test_state_loads_prelearned_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            source.write_text(json.dumps(VALID_LEARNING), encoding="utf-8")
            state = ExpertLearningState(Path(tmp) / "run" / "expert_learning.json")

            state.load_from(source)

            self.assertTrue(state.learned)
            self.assertTrue(state.trajectory_retrieved)
            self.assertEqual(json.loads(state.path.read_text(encoding="utf-8"))["grasp_object_part"], "upright bottle body")

    def test_state_preserves_prelearned_summary_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            source.write_text(
                json.dumps({**VALID_LEARNING, "summary_model": "gpt-5.6-sol"}),
                encoding="utf-8",
            )
            state = ExpertLearningState(Path(tmp) / "run" / "expert_learning.json")

            state.load_from(source)

            self.assertEqual(state.summary_model, "gpt-5.6-sol")
            self.assertEqual(read_summary_model(state.path), "gpt-5.6-sol")

    def test_read_summary_model_rejects_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.json"
            source.write_text(
                json.dumps({**VALID_LEARNING, "summary_model": 56}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-empty string"):
                read_summary_model(source)

    def test_gate_blocks_actions_until_learning_is_saved(self) -> None:
        self.assertIn("Expert-learning gate", expert_learning_gate_error(True, False) or "")
        self.assertIsNone(expert_learning_gate_error(True, True))
        self.assertIsNone(expert_learning_gate_error(False, False))


if __name__ == "__main__":
    unittest.main()
