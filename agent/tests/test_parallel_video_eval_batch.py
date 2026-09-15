from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_parallel_video_eval_batch.py"
SPEC = importlib.util.spec_from_file_location("run_parallel_video_eval_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


LEARNING_SUMMARY = {
    "grasp_object_part": "test object",
    "grasp_region": "model-selected center",
    "grasp_height_or_depth": "model-selected depth",
    "finger_placement": "opposite sides",
    "approach_strategy": "align, insert, close, lift",
    "posture_or_rotation": "task-aligned posture",
    "pre_close_checks": ["object is between both fingers"],
    "test_lift_rule": "lift directly after closing",
}


class FakeBatchProcess:
    next_pid = 9000

    def __init__(self, command: list[str], *, env: dict[str, str], commands: list[list[str]]):
        self.command = list(command)
        self.pid = FakeBatchProcess.next_pid
        FakeBatchProcess.next_pid += 1
        commands.append(self.command)
        self._write_artifacts(env)

    def poll(self) -> int:
        return 0

    def _write_artifacts(self, env: dict[str, str]) -> None:
        def argument(name: str) -> str:
            index = self.command.index(name)
            return self.command[index + 1]

        state_dir = Path(env["AGENT_STATE_DIR"])
        task = argument("--task")
        seed = int(argument("--seed"))
        run_id = argument("--run-id")
        run_dir = state_dir / "runs" / task / f"seed_{seed}" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        model = env["AGENT_MODEL"]
        if "--learning-only" in self.command:
            summary = {
                **LEARNING_SUMMARY,
                "grasp_region": f"learned only by {model}",
                "summary_model": model,
            }
            (run_dir / "expert_learning.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            return
        summary_path = Path(argument("--expert-learning-file"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = {
            "success": True,
            "summary_model": summary["summary_model"],
            "evaluation_model": model,
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


class ParallelVideoEvalBatchTest(unittest.TestCase):
    def test_empty_provider_content_is_retryable_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps({
                "success": False,
                "stop_reason": "provider response missing text content",
            }))
            outcome = BATCH.evaluation_outcome(path, 0)
            self.assertEqual(outcome["state"], "infrastructure_error")
            self.assertIsNone(outcome["success"])

    def test_infers_shared_summary_from_learning_file(self) -> None:
        mode = BATCH.resolve_experiment_mode(
            None,
            expert_demo_dir=None,
            expert_learning_file=Path("summary.json"),
        )

        self.assertEqual(mode, BATCH.SHARED_SUMMARY_MODE)

    def test_infers_model_specific_summary_from_demo(self) -> None:
        mode = BATCH.resolve_experiment_mode(
            None,
            expert_demo_dir=Path("demo"),
            expert_learning_file=None,
        )

        self.assertEqual(mode, BATCH.MODEL_SPECIFIC_SUMMARY_MODE)

    def test_shared_summary_requires_learning_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --expert-learning-file"):
            BATCH.resolve_experiment_mode(
                BATCH.SHARED_SUMMARY_MODE,
                expert_demo_dir=Path("demo"),
                expert_learning_file=None,
            )

    def test_model_specific_summary_accepts_prelearned_summary(self) -> None:
        mode = BATCH.resolve_experiment_mode(
            BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            expert_demo_dir=None,
            expert_learning_file=Path("summary.json"),
        )

        self.assertEqual(mode, BATCH.MODEL_SPECIFIC_SUMMARY_MODE)

    def test_shared_summary_allows_a_different_evaluation_model(self) -> None:
        BATCH.validate_model_lineage(
            BATCH.SHARED_SUMMARY_MODE,
            summary_model="gpt-5.6-sol",
            evaluation_model="qwen3.7-plus",
        )

    def test_model_specific_summary_accepts_the_same_model(self) -> None:
        BATCH.validate_model_lineage(
            BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            summary_model="gpt-5.6-sol",
            evaluation_model="gpt-5.6-sol",
        )

    def test_model_specific_summary_rejects_a_different_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be learned by the evaluation model"):
            BATCH.validate_model_lineage(
                BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
                summary_model="gpt-5.6-sol",
                evaluation_model="qwen3.7-plus",
            )

    def test_eval_command_passes_experiment_mode_to_each_seed(self) -> None:
        command = BATCH.eval_command(
            task="grasp_single_cube",
            config="demo_clean",
            seed=1000,
            active_arm="right",
            benchmark_dir=Path("benchmark"),
            run_id="eval_test",
            max_steps=50,
            experiment_mode=BATCH.SHARED_SUMMARY_MODE,
            expert_learning_file=Path("summary.json"),
        )

        mode_index = command.index("--experiment-mode")
        self.assertEqual(command[mode_index + 1], BATCH.SHARED_SUMMARY_MODE)

    def test_eval_command_passes_request_timeout_as_global_option(self) -> None:
        command = BATCH.eval_command(
            task="grasp_single_cube",
            config="demo_clean",
            seed=1000,
            active_arm="right",
            benchmark_dir=Path("benchmark"),
            run_id="eval_test",
            max_steps=50,
            experiment_mode=BATCH.SHARED_SUMMARY_MODE,
            request_timeout=180.0,
            expert_learning_file=Path("summary.json"),
        )

        timeout_index = command.index("--timeout")
        eval_index = command.index("eval")
        self.assertLess(timeout_index, eval_index)
        self.assertEqual(command[timeout_index + 1], "180.0")

    def test_eval_command_passes_optional_spatial_tool_and_shadow_audit(self) -> None:
        command = BATCH.eval_command(
            task="grasp_single_pen",
            config="demo_clean",
            seed=1000,
            active_arm="right",
            benchmark_dir=Path("benchmark"),
            run_id="eval_test",
            max_steps=50,
            experiment_mode=BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            expert_learning_file=Path("summary.json"),
            spatial_tool=True,
            spatial_close_audit=True,
            spatial_candidate_proposals=True,
            spatial_checkpoint=Path("perception.pt"),
            spatial_calibration=Path("calibration.json"),
            spatial_view_ranker=Path("ranker.pt"),
            spatial_outcome_checkpoint=Path("outcome.pt"),
            spatial_candidate_ranker=Path("candidate_ranker.json"),
            spatial_semantic_part_checkpoint=Path("semantic_part.pt"),
            spatial_intent_embedding_store=Path("intent_embeddings.json"),
            spatial_semantic_view_ranker=Path("semantic_view_ranker.json"),
            spatial_semantic_view_mode="shadow",
            spatial_max_grasp_candidates=4,
        )

        self.assertIn("--spatial-tool", command)
        self.assertIn("--spatial-close-audit", command)
        self.assertIn("--spatial-candidate-proposals", command)
        self.assertIn("--spatial-candidate-ranker", command)
        self.assertIn("candidate_ranker.json", command)
        self.assertIn("--spatial-semantic-part-checkpoint", command)
        self.assertIn("semantic_part.pt", command)
        self.assertIn("--spatial-intent-embedding-store", command)
        self.assertIn("--spatial-semantic-view-ranker", command)
        self.assertIn("semantic_view_ranker.json", command)
        semantic_mode_index = command.index("--spatial-semantic-view-mode")
        self.assertEqual(command[semantic_mode_index + 1], "shadow")
        max_candidates_index = command.index("--spatial-max-grasp-candidates")
        self.assertEqual(command[max_candidates_index + 1], "4")
        self.assertIn("perception.pt", command)
        self.assertIn("ranker.pt", command)

    def test_eval_command_passes_direct_grasp_frame_and_dynamic_encoder(self) -> None:
        command = BATCH.eval_command(
            task="grasp_single_cube",
            config="demo_clean",
            seed=1000,
            active_arm="right",
            benchmark_dir=Path("benchmark"),
            run_id="eval_direct_frame",
            max_steps=50,
            experiment_mode=BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            expert_learning_file=Path("summary.json"),
            spatial_candidate_proposals=True,
            spatial_direct_grasp_frame_checkpoint=Path("frame.pt"),
            spatial_intent_embedding_store=Path("intent_embeddings.json"),
            spatial_semantic_view_ranker=Path("realized_view_ranker.json"),
            spatial_semantic_view_mode="shadow",
            spatial_dynamic_intent_encoder_python=Path("pi0-python"),
            spatial_dynamic_intent_encoder_script=Path("encode.py"),
            spatial_dynamic_intent_encoder_snapshot=Path("snapshot"),
        )

        self.assertIn("--spatial-direct-grasp-frame-checkpoint", command)
        self.assertIn("frame.pt", command)
        self.assertIn("--spatial-dynamic-intent-encoder-python", command)
        self.assertIn("pi0-python", command)
        self.assertIn("--spatial-dynamic-intent-encoder-script", command)
        self.assertIn("--spatial-dynamic-intent-encoder-snapshot", command)

    def test_evaluation_outcome_marks_provider_quota_as_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "success": False,
                        "stop_reason": (
                            'HTTP 429 from provider: {"error":'
                            '{"code":"AccountQuotaExceeded"}}'
                        ),
                    }
                ),
                encoding="utf-8",
            )

            outcome = BATCH.evaluation_outcome(result_path, 0)

        self.assertEqual(outcome["state"], "infrastructure_error")
        self.assertIsNone(outcome["success"])

    def test_evaluation_outcome_keeps_environment_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps({"success": False, "stop_reason": "max_steps_reached"}),
                encoding="utf-8",
            )

            outcome = BATCH.evaluation_outcome(result_path, 0)

        self.assertEqual(outcome["state"], "completed")
        self.assertIs(outcome["success"], False)

    def test_learning_batch_isolates_models_and_supports_prelearned_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            demo_dir = root / "video_demo"
            benchmark_dir = root / "benchmark"
            demo_dir.mkdir()
            benchmark_dir.mkdir()
            commands: list[list[str]] = []

            def fake_popen(command, **kwargs):
                return FakeBatchProcess(command, env=kwargs["env"], commands=commands)

            common_environment = {
                "AGENT_API_KEY": "test-key",
                "AGENT_TEMPERATURE": "0",
                "AGENT_REASONING_EFFORT": "xhigh",
            }
            with patch.object(BATCH.subprocess, "Popen", side_effect=fake_popen):
                with patch.dict(
                    os.environ,
                    {**common_environment, "AGENT_MODEL": "gpt-5.6-sol"},
                    clear=False,
                ):
                    self._run_learning_batch(
                        state_dir=state_dir,
                        demo_dir=demo_dir,
                        benchmark_dir=benchmark_dir,
                        batch_id="gpt_independent",
                        seeds=[1000, 1001],
                    )
                with patch.dict(
                    os.environ,
                    {**common_environment, "AGENT_MODEL": "qwen3.7-plus"},
                    clear=False,
                ):
                    self._run_learning_batch(
                        state_dir=state_dir,
                        demo_dir=demo_dir,
                        benchmark_dir=benchmark_dir,
                        batch_id="qwen_independent",
                        seeds=[1000],
                    )

                    gpt_summary = (
                        state_dir
                        / "runs/grasp_single_cube/seed_0/learning_gpt_independent/expert_learning.json"
                    )
                    qwen_summary = (
                        state_dir
                        / "runs/grasp_single_cube/seed_0/learning_qwen_independent/expert_learning.json"
                    )
                    self._run_prelearned_summary_batch(
                        state_dir=state_dir,
                        summary_file=qwen_summary,
                        benchmark_dir=benchmark_dir,
                        batch_id="qwen_prelearned_own",
                        seeds=[1001],
                        experiment_mode=BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
                    )
                    self._run_prelearned_summary_batch(
                        state_dir=state_dir,
                        summary_file=gpt_summary,
                        benchmark_dir=benchmark_dir,
                        batch_id="qwen_with_gpt_summary",
                        seeds=[1002],
                        experiment_mode=BATCH.SHARED_SUMMARY_MODE,
                    )

            gpt_status = self._status(state_dir, "gpt_independent")
            qwen_status = self._status(state_dir, "qwen_independent")
            prelearned_status = self._status(state_dir, "qwen_prelearned_own")
            shared_status = self._status(state_dir, "qwen_with_gpt_summary")
            self.assertEqual(gpt_status["summary_model"], "gpt-5.6-sol")
            self.assertEqual(gpt_status["evaluation_model"], "gpt-5.6-sol")
            self.assertEqual(qwen_status["summary_model"], "qwen3.7-plus")
            self.assertEqual(qwen_status["evaluation_model"], "qwen3.7-plus")
            self.assertEqual(
                prelearned_status["experiment_mode"],
                BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            )
            self.assertEqual(prelearned_status["summary_model"], "qwen3.7-plus")
            self.assertEqual(prelearned_status["learning"]["state"], "reused")
            self.assertEqual(shared_status["summary_model"], "gpt-5.6-sol")
            self.assertEqual(shared_status["evaluation_model"], "qwen3.7-plus")
            self.assertEqual(shared_status["learning"]["state"], "reused")

            qwen_eval_commands = [
                command
                for command in commands
                if "eval_qwen_independent" in command and "--learning-only" not in command
            ]
            qwen_learning_commands = [
                command
                for command in commands
                if "learning_qwen_independent" in command and "--learning-only" in command
            ]
            self.assertEqual(len(qwen_eval_commands), 1)
            self.assertEqual(len(qwen_learning_commands), 1)
            qwen_summary = (
                state_dir
                / "runs/grasp_single_cube/seed_0/learning_qwen_independent/expert_learning.json"
            )
            qwen_summary_data = json.loads(qwen_summary.read_text(encoding="utf-8"))
            self.assertEqual(qwen_summary_data["grasp_region"], "learned only by qwen3.7-plus")
            self.assertIn(str(qwen_summary), qwen_eval_commands[0])
            self.assertNotIn("--expert-learning-file", qwen_learning_commands[0])
            self.assertNotIn("--learned-constraint-file", qwen_eval_commands[0])
            self.assertNotIn("--learned-constraint-file", qwen_learning_commands[0])
            self.assertNotIn("--session", qwen_learning_commands[0])
            self.assertNotIn("--session", qwen_eval_commands[0])

    def _run_learning_batch(
        self,
        *,
        state_dir: Path,
        demo_dir: Path,
        benchmark_dir: Path,
        batch_id: str,
        seeds: list[int],
    ) -> None:
        argv = [
            str(SCRIPT_PATH),
            "--task",
            "grasp_single_cube",
            "--experiment-mode",
            BATCH.MODEL_SPECIFIC_SUMMARY_MODE,
            "--expert-demo-dir",
            str(demo_dir),
            "--seeds",
            *(str(seed) for seed in seeds),
            "--batch-id",
            batch_id,
            "--benchmark-dir",
            str(benchmark_dir),
            "--state-dir",
            str(state_dir),
            "--poll-seconds",
            "0.001",
        ]
        with patch.object(sys, "argv", argv):
            BATCH.main()

    def _run_prelearned_summary_batch(
        self,
        *,
        state_dir: Path,
        summary_file: Path,
        benchmark_dir: Path,
        batch_id: str,
        seeds: list[int],
        experiment_mode: str,
    ) -> None:
        argv = [
            str(SCRIPT_PATH),
            "--task",
            "grasp_single_cube",
            "--experiment-mode",
            experiment_mode,
            "--expert-learning-file",
            str(summary_file),
            "--seeds",
            *(str(seed) for seed in seeds),
            "--batch-id",
            batch_id,
            "--benchmark-dir",
            str(benchmark_dir),
            "--state-dir",
            str(state_dir),
            "--poll-seconds",
            "0.001",
        ]
        with patch.object(sys, "argv", argv):
            BATCH.main()

    def _status(self, state_dir: Path, batch_id: str) -> dict[str, object]:
        path = (
            state_dir
            / "runs/grasp_single_cube/batches"
            / batch_id
            / "batch_status.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
