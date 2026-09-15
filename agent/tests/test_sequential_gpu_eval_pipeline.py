from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_sequential_gpu_eval_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("sequential_gpu_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class SequentialGpuEvalPipelineTest(unittest.TestCase):
    def test_parse_task_specs_preserves_order_and_step_caps(self) -> None:
        self.assertEqual(
            pipeline.parse_task_specs(["task_a:50", "task_b:200"]),
            [("task_a", 50), ("task_b", 200)],
        )
        with self.assertRaisesRegex(ValueError, "duplicate task"):
            pipeline.parse_task_specs(["task_a:50", "task_a:100"])

    def test_parse_seed_overrides(self) -> None:
        self.assertEqual(
            pipeline.parse_seed_overrides(["task_a:1002,1005", "task_b:1010"]),
            {"task_a": [1002, 1005], "task_b": [1010]},
        )
        with self.assertRaisesRegex(ValueError, "unique seeds"):
            pipeline.parse_seed_overrides(["task_a:1002,1002"])

    def test_cli_defaults_preserve_single_worker_shared_summary_mode(self) -> None:
        parser_args = [
            "--task-spec",
            "task_a:50",
            "--pipeline-id",
            "test",
            "--benchmark-dir",
            "/tmp/benchmark",
            "--dry-run",
        ]
        with mock.patch("sys.argv", [str(SCRIPT), *parser_args]):
            args = pipeline.parse_args()
        self.assertEqual(args.max_parallel_per_gpu, 1)
        self.assertEqual(args.experiment_mode, "shared-summary")

    def test_cli_cross_task_scheduler_is_opt_in(self) -> None:
        parser_args = [
            "--task-spec",
            "task_a:50",
            "--pipeline-id",
            "test",
            "--benchmark-dir",
            "/tmp/benchmark",
            "--cross-task",
            "--dry-run",
        ]
        with mock.patch("sys.argv", [str(SCRIPT), *parser_args]):
            args = pipeline.parse_args()
        self.assertTrue(args.cross_task)

    def test_round_robin_assigns_five_seeds_per_gpu(self) -> None:
        groups = pipeline.split_round_robin(list(range(1000, 1020)), [0, 1, 2, 3])
        self.assertEqual([len(seeds) for _, seeds in groups], [5, 5, 5, 5])
        self.assertEqual(groups[0], (0, [1000, 1004, 1008, 1012, 1016]))
        self.assertEqual(groups[3], (3, [1003, 1007, 1011, 1015, 1019]))

    def test_safe_seed_discovery_uses_latest_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = {
                "task": "task_a",
                "complete": False,
                "selected_seeds": list(range(20)),
            }
            complete = {
                "task": "task_a",
                "complete": True,
                "selected_seeds": list(range(1000, 1021)),
            }
            (root / "old.json").write_text(json.dumps(incomplete), encoding="utf-8")
            (root / "new.json").write_text(json.dumps(complete), encoding="utf-8")

            selected = pipeline.discover_safe_seed_sets(root)

        self.assertEqual(selected["task_a"]["selected_seeds"], list(range(1000, 1020)))

    def test_provider_failure_is_infrastructure_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "stop_reason": "HTTP 403 from provider: error code 1010",
                    }
                ),
                encoding="utf-8",
            )
            result = pipeline.classify_evaluation(
                {
                    "state": "completed",
                    "returncode": 0,
                    "run_dir": str(run_dir),
                }
            )
        self.assertEqual(result["category"], "INFRA")

    def test_missing_provider_text_is_infrastructure_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "stop_reason": "provider response missing text content",
                    }
                ),
                encoding="utf-8",
            )
            result = pipeline.classify_evaluation(
                {
                    "state": "completed",
                    "returncode": 0,
                    "run_dir": str(run_dir),
                }
            )
        self.assertEqual(result["category"], "INFRA")

    def test_environment_failure_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "result.json").write_text(
                json.dumps({"success": False, "stop_reason": "max_steps_reached"}),
                encoding="utf-8",
            )
            result = pipeline.classify_evaluation(
                {
                    "state": "completed",
                    "returncode": 0,
                    "run_dir": str(run_dir),
                }
            )
        self.assertEqual(result["category"], "FAILED")

    def test_running_action_count_excludes_read_only_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "seed.log"
            log.write_text(
                '\n'.join(
                    [
                        'assistant> {"action":"camera.view_topdown","reason":"inspect"}',
                        'assistant> {"tool":"camera.history","args":{"step":1}}',
                        'assistant> {"tool":"expert.retrieve","args":{"mode":"trajectory"}}',
                        'assistant> {"action":"gripper.move_world","axis":"z"}',
                    ]
                ),
                encoding="utf-8",
            )
            count = pipeline.action_count({"log": str(log)})

        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
