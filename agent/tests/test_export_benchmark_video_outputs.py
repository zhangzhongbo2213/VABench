from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_benchmark_video_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("export_benchmark_video_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportBenchmarkVideoOutputsTest(unittest.TestCase):
    def test_exports_actions_reasons_and_final_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / "task" / "seed_1000" / "eval"
            run_dir.mkdir(parents=True)
            events = [
                {
                    "type": "action",
                    "data": {
                        "step": 0,
                        "action": "camera.view_topdown",
                        "action_payload": "camera.view_topdown",
                        "reason": "inspect alignment",
                    },
                    "created_at": "2026-08-01T00:00:00Z",
                },
                {
                    "type": "action",
                    "data": {
                        "step": 1,
                        "action": "gripper.world_z_neg.20mm",
                        "action_payload": {
                            "type": "move_world",
                            "axis": "z",
                            "sign": "-",
                            "distance_mm": 20,
                        },
                        "reason": "insert toward the object",
                    },
                    "created_at": "2026-08-01T00:00:01Z",
                },
            ]
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task": "task",
                        "seed": 1000,
                        "success": False,
                        "stop_reason": "max_steps_reached",
                    }
                ),
                encoding="utf-8",
            )
            outputs = root / "outputs"
            outputs.mkdir()
            manifest_path = outputs / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "task": "task",
                                "query_task": "task",
                                "split": "original",
                                "output_model": "model",
                                "seed": 1000,
                                "success": False,
                                "manual_override": False,
                                "source_id": "local",
                                "source_run_dir": str(run_dir),
                                "batch_id": "batch",
                                "destination_relative": (
                                    "single_arm/task/original/model/"
                                    "seed_1000_FAILED.mp4"
                                ),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = exporter.export_manifest(
                manifest_path,
                roots={"local": Path("/")},
            )

            sidecar_path = (
                outputs / "single_arm/task/original/model/seed_1000_FAILED.json"
            )
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["sidecar_count"], 1)
            self.assertEqual(sidecar["action_count"], 2)
            self.assertEqual(sidecar["actions"][0]["reason"], "inspect alignment")
            self.assertEqual(
                sidecar["final_action"]["action"], "gripper.world_z_neg.20mm"
            )
            self.assertEqual(sidecar["final_reason"], "insert toward the object")
            self.assertEqual(updated_manifest["materialized_output_count"], 1)
            self.assertTrue(updated_manifest["videos"][0]["output_metadata_available"])

    def test_falls_back_to_result_steps_without_events(self) -> None:
        actions, source = exporter.action_rows(
            [],
            {
                "steps": [
                    {"step": 0, "action": None},
                    {"step": 1, "action": "gripper.close"},
                ]
            },
        )

        self.assertEqual(source, "result.json:steps")
        self.assertEqual(actions[0]["action"], "gripper.close")
        self.assertIsNone(actions[0]["reason"])

    def test_falls_back_to_archived_metadata_next_to_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stem = root / "seed_1000_FAILED"
            stem.with_suffix(".events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "action",
                        "data": {
                            "step": 4,
                            "action": "gripper.close",
                            "reason": "close after verification",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stem.with_suffix(".result.json").write_text(
                json.dumps({"task": "task", "seed": 1000, "success": False}),
                encoding="utf-8",
            )

            sidecar, report = exporter.build_sidecar(
                {"task": "task", "seed": 1000},
                run_dir=root / "unavailable-run",
                archived_stem=stem,
            )

            self.assertTrue(report["metadata_available"])
            self.assertEqual(sidecar["action_count"], 1)
            self.assertEqual(sidecar["final_reason"], "close after verification")

    def test_recovers_unlisted_hardlinked_local_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs/task/seed_1000/eval_batch"
            run_dir.mkdir(parents=True)
            source = run_dir / "model_replay.mp4"
            source.write_bytes(b"video")
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "evaluation_model": "model",
                        "summary_model": "model",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            destination = (
                output_dir / "single_arm/task/original/model/seed_1000_SUCCESS.mp4"
            )
            destination.parent.mkdir(parents=True)
            destination.hardlink_to(source)
            manifest = {"videos": []}

            recovered = exporter.recover_unlisted_local_videos(
                manifest,
                output_dir=output_dir,
                runs_root=root / "runs",
            )

            self.assertEqual(recovered, 1)
            self.assertEqual(manifest["videos"][0]["seed"], 1000)
            self.assertEqual(manifest["videos"][0]["source_run_dir"], str(run_dir))


if __name__ == "__main__":
    unittest.main()
