from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.robotwin.learned_constraints import LearnedConstraintSet, default_learned_constraint_file
from agent.robotwin.loop import run_robotwin_loop
from agent.robotwin.correction import task_scoped_update


class LoopConstraintsTest(unittest.TestCase):
    def test_learned_constraint_prompt_includes_active_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.json"
            state = LearnedConstraintSet.load_or_empty(path, task="grasp_single_bottle")
            state.apply_update(
                {
                    "preserve": ["keep multi-view checks"],
                    "append": [
                        {
                            "id": "insert_before_close",
                            "scope": "task",
                            "text": "Before closing, keep moving until the object is deep between both finger pads.",
                            "rationale": "previous run closed at fingertip contact",
                        }
                    ],
                    "revise": [],
                    "remove": [],
                    "unresolved_conflicts": [],
                },
                source_run="run_a",
            )

            prompt = LearnedConstraintSet.load_or_empty(path, task="grasp_single_bottle").prompt_text()

            self.assertIn("[insert_before_close scope=task]", prompt)
            self.assertIn("deep between both finger pads", prompt)
            self.assertIn("keep multi-view checks", prompt)

    def test_auto_loop_applies_correction_and_reruns_with_constraint_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            calls: list[dict[str, object]] = []

            def fake_eval(*_args, **kwargs) -> Path:
                calls.append(kwargs)
                run_dir = state_dir / "runs" / f"round_{len(calls)}"
                run_dir.mkdir(parents=True)
                success = len(calls) == 2
                (run_dir / "result.json").write_text(json.dumps({"success": success}), encoding="utf-8")
                return run_dir

            def fake_diagnosis(_client, run_dir: Path, output_dir: Path, **_kwargs) -> Path:
                output_dir.mkdir(parents=True)
                report = output_dir / "report.md"
                report.write_text("main failure: object was only at the fingertips", encoding="utf-8")
                return report

            def fake_correction(_client, **kwargs) -> Path:
                output_dir = kwargs["output_dir"]
                output_dir.mkdir(parents=True)
                update = {
                    "preserve": ["preserve expert learning"],
                    "append": [
                        {
                            "id": "full_insert_before_close",
                            "scope": "task",
                            "text": "If the object is only at the fingertips, keep the gripper open and advance until it is deep inside the gap.",
                            "rationale": "diagnosis found fingertip-only contact",
                        }
                    ],
                    "revise": [],
                    "remove": [],
                    "unresolved_conflicts": [],
                }
                path = output_dir / "constraint_update.json"
                path.write_text(json.dumps(update), encoding="utf-8")
                return path

            result = run_robotwin_loop(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                state_dir,
                task="grasp_single_bottle",
                config="demo_clean",
                seed=10,
                active_arm="right",
                max_steps=5,
                max_rounds=2,
                correction="auto",
                display=False,
                stream=False,
                eval_runner=fake_eval,
                diagnosis_runner=fake_diagnosis,
                correction_runner=fake_correction,
            )

            constraint_file = default_learned_constraint_file(state_dir, task="grasp_single_bottle")
            saved = json.loads(constraint_file.read_text(encoding="utf-8"))

            self.assertEqual(result.status, "success")
            self.assertEqual(len(calls), 2)
            self.assertIsNone(calls[0]["learned_constraint_file"])
            self.assertEqual(calls[1]["learned_constraint_file"], constraint_file)
            self.assertEqual(saved["constraints"][0]["id"], "full_insert_before_close")
            self.assertIn("fingertips", saved["constraints"][0]["text"])

    def test_correction_update_is_restricted_to_task_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.json"
            state = LearnedConstraintSet.load_or_empty(path, task="grasp_single_bottle")
            state.constraints.append(
                {
                    "id": "protected_general",
                    "scope": "general",
                    "text": "A general rule that auto correction must not edit.",
                    "status": "active",
                }
            )
            update = {
                "preserve": [],
                "append": [
                    {
                        "id": "maybe_general",
                        "scope": "general",
                        "text": "Close only after side/depth confirms the target is between both pads.",
                        "rationale": "diagnosis found topdown overlap was insufficient",
                    }
                ],
                "revise": [
                    {
                        "id": "protected_general",
                        "scope": "general",
                        "text": "Do not edit this general rule.",
                        "rationale": "attempted general revision",
                    }
                ],
                "remove": [{"id": "protected_general", "rationale": "attempted general removal"}],
                "unresolved_conflicts": [],
            }

            sanitized = task_scoped_update(update, current_constraints=state)

            self.assertEqual(sanitized["append"][0]["scope"], "task")
            self.assertEqual(sanitized["revise"], [])
            self.assertEqual(sanitized["remove"], [])
            self.assertTrue(any("protected_general" in item for item in sanitized["unresolved_conflicts"]))


if __name__ == "__main__":
    unittest.main()
