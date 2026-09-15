from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from agent.events import RunLogger
from agent.robotwin.eval import (
    default_expert_demo_dir,
    default_robotwin_run_id,
    robotwin_run_parent,
    validate_run_id,
)


class RunLayoutTest(unittest.TestCase):
    def test_default_run_id_uses_task_seed_test_number_and_short_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            runs = state / "runs"
            runs.mkdir()
            (runs / "data").mkdir()
            seed10 = runs / "grasp_single_bottle" / "seed_10"
            seed11 = runs / "grasp_single_bottle" / "seed_11"
            seed10.mkdir(parents=True)
            seed11.mkdir(parents=True)
            (seed10 / "test001_aaaaaaaa").mkdir()
            (seed10 / "test002_bbbbbbbb").mkdir()
            (seed11 / "test001_cccccccc").mkdir()

            run_id = default_robotwin_run_id(state, task="grasp_single_bottle", seed=10)

            self.assertRegex(run_id, r"^test003_[0-9a-f]{8}$")

    def test_default_run_id_sanitizes_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id = default_robotwin_run_id(Path(tmp), task="task name/with spaces", seed=2)

            self.assertTrue(re.match(r"^test001_[0-9a-f]{8}$", run_id))
            self.assertEqual(
                robotwin_run_parent(task="task name/with spaces", seed=2),
                Path("task_name_with_spaces") / "seed_2",
            )

    def test_run_logger_writes_under_task_and_seed_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            logger = RunLogger(
                state,
                "test001_aaaaaaaa",
                relative_parent=robotwin_run_parent(task="grasp_single_cube", seed=10),
                display=False,
            )

            self.assertEqual(
                logger.dir,
                state / "runs" / "grasp_single_cube" / "seed_10" / "test001_aaaaaaaa",
            )
            self.assertTrue(logger.dir.is_dir())

    def test_explicit_run_id_must_be_a_leaf_name(self) -> None:
        self.assertEqual(validate_run_id("custom_test"), "custom_test")
        for value in ("", "../escape", "nested/run", "/absolute"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_run_id(value)

    def test_default_expert_demo_dir_uses_runs_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            data = state / "runs" / "data" / "expert_demos"
            horizontal = data / "grasp_single_bottle_right_eepose_clean"
            upright = data / "grasp_single_bottle_upright_right_eepose_clean"
            horizontal.mkdir(parents=True)
            upright.mkdir(parents=True)

            self.assertEqual(
                default_expert_demo_dir(state, task="grasp_single_bottle", active_arm="right"),
                horizontal,
            )
            self.assertEqual(
                default_expert_demo_dir(state, task="grasp_single_bottle_upright", active_arm="right"),
                upright,
            )

    def test_default_expert_demo_dir_does_not_leak_related_task_demos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            original = state / "runs" / "data" / "expert_demos" / "grasp_single_bottle_right_eepose_clean"
            original.mkdir(parents=True)

            self.assertIsNone(
                default_expert_demo_dir(state, task="grasp_single_bottle_114", active_arm="right")
            )

    def test_default_expert_demo_dir_uses_both_for_dual_arm_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            demo = state / "runs" / "data" / "expert_demos" / "lift_pot_both_eepose_clean"
            demo.mkdir(parents=True)

            self.assertEqual(default_expert_demo_dir(state, task="lift_pot", active_arm=None), demo)

    def test_cube_handover_defaults_to_dual_arm_expert_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            demo = (
                state
                / "runs"
                / "data"
                / "expert_demos"
                / "handover_cube_to_target_both_eepose_clean"
            )
            demo.mkdir(parents=True)

            self.assertEqual(
                default_expert_demo_dir(
                    state,
                    task="handover_cube_to_target",
                    active_arm=None,
                ),
                demo,
            )


if __name__ == "__main__":
    unittest.main()
