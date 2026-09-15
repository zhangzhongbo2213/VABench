from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]


def class_base_names(module_path: Path, class_name: str) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [ast.unparse(base) for base in node.bases]
    raise AssertionError(f"class {class_name!r} not found in {module_path}")


class TaskVisualBaseTests(unittest.TestCase):
    def test_beat_block_hammer_uses_single_right_arm_base(self) -> None:
        bases = class_base_names(
            ROBOTWIN_ROOT / "envs" / "beat_block_hammer_right.py",
            "beat_block_hammer_right",
        )
        self.assertEqual(bases, ["SingleRightArmTask"])

    def test_place_shoe_uses_dual_arm_visual_base(self) -> None:
        bases = class_base_names(
            ROBOTWIN_ROOT / "envs" / "place_shoe.py",
            "place_shoe",
        )
        self.assertEqual(bases, ["DualArmTask"])

    def test_block_handover_uses_dual_arm_visual_base(self) -> None:
        bases = class_base_names(
            ROBOTWIN_ROOT / "envs" / "handover_horizontal_block.py",
            "handover_horizontal_block",
        )
        self.assertEqual(bases, ["DualArmTask"])


if __name__ == "__main__":
    unittest.main()
