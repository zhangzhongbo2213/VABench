"""Apply the two RoboTwin compatibility patches to a fresh environment."""
from pathlib import Path
import mplib
import sapien


def update(path, replacements):
    original = path.read_text()
    result = original
    for before, after in replacements:
        result = result.replace(before, after)
    if result != original:
        backup = path.with_suffix(path.suffix + ".vabench-original")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(result)
        print(f"Patched {path}")


update(Path(sapien.__file__).parent / "wrapper/urdf_loader.py", [
    ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
    ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
])
update(Path(mplib.__file__).parent / "planner.py", [
    ('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
     'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'),
])
