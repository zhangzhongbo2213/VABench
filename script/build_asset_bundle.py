"""Build the main-task asset archive from an existing RoboTwin assets directory."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.assets_dir.expanduser().resolve()
    spec = json.loads((ROOT / "configs/main_assets.json").read_text())
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / spec["archive"]
    if archive.exists():
        parser.error(f"Archive already exists: {archive}")

    selected = set()

    def include(path):
        path = path.resolve()
        path.relative_to(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)

    for relative in spec["shared_files"]:
        include(source / relative)
    robot = source / "embodiments" / spec["embodiment"]
    for relative in spec["robot_files"]:
        include(robot / relative)
    urdf = robot / spec["robot_urdf"]
    for mesh in ET.parse(urdf).iter("mesh"):
        path = urdf.parent / mesh.attrib["filename"]
        include(path)
        for texture in path.parent.iterdir():
            if texture.suffix.lower() in (".png", ".jpg", ".jpeg"):
                include(texture)
    for name, ids in spec["objects"].items():
        directory = source / "objects" / name
        for model_id in ids:
            include(directory / f"model_data{model_id}.json")
            for kind in ("collision", "visual"):
                # These main-task object models use self-contained GLB files.
                include(directory / kind / f"base{model_id}.glb")
    for name, models in spec["articulated_objects"].items():
        for model in models:
            directory = source / "objects" / name / model
            if not (directory / "mobility.urdf").is_file():
                raise FileNotFoundError(directory / "mobility.urdf")
            for path in directory.rglob("*"):
                if path.is_file():
                    include(path)

    with tempfile.TemporaryDirectory(prefix="va-bench-assets-") as temporary:
        assets = Path(temporary) / "assets"
        records = {}
        for path in sorted(selected):
            relative = path.relative_to(source)
            destination = assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            original_digest = sha256(path)
            if relative.name in ("curobo_left.yml", "curobo_right.yml"):
                text = destination.read_text()
                text, count = re.subn(
                    r"(?m)^([ \t]*(?:urdf_path|collision_spheres):[ \t]*)[^\n]*?assets/([^\n\"']+?)[\"']?[ \t]*$",
                    r"\1assets/\2",
                    text,
                )
                if count != 2:
                    raise ValueError(f"Expected two planner asset paths in {relative}")
                destination.write_text(text)
            record = {"size": destination.stat().st_size, "sha256": sha256(destination)}
            if record["sha256"] != original_digest:
                record["source_sha256"] = original_digest
                record["change"] = "Replace machine-specific planner paths with assets-relative paths."
            records[str(relative)] = record

        shutil.copyfile(ROOT / "LICENSE", assets / "LICENSE")
        (assets / "NOTICE.md").write_text(
            "# VA-Bench main-task assets\n\n"
            "Selected from the RoboTwin assets used to validate this VA-Bench release.\n"
            "Upstream code: https://github.com/RoboTwin-Platform/RoboTwin\n"
            "Upstream dataset: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0\n"
            "The upstream dataset declares the MIT license; the RoboTwin license is included.\n"
            "Original model metadata and attribution in the selected files are retained.\n\n"
            "Scope: 14 main tasks using vabench_eval or demo_clean, including every\n"
            "object variant those tasks can select. Background randomization and clutter\n"
            "require the full upstream assets. Both shared object indexes are retained\n"
            "because the environment imports them even when clutter is disabled.\n\n"
            "Geometry and textures are unchanged. Only the two cuRobo configuration\n"
            "files have their machine-specific asset paths made relative.\n"
        )
        for name in ("LICENSE", "NOTICE.md"):
            path = assets / name
            records[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "format_version": 1,
            "selection": spec,
            "tasks": [t["task"] for t in json.loads((ROOT / "configs/main_tasks.json").read_text())["tasks"]],
            "file_count": len(records),
            "total_bytes": sum(item["size"] for item in records.values()),
            "files": dict(sorted(records.items())),
        }
        (assets / "ASSET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with archive.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    for path in sorted(assets.rglob("*")):
                        if not path.is_file():
                            continue
                        info = tar.gettarinfo(str(path), arcname=str(path.relative_to(assets.parent)))
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        info.mode = 0o644
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
        shutil.copyfile(assets / "ASSET_MANIFEST.json", output / "ASSET_MANIFEST.json")
    checksum = sha256(archive)
    (output / (archive.name + ".sha256")).write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": checksum, "asset_files": len(records) + 1,
                      "uncompressed_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
