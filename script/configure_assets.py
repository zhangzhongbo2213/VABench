"""Link an existing RoboTwin assets directory without modifying its contents."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets_dir", type=Path)
    args = parser.parse_args()
    source = args.assets_dir.expanduser().resolve()
    for subdir in ("objects", "embodiments"):
        if not (source / subdir).is_dir():
            parser.error(f"Missing {source / subdir}; supply the assets directory, not the repository")
    destination = ROOT / "assets"
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source:
            print(f"Assets already configured: {source}")
            return
        parser.error(f"{destination} already exists; no changes made")
    destination.symlink_to(source, target_is_directory=True)
    print(f"Assets configured: {destination} -> {source}")


if __name__ == "__main__":
    main()
