from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import imageio.v2 as imageio


@dataclass(frozen=True)
class ExpertFrame:
    demo_id: str
    seed: int | None
    phase: str
    step: int | None
    view: str
    path: Path
    caption: str
    text_label: bool | None = None
    eepose_overlay: bool | None = None

    @property
    def clean_for_model(self) -> bool:
        return self.text_label is False and self.eepose_overlay is False


@dataclass(frozen=True)
class ExpertTrajectory:
    demo_id: str
    seed: int | None
    rows: list[dict[str, Any]]
    source_type: str = "trajectory"


class ExpertStore:
    def __init__(self, root: Path):
        self.root = root
        self.source_roots = self._resolve_source_roots()
        self._video_cache: Path | None = None
        self.frames, self.trajectories = self._load()

    def __del__(self) -> None:
        if self._video_cache is not None:
            shutil.rmtree(self._video_cache, ignore_errors=True)

    def summary(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "source_roots": [str(path) for path in self.source_roots],
            "composite": self.composite,
            "demos": self.demo_ids,
            "frames": len(self.frames),
            "clean_frames": sum(1 for frame in self.frames if frame.clean_for_model),
            "trajectories": len(self.trajectories),
            "seeds": self.seeds,
            "phases": self.phases,
            "views": self.views,
            "observation_only": self.observation_only,
            "video_only": self.video_only,
        }

    @property
    def composite(self) -> bool:
        return len(self.source_roots) > 1

    @property
    def demo_ids(self) -> list[str]:
        return sorted({trajectory.demo_id for trajectory in self.trajectories})

    @property
    def seeds(self) -> list[int]:
        return sorted({trajectory.seed for trajectory in self.trajectories if trajectory.seed is not None})

    @property
    def phases(self) -> list[str]:
        return sorted({frame.phase for frame in self.frames})

    @property
    def views(self) -> list[str]:
        return sorted({frame.view for frame in self.frames})

    @property
    def observation_only(self) -> bool:
        return bool(self.trajectories) and all(
            all(not row.get("target_eepose") and not row.get("command") for row in trajectory.rows)
            for trajectory in self.trajectories
        )

    @property
    def video_only(self) -> bool:
        return bool(self.trajectories) and all(
            trajectory.source_type == "video_only" for trajectory in self.trajectories
        )

    def find(self, args: dict[str, Any]) -> ExpertFrame:
        demo = str(args.get("demo", "")).lower()
        phase = str(args.get("phase", "")).lower()
        view = str(args.get("view", "")).lower()
        seed_arg = args.get("seed")
        seed = int(seed_arg) if seed_arg not in {None, ""} else None
        step_arg = args.get("step")
        step = int(step_arg) if step_arg not in {None, ""} else None
        candidates = self.frames
        if demo:
            candidates = [
                frame for frame in candidates if demo in frame.demo_id.lower()
            ]
        if seed is not None:
            candidates = [frame for frame in candidates if frame.seed == seed]
        if step is not None:
            candidates = [frame for frame in candidates if frame.step == step]
        if phase:
            candidates = [frame for frame in candidates if phase in frame.phase.lower()]
        if view:
            candidates = [frame for frame in candidates if frame.view.lower() == view]
        if not candidates:
            raise ValueError(f"no expert frame matches {args}")
        return candidates[0]

    def find_many(self, args: dict[str, Any], *, max_frames: int = 8) -> list[ExpertFrame]:
        steps = args.get("steps")
        if not isinstance(steps, list):
            return [self.find(args)]
        if not steps or len(steps) > max_frames:
            raise ValueError(f"steps must contain 1 to {max_frames} frame indices")
        result = []
        for step in steps:
            frame_args = dict(args)
            frame_args.pop("steps", None)
            frame_args["step"] = int(step)
            frame = self.find(frame_args)
            if frame not in result:
                result.append(frame)
        return result

    def trajectory_text(self, args: dict[str, Any]) -> str:
        demo = str(args.get("demo", "")).lower()
        seed_arg = args.get("seed")
        seed = int(seed_arg) if seed_arg not in {None, ""} else None
        candidates = self.trajectories
        if demo:
            candidates = [
                trajectory
                for trajectory in candidates
                if demo in trajectory.demo_id.lower()
            ]
        if seed is not None:
            candidates = [trajectory for trajectory in candidates if trajectory.seed == seed]
        if not candidates:
            raise ValueError(f"no expert trajectory matches {args}")
        max_rows = int(args.get("max_rows") or 20)
        lines = []
        for trajectory in candidates:
            lines.append(
                f"Expert trajectory {trajectory.demo_id} seed={trajectory.seed} source={trajectory.source_type}"
            )
            if self.composite and trajectory.source_type == "video_only":
                steps = [
                    int(row["step"])
                    for row in trajectory.rows
                    if row.get("step") is not None
                ]
                image_views = sorted(
                    {
                        str(view)
                        for row in trajectory.rows
                        for view in (row.get("images") or {}).keys()
                    }
                )
                lines.append(
                    "available_steps={start}..{end} frame_count={count} image_views={views}".format(
                        start=min(steps) if steps else None,
                        end=max(steps) if steps else None,
                        count=len(steps),
                        views=image_views,
                    )
                )
                continue
            observation_only = all(not row.get("target_eepose") and not row.get("command") for row in trajectory.rows)
            selected_rows = trajectory.rows if observation_only else trajectory.rows[:max_rows]
            for row in selected_rows:
                target = row.get("target_eepose", {})
                state = row.get("state", {})
                if not target and not row.get("command"):
                    image_views = sorted((row.get("images") or {}).keys())
                    if trajectory.source_type == "video_only":
                        lines.append(
                            f"step={row.get('step')} source=video_only_rgb image_views={image_views}"
                        )
                        continue
                    lines.append(
                        "step={step} observed_ee={observed_ee} observed_rpy={observed_rpy} "
                        "gripper={gripper} image_views={image_views}".format(
                            step=row.get("step"),
                            observed_ee=state.get("eepose_xyz"),
                            observed_rpy=state.get("eepose_rpy_deg"),
                            gripper=state.get("gripper_value"),
                            image_views=image_views,
                        )
                    )
                    continue
                lines.append(
                    "step={step} phase={phase} executed={executed} command={command} "
                    "target_xyz={target_xyz} target_rpy={target_rpy} "
                    "observed_ee={observed_ee} gripper={gripper}".format(
                        step=row.get("step"),
                        phase=row.get("phase"),
                        executed=row.get("executed"),
                        command=row.get("command"),
                        target_xyz=target.get("xyz"),
                        target_rpy=target.get("rpy_deg"),
                        observed_ee=state.get("eepose_xyz"),
                        gripper=state.get("gripper_value"),
                    )
                )
            if not observation_only and len(trajectory.rows) > max_rows:
                lines.append(f"... {len(trajectory.rows) - max_rows} additional rows omitted")
        return "\n".join(lines)

    def _load(self) -> tuple[list[ExpertFrame], list[ExpertTrajectory]]:
        frames: list[ExpertFrame] = []
        trajectories: list[ExpertTrajectory] = []
        if not self.source_roots:
            return frames, trajectories
        for source_root in self.source_roots:
            for manifest in sorted(source_root.rglob("expert_demo.json")):
                data = json.loads(manifest.read_text(encoding="utf-8"))
                demo_id = self._demo_id(source_root, manifest.parent.name)
                seed = data.get("seed")
                image_style = data.get("image_style", {})
                text_label = image_style.get("text_label") if isinstance(image_style, dict) else None
                eepose_overlay = image_style.get("eepose_overlay") if isinstance(image_style, dict) else None
                trajectory_path = manifest.parent / str(data.get("trajectory", "eepose_trajectory.jsonl"))
                rows: list[dict[str, Any]] = []
                if trajectory_path.exists():
                    rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    trajectories.append(ExpertTrajectory(demo_id=demo_id, seed=int(seed) if seed is not None else None, rows=rows))
                phases = data.get("phases", [])
                if not phases:
                    for row in rows:
                        images = row.get("images", {})
                        if not isinstance(images, dict):
                            continue
                        state = row.get("state", {})
                        step = row.get("step")
                        for view, rel_path in images.items():
                            path = manifest.parent / str(rel_path)
                            if not path.exists():
                                continue
                            frames.append(
                                ExpertFrame(
                                    demo_id=demo_id,
                                    seed=int(seed) if seed is not None else None,
                                    phase="observation",
                                    step=int(step) if step is not None else None,
                                    view=str(view),
                                    path=path,
                                    caption=(
                                        f"Expert {demo_id} observation step={step} view={view}. "
                                        f"observed_eepose_xyz={state.get('eepose_xyz')} "
                                        f"observed_rpy={state.get('eepose_rpy_deg')} "
                                        f"gripper={state.get('gripper_value')}."
                                    ),
                                    text_label=bool(text_label) if text_label is not None else None,
                                    eepose_overlay=bool(eepose_overlay) if eepose_overlay is not None else None,
                                )
                            )
                for phase in phases:
                    if not isinstance(phase, dict):
                        continue
                    images = phase.get("images", {})
                    if not isinstance(images, dict):
                        continue
                    for view, rel_path in images.items():
                        path = manifest.parent / str(rel_path)
                        if not path.exists():
                            continue
                        phase_name = str(phase.get("phase", "unknown"))
                        note = str(phase.get("note", ""))
                        step = phase.get("step")
                        target = phase.get("target_eepose", {})
                        state = phase.get("state", {})
                        geometry_note = ""
                        if isinstance(target, dict):
                            geometry_note += f" target_eepose_xyz={target.get('xyz')} target_rpy={target.get('rpy_deg')}."
                        if isinstance(state, dict):
                            geometry_note += f" observed_eepose_xyz={state.get('eepose_xyz')} gripper={state.get('gripper_value')}."
                        frames.append(
                            ExpertFrame(
                                demo_id=demo_id,
                                seed=int(seed) if seed is not None else None,
                                phase=phase_name,
                                step=int(step) if step is not None else None,
                                view=str(view),
                                path=path,
                                caption=f"Expert {demo_id} phase={phase_name} view={view}. {note}{geometry_note}".strip(),
                                text_label=bool(text_label) if text_label is not None else None,
                                eepose_overlay=bool(eepose_overlay) if eepose_overlay is not None else None,
                            )
                        )
        self._load_video_only(frames, trajectories)
        return frames, trajectories

    def _load_video_only(
        self,
        frames: list[ExpertFrame],
        trajectories: list[ExpertTrajectory],
    ) -> None:
        metadata_paths = sorted(
            metadata_path
            for source_root in self.source_roots
            for metadata_path in source_root.rglob("metadata.json")
        )
        if not metadata_paths:
            return
        self._video_cache = Path(tempfile.mkdtemp(prefix="agent_expert_video_"))
        cache_root = self._video_cache
        for metadata_path in metadata_paths:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if data.get("stored_eepose") is not False or not data.get("video"):
                continue
            video_path = metadata_path.parent / str(data["video"])
            if not video_path.is_file():
                continue
            source_root = next(
                root
                for root in self.source_roots
                if metadata_path == root or root in metadata_path.parents
            )
            demo_id = self._demo_id(source_root, metadata_path.parent.name)
            seed = int(data["seed"]) if data.get("seed") is not None else None
            view = str(data.get("view") or "video")
            decoded = decode_video_frames(video_path, cache_root / demo_id)
            expected = data.get("frame_count")
            if expected is not None and len(decoded) != int(expected):
                raise ValueError(
                    f"video frame count mismatch for {video_path}: expected {expected}, decoded {len(decoded)}"
                )
            rows = []
            for step, path in enumerate(decoded):
                rows.append({"step": step, "images": {view: f"video_frame:{step}"}, "state": {}})
                frames.append(
                    ExpertFrame(
                        demo_id=demo_id,
                        seed=seed,
                        phase="video_observation",
                        step=step,
                        view=view,
                        path=path,
                        caption=(
                            f"Expert {demo_id} video-only RGB observation frame={step}/{len(decoded) - 1} "
                            f"view={view}. No eepose, qpos, action, object coordinate, or phase label is provided."
                        ),
                        text_label=False,
                        eepose_overlay=False,
                    )
                )
            trajectories.append(
                ExpertTrajectory(
                    demo_id=demo_id,
                    seed=seed,
                    rows=rows,
                    source_type="video_only",
                )
            )

    def _resolve_source_roots(self) -> list[Path]:
        if not self.root.exists():
            return []
        manifest = self.root / "composite_sources.json"
        if not manifest.is_file():
            return [self.root]
        data = json.loads(manifest.read_text(encoding="utf-8"))
        raw_sources = data.get("sources") if isinstance(data, dict) else None
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"composite expert manifest has no sources: {manifest}")
        roots = []
        for item in raw_sources:
            raw_path = item.get("path") if isinstance(item, dict) else item
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"invalid composite expert source in {manifest}: {item!r}")
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"composite expert source not found: {path}")
            if path not in roots:
                roots.append(path)
        return roots

    def _demo_id(self, source_root: Path, local_id: str) -> str:
        if not self.composite:
            return local_id
        if local_id == source_root.name:
            return local_id
        return f"{source_root.name}/{local_id}"


def decode_video_frames(video_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    paths = []
    try:
        for index, frame in enumerate(reader):
            path = output_dir / f"{index:05d}.png"
            imageio.imwrite(path, frame)
            paths.append(path)
    finally:
        reader.close()
    return paths
