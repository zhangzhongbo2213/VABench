from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


FORMAT = "robotwin_benchmark_video_output_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one structured action/reason sidecar next to every video in a "
            "RoboTwin benchmark outputs manifest."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="SOURCE_ID=ROOT",
        help=(
            "Resolve source run paths for one manifest source. ROOT=/ uses the "
            "recorded absolute path; a cache root is prepended otherwise."
        ),
    )
    parser.add_argument(
        "--require-source-metadata",
        action="store_true",
        help="Fail after export when neither events.jsonl nor result.json is available.",
    )
    parser.add_argument(
        "--include-unlisted-local-videos",
        action="store_true",
        help=(
            "Recover MP4 files present under the output hierarchy but absent from "
            "the manifest by matching their hardlink inode to a local source run."
        ),
    )
    parser.add_argument(
        "--local-runs-root",
        help="Local runs root used by --include-unlisted-local-videos.",
    )
    return parser.parse_args()


def parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source_id, separator, root = value.partition("=")
        if not separator or not source_id or not root:
            raise ValueError(f"invalid --source-root {value!r}")
        roots[source_id] = Path(root).resolve()
    return roots


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return events, malformed
    with stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(event, dict):
                events.append(event)
    return events, malformed


def resolve_run_dir(item: dict[str, Any], roots: dict[str, Path]) -> Path | None:
    source_id = item.get("source_id")
    recorded = item.get("source_run_dir")
    if not isinstance(source_id, str) or not isinstance(recorded, str):
        return None
    root = roots.get(source_id)
    if root is None:
        return None
    if root == Path("/"):
        return Path(recorded)
    return root / recorded.lstrip("/")


def action_rows(
    events: list[dict[str, Any]], result: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    actions: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "action":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("action"), str):
            continue
        actions.append(
            {
                "index": len(actions),
                "step": data.get("step"),
                "action": data["action"],
                "reason": data.get("reason"),
                "action_payload": data.get("action_payload"),
                "created_at": event.get("created_at"),
            }
        )
    if actions:
        return actions, "events.jsonl"

    steps = result.get("steps")
    if not isinstance(steps, list):
        return [], "none"
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("action"), str):
            continue
        actions.append(
            {
                "index": len(actions),
                "step": step.get("step"),
                "action": step["action"],
                "reason": None,
                "action_payload": None,
                "created_at": None,
            }
        )
    return actions, "result.json:steps" if actions else "none"


def terminal_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if event.get("type") not in {
            "eval_error",
            "eval_stopped",
            "eval_finish",
        }:
            continue
        rows.append(
            {
                "type": event.get("type"),
                "data": event.get("data"),
                "created_at": event.get("created_at"),
            }
        )
    return rows


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "task",
        "seed",
        "run_id",
        "success",
        "stop_reason",
        "last_info",
        "session",
        "evaluation_model",
        "summary_model",
        "experiment_mode",
        "provider_routing",
        "max_tokens",
        "request_image_transport",
        "context_compaction",
    )
    return {field: result.get(field) for field in fields if field in result}


def sidecar_relative(item: dict[str, Any]) -> Path:
    relative = item.get("destination_relative")
    if not isinstance(relative, str):
        raise ValueError("manifest video is missing destination_relative")
    return Path(relative).with_suffix(".json")


def recover_unlisted_local_videos(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    runs_root: Path,
) -> int:
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("manifest does not contain a videos list")
    listed = {
        item.get("destination_relative") for item in videos if isinstance(item, dict)
    }
    unlisted = [
        path
        for arm_group in ("single_arm", "dual_arm")
        for path in (output_dir / arm_group).glob("*/*/*/seed_*_*.mp4")
        if str(path.relative_to(output_dir)) not in listed
    ]
    recoverable_items = {
        str(item.get("destination_relative")): item
        for item in videos
        if isinstance(item, dict)
        and item.get("selection_source") == "recovered_unlisted_hardlink"
    }
    targets = {str(path.relative_to(output_dir)): path for path in unlisted}
    targets.update(
        {
            relative: output_dir / relative
            for relative in recoverable_items
            if (output_dir / relative).is_file()
        }
    )
    if not targets:
        return 0

    by_inode: dict[tuple[int, int], Path] = {}
    wanted = {(path.stat().st_dev, path.stat().st_ino) for path in targets.values()}
    for source in runs_root.rglob("model_replay.mp4"):
        key = (source.stat().st_dev, source.stat().st_ino)
        if key not in wanted:
            continue
        current = by_inode.get(key)
        source_quality = metadata_quality(source.parent)
        if current is None or source_quality > metadata_quality(current.parent):
            by_inode[key] = source

    for relative_text, destination in sorted(targets.items()):
        relative = destination.relative_to(output_dir)
        arm_group, task, split, output_model, filename = relative.parts
        stem = destination.stem
        prefix, outcome = stem.rsplit("_", 1)
        if not prefix.startswith("seed_") or outcome not in {"SUCCESS", "FAILED"}:
            raise ValueError(f"unsupported unlisted output name: {relative}")
        seed = int(prefix.removeprefix("seed_"))
        key = (destination.stat().st_dev, destination.stat().st_ino)
        source = by_inode.get(key)
        if source is None:
            raise FileNotFoundError(
                f"cannot recover source run for unlisted output: {relative}"
            )
        run_dir = source.parent
        result = read_json(run_dir / "result.json")
        source_success = result.get("success")
        success = outcome == "SUCCESS"
        if isinstance(source_success, bool) and source_success != success:
            raise ValueError(f"source result outcome mismatch for {relative}")
        recovered_item = {
            "seed": seed,
            "success": success,
            "selection_source": "recovered_unlisted_hardlink",
            "manual_override": False,
            "source_id": "local",
            "source_run_dir": str(run_dir),
            "source_video": str(source),
            "source_frames_dir": str(run_dir / "model_frames"),
            "source_frame_count": len(list((run_dir / "model_frames").glob("*.png"))),
            "batch_id": str(result.get("run_id") or run_dir.name).removeprefix("eval_"),
            "evaluation_model": result.get("evaluation_model") or output_model,
            "summary_model": result.get("summary_model") or output_model,
            "max_steps": (result.get("last_info") or {}).get("max_steps"),
            "task": task,
            "query_task": f"{task}_generalization"
            if split == "generalization"
            else task,
            "arm_group": arm_group,
            "split": split,
            "csv_column": None,
            "output_model": output_model,
            "destination_relative": str(relative),
            "destination": str(destination),
            "bytes": destination.stat().st_size,
            "materialization_source": "existing_destination_recovered",
        }
        existing = recoverable_items.get(relative_text)
        if existing is None:
            videos.append(recovered_item)
        else:
            existing.update(recovered_item)
    manifest["planned_video_count"] = len(videos)
    manifest["materialized_video_count"] = len(videos)
    return len(unlisted)


def metadata_quality(run_dir: Path) -> tuple[int, int]:
    return (
        int((run_dir / "events.jsonl").is_file())
        + int((run_dir / "result.json").is_file()),
        -int("merged_eval_results" in run_dir.parts),
    )


def build_sidecar(
    item: dict[str, Any],
    *,
    run_dir: Path | None,
    archived_stem: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    events_path = run_dir / "events.jsonl" if run_dir is not None else None
    result_path = run_dir / "result.json" if run_dir is not None else None
    if archived_stem is not None:
        archived_events = archived_stem.with_suffix(".events.jsonl")
        archived_result = archived_stem.with_suffix(".result.json")
        if events_path is None or not events_path.is_file():
            events_path = archived_events if archived_events.is_file() else events_path
        if result_path is None or not result_path.is_file():
            result_path = archived_result if archived_result.is_file() else result_path
    events, malformed_events = (
        read_events(events_path)
        if events_path is not None and events_path.is_file()
        else ([], 0)
    )
    result = (
        read_json(result_path)
        if result_path is not None and result_path.is_file()
        else {}
    )
    actions, action_source = action_rows(events, result)
    final_action = actions[-1] if actions else None
    source_metadata_available = bool(events or result)
    sidecar = {
        "format": FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": item.get("destination_relative"),
            "task": item.get("task"),
            "query_task": item.get("query_task"),
            "split": item.get("split"),
            "model": item.get("output_model"),
            "seed": item.get("seed"),
            "outcome": "SUCCESS" if item.get("success") is True else "FAILED",
            "manual_override": bool(item.get("manual_override")),
        },
        "source": {
            "source_id": item.get("source_id"),
            "recorded_run_dir": item.get("source_run_dir"),
            "resolved_run_dir": str(run_dir) if run_dir is not None else None,
            "batch_id": item.get("batch_id"),
            "events_file": str(events_path) if events_path is not None else None,
            "result_file": str(result_path) if result_path is not None else None,
            "metadata_available": source_metadata_available,
            "malformed_event_lines": malformed_events,
        },
        "result": result_summary(result),
        "action_source": action_source,
        "action_count": len(actions),
        "actions": actions,
        "final_action": final_action,
        "final_reason": final_action.get("reason")
        if final_action is not None
        else None,
        "termination_events": terminal_events(events),
    }
    report = {
        "metadata_available": source_metadata_available,
        "events_available": bool(events),
        "result_available": bool(result),
        "action_count": len(actions),
        "reason_count": sum(
            isinstance(action.get("reason"), str) for action in actions
        ),
        "final_action": final_action.get("action") if final_action else None,
        "final_reason": final_action.get("reason") if final_action else None,
        "action_source": action_source,
    }
    return sidecar, report


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def update_manifest_csv(path: Path, videos: list[dict[str, Any]]) -> None:
    existing_fields: list[str] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            existing_fields = list((csv.DictReader(stream).fieldnames or []))
    added_fields = [
        "destination_output_relative",
        "output_metadata_available",
        "output_action_count",
        "output_reason_count",
        "output_final_action",
    ]
    fields = existing_fields or [
        "arm_group",
        "task",
        "split",
        "output_model",
        "seed",
        "success",
        "manual_override",
        "source_id",
        "source_run_dir",
        "source_video",
        "destination_relative",
    ]
    fields.extend(field for field in added_fields if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(videos)


def export_manifest(
    manifest_path: Path,
    *,
    roots: dict[str, Path],
    include_unlisted_local_videos: bool = False,
    local_runs_root: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("manifest does not contain a videos list")
    output_dir = manifest_path.parent
    recovered_count = 0
    if include_unlisted_local_videos:
        if local_runs_root is None:
            raise ValueError(
                "--local-runs-root is required with --include-unlisted-local-videos"
            )
        recovered_count = recover_unlisted_local_videos(
            manifest,
            output_dir=output_dir,
            runs_root=local_runs_root,
        )
    missing: list[dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict):
            continue
        relative = sidecar_relative(item)
        run_dir = resolve_run_dir(item, roots)
        archived_stem = (output_dir / item["destination_relative"]).with_suffix("")
        sidecar, report = build_sidecar(
            item,
            run_dir=run_dir,
            archived_stem=archived_stem,
        )
        destination = output_dir / relative
        write_json(destination, sidecar)
        item.update(
            {
                "destination_output_relative": str(relative),
                "destination_output": str(destination),
                "output_metadata_available": report["metadata_available"],
                "output_events_available": report["events_available"],
                "output_result_available": report["result_available"],
                "output_action_count": report["action_count"],
                "output_reason_count": report["reason_count"],
                "output_final_action": report["final_action"],
                "output_final_reason": report["final_reason"],
                "output_action_source": report["action_source"],
            }
        )
        if not report["metadata_available"]:
            missing.append(
                {
                    "destination_relative": item.get("destination_relative"),
                    "source_id": item.get("source_id"),
                    "source_run_dir": item.get("source_run_dir"),
                }
            )

    manifest.update(
        {
            "output_sidecar_format": FORMAT,
            "materialized_output_count": len(videos),
            "output_metadata_available_count": sum(
                bool(item.get("output_metadata_available")) for item in videos
            ),
            "output_reason_complete_count": sum(
                int(item.get("output_action_count") or 0)
                == int(item.get("output_reason_count") or 0)
                for item in videos
                if int(item.get("output_action_count") or 0) > 0
            ),
            "missing_output_metadata_count": len(missing),
            "missing_output_metadata": missing,
            "recovered_unlisted_video_count": int(
                manifest.get("recovered_unlisted_video_count") or 0
            )
            + recovered_count,
        }
    )
    write_json(manifest_path, manifest)
    update_manifest_csv(output_dir / "manifest.csv", videos)
    return {
        "video_count": len(videos),
        "sidecar_count": len(videos),
        "metadata_available_count": len(videos) - len(missing),
        "missing_metadata_count": len(missing),
        "action_count": sum(
            int(item.get("output_action_count") or 0) for item in videos
        ),
        "reason_count": sum(
            int(item.get("output_reason_count") or 0) for item in videos
        ),
        "recovered_unlisted_video_count": recovered_count,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    summary = export_manifest(
        manifest_path,
        roots=parse_source_roots(args.source_root),
        include_unlisted_local_videos=args.include_unlisted_local_videos,
        local_runs_root=(
            Path(args.local_runs_root).resolve() if args.local_runs_root else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if args.require_source_metadata and summary["missing_metadata_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
