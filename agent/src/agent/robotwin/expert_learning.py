from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


REQUIRED_LEARNING_FIELDS = (
    "grasp_object_part",
    "grasp_region",
    "grasp_height_or_depth",
    "finger_placement",
    "approach_strategy",
    "posture_or_rotation",
    "pre_close_checks",
    "test_lift_rule",
)

SUMMARY_MODEL_FIELD = "summary_model"
SHARED_SUMMARY_MODE = "shared-summary"
MODEL_SPECIFIC_SUMMARY_MODE = "model-specific-summary"
EXPERIMENT_MODES = (SHARED_SUMMARY_MODE, MODEL_SPECIFIC_SUMMARY_MODE)
AFFORDANCE_PROFILE_FIELDS = (
    "part_shape",
    "symmetry_class",
    "centering_tolerance",
    "vertical_tolerance",
    "avoid_ends",
    "requires_bilateral_contact",
    "requires_dual_arm",
)
AFFORDANCE_PROFILE_ENUMS = {
    "part_shape": {
        "unknown",
        "slender_cylinder",
        "broad_cylinder",
        "box",
        "handle",
        "flat",
        "irregular",
    },
    "symmetry_class": {
        "unknown",
        "continuous",
        "two_fold",
        "four_fold",
        "asymmetric",
    },
    "centering_tolerance": {"unknown", "strict", "moderate", "permissive"},
    "vertical_tolerance": {"unknown", "strict", "moderate", "permissive"},
}


@dataclass
class ExpertLearningState:
    path: Path
    trajectory_retrieved: bool = False
    retrieved_frames: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None

    @property
    def learned(self) -> bool:
        return self.summary is not None

    @property
    def summary_model(self) -> str | None:
        if not self.summary:
            return None
        return validate_summary_model(self.summary.get(SUMMARY_MODEL_FIELD))

    def save(self, summary: dict[str, Any], *, summary_model: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        saved_summary = dict(summary)
        existing_model = validate_summary_model(saved_summary.get(SUMMARY_MODEL_FIELD))
        if summary_model is not None:
            resolved_model = validate_summary_model(summary_model)
            if existing_model is not None and existing_model != resolved_model:
                raise ValueError(
                    "expert learning summary_model conflicts with the configured model: "
                    f"{existing_model!r} != {resolved_model!r}"
                )
            saved_summary[SUMMARY_MODEL_FIELD] = resolved_model
        self.summary = saved_summary
        self.path.write_text(json.dumps(self.summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_frame_retrieved(
        self,
        *,
        demo: str | None = None,
        seed: int | None,
        step: int | None,
        view: str,
    ) -> None:
        record = {"demo": demo, "seed": seed, "step": step, "view": view}
        if record not in self.retrieved_frames:
            self.retrieved_frames.append(record)

    @property
    def retrieved_frame_views(self) -> set[str]:
        return {str(item.get("view", "")) for item in self.retrieved_frames if item.get("view")}

    @property
    def retrieved_frame_counts_by_demo(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.retrieved_frames:
            demo = item.get("demo")
            if not isinstance(demo, str) or not demo:
                continue
            counts[demo] = counts.get(demo, 0) + 1
        return counts

    def load_from(self, source: Path) -> None:
        summary = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("expert learning file must contain a JSON object")
        error = validate_expert_learning_args(summary, trajectory_retrieved=True)
        if error:
            raise ValueError(error)
        self.trajectory_retrieved = True
        self.save(summary)

    def prompt_text(self) -> str:
        if not self.summary:
            return ""
        return (
            "\nSaved expert-learning constraints for this run. Preserve these as additive task experience unless later visual evidence proves a specific item wrong:\n"
            + json.dumps(
                {key: value for key, value in self.summary.items() if key != SUMMARY_MODEL_FIELD},
                indent=2,
                ensure_ascii=False,
            )
        )


def read_summary_model(path: Path) -> str | None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("expert learning file must contain a JSON object")
    return validate_summary_model(summary.get(SUMMARY_MODEL_FIELD))


def validate_summary_model(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expert learning summary_model must be a non-empty string")
    return value.strip()


def validate_experiment_lineage(
    experiment_mode: str,
    *,
    summary_model: str | None,
    evaluation_model: str,
) -> None:
    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"unsupported summary experiment mode: {experiment_mode!r}")
    if summary_model is None:
        raise ValueError(
            'expert_learning.json must contain a non-empty "summary_model" field'
        )
    if (
        experiment_mode == MODEL_SPECIFIC_SUMMARY_MODE
        and summary_model != evaluation_model
    ):
        raise ValueError(
            "model-specific-summary must be learned by the evaluation model: "
            f"{summary_model!r} != {evaluation_model!r}"
        )


def validate_expert_learning_args(
    args: dict[str, Any],
    *,
    trajectory_retrieved: bool,
    observation_only: bool = False,
    retrieved_frame_views: int = 0,
    video_only: bool = False,
    retrieved_frame_count: int = 0,
    required_video_demos: tuple[str, ...] = (),
    retrieved_frame_counts_by_demo: dict[str, int] | None = None,
    minimum_frames_per_video: int = 3,
    forbid_episode_inventory_counts: bool = False,
) -> str | None:
    if not trajectory_retrieved:
        return "expert.learn is blocked until expert.retrieve has returned an expert trajectory."
    if video_only and retrieved_frame_count < 3:
        return "expert.learn for video-only data is blocked until at least 3 distinct video frames have been inspected."
    if video_only and required_video_demos:
        counts = retrieved_frame_counts_by_demo or {}
        incomplete = [
            demo
            for demo in required_video_demos
            if counts.get(demo, 0) < minimum_frames_per_video
        ]
        if incomplete:
            return (
                "expert.learn for a composite video set is blocked until at least "
                f"{minimum_frames_per_video} distinct frames from every atomic demo have been inspected. "
                "Incomplete demos: "
                + ", ".join(incomplete)
            )
    missing = [field for field in REQUIRED_LEARNING_FIELDS if is_empty(args.get(field))]
    if missing:
        return "expert.learn is missing required fields: " + ", ".join(missing)
    checks = args.get("pre_close_checks")
    if not isinstance(checks, list) or not all(isinstance(item, str) and item.strip() for item in checks):
        return "expert.learn pre_close_checks must be a non-empty list of strings."
    profile = args.get("affordance_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            return "expert.learn affordance_profile must be an object."
        missing_profile = [
            field for field in AFFORDANCE_PROFILE_FIELDS if field not in profile
        ]
        if missing_profile:
            return (
                "expert.learn affordance_profile is missing fields: "
                + ", ".join(missing_profile)
            )
        for field, allowed in AFFORDANCE_PROFILE_ENUMS.items():
            if profile.get(field) not in allowed:
                return (
                    f"expert.learn affordance_profile {field} must be one of: "
                    + ", ".join(sorted(allowed))
                )
        for field in (
            "avoid_ends",
            "requires_bilateral_contact",
            "requires_dual_arm",
        ):
            if not isinstance(profile.get(field), bool):
                return f"expert.learn affordance_profile {field} must be boolean."
    for field in ("task_stage", "post_grasp_goal"):
        value = args.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return f"expert.learn {field} must be a non-empty string when provided."
    functional_constraints = args.get("functional_constraints")
    if functional_constraints is not None and (
        not isinstance(functional_constraints, list)
        or not all(
            isinstance(item, str) and item.strip()
            for item in functional_constraints
        )
    ):
        return "expert.learn functional_constraints must be a list of non-empty strings."
    required_arms = args.get("required_arms")
    if required_arms is not None and (
        isinstance(required_arms, bool) or required_arms not in (1, 2)
    ):
        return "expert.learn required_arms must be 1 or 2 when provided."
    semantic_ambiguity = args.get("semantic_ambiguity")
    if semantic_ambiguity is not None and (
        isinstance(semantic_ambiguity, bool)
        or not isinstance(semantic_ambiguity, (int, float))
        or not 0.0 <= float(semantic_ambiguity) <= 1.0
    ):
        return "expert.learn semantic_ambiguity must be a number in [0, 1]."
    if forbid_episode_inventory_counts:
        serialized = json.dumps(args, ensure_ascii=False)
        count_pattern = re.compile(
            r"\b(?:one|two|three|four|1|2|3|4)\s+"
            r"(?:visually\s+distinct\s+)?(?:cubes?|bottles?|pens?)\b",
            re.IGNORECASE,
        )
        labeled_object_pattern = re.compile(
            r"\b(?:cube|bottle|pen)\s+[A-D]\b",
            re.IGNORECASE,
        )
        if count_pattern.search(serialized) or labeled_object_pattern.search(serialized):
            return (
                "expert.learn for this compositional task must not encode the learning "
                "episode's per-type inventory counts or A/B object list in any field. "
                "Replace them with 'the five objects specified by the current episode'; "
                "retain only transferable per-object skills."
            )
    return None


def expert_learning_gate_error(required: bool, learned: bool) -> str | None:
    if required and not learned:
        return (
            "Expert-learning gate: before any environment action, first call "
            '{"tool":"expert.retrieve","args":{"mode":"trajectory"},"reason":"study all available expert trajectories"} '
            "and then save structured experience with "
            '{"tool":"expert.learn","args":{...},"reason":"summarize expert trajectory constraints"}.'
        )
    return None


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
