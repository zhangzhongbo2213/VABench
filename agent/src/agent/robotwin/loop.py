from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

from agent.client import ChatClient
from agent.session import Store

from .correction import diagnose_with_script, synthesize_constraint_update
from .eval import robotwin_run_parent, run_robotwin_eval
from .learned_constraints import LearnedConstraintSet, default_learned_constraint_file
from .spatial_tool import SpatialToolConfig


EvalRunner = Callable[..., Path]
DiagnosisRunner = Callable[..., Path]
CorrectionRunner = Callable[..., Path]


@dataclass
class LoopRound:
    index: int
    run_dir: Path
    success: bool
    diagnosis_report: Path | None = None
    constraint_update: Path | None = None
    learned_constraint_file: Path | None = None


@dataclass
class LoopResult:
    loop_dir: Path
    status: str
    rounds: list[LoopRound] = field(default_factory=list)
    manual_feedback_template: Path | None = None

    def save(self) -> Path:
        self.loop_dir.mkdir(parents=True, exist_ok=True)
        path = self.loop_dir / "loop_result.json"
        payload = {
            "status": self.status,
            "rounds": [
                {
                    "index": item.index,
                    "run_dir": str(item.run_dir),
                    "success": item.success,
                    "diagnosis_report": str(item.diagnosis_report) if item.diagnosis_report else None,
                    "constraint_update": str(item.constraint_update) if item.constraint_update else None,
                    "learned_constraint_file": str(item.learned_constraint_file) if item.learned_constraint_file else None,
                }
                for item in self.rounds
            ],
            "manual_feedback_template": str(self.manual_feedback_template) if self.manual_feedback_template else None,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def run_robotwin_loop(
    client: ChatClient,
    store: Store,
    state_dir: Path,
    *,
    task: str,
    config: str,
    seed: int,
    active_arm: str | None,
    max_steps: int,
    max_rounds: int = 2,
    correction: str = "auto",
    benchmark_dir: Path | None = None,
    session_id: str | None = None,
    model_view: str = "active",
    record_view: str = "gripper_follow",
    initial_camera_view: str = "center_high",
    camera_policy: str = "fine",
    width: int = 1280,
    height: int = 960,
    model_overlay: str = "none",
    model_debug_overlay: str = "eepose",
    record_overlay: str = "eepose",
    expert_demo_dir: Path | None = None,
    expert_learning_file: Path | None = None,
    local_constraint_profile: str = "generic",
    learned_constraint_file: Path | None = None,
    manual_feedback_file: Path | None = None,
    diagnosis_max_turns: int = 18,
    display: bool = True,
    stream: bool = True,
    spatial_tool_config: SpatialToolConfig | None = None,
    eval_runner: EvalRunner = run_robotwin_eval,
    diagnosis_runner: DiagnosisRunner = diagnose_with_script,
    correction_runner: CorrectionRunner = synthesize_constraint_update,
) -> LoopResult:
    if correction not in {"auto", "manual", "none"}:
        raise ValueError("correction must be one of: auto, manual, none")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    loop_dir = (
        state_dir
        / "runs"
        / "loops"
        / robotwin_run_parent(task=task, seed=seed)
        / f"loop_{uuid4().hex[:8]}"
    )
    loop_dir.mkdir(parents=True, exist_ok=True)
    constraint_file = learned_constraint_file or default_learned_constraint_file(state_dir, task=task)
    result = LoopResult(loop_dir=loop_dir, status="running")

    for round_index in range(1, max_rounds + 1):
        active_constraint_file = constraint_file if constraint_file.exists() else None
        if display:
            print(f"[loop] round {round_index}/{max_rounds} eval start", flush=True)
            if active_constraint_file:
                print(f"[loop] using learned constraints: {active_constraint_file}", flush=True)
        run_dir = eval_runner(
            client,
            store,
            state_dir,
            task=task,
            config=config,
            seed=seed,
            active_arm=active_arm,
            max_steps=max_steps,
            benchmark_dir=benchmark_dir,
            session_id=session_id,
            model_view=model_view,
            record_view=record_view,
            initial_camera_view=initial_camera_view,
            camera_policy=camera_policy,
            width=width,
            height=height,
            model_overlay=model_overlay,
            model_debug_overlay=model_debug_overlay,
            record_overlay=record_overlay,
            expert_demo_dir=expert_demo_dir,
            expert_learning_file=expert_learning_file,
            learned_constraint_file=active_constraint_file,
            local_constraint_profile=local_constraint_profile,
            display=display,
            stream=stream,
            spatial_tool_config=spatial_tool_config,
        )
        success = run_success(run_dir)
        round_result = LoopRound(index=round_index, run_dir=run_dir, success=success, learned_constraint_file=active_constraint_file)
        result.rounds.append(round_result)
        result.save()
        if display:
            print(f"[loop] round {round_index} eval finished success={success} run={run_dir}", flush=True)
        if success:
            result.status = "success"
            result.save()
            return result
        if round_index >= max_rounds or correction == "none":
            result.status = "failed"
            result.save()
            return result

        diagnosis_dir = run_dir / f"diagnosis_loop_round{round_index:03d}"
        if display:
            print(f"[loop] round {round_index} diagnosis start", flush=True)
        diagnosis_report = diagnosis_runner(client, run_dir, diagnosis_dir, max_turns=diagnosis_max_turns, display=display)
        round_result.diagnosis_report = diagnosis_report
        result.save()

        feedback_text = ""
        if correction == "manual":
            if manual_feedback_file is None:
                template = write_manual_feedback_template(diagnosis_dir, diagnosis_report)
                result.status = "manual_feedback_required"
                result.manual_feedback_template = template
                result.save()
                if display:
                    print(f"[loop] manual feedback required: {template}", flush=True)
                return result
            feedback_text = manual_feedback_file.read_text(encoding="utf-8", errors="replace")

        current_constraints = LearnedConstraintSet.load_or_empty(constraint_file, task=task)
        correction_dir = run_dir / f"correction_loop_round{round_index:03d}"
        update_path = correction_runner(
            client,
            task=task,
            run_dir=run_dir,
            diagnosis_report=diagnosis_report,
            current_constraints=current_constraints,
            output_dir=correction_dir,
            human_feedback=feedback_text,
            default_scope="task",
            display=display,
        )
        update = json.loads(update_path.read_text(encoding="utf-8"))
        current_constraints.apply_update(update, source_run=run_dir.name, diagnosis_report=diagnosis_report)
        round_result.constraint_update = update_path
        round_result.learned_constraint_file = constraint_file
        result.save()
        if display:
            print(f"[loop] learned constraints updated: {constraint_file}", flush=True)

    result.status = "failed"
    result.save()
    return result


def run_success(run_dir: Path) -> bool:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("success"))


def write_manual_feedback_template(diagnosis_dir: Path, diagnosis_report: Path) -> Path:
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    path = diagnosis_dir / "manual_feedback.md"
    path.write_text(
        "\n".join(
            [
                "# Manual Correction Feedback",
                "",
                f"Diagnosis report: {diagnosis_report}",
                "",
                "Write the protocol changes you want the agent to convert into task constraints.",
                "Keep these as instructions, not simulator coordinates.",
                "",
                "## Preserve",
                "- ",
                "",
                "## Add",
                "- ",
                "",
                "## Revise Or Remove",
                "- ",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
