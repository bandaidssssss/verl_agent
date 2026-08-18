#!/usr/bin/env python3
"""Replay one historical Proposal -> Feasibility decision without training.

The report for decision trial N contains the agent decision made after trials
1..N-1 completed.  For example, replaying ``trials/0007/trial_report.json``
loads only trials 1..6 and asks the agents to choose parameters for trial 7.

Typical use after editing the defaults near the top of this file:

    python replay_agent_prompts.py

Pass copied/edited prompt files to the last two options for A/B experiments.
Use ``proposal-only`` or ``feasibility-only`` to keep that role's original
recorded input fixed and isolate one prompt before testing the linked pipeline.
The source run is never modified, and no verl training command is started.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


from agents import AgentError, LLMRoleAgent
from config_utils import load_json, write_json
from orchestrator import (
    TuningOrchestrator,
    _baseline_proposal,
    _next_stage_baseline,
    _runs_automatic_baseline,
    determine_stage,
)
from vllm_metrics import summarize_vllm_metrics


ROOT = Path(__file__).resolve().parent

# =============================================================================
# 每次做 Prompt 实验时，通常只需要修改下面 5 项
# =============================================================================

# 一次完整实验的目录。脚本会读取其中的 trials.jsonl 和 trials/NNNN/。
DEFAULT_RUN_DIR = ROOT / "output" / "0731_1702_2026"

# 表示“第几次实验已经结束”。填 6 就是基于 trial 1..6 决定 trial 7。
DEFAULT_AFTER_TRIAL = 3

# 建议只修改测试副本，不直接修改正式 Prompt。
DEFAULT_PROPOSAL_PROMPT = ROOT / "prompts" / "proposal_test.md"
DEFAULT_FEASIBILITY_PROMPT = ROOT / "prompts" / "feasibility_test.md"

# 可选：pipeline / proposal-only / feasibility-only
DEFAULT_MODE = "pipeline"


# =============================================================================
# 其他参数一般不需要频繁修改，但也全部在这里提供默认值
# =============================================================================

# 如果填写某个 trials/NNNN 或 trial_report.json，则优先使用它，并自动推导
# run-dir 和 after-trial。保持 None 时使用上面的 RUN_DIR + AFTER_TRIAL。
DEFAULT_TRIAL_REPORT: Path | None = None

DEFAULT_BASE_CONFIG = ROOT / "config" / "base_parameters.json"
DEFAULT_AGENT_CONFIG = ROOT / "config" / "agent_config.json"

# None 表示自动保存到 output/prompt_replays/... 的带时间戳目录。
DEFAULT_OUTPUT_DIR: Path | None = None

# True：只渲染完整 Prompt，不调用 LLM；False：真正执行 Agent 回放。
DEFAULT_RENDER_ONLY = False

# True：隐藏逐条工具调用与回答；最终比较结果仍会打印。
DEFAULT_QUIET = False

# False：默认只回放紧凑的 JSON 历史，不复制每个 trial 的 train.log。
# 需要测试 read_trial_log_excerpt/read_trial_metrics 时，可传 --copy-logs。
DEFAULT_COPY_LOGS = False


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _read_jsonl_prefix(path: Path, after_trial: int) -> list[dict[str, Any]]:
    """Read only until trial IDs 1..after_trial have all been collected."""
    if not path.is_file():
        return []
    by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"history row {line_number} in {path} must be a JSON object"
                )
            trial_id = row.get("trial_id")
            if not isinstance(trial_id, int) or isinstance(trial_id, bool):
                continue
            if not 1 <= trial_id <= after_trial:
                continue
            if trial_id in by_id:
                raise ValueError(f"duplicate trial_id={trial_id} in {path}")
            by_id[trial_id] = row
            if len(by_id) == after_trial:
                break
    return [by_id[trial_id] for trial_id in sorted(by_id)]


def load_history_prefix(run_dir: Path, after_trial: int) -> list[dict[str, Any]]:
    """Load exactly the completed history visible after ``after_trial``."""
    if after_trial < 1:
        raise ValueError("--after-trial must be at least 1")

    rows = _read_jsonl_prefix(run_dir / "trials.jsonl", after_trial)
    if not rows:
        for trial_id in range(1, after_trial + 1):
            report_path = run_dir / "trials" / f"{trial_id:04d}" / "trial_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"cannot reconstruct history: missing {report_path} and no trials.jsonl"
                )
            rows.append(load_json(report_path))

    by_id = {int(row["trial_id"]): row for row in rows}

    missing = [
        trial_id for trial_id in range(1, after_trial + 1) if trial_id not in by_id
    ]
    if missing:
        raise ValueError(
            f"history is missing completed trial IDs {missing}; available IDs are {sorted(by_id)}"
        )
    return [copy.deepcopy(by_id[trial_id]) for trial_id in range(1, after_trial + 1)]


def _write_replay_history(
    source_run: Path,
    sandbox_dir: Path,
    history: Sequence[Mapping[str, Any]],
    *,
    copy_logs: bool = False,
) -> tuple[Path, list[int], list[int]]:
    """Write isolated history without carrying large trial artifacts by default.

    vLLM CSV samples are summarized into the replay history instead of copied.
    This keeps ``analyze_rollout_metrics`` accurate while preventing it from
    following an absolute path back into the historical run.
    """
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    history_path = sandbox_dir / "trials.jsonl"
    rewritten: list[dict[str, Any]] = []
    copied_logs: list[int] = []
    summarized_vllm_metrics: list[int] = []

    for source_row in history:
        row = copy.deepcopy(dict(source_row))
        trial_id = int(row["trial_id"])
        local_trial_dir = sandbox_dir / "trials" / f"{trial_id:04d}"
        source_log = source_run / "trials" / f"{trial_id:04d}" / "train.log"
        local_log = local_trial_dir / "train.log"
        if copy_logs and source_log.is_file():
            local_trial_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_log, local_log)
            copied_logs.append(trial_id)
            row["log_path"] = str(local_log.resolve())
        else:
            row["log_path"] = None

        source_vllm_metrics = (
            source_run / "trials" / f"{trial_id:04d}" / "vllm_metrics.csv"
        )
        if source_vllm_metrics.is_file():
            rollout_engine = row.get("rollout_engine")
            rollout_engine = (
                copy.deepcopy(dict(rollout_engine))
                if isinstance(rollout_engine, Mapping)
                else {}
            )
            rollout_engine["metrics"] = summarize_vllm_metrics(source_vllm_metrics)
            row["rollout_engine"] = rollout_engine
            summarized_vllm_metrics.append(trial_id)

        # The replay tool consumes the embedded summary above.  Never retain
        # an absolute source-run path or copy the raw CSV into the sandbox.
        row["vllm_metrics_path"] = None
        rewritten.append(row)

    with history_path.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return history_path, copied_logs, summarized_vllm_metrics


def _recorded_diagnosis(target_report: Mapping[str, Any]) -> Any:
    trace = target_report.get("agent_trace")
    if isinstance(trace, Mapping):
        proposal_trace = trace.get("proposal_conversation")
        if isinstance(proposal_trace, Mapping):
            context = proposal_trace.get("context")
            if isinstance(context, Mapping) and "diagnosis" in context:
                return copy.deepcopy(context.get("diagnosis"))
        diagnosis_trace = trace.get("diagnosis")
        if isinstance(diagnosis_trace, Mapping) and "result" in diagnosis_trace:
            return copy.deepcopy(diagnosis_trace.get("result"))
    return None


def _recorded_context(
    target_report: Mapping[str, Any], role: str
) -> dict[str, Any] | None:
    trace = target_report.get("agent_trace")
    if not isinstance(trace, Mapping):
        return None
    if role == "proposal":
        proposal = trace.get("proposal_conversation")
        context = proposal.get("context") if isinstance(proposal, Mapping) else None
        return copy.deepcopy(dict(context)) if isinstance(context, Mapping) else None
    reviews = trace.get("feasibility_reviews")
    if isinstance(reviews, list):
        for review in reversed(reviews):
            context = review.get("context") if isinstance(review, Mapping) else None
            if isinstance(context, Mapping):
                return copy.deepcopy(dict(context))
    return None


def _recorded_role_result(
    target_report: Mapping[str, Any], role: str
) -> dict[str, Any] | None:
    trace = target_report.get("agent_trace")
    if not isinstance(trace, Mapping):
        return None
    if role == "proposal":
        # proposal-only starts a fresh first turn from the frozen context.  If
        # the historical pipeline had rejection-feedback turns, compare with
        # its first proposal batch rather than its later repaired result.
        rejections = trace.get("rejections")
        if isinstance(rejections, list) and rejections:
            first = rejections[0]
            proposal = first.get("proposal") if isinstance(first, Mapping) else None
            if isinstance(proposal, Mapping):
                return copy.deepcopy(dict(proposal))
        proposal = trace.get("proposal_conversation")
        result = proposal.get("result") if isinstance(proposal, Mapping) else None
        return copy.deepcopy(dict(result)) if isinstance(result, Mapping) else None
    reviews = trace.get("feasibility_reviews")
    if isinstance(reviews, list):
        for review in reversed(reviews):
            result = review.get("result") if isinstance(review, Mapping) else None
            if isinstance(result, Mapping):
                return copy.deepcopy(dict(result))
    return None


def _prompt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _system_prompt(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "system":
            content = message.get("content")
            return str(content) if content is not None else None
    return None


def _save_rendered_prompts(result_dir: Path, trace: Mapping[str, Any]) -> list[str]:
    saved: list[str] = []
    proposal = trace.get("proposal_conversation")
    if isinstance(proposal, Mapping):
        content = _system_prompt(proposal.get("messages"))
        if content is not None:
            path = result_dir / "rendered_proposal.md"
            path.write_text(content, encoding="utf-8")
            saved.append(str(path))

    reviews = trace.get("feasibility_reviews")
    if isinstance(reviews, list):
        for index, review in enumerate(reviews, start=1):
            if not isinstance(review, Mapping):
                continue
            content = _system_prompt(review.get("messages"))
            if content is None:
                continue
            path = result_dir / f"rendered_feasibility_{index:02d}.md"
            path.write_text(content, encoding="utf-8")
            saved.append(str(path))
    return saved


def _change_targets(proposal: Any) -> dict[str, Any]:
    if not isinstance(proposal, Mapping):
        return {}
    targets = proposal.get("target_changes")
    if isinstance(targets, Mapping):
        return copy.deepcopy(dict(targets))
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in changes.items():
        result[str(key)] = value.get("to") if isinstance(value, Mapping) else value
    return result


def _behavior(report: Mapping[str, Any]) -> dict[str, Any]:
    proposal = report.get("proposal")
    feasibility = report.get("feasibility")
    proposal_map = proposal if isinstance(proposal, Mapping) else {}
    feasibility_map = feasibility if isinstance(feasibility, Mapping) else {}
    return {
        "stage": report.get("stage"),
        "decision": proposal_map.get("decision"),
        "candidate_id": proposal_map.get("candidate_id"),
        "reference_trial_id": proposal_map.get("reference_trial_id"),
        "changes": _change_targets(proposal_map),
        "feasibility_verdict": feasibility_map.get("verdict"),
        "selected_candidate_id": feasibility_map.get("selected_candidate_id"),
    }


def _semantic_behavior(report: Mapping[str, Any]) -> dict[str, Any]:
    behavior = _behavior(report)
    behavior.pop("candidate_id", None)
    behavior.pop("selected_candidate_id", None)
    return behavior


def _apply_recorded_transition_semantics(
    stage: str,
    parameters: dict[str, Any],
    proposal: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Mirror run() post-processing when stop advances to a stage baseline."""
    if proposal.get("decision") != "stop":
        return stage, parameters, proposal

    transition = _next_stage_baseline(stage, history)
    if transition is None:
        raise ValueError(
            "this stop decision cannot advance because the next stage has no valid reference; "
            "no trial report would be created for replay comparison"
        )
    next_stage, selected_parameters, reference = transition
    return (
        next_stage,
        copy.deepcopy(selected_parameters),
        _baseline_proposal(
            reference,
            "automatic baseline after Agent stopped the previous stage",
            proposal,
        ),
    )


def _default_result_dir(run_dir: Path, after_trial: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (
        ROOT
        / "output"
        / "prompt_replays"
        / run_dir.name
        / f"after_{after_trial:04d}"
        / timestamp
    )


def _resolve_case(args: argparse.Namespace) -> tuple[Path, int, Path]:
    if args.trial_report:
        target = _absolute(args.trial_report)
        if target.is_dir():
            target = target / "trial_report.json"
        if not target.is_file():
            raise FileNotFoundError(f"trial report does not exist: {target}")
        try:
            decision_trial = int(target.parent.name)
        except ValueError as exc:
            raise ValueError(
                "--trial-report must be .../trials/NNNN/trial_report.json or its NNNN directory"
            ) from exc
        run_dir = target.parents[2]
        after_trial = decision_trial - 1
        if args.after_trial is not None and args.after_trial != after_trial:
            raise ValueError(
                f"{target} is the decision for trial {decision_trial}, so --after-trial must be "
                f"{after_trial}, not {args.after_trial}"
            )
        return run_dir, after_trial, target

    if not args.run_dir:
        raise ValueError(
            "provide --run-dir with --after-trial, or provide --trial-report"
        )
    if args.after_trial is None:
        raise ValueError("--run-dir requires --after-trial")
    run_dir = _absolute(args.run_dir)
    after_trial = args.after_trial
    target = run_dir / "trials" / f"{after_trial + 1:04d}" / "trial_report.json"
    if not target.is_file():
        raise FileNotFoundError(
            f"comparison report does not exist: {target}. "
            "Choose N only when the original decision report for N+1 was recorded."
        )
    return run_dir, after_trial, target


def _render_only(
    result_dir: Path,
    orchestrator: TuningOrchestrator,
    target_report: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    rendered: list[str] = []
    role_agents = (
        ("proposal", orchestrator.agents.proposal),
        ("feasibility", orchestrator.agents.feasibility),
    )
    for role, agent in role_agents:
        if mode == "proposal-only" and role != "proposal":
            continue
        if mode == "feasibility-only" and role != "feasibility":
            continue
        context = _recorded_context(target_report, role)
        if context is None:
            continue
        conversation = agent.new_conversation(context)
        content = _system_prompt(conversation.messages)
        if content is None:
            continue
        path = result_dir / f"rendered_{role}.md"
        path.write_text(content, encoding="utf-8")
        rendered.append(str(path))
    if not rendered:
        raise ValueError(
            "the target report contains no recorded Proposal/Feasibility contexts"
        )
    return {"mode": f"render-only:{mode}", "rendered_prompts": rendered}


def _run_role_only(
    role: str,
    result_dir: Path,
    orchestrator: TuningOrchestrator,
    target_report: Mapping[str, Any],
) -> dict[str, Any]:
    context = _recorded_context(target_report, role)
    original = _recorded_role_result(target_report, role)
    if context is None or original is None:
        raise ValueError(
            f"the target report contains no recorded {role} context/result"
        )

    run = (
        orchestrator.agents.propose(context)
        if role == "proposal"
        else orchestrator.agents.review(context)
    )
    trace = run.as_trace()
    rendered_paths: list[str] = []
    content = _system_prompt(trace.get("messages"))
    if content is not None:
        path = result_dir / f"rendered_{role}.md"
        path.write_text(content, encoding="utf-8")
        rendered_paths.append(str(path))
    return {
        "mode": f"{role}-only",
        "comparison_basis": (
            "first Proposal response from the frozen recorded context"
            if role == "proposal"
            else "Feasibility response for the frozen recorded candidate set"
        ),
        "comparison": {
            "original": original,
            "replay": run.result,
            "same_output": original == run.result,
        },
        "replay_trace": trace,
        "rendered_prompts": rendered_paths,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Proposal and Feasibility after a selected completed trial, "
            "without launching training"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="Historical run directory containing trials.jsonl",
    )
    source.add_argument(
        "--trial-report",
        default=DEFAULT_TRIAL_REPORT,
        help=(
            "The NNNN trial directory or trial_report.json whose recorded decision should be "
            "replayed; NNNN means history is cut off at NNNN-1"
        ),
    )
    parser.add_argument(
        "--after-trial",
        type=int,
        default=DEFAULT_AFTER_TRIAL,
        help="Last completed trial to expose; the replay decides parameters for N+1",
    )
    parser.add_argument(
        "--proposal-prompt",
        default=DEFAULT_PROPOSAL_PROMPT,
        help="Proposal prompt template to test",
    )
    parser.add_argument(
        "--feasibility-prompt",
        default=DEFAULT_FEASIBILITY_PROMPT,
        help="Feasibility prompt template to test",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "proposal-only", "feasibility-only"],
        default=DEFAULT_MODE,
        help=(
            "pipeline tests the linked Proposal -> Validator -> Feasibility flow; the two "
            "role-only modes freeze that role's original recorded input"
        ),
    )
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
    )
    parser.add_argument(
        "--agent-config",
        default=DEFAULT_AGENT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for replay artifacts and comparison JSON",
    )
    parser.add_argument(
        "--render-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENDER_ONLY,
        help="Render edited prompts with the recorded contexts; make no LLM API calls",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_QUIET,
        help="Do not stream individual agent tool calls and answers",
    )
    parser.add_argument(
        "--copy-logs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_COPY_LOGS,
        help=(
            "Copy historical train.log files into the replay sandbox so log-reading tools "
            "remain available; disabled by default to avoid duplicate storage"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_dir, after_trial, target_report_path = _resolve_case(args)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        proposal_prompt = _absolute(args.proposal_prompt)
        feasibility_prompt = _absolute(args.feasibility_prompt)
        for role, path in (
            ("proposal", proposal_prompt),
            ("feasibility", feasibility_prompt),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{role} prompt does not exist: {path}")

        history = load_history_prefix(run_dir, after_trial)
        target_report = load_json(target_report_path)
        decision_trial = after_trial + 1
        recorded_trial_id = target_report.get("trial_id")
        if recorded_trial_id != decision_trial:
            raise ValueError(
                f"target report trial_id is {recorded_trial_id!r}; expected {decision_trial}"
            )

        required_role = "feasibility" if args.mode == "feasibility-only" else "proposal"
        if (
            _recorded_context(target_report, required_role) is None
            or _recorded_role_result(target_report, required_role) is None
        ):
            raise ValueError(
                f"trial {decision_trial} has no recorded {required_role} decision to replay; "
                "baseline/confirm reports without an Agent trace are not prompt test cases"
            )

        result_dir = (
            _absolute(args.output_dir)
            if args.output_dir
            else _default_result_dir(run_dir, after_trial)
        )
        try:
            result_dir.relative_to(run_dir)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"--output-dir must be outside the historical run so it cannot be modified: {run_dir}"
            )
        if result_dir.exists():
            raise ValueError(f"output directory already exists: {result_dir}")
        result_dir.mkdir(parents=True, exist_ok=False)
        sandbox_dir = result_dir / "sandbox"
        history_path, copied_logs, summarized_vllm_metrics = _write_replay_history(
            run_dir,
            sandbox_dir,
            history,
            copy_logs=args.copy_logs,
        )

        configured_base = load_json(_absolute(args.base_config))
        first_parameters = history[0].get("parameters")
        if isinstance(first_parameters, Mapping):
            base_parameters = copy.deepcopy(dict(first_parameters))
            base_parameters_source = f"trial {history[0]['trial_id']} parameters"
        else:
            base_parameters = configured_base
            base_parameters_source = str(_absolute(args.base_config))

        original_proposal_context = _recorded_context(target_report, "proposal")
        config = load_json(_absolute(args.agent_config))
        recorded_constraints = (
            original_proposal_context.get("constraints")
            if isinstance(original_proposal_context, Mapping)
            else None
        )
        if isinstance(recorded_constraints, Mapping):
            for key in (
                "min_proposal_candidates",
                "max_proposal_candidates",
                "max_parameter_changes",
                "preserve_hardware_token_budget",
                "resource_memory_reserve_mib",
                "throughput_memory_reserve_mib",
                "resource_memory_limit_pct",
                "throughput_memory_limit_pct",
            ):
                if key in recorded_constraints:
                    config[key] = copy.deepcopy(recorded_constraints[key])
        config.update(
            {
                "output_dir": str(sandbox_dir),
                "agent_mode": "llm",
                "stream_agent_events": not args.quiet,
            }
        )
        # TuningOrchestrator gives OUTPUT_PATH precedence over config.  Force
        # it to the replay sandbox inside this child process so a sourced user
        # environment can never redirect writes into the historical run.
        os.environ["OUTPUT_PATH"] = str(sandbox_dir)
        orchestrator = TuningOrchestrator(ROOT, base_parameters, config)
        if orchestrator.history_path != history_path.resolve():
            raise RuntimeError(
                f"isolated history mismatch: {orchestrator.history_path} != {history_path.resolve()}"
            )
        registry = orchestrator.agents.registry
        orchestrator.agents.proposal = LLMRoleAgent(
            "proposal", proposal_prompt, registry, config
        )
        orchestrator.agents.feasibility = LLMRoleAgent(
            "feasibility", feasibility_prompt, registry, config
        )

        metadata: dict[str, Any] = {
            "source_run": str(run_dir),
            "target_report": str(target_report_path),
            "after_trial": after_trial,
            "decision_trial": decision_trial,
            "mode": args.mode,
            "history_trial_ids": [row["trial_id"] for row in history],
            "copied_log_trial_ids": copied_logs,
            "summarized_vllm_metrics_trial_ids": summarized_vllm_metrics,
            "isolated_history_path": str(history_path),
            "proposal_prompt": {
                "path": str(proposal_prompt),
                "sha256": _prompt_digest(proposal_prompt),
            },
            "feasibility_prompt": {
                "path": str(feasibility_prompt),
                "sha256": _prompt_digest(feasibility_prompt),
            },
            "model": orchestrator.agents.proposal.model,
            "base_parameters_source": base_parameters_source,
            "reproducibility_note": (
                "LLM sampling, model versions, live GPU state, and local verl docs may differ from "
                "the historical run; compare repeated runs as behavioral experiments."
            ),
        }
        print(
            f"Replay mode={args.mode}: completed history ends at trial {after_trial}; "
            f"decision is for trial {decision_trial}; training is disabled.",
            flush=True,
        )

        if args.render_only:
            rendered = _render_only(result_dir, orchestrator, target_report, args.mode)
            output = {"metadata": metadata, **rendered}
            write_json(result_dir / "replay_result.json", output)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"\nSaved to: {result_dir}")
            return 0

        if args.mode in {"proposal-only", "feasibility-only"}:
            role = "proposal" if args.mode == "proposal-only" else "feasibility"
            role_output = _run_role_only(role, result_dir, orchestrator, target_report)
            output = {
                "metadata": {
                    **metadata,
                    "rendered_prompts": role_output.pop("rendered_prompts"),
                },
                **role_output,
            }
            write_json(result_dir / "replay_result.json", output)
            print(json.dumps(output["comparison"], ensure_ascii=False, indent=2))
            print(f"\nFull result: {result_dir / 'replay_result.json'}")
            return 0

        recorded_diagnosis = _recorded_diagnosis(target_report)
        if history[-1].get("result") != "success" and original_proposal_context is None:
            raise ValueError(
                "the last selected trial failed, but the target report has no recorded Proposal "
                "context from which to freeze Diagnosis"
            )

        def frozen_diagnosis(_: list[dict[str, Any]]) -> tuple[Any, Any]:
            if recorded_diagnosis is None:
                return None, None
            return (
                copy.deepcopy(recorded_diagnosis),
                {
                    "role": "diagnosis",
                    "source": "frozen_from_target_report",
                    "result": copy.deepcopy(recorded_diagnosis),
                },
            )

        # Freeze Diagnosis so this experiment changes only Proposal and
        # Feasibility prompts.  Assignment on the instance intentionally
        # replaces the bound method for this one replay process.
        orchestrator._diagnosis = frozen_diagnosis  # type: ignore[method-assign]

        derived_stage = determine_stage(history, config)
        recorded_stage = (
            original_proposal_context.get("current_stage")
            if isinstance(original_proposal_context, Mapping)
            else None
        )
        stage = str(recorded_stage or derived_stage)
        if stage in {"done", "stopped_unstable"}:
            raise ValueError(
                f"history after trial {after_trial} is already terminal: {stage}"
            )
        recorded_current = (
            original_proposal_context.get("current_parameters")
            if isinstance(original_proposal_context, Mapping)
            else None
        )
        recorded_reference = (
            original_proposal_context.get("reference_trial")
            if isinstance(original_proposal_context, Mapping)
            else None
        )
        if isinstance(recorded_current, Mapping) and isinstance(
            recorded_reference, Mapping
        ):
            current_parameters = copy.deepcopy(dict(recorded_current))
            reference = copy.deepcopy(dict(recorded_reference))
            starting_point_source = "recorded Proposal context"
        else:
            current_parameters, reference = orchestrator._starting_point(stage, history)
            starting_point_source = "current orchestrator logic"
        if _runs_automatic_baseline(stage, history):
            candidate = current_parameters
            proposal = _baseline_proposal(reference, "automatic stage baseline")
            feasibility = {"verdict": "valid", "reason": "stage baseline"}
            trace = None
        else:
            candidate, proposal, feasibility, trace = orchestrator._propose_candidate(
                stage, current_parameters, history, reference
            )
        trial_would_run = proposal.get("decision") != "blocked"
        if trial_would_run:
            stage, candidate, proposal = _apply_recorded_transition_semantics(
                stage, candidate, proposal, history
            )
            replay_parameters: dict[str, Any] | None = candidate
        else:
            # Production run() writes a blocked state and does not create the
            # next trial.  Keep that behavioral difference explicit rather
            # than pretending the unchanged parameters would be executed.
            replay_parameters = None

        replay_report = {
            "trial_id": decision_trial,
            "stage": stage,
            "trial_would_run": trial_would_run,
            "parameters": replay_parameters,
            "proposal": proposal,
            "feasibility": feasibility,
            "agent_trace": trace,
        }
        original_behavior = _behavior(target_report)
        replay_behavior = _behavior(replay_report)
        comparison = {
            "original": original_behavior,
            "replay": replay_behavior,
            "same_semantic_behavior": _semantic_behavior(target_report)
            == _semantic_behavior(replay_report),
            "same_candidate_labels": (
                original_behavior.get("candidate_id")
                == replay_behavior.get("candidate_id")
                and original_behavior.get("selected_candidate_id")
                == replay_behavior.get("selected_candidate_id")
            ),
            "same_selected_proposal": target_report.get("proposal") == proposal,
            "same_feasibility_review": target_report.get("feasibility") == feasibility,
            "trial_would_run": trial_would_run,
            "same_parameters": (
                target_report.get("parameters") == replay_parameters
                if trial_would_run
                else None
            ),
        }
        output = {
            "metadata": {
                **metadata,
                "stage": stage,
                "derived_stage": derived_stage,
                "starting_point_source": starting_point_source,
                "recorded_diagnosis": recorded_diagnosis,
            },
            "comparison": comparison,
            "original": {
                "parameters": target_report.get("parameters"),
                "proposal": target_report.get("proposal"),
                "feasibility": target_report.get("feasibility"),
            },
            "replay": replay_report,
        }
        rendered_paths = _save_rendered_prompts(result_dir, trace)
        output["metadata"]["rendered_prompts"] = rendered_paths
        write_json(result_dir / "replay_result.json", output)

        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"\nFull result: {result_dir / 'replay_result.json'}")
        if rendered_paths:
            print("Rendered prompts:")
            for path in rendered_paths:
                print(f"  {path}")
        return 0
    except (AgentError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
