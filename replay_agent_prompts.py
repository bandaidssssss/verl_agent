#!/usr/bin/env python3
"""Replay one recorded Proposal -> Feasibility decision from trial artifacts.

The selected target is a concrete ``trials/NNNN`` directory.  Its
``decision.json`` and ``agent_trace.json`` are the recorded decision; every
earlier trial is loaded through its own parameters, metrics, and decision
artifacts.  The source run is read-only and no training is launched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from agents import AgentError, AgentRun, AgentSet, LLMRoleAgent
from config_utils import load_json, write_json
from orchestrator import (
    TuningOrchestrator,
    _baseline_proposal,
    _next_stage_baseline,
    _runs_automatic_baseline,
    determine_stage,
)
from trial_storage import hydrate_trial, read_trial_indexes


ROOT = Path(__file__).resolve().parent

# Change these defaults for a local prompt experiment.
DEFAULT_TRIAL_DIR = ROOT / "output" / "0819_1134_2026" / "trials" / "0002"
DEFAULT_PROPOSAL_PROMPT = ROOT / "prompts" / "proposal_test.md"
DEFAULT_FEASIBILITY_PROMPT = ROOT / "prompts" / "feasibility_test.md"
DEFAULT_AGENT_CONFIG = ROOT / "config" / "agent_config.json"
DEFAULT_BASE_CONFIG = ROOT / "config" / "base_parameters.json"


@dataclass(frozen=True)
class ReplayCase:
    run_dir: Path
    trial_dir: Path
    trial_id: int
    report: dict[str, Any]
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    decision: dict[str, Any]
    agent_trace: dict[str, Any]


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.resolve()


def _json_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing trial artifact: {path}")
        return {}
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"trial artifact must be a JSON object: {path}")
    return value


def load_replay_case(path: str | Path) -> ReplayCase:
    candidate = _absolute(path)
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.name == "trials":
        raise ValueError("--trial-dir must name one trials/NNNN directory")
    if candidate.parent.name != "trials":
        raise ValueError("--trial-dir must be a trials/NNNN directory")
    try:
        trial_id = int(candidate.name)
    except ValueError as exc:
        raise ValueError("trial directory name must be a numeric trial ID") from exc
    if trial_id < 1:
        raise ValueError("trial ID must be positive")

    report = _json_object(candidate / "trial_report.json")
    if report.get("trial_id") != trial_id:
        raise ValueError(
            f"trial_report.json says trial_id={report.get('trial_id')!r}, expected {trial_id}"
        )
    decision = _json_object(candidate / "decision.json")
    return ReplayCase(
        run_dir=candidate.parents[1],
        trial_dir=candidate,
        trial_id=trial_id,
        report=report,
        parameters=_json_object(candidate / "parameters.json"),
        metrics=_json_object(candidate / "metrics.json"),
        decision=decision,
        agent_trace=_json_object(candidate / "agent_trace.json", required=False),
    )


def load_prior_trials(case: ReplayCase) -> list[dict[str, Any]]:
    """Hydrate only trials that existed when the target decision was made."""
    history_path = case.run_dir / "trials.jsonl"
    indexes = read_trial_indexes(history_path)
    by_id = {
        row.get("trial_id"): row
        for row in indexes
        if isinstance(row.get("trial_id"), int) and row["trial_id"] < case.trial_id
    }
    expected = set(range(1, case.trial_id))
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(
            f"history before trial {case.trial_id} is incomplete; missing trial IDs {missing}"
        )
    return [hydrate_trial(by_id[trial_id], history_path) for trial_id in sorted(by_id)]


def _system_prompt(trace: Mapping[str, Any]) -> str | None:
    messages = trace.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "system":
            content = message.get("content")
            return str(content) if content is not None else None
    return None


def _prompt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_result_dir(case: ReplayCase) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ROOT / "output" / "prompt_replays" / case.run_dir.name / case.trial_dir.name / stamp


def _recorded_diagnosis(case: ReplayCase) -> Any:
    if "diagnosis" in case.decision:
        return copy.deepcopy(case.decision["diagnosis"])
    return copy.deepcopy(case.agent_trace.get("diagnosis_summary"))


def _recorded_constraints(case: ReplayCase) -> Mapping[str, Any] | None:
    conversation = case.agent_trace.get("proposal_conversation")
    context = conversation.get("context") if isinstance(conversation, Mapping) else None
    constraints = context.get("constraints") if isinstance(context, Mapping) else None
    return constraints if isinstance(constraints, Mapping) else None


def _make_orchestrator(
    case: ReplayCase,
    history: list[dict[str, Any]],
    result_dir: Path,
    proposal_prompt: Path,
    feasibility_prompt: Path,
    agent_config: Path,
    base_config: Path,
    quiet: bool,
) -> TuningOrchestrator:
    config = _json_object(agent_config)
    constraints = _recorded_constraints(case)
    if constraints is not None:
        for key in (
            "min_proposal_candidates",
            "max_proposal_candidates",
            "max_parameter_changes",
            "preserve_hardware_token_budget",
            "resource_memory_reserve_mib",
            "throughput_memory_reserve_mib",
        ):
            if key in constraints:
                config[key] = copy.deepcopy(constraints[key])
    config.update(
        {
            "agent_mode": "llm",
            "output_dir": str(result_dir),
            "stream_agent_events": not quiet,
        }
    )
    base_parameters = (
        dict(history[0]["parameters"])
        if history and isinstance(history[0].get("parameters"), Mapping)
        else _json_object(base_config)
    )
    orchestrator = TuningOrchestrator(ROOT, base_parameters, config)

    source_history_path = case.run_dir / "trials.jsonl"
    agents = AgentSet(ROOT, "llm", config, source_history_path)
    agents.proposal = LLMRoleAgent("proposal", proposal_prompt, agents.registry, config)
    agents.feasibility = LLMRoleAgent(
        "feasibility", feasibility_prompt, agents.registry, config
    )
    orchestrator.agents = agents
    return orchestrator


def _apply_stop_transition(
    stage: str,
    parameters: dict[str, Any],
    proposal: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if proposal.get("decision") != "stop":
        return stage, parameters, proposal
    transition = _next_stage_baseline(stage, history)
    if transition is None:
        raise ValueError("stop cannot advance: no successful next-stage reference")
    next_stage, next_parameters, reference = transition
    return (
        next_stage,
        next_parameters,
        _baseline_proposal(
            reference,
            "automatic baseline after Agent stopped the previous stage",
            proposal,
        ),
    )


def _change_targets(proposal: Mapping[str, Any]) -> dict[str, Any]:
    changes = proposal.get("target_changes")
    if isinstance(changes, Mapping):
        return dict(changes)
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    return {
        str(key): value.get("to") if isinstance(value, Mapping) else value
        for key, value in changes.items()
    }


def _behavior(
    stage: str, proposal: Mapping[str, Any], feasibility: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage": stage,
        "decision": proposal.get("decision"),
        "reference_trial_id": proposal.get("reference_trial_id"),
        "changes": _change_targets(proposal),
        "feasibility_verdict": feasibility.get("verdict"),
        "selected_candidate_id": feasibility.get("selected_candidate_id"),
    }


def _save_rendered_prompts(
    result_dir: Path, trace: Mapping[str, Any]
) -> list[str]:
    saved: list[str] = []
    proposal = trace.get("proposal_conversation")
    if isinstance(proposal, Mapping):
        content = _system_prompt(proposal)
        if content is not None:
            path = result_dir / "rendered_proposal.md"
            path.write_text(content, encoding="utf-8")
            saved.append(str(path))
    reviews = trace.get("feasibility_reviews")
    if isinstance(reviews, list):
        for index, review in enumerate(reviews, start=1):
            if not isinstance(review, Mapping):
                continue
            content = _system_prompt(review)
            if content is None:
                continue
            path = result_dir / f"rendered_feasibility_{index:02d}.md"
            path.write_text(content, encoding="utf-8")
            saved.append(str(path))
    return saved


def render_proposal_context(
    case: ReplayCase,
    history: list[dict[str, Any]],
    orchestrator: TuningOrchestrator,
) -> dict[str, Any]:
    """Rebuild the current Proposal context without using the old trace context."""
    stage = determine_stage(history, orchestrator.config)
    if _runs_automatic_baseline(stage, history):
        raise ValueError("target decision is an orchestrator baseline, not a Proposal prompt")
    current, reference = orchestrator._starting_point(stage, history)
    diagnosis = _recorded_diagnosis(case)

    def frozen_diagnosis(_: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
        return diagnosis, {
            "role": "diagnosis",
            "source": "target_trial_decision_artifact",
            "result": copy.deepcopy(diagnosis),
        }

    proposal_agent = orchestrator.agents.proposal

    class CaptureAgents:
        def propose(self, context: Mapping[str, Any] | None = None, conversation: Any = None) -> AgentRun:
            if context is None or conversation is not None:
                raise RuntimeError("render-only capture expects the first Proposal turn")
            return AgentRun(
                {
                    "decision": "stop",
                    "reason": "render-only capture",
                    "candidates": [],
                },
                proposal_agent.new_conversation(context),
            )

    orchestrator._diagnosis = frozen_diagnosis  # type: ignore[method-assign]
    orchestrator.agents = CaptureAgents()  # type: ignore[assignment]
    _, _, _, trace = orchestrator._propose_candidate(stage, current, history, reference)
    return trace


def replay_pipeline(
    case: ReplayCase,
    history: list[dict[str, Any]],
    orchestrator: TuningOrchestrator,
) -> dict[str, Any]:
    stage_before = determine_stage(history, orchestrator.config)
    if stage_before in {"done", "stopped_unstable"}:
        raise ValueError(f"history is already terminal: {stage_before}")
    current, reference = orchestrator._starting_point(stage_before, history)
    diagnosis = _recorded_diagnosis(case)

    def frozen_diagnosis(_: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
        return diagnosis, {
            "role": "diagnosis",
            "source": "target_trial_decision_artifact",
            "result": copy.deepcopy(diagnosis),
        }

    orchestrator._diagnosis = frozen_diagnosis  # type: ignore[method-assign]
    if _runs_automatic_baseline(stage_before, history):
        parameters = current
        proposal = _baseline_proposal(reference, "automatic stage baseline")
        feasibility: dict[str, Any] = {"verdict": "valid", "reason": "stage baseline"}
        trace: dict[str, Any] = {}
    else:
        parameters, proposal, feasibility, trace = orchestrator._propose_candidate(
            stage_before, current, history, reference
        )

    trial_would_run = proposal.get("decision") != "blocked"
    stage_after = stage_before
    if trial_would_run:
        stage_after, parameters, proposal = _apply_stop_transition(
            stage_before, parameters, proposal, history
        )
    else:
        parameters = None
    return {
        "trial_id": case.trial_id,
        "stage_before": stage_before,
        "stage": stage_after,
        "trial_would_run": trial_would_run,
        "parameters": parameters,
        "proposal": proposal,
        "feasibility": feasibility,
        "agent_trace": trace,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one trial decision from concrete trial artifacts without training."
    )
    parser.add_argument(
        "--trial-dir",
        default=DEFAULT_TRIAL_DIR,
        help="Source trials/NNNN directory whose decision is replayed",
    )
    parser.add_argument("--proposal-prompt", default=DEFAULT_PROPOSAL_PROMPT)
    parser.add_argument("--feasibility-prompt", default=DEFAULT_FEASIBILITY_PROMPT)
    parser.add_argument("--agent-config", default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--output-dir",
        help="Replay result directory; defaults under output/prompt_replays",
    )
    parser.add_argument(
        "--render-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild and render the Proposal context without calling an LLM",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not stream Agent tool calls and responses during a pipeline replay",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        case = load_replay_case(args.trial_dir)
        proposal_prompt = _absolute(args.proposal_prompt)
        feasibility_prompt = _absolute(args.feasibility_prompt)
        agent_config = _absolute(args.agent_config)
        base_config = _absolute(args.base_config)
        for path in (proposal_prompt, feasibility_prompt, agent_config, base_config):
            if not path.is_file():
                raise FileNotFoundError(f"required file does not exist: {path}")

        result_dir = _absolute(args.output_dir) if args.output_dir else _default_result_dir(case)
        try:
            result_dir.relative_to(case.run_dir)
        except ValueError:
            pass
        else:
            raise ValueError("--output-dir must be outside the source run directory")
        result_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "source_trial_dir": str(case.trial_dir),
            "source_run_dir": str(case.run_dir),
            "trial_id": case.trial_id,
            "proposal_prompt": {"path": str(proposal_prompt), "sha256": _prompt_digest(proposal_prompt)},
            "feasibility_prompt": {
                "path": str(feasibility_prompt),
                "sha256": _prompt_digest(feasibility_prompt),
            },
        }
        if args.render_only:
            history = load_prior_trials(case)
            orchestrator = _make_orchestrator(
                case,
                history,
                result_dir,
                proposal_prompt,
                feasibility_prompt,
                agent_config,
                base_config,
                True,
            )
            trace = render_proposal_context(case, history, orchestrator)
            rendered = _save_rendered_prompts(result_dir, trace)
            output = {
                "metadata": {
                    **metadata,
                    "prior_trial_ids": [row["trial_id"] for row in history],
                },
                "mode": "render-only",
                "rendered_prompts": rendered,
            }
            write_json(result_dir / "replay_result.json", output)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        history = load_prior_trials(case)
        orchestrator = _make_orchestrator(
            case,
            history,
            result_dir,
            proposal_prompt,
            feasibility_prompt,
            agent_config,
            base_config,
            args.quiet,
        )
        replay = replay_pipeline(case, history, orchestrator)
        original_proposal = case.decision.get("proposal")
        original_feasibility = case.decision.get("feasibility")
        if not isinstance(original_proposal, Mapping) or not isinstance(
            original_feasibility, Mapping
        ):
            raise ValueError("target decision.json lacks proposal or feasibility")
        original = {
            "trial_id": case.trial_id,
            "stage": case.report.get("stage"),
            "parameters": case.parameters,
            "proposal": original_proposal,
            "feasibility": original_feasibility,
        }
        comparison = {
            "original": _behavior(str(original["stage"]), original_proposal, original_feasibility),
            "replay": _behavior(str(replay["stage"]), replay["proposal"], replay["feasibility"]),
            "same_parameters": case.parameters == replay["parameters"],
            "same_proposal": original_proposal == replay["proposal"],
            "same_feasibility": original_feasibility == replay["feasibility"],
        }
        rendered = _save_rendered_prompts(result_dir, replay["agent_trace"])
        output = {
            "metadata": {**metadata, "prior_trial_ids": [row["trial_id"] for row in history], "rendered_prompts": rendered},
            "mode": "pipeline",
            "comparison": comparison,
            "original": original,
            "replay": replay,
        }
        write_json(result_dir / "replay_result.json", output)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"Full result: {result_dir / 'replay_result.json'}")
        return 0
    except (AgentError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
