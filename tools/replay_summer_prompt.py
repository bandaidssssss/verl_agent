#!/usr/bin/env python3
"""Build and optionally run a Summer prompt from one recorded output run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_tools import ToolRegistry
from agents import AgentError, LLMRoleAgent
from config_utils import load_json, write_json
from prompt_context import select_stage_metrics
from trial_storage import hydrate_trial, read_trial_indexes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "output"
DEFAULT_PROMPT = ROOT / "prompts" / "summer.md"
DEFAULT_AGENT_CONFIG = ROOT / "config" / "agent_config.json"
TUNING_STAGES = {
    "hardware_tuning": ("hardware", "hardware"),
    "hardware_repair": ("hardware", "hardware"),
    "stability_tuning": ("stability", "stability"),
}


def _portable_identifier(value: Any, marker: str) -> Any:
    """Remove host-specific path prefixes while preserving dataset/model identity."""
    if isinstance(value, list):
        return [_portable_identifier(item, marker) for item in value]
    if not isinstance(value, str):
        return copy.deepcopy(value)
    normalized = value.replace("\\", "/")
    path_marker = f"/{marker.strip('/')}/"
    if path_marker in normalized:
        return normalized.rsplit(path_marker, 1)[1]
    if normalized.startswith("/"):
        return Path(normalized).name
    return value


def _run_context(run_path: Path, trials: list[Mapping[str, Any]]) -> dict[str, Any]:
    anchor = next(
        (trial for trial in trials if trial.get("source") == "reference_only"),
        trials[0] if trials else {},
    )
    parameters = anchor.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    platform = next(
        (trial.get("platform") for trial in trials if trial.get("platform") is not None),
        None,
    )
    return {
        "run_id": run_path.name,
        "algorithm": copy.deepcopy(parameters.get("algorithm.adv_estimator")),
        "model": _portable_identifier(
            parameters.get("actor_rollout_ref.model.path"), "model"
        ),
        "train_dataset": _portable_identifier(
            parameters.get("data.train_files"), "data"
        ),
        "evaluation_dataset": _portable_identifier(
            parameters.get("data.val_files"), "data"
        ),
        "platform": copy.deepcopy(platform),
        "workload": {
            "train_batch_size": copy.deepcopy(
                parameters.get("data.train_batch_size")
            ),
            "max_prompt_length": copy.deepcopy(
                parameters.get("data.max_prompt_length")
            ),
            "max_response_length": copy.deepcopy(
                parameters.get("data.max_response_length")
            ),
            "n_gpus_per_node": copy.deepcopy(
                parameters.get("trainer.n_gpus_per_node")
            ),
            "nnodes": copy.deepcopy(parameters.get("trainer.nnodes")),
        },
    }


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.resolve()


def resolve_run_dir(output_root: str | Path, date: str) -> Path:
    """Resolve an exact run directory name or one unambiguous date prefix."""
    root = _absolute(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"output root does not exist: {root}")
    if not isinstance(date, str) or not date.strip() or Path(date).name != date:
        raise ValueError("--date must be a run directory name or unambiguous name prefix")
    exact = root / date
    if (exact / "trials.jsonl").is_file():
        return exact.resolve()
    matches = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(date)
        and (path / "trials.jsonl").is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"no run under {root} matches date/name prefix {date!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"date/name prefix {date!r} is ambiguous: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def _trial_source(
    proposal: Mapping[str, Any], changes: Mapping[str, Any]
) -> str:
    if proposal.get("source") == "orchestrator" or proposal.get("decision") == "baseline":
        return "reference_only"
    if proposal.get("decision") == "modify" and changes:
        return "agent"
    return "reference_only"


def _compact_error(trial: Mapping[str, Any]) -> dict[str, Any] | None:
    structured = trial.get("structured_metrics")
    structured_error = (
        structured.get("error") if isinstance(structured, Mapping) else None
    )
    raw = structured_error if isinstance(structured_error, Mapping) else trial.get("error")
    if not isinstance(raw, Mapping):
        return None
    result = {
        key: copy.deepcopy(raw.get(key))
        for key in ("type", "failure_phase", "evidence")
        if raw.get(key) not in (None, [], {})
    }
    failure_phase = trial.get("failure_phase")
    if failure_phase is not None and "failure_phase" not in result:
        result["failure_phase"] = failure_phase
    return result or None


def build_summer_context(run_dir: str | Path) -> dict[str, Any]:
    """Build a deduplicated, stage-specific fact graph for one output run."""
    run_path = _absolute(run_dir)
    history_path = run_path / "trials.jsonl"
    indexes = read_trial_indexes(history_path)
    seen_ids: set[int] = set()
    trials: list[dict[str, Any]] = []
    hydrated_trials: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index in indexes:
        trial_id = index.get("trial_id")
        if not isinstance(trial_id, int) or isinstance(trial_id, bool):
            raise ValueError("every history row must contain an integer trial_id")
        if trial_id in seen_ids:
            raise ValueError(f"duplicate trial_id={trial_id} in {history_path}")
        seen_ids.add(trial_id)
        stage = index.get("stage")
        if stage not in TUNING_STAGES:
            continue
        stage_group, metric_stage = TUNING_STAGES[str(stage)]
        trial = hydrate_trial(index, history_path)
        proposal = trial.get("proposal")
        proposal = proposal if isinstance(proposal, Mapping) else {}
        changes = trial.get("changes")
        changes = changes if isinstance(changes, Mapping) else {}
        structured = trial.get("structured_metrics")
        structured = structured if isinstance(structured, Mapping) else {}
        metrics, missing_metrics = select_stage_metrics(structured, metric_stage)
        source = _trial_source(proposal, changes)

        row: dict[str, Any] = {
            "trial_id": trial_id,
            "reference_trial_id": index.get("reference_trial_id"),
            "stage": stage,
            "stage_group": stage_group,
            "source": source,
            "result": index.get("result"),
            "updates_target": index.get("updates_target"),
            "updates_completed": index.get("updates_completed"),
            "updates_executed": index.get("updates_executed"),
            "agent_hypothesis": (
                {
                    "reason": proposal.get("reason"),
                    "expected_effect": copy.deepcopy(
                        proposal.get("expected_effect", {})
                    ),
                }
                if source == "agent"
                else None
            ),
            "changes": copy.deepcopy(changes),
            "metrics": metrics,
            "resource": copy.deepcopy(index.get("resource", {})),
            "error": _compact_error(trial),
        }
        if missing_metrics:
            row["missing_metrics"] = missing_metrics
        termination = trial.get("termination")
        if isinstance(termination, Mapping) and termination:
            row["termination"] = copy.deepcopy(dict(termination))
        trials.append(row)
        hydrated_trials.append({**trial, "source": source})

    included_ids = {row["trial_id"] for row in trials}
    for row in trials:
        reference_id = row.get("reference_trial_id")
        if reference_id is not None and reference_id not in included_ids:
            warnings.append(
                f"trial {row['trial_id']} references trial {reference_id}, which has no "
                "hardware/stability fact row in this run"
            )

    return {
        "run_id": run_path.name,
        "run_context": _run_context(run_path, hydrated_trials),
        "stage_objectives": {
            "hardware": (
                "Judge whether the attempted direction improved end-to-end throughput "
                "without violating recorded resource safety. Phase metrics explain the result."
            ),
            "stability": (
                "Judge final usefulness by evaluation metrics. Step/window reward, KL, "
                "entropy, loss, clip, and learning-rate metrics explain the trajectory."
            ),
        },
        "trial_graph_rule": (
            "Each trial appears once with metrics for its own stage. Resolve comparisons "
            "through reference_trial_id within this trials array."
        ),
        "trials": trials,
        "warnings": warnings,
    }


def _prompt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally run the tool-free Summer summary for one dated output run."
        )
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Exact output run directory name or an unambiguous prefix such as 0903_1333",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--agent-config", default=DEFAULT_AGENT_CONFIG)
    parser.add_argument(
        "--output-dir",
        help="Result directory inside the selected run; defaults to summer_replays/<timestamp>",
    )
    parser.add_argument(
        "--render-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the fact context and rendered prompt without calling the LLM",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not stream the Summer Agent response",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_dir = resolve_run_dir(args.output_root, args.date)
        prompt_path = _absolute(args.prompt)
        config_path = _absolute(args.agent_config)
        for path in (prompt_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"required file does not exist: {path}")

        if args.output_dir:
            result_dir = _absolute(args.output_dir)
            try:
                result_dir.relative_to(run_dir)
            except ValueError as exc:
                raise ValueError("--output-dir must be inside the selected run") from exc
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_dir = run_dir / "summer_replays" / stamp
        result_dir.mkdir(parents=True, exist_ok=False)

        context = build_summer_context(run_dir)
        config = load_json(config_path)
        config["stream_agent_events"] = not args.quiet
        registry = ToolRegistry(ROOT, config, run_dir / "trials.jsonl")
        agent = LLMRoleAgent("summer", prompt_path, registry, config)
        conversation = agent.new_conversation({"trial": context})
        conversation.messages[1]["content"] = (
            "Summarize the supplied run facts and output exactly the required JSON object."
        )
        rendered_prompt = str(conversation.messages[0]["content"])

        write_json(result_dir / "summer_context.json", context)
        (result_dir / "rendered_summer.md").write_text(
            rendered_prompt + "\n", encoding="utf-8"
        )
        metadata = {
            "source_run_dir": str(run_dir),
            "run_id": run_dir.name,
            "prompt": {
                "path": str(prompt_path),
                "sha256": _prompt_digest(prompt_path),
            },
            "agent_config": str(config_path),
            "agent_trial_ids": [
                row["trial_id"]
                for row in context["trials"]
                if row.get("source") == "agent"
            ],
        }
        if args.render_only:
            output = {"metadata": metadata, "mode": "render-only"}
            write_json(result_dir / "summer_replay.json", output)
            print(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"Rendered prompt: {result_dir / 'rendered_summer.md'}")
            return 0

        run = agent.run(conversation=conversation)
        result = {
            **copy.deepcopy(run.result),
            "run_context": copy.deepcopy(context["run_context"]),
        }
        write_json(result_dir / "summer_result.json", result)
        trace = run.as_trace()
        trace["result"] = copy.deepcopy(result)
        write_json(result_dir / "summer_trace.json", trace)
        output = {
            "metadata": metadata,
            "mode": "llm",
            "result_path": str(result_dir / "summer_result.json"),
            "trace_path": str(result_dir / "summer_trace.json"),
        }
        write_json(result_dir / "summer_replay.json", output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (AgentError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
