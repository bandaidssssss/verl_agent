#!/usr/bin/env python3
"""Replay recorded proposals through memory_estimator_V3.

This script does not call an LLM and does not start training.  It reconstructs
each candidate from its recorded ``proposal.changes``, exposes only earlier
trials to the estimator, and compares the prediction with that trial's observed
per-phase memory.  With no ``--target-trial``, every discoverable trial is
processed and the complete terminal report is also saved as Markdown.

Typical use after editing the defaults near the top of this file:

    python tests/memory_all.py

To replay only one target trial:

    python tests/memory_all.py \
      --run-dir output/0807_1110_2026 \
      --target-trial 2
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_tools.memory_estimator_V3 import PHASES, estimate_phase_memory
from trial_storage import hydrate_trial


# =============================================================================
# 每次重新评估历史实验时，通常只需要修改下面 1 项
# =============================================================================

# 完整实验目录，里面应包含 trials.jsonl 或 trials/NNNN/trial_report.json。
DEFAULT_RUN_DIR = ROOT / "output" / "0827_1018_2026"

# =============================================================================
# 其他参数一般不需要修改
# =============================================================================

# None 表示优先读取目标 trial 当时 Agent context 中的 throughput 显存限制
# 或 reserve，其次读取 config/agent_config.json，最后使用 92%。
DEFAULT_MEMORY_LIMIT_MIB: float | None = None

# None 表示只打印结果，不写文件。也可以设置成一个 JSON 文件路径。
DEFAULT_OUTPUT_JSON: Path | None = None

# None 表示自动写入 DEFAULT_RUN_DIR/memory_replay_report.md。
DEFAULT_OUTPUT_MD: Path | None = None


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_trials(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "trials.jsonl"
    by_id: dict[int, dict[str, Any]] = {}
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {history_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"history row {line_number} must be a JSON object"
                    )
                trial_id = row.get("trial_id")
                if isinstance(trial_id, int) and not isinstance(trial_id, bool):
                    if trial_id in by_id:
                        raise ValueError(
                            f"duplicate trial_id={trial_id} in {history_path}"
                        )
                    by_id[trial_id] = row
    else:
        trials_dir = run_dir / "trials"
        for report_path in sorted(trials_dir.glob("*/trial_report.json")):
            row = _load_json(report_path)
            trial_id = row.get("trial_id")
            if isinstance(trial_id, int) and not isinstance(trial_id, bool):
                by_id[trial_id] = row

    if not by_id:
        raise FileNotFoundError(
            f"no trials found in {history_path} or {run_dir / 'trials'}"
        )
    trials = [by_id[trial_id] for trial_id in sorted(by_id)]

    # trials.jsonl stays deliberately small. The proposal, parameters,
    # structured metrics, and log facts used by V3 live in per-trial artifacts,
    # so hydrate those indexes before replaying them.
    history_path = run_dir / "trials.jsonl"
    if history_path.is_file():
        hydrated: list[dict[str, Any]] = []
        for trial in trials:
            if isinstance(trial.get("artifacts"), Mapping):
                hydrated.append(hydrate_trial(trial, history_path))
            else:
                hydrated.append(trial)
        trials = hydrated

    # ``hydrate_trial`` intentionally keeps agent traces out of the normal
    # in-memory trial shape. Replay needs the recorded constraint snapshot,
    # however, so load just that artifact here instead of silently applying the
    # current agent_config.json policy to an older proposal.
    with_traces: list[dict[str, Any]] = []
    for trial in trials:
        row = dict(trial)
        trial_id = row.get("trial_id")
        artifacts = row.get("artifacts")
        relative_trace = (
            artifacts.get("agent_trace")
            if isinstance(artifacts, Mapping)
            else None
        )
        candidates: list[Path] = []
        if isinstance(relative_trace, str):
            candidate = (run_dir / relative_trace).resolve()
            try:
                candidate.relative_to(run_dir)
            except ValueError:
                pass
            else:
                candidates.append(candidate)
        if isinstance(trial_id, int) and not isinstance(trial_id, bool):
            candidates.append(
                run_dir / "trials" / f"{trial_id:04d}" / "agent_trace.json"
            )
        trace_path = next((path for path in candidates if path.is_file()), None)
        if trace_path is not None:
            row["agent_trace"] = _load_json(trace_path)
        with_traces.append(row)
    return with_traces


def _target_report(
    run_dir: Path,
    trials: Sequence[Mapping[str, Any]],
    target_trial_id: int,
) -> dict[str, Any]:
    # Trial reports are compact summaries. ``_load_trials`` has already
    # hydrated their decision and measurement artifacts.
    for row in trials:
        if row.get("trial_id") == target_trial_id:
            return dict(row)

    # Retain a useful fallback for a legacy run discovered from artifacts.
    report_path = run_dir / "trials" / f"{target_trial_id:04d}" / "trial_report.json"
    if report_path.is_file():
        return _load_json(report_path)
    raise ValueError(f"target trial {target_trial_id} was not found in {run_dir}")


def _proposal_targets(proposal: Mapping[str, Any]) -> dict[str, Any]:
    target_changes = proposal.get("target_changes")
    if isinstance(target_changes, Mapping):
        return {str(key): value for key, value in target_changes.items()}

    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    targets: dict[str, Any] = {}
    for key, value in changes.items():
        targets[str(key)] = value.get("to") if isinstance(value, Mapping) else value
    return targets


def _proposal_change_details(proposal: Mapping[str, Any]) -> dict[str, Any]:
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in changes.items():
        if isinstance(value, Mapping):
            result[str(key)] = {
                "from": value.get("from"),
                "to": value.get("to"),
                "reason": value.get("reason"),
            }
        else:
            result[str(key)] = {"from": None, "to": value, "reason": None}
    return result


def _reference_id(proposal: Mapping[str, Any]) -> int:
    value = proposal.get("reference_trial_id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    descriptor = proposal.get("reference_trial")
    if isinstance(descriptor, Mapping):
        value = descriptor.get("trial_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise ValueError(
        "target trial proposal has no integer reference_trial_id; choose a "
        "non-baseline trial produced from an earlier empirical reference"
    )


def _memory_limit_from_report(
    report: Mapping[str, Any],
    reference_measurements: Mapping[str, Mapping[str, Any]],
    explicit_limit_mib: float | None,
) -> tuple[float, str]:
    capacities_mib = [
        float(row["gpu_capacity_mib"])
        for row in reference_measurements.values()
        if isinstance(row.get("gpu_capacity_mib"), (int, float))
        and not isinstance(row.get("gpu_capacity_mib"), bool)
    ]
    if not capacities_mib:
        raise ValueError("reference trial has no GPU capacity")
    capacity_mib = min(capacities_mib)

    if explicit_limit_mib is not None:
        return explicit_limit_mib, "command/default override"

    trace = report.get("agent_trace")
    proposal_conversation = (
        trace.get("proposal_conversation") if isinstance(trace, Mapping) else None
    )
    context = (
        proposal_conversation.get("context")
        if isinstance(proposal_conversation, Mapping)
        else None
    )
    constraints = context.get("constraints") if isinstance(context, Mapping) else None
    if isinstance(constraints, Mapping):
        for key in (
            "throughput_memory_limit_pct",
            "resource_memory_limit_pct",
        ):
            value = constraints.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limit_pct = float(value)
                return (
                    capacity_mib * limit_pct / 100.0,
                    f"recorded proposal context: {key}",
                )
        for key in (
            "throughput_memory_reserve_mib",
            "resource_memory_reserve_mib",
        ):
            value = constraints.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limit_mib = capacity_mib - float(value)
                return (
                    limit_mib,
                    f"recorded proposal context: {key}",
                )

    config_path = ROOT / "config" / "agent_config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        for key in (
            "throughput_memory_limit_pct",
            "resource_memory_limit_pct",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limit_pct = float(value)
                return (
                    capacity_mib * limit_pct / 100.0,
                    f"{config_path}: {key}",
                )
        for key in (
            "throughput_memory_reserve_mib",
            "resource_memory_reserve_mib",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limit_mib = capacity_mib - float(value)
                return (
                    limit_mib,
                    f"{config_path}: {key}",
                )
    return capacity_mib * 0.92, "fallback: 92% of GPU capacity"


def _actual_phase_measurements(
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read the persisted per-phase peaks in the estimator's native MiB unit."""

    structured = report.get("structured_metrics")
    resource = structured.get("resource") if isinstance(structured, Mapping) else None
    by_phase = resource.get("by_phase") if isinstance(resource, Mapping) else None
    if not isinstance(by_phase, Mapping):
        raise ValueError("trial has no metrics resource.by_phase")
    result: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        row = by_phase.get(phase)
        row = row if isinstance(row, Mapping) else {}
        used_mib = row.get("max_used_mib")
        capacity_mib = row.get("max_used_gpu_total_mib")
        result[phase] = {
            "memory_mib": (
                float(used_mib)
                if isinstance(used_mib, (int, float))
                and not isinstance(used_mib, bool)
                else None
            ),
            "gpu_capacity_mib": (
                float(capacity_mib)
                if isinstance(capacity_mib, (int, float))
                and not isinstance(capacity_mib, bool)
                else None
            ),
            "source": row.get("source"),
        }
    return result


def _candidate_from_proposal(
    reference_parameters: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = _proposal_targets(proposal)
    if not targets:
        raise ValueError(
            "target trial proposal has no changes/target_changes to replay"
        )

    details = _proposal_change_details(proposal)
    from_mismatches: dict[str, Any] = {}
    for key, detail in details.items():
        expected_from = detail.get("from")
        observed_from = reference_parameters.get(key)
        if expected_from != observed_from:
            from_mismatches[key] = {
                "proposal_from": expected_from,
                "reference_value": observed_from,
            }
    if from_mismatches:
        raise ValueError(
            "proposal 'from' values do not match the selected reference: "
            + json.dumps(from_mismatches, ensure_ascii=False)
        )

    candidate = dict(reference_parameters)
    candidate.update(targets)
    return candidate, details


def _parameter_comparison(
    candidate: Mapping[str, Any],
    actual_parameters: Any,
    changed_keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(actual_parameters, Mapping):
        return {
            "actual_parameters_available": False,
            "proposal_changes_match_actual": None,
            "full_candidate_matches_actual": None,
            "changed_parameter_mismatches": {},
        }
    mismatches = {
        key: {
            "proposal_candidate": candidate.get(key),
            "actual_trial": actual_parameters.get(key),
        }
        for key in changed_keys
        if candidate.get(key) != actual_parameters.get(key)
    }
    return {
        "actual_parameters_available": True,
        "proposal_changes_match_actual": not mismatches,
        "full_candidate_matches_actual": dict(candidate) == dict(actual_parameters),
        "changed_parameter_mismatches": mismatches,
    }


def replay_memory_estimate(
    run_dir: Path,
    target_trial_id: int,
    memory_limit_mib: float | None = None,
) -> dict[str, Any]:
    trials = _load_trials(run_dir)
    target = _target_report(run_dir, trials, target_trial_id)
    proposal = target.get("proposal")
    if not isinstance(proposal, Mapping):
        raise ValueError(f"target trial {target_trial_id} has no recorded proposal")

    reference_trial_id = _reference_id(proposal)
    history = [
        dict(row)
        for row in trials
        if isinstance(row.get("trial_id"), int)
        and int(row["trial_id"]) < target_trial_id
    ]
    reference = next(
        (row for row in history if row.get("trial_id") == reference_trial_id),
        None,
    )
    if reference is None:
        raise ValueError(
            f"reference trial {reference_trial_id} is not available before "
            f"target trial {target_trial_id}"
        )
    reference_parameters = reference.get("parameters")
    if not isinstance(reference_parameters, Mapping):
        raise ValueError(
            f"reference trial {reference_trial_id} has no parameter mapping"
        )

    candidate, change_details = _candidate_from_proposal(
        reference_parameters, proposal
    )
    reference_measurements = _actual_phase_measurements(reference)
    limit_mib, limit_source = _memory_limit_from_report(
        target,
        reference_measurements,
        memory_limit_mib,
    )
    estimate = estimate_phase_memory(
        reference,
        candidate,
        history,
        memory_limit_mib=limit_mib,
    )
    estimate_phases = estimate.get("phases")
    if not isinstance(estimate_phases, Mapping):
        raise ValueError("memory_estimator returned no phases object")
    actual_measurements = _actual_phase_measurements(target)
    comparison: dict[str, Any] = {}
    absolute_errors_mib: list[float] = []
    signed_errors_mib: list[float] = []
    actual_exceeds_limit_phases: list[str] = []
    for phase in PHASES:
        phase_estimate = estimate_phases.get(phase)
        if not isinstance(phase_estimate, Mapping):
            raise ValueError(f"memory_estimator returned no {phase} phase object")
        reference_mib = phase_estimate.get("reference_peak_mib")
        estimated_mib = phase_estimate.get("estimated_peak_mib")
        reference_mib = (
            float(reference_mib)
            if isinstance(reference_mib, (int, float))
            and not isinstance(reference_mib, bool)
            else None
        )
        predicted_mib = (
            float(estimated_mib)
            if isinstance(estimated_mib, (int, float))
            and not isinstance(estimated_mib, bool)
            else None
        )
        actual_mib = actual_measurements[phase]["memory_mib"]
        signed_error_mib = (
            predicted_mib - actual_mib
            if predicted_mib is not None and actual_mib is not None
            else None
        )
        absolute_error_mib = (
            abs(signed_error_mib) if signed_error_mib is not None else None
        )
        if absolute_error_mib is not None:
            absolute_errors_mib.append(absolute_error_mib)
            signed_errors_mib.append(signed_error_mib)
        if actual_mib is not None and actual_mib >= limit_mib:
            actual_exceeds_limit_phases.append(phase)
        comparison[phase] = {
            "status": phase_estimate.get("status"),
            "reference_mib": reference_mib,
            "delta_mib": (
                predicted_mib - reference_mib
                if predicted_mib is not None and reference_mib is not None
                else None
            ),
            "predicted_mib": predicted_mib,
            "actual_mib": round(actual_mib, 2) if actual_mib is not None else None,
            "signed_error_mib": (
                round(signed_error_mib, 2)
                if signed_error_mib is not None
                else None
            ),
            "absolute_error_mib": (
                round(absolute_error_mib, 2)
                if absolute_error_mib is not None
                else None
            ),
            "gpu_capacity_mib": reference_measurements[phase].get(
                "gpu_capacity_mib"
            ),
            "actual_source": actual_measurements[phase]["source"],
        }

    parameter_comparison = _parameter_comparison(
        candidate,
        target.get("parameters"),
        list(change_details),
    )
    return {
        "case": {
            "run_dir": str(run_dir),
            "target_trial_id": target_trial_id,
            "reference_trial_id": reference_trial_id,
            "history_trial_ids_exposed_to_estimator": [
                row.get("trial_id") for row in history
            ],
            "memory_limit_mib": limit_mib,
            "memory_limit_source": limit_source,
            "target_result": target.get("result"),
            "target_failure_phase": target.get("failure_phase"),
            "target_error": target.get("error"),
        },
        "proposal": {
            "candidate_id": proposal.get("candidate_id"),
            "decision": proposal.get("decision"),
            "reason": proposal.get("reason"),
            "changes": change_details,
        },
        "parameter_comparison": parameter_comparison,
        "estimate": estimate,
        "comparison": comparison,
        "summary": {
            "compared_phase_count": len(absolute_errors_mib),
            "mae_mib": (
                round(sum(absolute_errors_mib) / len(absolute_errors_mib), 2)
                if absolute_errors_mib
                else None
            ),
            "mean_signed_error_mib": (
                round(sum(signed_errors_mib) / len(signed_errors_mib), 2)
                if signed_errors_mib
                else None
            ),
            "max_underestimate_mib": (
                round(max([0.0, *(-value for value in signed_errors_mib)]), 2)
                if signed_errors_mib
                else None
            ),
            "actual_exceeds_limit_phases": actual_exceeds_limit_phases,
            "safety_decision_correct": (
                not actual_exceeds_limit_phases
                if estimate.get("safety") == "within_limit"
                else (
                    bool(actual_exceeds_limit_phases)
                    if estimate.get("safety") == "exceeds_limit"
                    else None
                )
            ),
        },
    }


def _format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{float(value):.{digits}f}"
    return str(value)


def _print_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    all_rows = [list(headers), *[list(row) for row in rows]]
    widths = [
        max(len(str(row[index])) for row in all_rows)
        for index in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        return " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row)
        )

    print(render(all_rows[0]))
    print("-+-".join("-" * width for width in widths))
    for row in all_rows[1:]:
        print(render(row))


def print_report(report: Mapping[str, Any]) -> None:
    case = report["case"]
    proposal = report["proposal"]
    print(
        f"Memory replay: {case['run_dir']} trial {case['target_trial_id']} "
        f"<- reference trial {case['reference_trial_id']}"
    )
    print(
        f"Estimator history: {case['history_trial_ids_exposed_to_estimator']} | "
        f"limit: {case['memory_limit_mib']:.2f} MiB "
        f"({case['memory_limit_source']})"
    )
    print(
        f"Recorded outcome: {case['target_result']} | "
        f"failure phase: {case['target_failure_phase']}"
    )
    print(
        f"Proposal candidate: {proposal['candidate_id']} | "
        f"decision: {proposal['decision']}"
    )

    print("\nProposal changes")
    change_rows = []
    for key, detail in proposal["changes"].items():
        change_rows.append(
            [
                key,
                repr(detail.get("from")),
                repr(detail.get("to")),
            ]
        )
    _print_table(change_rows, ("parameter", "from", "to"))

    sequence = report["estimate"].get("sequence_length", {})
    reference_sequence = sequence.get("reference", {})
    if reference_sequence:
        print(
            "\nEffective sequence length: "
            f"point={_format_value(reference_sequence.get('point_tokens'), 0)} "
            f"tokens, upper={_format_value(reference_sequence.get('upper_tokens'), 0)} "
            f"tokens ({reference_sequence.get('source')})"
        )

    print("\nCenter prediction versus observation (MiB)")
    comparison_rows = []
    for phase in PHASES:
        item = report["comparison"][phase]
        comparison_rows.append(
            [
                phase,
                str(item["status"]),
                _format_value(item["reference_mib"]),
                _format_value(item["delta_mib"]),
                _format_value(item["predicted_mib"]),
                _format_value(item["actual_mib"]),
                _format_value(item["signed_error_mib"]),
            ]
        )
    _print_table(
        comparison_rows,
        (
            "phase",
            "status",
            "ref",
            "delta",
            "estimate",
            "actual",
            "estimate-actual",
        ),
    )

    parameter_comparison = report["parameter_comparison"]
    summary = report["summary"]
    print("\nSummary")
    print(
        "  proposal changes match actual parameters: "
        f"{parameter_comparison['proposal_changes_match_actual']}"
    )
    print(
        "  full assembled candidate matches actual parameters: "
        f"{parameter_comparison['full_candidate_matches_actual']}"
    )
    print(f"  phase MAE: {_format_value(summary['mae_mib'])} MiB")
    print(
        "  max underestimate: "
        f"{_format_value(summary['max_underestimate_mib'])} MiB"
    )
    print("  note: compact estimator output contains center predictions only")


def _discover_trial_ids(run_dir: Path) -> list[int]:
    """Return complete indexed or report-backed trial IDs."""
    return sorted(
        int(row["trial_id"])
        for row in _load_trials(run_dir)
        if isinstance(row.get("trial_id"), int)
        and not isinstance(row.get("trial_id"), bool)
    )


def _render_report(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_report(report)
    return buffer.getvalue().rstrip()


def _render_batch(
    run_dir: Path,
    reports: Sequence[Mapping[str, Any]],
    skipped: Sequence[tuple[int, str]],
    output_md: Path,
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Memory replay batch: {run_dir}")
        print(f"Markdown report: {output_md}")
        print(f"Successful replays: {len(reports)} | skipped: {len(skipped)}")

        if reports:
            print("\nBatch summary")
            summary_rows = []
            for report in reports:
                case = report["case"]
                summary = report["summary"]
                summary_rows.append(
                    [
                        str(case["target_trial_id"]),
                        str(case["reference_trial_id"]),
                        str(summary["compared_phase_count"]),
                        _format_value(summary["mae_mib"]),
                        _format_value(summary["max_underestimate_mib"]),
                    ]
                )
            _print_table(
                summary_rows,
                (
                    "trial",
                    "reference",
                    "phases",
                    "MAE MiB",
                    "max under MiB",
                ),
            )

        for report in reports:
            print("\n" + "=" * 96)
            print(_render_report(report))

        if skipped:
            print("\n" + "=" * 96)
            print("Skipped trials")
            for trial_id, reason in skipped:
                print(f"  trial {trial_id}: {reason}")
    return buffer.getvalue().rstrip() + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded proposal with memory_estimator_V3 and compare "
            "the prediction against actual phase peaks"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Historical run containing trials.jsonl; when omitted, use "
            "DEFAULT_RUN_DIR from this file"
        ),
    )
    parser.add_argument(
        "--target-trial",
        type=int,
        default=None,
        help=(
            "Replay only this executed trial; when omitted, replay every trial "
            "discovered under the run directory"
        ),
    )
    parser.add_argument(
        "--memory-limit-mib",
        type=float,
        default=DEFAULT_MEMORY_LIMIT_MIB,
        help="Override the recorded/configured memory limit in MiB",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Optional path for all successful machine-readable reports",
    )
    parser.add_argument(
        "--output-md",
        default=DEFAULT_OUTPUT_MD,
        help=(
            "Markdown output path; defaults to "
            "<run-dir>/memory_replay_report.md"
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        run_dir = _absolute(
            DEFAULT_RUN_DIR if args.run_dir is None else args.run_dir
        )
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        if args.target_trial is not None and args.target_trial < 1:
            raise ValueError("--target-trial must be at least 1")

        target_trial_ids = (
            [args.target_trial]
            if args.target_trial is not None
            else _discover_trial_ids(run_dir)
        )
        reports: list[dict[str, Any]] = []
        skipped: list[tuple[int, str]] = []
        for target_trial_id in target_trial_ids:
            try:
                reports.append(
                    replay_memory_estimate(
                        run_dir,
                        target_trial_id,
                        args.memory_limit_mib,
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                skipped.append((target_trial_id, str(exc)))

        output_md = (
            _absolute(args.output_md)
            if args.output_md is not None
            else run_dir / "memory_replay_report.md"
        )
        terminal_output = _render_batch(run_dir, reports, skipped, output_md)
        print(terminal_output, end="")
        markdown = (
            "# Memory estimator replay report\n\n"
            f"Run directory: `{run_dir}`\n\n"
            "## Complete terminal output\n\n"
            "```text\n"
            f"{terminal_output}"
            "```\n"
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

        if args.output_json:
            output_path = _absolute(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {"reports": reports, "skipped": skipped},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"\nFull JSON report: {output_path}")
        return 0 if reports else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"memory replay failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
