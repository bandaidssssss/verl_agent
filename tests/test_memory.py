#!/usr/bin/env python3
"""Replay one recorded proposal through memory_estimator_V3.

This script does not call an LLM and does not start training.  It reconstructs
the candidate parameters from a recorded trial's ``proposal.changes``, exposes
only earlier trials to the estimator, and then compares the prediction with the
target trial's observed per-phase memory.

Typical use after editing the defaults near the top of this file:

    python tests/test_memory.py

Command-line values can override every frequently changed default:

    python tests/test_memory.py \
      --run-dir output/0807_1110_2026 \
      --target-trial 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_tools.memory_estimator_V3 import PHASES, estimate_phase_memory
from trial_storage import hydrate_trial


# =============================================================================
# 每次重新评估历史实验时，通常只需要修改下面 2 项
# =============================================================================

# 完整实验目录，里面必须包含当前格式的 trials.jsonl 及其 artifact。
DEFAULT_RUN_DIR = ROOT / "output" / "0819_0935_2026"

# 要重新评估哪一个已经执行过的 trial。脚本读取它的 proposal，并且只用
# trial_id 小于它的历史数据做预测；该 trial 的实测显存只用于最后对比。
DEFAULT_TARGET_TRIAL = 3


# =============================================================================
# 其他参数一般不需要修改
# =============================================================================

# None 表示读取当前 config/agent_config.json 中的显存限制。
DEFAULT_MEMORY_LIMIT_PCT: float | None = None

# None 表示只打印结果，不写文件。也可以设置成一个 JSON 文件路径。
DEFAULT_OUTPUT_JSON: Path | None = None


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
    if not history_path.is_file():
        raise FileNotFoundError(f"current-format trial index is missing: {history_path}")
    by_id: dict[int, dict[str, Any]] = {}
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
                raise ValueError(f"history row {line_number} must be a JSON object")
            trial_id = row.get("trial_id")
            if isinstance(trial_id, int) and not isinstance(trial_id, bool):
                if trial_id in by_id:
                    raise ValueError(
                        f"duplicate trial_id={trial_id} in {history_path}"
                    )
                by_id[trial_id] = row

    if not by_id:
        raise FileNotFoundError(f"no trials found in {history_path}")
    trials = [by_id[trial_id] for trial_id in sorted(by_id)]
    return [hydrate_trial(trial, history_path) for trial in trials]


def _target_report(
    trials: Sequence[Mapping[str, Any]],
    target_trial_id: int,
) -> dict[str, Any]:
    for row in trials:
        if row.get("trial_id") == target_trial_id:
            return dict(row)
    raise ValueError(f"target trial {target_trial_id} was not found")


def _proposal_targets(proposal: Mapping[str, Any]) -> dict[str, Any]:
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    targets: dict[str, Any] = {}
    for key, value in changes.items():
        if not isinstance(value, Mapping) or "to" not in value:
            raise ValueError(f"proposal changes[{key!r}] must contain from/to")
        targets[str(key)] = value["to"]
    return targets


def _proposal_change_details(proposal: Mapping[str, Any]) -> dict[str, Any]:
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in changes.items():
        if not isinstance(value, Mapping) or not {"from", "to"} <= set(value):
            raise ValueError(f"proposal changes[{key!r}] must contain from/to")
        result[str(key)] = {
            "from": value["from"],
            "to": value["to"],
            "reason": value.get("reason"),
        }
    return result


def _reference_id(proposal: Mapping[str, Any]) -> int:
    value = proposal.get("reference_trial_id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(
        "target trial proposal has no integer reference_trial_id; choose a "
        "non-baseline trial produced from an earlier empirical reference"
    )


def _memory_limit(
    explicit_limit_pct: float | None,
    reference_measurements: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, str]:
    capacities = [
        float(row["gpu_capacity_mib"])
        for row in reference_measurements.values()
        if isinstance(row.get("gpu_capacity_mib"), (int, float))
    ]
    if not capacities:
        raise ValueError("reference trial has no GPU capacity in metrics.json")
    capacity_mib = min(capacities)
    if explicit_limit_pct is not None:
        return (
            explicit_limit_pct,
            capacity_mib * explicit_limit_pct / 100.0,
            "command/default override",
        )

    config_path = ROOT / "config" / "agent_config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        reserve = config.get("throughput_memory_reserve_mib")
        if isinstance(reserve, (int, float)) and not isinstance(reserve, bool):
            limit_mib = capacity_mib - float(reserve)
            return (
                100.0 * limit_mib / capacity_mib,
                limit_mib,
                f"{config_path}: throughput_memory_reserve_mib",
            )
    raise ValueError(f"throughput_memory_reserve_mib is missing from {config_path}")


def _actual_phase_measurements(
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    structured = report.get("structured_metrics")
    resource = structured.get("resource") if isinstance(structured, Mapping) else None
    by_phase = resource.get("by_phase") if isinstance(resource, Mapping) else None
    if not isinstance(by_phase, Mapping):
        raise ValueError("trial has no current-format metrics resource.by_phase")
    result: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        row = by_phase.get(phase)
        row = row if isinstance(row, Mapping) else {}
        used_mb = row.get("max_used_mib")
        capacity_mb = row.get("max_used_gpu_total_mib")
        used_mb = float(used_mb) if isinstance(used_mb, (int, float)) else None
        capacity_mb = (
            float(capacity_mb) if isinstance(capacity_mb, (int, float)) else None
        )
        pct = (
            100.0 * used_mb / capacity_mb
            if used_mb is not None and capacity_mb is not None and capacity_mb > 0
            else None
        )
        result[phase] = {
            "memory_mb": used_mb,
            "memory_gib": used_mb / 1024.0 if used_mb is not None else None,
            "memory_pct": pct,
            "gpu_capacity_gib": (
                capacity_mb / 1024.0 if capacity_mb is not None else None
            ),
            "gpu_capacity_mib": capacity_mb,
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
            "target trial proposal has no changes to replay"
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
    memory_limit_pct: float | None = None,
) -> dict[str, Any]:
    trials = _load_trials(run_dir)
    target = _target_report(trials, target_trial_id)
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
    limit, limit_mib, limit_source = _memory_limit(
        memory_limit_pct,
        reference_measurements,
    )
    relative_estimate = estimate_phase_memory(
        reference,
        candidate,
        history,
        memory_limit_mib=limit_mib,
    )
    estimate_phases = relative_estimate.get("phases")
    if not isinstance(estimate_phases, Mapping):
        raise ValueError("memory_estimator returned no phases object")
    actual_measurements = _actual_phase_measurements(target)
    comparison: dict[str, Any] = {}
    absolute_errors_pct: list[float] = []
    signed_errors_pct: list[float] = []
    absolute_errors_gib: list[float] = []
    signed_errors_gib: list[float] = []
    actual_exceeds_limit_phases: list[str] = []
    for phase in PHASES:
        phase_estimate = estimate_phases.get(phase)
        if not isinstance(phase_estimate, Mapping):
            raise ValueError(f"memory_estimator returned no {phase} phase object")
        reference_mib = phase_estimate.get("reference_peak_mib")
        estimated_mib = phase_estimate.get("estimated_peak_mib")
        relative_change_pct = phase_estimate.get(
            "estimated_relative_change_pct"
        )
        reference_gib = (
            float(reference_mib) / 1024.0
            if isinstance(reference_mib, (int, float))
            else None
        )
        predicted_gib = (
            float(estimated_mib) / 1024.0
            if isinstance(estimated_mib, (int, float))
            else None
        )
        reference_pct = reference_measurements[phase].get("memory_pct")
        predicted_pct = (
            float(reference_pct) * (1.0 + float(relative_change_pct) / 100.0)
            if isinstance(reference_pct, (int, float))
            and isinstance(relative_change_pct, (int, float))
            else None
        )
        actual_pct = actual_measurements[phase]["memory_pct"]
        actual_gib = actual_measurements[phase]["memory_gib"]
        if (
            actual_gib is None
            and actual_pct is not None
            and isinstance(
                reference_measurements[phase].get("gpu_capacity_gib"),
                (int, float),
            )
        ):
            actual_gib = (
                float(reference_measurements[phase]["gpu_capacity_gib"])
                * float(actual_pct)
                / 100.0
            )
            actual_measurements[phase]["source"] = (
                "target_pct_times_reference_gpu_capacity"
            )
        signed_error_pct = (
            float(predicted_pct) - actual_pct
            if isinstance(predicted_pct, (int, float)) and actual_pct is not None
            else None
        )
        absolute_error_pct = (
            abs(signed_error_pct) if signed_error_pct is not None else None
        )
        signed_error_gib = (
            float(predicted_gib) - actual_gib
            if isinstance(predicted_gib, (int, float)) and actual_gib is not None
            else None
        )
        absolute_error_gib = (
            abs(signed_error_gib) if signed_error_gib is not None else None
        )
        if absolute_error_pct is not None:
            absolute_errors_pct.append(absolute_error_pct)
            signed_errors_pct.append(signed_error_pct)
        if absolute_error_gib is not None:
            absolute_errors_gib.append(absolute_error_gib)
            signed_errors_gib.append(signed_error_gib)
        if actual_gib is not None and actual_gib * 1024.0 >= limit_mib:
            actual_exceeds_limit_phases.append(phase)
        comparison[phase] = {
            "status": phase_estimate.get("status"),
            "reference_peak_gib": reference_gib,
            "estimated_delta_gib": (
                predicted_gib - reference_gib
                if predicted_gib is not None and reference_gib is not None
                else None
            ),
            "estimated_peak_gib": predicted_gib,
            "estimated_relative_change_pct": relative_change_pct,
            "actual_gib": round(actual_gib, 2) if actual_gib is not None else None,
            "signed_error_gib": (
                round(signed_error_gib, 2)
                if signed_error_gib is not None
                else None
            ),
            "absolute_error_gib": (
                round(absolute_error_gib, 2)
                if absolute_error_gib is not None
                else None
            ),
            "reference_pct": reference_pct,
            "estimated_pct": predicted_pct,
            "actual_pct": round(actual_pct, 2) if actual_pct is not None else None,
            "signed_error_pct": (
                round(signed_error_pct, 2)
                if signed_error_pct is not None
                else None
            ),
            "absolute_error_pct": (
                round(absolute_error_pct, 2)
                if absolute_error_pct is not None
                else None
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
            "memory_limit_pct": limit,
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
        "estimate": relative_estimate,
        "comparison": comparison,
        "summary": {
            "compared_phase_count": len(absolute_errors_pct),
            "mae_pct": (
                round(sum(absolute_errors_pct) / len(absolute_errors_pct), 2)
                if absolute_errors_pct
                else None
            ),
            "mean_signed_error_pct": (
                round(sum(signed_errors_pct) / len(signed_errors_pct), 2)
                if signed_errors_pct
                else None
            ),
            "max_underestimate_pct": (
                round(max([0.0, *(-value for value in signed_errors_pct)]), 2)
                if signed_errors_pct
                else None
            ),
            "mae_gib": (
                round(sum(absolute_errors_gib) / len(absolute_errors_gib), 2)
                if absolute_errors_gib
                else None
            ),
            "mean_signed_error_gib": (
                round(sum(signed_errors_gib) / len(signed_errors_gib), 2)
                if signed_errors_gib
                else None
            ),
            "max_underestimate_gib": (
                round(max([0.0, *(-value for value in signed_errors_gib)]), 2)
                if signed_errors_gib
                else None
            ),
            "actual_exceeds_limit_phases": actual_exceeds_limit_phases,
            "safety_decision_correct": (
                not actual_exceeds_limit_phases
                if relative_estimate.get("safety") == "within_limit"
                else (
                    bool(actual_exceeds_limit_phases)
                    if relative_estimate.get("safety") == "exceeds_limit"
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
        f"limit: {case['memory_limit_pct']:.2f}% "
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

    print("\nCenter prediction versus observation (GiB; percent in parentheses)")
    comparison_rows = []
    for phase in PHASES:
        item = report["comparison"][phase]
        comparison_rows.append(
            [
                phase,
                item["status"],
                _format_value(item["reference_peak_gib"]),
                _format_value(item["estimated_delta_gib"]),
                _format_value(item["estimated_peak_gib"]),
                _format_value(item["actual_gib"]),
                _format_value(item["signed_error_gib"]),
                (
                    f"{_format_value(item['estimated_pct'])}/"
                    f"{_format_value(item['actual_pct'])}"
                ),
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
            "estimate%/actual%",
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
    print(f"  phase MAE: {_format_value(summary['mae_gib'])} GiB")
    print(f"  phase MAE: {_format_value(summary['mae_pct'])} pct-points")
    print(
        "  max underestimate: "
        f"{_format_value(summary['max_underestimate_gib'])} GiB / "
        f"{_format_value(summary['max_underestimate_pct'])} pct-points"
    )
    print("  note: compact estimator output contains center predictions only")


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
            "Current-format run containing trials.jsonl; when omitted, use "
            "DEFAULT_RUN_DIR from this file"
        ),
    )
    parser.add_argument(
        "--target-trial",
        type=int,
        default=None,
        help=(
            "Executed trial whose recorded proposal should be replayed; when "
            "omitted, use DEFAULT_TARGET_TRIAL from this file"
        ),
    )
    parser.add_argument(
        "--memory-limit-pct",
        type=float,
        default=DEFAULT_MEMORY_LIMIT_PCT,
        help="Override the recorded/configured memory limit",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Optional path for the full machine-readable report",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        run_dir = _absolute(
            DEFAULT_RUN_DIR if args.run_dir is None else args.run_dir
        )
        target_trial_id = (
            DEFAULT_TARGET_TRIAL
            if args.target_trial is None
            else args.target_trial
        )
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        if target_trial_id < 1:
            raise ValueError("--target-trial must be at least 1")

        report = replay_memory_estimate(
            run_dir,
            target_trial_id,
            args.memory_limit_pct,
        )
        print_report(report)
        if args.output_json:
            output_path = _absolute(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str)
                + "\n",
                encoding="utf-8",
            )
            print(f"\nFull JSON report: {output_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"memory replay failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
