#!/usr/bin/env python3
"""Replay one recorded proposal through memory_estimator_V2.

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
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_tools.memory_estimator_V2 import PHASES, estimate_phase_memory


# =============================================================================
# 每次重新评估历史实验时，通常只需要修改下面 2 项
# =============================================================================

# 完整实验目录，里面应包含 trials.jsonl 或 trials/NNNN/trial_report.json。
DEFAULT_RUN_DIR = ROOT / "output" / "0807_1110_2026"

# 要重新评估哪一个已经执行过的 trial。脚本读取它的 proposal，并且只用
# trial_id 小于它的历史数据做预测；该 trial 的实测显存只用于最后对比。
DEFAULT_TARGET_TRIAL = 2


# =============================================================================
# 其他参数一般不需要修改
# =============================================================================

# None 表示优先读取目标 trial 当时 Agent context 中的 throughput 显存限制，
# 其次读取 config/agent_config.json，最后使用 92%。
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
    return [by_id[trial_id] for trial_id in sorted(by_id)]


def _target_report(
    run_dir: Path,
    trials: Sequence[Mapping[str, Any]],
    target_trial_id: int,
) -> dict[str, Any]:
    # The per-trial report usually contains a richer Agent trace, so prefer it
    # over the compact trials.jsonl row when it exists.
    report_path = (
        run_dir
        / "trials"
        / f"{target_trial_id:04d}"
        / "trial_report.json"
    )
    if report_path.is_file():
        return _load_json(report_path)
    for row in trials:
        if row.get("trial_id") == target_trial_id:
            return dict(row)
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
    report: Mapping[str, Any], explicit_limit: float | None
) -> tuple[float, str]:
    if explicit_limit is not None:
        return explicit_limit, "command/default override"

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
                return float(value), f"recorded proposal context: {key}"

    config_path = ROOT / "config" / "agent_config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        for key in (
            "throughput_memory_limit_pct",
            "resource_memory_limit_pct",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), f"{config_path}: {key}"
    return 92.0, "fallback"


def _phase_actual_peaks_pct(report: Mapping[str, Any]) -> dict[str, float]:
    memory = report.get("memory_by_phase_pct")
    if not isinstance(memory, Mapping):
        return {}
    result: dict[str, float] = {}
    for phase in PHASES:
        value = memory.get(phase)
        if isinstance(value, Mapping):
            value = value.get("max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[phase] = float(value)
    return result


def _actual_phase_measurements(
    run_dir: Path,
    target_trial_id: int,
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read target-trial peaks in both GiB and percent.

    ``memory_by_phase_pct`` in trial.jsonl is the authoritative phase summary.
    The raw CSV supplies GPU capacity so that percentage peaks can be converted
    to GiB; its per-phase used values are fallback data only.
    """

    pct_peaks = _phase_actual_peaks_pct(report)
    sample_candidates = [
        run_dir / "trials" / f"{target_trial_id:04d}" / "gpu_samples.csv"
    ]
    recorded = report.get("gpu_samples_path")
    if isinstance(recorded, str) and recorded:
        recorded_path = Path(recorded).expanduser()
        if recorded_path.is_file():
            sample_candidates.insert(0, recorded_path)
    samples_path = next((path for path in sample_candidates if path.is_file()), None)

    used_by_phase: dict[str, list[float]] = {phase: [] for phase in PHASES}
    totals: list[float] = []
    if samples_path is not None:
        with samples_path.open("r", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                phase = row.get("phase")
                if phase not in used_by_phase:
                    continue
                try:
                    used_mb = float(row["memory_used_mb"])
                    total_mb = float(row["memory_total_mb"])
                except (KeyError, TypeError, ValueError):
                    continue
                if used_mb >= 0 and total_mb > 0:
                    used_by_phase[phase].append(used_mb)
                    totals.append(total_mb)

    capacity_mb = sorted(totals)[len(totals) // 2] if totals else None
    result: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        pct = pct_peaks.get(phase)
        raw_csv_peak_mb = (
            max(used_by_phase[phase]) if used_by_phase[phase] else None
        )
        if pct is not None and capacity_mb is not None:
            used_mb = capacity_mb * pct / 100.0
            source = "trial.memory_by_phase_pct.max * gpu_samples.capacity"
        else:
            used_mb = raw_csv_peak_mb
            source = (
                "gpu_samples.raw_phase_peak_fallback"
                if raw_csv_peak_mb is not None
                else "trial_pct_without_capacity"
            )
        if pct is None and used_mb is not None and capacity_mb is not None:
            pct = 100.0 * used_mb / capacity_mb
        result[phase] = {
            "memory_mb": used_mb,
            "memory_gib": used_mb / 1024.0 if used_mb is not None else None,
            "memory_pct": pct,
            "gpu_capacity_gib": (
                capacity_mb / 1024.0 if capacity_mb is not None else None
            ),
            "source": source,
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
    memory_limit_pct: float | None = None,
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
    limit, limit_source = _memory_limit_from_report(target, memory_limit_pct)
    estimate = estimate_phase_memory(
        reference_parameters,
        candidate,
        history,
        limit,
        reference_trial_id,
    )
    if estimate.get("reference_trial_id") != reference_trial_id:
        raise ValueError(
            f"reference trial {reference_trial_id} has no phase memory observations"
        )

    actual_measurements = _actual_phase_measurements(
        run_dir, target_trial_id, target
    )
    comparison: dict[str, Any] = {}
    absolute_errors_pct: list[float] = []
    signed_errors_pct: list[float] = []
    absolute_errors_gib: list[float] = []
    signed_errors_gib: list[float] = []
    false_safe_phases: list[str] = []
    upper_bound_misses: list[str] = []
    for phase in PHASES:
        phase_estimate = estimate["phases"][phase]
        predicted_pct = phase_estimate.get("projected_pct")
        upper_pct = phase_estimate.get("upper_bound_pct")
        predicted_gib = phase_estimate.get("projected_memory_gib")
        upper_gib = phase_estimate.get("upper_bound_memory_gib")
        actual_pct = actual_measurements[phase]["memory_pct"]
        actual_gib = actual_measurements[phase]["memory_gib"]
        if (
            actual_gib is None
            and actual_pct is not None
            and isinstance(phase_estimate.get("gpu_capacity_gib"), (int, float))
        ):
            actual_gib = (
                float(phase_estimate["gpu_capacity_gib"])
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
        within_upper = (
            actual_gib <= float(upper_gib)
            if isinstance(upper_gib, (int, float)) and actual_gib is not None
            else (
                actual_pct <= float(upper_pct)
                if isinstance(upper_pct, (int, float)) and actual_pct is not None
                else None
            )
        )
        if absolute_error_pct is not None:
            absolute_errors_pct.append(absolute_error_pct)
            signed_errors_pct.append(signed_error_pct)
        if absolute_error_gib is not None:
            absolute_errors_gib.append(absolute_error_gib)
            signed_errors_gib.append(signed_error_gib)
        if within_upper is False:
            upper_bound_misses.append(phase)
        if (
            actual_pct is not None
            and actual_pct >= limit
            and isinstance(upper_pct, (int, float))
            and float(upper_pct) < limit
        ):
            false_safe_phases.append(phase)
        drivers = phase_estimate.get("drivers", {})
        affected_components = (
            drivers.get("affected_components")
            if isinstance(drivers, Mapping)
            else None
        )
        comparison[phase] = {
            "reference_gib": phase_estimate.get("reference_memory_gib"),
            "delta_gib": phase_estimate.get("delta_memory_gib"),
            "predicted_gib": predicted_gib,
            "upper_bound_gib": upper_gib,
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
            "gpu_capacity_gib": phase_estimate.get("gpu_capacity_gib"),
            "reference_pct": phase_estimate.get("reference_pct"),
            "delta_pct": phase_estimate.get("delta_pct"),
            "predicted_pct": predicted_pct,
            "upper_bound_pct": upper_pct,
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
            "actual_within_upper_bound": within_upper,
            "predicted_risk": phase_estimate.get("risk"),
            "confidence": phase_estimate.get("confidence"),
            "model": phase_estimate.get("model"),
            "affected_components": affected_components or [],
            "calibration": (
                drivers.get("calibration")
                if isinstance(drivers, Mapping)
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
            "upper_bound_miss_phases": upper_bound_misses,
            "false_safe_phases": false_safe_phases,
            "safe_decision_correct": not false_safe_phases,
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

    print("\nPrediction versus observation (GiB; percent in parentheses)")
    comparison_rows = []
    for phase in PHASES:
        item = report["comparison"][phase]
        comparison_rows.append(
            [
                phase,
                _format_value(item["reference_gib"]),
                _format_value(item["delta_gib"]),
                _format_value(item["predicted_gib"]),
                _format_value(item["upper_bound_gib"]),
                _format_value(item["actual_gib"]),
                _format_value(item["signed_error_gib"]),
                (
                    f"{_format_value(item['predicted_pct'])}/"
                    f"{_format_value(item['actual_pct'])}"
                ),
                _format_value(item["actual_within_upper_bound"]),
                str(item["predicted_risk"]),
                ",".join(item["affected_components"]) or "unchanged",
            ]
        )
    _print_table(
        comparison_rows,
        (
            "phase",
            "ref",
            "delta",
            "pred",
            "upper",
            "actual",
            "pred-actual",
            "pred%/actual%",
            "covered",
            "risk",
            "components",
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
    print(f"  upper-bound misses: {summary['upper_bound_miss_phases']}")
    print(f"  false-safe phases: {summary['false_safe_phases']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded proposal with memory_estimator_V2 and compare "
            "the prediction against actual phase peaks"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="Historical run containing trials.jsonl",
    )
    parser.add_argument(
        "--target-trial",
        type=int,
        default=DEFAULT_TARGET_TRIAL,
        help="Executed trial whose recorded proposal should be replayed",
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_dir = _absolute(args.run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        if args.target_trial < 1:
            raise ValueError("--target-trial must be at least 1")

        report = replay_memory_estimate(
            run_dir,
            args.target_trial,
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
