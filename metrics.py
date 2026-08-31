from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from tools.extract_log_facts import LogFactsAccumulator


NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
STEP_RE = re.compile(r"(?<![\w/])step:(\d+)")
MEMORY_RE = re.compile(
    rf"(?P<when>Before|After) (?P<name>generate_sequences|compute_log_prob|compute_ref_log_prob|update_actor|rollout offload),.*?"
    rf"device memory used/total \(GB\): (?P<used>{NUMBER})/(?P<total>{NUMBER})"
)
FATAL_PATTERNS = {
    "OOM": re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError", re.I),
    "NCCL_OR_DISTRIBUTED_FAILURE": re.compile(
        r"ChildFailedError|DistBackendError|connection reset|"
        r"NCCL[^\n]{0,80}(?:WARN|ERROR|unhandled|failed|failure)|"
        r"(?:WARN|ERROR)[^\n]{0,80}NCCL",
        re.I,
    ),
    "NAN_OR_INF": re.compile(r"\b(?:nan|inf)\b.*(?:loss|gradient|reward)|(?:loss|gradient|reward).*\b(?:nan|inf)\b", re.I),
}
PHASE_NAMES = {
    "generate_sequences": "rollout",
    # C550 / MetaX verl logs report the rollout boundary as an offload event.
    "rollout offload": "rollout",
    "compute_log_prob": "actor_log_prob",
    "compute_ref_log_prob": "ref_log_prob",
    "update_actor": "training",
}
TIMING_KEYS = {
    "rollout": "timing_s/gen",
    "actor_log_prob": "timing_s/old_log_prob",
    "ref_log_prob": "timing_s/ref",
    "training": "timing_s/update_actor",
}

# These are the primary time-series signals supplied to a proposal agent.  The
# names intentionally match the train-log keys so the same names can be passed
# to the bounded ``read_trial_metrics`` tool for a more detailed follow-up.
STABILITY_SERIES_METRICS = (
    "critic/rewards/mean",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/entropy",
    "actor/lr",
    "response_length/clip_ratio",
    "actor/kl_loss",
    "actor/grad_norm",
)
STABILITY_QUERY_METRICS = STABILITY_SERIES_METRICS + (
    "actor/pg_loss",
    "response_length/mean",
    "response/aborted_ratio",
)
THROUGHPUT_STEP_METRICS = (
    "perf/throughput",
    "perf/time_per_step",
    "perf/total_num_tokens",
    "perf/tgs/gen",
    "perf/tgs/actor",
    "perf/mfu/actor",
    *TIMING_KEYS.values(),
)
MATH_EVALUATION_METRIC = (
    "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"
)
EVALUATION_METRICS = (MATH_EVALUATION_METRIC,)


def parse_step_records(log_path: str | Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            pairs = {key: float(value) for key, value in PAIR_RE.findall(line)}
            if pairs:
                records.setdefault(int(step_match.group(1)), {}).update(pairs)
    return dict(sorted(records.items()))


def parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    match = STEP_RE.search(line)
    if not match:
        return None
    pairs = {key: float(value) for key, value in PAIR_RE.findall(line)}
    return (int(match.group(1)), pairs) if pairs else None


def parse_train_log(
    log_path: str | Path,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract step, phase-memory, and fatal-error facts in one file pass."""
    records: dict[int, dict[str, float]] = {}
    phase_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    error_type: str | None = None
    error_evidence: list[str] = []
    log_facts_accumulator = (
        LogFactsAccumulator(parameters, log_path)
        if isinstance(parameters, Mapping)
        else None
    )
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if log_facts_accumulator is not None:
                log_facts_accumulator.consume(line)
            parsed = parse_step_line(line)
            if parsed is not None:
                step, values = parsed
                # verl may log training and validation metrics on separate lines
                # with the same global step.  Merge them so a later validation
                # line does not discard the training metrics already collected.
                records.setdefault(step, {}).update(values)
            memory_match = MEMORY_RE.search(line)
            if memory_match:
                used = float(memory_match.group("used")) * 1024.0
                total = float(memory_match.group("total")) * 1024.0
                if total > 0:
                    phase_rows[PHASE_NAMES[memory_match.group("name")]].append(
                        (used, total)
                    )
            for label, pattern in FATAL_PATTERNS.items():
                if pattern.search(line):
                    error_type = error_type or label
                    if len(error_evidence) < 5:
                        error_evidence.append(line.strip()[:500])
    phase_memory: dict[str, Any] = {}
    for phase, rows in phase_rows.items():
        used_values = [used for used, _ in rows]
        max_used, max_total = max(rows, key=lambda row: row[0])
        phase_memory[phase] = {
            "mean_used_mib": mean(used_values),
            "p95_used_mib": _percentile(used_values, 0.95),
            "max_used_mib": max_used,
            "max_used_gpu_index": None,
            "max_used_gpu_total_mib": max_total,
            "min_free_mib": max_total - max_used,
            "sample_count": len(rows),
            "per_gpu_max_used_mib": {},
            "source": "train.log:device_memory_used_total",
        }
    return {
        "records": dict(sorted(records.items())),
        "phase_memory": phase_memory,
        "error_type": error_type,
        "error_evidence": error_evidence,
        "log_facts": (
            log_facts_accumulator.finalize(records)
            if log_facts_accumulator is not None
            else {}
        ),
    }


def metric_steps(
    records: Mapping[int, Mapping[str, float]], metrics: Iterable[str]
) -> list[dict[str, Any]]:
    allowed = set(metrics)
    return [
        {
            "step": step,
            "metrics": {
                key: value for key, value in row.items() if key in allowed
            },
        }
        for step, row in sorted(records.items())
        if any(key in allowed for key in row)
    ]


def records_from_metric_steps(steps: Any) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    if not isinstance(steps, list):
        return result
    for item in steps:
        if not isinstance(item, Mapping):
            continue
        step = item.get("step")
        values = item.get("metrics")
        if not isinstance(step, int) or isinstance(step, bool) or not isinstance(values, Mapping):
            continue
        result[step] = {
            str(key): float(value)
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    return dict(sorted(result.items()))


def summarize_health_events(path: str | Path | None) -> dict[str, Any]:
    """Compact health-event JSONL without copying decisions or traces."""
    target = Path(path) if path else None
    if target is None or not target.is_file():
        return {"available": False, "record_count": 0, "record_types": {}}
    counts: dict[str, int] = defaultdict(int)
    last_decision: dict[str, Any] | None = None
    warnings: list[str] = []
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_number}: {exc.msg}")
                continue
            if not isinstance(row, Mapping):
                warnings.append(f"line {line_number}: record is not an object")
                continue
            record_type = str(row.get("record_type", "unknown"))
            counts[record_type] += 1
            if record_type in {"agent_decision", "agent_error", "stop_applied"}:
                last_decision = {
                    key: row.get(key)
                    for key in (
                        "record_type",
                        "event_id",
                        "snapshot_step",
                        "step",
                        "verdict",
                        "action",
                        "confidence",
                        "reason_codes",
                    )
                    if key in row
                }
    return {
        "available": True,
        "record_count": sum(counts.values()),
        "record_types": dict(sorted(counts.items())),
        "last_decision": last_decision,
        "warnings": warnings,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    rows = list(values)
    return {
        "mean": mean(rows) if rows else None,
        "p95": _percentile(rows, 0.95),
        "max": max(rows) if rows else None,
    }


def build_metric_windows(
    records: Mapping[int, Mapping[str, float]],
    metrics: Iterable[str],
    window_size: int,
    *,
    start_step: int | None = None,
    end_step: int | None = None,
) -> dict[str, Any]:
    """返回指定窗口大小以及step开始以及结尾的metric mean。
    Return aligned, step-sorted metric windows without derived judgments.

    A window contains consecutive observed updates rather than an assumed
    numeric step range.  This keeps the output correct when a log has missing
    steps, while its start/end fields make the covered range explicit.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    metric_names = tuple(dict.fromkeys(str(metric) for metric in metrics))
    selected = [
        (step, row)
        for step, row in sorted(records.items())
        if (start_step is None or step >= start_step) and (end_step is None or step <= end_step)
    ]
    windows: list[dict[str, int]] = []
    values_by_metric: dict[str, list[float | None]] = {metric: [] for metric in metric_names}
    for offset in range(0, len(selected), window_size):
        group = selected[offset : offset + window_size]
        windows.append(
            {
                "start_step": group[0][0],
                "end_step": group[-1][0],
                "sample_count": len(group),
            }
        )
        for metric in metric_names:
            values = [row[metric] for _, row in group if metric in row]
            values_by_metric[metric].append(mean(values) if values else None)
    return {
        "step_range": [selected[0][0], selected[-1][0]] if selected else [None, None],
        "window_size": window_size,
        "windows": windows,
        "metrics": values_by_metric,
    }


def build_terminal_metric_window(
    records: Mapping[int, Mapping[str, float]],
    metrics: Iterable[str],
    window_size: int,
) -> dict[str, Any]:
    """Return the mean over the final observed updates, aligned from the end."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    metric_names = tuple(dict.fromkeys(str(metric) for metric in metrics))
    selected = sorted(records.items())[-window_size:]
    terminal_metrics: dict[str, float | None] = {}
    for metric in metric_names:
        values = [row[metric] for _, row in selected if metric in row]
        terminal_metrics[metric] = mean(values) if values else None
    return {
        "terminal_window": (
            {
                "start_step": selected[0][0],
                "end_step": selected[-1][0],
                "sample_count": len(selected),
            }
            if selected
            else None
        ),
        "terminal_metrics": terminal_metrics,
    }


def parse_phase_memory_from_log(log_path: str | Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = MEMORY_RE.search(line)
            if match:
                total = float(match.group("total"))
                if total > 0:
                    phase = PHASE_NAMES[match.group("name")]
                    values[phase].append(100.0 * float(match.group("used")) / total)
    return values


def parse_phase_memory_absolute_from_log(log_path: str | Path) -> dict[str, Any]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = MEMORY_RE.search(line)
            if not match:
                continue
            used = float(match.group("used")) * 1024.0
            total = float(match.group("total")) * 1024.0
            if total > 0:
                values[PHASE_NAMES[match.group("name")]].append((used, total))
    result: dict[str, Any] = {}
    for phase, rows in values.items():
        used_values = [used for used, _ in rows]
        max_used, max_total = max(rows, key=lambda row: row[0])
        result[phase] = {
            "mean_used_mib": mean(used_values),
            "p95_used_mib": _percentile(used_values, 0.95),
            "max_used_mib": max_used,
            "max_used_gpu_index": None,
            "max_used_gpu_total_mib": max_total,
            "min_free_mib": max_total - max_used,
            "sample_count": len(rows),
            "per_gpu_max_used_mib": {},
            "source": "train.log:device_memory_used_total",
        }
    return result


def parse_gpu_samples(
    csv_path: str | Path | None,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    memory: dict[str, list[float]] = defaultdict(list)
    utilization: dict[str, list[float]] = defaultdict(list)
    memory_by_gpu: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not csv_path or not Path(csv_path).exists():
        return memory, utilization, memory_by_gpu
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = row.get("phase") or "unknown"
            total = float(row["memory_total_mb"])
            if total > 0:
                percentage = 100.0 * float(row["memory_used_mb"]) / total
                memory[phase].append(percentage)
                memory_by_gpu[phase][row["gpu_index"]].append(percentage)
            utilization[phase].append(float(row["utilization_pct"]))
    return memory, utilization, memory_by_gpu


def parse_gpu_samples_absolute(csv_path: str | Path | None) -> dict[str, Any]:
    """Summarize SMI device-used memory in MiB without losing GPU identity."""
    used_by_phase: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    util_by_phase: dict[str, list[float]] = defaultdict(list)
    totals_by_gpu: dict[str, float] = {}
    rows = 0
    skipped_rows = 0
    warnings: list[str] = []
    target = Path(csv_path) if csv_path else None
    if target is None or not target.is_file():
        return {
            "devices": [],
            "by_phase": {},
            "utilization_by_phase_pct": {},
            "samples": 0,
            "warnings": ["gpu_samples.csv is unavailable"],
        }
    try:
        with target.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    gpu = str(row["gpu_index"])
                    phase = str(row.get("phase") or "unknown")
                    used = float(row["memory_used_mb"])
                    total = float(row["memory_total_mb"])
                    utilization = float(row["utilization_pct"])
                except (KeyError, TypeError, ValueError):
                    skipped_rows += 1
                    continue
                if total <= 0 or used < 0:
                    skipped_rows += 1
                    continue
                rows += 1
                totals_by_gpu[gpu] = total
                used_by_phase[phase].append((gpu, used, total))
                util_by_phase[phase].append(utilization)
    except OSError as exc:
        warnings.append(str(exc))
    if skipped_rows:
        warnings.append(f"ignored {skipped_rows} malformed GPU sample rows")

    by_phase: dict[str, Any] = {}
    for phase, values in used_by_phase.items():
        if phase in {"startup", "between_phases", "unknown"}:
            continue
        used_values = [used for _, used, _ in values]
        per_gpu: dict[str, list[float]] = defaultdict(list)
        for gpu, used, _ in values:
            per_gpu[gpu].append(used)
        max_gpu, max_used, max_total = max(values, key=lambda item: item[1])
        per_gpu_max = {gpu: max(gpu_rows) for gpu, gpu_rows in sorted(per_gpu.items())}
        min_free_gpu = min(
            per_gpu_max,
            key=lambda gpu: totals_by_gpu[gpu] - per_gpu_max[gpu],
        )
        by_phase[phase] = {
            "mean_used_mib": mean(used_values),
            "p95_used_mib": _percentile(used_values, 0.95),
            "max_used_mib": max_used,
            "max_used_gpu_index": max_gpu,
            "max_used_gpu_total_mib": max_total,
            "min_free_mib": totals_by_gpu[min_free_gpu] - per_gpu_max[min_free_gpu],
            "min_free_gpu_index": min_free_gpu,
            "sample_count": len(values),
            "per_gpu_max_used_mib": per_gpu_max,
            "source": "gpu_samples.csv",
        }
    return {
        "devices": [
            {"gpu_index": gpu, "total_memory_mib": total}
            for gpu, total in sorted(totals_by_gpu.items())
        ],
        "by_phase": by_phase,
        "utilization_by_phase_pct": {
            phase: _summary(values)
            for phase, values in util_by_phase.items()
            if phase != "unknown"
        },
        "samples": rows,
        "warnings": warnings,
    }


def _resource_section(
    absolute: Mapping[str, Any],
    *,
    expected_gpu_count: int | None,
    resource_reserve_mib: float,
    throughput_reserve_mib: float,
    resource_gate_updates: int,
    monitor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    devices = list(absolute.get("devices", []))
    by_phase = dict(absolute.get("by_phase", {}))
    observed_count = len(devices)
    expected = expected_gpu_count if expected_gpu_count is not None else observed_count
    resource_limits = {
        str(device["gpu_index"]): float(device["total_memory_mib"]) - resource_reserve_mib
        for device in devices
    }
    throughput_limits = {
        str(device["gpu_index"]): float(device["total_memory_mib"]) - throughput_reserve_mib
        for device in devices
    }
    phase_peaks = [
        (phase, values)
        for phase, values in by_phase.items()
        if isinstance(values, Mapping)
        and isinstance(values.get("max_used_mib"), (int, float))
    ]
    bottleneck_phase = None
    bottleneck: Mapping[str, Any] | None = None
    if phase_peaks:
        bottleneck_phase, bottleneck = max(
            phase_peaks, key=lambda item: float(item[1]["max_used_mib"])
        )
    max_used = float(bottleneck["max_used_mib"]) if bottleneck else None
    totals_by_gpu = {
        str(device["gpu_index"]): float(device["total_memory_mib"])
        for device in devices
    }
    peaks_by_gpu: dict[str, float] = {}
    for values in by_phase.values():
        if not isinstance(values, Mapping):
            continue
        for gpu, used in (values.get("per_gpu_max_used_mib") or {}).items():
            if isinstance(used, (int, float)) and not isinstance(used, bool):
                peaks_by_gpu[str(gpu)] = max(peaks_by_gpu.get(str(gpu), 0.0), float(used))
    free_by_gpu = {
        gpu: totals_by_gpu[gpu] - used
        for gpu, used in peaks_by_gpu.items()
        if gpu in totals_by_gpu
    }
    worst_gpu = min(free_by_gpu, key=free_by_gpu.get) if free_by_gpu else None
    min_free = free_by_gpu.get(worst_gpu) if worst_gpu is not None else None
    coverage_complete = observed_count == expected and observed_count > 0
    resource_exceeded = (
        min_free < resource_reserve_mib if min_free is not None else None
    )
    throughput_exceeded = (
        min_free < throughput_reserve_mib if min_free is not None else None
    )
    monitor_data = dict(monitor or {})
    resource_safe = (
        not resource_exceeded if coverage_complete and resource_exceeded is not None else None
    )
    throughput_safe = (
        not throughput_exceeded
        if coverage_complete and throughput_exceeded is not None
        else None
    )
    return {
        "unit": "MiB",
        "monitor": {
            "source": monitor_data.get("executable"),
            "platform": monitor_data.get("platform"),
            "sampling_scope": "local_node",
            "expected_gpu_count": expected,
            "observed_gpu_count": observed_count,
            "coverage_complete": coverage_complete,
            "samples_written": monitor_data.get("samples_written", absolute.get("samples", 0)),
            "sample_errors": monitor_data.get("sample_errors", 0),
            "warnings": list(absolute.get("warnings", [])),
        },
        "devices": devices,
        "policy": {
            "resource_reserve_mib": resource_reserve_mib,
            "throughput_reserve_mib": throughput_reserve_mib,
            "resource_gate_after_update": resource_gate_updates,
            "effective_resource_limit_mib_by_gpu": resource_limits,
            "effective_throughput_limit_mib_by_gpu": throughput_limits,
        },
        "by_phase": by_phase,
        "utilization_by_phase_pct": dict(
            absolute.get("utilization_by_phase_pct", {})
        ),
        "summary": {
            "memory_bottleneck_phase": bottleneck_phase,
            "memory_bottleneck_gpu_index": worst_gpu,
            "max_used_mib": max_used,
            "min_free_mib": min_free,
            "headroom_to_resource_reserve_mib": (
                min_free - resource_reserve_mib if min_free is not None else None
            ),
            "headroom_to_throughput_reserve_mib": (
                min_free - throughput_reserve_mib if min_free is not None else None
            ),
            "resource_limit_exceeded": resource_exceeded,
            "throughput_limit_exceeded": throughput_exceeded,
            "resource_safe": resource_safe,
            "throughput_safe": throughput_safe,
        },
    }


def build_structured_metrics(
    log_path: str | Path,
    gpu_samples_path: str | Path | None,
    *,
    warmup_updates: int = 5,
    reward_window: int = 5,
    reward_thresholds: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    stability_window_size: int = 5,
    evaluation_metrics: Iterable[str] = EVALUATION_METRICS,
    vllm_summary: Mapping[str, Any] | None = None,
    vllm_metrics_path: str | Path | None = None,
    health_events_path: str | Path | None = None,
    expected_gpu_count: int | None = None,
    resource_reserve_mib: float = 0.0,
    throughput_reserve_mib: float = 0.0,
    resource_gate_updates: int = 1,
    monitor: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    status: str = "final",
) -> dict[str, Any]:
    parsed_log = parse_train_log(log_path, parameters)
    records = parsed_log["records"]
    stable_records = {step: row for step, row in records.items() if step > warmup_updates}
    stable_rows = list(stable_records.values()) or list(records.values())

    def metric_summary(key: str) -> dict[str, float | None]:
        return _summary(row[key] for row in stable_rows if key in row)

    phase_duration = {
        phase: metric_summary(key) for phase, key in TIMING_KEYS.items()
    }
    duration_means = {
        phase: values["mean"]
        for phase, values in phase_duration.items()
        if values["mean"] is not None
    }
    stability = build_metric_windows(
        stable_records, STABILITY_QUERY_METRICS, stability_window_size
    )
    stability.update(
        build_terminal_metric_window(
            stable_records, STABILITY_QUERY_METRICS, stability_window_size
        )
    )
    stability["warmup_updates"] = warmup_updates
    stability["steps"] = metric_steps(records, STABILITY_QUERY_METRICS)
    stability["window_metrics"] = stability.pop("metrics")
    health_summary = summarize_health_events(health_events_path)
    stability["health"] = health_summary

    absolute = parse_gpu_samples_absolute(gpu_samples_path)
    if not absolute.get("by_phase"):
        log_phases = parsed_log["phase_memory"]
        if log_phases:
            absolute = {
                **absolute,
                "by_phase": log_phases,
                "warnings": [
                    warning
                    for warning in absolute.get("warnings", [])
                    if warning != "gpu_samples.csv is unavailable"
                ]
                + ["phase memory was recovered from train.log"],
            }
    resource = _resource_section(
        absolute,
        expected_gpu_count=expected_gpu_count,
        resource_reserve_mib=resource_reserve_mib,
        throughput_reserve_mib=throughput_reserve_mib,
        resource_gate_updates=resource_gate_updates,
        monitor=monitor,
    )
    error_type = parsed_log["error_type"]
    error_evidence = parsed_log["error_evidence"]
    reward_values = [
        row["critic/rewards/mean"]
        for row in records.values()
        if "critic/rewards/mean" in row
    ]
    evaluation_steps = metric_steps(records, evaluation_metrics)
    latest_evaluation = (
        dict(evaluation_steps[-1]["metrics"])
        if evaluation_steps
        else {}
    )
    return {
        "status": status,
        "latest_step": max(records, default=0),
        "source": {
            "parser_version": 1,
            "train_log": Path(log_path).name,
            "gpu_samples": Path(gpu_samples_path).name if gpu_samples_path else None,
            "vllm_metrics": Path(vllm_metrics_path).name if vllm_metrics_path else None,
            "health_events": Path(health_events_path).name if health_events_path else None,
            "warnings": list(absolute.get("warnings", []))
            + [f"health_events.jsonl: {warning}" for warning in health_summary.get("warnings", [])],
        },
        "throughput": {
            "summary": {
                "throughput": metric_summary("perf/throughput"),
                "time_per_step_s": metric_summary("perf/time_per_step"),
                "total_num_tokens": metric_summary("perf/total_num_tokens"),
                "generation_tgs": metric_summary("perf/tgs/gen"),
                "actor_tgs": metric_summary("perf/tgs/actor"),
                "actor_mfu": metric_summary("perf/mfu/actor"),
                "time_bottleneck": (
                    max(duration_means, key=duration_means.get)
                    if duration_means
                    else None
                ),
            },
            "phase_duration_s": phase_duration,
            "steps": metric_steps(records, THROUGHPUT_STEP_METRICS),
            "vllm": dict(vllm_summary or {}),
        },
        "stability": stability,
        "evaluation": {
            "steps": evaluation_steps,
            "latest_metrics": latest_evaluation,
        },
        "resource": resource,
        "end_to_end_reward": {
            "thresholds": compute_threshold_stats(records, reward_thresholds, reward_window),
            "peak_reward": max(reward_values) if reward_values else None,
        },
        "error": {
            "type": error_type or (None if records else "NO_STEP_METRICS"),
            "evidence": error_evidence,
        },
        "log_facts": parsed_log["log_facts"],
    }


def build_running_metrics(
    records: Mapping[int, Mapping[str, float]],
    *,
    snapshot_step: int,
    resource_snapshot: Mapping[str, Any] | None = None,
    expected_gpu_count: int | None = None,
    resource_reserve_mib: float = 0.0,
    throughput_reserve_mib: float = 0.0,
) -> dict[str, Any]:
    """Build the bounded live subset needed by read_current_trial_metrics."""
    visible = {
        step: dict(row)
        for step, row in records.items()
        if step <= snapshot_step
    }
    evaluation_steps = metric_steps(visible, EVALUATION_METRICS)
    snapshot = dict(resource_snapshot or {})
    gpu_rows = [row for row in snapshot.get("gpus", []) if isinstance(row, Mapping)]
    devices = [
        {
            "gpu_index": str(row.get("gpu_index")),
            "total_memory_mib": row.get("memory_total_mb"),
        }
        for row in gpu_rows
    ]
    free_values = [
        float(row["memory_total_mb"]) - float(row["memory_used_mb"])
        for row in gpu_rows
        if isinstance(row.get("memory_total_mb"), (int, float))
        and isinstance(row.get("memory_used_mb"), (int, float))
    ]
    coverage = bool(devices) and (
        expected_gpu_count is None or len(devices) == expected_gpu_count
    )
    min_free = min(free_values) if free_values else None
    resource_exceeded = min_free < resource_reserve_mib if min_free is not None else None
    throughput_exceeded = (
        min_free < throughput_reserve_mib if min_free is not None else None
    )
    resource = {
        "unit": "MiB",
        "monitor": {
            "source": snapshot.get("executable"),
            "platform": snapshot.get("platform"),
            "expected_gpu_count": expected_gpu_count,
            "observed_gpu_count": len(devices),
            "coverage_complete": coverage,
            "samples_written": snapshot.get("samples_written", 0),
            "sample_errors": snapshot.get("sample_errors", 0),
        },
        "devices": devices,
        "policy": {
            "resource_reserve_mib": resource_reserve_mib,
            "throughput_reserve_mib": throughput_reserve_mib,
            "effective_resource_limit_mib_by_gpu": {
                str(row["gpu_index"]): float(row["total_memory_mib"])
                - resource_reserve_mib
                for row in devices
                if isinstance(row.get("total_memory_mib"), (int, float))
            },
            "effective_throughput_limit_mib_by_gpu": {
                str(row["gpu_index"]): float(row["total_memory_mib"])
                - throughput_reserve_mib
                for row in devices
                if isinstance(row.get("total_memory_mib"), (int, float))
            },
        },
        "by_phase": {},
        "summary": {
            "min_free_mib": min_free,
            "resource_limit_exceeded": resource_exceeded,
            "throughput_limit_exceeded": throughput_exceeded,
            "resource_safe": (
                not resource_exceeded
                if coverage and resource_exceeded is not None
                else None
            ),
            "throughput_safe": (
                not throughput_exceeded
                if coverage and throughput_exceeded is not None
                else None
            ),
        },
    }
    return {
        "status": "running",
        "latest_step": max(visible, default=0),
        "source": {"parser_version": 1, "train_log": "train.log", "warnings": []},
        "throughput": {
            "summary": {},
            "phase_duration_s": {},
            "steps": metric_steps(visible, THROUGHPUT_STEP_METRICS),
            "vllm": {},
        },
        "stability": {
            "steps": metric_steps(visible, STABILITY_QUERY_METRICS),
            "windows": [],
            "window_metrics": {},
            "terminal_window": None,
            "terminal_metrics": {},
        },
        "evaluation": {
            "steps": evaluation_steps,
            "latest_metrics": (
                dict(evaluation_steps[-1]["metrics"])
                if evaluation_steps
                else {}
            ),
        },
        "resource": resource,
    }


def legacy_metrics_from_structured(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the old in-memory report shape while new artifacts stay classified."""
    throughput = metrics.get("throughput", {})
    summary = throughput.get("summary", {}) if isinstance(throughput, Mapping) else {}
    resource = metrics.get("resource", {})
    by_phase = resource.get("by_phase", {}) if isinstance(resource, Mapping) else {}
    memory_pct: dict[str, Any] = {}
    memory_gpu_pct: dict[str, Any] = {}
    for phase, values in by_phase.items() if isinstance(by_phase, Mapping) else []:
        if not isinstance(values, Mapping):
            continue
        total = values.get("max_used_gpu_total_mib")
        scale = 100.0 / float(total) if isinstance(total, (int, float)) and total else None
        memory_pct[phase] = {
            "mean": float(values["mean_used_mib"]) * scale
            if scale is not None and isinstance(values.get("mean_used_mib"), (int, float))
            else None,
            "p95": float(values["p95_used_mib"]) * scale
            if scale is not None and isinstance(values.get("p95_used_mib"), (int, float))
            else None,
            "max": float(values["max_used_mib"]) * scale
            if scale is not None and isinstance(values.get("max_used_mib"), (int, float))
            else None,
        }
        per_gpu = values.get("per_gpu_max_used_mib")
        if isinstance(per_gpu, Mapping):
            totals = {
                str(row.get("gpu_index")): row.get("total_memory_mib")
                for row in resource.get("devices", [])
                if isinstance(row, Mapping)
            }
            memory_gpu_pct[phase] = {
                str(gpu): {
                    "mean": None,
                    "p95": None,
                    "max": (
                        100.0 * float(used) / float(totals[str(gpu)])
                        if str(gpu) in totals and totals[str(gpu)]
                        else None
                    ),
                }
                for gpu, used in per_gpu.items()
            }
    stability = dict(metrics.get("stability", {}))
    stability.pop("steps", None)
    stability["metrics"] = stability.pop("window_metrics", {})
    resource_summary = resource.get("summary", {}) if isinstance(resource, Mapping) else {}
    known_pct = {
        phase: row.get("max")
        for phase, row in memory_pct.items()
        if isinstance(row, Mapping) and isinstance(row.get("max"), (int, float))
    }
    recorded_result = metrics.get("result")
    result = (
        str(recorded_result)
        if recorded_result in {"success", "fail", "early_stopped"}
        else (
            "fail"
            if metrics.get("error", {}).get("type")
            else ("success" if metrics.get("latest_step", 0) else "fail")
        )
    )
    return {
        "updates_completed": int(metrics.get("latest_step", 0) or 0),
        "result": result,
        "error": dict(metrics.get("error", {})),
        "memory_by_phase_pct": memory_pct,
        "memory_by_phase_gpu_pct": memory_gpu_pct,
        "memory_by_phase_mib": dict(by_phase) if isinstance(by_phase, Mapping) else {},
        "gpu_utilization_by_phase_pct": dict(
            resource.get("utilization_by_phase_pct", {})
        ),
        "performance": {
            **(dict(summary) if isinstance(summary, Mapping) else {}),
            "phase_duration_s": dict(throughput.get("phase_duration_s", {}))
            if isinstance(throughput, Mapping)
            else {},
        },
        "resource": {
            **(
                dict(resource_summary)
                if isinstance(resource_summary, Mapping)
                else {}
            ),
            "memory_bottleneck": resource_summary.get("memory_bottleneck_phase")
            if isinstance(resource_summary, Mapping)
            else None,
            "max_observed_memory_pct": max(known_pct.values()) if known_pct else None,
            "monitor_coverage_complete": (
                resource.get("monitor", {}).get("coverage_complete")
                if isinstance(resource.get("monitor"), Mapping)
                else None
            ),
            "policy": dict(resource.get("policy", {}))
            if isinstance(resource.get("policy"), Mapping)
            else {},
        },
        "stability": stability,
        "evaluation": dict(metrics.get("evaluation", {})),
        "end_to_end_reward": dict(metrics.get("end_to_end_reward", {})),
    }


def detect_error(log_path: str | Path) -> tuple[str | None, list[str]]:
    evidence: list[str] = []
    detected = None
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for label, pattern in FATAL_PATTERNS.items():
                if pattern.search(line):
                    detected = detected or label
                    if len(evidence) < 12:
                        evidence.append(line.strip()[-800:])
    return detected, evidence


def _recent_slope(records: Mapping[int, Mapping[str, float]], key: str, window: int) -> float | None:
    points = [(step, row[key]) for step, row in records.items() if key in row]
    if len(points) < 2:
        return None
    recent = points[-min(window, len(points)) :]
    delta_step = recent[-1][0] - recent[0][0]
    return (recent[-1][1] - recent[0][1]) / delta_step if delta_step else 0.0


def compute_threshold_stats(
    records: Mapping[int, Mapping[str, float]], thresholds: Iterable[float], window: int = 5
) -> dict[str, dict[str, float | int] | None]:
    steps = [step for step, row in sorted(records.items()) if "critic/rewards/mean" in row]
    result: dict[str, dict[str, float | int] | None] = {str(value): None for value in thresholds}
    if not steps:
        return result
    cumulative_time = 0.0
    cumulative_tokens = 0
    totals: list[tuple[float, int]] = []
    for step in steps:
        row = records[step]
        cumulative_time += row.get("perf/time_per_step", row.get("timing_s/step", 0.0))
        cumulative_tokens += int(row.get("perf/total_num_tokens", 0))
        totals.append((cumulative_time, cumulative_tokens))
    radius = window // 2
    for index, step in enumerate(steps):
        start = max(0, index - radius)
        end = min(len(steps), index + radius + 1)
        reward = mean(records[steps[pos]]["critic/rewards/mean"] for pos in range(start, end))
        for threshold in thresholds:
            key = str(threshold)
            if result[key] is None and reward >= threshold:
                result[key] = {
                    "step": step,
                    "cumulative_time_s": totals[index][0],
                    "cumulative_tokens": totals[index][1],
                    "moving_average_reward": reward,
                }
    return result


def analyze_trial(
    log_path: str | Path,
    gpu_samples_path: str | Path | None,
    warmup_updates: int = 5,
    reward_window: int = 5,
    reward_thresholds: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    stability_window_size: int = 5,
) -> dict[str, Any]:
    structured = build_structured_metrics(
        log_path,
        gpu_samples_path,
        warmup_updates=warmup_updates,
        reward_window=reward_window,
        reward_thresholds=reward_thresholds,
        stability_window_size=stability_window_size,
    )
    return legacy_metrics_from_structured(structured)
