from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
STEP_RE = re.compile(r"step:(\d+)")
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
)
STABILITY_QUERY_METRICS = STABILITY_SERIES_METRICS + (
    "actor/pg_loss",
    "response_length/mean",
    "response/aborted_ratio",
    "actor/kl_loss",
    "actor/grad_norm",
)


def parse_step_records(log_path: str | Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            pairs = {key: float(value) for key, value in PAIR_RE.findall(line)}
            if pairs:
                records[int(step_match.group(1))] = pairs
    return dict(sorted(records.items()))


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
    """Return aligned, step-sorted metric windows without derived judgments.

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
    records = parse_step_records(log_path)
    phase_log_memory = parse_phase_memory_from_log(log_path)
    sampled_memory, sampled_util, sampled_memory_by_gpu = parse_gpu_samples(gpu_samples_path)
    error_type, error_evidence = detect_error(log_path)
    stable_records = {step: row for step, row in records.items() if step > warmup_updates}
    stable_rows = list(stable_records.values())
    if not stable_rows:
        stable_rows = list(records.values())

    def metric_summary(key: str) -> dict[str, float | None]:
        return _summary(row[key] for row in stable_rows if key in row)

    phase_memory = {}
    phase_duration = {}
    for phase, timing_key in TIMING_KEYS.items():
        memory_values = sampled_memory.get(phase) or phase_log_memory.get(phase) or []
        phase_memory[phase] = _summary(memory_values)
        phase_duration[phase] = metric_summary(timing_key)

    known_phase_peaks = {
        phase: values["max"] for phase, values in phase_memory.items() if values["max"] is not None
    }
    duration_means = {
        phase: values["mean"] for phase, values in phase_duration.items() if values["mean"] is not None
    }
    reward_values = [row["critic/rewards/mean"] for row in records.values() if "critic/rewards/mean" in row]
    stability_series = build_metric_windows(
        stable_records,
        STABILITY_SERIES_METRICS,
        stability_window_size,
    )
    stability_series["warmup_updates"] = warmup_updates

    return {
        "updates_completed": max(records, default=0),
        "result": "fail" if error_type else ("success" if records else "fail"),
        "error": {"type": error_type or (None if records else "NO_STEP_METRICS"), "evidence": error_evidence},
        "memory_by_phase_pct": phase_memory,
        "memory_by_phase_gpu_pct": {
            phase: {gpu: _summary(values) for gpu, values in gpu_values.items()}
            for phase, gpu_values in sampled_memory_by_gpu.items()
            if phase not in {"startup", "between_phases", "unknown"}
        },
        "gpu_utilization_by_phase_pct": {
            phase: _summary(values) for phase, values in sampled_util.items() if phase != "unknown"
        },
        "performance": {
            "throughput": metric_summary("perf/throughput"),
            "time_per_step_s": metric_summary("perf/time_per_step"),
            "generation_tgs": metric_summary("perf/tgs/gen"),
            "actor_tgs": metric_summary("perf/tgs/actor"),
            "actor_mfu": metric_summary("perf/mfu/actor"),
            "phase_duration_s": phase_duration,
            "time_bottleneck": max(duration_means, key=duration_means.get) if duration_means else None,
        },
        "resource": {
            "memory_bottleneck": max(known_phase_peaks, key=known_phase_peaks.get) if known_phase_peaks else None,
            "max_observed_memory_pct": max(known_phase_peaks.values()) if known_phase_peaks else None,
        },
        "stability": stability_series,
        "end_to_end_reward": {
            "thresholds": compute_threshold_stats(records, reward_thresholds, reward_window),
            "peak_reward": max(reward_values) if reward_values else None,
        },
    }
