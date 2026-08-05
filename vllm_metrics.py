from __future__ import annotations

import argparse
import csv
import math
import os
import re
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import urlopen


DISABLE_LOG_STATS_PARAMETER = "actor_rollout_ref.rollout.disable_log_stats"
ROLLOUT_ENGINE_PARAMETER = "actor_rollout_ref.rollout.name"

VLLM_METRIC_COLUMNS = (
    "timestamp",
    "replica_rank",
    "server",
    "requests_running",
    "requests_waiting",
    "kv_cache_usage_pct",
    "preemptions_delta",
    "preemptions_source",
    "prompt_tokens_per_s",
    "generation_tokens_per_s",
    "iteration_tokens_mean",
    "iteration_tokens_p95_upper_bound",
)

_SERVER_RE = re.compile(
    r"vLLMHttpServer,\s*replica_rank:\s*(?P<rank>\d+).*?"
    r"master address:\s*(?P<host>[^,\s]+),\s*master port:\s*(?P<port>\d+)",
    re.I,
)
_AGENT_MANAGER_RE = re.compile(r"AgentLoopManager:\s*\[([^\]]+)\]", re.I)
_HOST_PORT_RE = re.compile(r"(?P<host>(?:\d{1,3}\.){3}\d{1,3}|[A-Za-z0-9_.-]+):(?P<port>\d+)")
_LOG_PREEMPTION_RE = re.compile(
    r"(?:total[_ ](?:number[_ ]of[_ ])?cumulative[_ ]preemptions?|"
    r"total_cumulative_preemption)\s*[=:]\s*(\d+)",
    re.I,
)
_LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{.*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)


def vllm_metrics_enabled(parameters: Mapping[str, Any]) -> bool:
    """Enable collection only when vLLM stats were explicitly enabled."""
    engine = str(parameters.get(ROLLOUT_ENGINE_PARAMETER, "")).lower()
    return engine == "vllm" and parameters.get(DISABLE_LOG_STATS_PARAMETER) is False


def parse_prometheus_metrics(data: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = dict(_LABEL_RE.findall(match.group("labels") or ""))
        metrics.setdefault(match.group("name"), []).append((labels, value))
    return metrics


def _total(
    metrics: Mapping[str, list[tuple[dict[str, str], float]]], *names: str
) -> float | None:
    for name in names:
        samples = metrics.get(name)
        if samples:
            return sum(value for _, value in samples)
    return None


def _maximum(
    metrics: Mapping[str, list[tuple[dict[str, str], float]]], name: str
) -> float | None:
    samples = metrics.get(name)
    return max((value for _, value in samples), default=None) if samples else None


def _histogram_buckets(
    metrics: Mapping[str, list[tuple[dict[str, str], float]]], name: str
) -> dict[str, float]:
    buckets: dict[str, float] = {}
    for labels, value in metrics.get(name, []):
        boundary = labels.get("le")
        if boundary is not None:
            buckets[boundary] = buckets.get(boundary, 0.0) + value
    return buckets


def metric_snapshot(data: str) -> dict[str, Any]:
    metrics = parse_prometheus_metrics(data)
    kv_ratio = _maximum(metrics, "vllm:kv_cache_usage_perc")
    return {
        "requests_running": _total(metrics, "vllm:num_requests_running"),
        "requests_waiting": _total(metrics, "vllm:num_requests_waiting"),
        "kv_cache_usage_pct": 100.0 * kv_ratio if kv_ratio is not None else None,
        "preemptions_total": _total(
            metrics,
            "vllm:num_preemptions_total",
            "vllm:num_preemptions",
        ),
        "preemptions_source": "prometheus",
        "prompt_tokens_total": _total(metrics, "vllm:prompt_tokens_total"),
        "generation_tokens_total": _total(metrics, "vllm:generation_tokens_total"),
        "iteration_tokens_count": _total(metrics, "vllm:iteration_tokens_total_count"),
        "iteration_tokens_sum": _total(metrics, "vllm:iteration_tokens_total_sum"),
        "iteration_tokens_buckets": _histogram_buckets(
            metrics, "vllm:iteration_tokens_total_bucket"
        ),
    }


def _counter_delta(current: Any, previous: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
        return None
    delta = float(current) - float(previous)
    return delta if delta >= 0 else None


def _counter_rate(current: Any, previous: Any, elapsed: float) -> float | None:
    delta = _counter_delta(current, previous)
    return delta / elapsed if delta is not None and elapsed > 0 else None


def _histogram_p95_upper_bound(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    count_delta: float,
) -> float | None:
    if count_delta <= 0 or not current or not previous:
        return None
    target = 0.95 * count_delta
    boundaries: list[tuple[float, str]] = []
    for label in current:
        if label == "+Inf":
            continue
        try:
            boundaries.append((float(label), label))
        except ValueError:
            continue
    for boundary, label in sorted(boundaries):
        delta = _counter_delta(current.get(label), previous.get(label))
        if delta is not None and delta >= target:
            return boundary
    return None


def compact_metric_row(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    elapsed: float,
) -> dict[str, float | str | None]:
    previous = previous or {}
    count_delta = _counter_delta(
        current.get("iteration_tokens_count"), previous.get("iteration_tokens_count")
    )
    sum_delta = _counter_delta(
        current.get("iteration_tokens_sum"), previous.get("iteration_tokens_sum")
    )
    iteration_mean = (
        sum_delta / count_delta
        if sum_delta is not None and count_delta is not None and count_delta > 0
        else None
    )
    preemptions_delta = _counter_delta(
        current.get("preemptions_total"), previous.get("preemptions_total")
    )
    if (
        preemptions_delta is None
        and isinstance(current.get("preemptions_total"), (int, float))
    ):
        # This counter is cumulative since engine startup.  The automatic
        # monitor starts with the engine, so zero is the correct first baseline.
        preemptions_delta = float(current["preemptions_total"])
    return {
        "requests_running": current.get("requests_running"),
        "requests_waiting": current.get("requests_waiting"),
        "kv_cache_usage_pct": current.get("kv_cache_usage_pct"),
        "preemptions_delta": preemptions_delta,
        "preemptions_source": (
            current.get("preemptions_source") if preemptions_delta is not None else ""
        ),
        "prompt_tokens_per_s": _counter_rate(
            current.get("prompt_tokens_total"), previous.get("prompt_tokens_total"), elapsed
        ),
        "generation_tokens_per_s": _counter_rate(
            current.get("generation_tokens_total"),
            previous.get("generation_tokens_total"),
            elapsed,
        ),
        "iteration_tokens_mean": iteration_mean,
        "iteration_tokens_p95_upper_bound": _histogram_p95_upper_bound(
            current.get("iteration_tokens_buckets", {}),
            previous.get("iteration_tokens_buckets", {}),
            count_delta or 0.0,
        ),
    }


def _row_is_useful(row: Mapping[str, Any]) -> bool:
    active_fields = (
        "requests_running",
        "requests_waiting",
        "kv_cache_usage_pct",
        "preemptions_delta",
        "prompt_tokens_per_s",
        "generation_tokens_per_s",
        "iteration_tokens_mean",
    )
    return any(
        isinstance(row.get(field), (int, float)) and float(row[field]) > 0
        for field in active_fields
    )


def _format_number(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


class VLLMMetricsSampler(threading.Thread):
    """Scrape compact vLLM scheduling metrics from endpoints discovered in logs."""

    def __init__(
        self,
        output_path: Path,
        interval: float = 5.0,
        fetcher: Callable[[str, float], str] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="vllm-metrics-sampler")
        self.output_path = output_path
        self.interval = max(0.1, float(interval))
        self.fetcher = fetcher or self._fetch
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._endpoints: dict[str, str] = {}
        self._using_manager_endpoints = False
        self._log_preemptions_total: float | None = None
        self.rows_written = 0
        self.scrapes_succeeded = 0
        self.scrape_errors = 0
        self.last_sample_timestamp: float | None = None

    @staticmethod
    def _fetch(server: str, timeout: float) -> str:
        return urlopen(f"http://{server}/metrics", timeout=timeout).read().decode(
            "utf-8", "replace"
        )

    def observe_log_line(self, line: str) -> None:
        server_endpoint: tuple[str, str] | None = None
        server_match = _SERVER_RE.search(line)
        if server_match:
            server_endpoint = (
                server_match.group("rank"),
                f"{server_match.group('host')}:{server_match.group('port')}",
            )
        manager_match = _AGENT_MANAGER_RE.search(line)
        manager_endpoints: list[tuple[str, str]] = []
        if manager_match:
            for index, match in enumerate(_HOST_PORT_RE.finditer(manager_match.group(1))):
                manager_endpoints.append(
                    (str(index), f"{match.group('host')}:{match.group('port')}")
                )
        preemption_match = _LOG_PREEMPTION_RE.search(line)
        with self._lock:
            # AgentLoopManager lists the actual serving endpoints used by the
            # rollout client.  Earlier vLLMHttpServer "master port" messages
            # are a startup fallback and must not remain as duplicate scrapes.
            if manager_endpoints:
                self._endpoints = {
                    server: rank for rank, server in manager_endpoints
                }
                self._using_manager_endpoints = True
            elif server_endpoint is not None and not self._using_manager_endpoints:
                rank, server = server_endpoint
                self._endpoints[server] = rank
            if preemption_match:
                self._log_preemptions_total = float(preemption_match.group(1))

    def _endpoint_snapshot(self) -> tuple[list[tuple[str, str]], float | None]:
        with self._lock:
            endpoints = sorted(
                ((rank, server) for server, rank in self._endpoints.items()),
                key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[1]),
            )
            return endpoints, self._log_preemptions_total

    def run(self) -> None:
        previous: dict[str, tuple[float, dict[str, Any]]] = {}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=VLLM_METRIC_COLUMNS)
            writer.writeheader()
            handle.flush()
            while not self.stop_event.is_set():
                endpoints, log_preemptions = self._endpoint_snapshot()
                for endpoint_index, (rank, server) in enumerate(endpoints):
                    now = time.time()
                    try:
                        data = self.fetcher(server, min(10.0, max(1.0, self.interval)))
                        current = metric_snapshot(data)
                        if (
                            current.get("preemptions_total") is None
                            and endpoint_index == 0
                            and log_preemptions is not None
                        ):
                            current["preemptions_total"] = log_preemptions
                            current["preemptions_source"] = "train_log"
                        old = previous.get(server)
                        elapsed = now - old[0] if old else self.interval
                        row = compact_metric_row(current, old[1] if old else None, elapsed)
                        previous[server] = (now, current)
                        self.scrapes_succeeded += 1
                        if not _row_is_useful(row):
                            continue
                        writer.writerow(
                            {
                                "timestamp": f"{now:.6f}",
                                "replica_rank": rank,
                                "server": server,
                                **{key: _format_number(value) for key, value in row.items()},
                            }
                        )
                        handle.flush()
                        self.rows_written += 1
                        self.last_sample_timestamp = now
                    except (OSError, URLError, ValueError):
                        self.scrape_errors += 1
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        endpoints, _ = self._endpoint_snapshot()
        return {
            "enabled": True,
            "interval_seconds": self.interval,
            "endpoints": [
                {"replica_rank": rank, "server": server} for rank, server in endpoints
            ],
            "rows_written": self.rows_written,
            "scrapes_succeeded": self.scrapes_succeeded,
            "scrape_errors": self.scrape_errors,
            "last_sample_timestamp": self.last_sample_timestamp,
        }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def summarize_vllm_metrics(path: str | Path | None) -> dict[str, Any]:
    metric_fields = (
        "requests_running",
        "requests_waiting",
        "kv_cache_usage_pct",
        "prompt_tokens_per_s",
        "generation_tokens_per_s",
        "iteration_tokens_mean",
        "iteration_tokens_p95_upper_bound",
    )
    values: dict[str, list[float]] = {field: [] for field in metric_fields}
    preemption_values: list[float] = []
    replicas: set[str] = set()
    rows = 0
    target = Path(path) if path else None
    if target is None or not target.is_file():
        return {
            "available": False,
            "samples": 0,
            "missing_metrics": list(metric_fields) + ["preemptions_delta"],
        }
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            replicas.add(row.get("replica_rank", ""))
            for field in metric_fields:
                raw = row.get(field, "")
                if raw not in (None, ""):
                    try:
                        values[field].append(float(raw))
                    except ValueError:
                        pass
            raw_preemptions = row.get("preemptions_delta", "")
            if raw_preemptions not in (None, ""):
                try:
                    preemption_values.append(float(raw_preemptions))
                except ValueError:
                    pass
    missing = [field for field, field_values in values.items() if not field_values]
    if not preemption_values:
        missing.append("preemptions_delta")
    waiting_values = values["requests_waiting"]
    return {
        "available": rows > 0,
        "samples": rows,
        "replicas": len({value for value in replicas if value != ""}),
        **{field: _summary(field_values) for field, field_values in values.items()},
        "waiting_positive_fraction": (
            sum(value > 0 for value in waiting_values) / len(waiting_values)
            if waiting_values
            else None
        ),
        "preemptions_total": sum(preemption_values) if preemption_values else None,
        "missing_metrics": missing,
    }


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _summary_value(summary: Mapping[str, Any], field: str, statistic: str) -> float | None:
    value = summary.get(field)
    return _numeric(value.get(statistic)) if isinstance(value, Mapping) else None


def assess_rollout_metrics(
    summary: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    rollout_memory_peak_pct: float | None = None,
    rollout_gpu_util_mean_pct: float | None = None,
    memory_limit_pct: float = 92.0,
) -> dict[str, Any]:
    running_p95 = _summary_value(summary, "requests_running", "p95")
    running_max = _summary_value(summary, "requests_running", "max")
    waiting_fraction = _numeric(summary.get("waiting_positive_fraction"))
    waiting_max = _summary_value(summary, "requests_waiting", "max")
    kv_p95 = _summary_value(summary, "kv_cache_usage_pct", "p95")
    kv_max = _summary_value(summary, "kv_cache_usage_pct", "max")
    preemptions = _numeric(summary.get("preemptions_total"))
    iteration_p95 = _summary_value(summary, "iteration_tokens_p95_upper_bound", "p95")
    memory_headroom = (
        memory_limit_pct - rollout_memory_peak_pct
        if rollout_memory_peak_pct is not None
        else None
    )
    queue_present = bool((waiting_fraction or 0.0) >= 0.05 and (waiting_max or 0.0) > 0)
    memory_pressure = bool(
        (kv_p95 is not None and kv_p95 >= 90.0)
        or (kv_max is not None and kv_max >= 98.0)
        or (preemptions is not None and preemptions > 0)
        or (memory_headroom is not None and memory_headroom < 5.0)
    )

    max_seqs = _numeric(parameters.get("actor_rollout_ref.rollout.max_num_seqs"))
    seq_ratio = running_p95 / max_seqs if running_p95 is not None and max_seqs else None
    seq_binding = bool(seq_ratio is not None and seq_ratio >= 0.95 and queue_present)
    if seq_binding and not memory_pressure:
        seq_status = "binding_consider_increase_one_step"
    elif seq_binding:
        seq_status = "binding_but_memory_pressure_blocks_increase"
    else:
        seq_status = "not_demonstrated"

    max_tokens = _numeric(
        parameters.get("actor_rollout_ref.rollout.max_num_batched_tokens")
    )
    token_ratio = iteration_p95 / max_tokens if iteration_p95 is not None and max_tokens else None
    token_binding = bool(token_ratio is not None and token_ratio >= 0.90 and queue_present)
    if iteration_p95 is None:
        token_status = "unknown_metric_not_exported"
    elif token_binding and not memory_pressure:
        token_status = "binding_consider_increase_one_step"
    elif token_binding:
        token_status = "binding_but_memory_pressure_blocks_increase"
    else:
        token_status = "not_demonstrated"

    configured_memory = _numeric(
        parameters.get("actor_rollout_ref.rollout.gpu_memory_utilization")
    )
    if kv_p95 is None:
        memory_status = "unknown_kv_metric_not_exported"
    elif (preemptions is not None and preemptions > 0) or kv_p95 >= 90.0:
        if (
            memory_headroom is not None
            and memory_headroom >= 5.0
            and configured_memory is not None
            and configured_memory < 0.95
        ):
            memory_status = "kv_capacity_binding_consider_small_increase"
        else:
            memory_status = "kv_capacity_binding_without_safe_physical_headroom"
    elif (
        kv_p95 < 70.0
        and (kv_max is None or kv_max < 85.0)
        and not (preemptions and preemptions > 0)
    ):
        memory_status = "not_binding_do_not_raise_for_low_compute_utilization"
    else:
        memory_status = "watch_no_clear_direction"

    return {
        "available": bool(summary.get("available")),
        "observed": {
            "rollout_gpu_compute_utilization_mean_pct": rollout_gpu_util_mean_pct,
            "rollout_physical_memory_peak_pct": rollout_memory_peak_pct,
            "physical_memory_headroom_to_limit_pct": memory_headroom,
            "requests_running_max": running_max,
            "requests_running_p95": running_p95,
            "requests_waiting_max": waiting_max,
            "waiting_positive_fraction": waiting_fraction,
            "kv_cache_usage_p95_pct": kv_p95,
            "kv_cache_usage_max_pct": kv_max,
            "preemptions_total": preemptions,
            "iteration_tokens_p95_upper_bound": iteration_p95,
            "missing_metrics": summary.get("missing_metrics", []),
        },
        "knobs": {
            "actor_rollout_ref.rollout.gpu_memory_utilization": {
                "configured": configured_memory,
                "status": memory_status,
                "binding_evidence": "KV-cache usage/preemption plus physical memory headroom",
            },
            "actor_rollout_ref.rollout.max_num_seqs": {
                "configured": max_seqs,
                "running_to_cap_ratio": seq_ratio,
                "status": seq_status,
                "binding_evidence": "running P95 approaches the cap while waiting remains positive",
            },
            "actor_rollout_ref.rollout.max_num_batched_tokens": {
                "configured": max_tokens,
                "iteration_p95_to_cap_ratio": token_ratio,
                "status": token_status,
                "binding_evidence": "iteration-token P95 approaches the cap while waiting remains positive",
            },
        },
        "guardrails": [
            "GPU compute utilization is an outcome; gpu_memory_utilization is a vLLM memory-budget fraction.",
            "Do not raise a scheduler ceiling unless its own binding evidence is present.",
            "Change only one of these three rollout capacity knobs per comparison trial.",
            "After a change, compare rollout duration, generation throughput, KV pressure, preemptions, and end-to-end throughput.",
            "A missing exporter metric means unknown, not zero.",
        ],
    }


def _follow_train_log(train_log: Path, sampler: VLLMMetricsSampler) -> None:
    position = 0
    while True:
        if train_log.exists():
            size = train_log.stat().st_size
            if size < position:
                position = 0
            with train_log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                for line in handle:
                    sampler.observe_log_line(line)
                position = handle.tell()
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect compact vLLM rollout metrics")
    parser.add_argument("train_log", type=Path)
    parser.add_argument("output_file", nargs="?", type=Path)
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("VLLM_METRICS_INTERVAL_SECONDS", "5")),
    )
    args = parser.parse_args()
    output = args.output_file or args.train_log.with_name("vllm_metrics.csv")
    sampler = VLLMMetricsSampler(output, args.interval)
    sampler.start()
    try:
        _follow_train_log(args.train_log, sampler)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        sampler.join(timeout=5)


if __name__ == "__main__":
    main()
