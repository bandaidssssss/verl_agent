from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from config_utils import load_json, read_jsonl
from metrics import legacy_metrics_from_structured


INDEX_SCHEMA_VERSION = 2


def trial_artifacts(trial_id: int) -> dict[str, str]:
    prefix = f"trials/{trial_id:04d}"
    return {
        "report": f"{prefix}/trial_report.json",
        "parameters": f"{prefix}/parameters.json",
        "parameter_groups": f"{prefix}/parameter_groups.json",
        "metrics": f"{prefix}/metrics.json",
        "log_facts": f"{prefix}/log_facts.json",
        "decision": f"{prefix}/decision.json",
        "agent_trace": f"{prefix}/agent_trace.json",
        "command": f"{prefix}/command.json",
        "log": f"{prefix}/train.log",
        "gpu_samples": f"{prefix}/gpu_samples.csv",
        "vllm_metrics": f"{prefix}/vllm_metrics.csv",
        "health_events": f"{prefix}/health_events.jsonl",
        "health_agent_traces": f"{prefix}/health_agent_traces.jsonl",
    }


def resolve_artifact(history_path: str | Path, relative_path: str) -> Path:
    root = Path(history_path).expanduser().resolve().parent
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("trial artifact is outside the configured output directory") from exc
    return target


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def _mean_metric(report: Mapping[str, Any], *path: str) -> float | None:
    value: Any = report
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _terminal_reward(report: Mapping[str, Any]) -> float | None:
    value = report.get("stability")
    if not isinstance(value, Mapping):
        return None
    terminal = value.get("terminal_metrics")
    reward = terminal.get("critic/rewards/mean") if isinstance(terminal, Mapping) else None
    return float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None


def build_trial_index(
    report: Mapping[str, Any],
    *,
    stability_healthy: bool | None = None,
) -> dict[str, Any]:
    trial_id = int(report["trial_id"])
    proposal = report.get("proposal")
    proposal = proposal if isinstance(proposal, Mapping) else {}
    resource = report.get("resource")
    resource = resource if isinstance(resource, Mapping) else {}
    error = report.get("error")
    error = error if isinstance(error, Mapping) else {}
    artifacts = trial_artifacts(trial_id)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "trial_id": trial_id,
        "stage": report.get("stage"),
        "result": report.get("result"),
        "updates_target": report.get("updates_target"),
        "updates_completed": report.get("updates_completed"),
        "reference_trial_id": proposal.get("reference_trial_id"),
        "changes": copy.deepcopy(proposal.get("changes", {})),
        "scores": {
            "throughput_mean": _mean_metric(report, "performance", "throughput"),
            "time_per_step_mean_s": _mean_metric(report, "performance", "time_per_step_s"),
            "terminal_reward": _terminal_reward(report),
            "stability_healthy": stability_healthy,
        },
        "resource": {
            "memory_bottleneck_phase": resource.get("memory_bottleneck_phase", resource.get("memory_bottleneck")),
            "max_used_mib": resource.get("max_used_mib"),
            "min_free_mib": resource.get("min_free_mib"),
            "resource_limit_exceeded": resource.get("resource_limit_exceeded"),
            "throughput_limit_exceeded": resource.get("throughput_limit_exceeded"),
            "resource_safe": resource.get("resource_safe"),
            "throughput_safe": resource.get("throughput_safe"),
            "monitor_coverage_complete": resource.get("monitor_coverage_complete"),
        },
        "error": {
            "type": error.get("type"),
            "failure_phase": report.get("failure_phase"),
        },
        "artifacts": artifacts,
    }


def compact_trial_report(
    report: Mapping[str, Any], index: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep one human-readable summary without duplicating detailed artifacts."""
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "trial_id": report.get("trial_id"),
        "stage": report.get("stage"),
        "platform": report.get("platform"),
        "result": report.get("result"),
        "updates_target": report.get("updates_target"),
        "updates_completed": report.get("updates_completed"),
        "return_code": report.get("return_code"),
        "stop_reason": report.get("stop_reason"),
        "failure_phase": report.get("failure_phase"),
        "scores": copy.deepcopy(index.get("scores", {})),
        "resource": copy.deepcopy(index.get("resource", {})),
        "error": copy.deepcopy(report.get("error", {})),
        "termination": copy.deepcopy(report.get("termination")),
        "rollout_engine": {
            "name": report.get("rollout_engine", {}).get("name")
            if isinstance(report.get("rollout_engine"), Mapping)
            else None,
            "disable_log_stats": report.get("rollout_engine", {}).get("disable_log_stats")
            if isinstance(report.get("rollout_engine"), Mapping)
            else None,
        },
        "artifacts": copy.deepcopy(index.get("artifacts", {})),
    }


def hydrate_trial(row: Mapping[str, Any], history_path: str | Path) -> dict[str, Any]:
    """Load one current-schema index into the in-memory report shape."""
    if row.get("schema_version") != INDEX_SCHEMA_VERSION or not isinstance(row.get("artifacts"), Mapping):
        raise ValueError(
            "unsupported trial index schema; only schema_version=2 is accepted"
        )
    result = copy.deepcopy(dict(row))
    artifacts = row["artifacts"]

    def artifact(name: str) -> Path | None:
        value = artifacts.get(name)
        return resolve_artifact(history_path, str(value)) if isinstance(value, str) else None

    report_path = artifact("report")
    if report_path is not None:
        result.update(_load_optional(report_path))
    parameter_path = artifact("parameters")
    if parameter_path is not None:
        parameters = _load_optional(parameter_path)
        if parameters:
            result["parameters"] = parameters
    metrics_path = artifact("metrics")
    if metrics_path is not None:
        structured = _load_optional(metrics_path)
        if structured:
            result.update(legacy_metrics_from_structured(structured))
            result["structured_metrics"] = structured
    decision_path = artifact("decision")
    if decision_path is not None:
        decision = _load_optional(decision_path)
        for key in ("proposal", "feasibility", "diagnosis"):
            if key in decision:
                result[key] = decision[key]
    log_facts_path = artifact("log_facts")
    if log_facts_path is not None:
        log_facts = _load_optional(log_facts_path)
        if log_facts:
            result["log_facts"] = log_facts
    for field, name in (
        ("log_path", "log"),
        ("gpu_samples_path", "gpu_samples"),
        ("vllm_metrics_path", "vllm_metrics"),
        ("health_events_path", "health_events"),
        ("health_agent_traces_path", "health_agent_traces"),
    ):
        path = artifact(name)
        result[field] = str(path) if path is not None and path.exists() else None
    return result


def read_trial_indexes(history_path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(history_path)
    for row in rows:
        if row.get("schema_version") != INDEX_SCHEMA_VERSION or not isinstance(
            row.get("artifacts"), Mapping
        ):
            raise ValueError(
                "unsupported trial index schema; only schema_version=2 is accepted"
            )
    return rows


def read_trials(history_path: str | Path) -> list[dict[str, Any]]:
    return [hydrate_trial(row, history_path) for row in read_trial_indexes(history_path)]


def read_metrics_for_trial(
    trial: Mapping[str, Any], history_path: str | Path
) -> dict[str, Any]:
    structured = trial.get("structured_metrics")
    if isinstance(structured, Mapping):
        return dict(structured)
    artifacts = trial.get("artifacts")
    relative = artifacts.get("metrics") if isinstance(artifacts, Mapping) else None
    if isinstance(relative, str):
        return _load_optional(resolve_artifact(history_path, relative))
    return {}
