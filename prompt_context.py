from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")
PHASE_MEMORY_FIELDS = ("mean_used_mib", "p95_used_mib", "max_used_mib")

# Add another summary metric here to expose it in hardware-stage prompts.
# Example: "total_num_tokens": "throughput.summary.total_num_tokens"
HARDWARE_SUMMARY_METRICS = {
    "throughput": "throughput.summary.throughput",
    "actor_mfu": "throughput.summary.actor_mfu",
}

# Add another metric name here to expose its window and terminal values in
# stability/confirm prompts.
STABILITY_METRICS = (
    "critic/rewards/mean",
    "actor/ppo_kl",
    "actor/kl_loss",
    "actor/entropy",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/lr",
)

SUMMARY_FIELDS = ("mean", "p95", "max")
_MISSING = object()


def _read_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _selected_mapping(
    value: Any,
    fields: Sequence[str],
    path: str,
    missing_metrics: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        missing_metrics.extend(f"{path}.{field}" for field in fields)
        return None
    selected: dict[str, Any] = {}
    for field in fields:
        if field in value:
            selected[field] = copy.deepcopy(value[field])
        else:
            missing_metrics.append(f"{path}.{field}")
    return selected or None


def _hardware_metrics(
    structured: Mapping[str, Any],
    summary_metrics: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    phase_memory: dict[str, Any] = {}
    for phase in PHASES:
        path = f"resource.by_phase.{phase}"
        selected = _selected_mapping(
            _read_path(structured, path), PHASE_MEMORY_FIELDS, path, missing
        )
        if selected is not None:
            phase_memory[phase] = selected

    summary: dict[str, Any] = {}
    for name, path in summary_metrics.items():
        selected = _selected_mapping(
            _read_path(structured, path), SUMMARY_FIELDS, path, missing
        )
        if selected is not None:
            summary[name] = selected

    metrics: dict[str, Any] = {}
    if phase_memory:
        metrics["phase_memory_mib"] = phase_memory
    if summary:
        metrics["summary"] = summary
    return metrics, missing


def _stability_metrics(
    structured: Mapping[str, Any], metric_names: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    metrics: dict[str, Any] = {}

    for name in ("step_range", "windows", "terminal_window"):
        path = f"stability.{name}"
        value = _read_path(structured, path)
        if value is _MISSING:
            missing.append(path)
        else:
            metrics[name] = copy.deepcopy(value)

    for collection in ("window_metrics", "terminal_metrics"):
        selected: dict[str, Any] = {}
        for metric_name in metric_names:
            path = f"stability.{collection}.{metric_name}"
            value = _read_path(structured, path)
            if value is _MISSING:
                missing.append(path)
            else:
                selected[metric_name] = copy.deepcopy(value)
        if selected:
            metrics[collection] = selected
    return metrics, missing


def select_stage_metrics(
    structured: Mapping[str, Any],
    stage: str,
    *,
    hardware_summary_metrics: Mapping[str, str] = HARDWARE_SUMMARY_METRICS,
    stability_metrics: Sequence[str] = STABILITY_METRICS,
) -> tuple[dict[str, Any], list[str]]:
    if stage == "hardware" or stage.startswith("hardware_"):
        return _hardware_metrics(structured, hardware_summary_metrics)
    if stage in {"stability", "stability_tuning", "confirm"}:
        return _stability_metrics(structured, stability_metrics)
    raise ValueError(f"unsupported metric stage: {stage}")


def compact_trial_for_prompt(
    trial: Mapping[str, Any],
    current_stage: str,
    editable_keys: Sequence[str],
    *,
    hardware_summary_metrics: Mapping[str, str] = HARDWARE_SUMMARY_METRICS,
    stability_metrics: Sequence[str] = STABILITY_METRICS,
) -> dict[str, Any]:
    parameters = trial.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    structured = trial.get("structured_metrics")
    structured = structured if isinstance(structured, Mapping) else {}

    result: dict[str, Any] = {
        "trial_id": trial.get("trial_id"),
        "stage": trial.get("stage"),
        "result": trial.get("result"),
        "updates_completed": trial.get("updates_completed"),
        "changes": copy.deepcopy(trial.get("changes", {})),
        "editable_parameter_values": {
            key: {
                "value": copy.deepcopy(parameters.get(key)),
                "explicitly_configured": key in parameters,
            }
            for key in editable_keys
        },
    }

    metrics, missing = select_stage_metrics(
        structured,
        current_stage,
        hardware_summary_metrics=hardware_summary_metrics,
        stability_metrics=stability_metrics,
    )
    result["metrics"] = metrics
    if missing:
        result["missing_metrics"] = missing

    error = trial.get("error")
    error = error if isinstance(error, Mapping) else {}
    failure = {
        "type": error.get("type"),
        "failure_phase": trial.get("failure_phase")
        or error.get("failure_phase"),
    }
    failure = {key: value for key, value in failure.items() if value is not None}
    if failure:
        result["failure"] = failure
    return result


def compact_reference_history(
    trials: Sequence[Mapping[str, Any]],
    current_stage: str,
    editable_keys: Sequence[str],
    *,
    required_trial_ids: Sequence[int] = (),
    limit: int = 8,
    hardware_summary_metrics: Mapping[str, str] = HARDWARE_SUMMARY_METRICS,
    stability_metrics: Sequence[str] = STABILITY_METRICS,
) -> list[dict[str, Any]]:
    recent = list(trials[-limit:]) if limit > 0 else []
    required = set(required_trial_ids)
    selected_ids = {trial.get("trial_id") for trial in recent}
    selected = [
        trial
        for trial in trials
        if trial.get("trial_id") in required or trial.get("trial_id") in selected_ids
    ]
    return [
        compact_trial_for_prompt(
            trial,
            current_stage,
            editable_keys,
            hardware_summary_metrics=hardware_summary_metrics,
            stability_metrics=stability_metrics,
        )
        for trial in selected
    ]


def compact_candidate_for_prompt(candidate: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id",
        "reference_trial_id",
        "reference_reason",
        "reason",
        "changes",
        "expected_effect",
        "confidence",
    )
    return {
        field: copy.deepcopy(candidate[field])
        for field in fields
        if field in candidate
    }
