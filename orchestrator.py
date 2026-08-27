from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from agents import AgentResponseError, AgentSet
from config_utils import append_jsonl, apply_changes, write_json
from prompt_context import compact_candidate_for_prompt, compact_reference_history
from prompting import rejection_feedback
from runtime_parameters import (
    effective_from_value,
    parameter_value_views,
    runtime_parameter_values,
)
from runner import run_trial
from trial_storage import (
    build_trial_index,
    compact_trial_report,
    read_trials,
    read_trial_indexes,
)
from validator import (
    IGNORED_PARAMETERS,
    editable_parameters,
    effective_parameters,
    parameter_groups,
    validate_candidate,
)


def _metric_mean(trial: Mapping[str, Any], *path: str) -> float | None:
    value: Any = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    scores = trial.get("scores")
    if isinstance(scores, Mapping):
        fallback = {
            ("performance", "throughput"): "throughput_mean",
            ("performance", "time_per_step_s"): "time_per_step_mean_s",
        }.get(tuple(path))
        candidate = scores.get(fallback) if fallback else None
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _hardware_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if str(trial.get("stage", "")).startswith("hardware")]


def _stability_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("stage") == "stability_tuning"]


def _confirm_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("stage") == "confirm"]


def _successful(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in trials if trial.get("result") == "success"]


def best_hardware_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for trial in _successful(_hardware_trials(trials)):
        throughput = _metric_mean(trial, "performance", "throughput")
        if throughput is not None:
            candidates.append((throughput, trial))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def best_stability_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for trial in _successful(_stability_trials(trials)):
        reward = _terminal_stability_value(trial, "critic/rewards/mean")
        if reward is not None:
            candidates.append((reward, trial))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def hardware_plateaued(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> bool:
    successful = _successful(_hardware_trials(trials))
    minimum = int(config.get("min_hardware_trials", 2))
    plateau_rounds = max(1, int(config.get("plateau_rounds", 2)))
    if len(successful) < minimum + plateau_rounds:
        return False
    scores = [_metric_mean(trial, "performance", "throughput") for trial in successful]
    scores = [score for score in scores if score is not None]
    if len(scores) < minimum + plateau_rounds:
        return False

    # Plateau patience starts after the most recent strict throughput record.
    # ``max`` returns the first index on a tie, so equal/noisy repeats do not
    # reset patience, while a new raw best always does.
    best_index = max(range(len(scores)), key=scores.__getitem__)
    rounds_since_best = len(scores) - best_index - 1
    return rounds_since_best >= plateau_rounds


def _reward_trend_degraded(
    reward_points: list[tuple[int, float]],
    config: Mapping[str, Any],
) -> bool:
    observation_steps = max(2, int(config.get("health_reward_trend_steps", 5)))
    window_size = max(1, int(config.get("health_reward_window_size", 3)))
    if window_size >= observation_steps or len(reward_points) < observation_steps:
        return False

    values = [value for _, value in reward_points]
    recent = values[-observation_steps:]
    recent_means = [
        mean(recent[index : index + window_size])
        for index in range(observation_steps - window_size + 1)
    ]
    tolerance = max(0.0, float(config.get("health_reward_trend_tolerance", 0.01)))
    non_improving = all(
        current <= previous + tolerance
        for previous, current in zip(recent_means, recent_means[1:])
    )
    historical_means = [
        mean(values[index : index + window_size])
        for index in range(len(values) - window_size + 1)
    ]
    drawdown = max(historical_means) - recent_means[-1]
    minimum_drawdown = max(
        0.0,
        float(config.get("health_reward_trend_min_drawdown", 0.15)),
    )
    return non_improving and drawdown >= minimum_drawdown


def _kl_changed_suddenly(
    kl_points: list[tuple[int, float]],
    config: Mapping[str, Any],
) -> bool:
    ratio_threshold = max(
        0.0,
        float(config.get("health_kl_change_ratio_threshold", 0.50)),
    )
    absolute_threshold = max(
        0.0,
        float(config.get("health_kl_change_absolute_threshold", 0.02)),
    )
    for (_, previous), (_, current) in zip(kl_points, kl_points[1:]):
        change = current - previous
        ratio = abs(change) / max(abs(previous), 1e-12)
        if abs(change) >= absolute_threshold and ratio >= ratio_threshold:
            return True
    return False


def _entropy_collapsed_suddenly(
    entropy_points: list[tuple[int, float]],
    config: Mapping[str, Any],
) -> bool:
    threshold = max(
        0.0,
        float(config.get("health_entropy_drop_ratio_threshold", 0.30)),
    )
    for (_, previous), (_, current) in zip(entropy_points, entropy_points[1:]):
        drop_ratio = (previous - current) / max(abs(previous), 1e-3)
        if drop_ratio >= threshold:
            return True
    return False


def stability_healthy(trial: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    scores = trial.get("scores")
    indexed = scores.get("stability_healthy") if isinstance(scores, Mapping) else None
    if isinstance(indexed, bool):
        return indexed
    # Fresh reports do not have the compact index score until persistence.
    if trial.get("result") != "success":
        return False
    reward_points = _complete_stability_points(trial, "critic/rewards/mean")
    if _reward_trend_degraded(reward_points, config):
        return False
    kl_points = _complete_stability_points(trial, "actor/kl_loss")
    if _kl_changed_suddenly(kl_points, config):
        return False
    entropy_points = _complete_stability_points(trial, "actor/entropy")
    if _entropy_collapsed_suddenly(entropy_points, config):
        return False
    return True


def determine_stage(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> str:
    if _confirm_trials(trials):
        return "done"
    start_stage = str(config.get("start_stage", "auto"))
    if start_stage not in {"auto", "hardware_tuning", "stability_tuning"}:
        raise ValueError(
            "start_stage must be auto, hardware_tuning, or stability_tuning"
        )
    if start_stage == "stability_tuning":
        stability = _stability_trials(trials)
        healthy = [trial for trial in stability if stability_healthy(trial, config)]
        if len(stability) >= int(config.get("max_stability_trials", 4)):
            return "confirm" if healthy else "stopped_unstable"
        if len(healthy) >= int(config.get("min_stability_trials", 2)):
            return "confirm"
        return "stability_tuning"
    hardware = _hardware_trials(trials)
    successful_hardware = _successful(hardware)
    if not successful_hardware:
        return "hardware_repair" if hardware else "hardware_tuning"
    if len(hardware) < int(config.get("min_hardware_trials", 2)):
        return "hardware_tuning"
    if len(hardware) < int(config.get("max_hardware_trials", 6)) and not hardware_plateaued(trials, config):
        return "hardware_tuning"

    stability = _stability_trials(trials)
    healthy = [trial for trial in stability if stability_healthy(trial, config)]
    if len(stability) >= int(config.get("max_stability_trials", 4)):
        return "confirm" if healthy else "stopped_unstable"
    if len(healthy) >= int(config.get("min_stability_trials", 2)):
        return "confirm"
    return "stability_tuning"


def trial_budget(stage: str, config: Mapping[str, Any]) -> int:
    if stage.startswith("hardware"):
        return int(config.get("hardware_trial_updates", 20))
    if stage == "stability_tuning":
        return int(config.get("stability_trial_updates", 80))
    if stage == "confirm":
        return int(config.get("confirm_trial_updates", 300))
    return 0


def _compact_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "trial_id",
        "stage",
        "result",
        "updates_completed",
        "parameters",
        "error",
        "resource",
        "memory_by_phase_pct",
        "performance",
        "rollout_engine",
        "stability",
        "health_monitor",
        "health_decisions",
        "termination",
        "diagnosis",
        "failure_phase",
        "checkpoint",
        "resume",
        "updates_executed",
        "proposal",
        "feasibility",
        "log_path",
        "vllm_metrics_path",
    ]
    return {key: copy.deepcopy(trial[key]) for key in keys if key in trial}


def _complete_stability_points(trial: Mapping[str, Any], metric: str) -> list[tuple[int, float]]:
    """Read complete structured-metrics windows for one stability metric."""
    stability = trial.get("stability")
    if not isinstance(stability, Mapping):
        return []
    windows = stability.get("windows")
    metrics = stability.get("metrics")
    window_size = stability.get("window_size")
    values = metrics.get(metric) if isinstance(metrics, Mapping) else None
    if isinstance(windows, list) and isinstance(values, list):
        points = []
        for window, value in zip(windows, values):
            if not isinstance(window, Mapping) or not isinstance(value, (int, float)):
                continue
            required = window_size if isinstance(window_size, int) else window.get("sample_count")
            if window.get("sample_count") != required:
                continue
            end_step = window.get("end_step")
            if isinstance(end_step, int):
                points.append((end_step, float(value)))
        return points

    return []


def _terminal_stability_value(
    trial: Mapping[str, Any], metric: str
) -> float | None:
    """Read the trailing metric mean used to compare completed stability trials."""
    scores = trial.get("scores")
    if metric == "critic/rewards/mean" and isinstance(scores, Mapping):
        indexed = scores.get("terminal_reward")
        if isinstance(indexed, (int, float)) and not isinstance(indexed, bool):
            return float(indexed)
    stability = trial.get("stability")
    if not isinstance(stability, Mapping):
        return None
    terminal_metrics = stability.get("terminal_metrics")
    if isinstance(terminal_metrics, Mapping):
        value = terminal_metrics.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    points = _complete_stability_points(trial, metric)
    return points[-1][1] if points else None


def _reference_descriptor(trial: Mapping[str, Any] | None, selection_reason: str) -> dict[str, Any]:
    if trial is None:
        return {
            "source": "base_parameters",
            "trial_id": None,
            "selection_reason": selection_reason,
        }
    # Do not embed the reference trial's own proposal/reference chain here.
    # A flat snapshot keeps each new report bounded and independently auditable.
    keys = (
        "trial_id",
        "stage",
        "result",
        "parameters",
        "resource",
        "memory_by_phase_pct",
        "performance",
        "error",
        "termination",
        "checkpoint",
    )
    compact = {key: copy.deepcopy(trial[key]) for key in keys if key in trial}
    compact.update(
        {
            "source": "trial",
            "trial_id": trial.get("trial_id"),
            "selection_reason": selection_reason,
        }
    )
    return compact


def _immutable_model_context(
    parameters: Mapping[str, Any],
    log_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_path = parameters.get("actor_rollout_ref.model.path")
    result: dict[str, Any] = {"model_path": model_path}
    persisted_config = (
        log_facts.get("model_config")
        if isinstance(log_facts, Mapping)
        else None
    )
    model_config: Mapping[str, Any] = (
        persisted_config if isinstance(persisted_config, Mapping) else {}
    )
    fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "torch_dtype",
        "max_position_embeddings",
    )
    if model_config:
        result["model_config"] = {
            key: model_config[key] for key in fields if key in model_config
        }
    if isinstance(log_facts, Mapping):
        megatron = log_facts.get("megatron")
        megatron = megatron if isinstance(megatron, Mapping) else {}
        profile = megatron.get("parameter_summary")
        if isinstance(profile, Mapping):
            result["logged_parameter_profile"] = {
                key: profile.get(key)
                for key in (
                    "most_loaded_shard_parameters",
                    "total_parameters",
                    "total_parameters_source",
                    "complete_tp_pp_coverage",
                    "reference_topology",
                )
            }
        resolved = megatron.get("resolved_config")
        if isinstance(resolved, Mapping):
            result["resolved_runtime"] = dict(resolved)
        workload = log_facts.get("workload")
        length = (
            workload.get("sequence_length")
            if isinstance(workload, Mapping)
            else None
        )
        if isinstance(length, Mapping):
            result["observed_sequence_length"] = dict(length)
    return result


def _baseline_proposal(
    reference: Mapping[str, Any],
    reason: str,
    transition_trigger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe an orchestrator-owned no-change trial without impersonating Agent output."""
    proposal: dict[str, Any] = {
        "decision": "baseline",
        "source": "orchestrator",
        "reason": reason,
        "reference_trial_id": reference.get("trial_id"),
        "reference_trial": copy.deepcopy(dict(reference)),
        "changes": {},
        "expected_effect": {},
    }
    if isinstance(transition_trigger, Mapping):
        proposal["transition_trigger"] = {
            "decision": transition_trigger.get("decision"),
            "reason": transition_trigger.get("reason"),
        }
    return proposal


def _next_stage_baseline(
    stage: str, trials: list[dict[str, Any]]
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    if stage.startswith("hardware"):
        selected = best_hardware_trial(trials)
        next_stage = "stability_tuning"
        reason = "best successful hardware trial used as stability baseline"
    elif stage == "stability_tuning":
        selected = best_stability_trial(trials)
        next_stage = "confirm"
        reason = (
            "best successful stability trial by terminal reward mean "
            "selected for confirmation"
        )
    else:
        return None
    if selected is None or not isinstance(selected.get("parameters"), Mapping):
        return None
    return (
        next_stage,
        dict(selected["parameters"]),
        _reference_descriptor(selected, reason),
    )


def _runs_automatic_baseline(stage: str, trials: list[dict[str, Any]]) -> bool:
    return (
        not trials
        or stage == "confirm"
        or (stage == "stability_tuning" and not _stability_trials(trials))
    )


def _stream_orchestrator_event(
    config: Mapping[str, Any], event: str, payload: Mapping[str, Any]
) -> None:
    if not bool(config.get("stream_agent_events", True)):
        return

    print(
        f"\n[Orchestrator] {event}\n"
        + json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
        flush=True,
    )


def _normalize_proposal_changes(
    proposal: Mapping[str, Any],
    reference_parameters: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_runtime_values: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Validate the Agent's provenance and derive executable target values."""
    violations: list[str] = []
    if proposal.get("decision") != "modify":
        violations.append("change objects require decision=modify")
    expected_reference_id = reference.get("trial_id")
    if proposal.get("reference_trial_id") != expected_reference_id:
        violations.append(
            "reference_trial_id must match the actual parameter source "
            f"{expected_reference_id!r}, got {proposal.get('reference_trial_id')!r}"
        )
    reference_reason = proposal.get("reference_reason")
    if not isinstance(reference_reason, str) or not reference_reason.strip():
        violations.append("reference_reason must explain why the reference trial is used")
    expected_effect = proposal.get("expected_effect")
    if not isinstance(expected_effect, Mapping) or not expected_effect:
        violations.append("expected_effect must remain a non-empty object for modify decisions")

    raw_changes = proposal.get("changes")
    if not isinstance(raw_changes, Mapping) or not raw_changes:
        violations.append("changes must be a non-empty object")
        return {}, {}, violations

    targets: dict[str, Any] = {}
    details: dict[str, dict[str, Any]] = {}
    for parameter, raw_detail in raw_changes.items():
        if not isinstance(parameter, str) or not parameter:
            violations.append("every change key must be a complete Hydra parameter name")
            continue
        if not isinstance(raw_detail, Mapping):
            violations.append(f"{parameter} change must contain from, to, and reason")
            continue
        missing = [name for name in ("from", "to", "reason") if name not in raw_detail]
        if missing:
            violations.append(f"{parameter} change is missing: {', '.join(missing)}")
            continue
        is_explicitly_configured = parameter in reference_parameters
        actual_from = reference_parameters.get(parameter)
        declared_from = raw_detail.get("from")
        target = raw_detail.get("to")
        reason = raw_detail.get("reason")
        if is_explicitly_configured and declared_from != actual_from:
            violations.append(
                f"{parameter} from must equal reference value "
                f"{actual_from!r}, got {declared_from!r}"
            )
        if not is_explicitly_configured and declared_from is not None:
            violations.append(
                f"{parameter} is not explicitly configured in the reference trial; from must be null"
            )
        if is_explicitly_configured and target == actual_from:
            violations.append(f"{parameter} is a no-op change: {actual_from!r} -> {target!r}")
        if not is_explicitly_configured and target is None:
            violations.append(f"{parameter} cannot add a null override")
        if not isinstance(reason, str) or not reason.strip():
            violations.append(f"{parameter} reason must be a non-empty string")
        targets[parameter] = target
        details[parameter] = {
            "from": actual_from,
            "effective_from": effective_from_value(
                parameter, reference_runtime_values
            ),
            "to": target,
            "reason": reason.strip() if isinstance(reason, str) else reason,
        }
    return targets, details, violations


def _resolve_candidate_reference(
    reference_trial_id: Any,
    reference_reason: Any,
    trials: list[dict[str, Any]],
    base_parameters: Mapping[str, Any],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any],
    list[str],
]:
    """Resolve one Proposal candidate's independently selected parameter source."""
    if reference_trial_id is None:
        return (
            dict(base_parameters),
            _reference_descriptor(
                None,
                reference_reason
                if isinstance(reference_reason, str) and reference_reason.strip()
                else "initial base parameters selected by Proposal",
            ),
            {},
            [],
        )
    if not isinstance(reference_trial_id, int) or isinstance(reference_trial_id, bool):
        return None, None, {}, ["reference_trial_id must be an integer trial ID or null"]
    trial = next(
        (row for row in trials if row.get("trial_id") == reference_trial_id),
        None,
    )
    if trial is None:
        return None, None, {}, [f"reference trial {reference_trial_id} does not exist"]
    parameters = trial.get("parameters")
    if not isinstance(parameters, Mapping):
        return None, None, {}, [f"reference trial {reference_trial_id} has no parameter map"]
    observed_runtime = runtime_parameter_values(
        trial.get("log_facts")
        if isinstance(trial.get("log_facts"), Mapping)
        else None
    )
    if not observed_runtime:
        return (
            None,
            None,
            {},
            [
                f"reference trial {reference_trial_id} has no resolved runtime_parameters; "
                "re-extract its train.log before using it as a reference"
            ],
        )
    return (
        dict(parameters),
        _reference_descriptor(
            trial,
            reference_reason
            if isinstance(reference_reason, str) and reference_reason.strip()
            else "recorded trial selected by Proposal",
        ),
        observed_runtime,
        [],
    )


def _feasibility_selection_violations(
    review: Mapping[str, Any],
    candidate_ids: set[str],
) -> list[str]:
    """Validate that Feasibility selected, but did not rewrite, a reviewed candidate."""
    violations: list[str] = []
    verdict = review.get("verdict")
    selected_id = review.get("selected_candidate_id")
    if verdict not in {"valid", "invalid"}:
        violations.append("verdict must be valid or invalid")
    if verdict == "valid" and selected_id not in candidate_ids:
        violations.append(
            "selected_candidate_id must identify a deterministically valid "
            f"candidate; got {selected_id!r}"
        )
    if verdict == "invalid" and selected_id is not None:
        violations.append(
            "selected_candidate_id must be null when verdict is invalid"
        )

    raw_candidate_reviews = review.get("candidate_reviews")
    if not isinstance(raw_candidate_reviews, list):
        violations.append("candidate_reviews must be an array covering every candidate")
        return violations
    reviewed_ids: list[str] = []
    review_verdicts: dict[str, Any] = {}
    for row in raw_candidate_reviews:
        if not isinstance(row, Mapping):
            violations.append("every candidate_reviews entry must be an object")
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            violations.append(
                "every candidate_reviews entry must contain a string candidate_id"
            )
            continue
        reviewed_ids.append(candidate_id)
        candidate_verdict = row.get("verdict")
        if candidate_verdict not in {"valid", "invalid"}:
            violations.append(
                f"candidate review {candidate_id!r} must use verdict valid or invalid"
            )
        review_verdicts[candidate_id] = candidate_verdict
    reviewed_set = set(reviewed_ids)
    if len(reviewed_ids) != len(reviewed_set):
        violations.append("candidate_reviews contains duplicate candidate IDs")
    if reviewed_set != candidate_ids:
        violations.append(
            "candidate_reviews must cover exactly the deterministically valid "
            f"candidate IDs; expected={sorted(candidate_ids)}, got={sorted(reviewed_set)}"
        )
    if verdict == "valid" and selected_id in candidate_ids:
        if review_verdicts.get(str(selected_id)) != "valid":
            violations.append(
                "the selected candidate's individual review must have verdict valid"
            )
    return violations


class TuningOrchestrator:
    def __init__(
        self,
        root: str | Path,
        base_parameters: Mapping[str, Any],
        agent_config: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.base_parameters = dict(base_parameters)
        self.config = dict(agent_config)
        self.output_dir = Path(os.getenv("OUTPUT_PATH", str(self.config["output_dir"]))).expanduser().resolve()
        self.history_path = self.output_dir / "trials.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.agents = AgentSet(
            self.root,
            str(self.config.get("agent_mode", "llm")),
            self.config,
            self.history_path,
        )

    def trials(self) -> list[dict[str, Any]]:
        return read_trials(self.history_path)

    def trial_indexes(self) -> list[dict[str, Any]]:
        return read_trial_indexes(self.history_path)

    def _starting_point(
        self, stage: str, trials: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if stage.startswith("hardware"):
            best = best_hardware_trial(trials)
            if best:
                return dict(best["parameters"]), _reference_descriptor(
                    best, "best successful hardware trial by throughput"
                )
            if trials:
                return dict(trials[-1]["parameters"]), _reference_descriptor(
                    trials[-1], "latest trial used for hardware failure repair"
                )
            return dict(self.base_parameters), _reference_descriptor(
                None, "initial base parameters; no completed trial exists"
            )
        if stage == "stability_tuning":
            stability = best_stability_trial(trials)
            best = stability or best_hardware_trial(trials)
            if not best:
                if str(self.config.get("start_stage", "auto")) == "stability_tuning":
                    return dict(self.base_parameters), _reference_descriptor(
                        None,
                        "initial base parameters used by direct stability mode",
                    )
                raise RuntimeError("stability tuning requires a successful hardware trial")
            reason = (
                "best successful stability trial by terminal reward mean"
                if stability is not None
                else "best successful hardware trial used as stability baseline"
            )
            return dict(best["parameters"]), _reference_descriptor(best, reason)
        if stage == "confirm":
            stability = best_stability_trial(trials)
            if stability is None:
                raise RuntimeError("confirmation requires a successful stability trial")
            reason = (
                "best successful stability trial by terminal reward mean "
                "selected for confirmation"
            )
            return dict(stability["parameters"]), _reference_descriptor(stability, reason)
        raise RuntimeError(f"unsupported stage: {stage}")

    def _starting_parameters(self, stage: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
        return self._starting_point(stage, trials)[0]

    def _confirm_resume_checkpoint(
        self,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        checkpoint = reference.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                "successful stability trial is missing its required checkpoint artifact"
            )
        return {
            "source_trial_id": reference["trial_id"],
            "global_step": checkpoint["global_step"],
            "path": checkpoint["path"],
        }

    def _diagnosis(
        self, trials: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not trials or trials[-1].get("result") == "success":
            return None, None
        if trials[-1].get("result") == "early_stopped":
            termination = trials[-1].get("termination", {})
            decision = termination.get("decision", {}) if isinstance(termination, Mapping) else {}
            diagnosis = {
                "failure_type": "TRAIN_UNHEALTHY",
                "training_substage": "training",
                "evidence": decision.get("evidence", []),
                "reason": decision.get("reason", "train_health Agent requested early stop"),
                "confidence": decision.get("confidence", 0.0),
                "reason_codes": decision.get("reason_codes", []),
            }
            return diagnosis, {
                "role": "train_health",
                "source": "persisted_early_stop_decision",
                "result": copy.deepcopy(decision),
            }
        context = {"trial": _compact_trial(trials[-1])}
        run = self.agents.diagnose(context)
        return run.result, run.as_trace()

    def _propose_candidate(
        self,
        stage: str,
        current: Mapping[str, Any],
        trials: list[dict[str, Any]],
        reference: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if reference is None:
            matching = next(
                (
                    trial
                    for trial in reversed(trials)
                    if isinstance(trial.get("parameters"), Mapping)
                    and dict(trial["parameters"]) == dict(current)
                ),
                None,
            )
            reference = _reference_descriptor(
                matching,
                "inferred from current parameters for direct proposal evaluation",
            )
        diagnosis, diagnosis_trace = self._diagnosis(trials)
        rejections: list[dict[str, Any]] = []
        max_rounds = int(self.config.get("max_validation_rounds", 3))
        history_limit = int(self.config.get("history_prompt_trials", 8))
        min_candidates = max(
            2, int(self.config.get("min_proposal_candidates", 2))
        )
        max_candidates = max(
            min_candidates, int(self.config.get("max_proposal_candidates", 3))
        )
        reference_trial_row = next(
            (
                trial
                for trial in trials
                if trial.get("trial_id") == reference.get("trial_id")
            ),
            None,
        )
        reference_structured = (
            reference_trial_row.get("structured_metrics")
            if isinstance(reference_trial_row, Mapping)
            else None
        )
        reference_resource = (
            reference_structured.get("resource")
            if isinstance(reference_structured, Mapping)
            else None
        )
        observed_devices = (
            reference_resource.get("devices")
            if isinstance(reference_resource, Mapping)
            else []
        )
        observed_device_memory = min(
            (
                float(row["total_memory_mib"])
                for row in observed_devices or []
                if isinstance(row, Mapping)
                and isinstance(row.get("total_memory_mib"), (int, float))
            ),
            default=None,
        )
        reference_log_facts = (
            reference_trial_row.get("log_facts")
            if isinstance(reference_trial_row, Mapping)
            else None
        )
        editable = editable_parameters(stage)
        reference_trial_id = reference.get("trial_id")
        required_reference_ids = (
            [reference_trial_id] if isinstance(reference_trial_id, int) else []
        )
        context = {
            "current_stage": stage,
            "mode": "failure_repair" if diagnosis else stage,
            "fixed_parameters": {
                key: value
                for key, value in current.items()
                if key not in set(editable) | IGNORED_PARAMETERS
            },
            "editable_parameter_values": parameter_value_views(
                current,
                reference_log_facts
                if isinstance(reference_log_facts, Mapping)
                else None,
                editable,
            ),
            "immutable_context": {
                "model": {
                    **_immutable_model_context(
                        current,
                        reference_log_facts
                        if isinstance(reference_log_facts, Mapping)
                        else None,
                    ),
                },
                "hardware": {
                    "platform": self.config.get("platform"),
                    "nnodes": current.get("trainer.nnodes"),
                    "gpus_per_node": current.get("trainer.n_gpus_per_node"),
                    "world_size": int(current.get("trainer.nnodes", 1))
                    * int(current.get("trainer.n_gpus_per_node", 1)),
                    "observed_device_memory_mib": observed_device_memory,
                    "resource_memory_reserve_mib": self.config.get(
                        "resource_memory_reserve_mib"
                    ),
                    "throughput_memory_reserve_mib": self.config.get(
                        "throughput_memory_reserve_mib"
                    ),
                },
                "workload": {
                    "algorithm": current.get("algorithm.adv_estimator"),
                    "train_batch_size": current.get("data.train_batch_size"),
                    "max_prompt_length": current.get("data.max_prompt_length"),
                    "max_response_length": current.get("data.max_response_length"),
                },
                "runtime_relationships": {
                    "ref_model_parallel_topology": "inherits_actor",
                    "entropy_calculation": (
                        "training calculate_entropy iff entropy_coeff != 0"
                    ),
                },
            },
            "default_reference": {
                "trial_id": reference_trial_id,
                "selection_reason": reference.get("selection_reason"),
            },
            "compact_reference_history": compact_reference_history(
                trials,
                stage,
                editable,
                required_trial_ids=required_reference_ids,
                limit=history_limit,
            ),
            "editable_parameters": editable,
            "constraints": {
                "min_proposal_candidates": min_candidates,
                "max_proposal_candidates": max_candidates,
                "max_parameter_changes": self.config.get("max_parameter_changes", 3),
                "preserve_hardware_token_budget": self.config.get("preserve_hardware_token_budget", True),
                "resource_memory_reserve_mib": self.config.get("resource_memory_reserve_mib", 3277),
                "throughput_memory_reserve_mib": self.config.get("throughput_memory_reserve_mib", 6554),
            },
            "diagnosis": diagnosis,
        }
        proposal_conversation = None
        trace: dict[str, Any] = {
            "diagnosis": diagnosis_trace,
            "diagnosis_summary": copy.deepcopy(diagnosis),
            "proposal_conversation": None,
            "feasibility_reviews": [],
            "candidate_validations": [],
            "rejections": rejections,
        }
        for attempt in range(1, max_rounds + 1):
            proposal_run = self.agents.propose(
                context if proposal_conversation is None else None,
                proposal_conversation,
            )
            proposal_conversation = proposal_run.conversation
            proposal = proposal_run.result
            trace["proposal_conversation"] = proposal_run.as_trace()
            decision = proposal.get("decision")
            raw_candidates = proposal.get("candidates")
            batch_violations: list[str] = []
            candidate_rows: list[Any] = []
            if decision not in {"modify", "stop"}:
                batch_violations.append(
                    "decision must be modify or stop; keep is not a valid Proposal decision"
                )
            batch_reason = proposal.get("reason")
            if not isinstance(batch_reason, str) or not batch_reason.strip():
                batch_violations.append(
                    "reason must be a non-empty batch-level explanation"
                )
            if decision == "stop" and raw_candidates not in (None, []):
                batch_violations.append(
                    "stop decisions must return an empty candidates array"
                )
            elif decision == "modify" and not isinstance(raw_candidates, list):
                batch_violations.append("candidates must be an array")
            elif decision == "modify":
                candidate_rows = raw_candidates
                if not min_candidates <= len(candidate_rows) <= max_candidates:
                    batch_violations.append(
                        f"modify decisions require {min_candidates} to {max_candidates} candidates, "
                        f"got {len(candidate_rows)}"
                    )
            if batch_violations:
                validation = {"valid": False, "violations": batch_violations}
                rejections.append(
                    {
                        "attempt": attempt,
                        "source": "deterministic_validator",
                        "proposal": proposal,
                        **validation,
                    }
                )
                _stream_orchestrator_event(
                    self.config,
                    "proposal_rejected",
                    {"attempt": attempt, "source": "proposal_schema", **validation},
                )
                proposal_conversation.add_user_message(
                    rejection_feedback(
                        attempt,
                        proposal,
                        {},
                        "deterministic_validator",
                        validation,
                    )
                )
                trace["proposal_conversation"] = proposal_conversation.as_trace()
                continue
            if decision == "stop":
                proposal.setdefault("reference_trial_id", reference.get("trial_id"))
                proposal["reference_trial"] = copy.deepcopy(dict(reference))
                proposal.setdefault("candidates", [])
                proposal.setdefault("changes", {})
                proposal.setdefault("expected_effect", {})
                return (
                    dict(current),
                    proposal,
                    {"verdict": "valid", "reason": "current stage stopped by Agent"},
                    trace,
                )

            validated: dict[str, dict[str, Any]] = {}
            candidate_results: list[dict[str, Any]] = []
            canonical_configurations: set[str] = set()
            seen_ids: set[str] = set()
            candidate_parameters_for_feedback: dict[str, Any] = {}
            for index, raw_candidate in enumerate(candidate_rows, start=1):
                violations: list[str] = []
                if not isinstance(raw_candidate, Mapping):
                    candidate_results.append(
                        {
                            "candidate_id": None,
                            "index": index,
                            "valid": False,
                            "violations": ["candidate must be an object"],
                        }
                    )
                    continue
                candidate_id = raw_candidate.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    violations.append("candidate_id must be a non-empty string")
                    normalized_id = f"candidate_at_index_{index}"
                else:
                    normalized_id = candidate_id.strip()
                    if len(normalized_id) > 64:
                        violations.append(
                            "candidate_id must not exceed 64 characters"
                        )
                    if normalized_id in seen_ids:
                        violations.append(
                            f"candidate_id {normalized_id!r} is duplicated in this proposal batch"
                        )
                    seen_ids.add(normalized_id)
                candidate_reason = raw_candidate.get("reason")
                if (
                    not isinstance(candidate_reason, str)
                    or not candidate_reason.strip()
                ):
                    violations.append(
                        "candidate reason must be a non-empty causal explanation"
                    )

                (
                    reference_parameters,
                    candidate_reference,
                    reference_runtime_values,
                    reference_violations,
                ) = (
                    _resolve_candidate_reference(
                        raw_candidate.get("reference_trial_id"),
                        raw_candidate.get("reference_reason"),
                        trials,
                        self.base_parameters,
                    )
                )
                violations.extend(reference_violations)
                normalized_proposal = copy.deepcopy(dict(raw_candidate))
                normalized_proposal["decision"] = "modify"
                normalized_proposal["candidate_id"] = normalized_id

                target_changes: dict[str, Any] = {}
                change_details: dict[str, dict[str, Any]] = {}
                if reference_parameters is not None and candidate_reference is not None:
                    target_changes, change_details, provenance_violations = (
                        _normalize_proposal_changes(
                            normalized_proposal,
                            reference_parameters,
                            candidate_reference,
                            reference_runtime_values,
                        )
                    )
                    violations.extend(provenance_violations)

                executable_parameters: dict[str, Any] | None = None
                if not violations and reference_parameters is not None:
                    executable_parameters = apply_changes(
                        reference_parameters, target_changes
                    )
                    deterministic = validate_candidate(
                        executable_parameters,
                        target_changes,
                        stage,
                        self.config,
                        self.base_parameters,
                        trials,
                        locked_parameters=current,
                        reference_runtime_parameters=reference_runtime_values,
                    )
                    if not deterministic.valid:
                        violations.extend(deterministic.violations)
                    else:
                        canonical = json.dumps(
                            effective_parameters(
                                executable_parameters,
                                reference_runtime_parameters=reference_runtime_values,
                                changed_keys=target_changes,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        if canonical in canonical_configurations:
                            violations.append(
                                "candidate resolves to the same complete configuration "
                                "as another candidate in this batch"
                            )
                        else:
                            canonical_configurations.add(canonical)

                if executable_parameters is not None:
                    candidate_parameters_for_feedback[normalized_id] = copy.deepcopy(
                        executable_parameters
                    )
                result: dict[str, Any] = {
                    "candidate_id": normalized_id,
                    "index": index,
                    "valid": not violations,
                    "violations": violations,
                }
                if any("not editable in stage" in row for row in violations):
                    result["editable_parameters"] = editable_parameters(stage)
                candidate_results.append(result)
                if violations:
                    continue

                normalized_proposal["changes"] = change_details
                normalized_proposal["target_changes"] = dict(target_changes)
                normalized_proposal["reference_trial"] = copy.deepcopy(
                    dict(candidate_reference)
                )
                packet = {
                    "candidate_id": normalized_id,
                    "reference_trial_id": normalized_proposal.get(
                        "reference_trial_id"
                    ),
                    "reference_reason": normalized_proposal.get("reference_reason"),
                    "reference_trial": copy.deepcopy(dict(candidate_reference)),
                    "reason": normalized_proposal.get("reason"),
                    "changes": copy.deepcopy(change_details),
                    "target_changes": dict(target_changes),
                    "candidate_parameters": copy.deepcopy(executable_parameters),
                    "expected_effect": copy.deepcopy(
                        normalized_proposal.get("expected_effect")
                    ),
                    "confidence": normalized_proposal.get("confidence"),
                }
                validated[normalized_id] = {
                    "proposal": normalized_proposal,
                    "parameters": executable_parameters,
                    "packet": packet,
                }

            trace["candidate_validations"].append(
                {"attempt": attempt, "candidates": copy.deepcopy(candidate_results)}
            )
            if len(validated) < min_candidates:
                validation = {
                    "valid": False,
                    "violations": [
                        f"only {len(validated)} proposal candidates passed deterministic "
                        f"validation; at least {min_candidates} are required before "
                        "Feasibility can select among them"
                    ],
                    "candidate_validations": candidate_results,
                }
                rejections.append(
                    {
                        "attempt": attempt,
                        "source": "deterministic_validator",
                        "proposal": proposal,
                        **validation,
                    }
                )
                _stream_orchestrator_event(
                    self.config,
                    "proposal_rejected",
                    {
                        "attempt": attempt,
                        "source": "deterministic_validator",
                        **validation,
                    },
                )
                proposal_conversation.add_user_message(
                    rejection_feedback(
                        attempt,
                        proposal,
                        candidate_parameters_for_feedback,
                        "deterministic_validator",
                        validation,
                    )
                )
                trace["proposal_conversation"] = proposal_conversation.as_trace()
                continue

            review_candidates = [
                compact_candidate_for_prompt(row["packet"])
                for row in validated.values()
            ]
            candidate_reference_ids = [
                row["reference_trial_id"]
                for row in review_candidates
                if isinstance(row.get("reference_trial_id"), int)
            ]
            review_run = self.agents.review(
                {
                    "current_stage": stage,
                    "candidates": review_candidates,
                    "compact_reference_history": compact_reference_history(
                        trials,
                        stage,
                        editable,
                        required_trial_ids=candidate_reference_ids,
                        limit=history_limit,
                    ),
                    "diagnosis": diagnosis,
                    "memory_limits": {
                        "unit": "MiB",
                        "resource_memory_reserve_mib": self.config.get("resource_memory_reserve_mib", 3277),
                        "throughput_memory_reserve_mib": self.config.get("throughput_memory_reserve_mib", 6554),
                        "effective_limit_formula": "device_total_memory_mib - reserve_mib",
                    },
                }
            )
            raw_review = review_run.result
            review = copy.deepcopy(raw_review)
            verdict = review.get("verdict")
            selection_violations = _feasibility_selection_violations(
                review, set(validated)
            )
            if selection_violations:
                review = {
                    **review,
                    "verdict": "invalid",
                    "reason": "Feasibility returned an invalid candidate selection",
                    "selection_violations": selection_violations,
                    "raw_verdict": verdict,
                }
            trace["feasibility_reviews"].append(
                {"attempt": attempt, **review_run.as_trace()}
            )
            if review.get("verdict") == "valid":
                trace["proposal_conversation"] = proposal_run.as_trace()
                selected = validated[str(review["selected_candidate_id"])]
                return (
                    copy.deepcopy(selected["parameters"]),
                    copy.deepcopy(selected["proposal"]),
                    review,
                    trace,
                )
            rejection = {
                "attempt": attempt,
                "source": "feasibility_agent",
                "proposal": proposal,
                "feasibility": review,
            }
            rejections.append(rejection)
            _stream_orchestrator_event(
                self.config,
                "proposal_rejected",
                {
                    "attempt": attempt,
                    "source": "feasibility_agent",
                    "feasibility": review,
                },
            )
            proposal_conversation.add_user_message(
                rejection_feedback(
                    attempt,
                    proposal,
                    candidate_parameters_for_feedback,
                    "feasibility_agent",
                    review,
                )
            )
            trace["proposal_conversation"] = proposal_conversation.as_trace()
        write_json(self.output_dir / "last_agent_rejection.json", trace)
        blocked = {
            "decision": "blocked",
            "reference_trial_id": reference.get("trial_id"),
            "reference_trial": copy.deepcopy(dict(reference)),
            "reason": f"no feasible proposal after {max_rounds} candidate rounds",
            "changes": {},
            "expected_effect": {},
            "rejection_count": len(rejections),
            "rejection_path": str(self.output_dir / "last_agent_rejection.json"),
        }
        _stream_orchestrator_event(self.config, "proposal_blocked", blocked)
        return (
            dict(current),
            blocked,
            {"verdict": "blocked", "reason": blocked["reason"]},
            trace,
        )

    def run(self, max_trials: int = 1, dry_run: bool = False) -> list[dict[str, Any]]:
        produced = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(max_trials):
            trial_indexes = self.trial_indexes()
            stage = determine_stage(trial_indexes, self.config)
            if stage in {"done", "stopped_unstable"}:
                write_json(
                    self.state_path,
                    {
                        "current_stage": stage,
                        "last_trial_id": len(trial_indexes),
                        "history_path": str(self.history_path),
                    },
                )
                break
            trials = self.trials()
            trial_id = len(trials) + 1
            parameters, reference = self._starting_point(stage, trials)
            proposal = _baseline_proposal(reference, "automatic stage baseline")
            review: dict[str, Any] = {"verdict": "valid", "reason": "stage baseline"}
            agent_trace: dict[str, Any] | None = None

            if not _runs_automatic_baseline(stage, trials):
                try:
                    parameters, proposal, review, agent_trace = self._propose_candidate(
                        stage, parameters, trials, reference
                    )
                except AgentResponseError as exc:
                    error_path = self.output_dir / "last_agent_error.json"
                    error_record = {
                        "stage": stage,
                        "reference_trial_id": reference.get("trial_id"),
                        **exc.as_dict(),
                    }
                    write_json(error_path, error_record)
                    blocked = {
                        "current_stage": "agent_response_blocked",
                        "resume_stage": stage,
                        "last_trial_id": len(trials),
                        "history_path": str(self.history_path),
                        "reference_trial_id": reference.get("trial_id"),
                        "agent_role": exc.role,
                        "reason": exc.reason,
                        "error_path": str(error_path),
                    }
                    write_json(self.state_path, blocked)
                    _stream_orchestrator_event(
                        self.config, "agent_response_blocked", blocked
                    )
                    break
                if proposal.get("decision") == "blocked":
                    write_json(
                        self.state_path,
                        {
                            "current_stage": "proposal_blocked",
                            "resume_stage": stage,
                            "last_trial_id": len(trials),
                            "history_path": str(self.history_path),
                            "reference_trial_id": reference.get("trial_id"),
                            "reason": proposal.get("reason"),
                            "rejection_path": proposal.get("rejection_path"),
                        },
                    )
                    break
                if proposal.get("decision") == "stop":
                    transition_trigger = copy.deepcopy(proposal)
                    transition = _next_stage_baseline(stage, trials)
                    if transition is None:
                        write_json(
                            self.state_path,
                            {
                                "current_stage": "stage_transition_blocked",
                                "resume_stage": stage,
                                "last_trial_id": len(trials),
                                "history_path": str(self.history_path),
                                "reason": (
                                    "Agent stopped the current stage, but no successful "
                                    "reference satisfies the next stage prerequisites"
                                ),
                                "proposal": transition_trigger,
                            },
                        )
                        break
                    stage, parameters, reference = transition
                    proposal = _baseline_proposal(
                        reference,
                        "automatic baseline after Agent stopped the previous stage",
                        transition_trigger,
                    )

            history_limit = int(self.config.get("history_prompt_trials", 8))

            resume_checkpoint = None
            if stage == "confirm":
                resume_checkpoint = self._confirm_resume_checkpoint(reference)

            def decide_train_health(context: Mapping[str, Any]) -> dict[str, Any]:
                enriched = dict(context)
                enriched["recent_trials"] = read_trial_indexes(self.history_path)[
                    -history_limit:
                ]
                run = self.agents.assess_health(enriched)
                return {"decision": run.result, "trace": run.as_trace()}

            report = run_trial(
                parameters,
                self.config,
                trial_id,
                stage,
                trial_budget(stage, self.config),
                dry_run=dry_run,
                health_decider=decide_train_health,
                resume_checkpoint=resume_checkpoint,
            )
            report["proposal"] = proposal
            report["feasibility"] = review
            if agent_trace is not None:
                report["agent_trace"] = agent_trace
            if not dry_run:
                trial_dir = self.output_dir / "trials" / f"{trial_id:04d}"
                write_json(
                    trial_dir / "decision.json",
                    {
                        "proposal": proposal,
                        "feasibility": review,
                        "diagnosis": (
                            agent_trace.get("diagnosis_summary")
                            if isinstance(agent_trace, Mapping)
                            else None
                        ),
                    },
                )
                write_json(trial_dir / "agent_trace.json", agent_trace or {})
                write_json(
                    trial_dir / "parameter_groups.json",
                    parameter_groups(report.get("parameters", {}), stage),
                )
                index = build_trial_index(
                    report,
                    stability_healthy=(
                        stability_healthy(report, self.config)
                        if stage == "stability_tuning"
                        else None
                    ),
                )
                compact_report = compact_trial_report(report, index)
                compact_report["artifacts"] = copy.deepcopy(index["artifacts"])
                write_json(trial_dir / "trial_report.json", compact_report)
                append_jsonl(self.history_path, index)
                if stage == "confirm":
                    write_json(self.output_dir / "final_result.json", compact_report)
            produced.append(report)
            write_json(
                self.state_path,
                {
                    "current_stage": stage if dry_run else determine_stage(self.trial_indexes(), self.config),
                    "last_trial_id": trial_id,
                    "history_path": str(self.history_path),
                },
            )
            if dry_run:
                break
        return produced
