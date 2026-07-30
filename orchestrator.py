from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

from agents import AgentResponseError, AgentSet
from config_utils import append_jsonl, apply_changes, read_jsonl, write_json
from prompting import rejection_feedback
from runner import run_trial
from validator import editable_parameters, validate_candidate


def _metric_mean(trial: Mapping[str, Any], *path: str) -> float | None:
    value: Any = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


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
        reward = _last_complete_stability_value(trial, "critic/rewards/mean")
        if reward is not None:
            candidates.append((reward, trial))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def hardware_plateaued(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> bool:
    successful = _successful(_hardware_trials(trials))
    minimum = int(config.get("min_hardware_trials", 2))
    plateau_rounds = int(config.get("plateau_rounds", 2))
    if len(successful) < minimum + plateau_rounds:
        return False
    scores = [_metric_mean(trial, "performance", "throughput") for trial in successful]
    scores = [score for score in scores if score is not None]
    if len(scores) < minimum + plateau_rounds:
        return False
    threshold = float(config.get("min_throughput_improvement", 0.02))
    best_before = max(scores[:-plateau_rounds])
    return all(score <= best_before * (1.0 + threshold) for score in scores[-plateau_rounds:])


def stability_healthy(trial: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if trial.get("result") != "success":
        return False
    reward_points = _complete_stability_points(trial, "critic/rewards/mean")
    slope = None
    if len(reward_points) >= 2:
        previous, current = reward_points[-2:]
        delta_step = current[0] - previous[0]
        slope = (current[1] - previous[1]) / delta_step if delta_step else 0.0
    kl_points = _complete_stability_points(trial, "actor/ppo_kl")
    kl_max = max((value for _, value in kl_points), default=None)
    if slope is not None and slope < float(config.get("reward_collapse_slope", -0.01)):
        return False
    if kl_max is not None and kl_max > float(config.get("kl_warning", 0.1)):
        return False
    return True


def determine_stage(trials: list[dict[str, Any]], config: Mapping[str, Any]) -> str:
    if _confirm_trials(trials):
        return "done"
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
        "stability",
        "health_monitor",
        "health_decisions",
        "termination",
        "diagnosis",
        "failure_phase",
        "proposal",
        "feasibility",
        "log_path",
    ]
    return {key: copy.deepcopy(trial[key]) for key in keys if key in trial}


def _complete_stability_points(trial: Mapping[str, Any], metric: str) -> list[tuple[int, float]]:
    """Read complete report windows for a metric, retaining old reports as fallback."""
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

    # Existing histories retain the legacy summary.  It is only a migration
    # fallback; newly written reports always use complete metric windows.
    legacy_key = {
        "critic/rewards/mean": "reward",
        "actor/ppo_kl": "actor_ppo_kl",
    }.get(metric)
    legacy_value = _metric_mean(trial, "stability", legacy_key) if legacy_key else None
    return [(0, legacy_value)] if legacy_value is not None else []


def _last_complete_stability_value(trial: Mapping[str, Any], metric: str) -> float | None:
    points = _complete_stability_points(trial, metric)
    return points[-1][1] if points else None


def _trial_stability_series(trial: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trial, Mapping):
        return None
    stability = trial.get("stability")
    if not isinstance(stability, Mapping) or not isinstance(stability.get("windows"), list):
        return None
    series = copy.deepcopy(dict(stability))
    series["trial_id"] = trial.get("trial_id")
    series["stage"] = trial.get("stage")
    return series


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


def _stream_orchestrator_event(
    config: Mapping[str, Any], event: str, payload: Mapping[str, Any]
) -> None:
    if not bool(config.get("stream_agent_events", True)):
        return
    import json

    print(
        f"\n[Orchestrator] {event}\n"
        + json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
        flush=True,
    )


def _normalize_proposal_changes(
    proposal: Mapping[str, Any],
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    """Validate the Agent's provenance and derive executable target values."""
    violations: list[str] = []
    if proposal.get("decision") != "modify":
        violations.append("decision must be modify, keep, or stop; change objects require decision=modify")
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
        is_explicitly_configured = parameter in current
        actual_from = current.get(parameter)
        declared_from = raw_detail.get("from")
        target = raw_detail.get("to")
        reason = raw_detail.get("reason")
        if is_explicitly_configured and declared_from != actual_from:
            violations.append(
                f"{parameter} from must equal current value {actual_from!r}, got {declared_from!r}"
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
            "to": target,
            "reason": reason.strip() if isinstance(reason, str) else reason,
        }
    return targets, details, violations


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
        return read_jsonl(self.history_path)

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
                raise RuntimeError("stability tuning requires a successful hardware trial")
            reason = (
                "best successful stability trial by reward"
                if stability is not None
                else "best successful hardware trial used as stability baseline"
            )
            return dict(best["parameters"]), _reference_descriptor(best, reason)
        if stage == "confirm":
            stability = best_stability_trial(trials)
            best = stability or best_hardware_trial(trials)
            if not best:
                raise RuntimeError("confirmation requires a successful candidate")
            reason = (
                "best successful stability trial selected for confirmation"
                if stability is not None
                else "best successful hardware trial selected for confirmation"
            )
            return dict(best["parameters"]), _reference_descriptor(best, reason)
        raise RuntimeError(f"unsupported stage: {stage}")

    def _starting_parameters(self, stage: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
        return self._starting_point(stage, trials)[0]

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
        context = {
            "current_stage": stage,
            "mode": "failure_repair" if diagnosis else stage,
            "current_parameters": dict(current),
            "reference_trial": copy.deepcopy(dict(reference)),
            "reference_stability_series": _trial_stability_series(
                next(
                    (
                        trial
                        for trial in trials
                        if trial.get("trial_id") == reference.get("trial_id")
                    ),
                    None,
                )
            ),
            "editable_parameters": editable_parameters(stage),
            "constraints": {
                "max_parameter_changes": self.config.get("max_parameter_changes", 3),
                "preserve_hardware_token_budget": self.config.get("preserve_hardware_token_budget", True),
                "resource_memory_limit_pct": self.config.get("resource_memory_limit_pct", 95.0),
                "throughput_memory_limit_pct": self.config.get("throughput_memory_limit_pct", 92.0),
            },
            "diagnosis": diagnosis,
            "recent_trials": [_compact_trial(trial) for trial in trials[-history_limit:]],
        }
        proposal_conversation = None
        trace: dict[str, Any] = {
            "diagnosis": diagnosis_trace,
            "proposal_conversation": None,
            "feasibility_reviews": [],
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
            if decision in {"keep", "stop"}:
                proposal.setdefault("reference_trial_id", reference.get("trial_id"))
                proposal["reference_trial"] = copy.deepcopy(dict(reference))
                return (
                    dict(current),
                    proposal,
                    {"verdict": "valid", "reason": "no parameter change"},
                    trace,
                )
            target_changes, change_details, proposal_violations = _normalize_proposal_changes(
                proposal, current, reference
            )
            if proposal_violations:
                candidate = dict(current)
                validation = {"valid": False, "violations": proposal_violations}
                rejections.append({"attempt": attempt, "source": "deterministic_validator", "proposal": proposal, **validation})
                _stream_orchestrator_event(
                    self.config,
                    "proposal_rejected",
                    {"attempt": attempt, "source": "proposal_schema", **validation},
                )
                proposal_conversation.add_user_message(
                    rejection_feedback(attempt, proposal, candidate, "deterministic_validator", validation)
                )
                trace["proposal_conversation"] = proposal_conversation.as_trace()
                continue
            proposal["changes"] = change_details
            proposal["target_changes"] = dict(target_changes)
            proposal["reference_trial"] = copy.deepcopy(dict(reference))
            candidate = apply_changes(current, target_changes)
            validation = validate_candidate(
                candidate, target_changes, stage, self.config, self.base_parameters, trials
            )
            if not validation.valid:
                validation_result = validation.as_dict()
                if any("not editable in stage" in row for row in validation_result["violations"]):
                    validation_result["editable_parameters"] = editable_parameters(stage)
                rejections.append(
                    {
                        "attempt": attempt,
                        "source": "deterministic_validator",
                        "proposal": proposal,
                        **validation_result,
                    }
                )
                _stream_orchestrator_event(
                    self.config,
                    "proposal_rejected",
                    {
                        "attempt": attempt,
                        "source": "deterministic_validator",
                        **validation_result,
                    },
                )
                proposal_conversation.add_user_message(
                    rejection_feedback(
                        attempt,
                        proposal,
                        candidate,
                        "deterministic_validator",
                        validation_result,
                    )
                )
                trace["proposal_conversation"] = proposal_conversation.as_trace()
                continue
            review_run = self.agents.review(
                {
                    "current_stage": stage,
                    "current_parameters": dict(current),
                    "candidate_parameters": candidate,
                    "changes": copy.deepcopy(change_details),
                    "target_changes": dict(target_changes),
                    "reference_trial": copy.deepcopy(dict(reference)),
                    "proposal_reason": proposal.get("reason"),
                    "last_trial": _compact_trial(trials[-1]) if trials else None,
                    "recent_trials": [_compact_trial(trial) for trial in trials[-history_limit:]],
                    "diagnosis": diagnosis,
                    "memory_limits": {
                        "resource": self.config.get("resource_memory_limit_pct", 95.0),
                        "throughput": self.config.get("throughput_memory_limit_pct", 92.0),
                    },
                }
            )
            review = review_run.result
            trace["feasibility_reviews"].append({"attempt": attempt, **review_run.as_trace()})
            if review.get("verdict") == "valid":
                trace["proposal_conversation"] = proposal_run.as_trace()
                return candidate, proposal, review, trace
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
                rejection_feedback(attempt, proposal, candidate, "feasibility_agent", review)
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
            trials = self.trials()
            stage = determine_stage(trials, self.config)
            if stage in {"done", "stopped_unstable"}:
                write_json(
                    self.state_path,
                    {
                        "current_stage": stage,
                        "last_trial_id": len(trials),
                        "history_path": str(self.history_path),
                    },
                )
                break
            trial_id = len(trials) + 1
            parameters, reference = self._starting_point(stage, trials)
            proposal: dict[str, Any] = {
                "decision": "keep",
                "reason": "stage baseline",
                "reference_trial_id": reference.get("trial_id"),
                "reference_trial": copy.deepcopy(reference),
                "changes": {},
                "expected_effect": {},
            }
            review: dict[str, Any] = {"verdict": "valid", "reason": "stage baseline"}
            agent_trace: dict[str, Any] | None = None

            first_hardware = not _hardware_trials(trials)
            if not first_hardware and stage != "confirm":
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
                if proposal.get("decision") in {"keep", "stop"}:
                    # The first stability trial now receives a proposal context
                    # based on the selected hardware trial.  A justified
                    # ``keep`` still needs to run that no-change stability
                    # baseline; it is not a terminal "no candidate" state.
                    if stage == "stability_tuning" and not _stability_trials(trials):
                        pass
                    elif stage.startswith("hardware") and best_hardware_trial(trials):
                        stage = "stability_tuning"
                        selected = best_hardware_trial(trials)
                        parameters = dict(selected["parameters"])
                        reference = _reference_descriptor(
                            selected, "best successful hardware trial used as stability baseline"
                        )
                    elif stage == "stability_tuning" and best_stability_trial(trials):
                        stage = "confirm"
                        selected = best_stability_trial(trials)
                        parameters = dict(selected["parameters"])
                        reference = _reference_descriptor(
                            selected, "best successful stability trial selected for confirmation"
                        )
                    else:
                        write_json(
                            self.state_path,
                            {
                                "current_stage": "stopped_no_candidate",
                                "last_trial_id": len(trials),
                                "history_path": str(self.history_path),
                                "proposal": proposal,
                            },
                        )
                        break
                    proposal = {
                        "decision": "keep",
                        "reason": "stage transition baseline",
                        "reference_trial_id": reference.get("trial_id"),
                        "reference_trial": copy.deepcopy(reference),
                        "changes": {},
                        "expected_effect": {},
                    }

            history_limit = int(self.config.get("history_prompt_trials", 8))

            def decide_train_health(context: Mapping[str, Any]) -> dict[str, Any]:
                enriched = dict(context)
                enriched["recent_trials"] = [
                    _compact_trial(trial) for trial in trials[-history_limit:]
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
            )
            report["proposal"] = proposal
            report["feasibility"] = review
            if agent_trace is not None:
                report["agent_trace"] = agent_trace
            if not dry_run:
                write_json(self.output_dir / "trials" / f"{trial_id:04d}" / "trial_report.json", report)
                append_jsonl(self.history_path, report)
                if stage == "confirm":
                    write_json(self.output_dir / "final_result.json", report)
            produced.append(report)
            write_json(
                self.state_path,
                {
                    "current_stage": stage if dry_run else determine_stage(self.trials(), self.config),
                    "last_trial_id": trial_id,
                    "history_path": str(self.history_path),
                },
            )
            if dry_run:
                break
        return produced
