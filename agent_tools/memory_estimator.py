from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def _number(parameters: Mapping[str, Any], key: str, default: float) -> float:
    value = parameters.get(key)
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{key} must be numeric or null, got {type(value).__name__}")


def _enabled(parameters: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = parameters.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be boolean or null, got {type(value).__name__}")


def _recompute_activation_factor(parameters: Mapping[str, Any]) -> float:
    value = parameters.get(
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity"
    )
    if value in (None, "none", "None", False):
        return 1.0
    # Recomputation only reduces the activation component, not model or
    # optimizer state.  Full recomputation saves more activations than a
    # selective policy.
    return 0.40 if str(value).lower() == "full" else 0.70


def _positive_ratio(candidate: float, reference: float, name: str) -> float:
    if candidate <= 0 or reference <= 0:
        raise ValueError(f"{name} values must be positive")
    return candidate / reference


def _sequence_length(parameters: Mapping[str, Any]) -> float:
    prompt = _number(parameters, "data.max_prompt_length", 1024)
    response = _number(parameters, "data.max_response_length", 4096)
    return max(1.0, prompt + response)


def _toggle_factor(
    parameters: Mapping[str, Any], key: str, enabled_factor: float
) -> float:
    return enabled_factor if _enabled(parameters, key, False) else 1.0


def _changed_keys(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> set[str]:
    keys = set(reference) | set(candidate)
    return {key for key in keys if reference.get(key) != candidate.get(key)}


def _bounded(value: float, lower: float = 0.25, upper: float = 4.0) -> float:
    return max(lower, min(upper, value))


def _rollout_projection(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    utilization_key = "actor_rollout_ref.rollout.gpu_memory_utilization"
    reference_util = _number(reference, utilization_key, 0.6)
    candidate_util = _number(candidate, utilization_key, 0.6)

    # vLLM's configured budget dominates the observed rollout peak.  The 0.86
    # elasticity is anchored by the local C550 observations
    # 0.5→0.7→0.8 == 59.15%→79.06%→88.97%.
    budget_ratio = _positive_ratio(
        candidate_util, reference_util, utilization_key
    ) ** 0.86

    relevant = {
        "actor_rollout_ref.model.path",
        utilization_key,
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        "actor_rollout_ref.rollout.max_num_seqs",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.enable_prefix_caching",
        "actor_rollout_ref.rollout.enable_chunked_prefill",
        "actor_rollout_ref.rollout.free_cache_engine",
        "data.max_prompt_length",
        "data.max_response_length",
    }
    changed = _changed_keys(reference, candidate) & relevant
    # Scheduler limits are capacity ceilings rather than allocations.  Keep
    # them out of the point estimate until matched trial evidence exists.
    uncalibrated = sorted(changed - {utilization_key})
    uncertainty = min(10.0, 1.5 + 1.5 * len(uncalibrated))
    return {
        "ratio": _bounded(budget_ratio),
        "model": "vllm_budget_relative",
        "drivers": {
            "gpu_memory_utilization": {
                "from": reference_util,
                "to": candidate_util,
                "elasticity": 0.86,
                "ratio": round(budget_ratio, 6),
            }
        },
        "uncalibrated_changes": uncalibrated,
        "uncertainty_pct": uncertainty,
        "confidence": "low" if uncalibrated else "medium",
    }


def _log_prob_projection(
    phase: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    actor_phase = phase == "actor_log_prob"
    micro_key = (
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
        if actor_phase
        else "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
    )
    # The colocated reference worker reuses the actor Megatron topology.  The
    # ref TP/PP/SP Hydra values may still be present for provenance, but they
    # are not runtime tuning knobs and must not influence this projection.
    tp_key = "actor_rollout_ref.actor.megatron.tensor_model_parallel_size"
    pp_key = "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"
    sequence_parallel_key = "actor_rollout_ref.actor.megatron.sequence_parallel"

    micro_ratio = _positive_ratio(
        _number(candidate, micro_key, 1),
        _number(reference, micro_key, 1),
        micro_key,
    )
    sequence_ratio = _positive_ratio(
        _sequence_length(candidate),
        _sequence_length(reference),
        "configured sequence length",
    )
    reference_parallel = max(
        1.0, _number(reference, tp_key, 1) * _number(reference, pp_key, 1)
    )
    candidate_parallel = max(
        1.0, _number(candidate, tp_key, 1) * _number(candidate, pp_key, 1)
    )
    state_ratio = _bounded(reference_parallel / candidate_parallel)

    packing_ratio = 1.0
    if actor_phase:
        packing_ratio *= _toggle_factor(
            candidate, sequence_parallel_key, 0.80
        ) / _toggle_factor(reference, sequence_parallel_key, 0.80)
        packing_ratio *= _toggle_factor(
            candidate,
            "actor_rollout_ref.actor.megatron.use_remove_padding",
            0.85,
        ) / _toggle_factor(
            reference,
            "actor_rollout_ref.actor.megatron.use_remove_padding",
            0.85,
        )
        packing_ratio *= _toggle_factor(
            candidate, "actor_rollout_ref.actor.use_dynamic_bsz", 0.90
        ) / _toggle_factor(
            reference, "actor_rollout_ref.actor.use_dynamic_bsz", 0.90
        )
    else:
        packing_ratio *= _toggle_factor(
            candidate, sequence_parallel_key, 0.80
        ) / _toggle_factor(reference, sequence_parallel_key, 0.80)

    activation_ratio = _bounded(
        micro_ratio
        * sequence_ratio
        * (reference_parallel / candidate_parallel) ** 0.5
        * packing_ratio
    )
    # Only the sharded model and activation components scale.  Runtime and
    # allocator residency remain fixed at the reference-trial level.
    fixed_share, model_share, activation_share = 0.55, 0.25, 0.20
    total_ratio = (
        fixed_share
        + model_share * state_ratio
        + activation_share * activation_ratio
    )

    recognized = {
        micro_key,
        tp_key,
        pp_key,
        "data.max_prompt_length",
        "data.max_response_length",
        sequence_parallel_key,
    }
    if actor_phase:
        recognized |= {
            "actor_rollout_ref.actor.megatron.use_remove_padding",
            "actor_rollout_ref.actor.use_dynamic_bsz",
        }
    relevant = recognized | {
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.rollout.free_cache_engine",
        (
            "actor_rollout_ref.actor.megatron.param_offload"
            if actor_phase
            else "actor_rollout_ref.ref.megatron.param_offload"
        ),
    }
    changed = _changed_keys(reference, candidate) & relevant
    uncalibrated = sorted(changed - recognized)
    known_changes = changed & recognized
    base_uncertainty = (
        (3.0 if known_changes else 2.0)
        if actor_phase
        else (5.0 if known_changes else 4.0)
    )
    uncertainty = min(
        12.0,
        base_uncertainty + 2.0 * len(uncalibrated),
    )
    return {
        "ratio": _bounded(total_ratio),
        "model": "fixed_model_activation_components",
        "component_shares": {
            "fixed_runtime": fixed_share,
            "sharded_model": model_share,
            "activation_workspace": activation_share,
        },
        "drivers": {
            "micro_batch_ratio": round(micro_ratio, 6),
            "configured_sequence_ratio": round(sequence_ratio, 6),
            "parallel_state_ratio": round(state_ratio, 6),
            "activation_ratio": round(activation_ratio, 6),
        },
        "uncalibrated_changes": uncalibrated,
        "uncertainty_pct": uncertainty,
        "confidence": "low" if uncalibrated else "medium",
    }


def _training_projection(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    micro_key = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
    tp_key = "actor_rollout_ref.actor.megatron.tensor_model_parallel_size"
    pp_key = "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"
    sequence_parallel_key = (
        "actor_rollout_ref.actor.megatron.sequence_parallel"
    )
    distributed_optimizer_key = (
        "actor_rollout_ref.actor.megatron.use_distributed_optimizer"
    )
    optimizer_offload_key = (
        "actor_rollout_ref.actor.megatron.optimizer_offload"
    )
    recompute_key = (
        "actor_rollout_ref.actor.megatron."
        "override_transformer_config.recompute_granularity"
    )
    entropy_key = "actor_rollout_ref.actor.entropy_coeff"

    micro_ratio = _positive_ratio(
        _number(candidate, micro_key, 1),
        _number(reference, micro_key, 1),
        micro_key,
    )
    sequence_ratio = _positive_ratio(
        _sequence_length(candidate),
        _sequence_length(reference),
        "configured sequence length",
    )
    reference_parallel = max(
        1.0, _number(reference, tp_key, 1) * _number(reference, pp_key, 1)
    )
    candidate_parallel = max(
        1.0, _number(candidate, tp_key, 1) * _number(candidate, pp_key, 1)
    )
    state_ratio = _bounded(reference_parallel / candidate_parallel)

    distributed_optimizer_ratio = _toggle_factor(
        candidate, distributed_optimizer_key, 0.60
    ) / _toggle_factor(reference, distributed_optimizer_key, 0.60)
    optimizer_offload_ratio = _toggle_factor(
        candidate, optimizer_offload_key, 0.25
    ) / _toggle_factor(reference, optimizer_offload_key, 0.25)
    optimizer_ratio = _bounded(
        state_ratio * distributed_optimizer_ratio * optimizer_offload_ratio
    )

    activation_ratio = _bounded(
        micro_ratio
        * sequence_ratio
        * (reference_parallel / candidate_parallel) ** 0.5
        * (
            _toggle_factor(candidate, sequence_parallel_key, 0.80)
            / _toggle_factor(reference, sequence_parallel_key, 0.80)
        )
        * (
            _toggle_factor(
                candidate,
                "actor_rollout_ref.actor.megatron.use_remove_padding",
                0.85,
            )
            / _toggle_factor(
                reference,
                "actor_rollout_ref.actor.megatron.use_remove_padding",
                0.85,
            )
        )
        * (
            _toggle_factor(
                candidate, "actor_rollout_ref.actor.use_dynamic_bsz", 0.90
            )
            / _toggle_factor(
                reference, "actor_rollout_ref.actor.use_dynamic_bsz", 0.90
            )
        )
        * (
            _recompute_activation_factor(candidate)
            / _recompute_activation_factor(reference)
        )
        * (
            (1.0 if float(candidate.get(entropy_key, 0.0) or 0.0) != 0.0 else 0.85)
            / (1.0 if float(reference.get(entropy_key, 0.0) or 0.0) != 0.0 else 0.85)
        )
    )

    fixed_share = 0.30
    model_gradient_share = 0.25
    optimizer_share = 0.25
    activation_share = 0.20
    total_ratio = (
        fixed_share
        + model_gradient_share * state_ratio
        + optimizer_share * optimizer_ratio
        + activation_share * activation_ratio
    )

    recognized = {
        micro_key,
        tp_key,
        pp_key,
        sequence_parallel_key,
        distributed_optimizer_key,
        optimizer_offload_key,
        recompute_key,
        entropy_key,
        "actor_rollout_ref.actor.megatron.use_remove_padding",
        "actor_rollout_ref.actor.use_dynamic_bsz",
        "data.max_prompt_length",
        "data.max_response_length",
    }
    relevant = recognized | {
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.megatron.param_offload",
        "actor_rollout_ref.rollout.free_cache_engine",
        (
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_method"
        ),
        (
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_num_layers"
        ),
        (
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_modules"
        ),
    }
    changed = _changed_keys(reference, candidate) & relevant
    uncalibrated = sorted(changed - recognized)
    known_changes = changed & recognized
    uncertainty = min(
        15.0,
        (4.0 if known_changes else 2.0) + 2.0 * len(uncalibrated),
    )
    return {
        "ratio": _bounded(total_ratio),
        "model": "fixed_model_optimizer_activation_components",
        "component_shares": {
            "fixed_runtime": fixed_share,
            "model_and_gradients": model_gradient_share,
            "optimizer_state": optimizer_share,
            "activation_workspace": activation_share,
        },
        "drivers": {
            "micro_batch_ratio": round(micro_ratio, 6),
            "configured_sequence_ratio": round(sequence_ratio, 6),
            "parallel_state_ratio": round(state_ratio, 6),
            "optimizer_ratio": round(optimizer_ratio, 6),
            "activation_ratio": round(activation_ratio, 6),
            "calculate_entropy": float(candidate.get(entropy_key, 0.0) or 0.0)
            != 0.0,
            "entropy_workspace_ratio": round(
                (1.0 if float(candidate.get(entropy_key, 0.0) or 0.0) != 0.0 else 0.85)
                / (1.0 if float(reference.get(entropy_key, 0.0) or 0.0) != 0.0 else 0.85),
                6,
            ),
        },
        "uncalibrated_changes": uncalibrated,
        "uncertainty_pct": uncertainty,
        "confidence": "low" if changed else "medium",
    }


def _phase_projection(
    phase: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if phase == "rollout":
        return _rollout_projection(reference, candidate)
    if phase in {"actor_log_prob", "ref_log_prob"}:
        return _log_prob_projection(phase, reference, candidate)
    return _training_projection(reference, candidate)


def _phase_peaks(trial: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    memory = trial.get("memory_by_phase_pct")
    if not isinstance(memory, Mapping):
        return result
    for phase in PHASES:
        value = memory.get(phase)
        if isinstance(value, Mapping):
            value = value.get("max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[phase] = float(value)
    return result


def _phase_peaks_mib(trial: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    memory = trial.get("memory_by_phase_mib")
    if not isinstance(memory, Mapping):
        structured = trial.get("structured_metrics")
        resource = structured.get("resource") if isinstance(structured, Mapping) else None
        memory = resource.get("by_phase") if isinstance(resource, Mapping) else None
    if not isinstance(memory, Mapping):
        return result
    for phase in PHASES:
        value = memory.get(phase)
        if isinstance(value, Mapping):
            value = value.get("max_used_mib", value.get("max"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[phase] = float(value)
    return result


def _same_parameters(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _reference_trial(
    current: Mapping[str, Any], trials: Sequence[Mapping[str, Any]], reference_trial_id: int | None
) -> Mapping[str, Any] | None:
    observed = [trial for trial in trials if _phase_peaks(trial) and isinstance(trial.get("parameters"), Mapping)]
    if reference_trial_id is not None:
        return next((trial for trial in observed if trial.get("trial_id") == reference_trial_id), None)
    exact = [trial for trial in observed if _same_parameters(trial.get("parameters", {}), current)]
    return (exact or observed)[-1] if observed else None


def estimate_phase_memory(
    current_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]] = (),
    memory_limit_pct: float = 92.0,
    reference_trial_id: int | None = None,
    memory_limit_mib: float | None = None,
) -> dict[str, Any]:
    reference = _reference_trial(current_parameters, trials, reference_trial_id)
    reference_parameters: Mapping[str, Any] = current_parameters
    peaks: dict[str, float] = {}
    peaks_mib: dict[str, float] = {}
    if reference:
        reference_parameters = reference.get("parameters", current_parameters)
        peaks = _phase_peaks(reference)
        peaks_mib = _phase_peaks_mib(reference)

    phases: dict[str, Any] = {}
    for phase in PHASES:
        projection = _phase_projection(
            phase, reference_parameters, candidate_parameters
        )
        raw_ratio = float(projection["ratio"])
        ratio = _bounded(raw_ratio, 0.2, 5.0)
        uncertainty = float(projection["uncertainty_pct"])
        projected = peaks.get(phase)
        projected = projected * ratio if projected is not None else None
        projected_mib = peaks_mib.get(phase)
        projected_mib = projected_mib * ratio if projected_mib is not None else None
        capacity_mib = (
            peaks_mib.get(phase) * 100.0 / peaks.get(phase)
            if phase in peaks_mib and phase in peaks and peaks[phase] > 0
            else None
        )
        uncertainty_mib = (
            uncertainty * capacity_mib / 100.0 if capacity_mib is not None else None
        )
        if projected is None:
            risk = "unknown_without_observed_anchor"
            headroom = None
            upper_bound = None
            upper_headroom = None
        else:
            headroom = memory_limit_pct - projected
            upper_bound = projected + uncertainty
            upper_headroom = memory_limit_pct - upper_bound
            if memory_limit_mib is not None and projected_mib is not None:
                upper_mib_for_risk = projected_mib + float(uncertainty_mib or 0.0)
                absolute_headroom = memory_limit_mib - upper_mib_for_risk
                risk = "high" if absolute_headroom <= 0 else (
                    "watch" if absolute_headroom < 2048 else "low"
                )
            else:
                risk = (
                    "high"
                    if upper_bound >= memory_limit_pct
                    else ("watch" if upper_headroom < 5.0 else "low")
                )
        phases[phase] = {
            "reference_pct": round(peaks[phase], 2) if phase in peaks else None,
            "pressure_ratio": round(ratio, 3),
            "raw_pressure_ratio": round(raw_ratio, 6),
            "projected_pct": round(projected, 2) if projected is not None else None,
            "uncertainty_pct": round(uncertainty, 2),
            "upper_bound_pct": (
                round(upper_bound, 2) if upper_bound is not None else None
            ),
            "reference_mib": round(peaks_mib[phase], 2) if phase in peaks_mib else None,
            "projected_mib": round(projected_mib, 2) if projected_mib is not None else None,
            "uncertainty_mib": round(uncertainty_mib, 2) if uncertainty_mib is not None else None,
            "upper_bound_mib": (
                round(projected_mib + uncertainty_mib, 2)
                if projected_mib is not None and uncertainty_mib is not None
                else None
            ),
            "headroom_to_limit_mib": (
                round(memory_limit_mib - projected_mib, 2)
                if memory_limit_mib is not None and projected_mib is not None
                else None
            ),
            "upper_headroom_to_limit_mib": (
                round(memory_limit_mib - projected_mib - uncertainty_mib, 2)
                if memory_limit_mib is not None
                and projected_mib is not None
                and uncertainty_mib is not None
                else None
            ),
            "headroom_to_limit_pct": round(headroom, 2) if headroom is not None else None,
            "upper_headroom_to_limit_pct": (
                round(upper_headroom, 2)
                if upper_headroom is not None
                else None
            ),
            "risk": risk,
            "confidence": projection["confidence"],
            "model": projection["model"],
            "component_shares": projection.get("component_shares"),
            "drivers": projection["drivers"],
            "uncalibrated_changes": projection["uncalibrated_changes"],
        }

    phase_confidence = [phase["confidence"] for phase in phases.values()]
    confidence = (
        "low"
        if "low" in phase_confidence
        else ("medium" if reference else "low")
    )
    return {
        "method": (
            "empirical_component_relative"
            if reference
            else "component_pressure_only"
        ),
        "confidence": confidence,
        "memory_limit_pct": memory_limit_pct,
        "memory_limit_mib": memory_limit_mib,
        "reference_trial_id": reference.get("trial_id") if reference else None,
        "phases": phases,
        "limitations": [
            "The estimator anchors a component-aware relative projection to one measured reference trial; it is not a tensor-allocation simulator.",
            "When absolute device capacity is available, risk is evaluated against upper_bound_mib and the effective absolute limit; percentage fields are derived display values.",
            "Scheduler caps and offload transitions without matched history are reported in uncalibrated_changes and widen uncertainty instead of scaling the full peak.",
            "Configured maximum sequence lengths are only workload bounds; actual token distributions may be lower.",
            "Absolute MiB projections require a prior trial with phase-tagged GPU memory observations.",
            "Live SMI snapshots cannot replace rollout/actor/ref/training phase samples.",
            "A real short resource-gate trial remains the final memory authority.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate verl phase memory from an empirical reference trial")
    parser.add_argument("--current", required=True, help="Current parameter JSON")
    parser.add_argument("--candidate", required=True, help="Candidate parameter JSON")
    parser.add_argument("--trials", help="Optional JSON array or JSONL trial history")
    parser.add_argument("--memory-limit-pct", type=float, default=92.0)
    args = parser.parse_args()

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    trials = []
    if args.trials:
        text = Path(args.trials).read_text(encoding="utf-8").strip()
        trials = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]
    print(json.dumps(estimate_phase_memory(current, candidate, trials, args.memory_limit_pct), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
