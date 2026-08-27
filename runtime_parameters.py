from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


TRAINING_DYNAMIC_KEY = "actor_rollout_ref.actor.use_dynamic_bsz"
TRAINING_MICRO_KEY = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
TRAINING_MAX_TOKENS_KEY = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"

ACTOR_LOG_PROB_DYNAMIC_KEY = (
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"
)
ACTOR_LOG_PROB_MICRO_KEY = (
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
)
ACTOR_LOG_PROB_MAX_TOKENS_KEY = (
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"
)

REF_LOG_PROB_DYNAMIC_KEY = "actor_rollout_ref.ref.log_prob_use_dynamic_bsz"
REF_LOG_PROB_MICRO_KEY = (
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
)
REF_LOG_PROB_MAX_TOKENS_KEY = (
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu"
)

BATCHING_PHASE_KEYS = {
    "training": {
        "dynamic": TRAINING_DYNAMIC_KEY,
        "micro": TRAINING_MICRO_KEY,
        "max_tokens": TRAINING_MAX_TOKENS_KEY,
    },
    "actor_log_prob": {
        "dynamic": ACTOR_LOG_PROB_DYNAMIC_KEY,
        "micro": ACTOR_LOG_PROB_MICRO_KEY,
        "max_tokens": ACTOR_LOG_PROB_MAX_TOKENS_KEY,
    },
    "ref_log_prob": {
        "dynamic": REF_LOG_PROB_DYNAMIC_KEY,
        "micro": REF_LOG_PROB_MICRO_KEY,
        "max_tokens": REF_LOG_PROB_MAX_TOKENS_KEY,
    },
}

BATCHING_PARAMETER_KEYS = frozenset(
    key for phase in BATCHING_PHASE_KEYS.values() for key in phase.values()
)


def runtime_parameter_values(log_facts: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(log_facts, Mapping):
        return {}
    runtime = log_facts.get("runtime_parameters")
    if not isinstance(runtime, Mapping) or runtime.get("available") is not True:
        return {}
    values = runtime.get("values")
    return copy.deepcopy(dict(values)) if isinstance(values, Mapping) else {}


def parameter_value_views(
    parameters: Mapping[str, Any],
    log_facts: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    runtime_values = runtime_parameter_values(log_facts)
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        explicitly_configured = key in parameters
        effective_available = key in runtime_values
        result[key] = {
            "configured_value": copy.deepcopy(parameters.get(key)),
            "explicitly_configured": explicitly_configured,
            "effective_value": copy.deepcopy(runtime_values.get(key)),
            "effective_source": (
                "train.log:resolved_hydra_config"
                if effective_available
                else "unavailable"
            ),
        }
    return result


def resolve_batching_parameters(
    parameters: Mapping[str, Any],
    runtime_values: Mapping[str, Any] | None = None,
    *,
    changed_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve batching values from explicit overrides or the observed runtime."""

    observed = runtime_values if isinstance(runtime_values, Mapping) else {}
    changed = set(changed_keys)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    for key in BATCHING_PARAMETER_KEYS:
        if key in parameters:
            values[key] = copy.deepcopy(parameters[key])
            sources[key] = (
                "candidate_override"
                if key in changed
                else "reference_override"
            )
        elif key in observed:
            values[key] = copy.deepcopy(observed[key])
            sources[key] = "observed_runtime"
        else:
            values[key] = None
            sources[key] = "unavailable"

    phases: dict[str, dict[str, Any]] = {}
    for phase, keys in BATCHING_PHASE_KEYS.items():
        phases[phase] = {
            "dynamic": values[keys["dynamic"]],
            "micro_batch_size_per_gpu": values[keys["micro"]],
            "max_token_len_per_gpu": values[keys["max_tokens"]],
            "sources": {
                "dynamic": sources[keys["dynamic"]],
                "micro_batch_size_per_gpu": sources[keys["micro"]],
                "max_token_len_per_gpu": sources[keys["max_tokens"]],
            },
        }
    return {"values": values, "sources": sources, "phases": phases}


def effective_from_value(
    key: str,
    runtime_values: Mapping[str, Any] | None,
) -> Any:
    if isinstance(runtime_values, Mapping) and key in runtime_values:
        return copy.deepcopy(runtime_values[key])
    return None
