from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


HARDWARE_PARAMETERS = {
    # Actor training batching.
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",

    # Actor training topology.
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
    "actor_rollout_ref.actor.megatron.sequence_parallel",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.actor.use_dynamic_bsz",

    # Rollout generation.
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.max_num_seqs",

    # Actor old-log-prob.
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    
    # Reference log-prob owns batching only. Its colocated Megatron model
    # follows the actor model-parallel topology in verl 0.7.
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
}

STABILITY_PARAMETERS = {
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.actor.optim.lr_warmup_steps",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.kl_loss_coef",
    "actor_rollout_ref.actor.kl_loss_type",
    "actor_rollout_ref.actor.entropy_coeff",
    "actor_rollout_ref.rollout.n",
}

IGNORED_PARAMETERS = {
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size",
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size",
    "actor_rollout_ref.ref.megatron.sequence_parallel",
}

CROSS_EFFECTS = {
    "actor_rollout_ref.actor.entropy_coeff": [
        "stability",
        "training_memory",
        "training_throughput",
    ],
    "actor_rollout_ref.rollout.n": [
        "stability",
        "rollout_throughput",
        "token_workload",
    ],
}

RANGES = {
    "data.train_batch_size": (1, 8192),
    "data.max_prompt_length": (128, 32768),
    "data.max_response_length": (128, 32768),
    "actor_rollout_ref.actor.optim.lr": (1e-7, 1e-5),
    "actor_rollout_ref.actor.optim.lr_warmup_steps": (0, 10000),
    "actor_rollout_ref.actor.kl_loss_coef": (0.0, 0.2),
    "actor_rollout_ref.actor.entropy_coeff": (0.0, 0.2),
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": (1, 256),
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": (1, 1048576),
    "actor_rollout_ref.actor.ppo_mini_batch_size": (1, 8192),
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": (1, 1024),
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": (1, 1048576),
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": (1, 1024),
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": (1, 1048576),
    "actor_rollout_ref.rollout.gpu_memory_utilization": (0.1, 0.95),
    "actor_rollout_ref.rollout.max_num_batched_tokens": (1024, 1048576),
    "actor_rollout_ref.rollout.max_num_seqs": (1, 4096),
    "actor_rollout_ref.rollout.n": (1, 64),
}

BATCHING_AUTHORITIES = (
    (
        "actor_rollout_ref.actor.use_dynamic_bsz",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    ),
    (
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    ),
    (
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
    ),
)


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "violations": self.violations}


def editable_parameters(stage: str) -> list[str]:
    if stage in {"hardware_repair", "hardware_tuning"}:
        return sorted(HARDWARE_PARAMETERS)
    if stage == "stability_tuning":
        return sorted(STABILITY_PARAMETERS)
    return []


def parameter_groups(parameters: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Classify one executable parameter snapshot from the validator policy."""
    editable = set(editable_parameters(stage))
    throughput = {
        key: parameters.get(key)
        for key in sorted(HARDWARE_PARAMETERS)
        if key in parameters
    }
    stability = {
        key: parameters.get(key)
        for key in sorted(STABILITY_PARAMETERS)
        if key in parameters
    }
    ignored = {
        key: parameters.get(key)
        for key in sorted(IGNORED_PARAMETERS)
        if key in parameters
    }
    fixed = {
        key: value
        for key, value in sorted(parameters.items())
        if key not in editable and key not in IGNORED_PARAMETERS
    }
    return {
        "trial_stage": stage,
        "editable_in_trial_stage": sorted(editable),
        "throughput_tuning": throughput,
        "stability_tuning": stability,
        "fixed": fixed,
        "ignored": ignored,
        "cross_effects": CROSS_EFFECTS,
    }


def hardware_token_budget(parameters: Mapping[str, Any]) -> int:
    return int(parameters["data.train_batch_size"]) * int(parameters["actor_rollout_ref.rollout.n"]) * (
        int(parameters["data.max_prompt_length"]) + int(parameters["data.max_response_length"])
    )


def effective_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the runtime-effective configuration used for duplicate checks."""
    return {
        key: value
        for key, value in parameters.items()
        if key not in IGNORED_PARAMETERS
    }


def validate_candidate(
    parameters: Mapping[str, Any],
    changes: Mapping[str, Any],
    stage: str,
    agent_config: Mapping[str, Any],
    base_parameters: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    locked_parameters: Mapping[str, Any] | None = None,
) -> ValidationResult:
    violations: list[str] = []
    editable = set(editable_parameters(stage))
    max_changes = int(agent_config.get("max_parameter_changes", 3))

    if len(changes) > max_changes:
        violations.append(f"changes contains {len(changes)} parameters; maximum is {max_changes}")

    for key, value in changes.items():
        if key not in editable:
            violations.append(f"{key} is not editable in stage {stage}")
            continue
        if key in base_parameters:
            original = base_parameters[key]
            if isinstance(original, bool) and not isinstance(value, bool):
                violations.append(f"{key} must be bool")
            elif isinstance(original, int) and not isinstance(original, bool) and not isinstance(value, int):
                violations.append(f"{key} must be int")
            elif isinstance(original, float) and not isinstance(value, (int, float)):
                violations.append(f"{key} must be numeric")

    if locked_parameters is not None:
        editable_or_ignored = editable | IGNORED_PARAMETERS
        all_keys = set(parameters) | set(locked_parameters)
        for key in sorted(all_keys - editable_or_ignored):
            configured = key in parameters
            locked = key in locked_parameters
            if configured != locked or parameters.get(key) != locked_parameters.get(key):
                violations.append(
                    f"{key} is fixed in stage {stage} and must match the current stage baseline"
                )

    for key, bounds in RANGES.items():
        if key not in parameters:
            continue
        value = parameters[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            violations.append(f"{key} must be numeric")
        elif not bounds[0] <= value <= bounds[1]:
            violations.append(f"{key} must be in [{bounds[0]}, {bounds[1]}]")

    required = [
        "data.train_batch_size",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    ]
    for dynamic_key, micro_key, max_tokens_key in BATCHING_AUTHORITIES:
        required.append(
            max_tokens_key if bool(parameters.get(dynamic_key, False)) else micro_key
        )
    missing = [
        key for key in required if key not in parameters or parameters.get(key) is None
    ]
    if missing:
        violations.append("missing required parameters: " + ", ".join(missing))
        return ValidationResult(False, violations)

    train_batch = int(parameters["data.train_batch_size"])
    rollout_n = int(parameters["actor_rollout_ref.rollout.n"])
    mini_batch = int(parameters["actor_rollout_ref.actor.ppo_mini_batch_size"])
    if (train_batch * rollout_n) % mini_batch != 0:
        violations.append("data.train_batch_size * rollout.n must be divisible by ppo_mini_batch_size")
    use_dynamic_bsz = bool(
        parameters.get("actor_rollout_ref.actor.use_dynamic_bsz", False)
    )
    if not use_dynamic_bsz:
        micro_value = parameters.get(
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
        )
        if (
            isinstance(micro_value, int)
            and not isinstance(micro_value, bool)
            and micro_value > 0
            and mini_batch % micro_value != 0
        ):
            violations.append("ppo_mini_batch_size must be divisible by ppo_micro_batch_size_per_gpu")

    num_gpus = int(parameters["trainer.n_gpus_per_node"]) * int(parameters["trainer.nnodes"])
    parallel_groups = [
        ("actor", "actor_rollout_ref.actor.megatron.tensor_model_parallel_size", "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"),
        ("rollout", "actor_rollout_ref.rollout.tensor_model_parallel_size", None),
    ]
    for name, tp_key, pp_key in parallel_groups:
        tp = int(parameters.get(tp_key, 1))
        pp = int(parameters.get(pp_key, 1)) if pp_key else 1
        if tp < 1 or pp < 1 or num_gpus % (tp * pp) != 0:
            violations.append(f"num_gpus must be divisible by {name} TP*PP")

    if stage in {"hardware_repair", "hardware_tuning"} and agent_config.get("preserve_hardware_token_budget", True):
        baseline = hardware_token_budget(base_parameters)
        candidate = hardware_token_budget(parameters)
        tolerance = float(agent_config.get("token_budget_tolerance", 0.0))
        if baseline and abs(candidate - baseline) / baseline > tolerance:
            violations.append(f"hardware token budget must remain {baseline}, got {candidate}")

    canonical = json.dumps(
        effective_parameters(parameters), sort_keys=True, separators=(",", ":")
    )
    for trial in history:
        previous = trial.get("parameters")
        if previous and json.dumps(
            effective_parameters(previous), sort_keys=True, separators=(",", ":")
        ) == canonical:
            violations.append(f"configuration duplicates trial {trial.get('trial_id', '?')}")
            break

    return ValidationResult(not violations, violations)
