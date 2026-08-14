from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")

MIB = 1024 * 1024
GIB_IN_MIB = 1024.0

MODEL_KEY = "actor_rollout_ref.model.path"
PROMPT_LENGTH_KEY = "data.max_prompt_length"
RESPONSE_LENGTH_KEY = "data.max_response_length"
TRAIN_BATCH_KEY = "data.train_batch_size"
ROLLOUT_N_KEY = "actor_rollout_ref.rollout.n"
ROLLOUT_UTILIZATION_KEY = "actor_rollout_ref.rollout.gpu_memory_utilization"

ACTOR_MICRO_KEY = "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
REF_MICRO_KEY = "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
TRAINING_MICRO_KEY = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
PPO_MINI_BATCH_KEY = "actor_rollout_ref.actor.ppo_mini_batch_size"
ACTOR_LOG_PROB_DYNAMIC_KEY = "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"
REF_LOG_PROB_DYNAMIC_KEY = "actor_rollout_ref.ref.log_prob_use_dynamic_bsz"
TRAINING_DYNAMIC_KEY = "actor_rollout_ref.actor.use_dynamic_bsz"
ACTOR_LOG_PROB_MAX_TOKENS_KEY = (
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"
)
REF_LOG_PROB_MAX_TOKENS_KEY = "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu"
TRAINING_MAX_TOKENS_KEY = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"

ACTOR_TP_KEY = "actor_rollout_ref.actor.megatron.tensor_model_parallel_size"
ACTOR_PP_KEY = "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"
ACTOR_CP_KEY = "actor_rollout_ref.actor.megatron.context_parallel_size"
ACTOR_EP_KEY = "actor_rollout_ref.actor.megatron.expert_model_parallel_size"
ACTOR_ETP_KEY = "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size"
ACTOR_VPP_KEY = "actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size"
REF_TP_KEY = "actor_rollout_ref.ref.megatron.tensor_model_parallel_size"
REF_PP_KEY = "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"
REF_CP_KEY = "actor_rollout_ref.ref.megatron.context_parallel_size"
REF_EP_KEY = "actor_rollout_ref.ref.megatron.expert_model_parallel_size"
REF_ETP_KEY = "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size"
REF_VPP_KEY = "actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size"

ACTOR_SP_KEY = "actor_rollout_ref.actor.megatron.sequence_parallel"
REF_SP_KEY = "actor_rollout_ref.ref.megatron.sequence_parallel"
ACTOR_REMOVE_PADDING_KEY = "actor_rollout_ref.actor.megatron.use_remove_padding"
REF_REMOVE_PADDING_KEY = "actor_rollout_ref.ref.megatron.use_remove_padding"
ACTOR_PARAM_OFFLOAD_KEY = "actor_rollout_ref.actor.megatron.param_offload"
ACTOR_GRAD_OFFLOAD_KEY = "actor_rollout_ref.actor.megatron.grad_offload"
ACTOR_OPTIMIZER_OFFLOAD_KEY = "actor_rollout_ref.actor.megatron.optimizer_offload"
REF_PARAM_OFFLOAD_KEY = "actor_rollout_ref.ref.megatron.param_offload"
MODEL_FUSED_KERNELS_KEY = "actor_rollout_ref.model.use_fused_kernels"
ACTOR_FUSED_KERNELS_KEY = "actor_rollout_ref.actor.use_fused_kernels"
REF_FUSED_KERNELS_KEY = "actor_rollout_ref.ref.use_fused_kernels"
LORA_RANK_KEY = "actor_rollout_ref.model.lora_rank"
LORA_ADAPTER_PATH_KEY = "actor_rollout_ref.model.lora_adapter_path"
LORA_TARGET_MODULES_KEY = "actor_rollout_ref.model.target_modules"
ENTROPY_COEFF_KEY = "actor_rollout_ref.actor.entropy_coeff"
PARAM_DTYPE_KEY = "actor_rollout_ref.actor.megatron.dtype"
USE_DISTRIBUTED_OPTIMIZER_KEY = (
    "actor_rollout_ref.actor.megatron.use_distributed_optimizer"
)
RECOMPUTE_GRANULARITY_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity"
)
RECOMPUTE_METHOD_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method"
)
RECOMPUTE_NUM_LAYERS_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers"
)
RECOMPUTE_MODULES_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules"
)
ATTENTION_BACKEND_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend"
)
FIRST_PIPELINE_LAYERS_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config."
    "num_layers_in_first_pipeline_stage"
)
LAST_PIPELINE_LAYERS_KEY = (
    "actor_rollout_ref.actor.megatron.override_transformer_config."
    "num_layers_in_last_pipeline_stage"
)

WORLD_SIZE_KEYS = ("trainer.n_gpus_per_node", "trainer.nnodes")
NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"([^\s:]+):({NUMBER})")
STEP_RE = re.compile(r"step:(\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PARAMETER_COUNT_RE = re.compile(
    r"number of parameters on \(tensor, pipeline\) model parallel rank "
    r"\((?P<tp_rank>\d+),\s*(?P<pp_rank>\d+)\):\s*(?P<count>\d+)"
)

MODEL_CONFIG_FIELDS = (
    "model_type",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "tie_word_embeddings",
    "torch_dtype",
    "vocab_size",
    "padded_vocab_size",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "expert_intermediate_size",
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "n_shared_experts",
    "moe_layer_freq",
    "decoder_sparse_step",
    "first_k_dense_replace",
    "mtp_num_layers",
    "num_nextn_predict_layers",
    "multi_latent_attention",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_pos_emb_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)

RESOLVED_CONFIG_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "sequence_parallel",
    "use_remove_padding",
    "use_distributed_optimizer",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "recompute_modules",
    "attention_backend",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "fp16",
    "bf16",
    "params_dtype",
    "num_layers",
    "mtp_num_layers",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "ffn_hidden_size",
    "kv_channels",
    "gated_linear_unit",
    "num_moe_experts",
    "multi_latent_attention",
    "moe_shared_expert_intermediate_size",
    "moe_layer_freq",
    "moe_ffn_hidden_size",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(parameters: Mapping[str, Any], key: str, default: float) -> float:
    value = parameters.get(key)
    if value is None:
        return default
    if _is_number(value):
        return float(value)
    raise ValueError(f"{key} must be numeric or null, got {type(value).__name__}")


def _changed_keys(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> set[str]:
    keys = set(reference) | set(candidate)
    return {key for key in keys if reference.get(key) != candidate.get(key)}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().rstrip(",")
    lowered = value.lower()
    if lowered == "none" or lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(NUMBER, value):
        return float(value)
    return value


def _resolve_recorded_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()

    # Trial histories are often copied from a remote cluster.  Preserve the
    # path below the repository's output/ directory so replay works locally.
    normalized = value.replace("\\", "/")
    marker = "/output/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        local = Path.cwd() / "output" / Path(suffix)
        if local.is_file():
            return local.resolve()
    return None


def _trial_log_path(trial: Mapping[str, Any]) -> Path | None:
    return _resolve_recorded_path(trial.get("log_path"))


def _gpu_samples_path(trial: Mapping[str, Any], log_path: Path | None) -> Path | None:
    recorded = _resolve_recorded_path(trial.get("gpu_samples_path"))
    if recorded is not None:
        return recorded
    if log_path is not None:
        sibling = log_path.parent / "gpu_samples.csv"
        if sibling.is_file():
            return sibling.resolve()
    return None


def _read_model_config(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    model_path = parameters.get(MODEL_KEY)
    if not isinstance(model_path, str) or not model_path:
        return {}, None
    path = Path(model_path).expanduser()
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return {}, None
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if not isinstance(value, Mapping):
        return {}, None
    return dict(value), str(config_path)


def _logged_parameter_profile(
    clean_text: str,
    resolved: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    """Recover the measured model shards printed by Megatron during startup.

    Ray may collapse identical messages from data-parallel replicas.  The rank
    tuple in this message is the tensor/pipeline rank, so one value per unique
    tuple is sufficient to reconstruct a TP/PP partition.  Actor and reference
    use the same model and, in the colocated Megatron worker, the same model-
    parallel topology; repeated initializations therefore do not need to be
    counted twice.
    """

    rank_parameters: dict[tuple[int, int], int] = {}
    matched_lines = 0
    for match in PARAMETER_COUNT_RE.finditer(clean_text):
        matched_lines += 1
        rank = (int(match.group("tp_rank")), int(match.group("pp_rank")))
        count = int(match.group("count"))
        # A repeated actor/ref or DP message should be identical.  Keeping the
        # maximum is conservative if a backend prints slightly different
        # wrapper counts for the two model instances.
        rank_parameters[rank] = max(rank_parameters.get(rank, 0), count)

    if not rank_parameters:
        return {}

    tp = int(resolved.get("tensor_model_parallel_size", 1) or 1)
    pp = int(resolved.get("pipeline_model_parallel_size", 1) or 1)
    ep = int(resolved.get("expert_model_parallel_size", 1) or 1)
    etp = int(resolved.get("expert_tensor_parallel_size", tp) or tp)
    expected_rank_count = max(1, tp * pp)
    in_topology = {
        rank: count
        for rank, count in rank_parameters.items()
        if rank[0] < tp and rank[1] < pp
    }
    if not in_topology:
        in_topology = rank_parameters

    complete_tp_pp_coverage = len(in_topology) == expected_rank_count
    most_loaded = max(in_topology.values())

    # With EP=1, summing every unique TP/PP shard reconstructs the fixed global
    # parameter count directly.  An EP rank is not present in the printed rank
    # tuple, so for MoE/EP logs that total is ambiguous; retain the observed
    # shard and let _parameter_footprint use an anchored analytical ratio.
    total_parameters: int | None = None
    total_source: str | None = None
    if complete_tp_pp_coverage and ep == 1:
        total_parameters = sum(in_topology.values())
        total_source = "sum_unique_logged_tp_pp_shards"
    elif ep == 1:
        total_parameters = most_loaded * expected_rank_count
        total_source = "max_logged_shard_scaled_by_reference_tp_pp"

    return {
        "source": f"{log_path}:number_of_parameters",
        "matched_log_lines": matched_lines,
        "rank_parameters": [
            {
                "tensor_rank": tp_rank,
                "pipeline_rank": pp_rank,
                "parameters": count,
            }
            for (tp_rank, pp_rank), count in sorted(in_topology.items())
        ],
        "observed_rank_count": len(in_topology),
        "expected_rank_count": expected_rank_count,
        "complete_tp_pp_coverage": complete_tp_pp_coverage,
        "most_loaded_shard_parameters": most_loaded,
        "total_parameters": total_parameters,
        "total_parameters_source": total_source,
        "reference_topology": {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
    }


def _extract_log_context(
    trial: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    log_path = _trial_log_path(trial)
    model_config, model_config_path = _read_model_config(parameters)
    if log_path is None:
        return {
            "log_path": None,
            "model_config": model_config,
            "model_config_source": model_config_path,
            "resolved": {},
            "parameter_profile": {},
            "length": _configured_length_profile(parameters),
        }

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    clean_text = ANSI_RE.sub("", text)

    # When the model path is unavailable during offline replay, recover the
    # simple scalar fields from the config JSON printed by transformers.
    log_model_config: dict[str, Any] = {}
    for field in MODEL_CONFIG_FIELDS:
        pattern = re.compile(
            rf'["\']{re.escape(field)}["\']\s*:\s*'
            rf'(?P<value>true|false|null|None|"[^"]*"|\'[^\']*\'|{NUMBER})',
            re.IGNORECASE,
        )
        match = pattern.search(clean_text)
        if match:
            log_model_config[field] = _parse_scalar(match.group("value"))
    if not model_config:
        model_config = log_model_config
        model_config_path = (
            f"{log_path}:printed_model_config" if log_model_config else None
        )

    resolved: dict[str, Any] = {}
    transformer_line = next(
        (line for line in clean_text.splitlines() if "TransformerConfig(" in line),
        "",
    )
    for field in RESOLVED_CONFIG_FIELDS:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(field)}=(?P<value>[^,\)]+)",
            transformer_line,
        )
        if match:
            resolved[field] = _parse_scalar(match.group("value"))

    parameter_profile = _logged_parameter_profile(clean_text, resolved, log_path)

    records: list[tuple[int, dict[str, float]]] = []
    for line in clean_text.splitlines():
        step_match = STEP_RE.search(line)
        if not step_match:
            continue
        pairs = {key: float(value) for key, value in PAIR_RE.findall(line)}
        if pairs:
            records.append((int(step_match.group(1)), pairs))

    length = _length_profile_from_records(records, parameters)
    return {
        "log_path": str(log_path),
        "model_config": model_config,
        "model_config_source": model_config_path,
        "resolved": resolved,
        "parameter_profile": parameter_profile,
        "length": length,
    }


def _configured_length_profile(parameters: Mapping[str, Any]) -> dict[str, Any]:
    prompt = _number(parameters, PROMPT_LENGTH_KEY, 1024.0)
    response = _number(parameters, RESPONSE_LENGTH_KEY, 4096.0)
    total = max(1.0, prompt + response)
    return {
        "point_tokens": total,
        "upper_tokens": total,
        "configured_upper_tokens": total,
        "source": "configured_maximum",
        "sampled_steps": 0,
    }


def _length_profile_from_records(
    records: Sequence[tuple[int, Mapping[str, float]]],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    configured = _configured_length_profile(parameters)
    if not records:
        return configured
    stable = [row for step, row in records if step > 5]
    if not stable:
        stable = [row for _, row in records]

    population = _number(parameters, TRAIN_BATCH_KEY, 0.0) * _number(
        parameters, ROLLOUT_N_KEY, 1.0
    )
    means: list[float] = []
    observed_upper: list[float] = []
    for row in stable:
        total_tokens = row.get("perf/total_num_tokens")
        if total_tokens is not None and population > 0:
            means.append(total_tokens / population)
        elif (
            row.get("prompt_length/mean") is not None
            and row.get("response_length/mean") is not None
        ):
            means.append(row["prompt_length/mean"] + row["response_length/mean"])
        if (
            row.get("prompt_length/max") is not None
            and row.get("response_length/max") is not None
        ):
            observed_upper.append(row["prompt_length/max"] + row["response_length/max"])

    point = _percentile(means, 0.95)
    if point is None:
        return configured
    configured_upper = float(configured["configured_upper_tokens"])
    upper_observed = _percentile(observed_upper, 0.95)
    upper = configured_upper
    if upper_observed is not None:
        upper = min(configured_upper, max(point, upper_observed))
    return {
        "point_tokens": max(1.0, min(point, configured_upper)),
        "upper_tokens": max(1.0, upper),
        "configured_upper_tokens": configured_upper,
        "source": "reference_train_log_stable_step_p95_mean_tokens",
        "sampled_steps": len(means),
        "mean_tokens_across_sampled_steps": (statistics.mean(means) if means else None),
    }


def _candidate_length_profile(
    reference_profile: Mapping[str, Any],
    reference_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    ref_bound = _configured_length_profile(reference_parameters)[
        "configured_upper_tokens"
    ]
    candidate_bound = _configured_length_profile(candidate_parameters)[
        "configured_upper_tokens"
    ]
    changed = any(
        reference_parameters.get(key) != candidate_parameters.get(key)
        for key in (PROMPT_LENGTH_KEY, RESPONSE_LENGTH_KEY)
    )
    if not changed:
        return dict(reference_profile)
    ratio = candidate_bound / max(1.0, float(ref_bound))
    return {
        "point_tokens": min(
            float(candidate_bound),
            max(1.0, float(reference_profile["point_tokens"]) * ratio),
        ),
        "upper_tokens": float(candidate_bound),
        "configured_upper_tokens": float(candidate_bound),
        "source": "reference_effective_length_scaled_by_configured_bound",
        "sampled_steps": reference_profile.get("sampled_steps", 0),
    }


def _phase_percentage_peaks(trial: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    memory = trial.get("memory_by_phase_pct")
    if not isinstance(memory, Mapping):
        return result
    for phase in PHASES:
        value = memory.get(phase)
        if isinstance(value, Mapping):
            value = value.get("max")
        if _is_number(value):
            result[phase] = float(value)
    return result


def _phase_measurements(
    trial: Mapping[str, Any], log_path: Path | None
) -> dict[str, dict[str, Any]]:
    """Read phase peaks from trial.jsonl and capacity from gpu_samples.csv.

    C550's phase boundary messages in train.log are not a reliable physical
    memory source.  ``memory_by_phase_pct`` is the authoritative phase summary;
    the CSV is used for GPU capacity and as a fallback only when that summary
    is absent.
    """

    pct_peaks = _phase_percentage_peaks(trial)
    samples_path = _gpu_samples_path(trial, log_path)
    used_by_phase: dict[str, list[float]] = {phase: [] for phase in PHASES}
    totals: list[float] = []

    if samples_path is not None:
        try:
            with samples_path.open("r", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    phase = row.get("phase")
                    if phase not in used_by_phase:
                        continue
                    try:
                        used = float(row["memory_used_mb"])
                        total = float(row["memory_total_mb"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if used >= 0 and total > 0:
                        used_by_phase[phase].append(used)
                        totals.append(total)
        except OSError:
            pass

    capacity_mb = statistics.median(totals) if totals else None
    result: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        pct = pct_peaks.get(phase)
        raw_csv_peak_mb = max(used_by_phase[phase]) if used_by_phase[phase] else None
        if pct is not None and capacity_mb is not None:
            used_mb = capacity_mb * pct / 100.0
            phase_source = "trial.memory_by_phase_pct.max * gpu_samples.capacity"
        else:
            used_mb = raw_csv_peak_mb
            phase_source = (
                f"{samples_path}:raw_phase_peak_fallback"
                if raw_csv_peak_mb is not None and samples_path is not None
                else (
                    "trial.memory_by_phase_pct.max_without_capacity"
                    if pct is not None
                    else None
                )
            )
        if pct is None and used_mb is not None and capacity_mb is not None:
            pct = 100.0 * used_mb / capacity_mb
        result[phase] = {
            "memory_mb": used_mb,
            "memory_pct": pct,
            "gpu_capacity_mb": capacity_mb,
            "source": phase_source,
        }
    return result


def _same_parameters(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _reference_trial(
    current: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    reference_trial_id: int | None,
) -> Mapping[str, Any] | None:
    observed = [
        trial
        for trial in trials
        if isinstance(trial, Mapping)
        and isinstance(trial.get("parameters"), Mapping)
        and _phase_percentage_peaks(trial)
    ]
    if reference_trial_id is not None:
        return next(
            (
                trial
                for trial in observed
                if trial.get("trial_id") == reference_trial_id
            ),
            None,
        )
    # Do not silently fall back to an arbitrary same-model trial.  The only
    # safe implicit anchor is an exact parameter match.
    exact = [
        trial
        for trial in observed
        if _same_parameters(trial.get("parameters", {}), current)
    ]
    return exact[-1] if exact else None


def _pick(
    parameters: Mapping[str, Any],
    model_config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    parameter_names: Sequence[str],
    config_names: Sequence[str],
    resolved_names: Sequence[str],
    default: Any = None,
) -> tuple[Any, str]:
    for name in parameter_names:
        if name in parameters and parameters[name] is not None:
            return parameters[name], f"parameters:{name}"
    for name in resolved_names:
        if name in resolved and resolved[name] is not None:
            return resolved[name], f"reference_train_log:{name}"
    for name in config_names:
        if name in model_config and model_config[name] is not None:
            return model_config[name], f"model_config:{name}"
    return default, "explicit_default"


def _required_positive_number(value: Any, name: str) -> float:
    if not _is_number(value) or float(value) <= 0:
        raise ValueError(
            f"cannot estimate compute memory: missing positive model field {name!r} "
            "from parameters, reference train.log, and model config.json"
        )
    return float(value)


def _known_gated_model(model_type: str) -> bool:
    lowered = model_type.lower()
    return any(
        family in lowered
        for family in (
            "qwen",
            "llama",
            "mistral",
            "mixtral",
            "deepseek",
            "gemma",
            "phi3",
        )
    )


def _model_architecture(
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    config = context.get("model_config", {})
    resolved = context.get("resolved", {})
    if not isinstance(config, Mapping):
        config = {}
    if not isinstance(resolved, Mapping):
        resolved = {}
    sources: dict[str, str] = {}

    def pick(
        name: str,
        parameter_names: Sequence[str],
        config_names: Sequence[str],
        resolved_names: Sequence[str],
        default: Any = None,
    ) -> Any:
        value, source = _pick(
            parameters,
            config,
            resolved,
            parameter_names,
            config_names,
            resolved_names,
            default,
        )
        sources[name] = source
        return value

    hidden = _required_positive_number(
        pick(
            "hidden_size",
            ("hidden_size", "model.hidden_size", "actor_rollout_ref.model.hidden_size"),
            ("hidden_size",),
            ("hidden_size",),
        ),
        "hidden_size",
    )
    layers = int(
        _required_positive_number(
            pick(
                "num_layers",
                ("num_layers", "num_hidden_layers"),
                ("num_hidden_layers", "num_layers"),
                ("num_layers",),
            ),
            "num_layers",
        )
    )
    heads = int(
        _required_positive_number(
            pick(
                "num_attention_heads",
                ("num_attention_heads",),
                ("num_attention_heads",),
                ("num_attention_heads",),
            ),
            "num_attention_heads",
        )
    )
    ffn = _required_positive_number(
        pick(
            "ffn_hidden_size",
            ("ffn_hidden_size",),
            ("intermediate_size", "ffn_hidden_size"),
            ("ffn_hidden_size",),
        ),
        "ffn_hidden_size",
    )
    vocab = int(
        _required_positive_number(
            pick(
                "padded_vocab_size",
                ("padded_vocab_size",),
                ("padded_vocab_size", "vocab_size"),
                ("padded_vocab_size",),
            ),
            "padded_vocab_size/vocab_size",
        )
    )
    kv_channels = pick(
        "kv_channels",
        ("kv_channels",),
        ("head_dim", "kv_channels"),
        ("kv_channels",),
        None,
    )
    if not _is_number(kv_channels) or float(kv_channels) <= 0:
        if hidden % heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        kv_channels = hidden / heads
        sources["kv_channels"] = "derived:hidden_size/num_attention_heads"

    query_groups = pick(
        "num_query_groups",
        ("num_query_groups",),
        ("num_key_value_heads", "num_query_groups"),
        ("num_query_groups",),
        heads,
    )
    query_groups = int(_required_positive_number(query_groups, "num_query_groups"))
    group_query = pick(
        "group_query_attention",
        ("group_query_attention",),
        ("group_query_attention",),
        (),
        query_groups != heads,
    )
    model_type = str(config.get("model_type", ""))
    gated = pick(
        "swiglu",
        ("swiglu",),
        ("swiglu",),
        ("gated_linear_unit",),
        _known_gated_model(model_type)
        and str(config.get("hidden_act", "")).lower() in {"silu", "swish"},
    )
    tied = pick(
        "tie_word_embeddings",
        ("tie_word_embeddings",),
        ("tie_word_embeddings",),
        (),
        True,
    )

    num_experts = pick(
        "num_experts",
        ("num_experts", "num_moe_experts"),
        ("num_experts", "num_local_experts", "n_routed_experts"),
        ("num_moe_experts",),
        None,
    )
    num_experts = (
        int(num_experts) if _is_number(num_experts) and num_experts > 0 else None
    )
    moe_ffn = pick(
        "moe_ffn_hidden_size",
        ("moe_ffn_hidden_size",),
        ("moe_intermediate_size", "expert_intermediate_size"),
        ("moe_ffn_hidden_size",),
        ffn if num_experts is not None else None,
    )
    shared_ffn = pick(
        "moe_shared_expert_intermediate_size",
        ("moe_shared_expert_intermediate_size",),
        ("moe_shared_expert_intermediate_size", "shared_expert_intermediate_size"),
        ("moe_shared_expert_intermediate_size",),
        None,
    )
    if shared_ffn is None and _is_number(config.get("n_shared_experts")):
        base = moe_ffn if _is_number(moe_ffn) else ffn
        shared_ffn = float(config["n_shared_experts"]) * float(base)
        sources["moe_shared_expert_intermediate_size"] = (
            "derived:n_shared_experts*moe_ffn_hidden_size"
        )

    moe_frequency = pick(
        "moe_layer_freq",
        ("moe_layer_freq",),
        ("moe_layer_freq", "decoder_sparse_step"),
        ("moe_layer_freq",),
        1,
    )
    first_dense = config.get("first_k_dense_replace", 0)
    if num_experts is None:
        moe_pattern = [0] * layers
    elif isinstance(moe_frequency, list):
        if len(moe_frequency) != layers:
            raise ValueError("moe_layer_freq list length must equal num_layers")
        moe_pattern = [1 if bool(value) else 0 for value in moe_frequency]
    else:
        frequency = int(moe_frequency)
        if frequency <= 0:
            raise ValueError("moe_layer_freq must be positive")
        dense_prefix = int(first_dense) if _is_number(first_dense) else 0
        moe_pattern = [
            1
            if index >= dense_prefix and (index - dense_prefix) % frequency == 0
            else 0
            for index in range(layers)
        ]

    multi_latent = bool(
        pick(
            "multi_latent_attention",
            ("multi_latent_attention",),
            ("multi_latent_attention",),
            ("multi_latent_attention",),
            "kv_lora_rank" in config,
        )
    )
    q_lora_rank = pick(
        "q_lora_rank", ("q_lora_rank",), ("q_lora_rank",), ("q_lora_rank",), None
    )
    kv_lora_rank = pick(
        "kv_lora_rank", ("kv_lora_rank",), ("kv_lora_rank",), ("kv_lora_rank",), None
    )
    qk_head_dim = pick(
        "qk_head_dim",
        ("qk_head_dim",),
        ("qk_head_dim", "qk_nope_head_dim"),
        ("qk_head_dim",),
        kv_channels,
    )
    qk_pos_dim = pick(
        "qk_pos_emb_head_dim",
        ("qk_pos_emb_head_dim",),
        ("qk_pos_emb_head_dim", "qk_rope_head_dim"),
        ("qk_pos_emb_head_dim",),
        0,
    )
    v_head_dim = pick(
        "v_head_dim",
        ("v_head_dim",),
        ("v_head_dim",),
        ("v_head_dim",),
        kv_channels,
    )
    if multi_latent and not _is_number(kv_lora_rank):
        raise ValueError("MLA model requires kv_lora_rank")

    mtp_layers = pick(
        "mtp_num_layers",
        ("mtp_num_layers",),
        ("mtp_num_layers", "num_nextn_predict_layers"),
        ("mtp_num_layers",),
        None,
    )
    mtp_layers = int(mtp_layers) if _is_number(mtp_layers) and mtp_layers > 0 else 0
    megatron_dtype = parameters.get(PARAM_DTYPE_KEY)
    if megatron_dtype is not None:
        dtype = str(megatron_dtype)
        dtype_source = f"parameters:{PARAM_DTYPE_KEY}"
    elif resolved.get("params_dtype") is not None:
        dtype = str(resolved["params_dtype"])
        dtype_source = "reference_train_log:params_dtype"
    else:
        dtype = str(config.get("torch_dtype", "bfloat16"))
        dtype_source = "model_config:torch_dtype"
    lowered_dtype = dtype.lower()
    bytes_per_weight = 4 if "float32" in lowered_dtype else 2
    activation_dtype_bytes = 4 if "float32" in lowered_dtype else 2
    sources["params_dtype"] = dtype_source

    return {
        "model_type": model_type or None,
        "hidden_size": hidden,
        "num_layers": layers,
        "num_attention_heads": heads,
        "ffn_hidden_size": ffn,
        "padded_vocab_size": vocab,
        "kv_channels": float(kv_channels),
        "num_query_groups": query_groups,
        "group_query_attention": bool(group_query),
        "swiglu": bool(gated),
        "untie_embeddings_and_output_weights": not bool(tied),
        "num_experts": num_experts,
        "moe_ffn_hidden_size": float(moe_ffn) if _is_number(moe_ffn) else None,
        "moe_shared_expert_intermediate_size": (
            float(shared_ffn) if _is_number(shared_ffn) else 0.0
        ),
        "moe_layer_pattern": moe_pattern,
        "multi_latent_attention": multi_latent,
        "q_lora_rank": float(q_lora_rank) if _is_number(q_lora_rank) else None,
        "kv_lora_rank": float(kv_lora_rank) if _is_number(kv_lora_rank) else None,
        "qk_head_dim": float(qk_head_dim),
        "qk_pos_emb_head_dim": float(qk_pos_dim),
        "v_head_dim": float(v_head_dim),
        "mtp_num_layers": mtp_layers,
        "params_dtype": dtype,
        "bytes_per_weight": bytes_per_weight,
        "activation_dtype_bytes": activation_dtype_bytes,
        "parameter_profile": (
            dict(context["parameter_profile"])
            if isinstance(context.get("parameter_profile"), Mapping)
            else {}
        ),
        "sources": sources,
        "model_config_source": context.get("model_config_source"),
    }


def _phase_keys(phase: str) -> dict[str, str]:
    if phase == "ref_log_prob":
        return {
            "micro": REF_MICRO_KEY,
            "tp": ACTOR_TP_KEY,
            "pp": ACTOR_PP_KEY,
            "cp": ACTOR_CP_KEY,
            "ep": ACTOR_EP_KEY,
            "etp": ACTOR_ETP_KEY,
            "vpp": ACTOR_VPP_KEY,
            "sp": ACTOR_SP_KEY,
            "param_offload": REF_PARAM_OFFLOAD_KEY,
            "dynamic_batch": REF_LOG_PROB_DYNAMIC_KEY,
            "dynamic_fallback": TRAINING_DYNAMIC_KEY,
            "max_tokens": REF_LOG_PROB_MAX_TOKENS_KEY,
            "max_tokens_fallback": TRAINING_MAX_TOKENS_KEY,
            "remove_padding": REF_REMOVE_PADDING_KEY,
            "remove_padding_fallback": ACTOR_REMOVE_PADDING_KEY,
        }
    if phase == "actor_log_prob":
        return {
            "micro": ACTOR_MICRO_KEY,
            "tp": ACTOR_TP_KEY,
            "pp": ACTOR_PP_KEY,
            "cp": ACTOR_CP_KEY,
            "ep": ACTOR_EP_KEY,
            "etp": ACTOR_ETP_KEY,
            "vpp": ACTOR_VPP_KEY,
            "sp": ACTOR_SP_KEY,
            "param_offload": ACTOR_PARAM_OFFLOAD_KEY,
            "dynamic_batch": ACTOR_LOG_PROB_DYNAMIC_KEY,
            "dynamic_fallback": TRAINING_DYNAMIC_KEY,
            "max_tokens": ACTOR_LOG_PROB_MAX_TOKENS_KEY,
            "max_tokens_fallback": TRAINING_MAX_TOKENS_KEY,
            "remove_padding": ACTOR_REMOVE_PADDING_KEY,
            "remove_padding_fallback": ACTOR_REMOVE_PADDING_KEY,
        }
    return {
        "micro": TRAINING_MICRO_KEY,
        "tp": ACTOR_TP_KEY,
        "pp": ACTOR_PP_KEY,
        "cp": ACTOR_CP_KEY,
        "ep": ACTOR_EP_KEY,
        "etp": ACTOR_ETP_KEY,
        "vpp": ACTOR_VPP_KEY,
        "sp": ACTOR_SP_KEY,
        "param_offload": ACTOR_PARAM_OFFLOAD_KEY,
        "dynamic_batch": TRAINING_DYNAMIC_KEY,
        "dynamic_fallback": TRAINING_DYNAMIC_KEY,
        "max_tokens": TRAINING_MAX_TOKENS_KEY,
        "max_tokens_fallback": TRAINING_MAX_TOKENS_KEY,
        "remove_padding": ACTOR_REMOVE_PADDING_KEY,
        "remove_padding_fallback": ACTOR_REMOVE_PADDING_KEY,
    }


def _runtime_value(
    parameters: Mapping[str, Any],
    key: str,
    resolved: Mapping[str, Any],
    resolved_name: str,
    default: Any,
) -> tuple[Any, str]:
    if key in parameters and parameters[key] is not None:
        return parameters[key], f"parameters:{key}"
    if resolved_name in resolved and resolved[resolved_name] is not None:
        return resolved[resolved_name], f"reference_train_log:{resolved_name}"
    return default, "framework_default"


def _runtime_args(
    phase: str,
    parameters: Mapping[str, Any],
    context: Mapping[str, Any],
    length_profile: Mapping[str, Any],
) -> dict[str, Any]:
    keys = _phase_keys(phase)
    resolved = context.get("resolved", {})
    if not isinstance(resolved, Mapping):
        resolved = {}
    sources: dict[str, str] = {}

    def read(name: str, key: str, resolved_name: str, default: Any) -> Any:
        value, source = _runtime_value(
            parameters, key, resolved, resolved_name, default
        )
        sources[name] = source
        return value

    def read_with_parameter_fallback(
        name: str,
        key: str,
        fallback_key: str,
        resolved_name: str,
        default: Any,
    ) -> Any:
        if key in parameters and parameters[key] is not None:
            sources[name] = f"parameters:{key}"
            return parameters[key]
        if fallback_key in parameters and parameters[fallback_key] is not None:
            sources[name] = f"parameters:{fallback_key}:config_interpolation"
            return parameters[fallback_key]
        value = resolved.get(resolved_name)
        if value is not None:
            sources[name] = f"reference_train_log:{resolved_name}"
            return value
        sources[name] = "framework_default"
        return default

    micro = float(read("micro_batch_size", keys["micro"], "micro_batch_size", 1))
    tp = int(
        read("tensor_model_parallel_size", keys["tp"], "tensor_model_parallel_size", 1)
    )
    pp = int(
        read(
            "pipeline_model_parallel_size",
            keys["pp"],
            "pipeline_model_parallel_size",
            1,
        )
    )
    cp = int(read("context_parallel_size", keys["cp"], "context_parallel_size", 1))
    ep = int(
        read("expert_model_parallel_size", keys["ep"], "expert_model_parallel_size", 1)
    )
    etp_key = keys["etp"]
    if etp_key in parameters and parameters[etp_key] is not None:
        etp = int(parameters[etp_key])
        sources["expert_tensor_parallel_size"] = f"parameters:{etp_key}"
    else:
        resolved_tp = resolved.get("tensor_model_parallel_size")
        resolved_etp = resolved.get("expert_tensor_parallel_size")
        if (
            _is_number(resolved_etp)
            and _is_number(resolved_tp)
            and int(resolved_etp) != int(resolved_tp)
        ):
            # Preserve an explicitly non-default expert-TP relationship from
            # the reference log.  The common ETP==TP case is a framework
            # default and must follow a candidate TP change.
            etp = int(resolved_etp)
            sources["expert_tensor_parallel_size"] = (
                "reference_train_log:expert_tensor_parallel_size"
            )
        else:
            etp = tp
            sources["expert_tensor_parallel_size"] = (
                "framework_default:tensor_model_parallel_size"
            )
    vpp_raw = read(
        "virtual_pipeline_model_parallel_size",
        keys["vpp"],
        "virtual_pipeline_model_parallel_size",
        None,
    )
    vpp = int(vpp_raw) if _is_number(vpp_raw) and vpp_raw > 0 else None
    sp = bool(read("sequence_parallel", keys["sp"], "sequence_parallel", False))
    if tp == 1 and sp:
        sp = False
        sources["sequence_parallel"] = "framework_forced_false_at_tp1"
    param_offload = bool(
        read("param_offload", keys["param_offload"], "param_offload", False)
    )
    dynamic_batch = bool(
        read_with_parameter_fallback(
            "dynamic_batch",
            keys["dynamic_batch"],
            keys["dynamic_fallback"],
            "use_dynamic_bsz",
            False,
        )
    )
    max_tokens = float(
        read_with_parameter_fallback(
            "max_token_len_per_gpu",
            keys["max_tokens"],
            keys["max_tokens_fallback"],
            "max_token_len_per_gpu",
            16384,
        )
    )
    remove_padding = bool(
        read_with_parameter_fallback(
            "remove_padding",
            keys["remove_padding"],
            keys["remove_padding_fallback"],
            "use_remove_padding",
            True,
        )
    )
    if min(micro, max_tokens, tp, pp, cp, ep, etp) <= 0:
        raise ValueError(f"invalid non-positive parallel or batch value for {phase}")

    gpus = int(_number(parameters, WORLD_SIZE_KEYS[0], 1.0))
    nodes = int(_number(parameters, WORLD_SIZE_KEYS[1], 1.0))
    world = gpus * nodes
    model_parallel = tp * pp * cp
    if world % model_parallel != 0:
        raise ValueError(
            f"world size {world} is not divisible by TP*PP*CP={model_parallel}"
        )
    dp = world // model_parallel

    def recompute_value(key: str, resolved_name: str, default: Any = None) -> Any:
        if key in parameters and parameters[key] is not None:
            sources[resolved_name] = f"parameters:{key}"
            return parameters[key]
        if resolved_name in resolved and resolved[resolved_name] is not None:
            sources[resolved_name] = f"reference_train_log:{resolved_name}"
            return resolved[resolved_name]
        sources[resolved_name] = "framework_default"
        return default

    recompute = recompute_value(RECOMPUTE_GRANULARITY_KEY, "recompute_granularity")
    recompute_method = recompute_value(RECOMPUTE_METHOD_KEY, "recompute_method")
    recompute_num_layers = recompute_value(
        RECOMPUTE_NUM_LAYERS_KEY, "recompute_num_layers"
    )
    recompute_modules = recompute_value(
        RECOMPUTE_MODULES_KEY, "recompute_modules", ["core_attn"]
    )
    attention_backend = recompute_value(
        ATTENTION_BACKEND_KEY, "attention_backend", "flash"
    )
    first_stage_layers_raw = parameters.get(
        FIRST_PIPELINE_LAYERS_KEY,
        resolved.get("num_layers_in_first_pipeline_stage"),
    )
    last_stage_layers_raw = parameters.get(
        LAST_PIPELINE_LAYERS_KEY,
        resolved.get("num_layers_in_last_pipeline_stage"),
    )
    use_distributed = bool(
        read(
            "use_distributed_optimizer",
            USE_DISTRIBUTED_OPTIMIZER_KEY,
            "use_distributed_optimizer",
            False,
        )
    )
    optimizer_offload = bool(parameters.get(ACTOR_OPTIMIZER_OFFLOAD_KEY, False))
    actor_param_offload = bool(parameters.get(ACTOR_PARAM_OFFLOAD_KEY, False))
    actor_grad_offload = bool(parameters.get(ACTOR_GRAD_OFFLOAD_KEY, False))
    ref_param_offload = bool(parameters.get(REF_PARAM_OFFLOAD_KEY, False))
    lora_rank = int(_number(parameters, LORA_RANK_KEY, 0.0))
    is_lora = lora_rank > 0 or bool(parameters.get(LORA_ADAPTER_PATH_KEY))
    model_fused = bool(parameters.get(MODEL_FUSED_KERNELS_KEY, False))
    actor_fused = bool(parameters.get(ACTOR_FUSED_KERNELS_KEY, model_fused))
    if int(resolved.get("mtp_num_layers", 0) or 0) > 0:
        actor_fused = False
        sources["actor_fused_kernels"] = "framework_forced_false_with_mtp"
    else:
        sources["actor_fused_kernels"] = (
            f"parameters:{ACTOR_FUSED_KERNELS_KEY}"
            if ACTOR_FUSED_KERNELS_KEY in parameters
            else f"parameters:{MODEL_FUSED_KERNELS_KEY}:config_interpolation"
        )
    if phase == "ref_log_prob" and not is_lora:
        use_fused_kernels = bool(parameters.get(REF_FUSED_KERNELS_KEY, False))
        sources["use_fused_kernels"] = (
            f"parameters:{REF_FUSED_KERNELS_KEY}"
            if REF_FUSED_KERNELS_KEY in parameters
            else "framework_default:false_for_independent_ref"
        )
    else:
        use_fused_kernels = actor_fused
        sources["use_fused_kernels"] = sources["actor_fused_kernels"]

    if phase == "training":
        local_samples = (
            _number(parameters, PPO_MINI_BATCH_KEY, micro * dp)
            * _number(parameters, ROLLOUT_N_KEY, 1.0)
            / dp
        )
    else:
        local_samples = (
            _number(parameters, TRAIN_BATCH_KEY, micro * dp)
            * _number(parameters, ROLLOUT_N_KEY, 1.0)
            / dp
        )
    point_sequence = float(length_profile["point_tokens"])
    upper_sequence = float(length_profile["upper_tokens"])
    if dynamic_batch:
        point_tokens_per_cp_rank = min(max_tokens, local_samples * point_sequence / cp)
        upper_tokens_per_cp_rank = min(max_tokens, local_samples * upper_sequence / cp)
        num_microbatches = max(
            1, math.ceil(local_samples * point_sequence / (max_tokens * cp))
        )
        token_source = "dynamic_token_cap_per_cp_rank"
    else:
        point_tokens_per_cp_rank = micro * point_sequence / cp
        upper_tokens_per_cp_rank = micro * upper_sequence / cp
        num_microbatches = max(1, math.ceil(local_samples / micro))
        token_source = (
            "fixed_microbatch_valid_token_proxy_per_cp_rank"
            if remove_padding
            else "fixed_microbatch_padded_token_proxy_per_cp_rank"
        )

    return {
        "phase": phase,
        "micro_batch_size": micro,
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "context_parallel_size": cp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "virtual_pipeline_model_parallel_size": vpp,
        "sequence_parallel": sp,
        "param_offload": param_offload,
        "optimizer_offload": optimizer_offload,
        "actor_param_offload": actor_param_offload,
        "actor_grad_offload": actor_grad_offload,
        "ref_param_offload": ref_param_offload,
        "dynamic_batch": dynamic_batch,
        "max_token_len_per_gpu": max_tokens,
        "remove_padding": remove_padding,
        "use_fused_kernels": use_fused_kernels,
        "is_lora": is_lora,
        "lora_rank": lora_rank,
        "lora_target_modules": parameters.get(LORA_TARGET_MODULES_KEY, "all-linear"),
        "calculate_entropy": (
            phase == "actor_log_prob"
            or (
                phase == "training" and _number(parameters, ENTROPY_COEFF_KEY, 0.0) != 0
            )
        ),
        "world_size": world,
        "data_parallel_size": dp,
        "use_distributed_optimizer": use_distributed,
        "recompute_granularity": recompute,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "recompute_modules": recompute_modules,
        "attention_backend": str(attention_backend),
        "num_layers_in_first_pipeline_stage": (
            int(first_stage_layers_raw)
            if _is_number(first_stage_layers_raw) and first_stage_layers_raw > 0
            else None
        ),
        "num_layers_in_last_pipeline_stage": (
            int(last_stage_layers_raw)
            if _is_number(last_stage_layers_raw) and last_stage_layers_raw > 0
            else None
        ),
        "num_microbatches": num_microbatches,
        "local_call_samples": local_samples,
        "response_length": _number(parameters, RESPONSE_LENGTH_KEY, 4096.0),
        "seq_length": point_sequence,
        "seq_length_upper": upper_sequence,
        "tokens_per_cp_rank": point_tokens_per_cp_rank,
        "tokens_per_cp_rank_upper": upper_tokens_per_cp_rank,
        "token_source": token_source,
        "requires_packed_shape_calibration": dynamic_batch or remove_padding,
        "sources": sources,
    }


def _attention_parameter_terms(architecture: Mapping[str, Any]) -> float:
    hidden = float(architecture["hidden_size"])
    heads = float(architecture["num_attention_heads"])
    if architecture["multi_latent_attention"]:
        q_lora_rank = architecture["q_lora_rank"]
        qk_head_dim = float(architecture["qk_head_dim"])
        qk_pos_dim = float(architecture["qk_pos_emb_head_dim"])
        kv_lora_rank = float(architecture["kv_lora_rank"])
        v_head_dim = float(architecture["v_head_dim"])
        if q_lora_rank is None:
            q_term = hidden * heads * (qk_head_dim + qk_pos_dim)
        else:
            q_term = float(q_lora_rank) * (
                hidden + heads * (qk_head_dim + qk_pos_dim) + 1
            )
        return (
            q_term
            + kv_lora_rank * (hidden + heads * (qk_head_dim + v_head_dim) + 1)
            + hidden * qk_pos_dim
            + heads * v_head_dim * hidden
        )

    query_projection_size = float(architecture["kv_channels"]) * heads
    ratio = query_projection_size / hidden
    groups = (
        float(architecture["num_query_groups"])
        if architecture["group_query_attention"]
        else heads
    )
    return 2 * hidden * hidden * (1 + groups / heads) * ratio


def _layer_parameter_terms(
    architecture: Mapping[str, Any], is_moe: bool
) -> tuple[float, float]:
    hidden = float(architecture["hidden_size"])
    gated = 1.5 if architecture["swiglu"] else 1.0
    attention = _attention_parameter_terms(architecture)
    if not is_moe:
        common = (
            2 * hidden * (float(architecture["ffn_hidden_size"]) * gated + 2)
            + attention
        )
        return common, 0.0

    shared = float(architecture["moe_shared_expert_intermediate_size"])
    common = 2 * hidden * (shared * gated + 2) + attention
    routed = (
        2
        * hidden
        * (
            float(architecture["moe_ffn_hidden_size"])
            * int(architecture["num_experts"])
            * gated
        )
    )
    return common, routed


def _stage_layer_counts(
    num_layers: int,
    pp: int,
    first_stage_layers: int | None = None,
    last_stage_layers: int | None = None,
) -> list[int]:
    if pp > num_layers:
        raise ValueError(f"PP={pp} cannot exceed num_layers={num_layers}")
    if pp == 1:
        if first_stage_layers not in (None, num_layers) or last_stage_layers not in (
            None,
            num_layers,
        ):
            raise ValueError("first/last pipeline layer overrides require PP > 1")
        return [num_layers]
    first = first_stage_layers
    last = last_stage_layers
    if first is not None and not 0 < first < num_layers:
        raise ValueError("num_layers_in_first_pipeline_stage is invalid")
    if last is not None and not 0 < last < num_layers:
        raise ValueError("num_layers_in_last_pipeline_stage is invalid")
    remaining = num_layers - (first or 0) - (last or 0)
    middle_stages = pp - int(first is not None) - int(last is not None)
    if middle_stages <= 0 or remaining < middle_stages:
        raise ValueError("pipeline stage layer overrides cannot partition num_layers")
    quotient, remainder = divmod(remaining, middle_stages)
    middle = [
        quotient + (1 if stage < remainder else 0) for stage in range(middle_stages)
    ]
    return (
        ([first] if first is not None else [])
        + middle
        + ([last] if last is not None else [])
    )


def _analytical_parameter_footprint(
    architecture: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    pp = int(runtime["pipeline_model_parallel_size"])
    tp = int(runtime["tensor_model_parallel_size"])
    ep = int(runtime["expert_model_parallel_size"])
    etp = int(runtime["expert_tensor_parallel_size"])
    layer_counts = _stage_layer_counts(
        int(architecture["num_layers"]),
        pp,
        runtime.get("num_layers_in_first_pipeline_stage"),
        runtime.get("num_layers_in_last_pipeline_stage"),
    )
    pattern = list(architecture["moe_layer_pattern"])

    stage_common = [0.0] * pp
    stage_routed = [0.0] * pp
    cursor = 0
    for stage, count in enumerate(layer_counts):
        for layer_index in range(cursor, cursor + count):
            common, routed = _layer_parameter_terms(
                architecture, bool(pattern[layer_index])
            )
            stage_common[stage] += common
            stage_routed[stage] += routed
        cursor += count

    hidden = float(architecture["hidden_size"])
    embedding = hidden * int(architecture["padded_vocab_size"])
    stage_common[0] += embedding
    if architecture["untie_embeddings_and_output_weights"]:
        stage_common[-1] += embedding
    stage_common[-1] += 2 * hidden

    mtp_layers = int(architecture["mtp_num_layers"])
    if mtp_layers:
        common, routed = _layer_parameter_terms(architecture, bool(pattern[-1]))
        stage_common[-1] += common * mtp_layers
        stage_routed[-1] += routed * mtp_layers

    sharded = [
        common / tp + routed / (max(1, ep) * max(1, etp))
        for common, routed in zip(stage_common, stage_routed)
    ]
    total = sum(stage_common) + sum(stage_routed)
    return {
        "total_parameters": total,
        "most_loaded_shard_parameters": max(sharded),
        "stage_shard_parameters": sharded,
        "pipeline_layer_counts": layer_counts,
        "routed_expert_sharding": {
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
    }


def _parameter_footprint(
    architecture: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Return fixed model parameters per device, anchored to the train log.

    The architecture formula remains the fallback and supplies relative stage
    and expert-sharding shares.  When Megatron printed its actual shard size,
    an unchanged topology uses that measurement verbatim.  A changed topology
    re-shards the logged total (or scales the logged shard by the analytical
    topology ratio when a global EP total cannot be reconstructed).
    """

    analytical = _analytical_parameter_footprint(architecture, runtime)
    profile = architecture.get("parameter_profile")
    if not isinstance(profile, Mapping) or not profile:
        return {
            **analytical,
            "parameter_source": "analytical_model_architecture",
        }

    logged_shard = profile.get("most_loaded_shard_parameters")
    reference_topology = profile.get("reference_topology")
    if not _is_number(logged_shard) or not isinstance(reference_topology, Mapping):
        return {
            **analytical,
            "parameter_source": "analytical_model_architecture",
        }

    topology_names = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    )
    candidate_topology = {name: int(runtime[name]) for name in topology_names}
    normalized_reference_topology = {
        name: int(reference_topology.get(name, 1) or 1) for name in topology_names
    }
    topology_unchanged = candidate_topology == normalized_reference_topology

    analytical_total = float(analytical["total_parameters"])
    analytical_shards = [float(value) for value in analytical["stage_shard_parameters"]]
    analytical_max = float(analytical["most_loaded_shard_parameters"])
    logged_total = profile.get("total_parameters")

    if topology_unchanged:
        anchored_max = float(logged_shard)
        scale = anchored_max / analytical_max if analytical_max > 0 else 1.0
        anchored_shards = [value * scale for value in analytical_shards]
        total = (
            float(logged_total)
            if _is_number(logged_total)
            else analytical_total * scale
        )
        source = "reference_train_log_exact_shard_unchanged_topology"
    elif _is_number(logged_total) and analytical_total > 0:
        # The analytical model is used only for each candidate stage's share;
        # the absolute fixed-parameter count comes from the reference log.
        total = float(logged_total)
        scale = total / analytical_total
        anchored_shards = [value * scale for value in analytical_shards]
        anchored_max = max(anchored_shards)
        source = "reference_train_log_total_resharded_for_candidate_topology"
    else:
        reference_runtime = dict(runtime)
        reference_runtime.update(normalized_reference_topology)
        reference_analytical = _analytical_parameter_footprint(
            architecture, reference_runtime
        )
        reference_analytical_max = float(
            reference_analytical["most_loaded_shard_parameters"]
        )
        scale = (
            float(logged_shard) / reference_analytical_max
            if reference_analytical_max > 0
            else 1.0
        )
        anchored_shards = [value * scale for value in analytical_shards]
        anchored_max = max(anchored_shards)
        total = analytical_total * scale
        source = "reference_train_log_shard_scaled_by_topology_ratio"

    return {
        **analytical,
        "total_parameters": total,
        "most_loaded_shard_parameters": anchored_max,
        "stage_shard_parameters": anchored_shards,
        "parameter_source": source,
        "logged_parameter_profile": dict(profile),
        "candidate_topology": candidate_topology,
        "analytical_total_parameters": analytical_total,
        "analytical_most_loaded_shard_parameters": analytical_max,
    }


def _normalized_modules(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {
            item.strip().strip("'\"")
            for item in value.strip("[]").split(",")
            if item.strip()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {str(item) for item in value}
    return {str(value)}


def _dense_selective_layer_bytes(
    architecture: Mapping[str, Any], runtime: Mapping[str, Any], tokens: float
) -> float:
    """Megatron's classic selective-recompute proxy, normalized to bytes.

    The 18/10/8/4 coefficients already describe BF16 bytes.  Scale them only
    when the resolved activation dtype is wider; do not multiply by dtype a
    second time.
    """

    hidden = float(architecture["hidden_size"])
    ffn = float(architecture["ffn_hidden_size"])
    tp = int(runtime["tensor_model_parallel_size"])
    dtype_scale = float(architecture.get("activation_dtype_bytes", 2)) / 2.0
    if runtime["sequence_parallel"]:
        value = tokens * hidden * (18.0 + 4.0 * ffn / hidden) / tp
    else:
        value = tokens * hidden * (10.0 + (8.0 + 4.0 * ffn / hidden) / tp)
    return value * dtype_scale


def _non_recomputed_attention_bytes(
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    tokens: float,
    sequence: float,
) -> float:
    """Extra tensors retained when core attention is not selectively recomputed."""

    tp = int(runtime["tensor_model_parallel_size"])
    act_bytes = int(architecture.get("activation_dtype_bytes", 2))
    heads = float(architecture["num_attention_heads"])
    q_width = float(architecture["kv_channels"]) * heads
    kv_width = float(architecture["kv_channels"]) * float(
        architecture["num_query_groups"]
    )
    linear_qkv = tokens * (q_width + 2.0 * kv_width) * act_bytes / tp
    if "flash" in str(runtime.get("attention_backend", "flash")).lower():
        return linear_qkv
    # The classic unfused attention score/probability term is quadratic in S.
    quadratic = 5.0 * tokens * heads * max(1.0, sequence) / tp
    return linear_qkv + quadratic


def _training_activation_bytes(
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    upper_sequence: bool = False,
) -> tuple[float, dict[str, Any]]:
    tokens = float(
        runtime["tokens_per_cp_rank_upper"]
        if upper_sequence
        else runtime["tokens_per_cp_rank"]
    )
    sequence = float(
        runtime["seq_length_upper"] if upper_sequence else runtime["seq_length"]
    )
    hidden = float(architecture["hidden_size"])
    layers = int(architecture["num_layers"])
    tp = int(runtime["tensor_model_parallel_size"])
    pp = int(runtime["pipeline_model_parallel_size"])
    vpp = runtime["virtual_pipeline_model_parallel_size"]
    num_microbatches = int(runtime["num_microbatches"])
    act_bytes = int(architecture.get("activation_dtype_bytes", 2))
    layer_counts = _stage_layer_counts(
        layers,
        pp,
        runtime.get("num_layers_in_first_pipeline_stage"),
        runtime.get("num_layers_in_last_pipeline_stage"),
    )

    granularity_raw = runtime.get("recompute_granularity")
    granularity = (
        str(granularity_raw).lower()
        if granularity_raw not in (None, False, "none", "None")
        else "none"
    )
    method = str(runtime.get("recompute_method") or "uniform").lower()
    modules = _normalized_modules(runtime.get("recompute_modules"))
    selective_per_layer = _dense_selective_layer_bytes(architecture, runtime, tokens)
    normal_per_layer = selective_per_layer + _non_recomputed_attention_bytes(
        architecture, runtime, tokens, sequence
    )
    checkpoint_input = (
        tokens * hidden * act_bytes / (tp if runtime["sequence_parallel"] else 1)
    )

    recompute_num_raw = runtime.get("recompute_num_layers")
    recompute_num = (
        min(layers, int(recompute_num_raw))
        if _is_number(recompute_num_raw) and int(recompute_num_raw) > 0
        else layers
    )
    remaining_block_recompute = recompute_num if method == "block" else 0
    stage_body_bytes: list[float] = []
    stage_in_flight: list[int] = []
    for stage, stage_layers in enumerate(layer_counts):
        in_flight = 1 if pp == 1 else min(num_microbatches, max(1, pp - stage))
        stage_in_flight.append(in_flight)
        if granularity == "full" and method == "uniform":
            checkpoints = math.ceil(stage_layers / max(1, recompute_num))
            one_microbatch = checkpoints * checkpoint_input
        elif granularity == "full" and method == "block":
            recomputed_here = min(stage_layers, remaining_block_recompute)
            remaining_block_recompute -= recomputed_here
            one_microbatch = (
                recomputed_here * checkpoint_input
                + (stage_layers - recomputed_here) * normal_per_layer
            )
        elif granularity == "selective" and (not modules or "core_attn" in modules):
            one_microbatch = stage_layers * selective_per_layer
        else:
            one_microbatch = stage_layers * normal_per_layer
        body = one_microbatch * in_flight
        if vpp is not None:
            body *= 1.0 + (pp - 1) / (pp * int(vpp))
        stage_body_bytes.append(body)

    # First stage owns embedding inputs for every in-flight micro-batch.
    embedding_bytes = tokens * (8.0 + hidden) * stage_in_flight[0]
    stage_peak_bytes = list(stage_body_bytes)
    stage_peak_bytes[0] += embedding_bytes

    # Last stage owns final norm and, unless fused, TP-sharded vocabulary logits.
    final_norm_bytes = tokens * hidden * act_bytes
    logits_one_copy = 0.0
    logits_copies = 0
    if not runtime["use_fused_kernels"]:
        logits_one_copy = (
            tokens * float(architecture["padded_vocab_size"]) / tp * act_bytes
        )
        logits_copies = 2 if runtime.get("calculate_entropy") else 1
    stage_peak_bytes[-1] += final_norm_bytes + logits_one_copy * logits_copies

    peak = max(stage_peak_bytes)
    limitations: list[str] = []
    if architecture.get("num_experts") is not None:
        limitations.append("moe_routing_and_dispatch_workspace_require_rank_histograms")
    if architecture.get("multi_latent_attention"):
        limitations.append("mla_activation_liveness_requires_calibration")
    if runtime.get("dynamic_batch") and not runtime.get("remove_padding"):
        limitations.append(
            "dynamic_bshd_padded_shape_is_not_recoverable_from_token_cap"
        )
    if granularity == "selective" and modules and "core_attn" not in modules:
        limitations.append("non_core_attn_selective_modules_use_no_recompute_proxy")

    return peak, {
        "formula": "training_saved_activation_by_pipeline_stage",
        "tokens_per_cp_rank": tokens,
        "token_source": runtime.get("token_source"),
        "sequence_length": sequence,
        "micro_batch_size": runtime["micro_batch_size"],
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "context_parallel_size": runtime["context_parallel_size"],
        "sequence_parallel": runtime["sequence_parallel"],
        "recompute_granularity": granularity,
        "recompute_method": method,
        "recompute_num_layers": recompute_num,
        "recompute_modules": sorted(modules),
        "num_microbatches": num_microbatches,
        "pipeline_layer_counts": layer_counts,
        "stage_in_flight_microbatches": stage_in_flight,
        "stage_body_bytes": stage_body_bytes,
        "stage_peak_bytes": stage_peak_bytes,
        "peak_pipeline_stage": stage_peak_bytes.index(peak),
        "embedding_bytes_first_stage": embedding_bytes,
        "final_norm_bytes_last_stage": final_norm_bytes,
        "vocab_logits_one_copy_bytes": logits_one_copy,
        "vocab_logits_copies": logits_copies,
        "requires_calibration": bool(limitations)
        or runtime.get("requires_packed_shape_calibration", False),
        "limitations": limitations,
        "upper_sequence": upper_sequence,
    }


def _log_prob_activation_bytes(
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    phase: str,
    upper_sequence: bool = False,
) -> tuple[float, dict[str, Any]]:
    if phase not in {"actor_log_prob", "ref_log_prob"}:
        raise ValueError(f"invalid log-prob phase: {phase}")
    tokens = float(
        runtime["tokens_per_cp_rank_upper"]
        if upper_sequence
        else runtime["tokens_per_cp_rank"]
    )
    hidden = float(architecture["hidden_size"])
    tp = int(runtime["tensor_model_parallel_size"])
    pp = int(runtime["pipeline_model_parallel_size"])
    act_bytes = int(architecture.get("activation_dtype_bytes", 2))

    # This is a one-live-layer proxy.  Forward-only log-prob must not retain
    # one copy for every transformer layer as training does.
    body_live_bytes = _dense_selective_layer_bytes(architecture, runtime, tokens)
    pipeline_buffer_bytes = tokens * hidden * act_bytes if pp > 1 else 0.0
    non_last_stage_bytes = body_live_bytes + pipeline_buffer_bytes

    logits_one_copy = 0.0
    logits_copies = 0
    if not runtime["use_fused_kernels"]:
        logits_one_copy = (
            tokens * float(architecture["padded_vocab_size"]) / tp * act_bytes
        )
        logits_copies = 2 if phase == "actor_log_prob" else 1
    vocab_bytes = logits_one_copy * logits_copies
    output_fields = 2 if phase == "actor_log_prob" else 1
    result_accumulation_bytes = (
        2.0
        * output_fields
        * float(runtime["local_call_samples"])
        * float(runtime["response_length"])
        * 4.0
    )
    last_stage_bytes = body_live_bytes + vocab_bytes + result_accumulation_bytes
    peak = max(non_last_stage_bytes, last_stage_bytes)

    limitations: list[str] = []
    if runtime["use_fused_kernels"]:
        limitations.append("fused_cross_entropy_workspace_requires_calibration")
    if (
        runtime.get("is_lora")
        and phase == "ref_log_prob"
        and runtime["use_fused_kernels"]
    ):
        limitations.append("lora_ref_reuses_actor_fused_entropy_kernel")
    if architecture.get("num_experts") is not None:
        limitations.append("moe_routing_and_dispatch_workspace_require_rank_histograms")
    if architecture.get("multi_latent_attention"):
        limitations.append("mla_live_activation_requires_calibration")
    if runtime.get("dynamic_batch") and not runtime.get("remove_padding"):
        limitations.append(
            "dynamic_bshd_padded_shape_is_not_recoverable_from_token_cap"
        )

    return peak, {
        "formula": "forward_only_one_live_layer_plus_last_stage_logits",
        "phase": phase,
        "tokens_per_cp_rank": tokens,
        "token_source": runtime.get("token_source"),
        "body_live_bytes": body_live_bytes,
        "pipeline_buffer_bytes": pipeline_buffer_bytes,
        "non_last_stage_bytes": non_last_stage_bytes,
        "last_stage_bytes": last_stage_bytes,
        "peak_stage_kind": (
            "last" if last_stage_bytes >= non_last_stage_bytes else "non_last"
        ),
        "vocab_logits_one_copy_bytes": logits_one_copy,
        "vocab_logits_copies": logits_copies,
        "vocab_bytes": vocab_bytes,
        "result_accumulation_bytes": result_accumulation_bytes,
        "use_fused_kernels": runtime["use_fused_kernels"],
        "requires_calibration": True,
        "limitations": limitations,
        "upper_sequence": upper_sequence,
    }


def _activation_bytes(
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    upper_sequence: bool = False,
) -> tuple[float, dict[str, Any]]:
    phase = str(runtime.get("phase", "training"))
    if phase in {"actor_log_prob", "ref_log_prob"}:
        return _log_prob_activation_bytes(
            architecture,
            runtime,
            phase=phase,
            upper_sequence=upper_sequence,
        )
    return _training_activation_bytes(
        architecture, runtime, upper_sequence=upper_sequence
    )


def _lora_trainable_shard_parameters(
    architecture: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Conservative structural estimate for Megatron LoRA trainable weights."""

    rank = int(runtime.get("lora_rank", 0))
    if rank <= 0:
        return 0.0, {"source": "lora_adapter_path_without_rank", "approximate": True}
    hidden = float(architecture["hidden_size"])
    ffn = float(architecture["ffn_hidden_size"])
    heads = float(architecture["num_attention_heads"])
    q_width = float(architecture["kv_channels"]) * heads
    kv_width = float(architecture["kv_channels"]) * float(
        architecture["num_query_groups"]
    )
    gated_width = ffn * (2.0 if architecture["swiglu"] else 1.0)
    terms = {
        "linear_qkv": rank * (hidden + q_width + 2.0 * kv_width),
        "linear_proj": rank * (q_width + hidden),
        "linear_fc1": rank * (hidden + gated_width),
        "linear_fc2": rank * (ffn + hidden),
    }
    raw_targets = runtime.get("lora_target_modules", "all-linear")
    if isinstance(raw_targets, str):
        targets = {raw_targets}
    elif isinstance(raw_targets, Sequence):
        targets = {str(item) for item in raw_targets}
    else:
        targets = {"all-linear"}
    aliases = {
        "q_proj": "linear_qkv",
        "k_proj": "linear_qkv",
        "v_proj": "linear_qkv",
        "o_proj": "linear_proj",
        "gate_proj": "linear_fc1",
        "up_proj": "linear_fc1",
        "down_proj": "linear_fc2",
    }
    if "all-linear" in targets or "all" in targets:
        selected = set(terms)
    else:
        selected = {
            canonical
            for target in targets
            for alias, canonical in aliases.items()
            if alias in target
        } | {name for name in terms if any(name in target for target in targets)}
    if not selected:
        selected = set(terms)
    total = sum(terms[name] for name in selected) * int(architecture["num_layers"])
    pp = int(runtime["pipeline_model_parallel_size"])
    tp = int(runtime["tensor_model_parallel_size"])
    # LoRA matrices follow their target linear's PP/TP placement.  This is an
    # upper-oriented even-stage approximation until trainable numel is logged.
    shard = total / max(1, pp * tp)
    return shard, {
        "source": "analytical_lora_target_linears",
        "rank": rank,
        "targets": sorted(targets),
        "selected_megatron_linears": sorted(selected),
        "global_trainable_parameters": total,
        "approximate": True,
    }


def _phase_residency_components(
    phase: str,
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    footprint: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    shard = float(footprint["most_loaded_shard_parameters"])
    weight_bytes = int(architecture["bytes_per_weight"])
    is_lora = bool(runtime.get("is_lora"))
    if is_lora:
        trainable_shard, trainable_details = _lora_trainable_shard_parameters(
            architecture, runtime
        )
    else:
        trainable_shard = shard
        trainable_details = {
            "source": "full_parameter_training",
            "approximate": False,
        }

    actor_offload = bool(runtime.get("actor_param_offload"))
    ref_offload = bool(runtime.get("ref_param_offload"))
    grad_offload = bool(runtime.get("actor_grad_offload"))
    optimizer_offload = bool(runtime.get("optimizer_offload"))

    if is_lora:
        resident_model_copies = 1.0
        actor_params_active = True
    elif phase == "actor_log_prob":
        resident_model_copies = 1.0 + (0.0 if ref_offload else 1.0)
        actor_params_active = True
    elif phase == "ref_log_prob":
        resident_model_copies = 1.0 + (0.0 if actor_offload else 1.0)
        actor_params_active = not actor_offload
    else:
        resident_model_copies = 1.0 + (0.0 if ref_offload else 1.0)
        actor_params_active = True

    if phase == "training":
        gradients_resident = True
        optimizer_resident = True
    else:
        gradients_resident = (
            actor_params_active and not grad_offload and not actor_offload
        )
        optimizer_resident = not optimizer_offload

    weights = shard * resident_model_copies * weight_bytes
    gradients = trainable_shard * 4.0 if gradients_resident else 0.0
    if runtime["use_distributed_optimizer"]:
        optimizer_bpp = 12.0 / int(runtime["data_parallel_size"])
    else:
        optimizer_bpp = 12.0
    optimizer = trainable_shard * optimizer_bpp if optimizer_resident else 0.0
    return {
        "resident_model_weights_mb": weights / MIB,
        "resident_gradients_mb": gradients / MIB,
        "resident_optimizer_state_mb": optimizer / MIB,
    }, {
        "model_shard_parameters": shard,
        "resident_model_copies": resident_model_copies,
        "trainable_shard_parameters": trainable_shard,
        "trainable_parameter_details": trainable_details,
        "gradients_resident": gradients_resident,
        "optimizer_resident": optimizer_resident,
        "optimizer_bytes_per_trainable_parameter": optimizer_bpp,
        "is_lora": is_lora,
    }


def _component_dependencies(phase: str) -> dict[str, set[str]]:
    keys = _phase_keys(phase)
    residency_controls = {
        ACTOR_PARAM_OFFLOAD_KEY,
        ACTOR_GRAD_OFFLOAD_KEY,
        ACTOR_OPTIMIZER_OFFLOAD_KEY,
        REF_PARAM_OFFLOAD_KEY,
        LORA_RANK_KEY,
        LORA_ADAPTER_PATH_KEY,
        LORA_TARGET_MODULES_KEY,
        USE_DISTRIBUTED_OPTIMIZER_KEY,
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    }
    model = {
        MODEL_KEY,
        PARAM_DTYPE_KEY,
        keys["tp"],
        keys["pp"],
        keys["ep"],
        keys["etp"],
        FIRST_PIPELINE_LAYERS_KEY,
        LAST_PIPELINE_LAYERS_KEY,
        *residency_controls,
    }
    activation = {
        MODEL_KEY,
        PARAM_DTYPE_KEY,
        keys["micro"],
        keys["tp"],
        keys["pp"],
        keys["cp"],
        keys["vpp"],
        keys["sp"],
        keys["dynamic_batch"],
        keys["dynamic_fallback"],
        keys["max_tokens"],
        keys["max_tokens_fallback"],
        keys["remove_padding"],
        keys["remove_padding_fallback"],
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
        TRAIN_BATCH_KEY,
        ROLLOUT_N_KEY,
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    }
    optimizer: set[str] = set()
    if phase == "training":
        activation |= {
            PPO_MINI_BATCH_KEY,
            RECOMPUTE_GRANULARITY_KEY,
            RECOMPUTE_METHOD_KEY,
            RECOMPUTE_NUM_LAYERS_KEY,
            RECOMPUTE_MODULES_KEY,
            ATTENTION_BACKEND_KEY,
            ENTROPY_COEFF_KEY,
            ACTOR_FUSED_KERNELS_KEY,
            MODEL_FUSED_KERNELS_KEY,
            FIRST_PIPELINE_LAYERS_KEY,
            LAST_PIPELINE_LAYERS_KEY,
        }
        optimizer = set(residency_controls) | model | {keys["cp"]}
    elif phase == "actor_log_prob":
        activation |= {ACTOR_FUSED_KERNELS_KEY, MODEL_FUSED_KERNELS_KEY}
    else:
        activation.add(REF_FUSED_KERNELS_KEY)
        # LoRA ref executes the actor module and therefore follows actor fused state.
        activation |= {ACTOR_FUSED_KERNELS_KEY, MODEL_FUSED_KERNELS_KEY}
    uncalibrated = {
        keys["dynamic_batch"],
        keys["remove_padding"],
        "actor_rollout_ref.rollout.free_cache_engine",
        ROLLOUT_UTILIZATION_KEY,
        LORA_RANK_KEY,
        LORA_ADAPTER_PATH_KEY,
        LORA_TARGET_MODULES_KEY,
    }
    if phase in {"actor_log_prob", "training"}:
        uncalibrated |= {ACTOR_FUSED_KERNELS_KEY, MODEL_FUSED_KERNELS_KEY}
    if phase == "ref_log_prob":
        uncalibrated |= {
            REF_FUSED_KERNELS_KEY,
            ACTOR_FUSED_KERNELS_KEY,
            MODEL_FUSED_KERNELS_KEY,
        }
    return {
        "model": model,
        "activation": activation,
        "optimizer": optimizer,
        "uncalibrated": uncalibrated,
        "all": model | activation | optimizer | uncalibrated,
    }


def _looks_memory_sensitive(key: str) -> bool:
    if key in {
        MODEL_KEY,
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
        TRAIN_BATCH_KEY,
        ROLLOUT_N_KEY,
        *WORLD_SIZE_KEYS,
    }:
        return True
    markers = (
        "parallel_size",
        "micro_batch",
        "max_token",
        "dynamic_bsz",
        "remove_padding",
        "offload",
        "recompute",
        "fused_kernel",
        "attention_backend",
        "lora",
        "target_modules",
        "num_layers_in_",
        "distributed_optimizer",
        "entropy_coeff",
        "gpu_memory_utilization",
        "free_cache_engine",
    )
    return key.startswith("actor_rollout_ref.") and any(
        marker in key for marker in markers
    )


def _known_compute_memory_keys() -> set[str]:
    known: set[str] = set()
    for phase in ("actor_log_prob", "ref_log_prob", "training"):
        known |= _component_dependencies(phase)["all"]
    return known


def _component_values(
    phase: str,
    architecture: Mapping[str, Any],
    runtime: Mapping[str, Any],
    needed: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    if architecture.get("mtp_num_layers") and runtime.get("use_fused_kernels"):
        runtime = dict(runtime)
        runtime["use_fused_kernels"] = False
        runtime["fused_kernels_forced_off_reason"] = "mtp_enabled"
    values: dict[str, float] = {}
    details: dict[str, Any] = {}
    footprint: dict[str, Any] | None = None
    if needed & {"model", "optimizer"}:
        footprint = _parameter_footprint(architecture, runtime)
        details["parameter_footprint"] = footprint
    if needed & {"model", "optimizer"} and footprint is not None:
        residency_values, residency_details = _phase_residency_components(
            phase, architecture, runtime, footprint
        )
        if "model" in needed:
            values["resident_model_weights_mb"] = residency_values[
                "resident_model_weights_mb"
            ]
            values["resident_gradients_mb"] = residency_values["resident_gradients_mb"]
        if "optimizer" in needed or "model" in needed:
            # Optimizer residency changes with offload even outside training,
            # and training always loads it before update_policy.
            values["resident_optimizer_state_mb"] = residency_values[
                "resident_optimizer_state_mb"
            ]
        details["phase_residency"] = residency_details
    if "activation" in needed:
        activation, activation_details = _activation_bytes(
            architecture, runtime, upper_sequence=False
        )
        activation_upper, upper_details = _activation_bytes(
            architecture, runtime, upper_sequence=True
        )
        values["activation_mb"] = activation / MIB
        values["activation_upper_sequence_mb"] = activation_upper / MIB
        details["activation"] = activation_details
        details["activation_upper"] = upper_details
    return values, details


def _activation_calibration(
    phase: str,
    reference_parameters: Mapping[str, Any],
    reference_architecture: Mapping[str, Any],
    reference_runtime: Mapping[str, Any],
    reference_activation_mb: float,
    reference_measurement: Mapping[str, Any],
    reference_context: Mapping[str, Any],
    reference_length: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float, list[dict[str, Any]]]:
    """Calibrate analytical activation deltas with earlier comparable trials."""

    reference_mb = reference_measurement.get("memory_mb")
    reference_capacity_mb = reference_measurement.get("gpu_capacity_mb")
    if reference_mb is None:
        return None, 0.0, []

    dependencies = _component_dependencies(phase)
    calibration_keys = {
        _phase_keys(phase)["micro"],
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
    }
    observations: list[dict[str, Any]] = []
    slopes: list[float] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        parameters = trial.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        if trial.get("trial_id") == reference_context.get("trial_id"):
            continue
        all_changed = _changed_keys(reference_parameters, parameters)
        memory_regime_changes = {
            key for key in all_changed if _looks_memory_sensitive(key)
        }
        if memory_regime_changes - calibration_keys:
            continue
        relevant = all_changed & dependencies["all"]
        if not relevant or relevant - calibration_keys:
            continue
        if relevant & dependencies["uncalibrated"]:
            continue

        trial_context = _extract_log_context(trial, parameters)
        trial_length = trial_context["length"]
        if trial_length.get("source") == "configured_maximum" and not any(
            reference_parameters.get(key) != parameters.get(key)
            for key in (PROMPT_LENGTH_KEY, RESPONSE_LENGTH_KEY)
        ):
            # An OOM may happen before a step metric is emitted.  It still
            # supplies a valid phase peak; use the reference workload length
            # rather than incorrectly switching this one observation to the
            # configured maximum.
            trial_length = dict(reference_length)
        trial_runtime = _runtime_args(phase, parameters, trial_context, trial_length)
        trial_activation, _ = _activation_bytes(
            reference_architecture, trial_runtime, upper_sequence=False
        )
        theoretical_delta_mb = trial_activation / MIB - reference_activation_mb
        if abs(theoretical_delta_mb) < 1e-6:
            continue

        measurement = _phase_measurements(trial, _trial_log_path(trial))[phase]
        actual_mb = measurement.get("memory_mb")
        if actual_mb is None and measurement.get("memory_pct") is not None:
            capacity = measurement.get("gpu_capacity_mb") or reference_capacity_mb
            if capacity is not None:
                actual_mb = float(capacity) * float(measurement["memory_pct"]) / 100.0
        if actual_mb is None:
            continue
        actual_delta_mb = float(actual_mb) - float(reference_mb)
        slope = actual_delta_mb / theoretical_delta_mb
        if not math.isfinite(slope) or slope <= 0:
            continue
        slopes.append(slope)
        observations.append(
            {
                "trial_id": trial.get("trial_id"),
                "changed_parameters": sorted(relevant),
                "theoretical_activation_delta_mb": theoretical_delta_mb,
                "observed_phase_delta_mb": actual_delta_mb,
                "delta_multiplier": slope,
                "result": trial.get("result"),
                "error_type": (
                    trial.get("error", {}).get("type")
                    if isinstance(trial.get("error"), Mapping)
                    else None
                ),
            }
        )

    if not slopes:
        return None, 0.0, observations
    multiplier = statistics.median(slopes)
    residual = max(
        (
            abs(
                float(item["observed_phase_delta_mb"])
                - multiplier * float(item["theoretical_activation_delta_mb"])
            )
            for item in observations
        ),
        default=0.0,
    )
    return multiplier, residual, observations


def _compute_projection(
    phase: str,
    measurement: Mapping[str, Any],
    reference_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    reference_context: Mapping[str, Any],
    candidate_context: Mapping[str, Any],
    reference_length: Mapping[str, Any],
    candidate_length: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    changed = _changed_keys(reference_parameters, candidate_parameters)
    dependencies = _component_dependencies(phase)
    relevant_changes = changed & dependencies["all"]
    unknown_memory_changes = {
        key
        for key in changed
        if _looks_memory_sensitive(key) and key not in _known_compute_memory_keys()
    }
    reference_mb = measurement.get("memory_mb")
    capacity_mb = measurement.get("gpu_capacity_mb")
    reference_pct = measurement.get("memory_pct")

    if not changed:
        return {
            "reference_mb": reference_mb,
            "projected_mb": reference_mb,
            "delta_mb": 0.0 if reference_mb is not None else None,
            "uncertainty_mb": 0.0 if reference_mb is not None else None,
            "confidence": "high",
            "model": "unchanged_reference_phase",
            "drivers": {"affected_components": []},
            "uncalibrated_changes": [],
        }

    if not relevant_changes and not unknown_memory_changes:
        variability_mb = (
            max(512.0, 0.05 * float(capacity_mb)) if capacity_mb is not None else 2048.0
        )
        return {
            "reference_mb": reference_mb,
            "projected_mb": reference_mb,
            "delta_mb": 0.0 if reference_mb is not None else None,
            "uncertainty_mb": variability_mb if reference_mb is not None else None,
            "confidence": "medium",
            "model": "unchanged_reference_phase_with_run_variability",
            "drivers": {
                "affected_components": [],
                "changed_parameters": [],
                "variability_floor_mb": variability_mb,
            },
            "uncalibrated_changes": [],
        }

    if not relevant_changes:
        uncertainty_mb = (
            0.08 * float(capacity_mb) * len(unknown_memory_changes)
            if capacity_mb is not None
            else 2048.0 * len(unknown_memory_changes)
        )
        return {
            "reference_mb": reference_mb,
            "projected_mb": reference_mb,
            "delta_mb": 0.0 if reference_mb is not None else None,
            "uncertainty_mb": uncertainty_mb if reference_mb is not None else None,
            "confidence": "low",
            "model": "reference_peak_with_unknown_memory_sensitive_changes",
            "drivers": {
                "affected_components": [],
                "changed_parameters": sorted(unknown_memory_changes),
            },
            "uncalibrated_changes": sorted(unknown_memory_changes),
        }

    affected: set[str] = set()
    for component in ("model", "activation", "optimizer"):
        if relevant_changes & dependencies[component]:
            affected.add(component)
    uncalibrated = sorted(
        (relevant_changes & dependencies["uncalibrated"]) | unknown_memory_changes
    )

    # A model change requires metadata from the candidate model.  Otherwise
    # both sides share one architecture object and only runtime components move.
    reference_architecture = _model_architecture(
        reference_parameters, reference_context
    )
    if MODEL_KEY in relevant_changes:
        candidate_architecture = _model_architecture(
            candidate_parameters, candidate_context
        )
    else:
        candidate_architecture = reference_architecture

    reference_runtime = _runtime_args(
        phase, reference_parameters, reference_context, reference_length
    )
    candidate_runtime = _runtime_args(
        phase, candidate_parameters, reference_context, candidate_length
    )
    reference_components, reference_details = _component_values(
        phase, reference_architecture, reference_runtime, affected
    )
    candidate_components, candidate_details = _component_values(
        phase, candidate_architecture, candidate_runtime, affected
    )
    structural_uncertainties: list[str] = []
    if candidate_architecture.get("num_experts") is not None:
        structural_uncertainties.append("moe_route_imbalance_and_expert_dp_state")
    if candidate_architecture.get("multi_latent_attention"):
        structural_uncertainties.append("mla_activation_liveness")
    if candidate_runtime.get("is_lora") and affected & {"model", "optimizer"}:
        structural_uncertainties.append("lora_trainable_shard_is_analytical")
    if candidate_runtime.get("dynamic_batch") and not candidate_runtime.get(
        "remove_padding"
    ):
        structural_uncertainties.append("dynamic_bshd_padded_microbatch_shape")

    raw_delta_components: dict[str, float] = {}
    for name in set(reference_components) | set(candidate_components):
        if name.endswith("_upper_sequence_mb"):
            continue
        raw_delta_components[name] = candidate_components.get(
            name, 0.0
        ) - reference_components.get(name, 0.0)
    delta_components = dict(raw_delta_components)

    activation_multiplier = None
    calibration_residual_mb = 0.0
    calibration_observations: list[dict[str, Any]] = []
    if "activation" in affected:
        activation_multiplier, calibration_residual_mb, calibration_observations = (
            _activation_calibration(
                phase,
                reference_parameters,
                reference_architecture,
                reference_runtime,
                reference_components["activation_mb"],
                measurement,
                {**reference_context, "trial_id": None},
                reference_length,
                trials,
            )
        )
        if activation_multiplier is not None:
            delta_components["activation_mb"] = (
                raw_delta_components.get("activation_mb", 0.0) * activation_multiplier
            )
    delta_mb = sum(delta_components.values())

    reference_activation_upper = reference_components.get(
        "activation_upper_sequence_mb"
    )
    candidate_activation_upper = candidate_components.get(
        "activation_upper_sequence_mb"
    )
    sequence_delta_gap = 0.0
    if (
        reference_activation_upper is not None
        and candidate_activation_upper is not None
    ):
        upper_delta = candidate_activation_upper - reference_activation_upper
        sequence_delta_gap = max(
            0.0, upper_delta - delta_components.get("activation_mb", 0.0)
        )

    topology_changed = bool(
        relevant_changes
        & (dependencies["model"] | {ACTOR_CP_KEY, REF_CP_KEY, ACTOR_SP_KEY, REF_SP_KEY})
    )
    base_uncertainty = max(
        256.0,
        calibration_residual_mb,
        abs(delta_mb) * (0.20 if topology_changed else 0.12),
    )
    uncertainty_mb = base_uncertainty + sequence_delta_gap
    if topology_changed:
        uncertainty_mb += 512.0
    if uncalibrated:
        if capacity_mb is not None:
            uncertainty_mb += 0.08 * float(capacity_mb) * len(uncalibrated)
        else:
            uncertainty_mb += 2048.0 * len(uncalibrated)
    if structural_uncertainties:
        if capacity_mb is not None:
            uncertainty_mb += 0.05 * float(capacity_mb) * len(structural_uncertainties)
        else:
            uncertainty_mb += 1024.0 * len(structural_uncertainties)

    projected_mb = (
        max(0.0, float(reference_mb) + delta_mb) if reference_mb is not None else None
    )
    if reference_mb is None and reference_pct is not None and capacity_mb is not None:
        reference_mb = float(reference_pct) * float(capacity_mb) / 100.0
        projected_mb = max(0.0, reference_mb + delta_mb)

    # With only one anchor, a large activation increase can trigger allocator,
    # workspace, and kernel-regime jumps that the analytical tensor formula
    # misses.  Scaling the whole observed peak by the analytical activation
    # ratio is deliberately an upper bound, not the point estimate.  Once a
    # comparable trial exists its observed multiplier calibrates the point.
    reference_activation = reference_components.get("activation_mb")
    candidate_activation = candidate_components.get("activation_mb")
    conservative_upper_mb = None
    if (
        reference_mb is not None
        and reference_activation not in (None, 0)
        and candidate_activation is not None
        and candidate_activation > reference_activation
    ):
        activation_ratio = candidate_activation / reference_activation
        conservative_upper_mb = float(reference_mb) * activation_ratio
        if projected_mb is not None:
            uncertainty_mb = max(
                uncertainty_mb,
                conservative_upper_mb - projected_mb,
            )

    return {
        "reference_mb": reference_mb,
        "projected_mb": projected_mb,
        "delta_mb": delta_mb if reference_mb is not None else None,
        "uncertainty_mb": uncertainty_mb if reference_mb is not None else None,
        "confidence": (
            "low"
            if uncalibrated or structural_uncertainties
            else (
                "medium"
                if topology_changed or activation_multiplier is not None
                else "low"
            )
        ),
        "model": "reference_peak_plus_changed_component_deltas",
        "drivers": {
            "affected_components": sorted(affected),
            "changed_parameters": sorted(relevant_changes),
            "delta_components_mb": delta_components,
            "raw_analytical_delta_components_mb": raw_delta_components,
            "reference_components_mb": {
                key: value
                for key, value in reference_components.items()
                if not key.endswith("_upper_sequence_mb")
            },
            "candidate_components_mb": {
                key: value
                for key, value in candidate_components.items()
                if not key.endswith("_upper_sequence_mb")
            },
            "reference_runtime": reference_runtime,
            "candidate_runtime": candidate_runtime,
            "reference_details": reference_details,
            "candidate_details": candidate_details,
            "sequence_upper_delta_gap_mb": sequence_delta_gap,
            "conservative_activation_upper_mb": conservative_upper_mb,
            "activation_delta_calibration": {
                "multiplier": activation_multiplier,
                "residual_mb": calibration_residual_mb,
                "observations": calibration_observations,
            },
            "structural_uncertainties": structural_uncertainties,
            "architecture": reference_architecture,
        },
        "uncalibrated_changes": uncalibrated,
    }


def _same_values(
    left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str]
) -> bool:
    return all(left.get(key) == right.get(key) for key in keys)


def _usable_calibration_trial(trial: Mapping[str, Any], phase: str) -> bool:
    if trial.get("result") not in (None, "success", "early_stopped"):
        return False
    return phase in _phase_percentage_peaks(trial)


def _rollout_regime_keys() -> tuple[str, ...]:
    return (
        MODEL_KEY,
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.enable_prefix_caching",
        "actor_rollout_ref.rollout.enable_chunked_prefill",
        "actor_rollout_ref.rollout.free_cache_engine",
        "actor_rollout_ref.rollout.enforce_eager",
    )


def _rollout_observations(
    reference_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | None]]:
    result: list[dict[str, float | int | None]] = []
    for trial in trials:
        if not isinstance(trial, Mapping) or not _usable_calibration_trial(
            trial, "rollout"
        ):
            continue
        parameters = trial.get("parameters")
        if not isinstance(parameters, Mapping) or not _same_values(
            reference_parameters, parameters, _rollout_regime_keys()
        ):
            continue
        result.append(
            {
                "trial_id": trial.get("trial_id"),
                "gpu_memory_utilization": _number(
                    parameters, ROLLOUT_UTILIZATION_KEY, 0.6
                ),
                "observed_pct": _phase_percentage_peaks(trial)["rollout"],
            }
        )
    return result


def _rollout_utilization_slope(
    observations: Sequence[Mapping[str, float | int | None]],
) -> tuple[float, float, bool]:
    slopes: list[float] = []
    for index, left in enumerate(observations):
        x_left = float(left["gpu_memory_utilization"])
        y_left = float(left["observed_pct"])
        for right in observations[index + 1 :]:
            x_right = float(right["gpu_memory_utilization"])
            y_right = float(right["observed_pct"])
            dx = 100.0 * (x_right - x_left)
            if abs(dx) < 1e-9:
                continue
            slope = (y_right - y_left) / dx
            if math.isfinite(slope) and slope >= 0:
                slopes.append(slope)
    if not slopes:
        return 1.0, 0.0, False
    slope = statistics.median(slopes)
    intercepts = [
        float(item["observed_pct"])
        - slope * 100.0 * float(item["gpu_memory_utilization"])
        for item in observations
    ]
    intercept = statistics.median(intercepts)
    residual = max(
        (
            abs(
                float(item["observed_pct"])
                - (intercept + slope * 100.0 * float(item["gpu_memory_utilization"]))
            )
            for item in observations
        ),
        default=0.0,
    )
    return slope, residual, True


def _rollout_projection(
    measurement: Mapping[str, Any],
    reference_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    relevant = {
        MODEL_KEY,
        ROLLOUT_UTILIZATION_KEY,
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        "actor_rollout_ref.rollout.max_num_seqs",
        ROLLOUT_N_KEY,
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.enable_prefix_caching",
        "actor_rollout_ref.rollout.enable_chunked_prefill",
        "actor_rollout_ref.rollout.free_cache_engine",
        "actor_rollout_ref.rollout.enforce_eager",
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
    }
    changed = _changed_keys(reference_parameters, candidate_parameters) & relevant
    reference_mb = measurement.get("memory_mb")
    capacity_mb = measurement.get("gpu_capacity_mb")
    if not changed:
        return {
            "reference_mb": reference_mb,
            "projected_mb": reference_mb,
            "delta_mb": 0.0 if reference_mb is not None else None,
            "uncertainty_mb": 0.0 if reference_mb is not None else None,
            "confidence": "high",
            "model": "unchanged_reference_phase",
            "drivers": {"affected_components": []},
            "uncalibrated_changes": [],
        }

    reference_util = _number(reference_parameters, ROLLOUT_UTILIZATION_KEY, 0.6)
    candidate_util = _number(candidate_parameters, ROLLOUT_UTILIZATION_KEY, 0.6)
    if not 0 < reference_util <= 1 or not 0 < candidate_util <= 1:
        raise ValueError(f"{ROLLOUT_UTILIZATION_KEY} must be in (0, 1]")
    observations = _rollout_observations(reference_parameters, trials)
    slope, residual_pct, calibrated = _rollout_utilization_slope(observations)
    delta_capacity_fraction = slope * (candidate_util - reference_util)
    delta_mb = (
        float(capacity_mb) * delta_capacity_fraction
        if capacity_mb is not None
        else None
    )
    projected_mb = (
        max(0.0, float(reference_mb) + float(delta_mb))
        if reference_mb is not None and delta_mb is not None
        else None
    )
    uncalibrated = sorted(changed - {ROLLOUT_UTILIZATION_KEY})
    uncertainty_pct = max(2.0 if calibrated else 3.0, residual_pct + 1.0)
    uncertainty_pct += 2.5 * len(uncalibrated)
    uncertainty_mb = (
        float(capacity_mb) * uncertainty_pct / 100.0
        if capacity_mb is not None
        else None
    )
    return {
        "reference_mb": reference_mb,
        "projected_mb": projected_mb,
        "delta_mb": delta_mb,
        "uncertainty_mb": uncertainty_mb,
        "confidence": "low" if uncalibrated else ("high" if calibrated else "medium"),
        "model": "reference_peak_plus_vllm_utilization_capacity_delta",
        "drivers": {
            "affected_components": ["vllm_memory_budget"],
            "gpu_memory_utilization": {
                "from": reference_util,
                "to": candidate_util,
                "calibrated_slope": slope,
                "delta_gpu_capacity_fraction": delta_capacity_fraction,
            },
            "calibration": (
                "matched_trial_pairwise_median_slope"
                if calibrated
                else "one_capacity_pct_per_utilization_pct_prior"
            ),
            "calibration_observations": observations,
            "calibration_residual_pct": residual_pct,
        },
        "uncalibrated_changes": uncalibrated,
    }


def _format_phase_result(
    measurement: Mapping[str, Any],
    projection: Mapping[str, Any],
    memory_limit_pct: float,
) -> dict[str, Any]:
    capacity_mb = measurement.get("gpu_capacity_mb")
    reference_mb = projection.get("reference_mb")
    projected_mb = projection.get("projected_mb")
    delta_mb = projection.get("delta_mb")
    uncertainty_mb = projection.get("uncertainty_mb")
    reference_pct = measurement.get("memory_pct")
    if capacity_mb is not None:
        capacity_mb = float(capacity_mb)
        if reference_mb is not None:
            reference_pct = 100.0 * float(reference_mb) / capacity_mb
        projected_pct = (
            100.0 * float(projected_mb) / capacity_mb
            if projected_mb is not None
            else None
        )
        delta_pct = (
            100.0 * float(delta_mb) / capacity_mb if delta_mb is not None else None
        )
        uncertainty_pct = (
            100.0 * float(uncertainty_mb) / capacity_mb
            if uncertainty_mb is not None
            else None
        )
    else:
        projected_pct = reference_pct if delta_mb == 0 else None
        delta_pct = 0.0 if delta_mb == 0 else None
        uncertainty_pct = None

    upper_mb = (
        float(projected_mb) + float(uncertainty_mb)
        if projected_mb is not None and uncertainty_mb is not None
        else projected_mb
    )
    upper_pct = (
        100.0 * float(upper_mb) / float(capacity_mb)
        if upper_mb is not None and capacity_mb is not None
        else projected_pct
    )
    headroom = (
        memory_limit_pct - float(projected_pct) if projected_pct is not None else None
    )
    upper_headroom = (
        memory_limit_pct - float(upper_pct) if upper_pct is not None else None
    )
    if upper_pct is None:
        risk = "unknown_without_absolute_anchor"
    elif upper_pct >= memory_limit_pct:
        risk = "high"
    elif upper_headroom is not None and upper_headroom < 5:
        risk = "watch"
    else:
        risk = "low"
    ratio = (
        float(projected_mb) / float(reference_mb)
        if projected_mb is not None and reference_mb not in (None, 0)
        else None
    )
    return {
        "reference_memory_mb": _round(float(reference_mb))
        if reference_mb is not None
        else None,
        "reference_memory_gib": _round(float(reference_mb) / GIB_IN_MIB)
        if reference_mb is not None
        else None,
        "delta_memory_mb": _round(float(delta_mb)) if delta_mb is not None else None,
        "delta_memory_gib": _round(float(delta_mb) / GIB_IN_MIB)
        if delta_mb is not None
        else None,
        "projected_memory_mb": _round(float(projected_mb))
        if projected_mb is not None
        else None,
        "projected_memory_gib": _round(float(projected_mb) / GIB_IN_MIB)
        if projected_mb is not None
        else None,
        "upper_bound_memory_mb": _round(float(upper_mb))
        if upper_mb is not None
        else None,
        "upper_bound_memory_gib": _round(float(upper_mb) / GIB_IN_MIB)
        if upper_mb is not None
        else None,
        "gpu_capacity_mb": _round(float(capacity_mb))
        if capacity_mb is not None
        else None,
        "gpu_capacity_gib": _round(float(capacity_mb) / GIB_IN_MIB)
        if capacity_mb is not None
        else None,
        # Compatibility fields retained for the registry/prompts and replay tool.
        "reference_pct": _round(float(reference_pct))
        if reference_pct is not None
        else None,
        "pressure_ratio": _round(ratio, 3),
        "raw_pressure_ratio": _round(ratio, 6),
        "delta_pct": _round(delta_pct),
        "projected_pct": _round(projected_pct),
        "uncertainty_pct": _round(uncertainty_pct),
        "upper_bound_pct": _round(upper_pct),
        "headroom_to_limit_pct": _round(headroom),
        "upper_headroom_to_limit_pct": _round(upper_headroom),
        "risk": risk,
        "confidence": projection["confidence"],
        "model": projection["model"],
        "component_shares": None,
        "drivers": projection["drivers"],
        "uncalibrated_changes": projection["uncalibrated_changes"],
        "reference_measurement_source": measurement.get("source"),
    }


def estimate_phase_memory(
    current_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]] = (),
    memory_limit_pct: float = 92.0,
    reference_trial_id: int | None = None,
) -> dict[str, Any]:
    """Estimate candidate phase peaks from a measured reference plus deltas.

    ``candidate_parameters`` is already the canonical, fully assembled
    candidate produced by ``agent_tools.registry``.  This function never
    overlays or rebuilds it; it only compares it with the selected reference
    trial and recomputes components affected by the changed keys.
    """

    reference = _reference_trial(current_parameters, trials, reference_trial_id)
    if reference is None:
        return {
            "method": "no_observed_anchor",
            "version": 3,
            "confidence": "low",
            "memory_limit_pct": memory_limit_pct,
            "reference_trial_id": None,
            "changed_parameters": {},
            "phases": {
                phase: {
                    "reference_pct": None,
                    "projected_pct": None,
                    "projected_memory_gib": None,
                    "upper_bound_pct": None,
                    "risk": "unknown_without_observed_anchor",
                    "confidence": "low",
                    "model": "no_observed_anchor",
                    "component_shares": None,
                    "drivers": {},
                    "uncalibrated_changes": [],
                }
                for phase in PHASES
            },
            "limitations": ["An explicit phase-measured reference trial is required."],
        }

    reference_parameters = reference.get("parameters")
    if not isinstance(reference_parameters, Mapping):
        raise ValueError("reference trial has no parameter mapping")

    changed = _changed_keys(reference_parameters, candidate_parameters)
    changed_parameters = {
        key: {
            "from": reference_parameters.get(key),
            "to": candidate_parameters.get(key),
        }
        for key in sorted(changed)
    }

    reference_context = _extract_log_context(reference, reference_parameters)
    reference_length = reference_context["length"]
    candidate_length = _candidate_length_profile(
        reference_length, reference_parameters, candidate_parameters
    )
    if MODEL_KEY in changed:
        candidate_model_config, candidate_model_source = _read_model_config(
            candidate_parameters
        )
        candidate_context = {
            "model_config": candidate_model_config,
            "model_config_source": candidate_model_source,
            "resolved": reference_context.get("resolved", {}),
        }
    else:
        candidate_context = reference_context

    log_path = _trial_log_path(reference)
    measurements = _phase_measurements(reference, log_path)
    phases: dict[str, Any] = {}
    for phase in PHASES:
        measurement = measurements[phase]
        if phase == "rollout":
            projection = _rollout_projection(
                measurement,
                reference_parameters,
                candidate_parameters,
                trials,
            )
        else:
            projection = _compute_projection(
                phase,
                measurement,
                reference_parameters,
                candidate_parameters,
                reference_context,
                candidate_context,
                reference_length,
                candidate_length,
                trials,
            )
        phases[phase] = _format_phase_result(measurement, projection, memory_limit_pct)

    confidence_values = [phase["confidence"] for phase in phases.values()]
    confidence = (
        "low"
        if "low" in confidence_values
        else (
            "high" if all(value == "high" for value in confidence_values) else "medium"
        )
    )
    return {
        "method": "measured_reference_plus_changed_component_deltas",
        "version": 3,
        "confidence": confidence,
        "memory_limit_pct": memory_limit_pct,
        "reference_trial_id": reference.get("trial_id"),
        "changed_parameters": changed_parameters,
        "sequence_length": {
            "reference": reference_length,
            "candidate": candidate_length,
        },
        "reference_log_path": reference_context.get("log_path"),
        "resolved_parameter_sources": reference_context.get("resolved", {}),
        "logged_parameter_profile": reference_context.get("parameter_profile", {}),
        "phases": phases,
        "limitations": [
            (
                "Compute estimates add only changed model-state, optimizer, and "
                "activation components to the measured reference peak."
            ),
            (
                "Fixed model-state parameters use Megatron's logged per-rank count; "
                "topology changes re-shard that measured total when TP/PP coverage "
                "is available."
            ),
            (
                "In colocated Megatron workers, ref log-prob uses the actor's "
                "effective model-parallel topology; ref keeps only its own batch "
                "and offload controls."
            ),
            (
                "Fixed micro-batches use stable-step effective tokens and configured "
                "padded length as point/upper proxies. Dynamic phases use their own "
                "per-CP-rank token cap; exact packed shapes still require profiling."
            ),
            (
                "MoE routed weights account for EP and expert-TP sharding, but "
                "expert routing workspace and token imbalance remain runtime-dependent."
            ),
            (
                "Training activation distinguishes no/selective/full recompute and "
                "pipeline stages, but non-core modules, VPP schedules, MLA, and kernel "
                "workspaces remain calibration-dependent."
            ),
            (
                "Log-prob uses a forward-only one-live-layer proxy and explicit "
                "last-stage vocabulary copies. Fused-kernel workspace and MoE rank "
                "imbalance are intentionally not assigned universal constants."
            ),
            (
                "Phase residency includes colocated actor/ref weights and active "
                "training optimizer state. LoRA trainable state is an analytical "
                "target-module approximation until per-rank trainable numel is logged."
            ),
            (
                "Rollout is driven by vLLM gpu_memory_utilization; scheduler, cache, "
                "and rollout-topology changes without matched trials are uncalibrated."
            ),
            "A short resource-gate trial remains the final OOM authority.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate verl phase memory from a measured reference and changed "
            "analytical components"
        )
    )
    parser.add_argument("--current", required=True, help="Reference parameter JSON")
    parser.add_argument(
        "--candidate", required=True, help="Fully assembled candidate parameter JSON"
    )
    parser.add_argument("--trials", help="Optional JSON array or JSONL trial history")
    parser.add_argument("--memory-limit-pct", type=float, default=92.0)
    parser.add_argument("--reference-trial-id", type=int)
    args = parser.parse_args()

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    if args.trials:
        text = Path(args.trials).read_text(encoding="utf-8").strip()
        trials = (
            json.loads(text)
            if text.startswith("[")
            else [json.loads(line) for line in text.splitlines() if line]
        )
    print(
        json.dumps(
            estimate_phase_memory(
                current,
                candidate,
                trials,
                args.memory_limit_pct,
                args.reference_trial_id,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
