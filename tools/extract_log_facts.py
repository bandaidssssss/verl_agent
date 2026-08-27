from __future__ import annotations

import ast
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TASK_RUNNER_PREFIX_RE = re.compile(r"^.*?\(TaskRunner pid=\d+\)\s*")
RESOLVED_CONFIG_START = "{'actor_rollout_ref':"
MAX_RESOLVED_CONFIG_LINES = 5000
MAX_RESOLVED_CONFIG_CHARS = 2_000_000
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

PROMPT_LENGTH_KEY = "data.max_prompt_length"
RESPONSE_LENGTH_KEY = "data.max_response_length"
TRAIN_BATCH_KEY = "data.train_batch_size"
ROLLOUT_N_KEY = "actor_rollout_ref.rollout.n"

RANK_PATTERNS = {
    "global_rank": (
        re.compile(r"\bglobal[_ ]rank\s*[=:]\s*(\d+)", re.I),
        re.compile(r"\[rank\s*[:=]?\s*(\d+)\]", re.I),
    ),
    "data_parallel_rank": (
        re.compile(r"\b(?:data|dp)[_ ]parallel[_ ]rank\s*[=:]\s*(\d+)", re.I),
        re.compile(r"\bdp[_ ]rank\s*[=:]\s*(\d+)", re.I),
    ),
    "expert_parallel_rank": (
        re.compile(r"\b(?:expert|ep)[_ ]parallel[_ ]rank\s*[=:]\s*(\d+)", re.I),
        re.compile(r"\bep[_ ]rank\s*[=:]\s*(\d+)", re.I),
    ),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(parameters: Mapping[str, Any], key: str, default: float) -> float:
    value = parameters.get(key)
    return float(value) if _is_number(value) else default


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().rstrip(",")
    lowered = value.lower()
    if lowered in {"none", "null"}:
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


def _flatten_mapping(
    value: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, key))
        else:
            flattened[key] = item
    return flattened


def _resolved_runtime_parameters(
    lines: Sequence[str], warnings: list[str]
) -> dict[str, Any]:
    if not lines:
        warnings.append("resolved Hydra configuration was not found in train.log")
        return {
            "available": False,
            "source": "train.log:resolved_hydra_config",
            "values": {},
        }
    try:
        config = ast.literal_eval("\n".join(lines))
    except (SyntaxError, ValueError) as exc:
        warnings.append(f"resolved Hydra configuration could not be parsed: {exc}")
        return {
            "available": False,
            "source": "train.log:resolved_hydra_config",
            "values": {},
        }
    if not isinstance(config, Mapping):
        warnings.append("resolved Hydra configuration was not a mapping")
        return {
            "available": False,
            "source": "train.log:resolved_hydra_config",
            "values": {},
        }
    return {
        "available": True,
        "source": "train.log:resolved_hydra_config",
        "values": _flatten_mapping(config),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _configured_length_profile(parameters: Mapping[str, Any]) -> dict[str, Any]:
    total = max(
        1.0,
        _number(parameters, PROMPT_LENGTH_KEY, 1024.0)
        + _number(parameters, RESPONSE_LENGTH_KEY, 4096.0),
    )
    return {
        "point_tokens": total,
        "upper_tokens": total,
        "configured_upper_tokens": total,
        "source": "configured_maximum",
        "sampled_steps": 0,
    }


def _length_profile(
    records: Mapping[int, Mapping[str, float]], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    configured = _configured_length_profile(parameters)
    selected = [row for step, row in sorted(records.items()) if step > 5]
    if not selected:
        selected = [row for _, row in sorted(records.items())]
    population = _number(parameters, TRAIN_BATCH_KEY, 0.0) * _number(
        parameters, ROLLOUT_N_KEY, 1.0
    )
    means: list[float] = []
    observed_upper: list[float] = []
    for row in selected:
        total_tokens = row.get("perf/total_num_tokens")
        if total_tokens is not None and population > 0:
            means.append(float(total_tokens) / population)
        elif (
            row.get("prompt_length/mean") is not None
            and row.get("response_length/mean") is not None
        ):
            means.append(
                float(row["prompt_length/mean"])
                + float(row["response_length/mean"])
            )
        if (
            row.get("prompt_length/max") is not None
            and row.get("response_length/max") is not None
        ):
            observed_upper.append(
                float(row["prompt_length/max"])
                + float(row["response_length/max"])
            )
    point = _percentile(means, 0.95)
    if point is None:
        return configured
    configured_upper = float(configured["configured_upper_tokens"])
    observed = _percentile(observed_upper, 0.95)
    upper = (
        configured_upper
        if observed is None
        else min(configured_upper, max(point, observed))
    )
    return {
        "point_tokens": max(1.0, min(point, configured_upper)),
        "upper_tokens": max(1.0, upper),
        "configured_upper_tokens": configured_upper,
        "source": "train_log_stable_step_p95_mean_tokens",
        "sampled_steps": len(means),
        "mean_tokens_across_sampled_steps": (
            statistics.mean(means) if means else None
        ),
    }


def _rank_metadata(line: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, patterns in RANK_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                result[name] = int(match.group(1))
                break
    return result


def _rank_facts(
    lines: Sequence[str], resolved: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    unique: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {}
    matched_lines = 0
    warnings: list[str] = []
    for line in lines:
        match = PARAMETER_COUNT_RE.search(line)
        if not match:
            continue
        matched_lines += 1
        coordinates = {
            **_rank_metadata(line),
            "tensor_rank": int(match.group("tp_rank")),
            "pipeline_rank": int(match.group("pp_rank")),
        }
        identity = tuple(sorted(coordinates.items()))
        count = int(match.group("count"))
        existing = unique.get(identity)
        if existing is None:
            unique[identity] = {
                **coordinates,
                "parameters": count,
                "occurrences": 1,
            }
            continue
        existing["occurrences"] = int(existing["occurrences"]) + 1
        if count != existing["parameters"]:
            values = set(existing.get("conflicting_values", []))
            values.update((int(existing["parameters"]), count))
            existing["conflicting_values"] = sorted(values)

    rows = [unique[key] for key in sorted(unique)]
    conflicts = [row for row in rows if row.get("conflicting_values")]
    if conflicts:
        warnings.append(
            "conflicting Megatron parameter counts were logged for the same rank identity"
        )
    if not rows:
        return [], {}, ["Megatron per-rank parameter counts were not found in train.log"]

    # Parameter totals are reconstructed from unique TP/PP shards. Global and
    # data-parallel rank metadata remain in rank_parameter_counts for audit but
    # must not multiply replicated model shards.
    shard_values: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        shard = (int(row["tensor_rank"]), int(row["pipeline_rank"]))
        values = shard_values.setdefault(shard, set())
        values.add(int(row["parameters"]))
        values.update(int(value) for value in row.get("conflicting_values", []))
    shard_conflicts = {
        shard: sorted(values) for shard, values in shard_values.items() if len(values) > 1
    }
    if shard_conflicts and not conflicts:
        warnings.append(
            "conflicting Megatron parameter counts were logged for one or more TP/PP shards"
        )

    tp = int(resolved.get("tensor_model_parallel_size", 1) or 1)
    pp = int(resolved.get("pipeline_model_parallel_size", 1) or 1)
    ep = int(resolved.get("expert_model_parallel_size", 1) or 1)
    etp = int(resolved.get("expert_tensor_parallel_size", tp) or tp)
    expected = max(1, tp * pp)
    selected = {
        shard: next(iter(values))
        for shard, values in shard_values.items()
        if shard[0] < tp and shard[1] < pp and len(values) == 1
    }
    complete = len(selected) == expected and not shard_conflicts
    most_loaded = max(selected.values()) if selected else None
    total = sum(selected.values()) if complete and ep == 1 else None
    total_source = "sum_unique_logged_tp_pp_shards" if total is not None else None
    summary = {
        "source": "train.log:number_of_parameters",
        "matched_log_lines": matched_lines,
        "unique_rank_count": len(rows),
        "observed_tp_pp_shard_count": len(selected),
        "expected_tp_pp_shard_count": expected,
        "complete_tp_pp_coverage": complete,
        "most_loaded_shard_parameters": most_loaded,
        "total_parameters": total,
        "total_parameters_source": total_source,
        "reference_topology": {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
    }
    if shard_conflicts:
        summary["conflicting_tp_pp_shards"] = [
            {
                "tensor_rank": shard[0],
                "pipeline_rank": shard[1],
                "values": values,
            }
            for shard, values in sorted(shard_conflicts.items())
        ]
    return rows, summary, warnings


def build_log_facts(
    fact_lines: Sequence[str],
    records: Mapping[int, Mapping[str, float]],
    parameters: Mapping[str, Any],
    log_path: str | Path,
    resolved_config_lines: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the estimator-neutral log-facts artifact from one parser pass."""
    text = "\n".join(fact_lines)
    warnings: list[str] = []
    model_config: dict[str, Any] = {}
    for field in MODEL_CONFIG_FIELDS:
        match = re.search(
            rf'["\']{re.escape(field)}["\']\s*:\s*'
            rf'(?P<value>true|false|null|None|"[^"]*"|\'[^\']*\'|{NUMBER})',
            text,
            re.IGNORECASE,
        )
        if match:
            model_config[field] = _parse_scalar(match.group("value"))
    if not model_config:
        warnings.append("printed model configuration was not found in train.log")

    transformer = next(
        (line for line in fact_lines if "TransformerConfig(" in line), ""
    )
    resolved: dict[str, Any] = {}
    for field in RESOLVED_CONFIG_FIELDS:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(field)}=(?P<value>[^,\)]+)",
            transformer,
        )
        if match:
            resolved[field] = _parse_scalar(match.group("value"))
    if not resolved:
        warnings.append("Megatron TransformerConfig was not found in train.log")

    ranks, parameter_summary, rank_warnings = _rank_facts(fact_lines, resolved)
    warnings.extend(rank_warnings)
    path = Path(log_path)
    return {
        "source": {
            "train_log": path.name,
            "warnings": warnings,
        },
        "model_config": model_config,
        "runtime_parameters": _resolved_runtime_parameters(
            resolved_config_lines, warnings
        ),
        "megatron": {
            "resolved_config": resolved,
            "rank_parameter_counts": ranks,
            "parameter_summary": parameter_summary,
        },
        "workload": {
            "sequence_length": _length_profile(records, parameters),
        },
    }


class LogFactsAccumulator:
    """Collect trial facts while the unified metrics parser reads train.log."""

    def __init__(self, parameters: Mapping[str, Any], log_path: str | Path) -> None:
        self.parameters = dict(parameters)
        self.log_path = Path(log_path)
        self._fact_lines: list[str] = []
        self._resolved_config_lines: list[str] = []
        self._resolved_config_started = False
        self._resolved_config_complete = False
        self._resolved_config_chars = 0
        self._resolved_config_depth = 0
        self._resolved_config_quote: str | None = None
        self._resolved_config_escaped = False

    def _update_resolved_config_depth(self, line: str) -> None:
        for character in line:
            if self._resolved_config_escaped:
                self._resolved_config_escaped = False
                continue
            if character == "\\" and self._resolved_config_quote is not None:
                self._resolved_config_escaped = True
                continue
            if self._resolved_config_quote is not None:
                if character == self._resolved_config_quote:
                    self._resolved_config_quote = None
                continue
            if character in {"'", '"'}:
                self._resolved_config_quote = character
            elif character in "{[(":
                self._resolved_config_depth += 1
            elif character in "}])":
                self._resolved_config_depth -= 1

    def consume(self, line: str) -> None:
        clean = ANSI_RE.sub("", line)
        config_line = TASK_RUNNER_PREFIX_RE.sub("", clean).rstrip("\n")
        if not self._resolved_config_started and config_line.startswith(
            RESOLVED_CONFIG_START
        ):
            self._resolved_config_started = True
        if self._resolved_config_started and not self._resolved_config_complete:
            if (
                len(self._resolved_config_lines) < MAX_RESOLVED_CONFIG_LINES
                and self._resolved_config_chars + len(config_line)
                <= MAX_RESOLVED_CONFIG_CHARS
            ):
                self._resolved_config_lines.append(config_line)
                self._resolved_config_chars += len(config_line)
                self._update_resolved_config_depth(config_line)
                if self._resolved_config_depth == 0:
                    self._resolved_config_complete = True
            else:
                self._resolved_config_complete = True
        if (
            "TransformerConfig(" in clean
            or PARAMETER_COUNT_RE.search(clean)
            or any(field in clean for field in MODEL_CONFIG_FIELDS)
        ):
            self._fact_lines.append(clean.rstrip("\n"))

    def finalize(self, records: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
        return build_log_facts(
            self._fact_lines,
            records,
            self.parameters,
            self.log_path,
            self._resolved_config_lines,
        )
