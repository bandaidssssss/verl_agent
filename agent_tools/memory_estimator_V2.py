from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")

MODEL_KEY = "actor_rollout_ref.model.path"
PROMPT_LENGTH_KEY = "data.max_prompt_length"
RESPONSE_LENGTH_KEY = "data.max_response_length"
ROLLOUT_UTILIZATION_KEY = "actor_rollout_ref.rollout.gpu_memory_utilization"

ACTOR_MICRO_KEY = "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
REF_MICRO_KEY = "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
TRAINING_MICRO_KEY = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"

ACTOR_TP_KEY = "actor_rollout_ref.actor.megatron.tensor_model_parallel_size"
ACTOR_PP_KEY = "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size"
REF_TP_KEY = "actor_rollout_ref.ref.megatron.tensor_model_parallel_size"
REF_PP_KEY = "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size"

ACTOR_SP_KEY = "actor_rollout_ref.actor.megatron.sequence_parallel"
REF_SP_KEY = "actor_rollout_ref.ref.megatron.sequence_parallel"
ACTOR_REMOVE_PADDING_KEY = (
    "actor_rollout_ref.actor.megatron.use_remove_padding"
)
ACTOR_DYNAMIC_BATCH_KEY = "actor_rollout_ref.actor.use_dynamic_bsz"
RECOMPUTE_GRANULARITY_KEY = (
    "actor_rollout_ref.actor.megatron."
    "override_transformer_config.recompute_granularity"
)


def _number(parameters: Mapping[str, Any], key: str, default: float) -> float:
    value = parameters.get(key)
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{key} must be numeric or null, got {type(value).__name__}")


def _changed_keys(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> set[str]:
    keys = set(reference) | set(candidate)
    return {key for key in keys if reference.get(key) != candidate.get(key)}


def _sequence_length(parameters: Mapping[str, Any]) -> float:
    prompt = _number(parameters, PROMPT_LENGTH_KEY, 1024)
    response = _number(parameters, RESPONSE_LENGTH_KEY, 4096)
    return max(1.0, prompt + response)


def _recompute_factor(parameters: Mapping[str, Any]) -> float:
    value = parameters.get(RECOMPUTE_GRANULARITY_KEY)
    if value in (None, "none", "None", False):
        return 1.0
    # These are deliberately priors, not shares of total phase memory.  They
    # only scale the analytical activation term and are reported as
    # uncalibrated whenever the recompute regime changes.
    return 0.40 if str(value).lower() == "full" else 0.70


def _phase_peaks(trial: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
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
        if _phase_peaks(trial)
        and isinstance(trial.get("parameters"), Mapping)
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
    exact = [
        trial
        for trial in observed
        if _same_parameters(trial.get("parameters", {}), current)
    ]
    if exact:
        return exact[-1]
    same_model = [
        trial
        for trial in observed
        if trial.get("parameters", {}).get(MODEL_KEY) == current.get(MODEL_KEY)
    ]
    return same_model[-1] if same_model else None


def _read_model_config(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    model_path = parameters.get(MODEL_KEY)
    if not isinstance(model_path, str) or not model_path:
        return {}
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return {}
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _shape_number(
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
    names: Sequence[str],
    default: float,
) -> tuple[float, str]:
    for name in names:
        value = parameters.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), f"parameters:{name}"
    for name in names:
        short_name = name.rsplit(".", 1)[-1]
        value = config.get(short_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), f"model_config:{short_name}"
    return default, "decoder_default"


def _model_shape(parameters: Mapping[str, Any]) -> dict[str, Any]:
    config = _read_model_config(parameters)
    hidden, hidden_source = _shape_number(
        parameters,
        config,
        ("hidden_size", "model.hidden_size", "actor_rollout_ref.model.hidden_size"),
        4096.0,
    )
    layers, layers_source = _shape_number(
        parameters,
        config,
        (
            "num_hidden_layers",
            "num_layers",
            "model.num_hidden_layers",
            "actor_rollout_ref.model.num_hidden_layers",
        ),
        32.0,
    )
    heads, heads_source = _shape_number(
        parameters,
        config,
        (
            "num_attention_heads",
            "model.num_attention_heads",
            "actor_rollout_ref.model.num_attention_heads",
        ),
        max(1.0, hidden / 128.0),
    )
    return {
        "hidden_size": max(1.0, hidden),
        "num_layers": max(1.0, layers),
        "num_attention_heads": max(1.0, heads),
        "sources": {
            "hidden_size": hidden_source,
            "num_layers": layers_source,
            "num_attention_heads": heads_source,
        },
    }


def _phase_keys(phase: str) -> tuple[str, str, str]:
    if phase == "actor_log_prob":
        return ACTOR_MICRO_KEY, ACTOR_TP_KEY, ACTOR_PP_KEY
    if phase == "ref_log_prob":
        return REF_MICRO_KEY, REF_TP_KEY, REF_PP_KEY
    return TRAINING_MICRO_KEY, ACTOR_TP_KEY, ACTOR_PP_KEY


def _dynamic_activation_bytes(
    phase: str, parameters: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Return the Frenzy activation prior for a compute phase.

    The value is useful both as an absolute prior when model metadata exists
    and, more importantly here, as a relative dynamic-memory pressure.  The
    estimator calibrates its observed percentage-point slope from matched
    trials, so backend-specific constant factors cancel.
    """

    micro_key, tp_key, _ = _phase_keys(phase)
    batch = _number(parameters, micro_key, 1.0)
    tensor_parallel = max(1.0, _number(parameters, tp_key, 1.0))
    sequence = _sequence_length(parameters)
    shape = _model_shape(parameters)
    hidden = float(shape["hidden_size"])
    layers = float(shape["num_layers"])
    heads = float(shape["num_attention_heads"])

    # Frenzy: s*b*h*l*(10 + 24/t + 5*a*s/(h*t)).
    bracket = (
        10.0
        + 24.0 / tensor_parallel
        + 5.0 * heads * sequence / (hidden * tensor_parallel)
    )
    value = sequence * batch * hidden * layers * bracket
    if phase == "training":
        value *= _recompute_factor(parameters)

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid analytical activation value for {phase}")
    return value, {
        "micro_batch_key": micro_key,
        "micro_batch_size": batch,
        "sequence_length": sequence,
        "tensor_parallel_size": tensor_parallel,
        "hidden_size": hidden,
        "num_layers": layers,
        "num_attention_heads": heads,
        "recompute_factor": (
            _recompute_factor(parameters) if phase == "training" else 1.0
        ),
        "shape_sources": shape["sources"],
        "formula": "s*b*h*l*(10 + 24/t + 5*a*s/(h*t))",
    }


def _same_values(
    left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str]
) -> bool:
    return all(left.get(key) == right.get(key) for key in keys)


def _calibration_regime_keys(phase: str) -> tuple[str, ...]:
    if phase == "actor_log_prob":
        return (
            MODEL_KEY,
            ROLLOUT_UTILIZATION_KEY,
            ACTOR_TP_KEY,
            ACTOR_PP_KEY,
            ACTOR_SP_KEY,
            ACTOR_REMOVE_PADDING_KEY,
            ACTOR_DYNAMIC_BATCH_KEY,
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.rollout.free_cache_engine",
        )
    if phase == "ref_log_prob":
        return (
            MODEL_KEY,
            ROLLOUT_UTILIZATION_KEY,
            REF_TP_KEY,
            REF_PP_KEY,
            REF_SP_KEY,
            "actor_rollout_ref.ref.megatron.param_offload",
            "actor_rollout_ref.rollout.free_cache_engine",
        )
    return (
        MODEL_KEY,
        ROLLOUT_UTILIZATION_KEY,
        ACTOR_TP_KEY,
        ACTOR_PP_KEY,
        ACTOR_SP_KEY,
        ACTOR_REMOVE_PADDING_KEY,
        ACTOR_DYNAMIC_BATCH_KEY,
        "actor_rollout_ref.actor.megatron.param_offload",
        "actor_rollout_ref.actor.megatron.optimizer_offload",
        "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
        RECOMPUTE_GRANULARITY_KEY,
        "actor_rollout_ref.rollout.free_cache_engine",
    )


def _usable_calibration_trial(trial: Mapping[str, Any], phase: str) -> bool:
    # Failed resource gates often contain partial phase labels after the job
    # stopped elsewhere.  They remain valid as an explicitly selected anchor,
    # but are too noisy to train a slope automatically.
    if trial.get("result") not in (None, "success", "early_stopped"):
        return False
    return phase in _phase_peaks(trial)


def _matched_dynamic_observations(
    phase: str,
    reference_parameters: Mapping[str, Any],
    reference_dynamic: float,
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int | None]]:
    observations: list[dict[str, float | int | None]] = []
    regime_keys = _calibration_regime_keys(phase)
    for trial in trials:
        if not isinstance(trial, Mapping) or not _usable_calibration_trial(
            trial, phase
        ):
            continue
        parameters = trial.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        if not _same_values(reference_parameters, parameters, regime_keys):
            continue
        dynamic, _ = _dynamic_activation_bytes(phase, parameters)
        peak = _phase_peaks(trial)[phase]
        observations.append(
            {
                "trial_id": trial.get("trial_id"),
                "dynamic_ratio_to_reference": dynamic / reference_dynamic,
                "observed_pct": peak,
            }
        )
    return observations


def _robust_dynamic_slope(
    observations: Sequence[Mapping[str, float | int | None]],
) -> tuple[float | None, float]:
    slopes: list[float] = []
    for index, left in enumerate(observations):
        x_left = float(left["dynamic_ratio_to_reference"])
        y_left = float(left["observed_pct"])
        for right in observations[index + 1 :]:
            x_right = float(right["dynamic_ratio_to_reference"])
            y_right = float(right["observed_pct"])
            dx = x_right - x_left
            if abs(dx) < 1e-9:
                continue
            slope = (y_right - y_left) / dx
            if math.isfinite(slope) and slope >= 0:
                slopes.append(slope)
    if not slopes:
        return None, 0.0

    slope = statistics.median(slopes)
    intercepts = [
        float(item["observed_pct"])
        - slope * float(item["dynamic_ratio_to_reference"])
        for item in observations
    ]
    intercept = statistics.median(intercepts)
    residuals = [
        abs(
            float(item["observed_pct"])
            - (
                intercept
                + slope * float(item["dynamic_ratio_to_reference"])
            )
        )
        for item in observations
    ]
    return slope, max(residuals, default=0.0)


def _compute_relevant_keys(phase: str) -> set[str]:
    micro_key, tp_key, pp_key = _phase_keys(phase)
    keys = {
        MODEL_KEY,
        micro_key,
        tp_key,
        pp_key,
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
        "actor_rollout_ref.rollout.free_cache_engine",
    }
    if phase == "actor_log_prob":
        keys |= {
            ACTOR_SP_KEY,
            ACTOR_REMOVE_PADDING_KEY,
            ACTOR_DYNAMIC_BATCH_KEY,
            "actor_rollout_ref.actor.megatron.param_offload",
        }
    elif phase == "ref_log_prob":
        keys |= {
            REF_SP_KEY,
            "actor_rollout_ref.ref.megatron.param_offload",
        }
    else:
        keys |= {
            ACTOR_SP_KEY,
            ACTOR_REMOVE_PADDING_KEY,
            ACTOR_DYNAMIC_BATCH_KEY,
            RECOMPUTE_GRANULARITY_KEY,
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.actor.megatron.optimizer_offload",
            "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_method",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_num_layers",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_modules",
        }
    return keys


def _compute_projection(
    phase: str,
    reference_peak: float,
    reference_parameters: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_dynamic, reference_inputs = _dynamic_activation_bytes(
        phase, reference_parameters
    )
    candidate_dynamic, candidate_inputs = _dynamic_activation_bytes(
        phase, candidate
    )
    dynamic_ratio = candidate_dynamic / reference_dynamic
    observations = _matched_dynamic_observations(
        phase, reference_parameters, reference_dynamic, trials
    )
    calibrated_slope, calibration_residual = _robust_dynamic_slope(
        observations
    )

    changed = _changed_keys(reference_parameters, candidate)
    relevant_changes = changed & _compute_relevant_keys(phase)
    micro_key, tp_key, pp_key = _phase_keys(phase)
    analytical_dynamic_keys = {
        micro_key,
        tp_key,
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
    }
    if phase == "training":
        analytical_dynamic_keys.add(RECOMPUTE_GRANULARITY_KEY)

    # TP also changes static states; recompute policy details and PP/offload
    # changes cannot be valued by an activation-only formula.  We still
    # calculate the known dynamic delta, but expose the missing static delta.
    static_or_regime_keys = {
        MODEL_KEY,
        tp_key,
        pp_key,
        "actor_rollout_ref.rollout.free_cache_engine",
    }
    if phase == "actor_log_prob":
        static_or_regime_keys |= {
            ACTOR_SP_KEY,
            ACTOR_REMOVE_PADDING_KEY,
            ACTOR_DYNAMIC_BATCH_KEY,
            "actor_rollout_ref.actor.megatron.param_offload",
        }
    elif phase == "ref_log_prob":
        static_or_regime_keys |= {
            REF_SP_KEY,
            "actor_rollout_ref.ref.megatron.param_offload",
        }
    else:
        static_or_regime_keys |= {
            ACTOR_SP_KEY,
            ACTOR_REMOVE_PADDING_KEY,
            ACTOR_DYNAMIC_BATCH_KEY,
            "actor_rollout_ref.actor.megatron.param_offload",
            "actor_rollout_ref.actor.megatron.optimizer_offload",
            "actor_rollout_ref.actor.megatron.use_distributed_optimizer",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_method",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_num_layers",
            "actor_rollout_ref.actor.megatron."
            "override_transformer_config.recompute_modules",
        }
        if RECOMPUTE_GRANULARITY_KEY in changed:
            static_or_regime_keys.add(RECOMPUTE_GRANULARITY_KEY)

    uncalibrated = sorted(
        (relevant_changes - analytical_dynamic_keys)
        | (relevant_changes & static_or_regime_keys)
    )

    if calibrated_slope is not None:
        delta = calibrated_slope * (dynamic_ratio - 1.0)
        calibration = "matched_trial_pairwise_median_slope"
        confidence = "medium" if not uncalibrated else "low"
        base_uncertainty = max(2.0, calibration_residual + 1.0)
    elif math.isclose(dynamic_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        delta = 0.0
        calibration = "unchanged_dynamic_pressure"
        confidence = "medium" if not uncalibrated else "low"
        base_uncertainty = 2.0
    elif dynamic_ratio > 1.0:
        # With one anchor there is no defensible way to separate the static
        # floor from dynamic memory.  Treating the full observed peak as the
        # dynamic scale is intentionally conservative for increases and, in
        # particular, prevents large micro-batch jumps from being declared
        # safe on an arbitrary 20% activation-share assumption.
        delta = reference_peak * (dynamic_ratio - 1.0)
        calibration = "single_anchor_conservative_increase"
        confidence = "low"
        base_uncertainty = max(4.0, min(12.0, abs(delta) * 0.10))
    else:
        # Do not promise memory savings without a second observation that can
        # identify the dynamic slope.  The resource gate may later provide the
        # calibration point needed to value the decrease.
        delta = 0.0
        calibration = "single_anchor_no_uncalibrated_savings"
        confidence = "low"
        base_uncertainty = 5.0

    # A changed static/residency regime can move tens of percentage points.
    # V2 intentionally declines to price that move until an absolute static
    # model is added; a wide guard band prevents the analytical activation
    # delta from presenting false precision in the meantime.
    regime_changes = sorted(relevant_changes & static_or_regime_keys)
    uncertainty = min(
        35.0,
        base_uncertainty
        + 3.0 * len(set(uncalibrated) - set(regime_changes))
        + 10.0 * len(regime_changes),
    )
    projected = max(0.0, reference_peak + delta)
    ratio = projected / reference_peak if reference_peak > 0 else 1.0
    return {
        "projected_pct": projected,
        "delta_pct": delta,
        "ratio": ratio,
        "model": "empirical_anchor_plus_frenzy_dynamic_delta",
        "drivers": {
            "dynamic_formula": reference_inputs["formula"],
            "dynamic_ratio": dynamic_ratio,
            "reference_dynamic_value": reference_dynamic,
            "candidate_dynamic_value": candidate_dynamic,
            "reference_inputs": reference_inputs,
            "candidate_inputs": candidate_inputs,
            "calibration": calibration,
            "calibrated_slope_pct_per_dynamic_ratio": calibrated_slope,
            "calibration_observations": observations,
            "calibration_residual_pct": calibration_residual,
            "static_or_residency_regime_changes": regime_changes,
        },
        "uncalibrated_changes": uncalibrated,
        "uncertainty_pct": uncertainty,
        "confidence": confidence,
    }


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
                "observed_pct": _phase_peaks(trial)["rollout"],
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
        # A one-point increase in gpu_memory_utilization corresponds to one
        # percentage point of physical GPU capacity before calibration.
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
                - (
                    intercept
                    + slope
                    * 100.0
                    * float(item["gpu_memory_utilization"])
                )
            )
            for item in observations
        ),
        default=0.0,
    )
    return slope, residual, True


def _rollout_projection(
    reference_peak: float,
    reference_parameters: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_util = _number(
        reference_parameters, ROLLOUT_UTILIZATION_KEY, 0.6
    )
    candidate_util = _number(candidate, ROLLOUT_UTILIZATION_KEY, 0.6)
    if not 0.0 < reference_util <= 1.0 or not 0.0 < candidate_util <= 1.0:
        raise ValueError(
            f"{ROLLOUT_UTILIZATION_KEY} must be in the interval (0, 1]"
        )

    observations = _rollout_observations(reference_parameters, trials)
    slope, residual, calibrated = _rollout_utilization_slope(observations)
    delta = slope * 100.0 * (candidate_util - reference_util)
    projected = max(0.0, reference_peak + delta)

    relevant = {
        MODEL_KEY,
        ROLLOUT_UTILIZATION_KEY,
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        "actor_rollout_ref.rollout.max_num_seqs",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.enable_prefix_caching",
        "actor_rollout_ref.rollout.enable_chunked_prefill",
        "actor_rollout_ref.rollout.free_cache_engine",
        "actor_rollout_ref.rollout.enforce_eager",
        PROMPT_LENGTH_KEY,
        RESPONSE_LENGTH_KEY,
    }
    changed = _changed_keys(reference_parameters, candidate) & relevant
    uncalibrated = sorted(changed - {ROLLOUT_UTILIZATION_KEY})
    uncertainty = min(
        20.0,
        max(2.0 if calibrated else 3.0, residual + 1.0)
        + 2.5 * len(uncalibrated),
    )
    ratio = projected / reference_peak if reference_peak > 0 else 1.0
    return {
        "projected_pct": projected,
        "delta_pct": delta,
        "ratio": ratio,
        "model": "vllm_utilization_absolute_delta",
        "drivers": {
            "gpu_memory_utilization": {
                "from": reference_util,
                "to": candidate_util,
                "delta_gpu_capacity_pct": 100.0
                * (candidate_util - reference_util),
                "calibrated_slope": slope,
                "projected_delta_pct": delta,
            },
            "calibration": (
                "matched_trial_pairwise_median_slope"
                if calibrated
                else "one_capacity_pct_per_utilization_pct_prior"
            ),
            "calibration_observations": observations,
            "calibration_residual_pct": residual,
        },
        "uncalibrated_changes": uncalibrated,
        "uncertainty_pct": uncertainty,
        "confidence": (
            "low" if uncalibrated else ("high" if calibrated else "medium")
        ),
    }


def _phase_projection(
    phase: str,
    reference_peak: float,
    reference_parameters: Mapping[str, Any],
    candidate: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if phase == "rollout":
        return _rollout_projection(
            reference_peak, reference_parameters, candidate, trials
        )
    return _compute_projection(
        phase, reference_peak, reference_parameters, candidate, trials
    )


def estimate_phase_memory(
    current_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]] = (),
    memory_limit_pct: float = 92.0,
    reference_trial_id: int | None = None,
) -> dict[str, Any]:
    reference = _reference_trial(
        current_parameters, trials, reference_trial_id
    )
    reference_parameters: Mapping[str, Any] = current_parameters
    peaks: dict[str, float] = {}
    if reference:
        reference_parameters = reference.get(
            "parameters", current_parameters
        )
        peaks = _phase_peaks(reference)

    phases: dict[str, Any] = {}
    for phase in PHASES:
        reference_peak = peaks.get(phase)
        if reference_peak is None:
            phases[phase] = {
                "reference_pct": None,
                "pressure_ratio": None,
                "raw_pressure_ratio": None,
                "delta_pct": None,
                "projected_pct": None,
                "uncertainty_pct": None,
                "upper_bound_pct": None,
                "headroom_to_limit_pct": None,
                "upper_headroom_to_limit_pct": None,
                "risk": "unknown_without_observed_anchor",
                "confidence": "low",
                "model": "no_observed_anchor",
                "component_shares": None,
                "drivers": {},
                "uncalibrated_changes": [],
            }
            continue

        projection = _phase_projection(
            phase,
            reference_peak,
            reference_parameters,
            candidate_parameters,
            trials,
        )
        projected = float(projection["projected_pct"])
        ratio = float(projection["ratio"])
        uncertainty = float(projection["uncertainty_pct"])
        upper_bound = projected + uncertainty
        headroom = memory_limit_pct - projected
        upper_headroom = memory_limit_pct - upper_bound
        risk = (
            "high"
            if upper_bound >= memory_limit_pct
            else ("watch" if upper_headroom < 5.0 else "low")
        )
        phases[phase] = {
            "reference_pct": round(reference_peak, 2),
            "pressure_ratio": round(ratio, 3),
            "raw_pressure_ratio": round(ratio, 6),
            "delta_pct": round(float(projection["delta_pct"]), 2),
            "projected_pct": round(projected, 2),
            "uncertainty_pct": round(uncertainty, 2),
            "upper_bound_pct": round(upper_bound, 2),
            "headroom_to_limit_pct": round(headroom, 2),
            "upper_headroom_to_limit_pct": round(upper_headroom, 2),
            "risk": risk,
            "confidence": projection["confidence"],
            "model": projection["model"],
            # Retained for callers migrating from V1.  V2 intentionally does
            # not guess fixed component shares.
            "component_shares": None,
            "drivers": projection["drivers"],
            "uncalibrated_changes": projection["uncalibrated_changes"],
        }

    phase_confidence = [item["confidence"] for item in phases.values()]
    confidence = (
        "low"
        if "low" in phase_confidence
        else (
            "high"
            if phase_confidence
            and all(x == "high" for x in phase_confidence)
            else "medium"
        )
    )
    return {
        "method": (
            "empirical_anchor_plus_analytical_delta"
            if reference
            else "no_observed_anchor"
        ),
        "version": 2,
        "confidence": confidence,
        "memory_limit_pct": memory_limit_pct,
        "reference_trial_id": reference.get("trial_id") if reference else None,
        "phases": phases,
        "limitations": [
            (
                "Rollout point estimates are driven by the absolute change in "
                "vLLM gpu_memory_utilization; other rollout changes widen "
                "uncertainty."
            ),
            (
                "Compute phases add a Frenzy analytical activation delta to "
                "an observed phase peak instead of multiplying the full peak "
                "by fixed component shares."
            ),
            (
                "Matched same-regime trials calibrate the dynamic-memory "
                "slope; with one anchor, increases are conservative and "
                "uncalibrated decreases claim no savings."
            ),
            (
                "TP, PP, SP, offload, detailed recompute, model, and "
                "cache-engine changes can alter static or residency memory "
                "and remain uncalibrated without a matched regime."
            ),
            (
                "Configured maximum sequence lengths are workload bounds; "
                "actual token distributions and padding behavior can change "
                "realized activation memory."
            ),
            "A real short resource-gate trial remains the final memory authority.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate verl phase memory from an empirical anchor and "
            "analytical dynamic-memory deltas"
        )
    )
    parser.add_argument("--current", required=True, help="Current parameter JSON")
    parser.add_argument("--candidate", required=True, help="Candidate parameter JSON")
    parser.add_argument("--trials", help="Optional JSON array or JSONL trial history")
    parser.add_argument("--memory-limit-pct", type=float, default=92.0)
    parser.add_argument("--reference-trial-id", type=int)
    args = parser.parse_args()

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    trials = []
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
