from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def json_block(value: Any) -> str:
    if value in (None, {}, []):
        return "Not provided."
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"


def _metric(trial: Mapping[str, Any], *path: str) -> Any:
    value: Any = trial
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("mean")
    return value


def _cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text[:120] + ("..." if len(text) > 120 else "")


def trial_history_table(trials: Sequence[Mapping[str, Any]]) -> str:
    if not trials:
        return "No trial history is available."
    header = (
        "|Trial|Stage|Result|Changes|Throughput|Step(s)|Memory bottleneck|Peak memory %|Failure type|\n"
        "|---:|---|---|---|---:|---:|---|---:|---|"
    )
    rows = [header]
    for trial in trials:
        changes = trial.get("proposal", {}).get("changes") if isinstance(trial.get("proposal"), Mapping) else None
        rendered_changes = []
        for key, value in (changes or {}).items():
            if isinstance(value, Mapping) and "from" in value and "to" in value:
                rendered_changes.append(f"{key}: {value['from']}→{value['to']}")
            else:
                rendered_changes.append(f"{key}={value}")
        change_text = ", ".join(rendered_changes) or "baseline/keep"
        rows.append(
            "|".join(
                [
                    "",
                    _cell(trial.get("trial_id")),
                    _cell(trial.get("stage")),
                    _cell(trial.get("result")),
                    _cell(change_text),
                    _cell(_metric(trial, "performance", "throughput")),
                    _cell(_metric(trial, "performance", "time_per_step_s")),
                    _cell(_metric(trial, "resource", "memory_bottleneck")),
                    _cell(_metric(trial, "resource", "max_observed_memory_pct"), 1),
                    _cell(_metric(trial, "error", "type")),
                    "",
                ]
            )
        )
    return "\n".join(rows)


def available_tools_markdown(tool_definitions: Sequence[Mapping[str, Any]]) -> str:
    if not tool_definitions:
        return "No tools are available to this role."
    lines = []
    for tool in tool_definitions:
        lines.append(f"- `{tool['name']}`：{tool['description']}")
    return "\n".join(lines)


def render_prompt(
    template: str,
    context: Mapping[str, Any],
    tool_definitions: Sequence[Mapping[str, Any]],
) -> str:
    replacements = {
        "CURRENT_STAGE": f"`{context.get('current_stage', 'unknown')}`",
        "MODE": f"`{context.get('mode', 'unknown')}`",
        "CURRENT_PARAMETERS": json_block(context.get("current_parameters")),
        "REFERENCE_TRIAL": json_block(context.get("reference_trial")),
        "REFERENCE_STABILITY_SERIES": json_block(context.get("reference_stability_series")),
        "CANDIDATE_PARAMETERS": json_block(context.get("candidate_parameters")),
        "CHANGES": json_block(context.get("changes")),
        "TARGET_CHANGES": json_block(context.get("target_changes")),
        "EDITABLE_PARAMETERS": json_block(context.get("editable_parameters")),
        "CONSTRAINTS": json_block(context.get("constraints")),
        "DIAGNOSIS": json_block(context.get("diagnosis")),
        "HEALTH_EVENT": json_block(context.get("health_event")),
        "TRIAL": json_block(context.get("trial")),
        "LAST_TRIAL": json_block(context.get("last_trial")),
        "PROPOSAL_REASON": _cell(context.get("proposal_reason")),
        "MEMORY_LIMITS": json_block(context.get("memory_limits")),
        "TRIAL_HISTORY": trial_history_table(context.get("recent_trials", [])),
        "AVAILABLE_TOOLS": available_tools_markdown(tool_definitions),
    }
    rendered = template
    for name, value in replacements.items():
        rendered = rendered.replace("{" + name + "}", value)
    return re.sub(r"\{[A-Z][A-Z0-9_]*\}", "Not provided.", rendered)


def rejection_feedback(
    attempt: int,
    proposal: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source: str,
    result: Mapping[str, Any],
) -> str:
    return (
        f"## Proposal Attempt {attempt} Was Rejected\n\n"
        f"- Rejection source: `{source}`\n\n"
        f"- Proposed changes:\n{json_block(proposal)}\n\n"
        f"- The modified parameters are:\n{json_block(candidate)}\n\n"
        f"- Validation result:\n{json_block(result)}\n\n"
        "Treat this rejection and its reason as evidence for the next proposal. If a field is not "
        "in the editable whitelist for the current stage, abandon that field and choose an allowed "
        "parameter. If a field is editable but absent from the reference trial's explicit "
        "configuration, use `from: null` to add an override. You may continue using tools to verify "
        "parameter semantics, memory, or verl documentation, but after tool use you must return a "
        "complete Proposal rather than tool arguments. The final response must still be exactly the "
        "required JSON object."
    )
