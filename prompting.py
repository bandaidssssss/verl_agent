from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def json_block(value: Any) -> str:
    if value in (None, {}, []):
        return "未提供"
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
        return "暂无历史 trial。"
    header = (
        "|Trial|阶段|结果|修改|吞吐|Step(s)|显存瓶颈|峰值显存%|失败类型|\n"
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
        return "当前角色没有可调用工具。"
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
    return re.sub(r"\{[A-Z][A-Z0-9_]*\}", "未提供", rendered)


def rejection_feedback(
    attempt: int,
    proposal: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source: str,
    result: Mapping[str, Any],
) -> str:
    return (
        f"## 第 {attempt} 次建议被拒绝\n\n"
        f"- 拒绝来源：`{source}`\n\n"
        f"- Proposed changes:\n{json_block(proposal)}\n\n"
        f"- The modified parameters are:\n{json_block(candidate)}\n\n"
        f"- Validation result:\n{json_block(result)}\n\n"
        "请把这次失败及其原因作为后续推理证据。如果字段不在当前阶段 editable 白名单，必须放弃该字段并选择白名单内参数；"
        "如果字段可编辑但参考 trial 未显式配置，使用 from:null 表示新增 override。你可以继续调用工具核查参数、显存或 verl 文档，"
        "但完成工具调用后必须输出完整 Proposal，不能把工具参数当作最终答案。最终仍只输出约定的 JSON 对象。"
    )
