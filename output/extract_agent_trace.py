#!/usr/bin/env python3
"""Generate the historical-style Agent Markdown report from schema-v2 trials.

The authoritative run layout is ``trials.jsonl`` plus per-trial artifacts under
``trials/NNNN``.  The index is intentionally small, so this script loads only
the artifacts needed for the human-readable report; it never reparses train.log.

Usage:
    python3 output/extract_agent_trace.py output/0817_0924_2026
    python3 output/extract_agent_trace.py output/0817_0924_2026 --output-name report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent / "0825_1133_2026"
DEFAULT_OUTPUT_NAME = "agent_report.md"
PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}
    return value if isinstance(value, dict) else {"_read_error": "JSON root is not an object"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"_read_error": f"line {line_no}: {exc}"})
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _artifact_path(run_dir: Path, index: Mapping[str, Any], name: str) -> Path:
    artifacts = index.get("artifacts")
    relative = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(relative, str):
        trial_id = index.get("trial_id")
        relative = f"trials/{int(trial_id):04d}/{name}.json" if isinstance(trial_id, int) else name
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        raise ValueError(f"artifact path escapes run directory: {relative}")
    return candidate


def _fmt(value: Any, spec: str = ".1f") -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    return "-"


def _markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _compact_json(value: Any, max_len: int = 130) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= max_len else text[:max_len] + "…"


def _tool_calls(record: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    calls = record.get("tool_calls")
    if isinstance(calls, list):
        return [call for call in calls if isinstance(call, dict)]
    return []


def _result(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    for key in ("result", "review", "decision"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return dict(record)


def _tool_table(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "_无工具调用_\n"
    lines = ["| 轮次 | 工具 | 参数 / 查询内容 | 状态 |", "|---|---|---|---|"]
    for call in calls:
        arguments = call.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        name = str(call.get("name", "?"))
        if name == "parameter_understanding":
            summary = ", ".join(map(str, arguments.get("items", [])))
        elif name == "memory_estimator":
            changes = arguments.get("changes")
            keys = ", ".join(changes) if isinstance(changes, Mapping) else ""
            summary = f"ref_trial={arguments.get('reference_trial_id', '?')}; {keys}"
        elif name in {"tuning_strategies", "query_trial_history", "read_trial_metrics"}:
            summary = _compact_json(arguments)
        elif name == "search_verl_docs":
            summary = f"查询: {arguments.get('query', '')}"
        else:
            summary = _compact_json(arguments)
        lines.append(
            f"| {call.get('tool_round', '?')} | `{name}` | {_markdown(summary)} | {_markdown(call.get('status', '?'))} |"
        )
    return "\n".join(lines) + "\n"


def _proposal_section(proposal: Mapping[str, Any]) -> str:
    if not proposal:
        return "_无 Proposal 决策记录_\n"
    lines = [
        "#### Proposal Agent 决策",
        "",
        f"- **决策**: `{proposal.get('decision', '?')}`",
        f"- **原因**: {proposal.get('reason') or '-'}",
        f"- **参考 Trial**: {proposal.get('reference_trial_id', '-')}",
    ]
    candidates = proposal.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            lines.extend(["", f"**候选 `{candidate.get('candidate_id', '?')}`**"])
            lines.append(f"- {candidate.get('reason') or '-'}")
            _append_changes(lines, candidate.get("changes"))
    else:
        _append_changes(lines, proposal.get("changes"))
    return "\n".join(lines) + "\n"


def _append_changes(lines: list[str], changes: Any) -> None:
    if not isinstance(changes, Mapping) or not changes:
        return
    lines.extend(["", "| 参数 | 旧值 | 新值 | 原因 |", "|---|---|---|---|"])
    for key, value in changes.items():
        if isinstance(value, Mapping):
            lines.append(
                f"| `{key}` | `{_markdown(value.get('from', '-'))}` | `{_markdown(value.get('to', '-'))}` | {_markdown(value.get('reason', ''))} |"
            )
        else:
            lines.append(f"| `{key}` | - | `{_markdown(value)}` | |")


def _diagnosis_section(diagnosis: Mapping[str, Any]) -> str:
    result = _result(diagnosis)
    lines = ["#### Diagnosis Agent 诊断", ""]
    calls = _tool_calls(diagnosis)
    if calls:
        lines.extend([f"**Diagnosis 工具调用 ({len(calls)} 次):**", "", _tool_table(calls)])
    lines.extend(
        [
            f"- **失败类型**: `{result.get('failure_type', result.get('type', '?'))}`",
            f"- **训练子阶段**: `{result.get('training_substage', result.get('failure_phase', '?'))}`",
            f"- **置信度**: {result.get('confidence', '-')}",
            f"- **原因**: {result.get('reason') or '-'}",
        ]
    )
    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        lines.append("- **证据**:")
        lines.extend(f"  - {_markdown(item)}" for item in evidence)
    return "\n".join(lines) + "\n"


def _feasibility_section(reviews: Any) -> str:
    if isinstance(reviews, Mapping):
        reviews = [reviews]
    if not isinstance(reviews, list) or not reviews:
        return ""
    lines = ["#### Feasibility Agent 审查", ""]
    for index, review in enumerate(reviews, start=1):
        if not isinstance(review, Mapping):
            continue
        result = _result(review)
        calls = _tool_calls(review)
        lines.extend([f"**审查 #{index}（attempt={review.get('attempt', index)}）**", ""])
        if calls:
            lines.extend([f"**Feasibility 工具调用 ({len(calls)} 次):**", "", _tool_table(calls)])
        lines.extend(
            [
                f"- **判定**: `{result.get('verdict', '?')}`",
                f"- **选中候选**: `{result.get('selected_candidate_id', '-')}`",
                f"- **原因**: {result.get('reason') or '-'}",
            ]
        )
        risks = result.get("risks")
        if isinstance(risks, list) and risks:
            lines.append("- **风险**:")
            lines.extend(f"  - {_markdown(risk)}" for risk in risks)
        lines.append("")
    return "\n".join(lines)


def _health_section(events_path: Path, traces_path: Path) -> str:
    events = _read_jsonl(events_path)
    decisions = [event for event in events if event.get("record_type") in {"agent_decision", "agent_error"}]
    if not decisions:
        return ""
    trace_count = len(_read_jsonl(traces_path))
    lines = ["### Trial 运行中的 Health Monitor 行为", ""]
    for index, decision in enumerate(decisions, start=1):
        lines.extend(
            [
                f"#### Health 决策 #{index}: `{decision.get('event_id', '?')}`",
                "",
                f"- **判定**: `{decision.get('verdict', '?')}`",
                f"- **动作**: `{decision.get('action', '?')}`",
                f"- **snapshot step**: {decision.get('snapshot_step', '-')}",
                f"- **置信度**: {decision.get('confidence', '-')}",
                f"- **原因**: {decision.get('reason') or '-'}",
            ]
        )
    if trace_count:
        lines.extend(["", f"_完整 Health Agent trace：{trace_count} 条，见 `{traces_path.name}`。_"])
    return "\n".join(lines) + "\n"


def _metrics_section(metrics: Mapping[str, Any]) -> str:
    throughput = metrics.get("throughput") if isinstance(metrics.get("throughput"), Mapping) else {}
    summary = throughput.get("summary") if isinstance(throughput.get("summary"), Mapping) else {}
    lines = ["### 关键指标", ""]
    if summary:
        lines.extend(["| 指标 | 均值 | P95 | 最大值 |", "|---|---:|---:|---:|"])
        for title, key, spec in (
            ("吞吐量 (tok/s)", "throughput", ".1f"),
            ("每步耗时 (s)", "time_per_step_s", ".1f"),
            ("生成 TGS", "generation_tgs", ".1f"),
            ("Actor TGS", "actor_tgs", ".1f"),
            ("Actor MFU", "actor_mfu", ".4f"),
        ):
            value = summary.get(key)
            if isinstance(value, Mapping):
                lines.append(f"| {title} | {_fmt(value.get('mean'), spec)} | {_fmt(value.get('p95'), spec)} | {_fmt(value.get('max'), spec)} |")
        if summary.get("time_bottleneck"):
            lines.append(f"| **时间瓶颈** | {_markdown(summary['time_bottleneck'])} | | |")
        lines.append("")

    resource = metrics.get("resource") if isinstance(metrics.get("resource"), Mapping) else {}
    by_phase = resource.get("by_phase") if isinstance(resource.get("by_phase"), Mapping) else {}
    utilization = resource.get("utilization_by_phase_pct") if isinstance(resource.get("utilization_by_phase_pct"), Mapping) else {}
    durations = throughput.get("phase_duration_s") if isinstance(throughput.get("phase_duration_s"), Mapping) else {}
    if any(isinstance(data, Mapping) for data in (by_phase, utilization, durations)):
        lines.extend(
            [
                "**分阶段耗时、显存与 GPU 利用率:**",
                "",
                "| 阶段 | 耗时均值 (s) | 显存均值 (MiB) | 显存 P95 (MiB) | 显存峰值 (MiB) | GPU 利用率均值 (%) | GPU 利用率 P95 (%) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for phase in PHASES:
            phase_memory = by_phase.get(phase) if isinstance(by_phase.get(phase), Mapping) else {}
            phase_duration = durations.get(phase) if isinstance(durations.get(phase), Mapping) else {}
            phase_util = utilization.get(phase) if isinstance(utilization.get(phase), Mapping) else {}
            if not (phase_memory or phase_duration or phase_util):
                continue
            lines.append(
                f"| {phase} | {_fmt(phase_duration.get('mean'))} | {_fmt(phase_memory.get('mean_used_mib'))} | {_fmt(phase_memory.get('p95_used_mib'))} | {_fmt(phase_memory.get('max_used_mib'))} | {_fmt(phase_util.get('mean'))} | {_fmt(phase_util.get('p95'))} |"
            )
        lines.append("")
    resource_summary = resource.get("summary") if isinstance(resource.get("summary"), Mapping) else {}
    if resource_summary:
        lines.extend(
            [
                f"- **最高显存阶段**: {resource_summary.get('memory_bottleneck_phase', '-')}",
                f"- **总体显存峰值**: {_fmt(resource_summary.get('max_used_mib'))} MiB",
                f"- **最小剩余显存**: {_fmt(resource_summary.get('min_free_mib'))} MiB",
                f"- **Resource Gate**: `{'safe' if resource_summary.get('resource_safe') else 'unsafe'}`",
                "",
            ]
        )
    stability = metrics.get("stability") if isinstance(metrics.get("stability"), Mapping) else {}
    windows = stability.get("windows") if isinstance(stability.get("windows"), list) else []
    window_metrics = stability.get("window_metrics") if isinstance(stability.get("window_metrics"), Mapping) else {}
    if windows:
        lines.extend([f"**稳定性时序（每 {stability.get('window_size', '?')} step 一个 window）:**", "", "| Step window | Reward | PPO KL | Clip Fraction | Entropy | LR |", "|---|---:|---:|---:|---:|---:|"])
        keys = ("critic/rewards/mean", "actor/ppo_kl", "actor/pg_clipfrac", "actor/entropy", "actor/lr")
        specs = (".4f", ".8f", ".6f", ".4f", ".2e")
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                continue
            values = []
            for key, spec in zip(keys, specs):
                series = window_metrics.get(key)
                value = series[index] if isinstance(series, list) and index < len(series) else None
                values.append(_fmt(value, spec))
            label = f"{window.get('start_step', '?')}–{window.get('end_step', '?')} (n={window.get('sample_count', '?')})"
            lines.append("| " + label + " | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def _parameter_diff(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> str:
    if previous is None:
        return "#### 初始参数（基准）\n\n_完整参数见本 Trial 的 `parameters.json`。_\n"
    changes = [key for key in sorted(set(current) | set(previous)) if current.get(key) != previous.get(key)]
    if not changes:
        return "#### 参数变更\n\n_参数无变化。_\n"
    lines = ["#### 参数变更（相比上一 Trial）", "", "| 参数 | 旧值 | 新值 |", "|---|---|---|"]
    lines.extend(f"| `{key}` | `{_markdown(previous.get(key))}` | `{_markdown(current.get(key))}` |" for key in changes)
    return "\n".join(lines) + "\n"


def _trace_source_id(trace: Mapping[str, Any], fallback: int | None) -> int | None:
    proposal = trace.get("proposal_conversation")
    context = proposal.get("context") if isinstance(proposal, Mapping) else None
    last_trial = context.get("last_trial") if isinstance(context, Mapping) else None
    if isinstance(last_trial, Mapping) and isinstance(last_trial.get("trial_id"), int):
        return last_trial["trial_id"]
    return fallback


def process_experiment(run_dir: Path, output_path: Path) -> None:
    indexes = _read_jsonl(run_dir / "trials.jsonl")
    if not indexes:
        raise FileNotFoundError(f"no readable trial index at {run_dir / 'trials.jsonl'}")
    trials: list[dict[str, Any]] = []
    for index in indexes:
        if not isinstance(index.get("trial_id"), int):
            continue
        trials.append(
            {
                "index": index,
                "parameters": _read_json(_artifact_path(run_dir, index, "parameters")),
                "metrics": _read_json(_artifact_path(run_dir, index, "metrics")),
                "decision": _read_json(_artifact_path(run_dir, index, "decision")),
                "trace": _read_json(_artifact_path(run_dir, index, "agent_trace")),
            }
        )
    actions_by_source: dict[int, list[dict[str, Any]]] = {}
    for position, trial in enumerate(trials):
        trace = trial["trace"]
        if not trace or trace.get("_read_error"):
            continue
        fallback = trials[position - 1]["index"]["trial_id"] if position else None
        source = _trace_source_id(trace, fallback)
        if isinstance(source, int):
            actions_by_source.setdefault(source, []).append(trial)

    state = _read_json(run_dir / "state.json")
    lines = [
        f"# Agent 实验报告: `{run_dir.name}`",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据来源**: `{run_dir}`",
        f"**总 Trial 数**: {len(trials)}",
        "",
        "## 实验概览",
        "",
        f"- **最终阶段**: `{state.get('current_stage', '?')}`",
        f"- **总 Trial 数**: {state.get('last_trial_id', len(trials))}",
        "",
        "| Trial | 阶段 | 结果 | 吞吐量 (tok/s) | 每步耗时 (s) | 末窗口 Reward | 峰值显存 (MiB) | Resource Gate | 后续 Agent trace |",
        "|---|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for trial in trials:
        index = trial["index"]
        scores = index.get("scores") if isinstance(index.get("scores"), Mapping) else {}
        resource = index.get("resource") if isinstance(index.get("resource"), Mapping) else {}
        trace_count = len(actions_by_source.get(index["trial_id"], []))
        lines.append(
            f"| {index['trial_id']} | {index.get('stage', '?')} | {index.get('result', '?')} | {_fmt(scores.get('throughput_mean'))} | {_fmt(scores.get('time_per_step_mean_s'))} | {_fmt(scores.get('terminal_reward'), '.4f')} | {_fmt(resource.get('max_used_mib'))} | {'safe' if resource.get('resource_safe') else 'unsafe'} | {trace_count or '-'} |"
        )
    lines.extend(["", "---", "", "## 逐 Trial 详细分析", ""])

    previous_parameters: Mapping[str, Any] | None = None
    for position, trial in enumerate(trials):
        index, parameters, metrics, decision = trial["index"], trial["parameters"], trial["metrics"], trial["decision"]
        trial_dir = _artifact_path(run_dir, index, "report").parent
        lines.extend(
            [
                f"### Trial {index['trial_id']}: {index.get('stage', '?')}",
                "",
                f"- **结果**: `{index.get('result', '?')}` | **完成步数**: {index.get('updates_completed', '?')}/{index.get('updates_target', '?')}",
            ]
        )
        error = index.get("error") if isinstance(index.get("error"), Mapping) else {}
        if error.get("type"):
            lines.append(f"- **错误类型**: `{error['type']}`")
        lines.append("")
        lines.append(_parameter_diff(parameters, previous_parameters))
        previous_parameters = parameters
        lines.append(_metrics_section(metrics))
        lines.append(_health_section(_artifact_path(run_dir, index, "health_events"), _artifact_path(run_dir, index, "health_agent_traces")))
        actions = actions_by_source.get(index["trial_id"], [])
        lines.extend(["### 本 Trial 完成后的 Agent 行为", ""])
        if actions:
            for target in actions:
                target_id = target["index"]["trial_id"]
                lines.extend([f"_以下行为用于生成 Trial {target_id} 的候选配置。_", ""])
                trace = target["trace"]
                diagnosis = trace.get("diagnosis") if isinstance(trace.get("diagnosis"), Mapping) else target["decision"].get("diagnosis")
                if isinstance(diagnosis, Mapping):
                    lines.append(_diagnosis_section(diagnosis))
                proposal_trace = trace.get("proposal_conversation") if isinstance(trace.get("proposal_conversation"), Mapping) else {}
                calls = _tool_calls(proposal_trace)
                if calls:
                    lines.extend([f"**Proposal 工具调用 ({len(calls)} 次):**", "", _tool_table(calls)])
                proposal = _result(proposal_trace) or target["decision"].get("proposal", {})
                lines.append(_proposal_section(proposal if isinstance(proposal, Mapping) else {}))
                lines.append(_feasibility_section(trace.get("feasibility_reviews")))
        elif position == len(trials) - 1:
            lines.append("_这是最后一个 Trial，尚无后续 Agent trace。_")
        else:
            lines.append("_该 Trial 完成后没有记录 Diagnosis、Proposal 或 Feasibility trace。_")
        lines.extend(["", f"📁 Trial artifacts: `{trial_dir.relative_to(run_dir)}`", "", "---", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 报告已生成: {output_path}")
    print(f"   共 {len(trials)} 个 trial")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 schema-v2 trial artifacts 生成 Agent Markdown 报告。")
    parser.add_argument("experiment_dir", nargs="?", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="写入实验目录内的 Markdown 文件名")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name 必须是文件名，不能是路径")
    process_experiment(args.experiment_dir.expanduser().resolve(), args.experiment_dir.expanduser().resolve() / args.output_name)


if __name__ == "__main__":
    main()
