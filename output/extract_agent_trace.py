#!/usr/bin/env python3
"""
提取实验目录中每个 trial 的 agent 行为（工具调用、参数变更、决策原因）并生成 Markdown 报告。

用法:
    python3 output/extract_agent_trace.py
    python3 extract_agent_trace.py ssh_agent/output/0720_1656_2026 [output.md]

直接修改下方 ``DEFAULT_EXPERIMENT_DIR`` 即可更换无参数运行时的实验目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent / "0724_1741_2026"
DEFAULT_OUTPUT_NAME = "agent_report.md"
PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def _compact_json(obj, max_len=120):
    """紧凑的 JSON 字符串，超长截断。"""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def extract_tool_calls(tool_calls: list[dict]) -> str:
    """格式化工具调用列表为 Markdown 表格。"""
    if not tool_calls:
        return "_无工具调用_\n"

    lines = [
        "| 轮次 | 工具 | 参数 / 查询内容 | 状态 |",
        "|---|---|---|---|",
    ]
    # Group by tool_round
    by_round: dict[int, list[dict]] = {}
    for tc in tool_calls:
        by_round.setdefault(tc.get("tool_round", 0), []).append(tc)

    for rnd in sorted(by_round):
        for tc in by_round[rnd]:
            name = tc.get("name", "?")
            args = tc.get("arguments", {})
            status = tc.get("status", "?")
            # 简化参数展示
            if name == "parameter_understanding":
                items = args.get("items", [])
                arg_summary = ", ".join(items[:6])
                if len(items) > 6:
                    arg_summary += f" …共 {len(items)} 个"
            elif name == "memory_estimator":
                changes = args.get("changes", {})
                ref = args.get("reference_trial_id", "?")
                param_names = ", ".join(changes.keys())
                arg_summary = f"ref_trial={ref}, 预测参数: {param_names}"
            elif name == "tuning_strategies":
                items = args.get("items", [])
                arg_summary = ", ".join(items)
            elif name == "query_trial_history":
                arg_summary = f"stage={args.get('stage')}, limit={args.get('limit')}"
            elif name == "search_verl_docs":
                arg_summary = f'查询: "{args.get("query", "")}"'
            else:
                arg_summary = _compact_json(args, 100)
            lines.append(f"| {rnd} | `{name}` | {arg_summary} | {status} |")
    return "\n".join(lines) + "\n"


def extract_proposal_changes(proposal: dict | None) -> str:
    """格式化 proposal 中的参数变更为 Markdown 表格。"""
    if not proposal:
        return "_无 proposal_\n"

    decision = proposal.get("decision", "?")
    reason = proposal.get("reason", "")
    confidence = proposal.get("confidence")
    changes = proposal.get("changes", {})

    lines = [
        f"- **决策**: `{decision}`",
        f"- **原因**: {reason}",
    ]
    if confidence is not None:
        lines.append(f"- **置信度**: {confidence}")
    lines.append("")

    if changes and isinstance(changes, dict) and decision == "modify":
        # 判断 changes 是否是 {key: {from, to, reason}} 结构
        first_val = next(iter(changes.values()), None)
        if isinstance(first_val, dict) and "from" in first_val:
            lines.append("| 参数 | 旧值 | 新值 | 原因 |")
            lines.append("|---|---|---|---|")
            for key, ch in changes.items():
                lines.append(
                    f"| `{key}` | `{ch.get('from')}` | `{ch.get('to')}` "
                    f"| {ch.get('reason', '')} |"
                )
        else:
            # 简易格式 {key: value}
            lines.append("| 参数 | 目标值 |")
            lines.append("|---|---|")
            for key, val in changes.items():
                lines.append(f"| `{key}` | `{val}` |")
        lines.append("")

    return "\n".join(lines)


def _feasibility_result(review: dict) -> dict:
    """兼容新旧 trace：新格式把最终判定放在 review.result 中。"""
    for key in ("result", "review", "decision"):
        value = review.get(key)
        if isinstance(value, dict):
            return value
    return review


def _feasibility_tool_calls(review: dict) -> list[dict]:
    """从 feasibility Agent 运行记录中提取工具调用。"""
    tool_calls = review.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls

    # 兼容曾经把 Agent 运行信息包在 conversation/trace 中的格式。
    for key in ("conversation", "trace"):
        nested = review.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("tool_calls"), list):
            return nested["tool_calls"]
    return []


def extract_feasibility(feas: list[dict] | dict | None) -> str:
    """格式化 feasibility Agent 的工具调用、判定、风险和显存预测。"""
    if not feas:
        return ""
    if isinstance(feas, dict):
        feas = [feas]

    lines = ["#### Feasibility Agent 审查", ""]
    for i, review in enumerate(feas):
        if not isinstance(review, dict):
            continue

        result = _feasibility_result(review)
        tool_calls = _feasibility_tool_calls(review)
        verdict = result.get("verdict", "?")
        reason = result.get("reason", "")
        attempt = review.get("attempt", i + 1)

        lines.append(f"**审查 #{i + 1}（attempt={attempt}）**")
        lines.append("")
        if tool_calls:
            lines.append(f"**Feasibility 工具调用 ({len(tool_calls)} 次):**")
            lines.append("")
            lines.append(extract_tool_calls(tool_calls))
        else:
            lines.append("_Feasibility Agent 无工具调用_")
            lines.append("")

        lines.append(f"- **判定**: `{verdict}`")
        lines.append(f"- **原因**: {reason or '-'}")

        risks = result.get("risks", [])
        if risks:
            lines.append("- **风险**:")
            for risk in risks:
                lines.append(f"  - {risk}")

        predicted = result.get("predicted_memory_pct")
        if isinstance(predicted, dict) and predicted:
            lines.append("")
            lines.append("**预测显存占用 (%):**")
            lines.append("")
            lines.append("| rollout | actor_log_prob | ref_log_prob | training |")
            lines.append("|---:|---:|---:|---:|")
            lines.append(
                "| "
                + " | ".join(
                    _fmt(predicted.get(phase), ".2f")
                    for phase in (
                        "rollout",
                        "actor_log_prob",
                        "ref_log_prob",
                        "training",
                    )
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def extract_diagnosis(diagnosis: dict | None) -> str:
    """格式化 Diagnosis Agent 的工具调用和最终诊断。"""
    if not isinstance(diagnosis, dict):
        return ""

    result = _feasibility_result(diagnosis)
    tool_calls = _feasibility_tool_calls(diagnosis)
    lines = ["#### Diagnosis Agent 诊断", ""]

    if tool_calls:
        lines.append(f"**Diagnosis 工具调用 ({len(tool_calls)} 次):**")
        lines.append("")
        lines.append(extract_tool_calls(tool_calls))
    else:
        lines.append("_Diagnosis Agent 无工具调用_")
        lines.append("")

    lines.append(f"- **失败类型**: `{result.get('failure_type', '?')}`")
    lines.append(f"- **训练子阶段**: `{result.get('training_substage', '?')}`")
    if result.get("confidence") is not None:
        lines.append(f"- **置信度**: {result['confidence']}")
    lines.append(f"- **原因**: {result.get('reason') or '-'}")

    evidence = result.get("evidence", [])
    if evidence:
        lines.append("- **证据**:")
        for item in evidence:
            lines.append(f"  - {item}")
    lines.append("")
    return "\n".join(lines)


def extract_health_decisions(decisions: list[dict] | None) -> str:
    """格式化 Trial 运行期间 Health Monitor 的决策。"""
    if not decisions:
        return ""

    lines = ["### Trial 运行中的 Health Monitor 行为", ""]
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            continue

        event_id = decision.get("event_id", f"event-{index + 1}")
        lines.append(f"#### Health 决策 #{index + 1}: `{event_id}`")
        lines.append("")
        lines.append(f"- **判定**: `{decision.get('verdict', '?')}`")
        lines.append(f"- **动作**: `{decision.get('action', '?')}`")
        if decision.get("confidence") is not None:
            lines.append(f"- **置信度**: {decision['confidence']}")
        if decision.get("observe_for_updates") is not None:
            lines.append(
                f"- **继续观察步数**: {decision['observe_for_updates']}"
            )
        lines.append(f"- **原因**: {decision.get('reason') or '-'}")

        reason_codes = decision.get("reason_codes", [])
        if reason_codes:
            lines.append(
                "- **原因代码**: "
                + ", ".join(f"`{code}`" for code in reason_codes)
            )

        evidence = decision.get("evidence", [])
        if evidence:
            lines.append("- **支持证据**:")
            for item in evidence:
                lines.append(f"  - {item}")

        counterevidence = decision.get("counterevidence", [])
        if counterevidence:
            lines.append("- **反向证据**:")
            for item in counterevidence:
                lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines)


def extract_rejections(rejections: list[dict] | None) -> str:
    """格式化 rejections。"""
    if not rejections:
        return ""
    lines = ["### 建议被拒绝记录", ""]
    for i, r in enumerate(rejections):
        source = r.get("source", "?")
        violations = r.get("violations", [])
        lines.append(f"- **#{i+1}** 来源: `{source}`")
        for v in violations:
            lines.append(f"  - ❌ {v}")
    lines.append("")
    return "\n".join(lines)


def _trace_source_trial_id(trace: dict) -> int | None:
    """找出一次 Agent trace 实际分析的是哪个已完成 Trial。"""
    proposal = trace.get("proposal_conversation")
    if isinstance(proposal, dict):
        context = proposal.get("context")
        if isinstance(context, dict):
            last_trial = context.get("last_trial")
            if isinstance(last_trial, dict) and isinstance(
                last_trial.get("trial_id"), int
            ):
                return last_trial["trial_id"]

    diagnosis = trace.get("diagnosis")
    if isinstance(diagnosis, dict):
        context = diagnosis.get("context")
        if isinstance(context, dict):
            diagnosed_trial = context.get("trial")
            if isinstance(diagnosed_trial, dict) and isinstance(
                diagnosed_trial.get("trial_id"), int
            ):
                return diagnosed_trial["trial_id"]
    return None


def _trace_tool_counts(trace: dict) -> tuple[int, int, int]:
    """返回 Diagnosis、Proposal、Feasibility 的工具调用数。"""
    diagnosis = trace.get("diagnosis")
    diagnosis_count = (
        len(_feasibility_tool_calls(diagnosis))
        if isinstance(diagnosis, dict)
        else 0
    )

    proposal = trace.get("proposal_conversation")
    proposal_count = (
        len(_feasibility_tool_calls(proposal))
        if isinstance(proposal, dict)
        else 0
    )

    feasibility = trace.get("feasibility_reviews") or []
    if isinstance(feasibility, dict):
        feasibility = [feasibility]
    feasibility_count = sum(
        len(_feasibility_tool_calls(review))
        for review in feasibility
        if isinstance(review, dict)
    )
    return diagnosis_count, proposal_count, feasibility_count


def _fmt(val, spec=".1f"):
    """安全格式化数值，处理 None。"""
    if val is None:
        return "-"
    return format(val, spec)


def extract_metrics(trial: dict) -> str:
    """提取关键指标。"""
    lines = ["### 关键指标", ""]

    perf = trial.get("performance", {})
    if perf:
        tp = perf.get("throughput", {})
        tps = perf.get("time_per_step_s", {})
        tgs = perf.get("generation_tgs", {})
        mfu = perf.get("actor_mfu", {})
        bottleneck = perf.get("time_bottleneck", "")
        lines.append("| 指标 | 均值 | P95 | 最大值 |")
        lines.append("|---|---|---|---|")
        if tp:
            lines.append(f"| 吞吐量 (tok/s) | {_fmt(tp.get('mean'), '.1f')} | {_fmt(tp.get('p95'), '.1f')} | {_fmt(tp.get('max'), '.1f')} |")
        if tps:
            lines.append(f"| 每步耗时 (s) | {_fmt(tps.get('mean'), '.1f')} | {_fmt(tps.get('p95'), '.1f')} | {_fmt(tps.get('max'), '.1f')} |")
        if tgs:
            lines.append(f"| 生成 tgs | {_fmt(tgs.get('mean'), '.1f')} | {_fmt(tgs.get('p95'), '.1f')} | {_fmt(tgs.get('max'), '.1f')} |")
        if mfu:
            lines.append(f"| Actor MFU | {_fmt(mfu.get('mean'), '.4f')} | {_fmt(mfu.get('p95'), '.4f')} | {_fmt(mfu.get('max'), '.4f')} |")
        if bottleneck:
            lines.append(f"| **时间瓶颈** | {bottleneck} |||")
        lines.append("")

    stability = trial.get("stability", {})
    if stability:
        reward = stability.get("reward", {})
        kl = stability.get("actor_ppo_kl", {})
        entropy = stability.get("actor_entropy", {})
        clipfrac = stability.get("actor_pg_clipfrac", {})
        resp_len = stability.get("response_length", {})

        lines.append("**稳定性指标:**")
        lines.append("")
        lines.append("| 指标 | 均值 | P95 | 最大 |")
        lines.append("|---|---|---|---|")
        if reward:
            lines.append(f"| Reward | {_fmt(reward.get('mean'), '.4f')} | {_fmt(reward.get('p95'), '.4f')} | {_fmt(reward.get('max'), '.4f')} |")
        lines.append(f"| Reward 斜率 | {_fmt(stability.get('reward_slope'), '.6f')} |||")
        if kl:
            lines.append(f"| Actor PPO KL | {_fmt(kl.get('mean'), '.8f')} | {_fmt(kl.get('p95'), '.8f')} | {_fmt(kl.get('max'), '.8f')} |")
        if entropy:
            lines.append(f"| Actor Entropy | {_fmt(entropy.get('mean'), '.4f')} | {_fmt(entropy.get('p95'), '.4f')} | {_fmt(entropy.get('max'), '.4f')} |")
        if clipfrac:
            lines.append(f"| Clip Fraction | {_fmt(clipfrac.get('mean'), '.6f')} | {_fmt(clipfrac.get('p95'), '.6f')} | {_fmt(clipfrac.get('max'), '.6f')} |")
        if resp_len:
            lines.append(f"| Response Length | {_fmt(resp_len.get('mean'), '.1f')} | {_fmt(resp_len.get('p95'), '.1f')} | {_fmt(resp_len.get('max'), '.1f')} |")
        lines.append("")

    phase_duration = (
        perf.get("phase_duration_s", {}) if isinstance(perf, dict) else {}
    )
    phase_memory = trial.get("memory_by_phase_pct", {})
    phase_gpu_utilization = trial.get("gpu_utilization_by_phase_pct", {})
    resource = trial.get("resource", {})
    phase_metrics = (phase_duration, phase_memory, phase_gpu_utilization)
    if any(isinstance(metric, dict) and metric for metric in phase_metrics):
        lines.append("**分阶段耗时、显存与 GPU 利用率:**")
        lines.append("")
        lines.append(
            "| 阶段 | 耗时均值 (s) | 显存均值 (%) | 显存峰值 (%) | "
            "GPU 利用率均值 (%) | GPU 利用率 P95 (%) | "
            "GPU 利用率峰值 (%) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pname in PHASES:
            duration = (
                phase_duration.get(pname, {})
                if isinstance(phase_duration, dict)
                else {}
            )
            memory = (
                phase_memory.get(pname, {})
                if isinstance(phase_memory, dict)
                else {}
            )
            utilization = (
                phase_gpu_utilization.get(pname, {})
                if isinstance(phase_gpu_utilization, dict)
                else {}
            )
            duration = duration if isinstance(duration, dict) else {}
            memory = memory if isinstance(memory, dict) else {}
            utilization = utilization if isinstance(utilization, dict) else {}
            if not any(
                summary for summary in (duration, memory, utilization)
            ):
                continue
            lines.append(
                f"| {pname} | {_fmt(duration.get('mean'), '.1f')} | "
                f"{_fmt(memory.get('mean'), '.1f')} | "
                f"{_fmt(memory.get('max'), '.1f')} | "
                f"{_fmt(utilization.get('mean'), '.1f')} | "
                f"{_fmt(utilization.get('p95'), '.1f')} | "
                f"{_fmt(utilization.get('max'), '.1f')} |"
            )
        lines.append("")

    if isinstance(resource, dict) and resource:
        relative_peak_phase = resource.get("memory_bottleneck", "-")
        observed_peak = resource.get("max_observed_memory_pct")
        lines.append(
            f"- **相对最高显存阶段**: {relative_peak_phase} "
            "（仅表示四阶段中峰值最高，不等于存在显存压力）"
        )
        lines.append(f"- **总体实测峰值**: {_fmt(observed_peak, '.1f')}%")
        lines.append("")

    return "\n".join(lines)


def extract_param_diff(current: dict, previous: dict | None) -> str:
    """对比两个 trial 的参数差异。"""
    if not previous:
        return ""
    diffs = []
    for key in sorted(set(list(current.keys()) + list(previous.keys()))):
        cv = current.get(key)
        pv = previous.get(key)
        if cv != pv:
            diffs.append(f"| `{key}` | `{pv}` | `{cv}` |")

    if not diffs:
        return "_参数无变化_\n"

    lines = ["| 参数 | 旧值 | 新值 |", "|---|---|---|"]
    lines.extend(diffs)
    return "\n".join(lines) + "\n"


# ── 主逻辑 ──────────────────────────────────────────────

def process_experiment(exp_dir: Path, output_path: Path) -> None:
    trials_path = exp_dir / "trials.jsonl"
    if not trials_path.exists():
        print(f"错误: 找不到 {trials_path}")
        sys.exit(1)

    state_path = exp_dir / "state.json"
    state_info = ""
    if state_path.exists():
        state = json.load(state_path.open(encoding="utf-8"))
        state_info = (
            f"- **最终阶段**: `{state.get('current_stage', '?')}`\n"
            f"- **总 Trial 数**: {state.get('last_trial_id', '?')}\n"
        )

    # 读取所有 trial
    trials = []
    with trials_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    # agent_trace 随“候选 Trial”保存，但其中的 Agent 行为发生在上一个
    # Trial 完成之后。报告按被分析的源 Trial 归属，避免 Trial 5 这类
    # 阶段基准 Trial 显示成“没有 Agent 行为”。
    actions_by_source: dict[int, list[tuple[dict, dict]]] = {}
    for index, target_trial in enumerate(trials):
        trace = target_trial.get("agent_trace")
        if not isinstance(trace, dict) or not trace:
            continue
        source_id = _trace_source_trial_id(trace)
        if source_id is None and index > 0:
            source_id = trials[index - 1].get("trial_id")
        if isinstance(source_id, int):
            actions_by_source.setdefault(source_id, []).append((target_trial, trace))

    # 提取实验时间戳
    exp_name = exp_dir.name
    ts_match = None
    for part in exp_name.split("_"):
        if len(part) == 4 and part.isdigit():
            ts_match = part
            break

    lines: list[str] = []
    lines.append(f"# Agent 实验报告: `{exp_name}`")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据来源**: `{exp_dir}`")
    lines.append(f"**总 Trial 数**: {len(trials)}")
    lines.append("")
    lines.append("## 实验概览")
    lines.append("")
    lines.append(state_info)
    lines.append("")

    # 试验总览表
    lines.append("| Trial | 阶段 | 结果 | 吞吐量 | Reward (均值) | Reward (最大) | 显存峰值% | Health 决策 | 完成后的 Agent 工具调用（D + P + F） |")
    lines.append("|---|---|---|---:|---:|---:|---:|:---:|:---:|")
    for t in trials:
        tid = t.get("trial_id", "?")
        stage = t.get("stage", "?")
        result = t.get("result", "?")
        tp = t.get("performance", {}).get("throughput", {}).get("mean", "-")
        if isinstance(tp, (int, float)):
            tp = f"{tp:.0f}"
        rw_mean = t.get("stability", {}).get("reward", {}).get("mean", "-")
        if isinstance(rw_mean, (int, float)):
            rw_mean = f"{rw_mean:.4f}"
        rw_max = t.get("stability", {}).get("reward", {}).get("max", "-")
        if isinstance(rw_max, (int, float)):
            rw_max = f"{rw_max:.4f}"
        mem = t.get("resource", {}).get("max_observed_memory_pct", "-")
        if isinstance(mem, (int, float)):
            mem = f"{mem:.1f}%"
        diagnosis_tools = 0
        proposal_tools = 0
        feasibility_tools = 0
        for _, trace in actions_by_source.get(tid, []):
            d_count, p_count, f_count = _trace_tool_counts(trace)
            diagnosis_tools += d_count
            proposal_tools += p_count
            feasibility_tools += f_count
        total_tools = diagnosis_tools + proposal_tools + feasibility_tools
        tool_str = (
            f"{diagnosis_tools} + {proposal_tools} + "
            f"{feasibility_tools} = {total_tools}"
            if total_tools
            else "-"
        )
        health_count = len(t.get("health_decisions") or [])
        health_str = str(health_count) if health_count else "-"

        lines.append(
            f"| {tid} | {stage} | {result} | {tp} | {rw_mean} | {rw_max} "
            f"| {mem} | {health_str} | {tool_str} |"
        )
    lines.append("")

    # ── 逐 Trial 详细展开 ──
    lines.append("---")
    lines.append("")
    lines.append("## 逐 Trial 详细分析")
    lines.append("")

    prev_params = None
    for idx, t in enumerate(trials):
        tid = t.get("trial_id", "?")
        stage = t.get("stage", "?")
        result = t.get("result", "?")
        updates = t.get("updates_completed", "?")

        lines.append(f"### Trial {tid}: {stage}")
        lines.append("")
        lines.append(f"- **结果**: `{result}` | **完成步数**: {updates}/{t.get('updates_target', '?')}")
        error = t.get("error", {})
        if error and error.get("type"):
            lines.append(f"- **错误类型**: {error['type']}")
        lines.append("")

        # 参数对比
        params = t.get("parameters", {})
        if prev_params is not None and params != prev_params:
            lines.append("#### 参数变更（相比上一 Trial）")
            lines.append("")
            lines.append(extract_param_diff(params, prev_params))
            lines.append("")
        elif idx == 0:
            lines.append("#### 初始参数（基准）")
            lines.append("")
            lines.append(f"_完整参数见 `trials/{tid:04d}/parameters.json`_")
            lines.append("")
        prev_params = params

        # 关键指标
        lines.append(extract_metrics(t))

        # Health Monitor 决策发生在当前 Trial 运行期间。
        health_decisions = t.get("health_decisions") or []
        if health_decisions:
            lines.append(extract_health_decisions(health_decisions))

        # 其他 Agent 行为按“该 Trial 完成之后”归属；trace 本身保存在目标 Trial。
        actions = actions_by_source.get(tid, [])
        if actions:
            lines.append("### 本 Trial 完成后的 Agent 行为")
            lines.append("")

            for target_trial, trace in actions:
                target_id = target_trial.get("trial_id", "?")
                lines.append(f"_以下行为用于生成 Trial {target_id} 的候选配置。_")
                lines.append("")

                diagnosis = trace.get("diagnosis")
                if diagnosis:
                    lines.append(extract_diagnosis(diagnosis))

                conv = trace.get("proposal_conversation") or {}
                tool_calls = _feasibility_tool_calls(conv)
                if conv:
                    lines.append("#### Proposal Agent 决策")
                    lines.append("")
                    if tool_calls:
                        lines.append(
                            f"**Proposal 工具调用 ({len(tool_calls)} 次):**"
                        )
                        lines.append("")
                        lines.append(extract_tool_calls(tool_calls))
                        lines.append("")

                    proposal = conv.get("result")
                    if not isinstance(proposal, dict):
                        proposal = target_trial.get("proposal", {})
                    lines.append(extract_proposal_changes(proposal))
                    lines.append("")

                feas = trace.get("feasibility_reviews", [])
                if feas:
                    lines.append(extract_feasibility(feas))

                rejs = trace.get("rejections", [])
                if rejs:
                    lines.append(extract_rejections(rejs))
        else:
            lines.append("### 本 Trial 完成后的 Agent 行为")
            lines.append("")
            if idx == len(trials) - 1:
                lines.append("_这是最后一个 Trial，记录中没有后续 Agent trace。_")
            else:
                lines.append("_该 Trial 完成后没有记录 Diagnosis、Proposal 或 Feasibility trace。_")
            lines.append("")

        # 实验日志路径
        log_path = t.get("log_path", "")
        if log_path:
            lines.append(f"📄 [训练日志]({log_path})")
            lines.append("")

        lines.append("---")
        lines.append("")

    # 写入文件
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 报告已生成: {output_path}")
    print(f"   共 {len(trials)} 个 trial")


def main():
    parser = argparse.ArgumentParser(
        description="提取实验目录中的 Agent 行为并生成 Markdown 报告。"
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"实验目录（默认：{DEFAULT_EXPERIMENT_DIR}）",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="写入实验目录内的 Markdown 文件名",
    )
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name 必须是文件名，不能是路径")

    exp_dir = args.experiment_dir.expanduser().resolve()
    output_path = exp_dir / args.output_name
    process_experiment(exp_dir, output_path)


if __name__ == "__main__":
    main()
