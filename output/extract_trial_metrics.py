#!/usr/bin/env python3
"""Generate a per-trial parameter, memory, and throughput Markdown report.

Usage:
    python3 output/extract_trial_metrics.py
    python3 output/extract_trial_metrics.py output/0723_1550_2026
    python3 output/extract_trial_metrics.py output/0723_1550_2026 \
        --output-name trial_metrics_report.md

Edit ``DEFAULT_EXPERIMENT_DIR`` below to change the no-argument input. The
report is always written inside the selected experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import parse_step_records


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent / "0724_1741_2026"
DEFAULT_OUTPUT_NAME = "trial_metrics_report.md"

PHASE_ORDER = (
    "startup",
    "rollout",
    "actor_log_prob",
    "ref_log_prob",
    "training",
    "between_phases",
    "unknown",
)
TRAINING_PHASES = ("rollout", "actor_log_prob", "ref_log_prob", "training")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _markdown(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return rendered.replace("|", "\\|").replace("\n", "<br>")


def _trial_directories(experiment_dir: Path) -> list[Path]:
    trials_dir = experiment_dir / "trials"
    if not trials_dir.is_dir():
        raise FileNotFoundError(f"trial directory does not exist: {trials_dir}")
    return sorted(
        (
            path
            for path in trials_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def _parameter_changes(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if previous is None:
        return []
    changes: list[dict[str, Any]] = []
    for name in sorted(set(previous) | set(current)):
        before_present = name in previous
        after_present = name in current
        before = previous.get(name)
        after = current.get(name)
        if before_present and after_present and before == after:
            continue
        change_type = (
            "新增"
            if not before_present
            else "移除"
            if not after_present
            else "修改"
        )
        changes.append(
            {
                "parameter": name,
                "type": change_type,
                "before": before if before_present else None,
                "after": after if after_present else None,
            }
        )
    return changes


def _phase_memory_peaks(csv_path: Path) -> dict[str, dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    if not csv_path.exists():
        return peaks
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                used_mb = float(row["memory_used_mb"])
                total_mb = float(row["memory_total_mb"])
                utilization = float(row["utilization_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            if total_mb <= 0:
                continue
            phase = row.get("phase") or "unknown"
            memory_pct = 100.0 * used_mb / total_mb
            if phase not in peaks or memory_pct > peaks[phase]["memory_pct"]:
                peaks[phase] = {
                    "memory_pct": memory_pct,
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "gpu_index": row.get("gpu_index", "?"),
                    "gpu_utilization_pct": utilization,
                    "timestamp": row.get("timestamp", ""),
                }
    return peaks


def _step_throughput(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    records = parse_step_records(log_path)
    return [
        (step, row["perf/throughput"])
        for step, row in records.items()
        if "perf/throughput" in row
    ]


def _ordered_phases(peaks: Mapping[str, Any]) -> list[str]:
    order = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    return sorted(peaks, key=lambda phase: (order.get(phase, len(order)), phase))


def _load_trial(trial_dir: Path) -> dict[str, Any]:
    report = _read_json(trial_dir / "trial_report.json")
    parameters = _read_json(trial_dir / "parameters.json")
    if not parameters and isinstance(report.get("parameters"), dict):
        parameters = dict(report["parameters"])
    memory_peaks = _phase_memory_peaks(trial_dir / "gpu_samples.csv")
    throughput = _step_throughput(trial_dir / "train.log")
    return {
        "trial_id": int(trial_dir.name),
        "stage": report.get("stage", "?"),
        "result": report.get("result", "?"),
        "updates_completed": report.get("updates_completed"),
        "updates_target": report.get("updates_target"),
        "parameters": parameters,
        "memory_peaks": memory_peaks,
        "throughput": throughput,
        "trial_dir": trial_dir,
    }


def _fmt_number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def generate_report(experiment_dir: Path, output_name: str) -> Path:
    experiment_dir = experiment_dir.expanduser().resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"experiment directory does not exist: {experiment_dir}"
        )
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a file name, not a path")

    trials = [_load_trial(path) for path in _trial_directories(experiment_dir)]
    previous_parameters: Mapping[str, Any] | None = None
    for trial in trials:
        trial["changes"] = _parameter_changes(
            previous_parameters, trial["parameters"]
        )
        previous_parameters = trial["parameters"]

    lines: list[str] = [
        f"# Trial 参数、显存与吞吐报告：`{experiment_dir.name}`",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 实验目录：`{experiment_dir}`",
        f"- Trial 数量：{len(trials)}",
        "",
        "## Trial 总览",
        "",
        "| Trial | 阶段 | 结果 | 参数变化数 | Throughput step 数 | Throughput 均值 | Throughput 最大值 | Rollout 峰值显存 | Actor log-prob 峰值 | Ref log-prob 峰值 | Training 峰值 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for trial in trials:
        throughput_values = [value for _, value in trial["throughput"]]
        peaks = trial["memory_peaks"]
        peak_cells = [
            _fmt_number(
                peaks.get(phase, {}).get("memory_pct")
                if phase in peaks
                else None
            )
            + ("%"
               if phase in peaks
               else "")
            for phase in TRAINING_PHASES
        ]
        lines.append(
            "| {trial_id} | {stage} | {result} | {change_count} | "
            "{step_count} | {throughput_mean} | {throughput_max} | "
            "{peak_cells} |".format(
                trial_id=trial["trial_id"],
                stage=trial["stage"],
                result=trial["result"],
                change_count=(
                    "-" if trial["trial_id"] == trials[0]["trial_id"]
                    else len(trial["changes"])
                ),
                step_count=len(throughput_values),
                throughput_mean=_fmt_number(
                    mean(throughput_values) if throughput_values else None
                ),
                throughput_max=_fmt_number(
                    max(throughput_values) if throughput_values else None
                ),
                peak_cells=" | ".join(peak_cells),
            )
        )

    for index, trial in enumerate(trials):
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Trial {trial['trial_id']:04d}",
                "",
                f"- 阶段：`{trial['stage']}`",
                f"- 结果：`{trial['result']}`",
                (
                    f"- 完成步数：{trial['updates_completed']}/"
                    f"{trial['updates_target']}"
                ),
                "",
                "### 相比上一 Trial 的参数变化",
                "",
            ]
        )

        if index == 0:
            lines.append(
                f"_首个 Trial，为基准参数，共 {len(trial['parameters'])} 项。_"
            )
        elif not trial["changes"]:
            lines.append("_参数无变化。_")
        else:
            lines.extend(
                [
                    "| 参数 | 类型 | 旧值 | 新值 |",
                    "|---|---|---|---|",
                ]
            )
            for change in trial["changes"]:
                lines.append(
                    f"| `{change['parameter']}` | {change['type']} | "
                    f"`{_markdown(change['before'])}` | "
                    f"`{_markdown(change['after'])}` |"
                )

        lines.extend(["", "### 各阶段显存峰值", ""])
        if not trial["memory_peaks"]:
            lines.append("_未找到有效的 `gpu_samples.csv` 数据。_")
        else:
            lines.extend(
                [
                    "| 阶段 | 峰值显存 | 已用/总显存 (MB) | GPU | 峰值时 GPU 利用率 | 时间戳 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for phase in _ordered_phases(trial["memory_peaks"]):
                peak = trial["memory_peaks"][phase]
                lines.append(
                    f"| `{phase}` | {peak['memory_pct']:.2f}% | "
                    f"{peak['used_mb']:.0f}/{peak['total_mb']:.0f} | "
                    f"{peak['gpu_index']} | {peak['gpu_utilization_pct']:.1f}% | "
                    f"{peak['timestamp']} |"
                )

        throughput_values = [value for _, value in trial["throughput"]]
        lines.extend(["", "### 每个 Step 的 Throughput", ""])
        if not throughput_values:
            lines.append("_未从 `train.log` 提取到 `perf/throughput`。_")
        else:
            lines.extend(
                [
                    f"- Step 数：{len(throughput_values)}",
                    f"- 均值：{mean(throughput_values):.4f}",
                    f"- 最小值：{min(throughput_values):.4f}",
                    f"- 最大值：{max(throughput_values):.4f}",
                    "",
                    "| Step | Throughput |",
                    "|---:|---:|",
                ]
            )
            lines.extend(
                f"| {step} | {value:.6f} |"
                for step, value in trial["throughput"]
            )

    output_path = experiment_dir / output_name
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown report containing inter-trial parameter "
            "changes, per-phase GPU memory peaks, and per-step throughput."
        )
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"experiment directory (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Markdown file name written inside the experiment directory",
    )
    args = parser.parse_args()
    try:
        output_path = generate_report(args.experiment_dir, args.output_name)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"报告已生成：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
