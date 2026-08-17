#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_json, write_json_atomic
from metrics import build_structured_metrics
from vllm_metrics import summarize_vllm_metrics


def extract_trial_metrics(
    trial_dir: str | Path,
    config: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    expected_gpu_count: int | None = None,
    vllm_summary: Mapping[str, Any] | None = None,
    monitor: Mapping[str, Any] | None = None,
    write_metrics: bool = True,
) -> dict[str, Any]:
    """Own the single-pass extraction and persistence of trial facts."""
    trial_path = Path(trial_dir).expanduser().resolve()
    parameters_path = trial_path / "parameters.json"
    effective_parameters = (
        dict(parameters)
        if isinstance(parameters, Mapping)
        else load_json(parameters_path)
        if parameters_path.is_file()
        else {}
    )
    vllm_path = trial_path / "vllm_metrics.csv"
    effective_vllm_summary = (
        dict(vllm_summary)
        if isinstance(vllm_summary, Mapping)
        else summarize_vllm_metrics(vllm_path if vllm_path.exists() else None)
    )
    metrics = build_structured_metrics(
        trial_path / "train.log",
        trial_path / "gpu_samples.csv",
        warmup_updates=int(config.get("warmup_updates", 5)),
        reward_window=int(config.get("reward_window", 5)),
        reward_thresholds=config.get("reward_thresholds", [0.0, 0.1, 0.2, 0.3]),
        stability_window_size=int(config.get("stability_window_size", 5)),
        vllm_summary=effective_vllm_summary,
        vllm_metrics_path=vllm_path if vllm_path.exists() else None,
        health_events_path=(
            trial_path / "health_events.jsonl"
            if (trial_path / "health_events.jsonl").exists()
            else None
        ),
        expected_gpu_count=expected_gpu_count,
        resource_reserve_mib=float(config.get("resource_memory_reserve_mib", 3277)),
        throughput_reserve_mib=float(config.get("throughput_memory_reserve_mib", 6554)),
        resource_gate_updates=int(config.get("resource_gate_updates", 1)),
        monitor=monitor,
        parameters=effective_parameters,
    )
    log_facts = metrics.pop("log_facts", {})
    write_json_atomic(trial_path / "log_facts.json", log_facts)
    metrics["source"]["log_facts"] = "log_facts.json"
    if write_metrics:
        write_json_atomic(trial_path / "metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse one trial's raw artifacts into classified metrics.json"
    )
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--agent-config")
    parser.add_argument("--expected-gpu-count", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir).expanduser().resolve()
    config = {}
    if args.agent_config:
        config = json.loads(Path(args.agent_config).read_text(encoding="utf-8"))
    metrics = extract_trial_metrics(
        trial_dir,
        config,
        expected_gpu_count=args.expected_gpu_count,
        write_metrics=args.output is None,
    )
    output = Path(args.output).expanduser().resolve() if args.output else trial_dir / "metrics.json"
    if args.output:
        write_json_atomic(output, metrics)
    print(json.dumps({"output": str(output), "latest_step": metrics["latest_step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
