#!/usr/bin/env python3
"""Convert recorded output runs into canonical Summer experience summaries."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import AgentError, AgentResponseError, AgentSet
from config_utils import load_json, write_json_atomic
from summary_store import rebuild_summary_index
from tools.replay_summer_prompt import build_summer_context, resolve_run_dir
from trial_storage import read_trial_indexes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "output"
DEFAULT_AGENT_CONFIG = ROOT / "config" / "agent_config.json"


def discover_run_dirs(output_root: str | Path) -> list[Path]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output root does not exist: {root}")
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and (path / "trials.jsonl").is_file()
    )


def _latest_trial_id(run_dir: Path) -> int:
    trial_ids = [
        row.get("trial_id")
        for row in read_trial_indexes(run_dir / "trials.jsonl")
        if isinstance(row.get("trial_id"), int)
        and not isinstance(row.get("trial_id"), bool)
    ]
    if not trial_ids:
        raise ValueError(f"run has no recorded trials: {run_dir}")
    return max(trial_ids)


def backfill_run(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    result_path = run_path / "summer" / "summer_result.json"
    result_existed = result_path.is_file()
    if result_existed and not overwrite:
        return {
            "run_id": run_path.name,
            "status": "skipped",
            "reason": "summer_result.json already exists",
            "result_path": str(result_path),
        }

    latest_trial_id = _latest_trial_id(run_path)
    trace_path = run_path / "trials" / f"{latest_trial_id:04d}" / "agent_trace.json"
    current_trace = load_json(trace_path) if trace_path.is_file() else {}
    if not isinstance(current_trace, dict):
        current_trace = {}

    context = build_summer_context(run_path)
    agents = AgentSet(ROOT, "llm", config, run_path / "trials.jsonl")
    run = agents.summarize({"trial": context})
    raw_result = copy.deepcopy(run.result)
    raw_result.pop("run_context", None)
    result = {
        "run_context": copy.deepcopy(context["run_context"]),
        **raw_result,
    }

    trace = run.as_trace()
    trace["result"] = copy.deepcopy(result)
    current_trace["summer"] = trace
    write_json_atomic(result_path, result)
    write_json_atomic(trace_path, current_trace)
    return {
        "run_id": run_path.name,
        "status": "overwritten" if result_existed else "created",
        "trial_id": latest_trial_id,
        "result_path": str(result_path),
        "trace_path": str(trace_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Summer Agent over existing output runs and publish their "
            "summaries for query_tuning_summaries."
        )
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--agent-config", default=DEFAULT_AGENT_CONFIG)
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        help=(
            "Only migrate this exact run name or unambiguous prefix. Repeat for multiple "
            "runs; omit to migrate every run under output-root."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing canonical summer/summer_result.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable streamed Agent events while retaining one result line per run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.agent_config).expanduser().resolve()
    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"agent config does not exist: {config_path}")
        config = load_json(config_path)
        config["stream_agent_events"] = not args.quiet
        if args.dates:
            run_dirs = []
            seen: set[Path] = set()
            for date in args.dates:
                run_dir = resolve_run_dir(output_root, date)
                if run_dir not in seen:
                    run_dirs.append(run_dir)
                    seen.add(run_dir)
        else:
            run_dirs = discover_run_dirs(output_root)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    outcomes: list[dict[str, Any]] = []
    failures = 0
    for run_dir in run_dirs:
        try:
            outcome = backfill_run(run_dir, config, overwrite=args.overwrite)
        except (AgentError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            failures += 1
            error = (
                exc.as_dict()
                if isinstance(exc, AgentResponseError)
                else {"error": f"{type(exc).__name__}: {exc}"}
            )
            outcome = {"run_id": run_dir.name, "status": "failed", **error}
        outcomes.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False))

    try:
        index_path = rebuild_summary_index(output_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error rebuilding summary index: {exc}", file=sys.stderr)
        return 2

    totals = {
        status: sum(row.get("status") == status for row in outcomes)
        for status in ("created", "overwritten", "skipped", "failed")
    }
    print(
        json.dumps(
            {
                "summary_index": str(index_path),
                "runs": len(outcomes),
                **totals,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
