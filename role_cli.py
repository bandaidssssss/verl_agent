#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentSet
from config_utils import load_json, write_json


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one verl tuning agent role")
    parser.add_argument(
        "--role",
        choices=["proposal", "feasibility", "diagnosis", "train_health", "summer"],
        required=True,
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--output")
    parser.add_argument("--trace-output", help="Optional full conversation/tool trace JSON")
    parser.add_argument("--agent-config", default=str(ROOT / "config" / "agent_config.json"))
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    agent_config = load_json(args.agent_config)
    output_dir = Path(os.getenv("OUTPUT_PATH", str(agent_config.get("output_dir", "output"))))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    agents = AgentSet(
        ROOT,
        "rules" if args.rules_only else "llm",
        agent_config,
        output_dir / "trials.jsonl",
    )
    context = load_json(args.context)
    if args.role == "proposal":
        run = agents.propose(context)
    elif args.role == "feasibility":
        run = agents.review(context)
    elif args.role == "diagnosis":
        run = agents.diagnose(context)
    elif args.role == "train_health":
        run = agents.assess_health(context)
    else:
        run = agents.summarize(context)
    result = run.result
    if args.output:
        write_json(args.output, result)
    if args.trace_output:
        write_json(args.trace_output, run.as_trace())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
