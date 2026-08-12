#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import load_json
from orchestrator import TuningOrchestrator


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-aware verl GRPO tuning agent")
    parser.add_argument("--base-config", default=str(ROOT / "config" / "base_parameters.json"))
    parser.add_argument("--agent-config", default=str(ROOT / "config" / "agent_config.json"))
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument(
        "--start-stage",
        choices=["auto", "hardware_tuning", "stability_tuning"],
        help="Override the configured initial stage; stability_tuning skips hardware search",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rules-only", action="store_true", help="Skip LLM calls; useful for baseline and tests")
    args = parser.parse_args()

    base_parameters = load_json(args.base_config)
    agent_config = load_json(args.agent_config)
    if os.getenv("PLATFORM"):
        agent_config["platform"] = os.environ["PLATFORM"]
    if os.getenv("VERL_ROOT"):
        agent_config["verl_root"] = os.environ["VERL_ROOT"]
    if os.getenv("OUTPUT_PATH"):
        agent_config["output_dir"] = os.environ["OUTPUT_PATH"]
    if os.getenv("VERL_ENV_SCRIPT"):
        agent_config["environment_script"] = os.environ["VERL_ENV_SCRIPT"]
    if os.getenv("START_STAGE"):
        agent_config["start_stage"] = os.environ["START_STAGE"]
    if args.start_stage:
        agent_config["start_stage"] = args.start_stage
    if args.rules_only:
        agent_config["agent_mode"] = "rules"
    orchestrator = TuningOrchestrator(ROOT, base_parameters, agent_config)
    reports = orchestrator.run(max_trials=args.max_trials, dry_run=args.dry_run)
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
