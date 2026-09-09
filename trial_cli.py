#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import load_json
from distributed_launch import prepare_cluster


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one monitored verl trial")
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parameters = load_json(args.parameters)
    if not prepare_cluster(parameters, dry_run=args.dry_run):
        return 0
    from runner import run_trial

    result = run_trial(
        parameters,
        load_json(args.agent_config),
        args.trial_id,
        args.stage,
        args.updates,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
