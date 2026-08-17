#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import write_json
from metrics import analyze_trial



# warmup-updates 前面几步不统计throughput，reward之类的，前几步可能不稳定
# stability-window-size        STABILITY_QUERY_METRICS按照多大的窗口进行取mean

#reward-window 这个是江老师推荐几个窗口均值，判断达到 reward-thresholds 的cumulative_time_s，step，和cumulative_tokens

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a verl train log and optional GPU samples")
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu-samples")
    parser.add_argument("--output")
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--reward-window", type=int, default=5)
    parser.add_argument("--stability-window-size", type=int, default=5)
    parser.add_argument("--reward-thresholds", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    args = parser.parse_args()

    result = analyze_trial(
        args.log,
        args.gpu_samples,
        warmup_updates=args.warmup_updates,
        reward_window=args.reward_window,
        stability_window_size=args.stability_window_size,
        reward_thresholds=args.reward_thresholds,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
