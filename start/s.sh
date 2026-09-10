#!/usr/bin/env bash
# Nanhu unified launcher: run this same command once in each allocated Pod.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

# Nanhu WORLD_SIZE is the node count. Prefer platform metadata over stale
# NODE_RANK values left over from a previous manual launch.
LAUNCH_NODES=${WORLD_SIZE:-${NNODES:-${SLURM_NNODES:-}}}
LAUNCH_RANK=${POD_RANK:-${RANK:-${SLURM_NODEID:-${NODE_RANK:-}}}}
if [[ -n "$LAUNCH_NODES" && ! "$LAUNCH_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid platform node count: $LAUNCH_NODES" >&2
    exit 2
fi
if [[ -n "$LAUNCH_RANK" && ! "$LAUNCH_RANK" =~ ^[0-9]+$ ]]; then
    echo "Invalid platform node rank: $LAUNCH_RANK" >&2
    exit 2
fi
if [[ -n "$LAUNCH_RANK" ]]; then
    LAUNCH_RANK=$((10#$LAUNCH_RANK))
fi
if [[ -n "$LAUNCH_NODES" && -n "$LAUNCH_RANK" ]] && (( LAUNCH_RANK >= LAUNCH_NODES )); then
    echo "Node rank must be smaller than node count." >&2
    exit 2
fi

# Credentials are needed only by the driver, not the worker launcher.
if [[ "${LAUNCH_RANK:-0}" == "0" && -f "$PROJECT_DIR/env.sh" ]]; then
    source "$PROJECT_DIR/env.sh"
fi
[[ -z "$LAUNCH_NODES" ]] || export NNODES="$LAUNCH_NODES"
[[ -z "$LAUNCH_RANK" ]] || export NODE_RANK="$LAUNCH_RANK"
export PLATFORM=${PLATFORM:-C550}
export MAX_TRIALS=${MAX_TRIALS:-10}
export RAY_START_TIMEOUT=${RAY_START_TIMEOUT:-${START_TIMEOUT:-600}}
export RAY_CLUSTER_MODE=managed
if [[ "${1:-}" == "--existing" ]]; then
    export RAY_CLUSTER_MODE=existing
    export RAY_ADDRESS=auto
    shift
fi
exec bash "$PROJECT_DIR/run_circle.sh" "$@"
