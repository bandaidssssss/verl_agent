#!/usr/bin/env bash
# Short master entry point. Use --existing when Ray is already running.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/env.sh" ]]; then
    source "$PROJECT_DIR/env.sh"
fi
export PLATFORM=${PLATFORM:-C550}
export MAX_TRIALS=${MAX_TRIALS:-10}
export NODE_RANK=0
export RAY_CLUSTER_MODE=managed
if [[ "${1:-}" == "--existing" ]]; then
    export RAY_CLUSTER_MODE=existing
    export RAY_ADDRESS=auto
    shift
fi
exec bash "$PROJECT_DIR/run_circle.sh" "$@"
