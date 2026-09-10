#!/usr/bin/env bash
# Short worker entry point; use the platform's rank for more than two nodes.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
export PLATFORM=${PLATFORM:-C550}
export NODE_RANK=${POD_RANK:-${SLURM_NODEID:-1}}
if [[ "$NODE_RANK" == "0" ]]; then
    echo "This is rank 0. Use start/m.sh on master and start/w.sh on worker." >&2
    exit 2
fi
export RAY_CLUSTER_MODE=managed
exec bash "$PROJECT_DIR/run_circle.sh" "$@"
