#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PLATFORM=${PLATFORM:-C550}
export MAX_TRIALS=${MAX_TRIALS:-10}
RUN_TIME=$(date +%m%d_%H%M_%Y)
export OUTPUT_PATH=${OUTPUT_PATH:-${SCRIPT_DIR}/output}/${RUN_TIME}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_PATH}"
RUN_LOG="${OUTPUT_PATH}/run_circle.log"
LAUNCH_RANK=${NODE_RANK:-${POD_RANK:-${SLURM_NODEID:-${RANK:-0}}}}
if [[ "$LAUNCH_RANK" != "0" ]]; then
    RUN_LOG="${OUTPUT_PATH}/worker_${LAUNCH_RANK}.log"
fi

source "${SCRIPT_DIR}/train/setup_environment.sh"

python3 "${SCRIPT_DIR}/run_agent.py" \
    --base-config "${BASE_CONFIG_FILE:-${SCRIPT_DIR}/config/base_parameters.json}" \
    --agent-config "${AGENT_CONFIG_FILE:-${SCRIPT_DIR}/config/agent_config.json}" \
    --max-trials "${MAX_TRIALS}" \
    "$@" 2>&1 | tee -a "${RUN_LOG}"
