#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PARAMETERS=${1:?Usage: run_verl.sh PARAMETERS.json STAGE TRIAL_ID UPDATES}
STAGE=${2:?Usage: run_verl.sh PARAMETERS.json STAGE TRIAL_ID UPDATES}
TRIAL_ID=${3:?Usage: run_verl.sh PARAMETERS.json STAGE TRIAL_ID UPDATES}
UPDATES=${4:?Usage: run_verl.sh PARAMETERS.json STAGE TRIAL_ID UPDATES}
export PLATFORM=${PLATFORM:-V5000}
export OUTPUT_PATH=${OUTPUT_PATH:-${SCRIPT_DIR}/output}
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/train/setup_environment.sh"

exec python3 "${SCRIPT_DIR}/trial_cli.py" \
    --parameters "${PARAMETERS}" \
    --stage "${STAGE}" \
    --trial-id "${TRIAL_ID}" \
    --updates "${UPDATES}" \
    --agent-config "${AGENT_CONFIG_FILE:-${SCRIPT_DIR}/config/agent_config.json}"
