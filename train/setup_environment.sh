#!/usr/bin/env bash
# Sourced by both launchers, before Ray starts on each node.
SETUP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PLATFORM=${PLATFORM:-V5000}
if [[ -z "${VERL_ENV_SCRIPT:-}" ]]; then
    case "$(printf '%s' "$PLATFORM" | tr '[:lower:]' '[:upper:]')" in
        V5000) export VERL_ENV_SCRIPT="$SETUP_ROOT/train/env_V5000.sh" ;;
        C550|METAX) export VERL_ENV_SCRIPT="$SETUP_ROOT/train/env_C550.sh" ;;
        A100|NVIDIA|CUDA) export VERL_ENV_SCRIPT="$SETUP_ROOT/train/env_NVIDIA.sh" ;;
        *) echo "Unsupported PLATFORM=$PLATFORM. Set VERL_ENV_SCRIPT explicitly." >&2; exit 2 ;;
    esac
fi
source "$VERL_ENV_SCRIPT"
