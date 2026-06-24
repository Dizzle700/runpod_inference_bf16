#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_ROOT="${SAFETENSORS_VOLUME_ROOT:-${GGUF_VOLUME_ROOT:-/workspace}}"
VENV_DIR="${SAFETENSORS_VENV_DIR:-$VOLUME_ROOT/.venvs/safetensors-rig}"
LOG_FILE="${SAFETENSORS_STARTUP_LOG:-$VOLUME_ROOT/logs/safetensors-rig/startup.log}"

mkdir -p "$(dirname -- "$LOG_FILE")"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=== Safetensors Rig startup: $(date --iso-8601=seconds) ==="

export SAFETENSORS_VOLUME_ROOT="$VOLUME_ROOT"
export SAFETENSORS_MODELS_DIR="${SAFETENSORS_MODELS_DIR:-$VOLUME_ROOT/models/safetensors}"
export SAFETENSORS_STATE_DIR="${SAFETENSORS_STATE_DIR:-$VOLUME_ROOT/.state/safetensors-rig}"
export SAFETENSORS_LOG_DIR="${SAFETENSORS_LOG_DIR:-$VOLUME_ROOT/logs/safetensors-rig}"
export SAFETENSORS_API_KEY="${SAFETENSORS_API_KEY:-${GGUF_API_KEY:-}}"
export SAFETENSORS_PANEL_USER="${SAFETENSORS_PANEL_USER:-${GGUF_PANEL_USER:-}}"
export SAFETENSORS_PANEL_PASSWORD="${SAFETENSORS_PANEL_PASSWORD:-${GGUF_PANEL_PASSWORD:-}}"
export VLLM_PYTHON="${VLLM_PYTHON:-$VENV_DIR/bin/python}"
export HF_HOME="${HF_HOME:-$VOLUME_ROOT/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VOLUME_ROOT/.hf/hub}"

: "${SAFETENSORS_API_KEY:?Set SAFETENSORS_API_KEY as a RunPod secret}"
: "${SAFETENSORS_PANEL_USER:?Set SAFETENSORS_PANEL_USER as a RunPod secret}"
: "${SAFETENSORS_PANEL_PASSWORD:?Set SAFETENSORS_PANEL_PASSWORD as a RunPod secret}"

if [[ "${SAFETENSORS_SKIP_INSTALL:-0}" != "1" || ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/.safetensors-rig-installed" ]]; then
    bash "$SCRIPT_DIR/install_runpod.sh"
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/app.py"
