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
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

: "${SAFETENSORS_API_KEY:?Set SAFETENSORS_API_KEY as a RunPod secret}"
: "${SAFETENSORS_PANEL_USER:?Set SAFETENSORS_PANEL_USER as a RunPod secret}"
: "${SAFETENSORS_PANEL_PASSWORD:?Set SAFETENSORS_PANEL_PASSWORD as a RunPod secret}"

VLLM_CONSTRAINTS="${SAFETENSORS_VLLM_CONSTRAINTS:-$SCRIPT_DIR/constraints-vllm-afnext.txt}"
INSTALL_FINGERPRINT_FILE="$VENV_DIR/.safetensors-rig-requirements.sha256"
CURRENT_FINGERPRINT="$(
    {
        sha256sum "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/requirements-vllm.txt"
        if [[ -f "$VLLM_CONSTRAINTS" ]]; then
            sha256sum "$VLLM_CONSTRAINTS"
        fi
        printf '%s\n' \
            "system_site_packages=${SAFETENSORS_VENV_SYSTEM_SITE_PACKAGES:-1}" \
            "install_vllm=${SAFETENSORS_INSTALL_VLLM:-auto}" \
            "expected_vllm=${SAFETENSORS_EXPECTED_VLLM_VERSION:-0.20.0}"
    } | sha256sum | cut -d' ' -f1
)"
INSTALLED_FINGERPRINT=""
if [[ -f "$INSTALL_FINGERPRINT_FILE" ]]; then
    INSTALLED_FINGERPRINT="$(<"$INSTALL_FINGERPRINT_FILE")"
fi

NEEDS_INSTALL=0
if [[ ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/.safetensors-rig-installed" || "$CURRENT_FINGERPRINT" != "$INSTALLED_FINGERPRINT" ]]; then
    NEEDS_INSTALL=1
fi

if [[ "$NEEDS_INSTALL" == "1" && "${SAFETENSORS_SKIP_INSTALL:-0}" == "1" ]]; then
    echo "Installation is stale or missing, but SAFETENSORS_SKIP_INSTALL=1." >&2
    exit 1
fi
if [[ "$NEEDS_INSTALL" == "1" ]]; then
    bash "$SCRIPT_DIR/install_runpod.sh"
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/app.py"
