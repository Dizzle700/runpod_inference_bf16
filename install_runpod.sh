#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_ROOT="${SAFETENSORS_VOLUME_ROOT:-${GGUF_VOLUME_ROOT:-/workspace}}"
VENV_DIR="${SAFETENSORS_VENV_DIR:-$VOLUME_ROOT/.venvs/safetensors-rig}"
PYTHON_EXE="${PYTHON_EXE:-python3}"

info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
success() { printf '\033[0;32m%s\033[0m\n' "$*"; }
error() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }
trap 'error "Installation failed at line $LINENO (exit $?)."' ERR

if [[ ! -d "$VOLUME_ROOT" ]]; then
    error "Persistent volume root does not exist: $VOLUME_ROOT"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
info "Installing Python/runtime prerequisites..."
apt-get update
apt-get install -y --no-install-recommends ca-certificates git python3-venv

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    info "Creating the persistent Python environment..."
    "$PYTHON_EXE" -m venv "$VENV_DIR"
fi

info "Installing vLLM and control-panel dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p \
    "$VOLUME_ROOT/models/safetensors" \
    "$VOLUME_ROOT/.state/safetensors-rig" \
    "$VOLUME_ROOT/logs/safetensors-rig" \
    "$VOLUME_ROOT/.hf/hub"
touch "$VENV_DIR/.safetensors-rig-installed"

success "Safetensors Rig is installed in $VENV_DIR"
