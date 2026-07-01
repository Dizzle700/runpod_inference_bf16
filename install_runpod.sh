#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOLUME_ROOT="${SAFETENSORS_VOLUME_ROOT:-${GGUF_VOLUME_ROOT:-/workspace}}"
VENV_DIR="${SAFETENSORS_VENV_DIR:-$VOLUME_ROOT/.venvs/safetensors-rig}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
VENV_SYSTEM_SITE_PACKAGES="${SAFETENSORS_VENV_SYSTEM_SITE_PACKAGES:-1}"
INSTALL_VLLM="${SAFETENSORS_INSTALL_VLLM:-auto}"
EXPECTED_VLLM_VERSION="${SAFETENSORS_EXPECTED_VLLM_VERSION:-0.11.0}"
VLLM_CONSTRAINTS="${SAFETENSORS_VLLM_CONSTRAINTS:-$SCRIPT_DIR/constraints-vllm-torch280.txt}"
UPGRADE_PIP="${SAFETENSORS_UPGRADE_PIP:-0}"

info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
success() { printf '\033[0;32m%s\033[0m\n' "$*"; }
error() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }
trap 'error "Installation failed at line $LINENO (exit $?)."' ERR

enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

package_available() {
    "$VENV_DIR/bin/python" - "$1" <<'PY'
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

install_vllm() {
    local pip_args=(-r "$SCRIPT_DIR/requirements-vllm.txt")
    if [[ -f "$VLLM_CONSTRAINTS" ]]; then
        pip_args=(-c "$VLLM_CONSTRAINTS" "${pip_args[@]}")
    fi
    "$VENV_DIR/bin/python" -m pip install "${pip_args[@]}"
}

vllm_expected_version() {
    "$VENV_DIR/bin/python" - "$EXPECTED_VLLM_VERSION" <<'PY'
import importlib.metadata
import sys

try:
    installed = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0 if installed == sys.argv[1] else 1)
PY
}

if [[ ! -d "$VOLUME_ROOT" ]]; then
    error "Persistent volume root does not exist: $VOLUME_ROOT"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
info "Installing Python/runtime prerequisites..."
apt-get update
apt-get install -y --no-install-recommends ca-certificates git python3-venv

venv_args=("$PYTHON_EXE" -m venv)
if enabled "$VENV_SYSTEM_SITE_PACKAGES"; then
    venv_args+=(--system-site-packages)
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    info "Creating the persistent Python environment..."
    "${venv_args[@]}" "$VENV_DIR"
elif enabled "$VENV_SYSTEM_SITE_PACKAGES" && grep -Eiq '^include-system-site-packages *= *false' "$VENV_DIR/pyvenv.cfg"; then
    info "Reconfiguring the Python environment to see RunPod template packages..."
    "${venv_args[@]}" "$VENV_DIR"
fi

if enabled "$UPGRADE_PIP"; then
    info "Upgrading pip..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

info "Installing control-panel dependencies..."
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

case "${INSTALL_VLLM,,}" in
    auto|"")
        if vllm_expected_version; then
            info "Using existing vLLM $EXPECTED_VLLM_VERSION; skipping vLLM install."
        else
            info "vLLM $EXPECTED_VLLM_VERSION is not installed; installing pinned vLLM..."
            install_vllm
        fi
        ;;
    1|true|yes|on)
        info "Installing/updating vLLM..."
        install_vllm
        ;;
    0|false|no|off|skip)
        info "Skipping vLLM installation because SAFETENSORS_INSTALL_VLLM=$INSTALL_VLLM."
        ;;
    *)
        error "Unsupported SAFETENSORS_INSTALL_VLLM value: $INSTALL_VLLM (use auto, 1, or 0)"
        exit 1
        ;;
esac

if ! package_available vllm; then
    error "vLLM is not importable. Set SAFETENSORS_INSTALL_VLLM=1/auto or use a RunPod template with vLLM preinstalled."
    exit 1
fi

mkdir -p \
    "$VOLUME_ROOT/models/safetensors" \
    "$VOLUME_ROOT/.state/safetensors-rig" \
    "$VOLUME_ROOT/logs/safetensors-rig" \
    "$VOLUME_ROOT/.hf/hub"
touch "$VENV_DIR/.safetensors-rig-installed"

success "Safetensors Rig is installed in $VENV_DIR"
