#!/usr/bin/env bash

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="$SCRIPT_DIR/runpod_variables.template.env"
TARGET_FILE="$SCRIPT_DIR/runpod_variables.env"

echo -e "${BLUE}=== Safetensors RunPod Environment Generator ===${NC}"

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo -e "${RED}Error: Template file '$TEMPLATE_FILE' not found!${NC}"
    exit 1
fi

# Check if target file already exists
if [ -f "$TARGET_FILE" ]; then
    echo -e "${YELLOW}Warning: '$TARGET_FILE' already exists.${NC}"
    read -rp "Do you want to overwrite it? All existing variables will be lost! (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Aborted. No changes were made.${NC}"
        exit 0
    fi
fi

# Generate secure random strings
echo -e "${BLUE}Generating secure API keys and passwords...${NC}"

# Use openssl if available, fallback to python
if command -v openssl >/dev/null 2>&1; then
    API_KEY=$(openssl rand -hex 32)
    PANEL_PASS=$(openssl rand -hex 12)
else
    API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    PANEL_PASS=$(python3 -c "import secrets; print(secrets.token_hex(12))")
fi

PANEL_USER="admin"

echo -e "${GREEN}API Key and Password successfully generated.${NC}"

# Create runpod_variables.env by replacing placeholders.
# Pass values as arguments to Python to avoid shell injection issues.
python3 - "$TEMPLATE_FILE" "$TARGET_FILE" "$API_KEY" "$PANEL_USER" "$PANEL_PASS" <<'PY'
import sys

template_path, target_path, api_key, panel_user, panel_pass = sys.argv[1:6]

with open(template_path, 'r') as f:
    content = f.read()

content = content.replace('CHANGE_ME_GENERATE_WITH_OPENSSL_RAND_HEX_32', api_key)
content = content.replace('CHANGE_ME_PANEL_USER', panel_user)
content = content.replace('CHANGE_ME_STRONG_PANEL_PASSWORD', panel_pass)

with open(target_path, 'w') as f:
    f.write(content)
PY

echo -e "${GREEN}Successfully created '$TARGET_FILE'!${NC}"
echo -e "----------------------------------------"
echo -e "${BLUE}Generated Credentials Summary:${NC}"
echo -e "  ${YELLOW}SAFETENSORS_API_KEY:${NC}        $API_KEY"
echo -e "  ${YELLOW}SAFETENSORS_PANEL_USER:${NC}     $PANEL_USER"
echo -e "  ${YELLOW}SAFETENSORS_PANEL_PASSWORD:${NC} $PANEL_PASS"
echo -e "----------------------------------------"
echo -e "${BLUE}What to do next:${NC}"
echo -e "1. Open ${GREEN}$TARGET_FILE${NC} and fill in your ${YELLOW}HF_TOKEN${NC} (if using private/gated HF models)."
echo -e "2. Copy the variables from ${GREEN}$TARGET_FILE${NC} to your RunPod Environment Variables interface."
echo -e "3. Since ${GREEN}$TARGET_FILE${NC} is listed in ${GREEN}.gitignore${NC}, it will not be committed to Git."
echo -e "----------------------------------------"
