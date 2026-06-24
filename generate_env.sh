#!/usr/bin/env bash

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TEMPLATE_FILE="runpod_variables.template.env"
TARGET_FILE="runpod_variables.env"

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
    PANEL_PASS=$(openssl rand -base64 15 | tr -d '/+=') # Keep it clean and URL/shell safe
else
    API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    PANEL_PASS=$(python3 -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16)))")
fi

PANEL_USER="admin"

echo -e "${GREEN}API Key and Password successfully generated.${NC}"

# Create runpod_variables.env by replacing placeholders
python3 -c "
with open('$TEMPLATE_FILE', 'r') as f:
    content = f.read()

content = content.replace('CHANGE_ME_GENERATE_WITH_OPENSSL_RAND_HEX_32', '$API_KEY')
content = content.replace('CHANGE_ME_PANEL_USER', '$PANEL_USER')
content = content.replace('CHANGE_ME_STRONG_PANEL_PASSWORD', '$PANEL_PASS')

with open('$TARGET_FILE', 'w') as f:
    f.write(content)
"

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
