#!/bin/bash
# Setup script for Claude Code Cloud environment

set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv
uv --version

# Install gh CLI if not present (for gists)
if ! command -v gh &> /dev/null; then
    echo "Installing GitHub CLI..."
    (type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
    && sudo mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y
fi

# Sync project dependencies
cd "$CLAUDE_PROJECT_DIR"
if [ -f "pyproject.toml" ]; then
    echo "Syncing project dependencies..."
    uv sync --quiet
fi

# Create directories if they don't exist
mkdir -p tools shared data outputs

echo "Environment ready!"
exit 0
