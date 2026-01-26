#!/bin/bash
# Setup script for Claude Code Cloud environment
# Enhanced for Cloud Web optimization

set -e

echo "=== AI Playground Environment Setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv
echo "uv version: $(uv --version)"

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

echo "gh version: $(gh --version | head -1)"

# Change to project directory
cd "$CLAUDE_PROJECT_DIR"

# Sync project dependencies
if [ -f "pyproject.toml" ]; then
    echo "Syncing project dependencies..."
    uv sync --quiet
fi

# Create directories if they don't exist
mkdir -p tools shared data data/samples outputs .cache/market-data

# =============================================================================
# Pre-download NLP models for pdf-skill (optional, runs once)
# =============================================================================
if command -v python3 &> /dev/null; then
    echo "Checking NLP models..."
    python3 -c "
try:
    import spacy
    try:
        spacy.load('en_core_web_sm')
        print('  spacy model: OK (en_core_web_sm)')
    except OSError:
        print('  spacy model: Downloading en_core_web_sm...')
        import subprocess
        subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'],
                      capture_output=True)
        print('  spacy model: Installed')
except ImportError:
    print('  spacy: Not installed (pdf-skill will install if needed)')
" 2>/dev/null || echo "  NLP model check skipped"
fi

# =============================================================================
# Verify data source connectivity (informational)
# =============================================================================
echo ""
echo "Checking data source connectivity..."

check_url() {
    local name=$1
    local url=$2
    if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
        echo "  $name: OK"
    else
        echo "  $name: BLOCKED (need Full Internet)"
    fi
}

check_url "Stooq" "https://stooq.pl"
check_url "NBP" "https://api.nbp.pl"
check_url "Yahoo" "https://query1.finance.yahoo.com"
check_url "GitHub" "https://api.github.com"
check_url "PyPI" "https://pypi.org"

# =============================================================================
# Setup marketplace (for Cloud VM)
# =============================================================================
echo ""
echo "Setting up plugin marketplaces..."

if command -v claude &> /dev/null; then
    # Add qute-marketplace from GitHub if not already added
    claude plugin marketplace add https://github.com/tomlupo/qute-marketplace.git 2>/dev/null || true
    echo "  qute-marketplace: configured"
else
    echo "  Claude CLI not found, skipping marketplace setup"
fi

# =============================================================================
# Environment summary
# =============================================================================
echo ""
echo "=== Environment Ready ==="
echo "  Project: $CLAUDE_PROJECT_DIR"
echo "  Python: $(python3 --version 2>/dev/null || echo 'N/A')"
echo "  uv: $(uv --version)"
echo "  gh: $(gh --version 2>/dev/null | head -1 || echo 'N/A')"
echo ""
echo "Tips:"
echo "  - Use 'uv run' to run Python scripts"
echo "  - Use '/qrd' skill for research specifications"
echo "  - Use '&' prefix for parallel task execution"
echo "  - Sample data available in data/samples/"

exit 0
