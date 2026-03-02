#!/bin/bash
# Setup script for Oracle Cloud Always Free ARM instance.
# Run on a fresh Ubuntu 22.04 ARM instance.

set -euo pipefail

echo "=== Setting up arxiv-context-feed on Oracle Cloud ==="

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    cmake build-essential \
    git curl wget

# Install GitHub CLI
if ! command -v gh &> /dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
        sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update && sudo apt-get install -y gh
fi

# Clone repo
REPO_DIR="$HOME/arxiv-context-feed"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/YOUR_USERNAME/arxiv-context-feed.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install package with local-llm extras
pip install --upgrade pip
pip install -e ".[local-llm,reddit]"

# Download model
bash scripts/download_model.sh

# Create log directory
mkdir -p "$HOME/arxiv-feed-logs"

# Install systemd service
sudo cp deploy/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Wait for server
echo "Waiting for llama.cpp server to start..."
for i in $(seq 1 60); do
    if curl -s -o /dev/null http://127.0.0.1:8080/health; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Install crontab
crontab deploy/crontab
echo "Crontab installed"

# Create .env file template
if [ ! -f "$REPO_DIR/.env" ]; then
    cat > "$REPO_DIR/.env" << 'EOF'
# Oracle Cloud Environment Variables
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=not-needed
JUDGE_PROVIDER=local

# GitHub (for creating issues)
GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE

# Reddit (optional)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USERNAME=
REDDIT_PASSWORD=

# Citation API keys (comma-separated for team, optional)
OPENALEX_API_KEYS=
EOF
    echo "Created .env template - edit with your credentials"
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit .env with your credentials"
echo "  2. Run: gh auth login"
echo "  3. Test: contextual-arxiv-feed run-analyze --dry-run"
echo "  4. Check health: bash deploy/healthcheck.sh"
