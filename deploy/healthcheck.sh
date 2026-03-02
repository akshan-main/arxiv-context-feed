#!/bin/bash
# Health check script for Oracle Cloud deployment.
# Run manually or via monitoring to verify all components are running.

set -euo pipefail

echo "=== arxiv-context-feed Health Check ==="
echo "Date: $(date -u)"
echo ""

# Check llama.cpp server
echo "--- llama.cpp Server ---"
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/health | grep -q "200"; then
    echo "Status: HEALTHY"
else
    echo "Status: DOWN"
fi

# Check systemd service
echo ""
echo "--- Systemd Service ---"
if systemctl is-active --quiet llama-server; then
    echo "Status: ACTIVE"
    systemctl show llama-server --property=ActiveEnterTimestamp --no-pager
else
    echo "Status: INACTIVE"
fi

# Check crontab
echo ""
echo "--- Crontab ---"
CRON_COUNT=$(crontab -l 2>/dev/null | grep -c "arxiv" || true)
echo "arXiv cron jobs: ${CRON_COUNT}"

# Check disk usage
echo ""
echo "--- Disk Usage ---"
df -h / | tail -1

# Check RAM
echo ""
echo "--- RAM Usage ---"
free -h | head -2

# Check model file
echo ""
echo "--- Model File ---"
MODEL_PATH="$HOME/.cache/arxiv-feed/models/qwen2.5-14b-instruct-q4_k_m.gguf"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "Model: ${SIZE} at ${MODEL_PATH}"
else
    echo "Model: NOT FOUND"
fi

# Check recent logs
echo ""
echo "--- Recent Logs ---"
LOG_DIR="$HOME/arxiv-feed-logs"
if [ -d "$LOG_DIR" ]; then
    LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Latest log: $LATEST"
        echo "Last 5 lines:"
        tail -5 "$LATEST"
    else
        echo "No log files found"
    fi
else
    echo "Log directory not found: $LOG_DIR"
fi

echo ""
echo "=== Health Check Complete ==="
