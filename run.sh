#!/bin/bash
# run.sh — AI Trader launcher
#
# Usage:
#   ./run.sh          → defaults to paper mode
#   ./run.sh paper    → Yahoo Finance data, paper fills, 24/7
#   ./run.sh sim      → Zerodha data, paper fills, market hours
#   ./run.sh live     → Zerodha data, real orders, market hours

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Default mode is paper — safe, no credentials needed
MODE="${1:-paper}"

# Validate mode
case "$MODE" in
    paper|sim|live)
        ;;
    *)
        echo "Usage: $0 [paper|sim|live]"
        echo "  paper — Yahoo Finance, paper fills, 24/7 (default)"
        echo "  sim   — Zerodha data, paper fills, market hours"
        echo "  live  — Zerodha data, real orders, market hours"
        exit 1
        ;;
esac

echo "Starting AI Trader in '$MODE' mode..."
python3 main.py "$MODE"
