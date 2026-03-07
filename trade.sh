#!/bin/bash
#
# trade.sh — helper launcher for AI Trader
#
# Usage:
#   ./trade.sh [watchlist] [mode]
#
# Watchlists:
#   core     → SBIN, ADANIENT
#   nifty50  → All NIFTY 50 stocks
#
# Modes:
#   paper (default)
#   sim
#   live
#
# Examples:
#   ./trade.sh core
#   ./trade.sh nifty50
#   ./trade.sh core sim
#   ./trade.sh nifty50 sim
#   ./trade.sh core live
#   ./trade.sh nifty50 live

WATCHLIST="${1:-core}"
MODE="${2:-paper}"

WATCHLIST_MODE=$WATCHLIST ./run.sh $MODE
