
"""
main.py — trading bot entry with mode switch

Modes:
paper -> Yahoo data + paper trading
sim   -> Zerodha data + paper trading
live  -> Zerodha data + live trading
"""

import sys
from config import config
from logger import get_logger
from kite_client import kite_client
from data_fetcher import data_fetcher

logger = get_logger("Main")

# ---------------- Mode switch ----------------

mode = "paper"

if len(sys.argv) > 1:
    mode = sys.argv[1]

if mode == "paper":
    config.data.provider = "yahoo"
    config.trading.paper_trading = True

elif mode == "sim":
    config.data.provider = "zerodha"
    config.trading.paper_trading = True

elif mode == "live":
    config.data.provider = "zerodha"
    config.trading.paper_trading = False

logger.info("Running mode: %s", mode)
logger.info("Data provider: %s", config.data.provider)
logger.info("Paper trading: %s", config.trading.paper_trading)

# ---------------- Initialise Zerodha if needed ----------------

if config.data.provider == "zerodha":
    try:
        kite_client.initialise()
    except Exception as e:
        logger.warning("Kite init failed: %s", e)

# ---------------- Start bot ----------------

from scheduler import start_scheduler

start_scheduler()
