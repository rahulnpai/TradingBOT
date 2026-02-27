"""
logger.py — Structured, colored, rotating logger for AI Trader.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


# ---------------------------------------------------------------------------
# Colored console formatter
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    logging.DEBUG:    Fore.CYAN,
    logging.INFO:     Fore.GREEN,
    logging.WARNING:  Fore.YELLOW,
    logging.ERROR:    Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class ColoredFormatter(logging.Formatter):
    FMT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        formatter = logging.Formatter(
            fmt=f"{color}{self.FMT}{Style.RESET_ALL}",
            datefmt=self.DATE_FMT,
        )
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    FMT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    log_file: str = "logs/ai_trader.log",
    level: str = "INFO",
) -> logging.Logger:
    """Return a named logger.  Loggers are cached so init only happens once."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ---- console handler -------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)

    # ---- rotating file handler -------------------------------------------
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=7,
        encoding="utf-8",
    )
    fh.setFormatter(PlainFormatter())
    logger.addHandler(fh)

    _loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Trade-specific event logger (writes CSV rows for audit trail)
# ---------------------------------------------------------------------------

class TradeLogger:
    """Appends trade events to a daily CSV for audit / back-analysis."""

    def __init__(self, trade_log_dir: str = "logs") -> None:
        os.makedirs(trade_log_dir, exist_ok=True)
        self._dir = trade_log_dir
        self._log = get_logger("TradeLogger")

    def _get_path(self) -> str:
        from datetime import date
        return os.path.join(self._dir, f"trades_{date.today().isoformat()}.csv")

    def log_trade(self, event: dict) -> None:
        import csv
        path = self._get_path()
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=event.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(event)
        self._log.info(
            "TRADE | %s | %s | qty=%s | price=%.2f | sl=%.2f | tgt=%.2f",
            event.get("action", "?"),
            event.get("symbol", "?"),
            event.get("quantity", "?"),
            float(event.get("entry_price", 0)),
            float(event.get("stoploss", 0)),
            float(event.get("target", 0)),
        )
