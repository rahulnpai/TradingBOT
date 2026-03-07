"""
market_analytics.py — Market status, regime detection, and sector rotation.

Provides:
  - NIFTY / BANKNIFTY / VIX status bar (refreshed every 60 s)
  - Market regime: TRENDING | SIDEWAYS | VOLATILE  (from NIFTY daily data)
  - Sector performance scanner with confidence-boost helper

All network calls run in a daemon background thread so the main trading loop
and Rich dashboard are never blocked.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from logger import get_logger

log = get_logger("MarketAnalytics")

# ---------------------------------------------------------------------------
# Sector definitions
# ---------------------------------------------------------------------------

SECTOR_MAP: Dict[str, List[str]] = {
    "BANKING":  ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                 "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV"],
    "IT":       ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "FMCG":     ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA", "TATACONSUM"],
    "PHARMA":   ["SUNPHARMA", "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP"],
    "METALS":   ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "AUTO":     ["MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "M&M"],
    "ENERGY":   ["RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC"],
}

# Reverse lookup: symbol → sector name
SYMBOL_TO_SECTOR: Dict[str, str] = {
    sym: sector
    for sector, syms in SECTOR_MAP.items()
    for sym in syms
}

# ---------------------------------------------------------------------------
# Shared mutable state — always access under _lock
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# index name → (price, pct_change)
_index_status: Dict[str, Tuple[float, float]] = {}

# "TRENDING" | "SIDEWAYS" | "VOLATILE" | "UNKNOWN"
_regime: str = "UNKNOWN"

# sector → average % change today
_sector_perf: Dict[str, float] = {}

# top-N sector names that have positive performance
_strong_sectors: List[str] = []

# Rate-limit timestamps
_last_index_update:  Optional[datetime] = None
_last_regime_update: Optional[datetime] = None
_last_sector_update: Optional[datetime] = None

_INDEX_REFRESH_SEC  = 60
_REGIME_REFRESH_SEC = 300   # every 5 min — NIFTY daily data rarely changes intraday
_SECTOR_REFRESH_SEC = 300   # every 5 min — ~30 Yahoo requests, keep it infrequent

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _yahoo_quote(ticker_sym: str) -> Tuple[float, float]:
    """Return (last_price, pct_change_today). Returns (0, 0) on any error."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_sym)
        fi = t.fast_info
        price = float(fi.last_price or 0)
        prev  = float(fi.previous_close or 0)
        pct   = round((price - prev) / prev * 100, 2) if prev > 0 else 0.0
        return price, pct
    except Exception as e:
        log.debug("_yahoo_quote(%s) failed: %s", ticker_sym, e)
        return 0.0, 0.0


def _yahoo_day_pct(symbol_ns: str) -> float:
    """Return today's % change for a .NS symbol. 0.0 on failure."""
    try:
        import yfinance as yf
        fi = yf.Ticker(symbol_ns).fast_info
        price = float(fi.last_price or 0)
        prev  = float(fi.previous_close or 0)
        if price > 0 and prev > 0:
            return (price - prev) / prev * 100
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Refresh functions (called from background thread)
# ---------------------------------------------------------------------------

def refresh_index_status() -> None:
    """Fetch NIFTY50, BANKNIFTY, INDIA VIX from Yahoo Finance."""
    global _last_index_update
    with _lock:
        last = _last_index_update
    now = datetime.now()
    if last and (now - last).total_seconds() < _INDEX_REFRESH_SEC:
        return

    result: Dict[str, Tuple[float, float]] = {}
    for key, ticker in [
        ("NIFTY",     "^NSEI"),
        ("BANKNIFTY", "^NSEBANK"),
        ("VIX",       "^INDIAVIX"),
    ]:
        result[key] = _yahoo_quote(ticker)

    with _lock:
        _index_status.update(result)
        _last_index_update = now
    log.debug("Index status refreshed: %s", {k: v[0] for k, v in result.items()})


def refresh_market_regime() -> None:
    """
    Determine market regime from NIFTY 50 daily candles.

    Rules:
      VOLATILE  — current ATR > 1.3 × 14-period average ATR
      TRENDING  — EMA20 > EMA50  AND  last close > EMA20
      SIDEWAYS  — everything else
    """
    global _last_regime_update
    with _lock:
        last = _last_regime_update
    now = datetime.now()
    if last and (now - last).total_seconds() < _REGIME_REFRESH_SEC:
        return

    try:
        import yfinance as yf
        df = yf.Ticker("^NSEI").history(period="60d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 25:
            return

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # True Range and ATR(14)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean()
        atr_avg = float(atr.mean())
        atr_cur = float(atr.iloc[-1])

        price_cur  = float(close.iloc[-1])
        ema20_cur  = float(ema20.iloc[-1])
        ema50_cur  = float(ema50.iloc[-1])

        if atr_cur > atr_avg * 1.3:
            regime = "VOLATILE"
        elif ema20_cur > ema50_cur and price_cur > ema20_cur:
            regime = "TRENDING"
        else:
            regime = "SIDEWAYS"

        with _lock:
            global _regime
            _regime = regime
            _last_regime_update = now

        log.debug(
            "Regime: %s  price=%.0f ema20=%.0f ema50=%.0f atr=%.1f avg_atr=%.1f",
            regime, price_cur, ema20_cur, ema50_cur, atr_cur, atr_avg,
        )
    except Exception as e:
        log.debug("refresh_market_regime failed: %s", e)


def refresh_sector_performance() -> None:
    """Compute average % change today for each sector bucket."""
    global _last_sector_update
    with _lock:
        last = _last_sector_update
    now = datetime.now()
    if last and (now - last).total_seconds() < _SECTOR_REFRESH_SEC:
        return

    perf: Dict[str, float] = {}
    for sector, symbols in SECTOR_MAP.items():
        changes = [
            pct
            for sym in symbols
            if (pct := _yahoo_day_pct(f"{sym}.NS")) != 0.0
        ]
        perf[sector] = round(sum(changes) / len(changes), 2) if changes else 0.0

    sorted_sectors = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    strong = [s for s, v in sorted_sectors[:3] if v > 0]

    with _lock:
        _sector_perf.update(perf)
        global _strong_sectors
        _strong_sectors = strong
        _last_sector_update = now

    log.debug("Sector performance: %s", perf)


# ---------------------------------------------------------------------------
# Public readers — thread-safe snapshots
# ---------------------------------------------------------------------------

def get_index_status() -> Dict[str, Tuple[float, float]]:
    with _lock:
        return dict(_index_status)


def get_regime() -> str:
    with _lock:
        return _regime


def get_sector_performance() -> Dict[str, float]:
    with _lock:
        return dict(_sector_perf)


def get_strong_sectors() -> List[str]:
    with _lock:
        return list(_strong_sectors)


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    return SYMBOL_TO_SECTOR.get(symbol)


def sector_confidence_boost(symbol: str) -> float:
    """
    Return +0.05 if the symbol belongs to a currently top-performing sector,
    otherwise 0.0.
    """
    sector = SYMBOL_TO_SECTOR.get(symbol)
    if not sector:
        return 0.0
    with _lock:
        return 0.05 if sector in _strong_sectors else 0.0


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------

_bg_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def start_background_refresh() -> None:
    """
    Launch the daemon thread that periodically refreshes market analytics.
    Safe to call multiple times — only one thread is ever running.
    """
    global _bg_thread
    if _bg_thread and _bg_thread.is_alive():
        return
    _stop_event.clear()
    _bg_thread = threading.Thread(
        target=_bg_loop, daemon=True, name="MarketAnalytics"
    )
    _bg_thread.start()
    log.info("Market analytics background thread started.")


def stop_background_refresh() -> None:
    _stop_event.set()


def _bg_loop() -> None:
    """
    Main loop for the background thread.
    - Index status: every 60 s
    - Regime: every 5 min (piggybacked on index refresh loop)
    - Sector: every 5 min (expensive — ~30 HTTP requests)
    """
    # Immediate first fetch (lightweight)
    try:
        refresh_index_status()
        refresh_market_regime()
    except Exception as e:
        log.debug("Initial market analytics fetch failed: %s", e)

    while not _stop_event.is_set():
        # Sleep in short increments so shutdown is responsive
        for _ in range(6):   # 6 × 10 s = 60 s total
            if _stop_event.is_set():
                return
            time.sleep(10)

        try:
            refresh_index_status()
            refresh_market_regime()
            refresh_sector_performance()
        except Exception as e:
            log.debug("Background analytics refresh error: %s", e)
