"""
data_fetcher.py — Historical and live market data with disk caching.

Data sources:
  - Yahoo Finance (config.data_provider == "yahoo"): used for paper mode.
    No Zerodha credentials required. Works 24/7 with real historical candles.
  - Zerodha KiteConnect (config.data_provider == "zerodha"): used for sim/live.
    Requires valid Kite access token.

No synthetic data is generated anywhere in this module.
"""

from __future__ import annotations

import os
import pickle
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from config import config
from kite_client import kite_client
from logger import get_logger

log = get_logger("DataFetcher")

IST = pytz.timezone("Asia/Kolkata")
CACHE_EXT = ".pkl"

# Maximum lookback Yahoo Finance supports per interval
_YAHOO_MAX_DAYS: Dict[str, int] = {
    "1m":  6,
    "2m":  59,
    "5m":  59,
    "15m": 59,
    "30m": 59,
    "90m": 59,
    "60m": 729,
    "1h":  729,
    "1d":  9999,
    "5d":  9999,
    "1wk": 9999,
    "1mo": 9999,
}

# Kite interval → Yahoo interval
_YAHOO_INTERVAL_MAP: Dict[str, str] = {
    "minute":   "1m",
    "3minute":  "5m",    # 3m not available in Yahoo; use 5m
    "5minute":  "5m",
    "10minute": "15m",   # 10m not available; use 15m
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "60m",
    "day":      "1d",
}


class DataFetcher:

    def __init__(self) -> None:
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.data_dir,  exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical(
        self,
        symbol: str,
        interval: str,
        days: int = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame for *symbol* over the past *days*."""
        if days is None:
            days = config.history_days
        end_dt   = datetime.now(tz=IST)
        start_dt = end_dt - timedelta(days=days)
        return self._fetch(symbol, interval, start_dt, end_dt, use_cache)

    def get_intraday(
        self,
        symbol: str,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """
        Today's intraday candles (no cache — always fresh).
        In Yahoo mode returns last 5 days of intraday candles so the bot
        has data even when the market is currently closed.
        """
        if config.data_provider == "yahoo":
            return self._yahoo_intraday(symbol, interval)

        # Zerodha: today's session only
        now      = datetime.now(tz=IST)
        start_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        return self._fetch(symbol, interval, start_dt, now, use_cache=False)

    def get_quote(self, symbol: str) -> float:
        """Return current last-traded price."""
        if config.data_provider == "yahoo":
            return self._yahoo_last_price(symbol)

        # Zerodha (sim or live)
        try:
            quote = kite_client.get_quote(symbol)
            return float(quote.get("last_price", 0.0))
        except Exception as e:
            log.error("get_quote(%s) via Kite failed: %s — falling back to Yahoo", symbol, e)
            return self._yahoo_last_price(symbol)

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """LTP for multiple symbols at once."""
        if config.data_provider == "yahoo":
            return {s: self._yahoo_last_price(s) for s in symbols}

        # Zerodha
        try:
            return kite_client.get_ltp(symbols)
        except Exception as e:
            log.error("get_batch_quotes via Kite failed: %s — falling back to Yahoo", e)
            return {s: self._yahoo_last_price(s) for s in symbols}

    def get_swing_data(
        self, symbol: str, days: int = 365
    ) -> pd.DataFrame:
        """Daily candles for swing analysis."""
        return self.get_historical(symbol, "day", days=days, use_cache=True)

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _fetch(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        use_cache: bool,
    ) -> pd.DataFrame:
        cache_key = self._cache_key(symbol, interval, start_dt.date(), end_dt.date())

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                log.debug("Cache hit: %s", cache_key)
                return cached

        if config.data_provider == "yahoo":
            df = self._fetch_from_yahoo(symbol, interval, start_dt, end_dt)
        else:
            df = self._fetch_from_kite(symbol, interval, start_dt, end_dt)

        if df is not None and not df.empty:
            if use_cache:
                self._save_cache(cache_key, df)
        return df if df is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    # Yahoo Finance fetcher
    # ------------------------------------------------------------------

    def _fetch_from_yahoo(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.critical("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        yf_interval = _YAHOO_INTERVAL_MAP.get(interval, "5m")
        max_days    = _YAHOO_MAX_DAYS.get(yf_interval, 59)

        # Cap start date to Yahoo's limit for this interval
        earliest_start = datetime.now(tz=IST) - timedelta(days=max_days)
        if start_dt < earliest_start:
            log.debug(
                "Yahoo: capping %s [%s] start from %s to %s (provider limit %d days)",
                symbol, yf_interval,
                start_dt.date(), earliest_start.date(), max_days,
            )
            start_dt = earliest_start

        ticker_sym = f"{symbol}.NS"
        # Yahoo end is exclusive — add 1 day to include end_dt
        end_date = end_dt.date() + timedelta(days=1)

        try:
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(
                start=start_dt.date(),
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )
        except Exception as e:
            log.error("Yahoo fetch failed for %s [%s]: %s", symbol, interval, e)
            return pd.DataFrame()

        if df is None or df.empty:
            log.warning("Yahoo returned no data for %s [%s]", symbol, interval)
            return pd.DataFrame()

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            log.error("Yahoo response missing columns %s for %s", missing, symbol)
            return pd.DataFrame()

        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Ensure tz-aware IST index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        df.index.name = "datetime"
        df.sort_index(inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        log.debug("Yahoo returned %d candles for %s [%s]", len(df), symbol, interval)
        return df

    def _yahoo_intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        """Last 5 days of intraday candles from Yahoo (works even when market closed)."""
        try:
            import yfinance as yf
        except ImportError:
            log.critical("yfinance not installed. Run: pip install yfinance")
            return pd.DataFrame()

        yf_interval = _YAHOO_INTERVAL_MAP.get(interval, "5m")
        ticker_sym  = f"{symbol}.NS"

        try:
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(period="5d", interval=yf_interval,
                                auto_adjust=True, prepost=False)
        except Exception as e:
            log.error("Yahoo intraday fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        df = df[["open", "high", "low", "close", "volume"]].copy()

        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        df.index.name = "datetime"
        df.sort_index(inplace=True)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return df

    def _yahoo_last_price(self, symbol: str) -> float:
        """Return the most recent available price from Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            # fast_info.last_price is quickest and works outside market hours
            price = ticker.fast_info.last_price
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        # Fallback: last close from recent history
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{symbol}.NS").history(period="5d", interval="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            log.error("Yahoo last price fallback failed for %s: %s", symbol, e)
        return 0.0

    # ------------------------------------------------------------------
    # Zerodha / Kite fetcher
    # ------------------------------------------------------------------

    def _fetch_from_kite(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        try:
            raw = kite_client.get_historical_data(symbol, interval, start_dt, end_dt)
            if not raw:
                log.warning("No data returned from Kite for %s", symbol)
                return pd.DataFrame()
            df = pd.DataFrame(raw)
            df.rename(columns={"date": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
            log.debug("Kite returned %d candles for %s [%s]", len(df), symbol, interval)
            return df
        except Exception as e:
            log.error("Kite historical fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, symbol: str, interval: str, sd: date, ed: date) -> str:
        return f"{symbol}_{interval}_{sd}_{ed}"

    def _cache_path(self, key: str) -> str:
        return os.path.join(config.cache_dir, key + CACHE_EXT)

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning("Cache read failed (%s): %s", key, e)
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            log.warning("Cache write failed (%s): %s", key, e)


# Singleton
data_fetcher = DataFetcher()
