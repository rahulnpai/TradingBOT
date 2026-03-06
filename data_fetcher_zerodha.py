
"""
data_fetcher.py — Historical and live market data with disk caching.
Modified to always prefer real Zerodha Kite historical data even in paper mode.
"""

from __future__ import annotations

import os
import pickle
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz

from config import config
from kite_client import kite_client
from logger import get_logger

log = get_logger("DataFetcher")

IST = pytz.timezone("Asia/Kolkata")
CACHE_EXT = ".pkl"


class DataFetcher:

    def __init__(self) -> None:
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical(
        self,
        symbol: str,
        interval: str,
        days: int = 60,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame for *symbol* over the past *days*."""
        end_dt   = datetime.now(tz=IST)
        start_dt = end_dt - timedelta(days=days)
        return self._fetch(symbol, interval, start_dt, end_dt, use_cache)

    def get_intraday(
        self,
        symbol: str,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """Today's intraday candles."""
        now      = datetime.now(tz=IST)
        start_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        return self._fetch(symbol, interval, start_dt, now, use_cache=False)

    def get_quote(self, symbol: str) -> float:
        """Return current last-traded price."""
        try:
            quote = kite_client.get_quote(symbol)
            return float(quote.get("last_price", 0.0))
        except Exception as e:
            log.error("get_quote(%s) failed: %s", symbol, e)
            return 0.0

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """LTP for multiple symbols at once."""
        try:
            return kite_client.get_ltp(symbols)
        except Exception as e:
            log.error("get_batch_quotes failed: %s", e)
            return {}

    def get_swing_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        return self.get_historical(symbol, "day", days=days, use_cache=True)

    # ------------------------------------------------------------------
    # Internal helpers
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

        # ALWAYS use real Zerodha historical data
        df = self._fetch_from_kite(symbol, interval, start_dt, end_dt)

        if df is not None and not df.empty:
            if use_cache:
                self._save_cache(cache_key, df)

        return df if df is not None else pd.DataFrame()

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
