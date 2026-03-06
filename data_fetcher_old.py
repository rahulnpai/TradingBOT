"""
data_fetcher.py — Historical and live market data with disk caching.

In paper trading mode the module can also generate synthetic intraday data
so the system runs without a valid Kite token (useful for dry-runs / CI).
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
        os.makedirs(config.data_dir,  exist_ok=True)

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
        """Today's intraday candles (no cache — always fresh)."""
        now      = datetime.now(tz=IST)
        start_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
        return self._fetch(symbol, interval, start_dt, now, use_cache=False)

    def get_quote(self, symbol: str) -> float:
        """Return current last-traded price."""
        if config.trading.paper_trading:
            return self._mock_ltp(symbol)
        try:
            quote = kite_client.get_quote(symbol)
            return float(quote.get("last_price", 0.0))
        except Exception as e:
            log.error("get_quote(%s) failed: %s", symbol, e)
            return 0.0

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """LTP for multiple symbols at once."""
        if config.trading.paper_trading:
            return {s: self._mock_ltp(s) for s in symbols}
        try:
            return kite_client.get_ltp(symbols)
        except Exception as e:
            log.error("get_batch_quotes failed: %s", e)
            return {}

    def get_swing_data(
        self, symbol: str, days: int = 365
    ) -> pd.DataFrame:
        """Daily candles for swing analysis."""
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
        
        '''
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                log.debug("Cache hit: %s", cache_key)
                return cached

        if config.trading.paper_trading:
            df = self._fetch_live_or_mock(symbol, interval, start_dt, end_dt)
        else:
            df = self._fetch_from_kite(symbol, interval, start_dt, end_dt)

       '''
       if use_cache:
       cached = self._load_cache(cache_key)
       if cached is not None:
       log.debug("Cache hit: %s", cache_key)
       return cached

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

    def _fetch_live_or_mock(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Try real Kite call; fall back to synthetic data in paper mode."""
        try:
            if kite_client._kite is not None:
                return self._fetch_from_kite(symbol, interval, start_dt, end_dt)
        except Exception:
            pass
        log.info("Generating synthetic data for %s [%s] (paper mode)", symbol, interval)
        return self._generate_synthetic(symbol, interval, start_dt, end_dt)

    # ------------------------------------------------------------------
    # Synthetic data generator (paper trading / testing)
    # ------------------------------------------------------------------

    _BASE_PRICES: Dict[str, float] = {
        "RELIANCE": 2950.0, "TCS": 3800.0, "INFY": 1780.0,
        "HDFCBANK": 1670.0, "ICICIBANK": 1120.0, "SBIN": 830.0,
        "WIPRO": 560.0,  "HCLTECH": 1680.0, "AXISBANK": 1200.0,
        "KOTAKBANK": 1870.0,
    }

    def _generate_synthetic(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        freq_map = {
            "minute":  "1min", "3minute":  "3min",  "5minute": "5min",
            "10minute":"10min", "15minute":"15min", "30minute":"30min",
            "60minute":"60min", "day": "1D",
        }
        freq   = freq_map.get(interval, "5min")
        times  = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz=IST)

        # Filter to market hours for intraday
        if "D" not in freq:
            times = times[
                ((times.hour > 9) | ((times.hour == 9) & (times.minute >= 15))) &
                ((times.hour < 15) | ((times.hour == 15) & (times.minute <= 30)))
            ]

        base = self._BASE_PRICES.get(symbol, 1000.0)
        n = len(times)
        if n == 0:
            return pd.DataFrame()

        rng   = np.random.default_rng(hash(symbol) % (2**31))
        pct   = rng.normal(0.0001, 0.003, n)
        close = base * np.exp(np.cumsum(pct))

        high   = close * (1 + rng.uniform(0.001, 0.008, n))
        low    = close * (1 - rng.uniform(0.001, 0.008, n))
        open_  = low + rng.uniform(0, 1, n) * (high - low)
        volume = rng.integers(50_000, 500_000, n).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            index=times,
        )
        df.index.name = "datetime"
        return df

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

    # ------------------------------------------------------------------

    @staticmethod
    def _mock_ltp(symbol: str) -> float:
        base = DataFetcher._BASE_PRICES.get(symbol, 1000.0)
        return round(base * random.uniform(0.98, 1.02), 2)


# Singleton
data_fetcher = DataFetcher()
