
"""
data_fetcher.py — Market data provider with Zerodha → Yahoo fallback.

Behavior:
1. Try Zerodha Kite historical API first.
2. If permission error / API unavailable → fallback to Yahoo Finance.
3. Returns pandas DataFrame with standard columns: open, high, low, close, volume.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from config import config
from kite_client import kite_client
from logger import get_logger

log = get_logger("DataFetcher")

IST = pytz.timezone("Asia/Kolkata")
CACHE_EXT = ".pkl"

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False
    log.warning("yfinance not installed — Yahoo fallback disabled.")


class DataFetcher:

    def __init__(self) -> None:
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)

    def get_historical(self, symbol: str, interval: str, days: int = 60, use_cache: bool = True) -> pd.DataFrame:
        end_dt = datetime.now(tz=IST)
        start_dt = end_dt - timedelta(days=days)
        return self._fetch(symbol, interval, start_dt, end_dt, use_cache)

    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        try:
            return kite_client.get_ltp(symbols)
        except Exception as e:
            log.warning("LTP via Zerodha failed — returning empty. %s", e)
            return {}

    def _fetch(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime, use_cache: bool) -> pd.DataFrame:

        cache_key = f"{symbol}_{interval}_{start_dt.date()}_{end_dt.date()}"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        df = self._fetch_from_kite(symbol, interval, start_dt, end_dt)

        if df is None or df.empty:
            log.warning("Zerodha data failed for %s — trying Yahoo Finance.", symbol)
            df = self._fetch_from_yahoo(symbol, interval, start_dt, end_dt)

        if df is not None and not df.empty and use_cache:
            self._save_cache(cache_key, df)

        return df if df is not None else pd.DataFrame()

    def _fetch_from_kite(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        try:
            raw = kite_client.get_historical_data(symbol, interval, start_dt, end_dt)
            if not raw:
                return None

            df = pd.DataFrame(raw)
            df.rename(columns={"date": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            return df

        except Exception as e:
            log.warning("Kite historical failed: %s", e)
            return None

    def _fetch_from_yahoo(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:

        if not YF_AVAILABLE:
            return None

        try:
            ticker = yf.Ticker(symbol + ".NS")

            yf_interval = {
                "5minute": "5m",
                "15minute": "15m",
                "day": "1d",
            }.get(interval, "5m")

            df = ticker.history(start=start_dt, end=end_dt, interval=yf_interval)

            if df.empty:
                return None

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            return df

        except Exception as e:
            log.error("Yahoo fetch failed: %s", e)
            return None

    def _cache_path(self, key: str) -> str:
        return os.path.join(config.cache_dir, key + CACHE_EXT)

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass


data_fetcher = DataFetcher()
