
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import pytz
import yfinance as yf

from config import config
from kite_client import kite_client
from logger import get_logger

log = get_logger("DataFetcher")
IST = pytz.timezone("Asia/Kolkata")


class DataFetcher:

    def get_historical(self, symbol, interval="5minute", days=None):

        # Use configured history length if not provided
        if days is None:
            days = config.data.history_days

        end_dt = datetime.now(tz=IST)
        start_dt = end_dt - timedelta(days=days)

        if config.data.provider == "zerodha":
            return self._fetch_from_kite(symbol, interval, start_dt, end_dt)
        else:
            return self._fetch_from_yahoo(symbol, interval, start_dt, end_dt)

    # -------- Zerodha data --------

    def _fetch_from_kite(self, symbol, interval, start_dt, end_dt) -> Optional[pd.DataFrame]:
        try:
            raw = kite_client.get_historical_data(symbol, interval, start_dt, end_dt)
            if not raw:
                return pd.DataFrame()

            df = pd.DataFrame(raw)
            df.rename(columns={"date": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            return df

        except Exception as e:
            log.error("Zerodha fetch failed: %s", e)
            return pd.DataFrame()

    # -------- Yahoo fallback --------

    def _fetch_from_yahoo(self, symbol, interval, start_dt, end_dt):

        ticker = yf.Ticker(symbol + ".NS")

        yf_interval = {
            "5minute": "5m",
            "15minute": "15m",
            "day": "1d"
        }.get(interval, "5m")

        df = ticker.history(start=start_dt, end=end_dt, interval=yf_interval)

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        return df


data_fetcher = DataFetcher()
