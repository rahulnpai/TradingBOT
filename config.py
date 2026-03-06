
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------- Trading settings ----------------

@dataclass
class TradingConfig:
    paper_trading: bool = True
    trading_mode: str = "intraday"

    # Safer testing capital
    capital: int = 5000

    symbols = ["ADANIENT", "SBIN"]

# ---------------- Data provider settings ----------------

@dataclass
class DataConfig:
    provider: str = "yahoo"   # "yahoo" for testing, "zerodha" for live

    # More history for better indicators (EMA200 etc.)
    history_days: int = 120

# ---------------- Zerodha credentials ----------------

@dataclass
class KiteConfig:
    api_key: str = os.getenv("KITE_API_KEY", "")
    api_secret: str = os.getenv("KITE_API_SECRET", "")
    access_token: str = os.getenv("KITE_ACCESS_TOKEN", "")

# ---------------- Main config object ----------------

class Config:
    def __init__(self):
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.kite = KiteConfig()

        self.cache_dir = "cache"
        self.data_dir = "data"

config = Config()
