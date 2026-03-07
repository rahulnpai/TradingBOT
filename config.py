"""
config.py — Central configuration for AI Trader.
All tuneable parameters live here; secrets come from .env.
"""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class KiteConfig:
    api_key: str         = os.getenv("KITE_API_KEY",      "")
    api_secret: str      = os.getenv("KITE_API_SECRET",   "")
    access_token: str    = os.getenv("KITE_ACCESS_TOKEN", "")
    redirect_url: str    = os.getenv("KITE_REDIRECT_URL", "http://localhost:8080")
    token_file: str      = ".kite_token"          # persisted access token


@dataclass
class TradingConfig:
    # ---- mode ------------------------------------------------------------
    paper_trading: bool   = True          # False → live orders
    trading_mode: str     = "intraday"    # "intraday" | "swing"

    # ---- capital ---------------------------------------------------------
    capital: float        = float(os.getenv("TRADING_CAPITAL", "100000"))

    # ---- watchlist -------------------------------------------------------
    watchlist_mode: str = os.getenv("WATCHLIST_MODE", "core")
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "WIPRO", "HCLTECH", "AXISBANK", "KOTAKBANK",
    ])
    exchange: str = "NSE"

    # ---- timeframes ------------------------------------------------------
    intraday_interval: str = "5minute"   # Kite interval string
    swing_interval: str    = "day"

    # ---- market hours (IST, 24-h) ----------------------------------------
    market_open_h: int  = 9
    market_open_m: int  = 15
    market_close_h: int = 15
    market_close_m: int = 30
    intraday_exit_h: int = 15  # force-close all intraday by 15:10
    intraday_exit_m: int = 10

    # ---- data lookback ---------------------------------------------------
    lookback_candles: int = 250


@dataclass
class RiskConfig:
    capital: float                   = float(os.getenv("TRADING_CAPITAL", "100000"))
    max_capital_per_trade_pct: float = 0.02   # 2 % per trade
    daily_loss_limit_pct: float      = 0.03   # 3 % daily hard stop
    max_open_positions: int          = 10
    default_sl_pct: float            = 0.015  # 1.5 % stoploss
    default_target_pct: float        = 0.030  # 3 % target  → 1:2 RR
    trailing_sl: bool                = True
    trailing_sl_pct: float           = 0.010  # 1 % trailing

    # ---- professional risk controls --------------------------------------
    max_trades_per_symbol: int       = 3      # max entries per symbol per day
    cooldown_minutes: int            = 15     # no-trade window after SL exit


@dataclass
class IndicatorConfig:
    # RSI
    rsi_period: int       = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float   = 30.0

    # MACD
    macd_fast: int   = 12
    macd_slow: int   = 26
    macd_signal: int = 9

    # EMAs
    ema_9:   int = 9
    ema_21:  int = 21
    ema_50:  int = 50
    ema_200: int = 200

    # VWAP resets daily (only relevant for intraday data)
    # Volume breakout
    vol_ma_period: int          = 20
    vol_breakout_multiplier: float = 1.5


@dataclass
class AIConfig:
    enabled: bool      = True
    ollama_url: str    = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model: str         = os.getenv("AI_MODEL",   "mistral")   # or deepseek-r1
    timeout: int       = 90       # seconds per request (fallback if model not in timeout ladder)
    min_confidence: float = 0.55  # accept AI confirmation above this threshold

    # Throttle: minimum seconds between AI calls per symbol
    call_interval_sec: int = 300   # 5 min

    # Multi-model fallback — set AI_MODELS env var to override order
    models: List[str] = field(default_factory=lambda: [
        m.strip()
        for m in os.getenv("AI_MODELS", "llama3,mistral,deepseek-coder").split(",")
        if m.strip()
    ])


@dataclass
class BacktestConfig:
    start_date: str        = "2024-01-01"
    end_date: str          = "2024-12-31"
    initial_capital: float = 100_000.0
    brokerage_per_trade: float = 20.0   # flat ₹ per side
    slippage_pct: float        = 0.001  # 0.1 % market impact


@dataclass
class Config:
    kite:      KiteConfig      = field(default_factory=KiteConfig)
    trading:   TradingConfig   = field(default_factory=TradingConfig)
    risk:      RiskConfig      = field(default_factory=RiskConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    ai:        AIConfig        = field(default_factory=AIConfig)
    backtest:  BacktestConfig  = field(default_factory=BacktestConfig)

    # ---- runtime mode ----------------------------------------------------
    data_provider: str = "yahoo"    # "yahoo" | "zerodha"
    history_days: int  = 120        # candle lookback window for all modes

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file:  str = "logs/ai_trader.log"
    data_dir:  str = "data"
    cache_dir: str = "cache"

    def __post_init__(self) -> None:
        # Keep risk.capital in sync with trading.capital (trading takes precedence)
        self.risk.capital = self.trading.capital

    def validate(self) -> None:
        """Raise ValueError if critical config is missing for the active data provider."""
        if self.data_provider == "zerodha":
            if not self.kite.api_key or not self.kite.api_secret:
                raise ValueError(
                    "KITE_API_KEY and KITE_API_SECRET must be set for Zerodha data provider."
                )
            if not self.kite.access_token:
                raise ValueError(
                    "KITE_ACCESS_TOKEN must be set for Zerodha data provider. "
                    "Run 'python kite_client.py --login' first."
                )


# ---------------------------------------------------------------------------
# Watchlist definitions
# ---------------------------------------------------------------------------

CORE_SYMBOLS = ["SBIN", "ADANIENT"]

NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT",
    "AXISBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", "HCLTECH", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "POWERGRID",
    "NTPC", "ONGC", "WIPRO", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HINDUNILVR", "BAJAJFINSV", "BAJAJ-AUTO",
    "BRITANNIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HEROMOTOCO",
    "HINDALCO", "INDUSINDBK", "M&M", "SBILIFE", "TATACONSUM", "TATASTEEL",
    "TECHM", "UPL", "BPCL", "CIPLA", "APOLLOHOSP", "SHREECEM",
]

# Singleton — import this everywhere
config = Config()

# Apply watchlist based on WATCHLIST_MODE env var
if config.trading.watchlist_mode == "nifty50":
    config.trading.symbols = NIFTY50_SYMBOLS
else:
    config.trading.symbols = CORE_SYMBOLS
