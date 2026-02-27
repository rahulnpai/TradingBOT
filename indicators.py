"""
indicators.py — Pure pandas/numpy technical indicator calculations.
All functions accept a DataFrame and return a Series or the enriched DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger("Indicators")


# ---------------------------------------------------------------------------
# RSI  (Wilder's smoothing = EWM with alpha = 1/period)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("rsi")


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


# ---------------------------------------------------------------------------
# VWAP  (cumulative intraday — resets each day)
# ---------------------------------------------------------------------------

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume

    # Group by date so VWAP resets daily
    dates = pd.Series(high.index.date if hasattr(high.index, "date") else
                       pd.to_datetime(high.index).date, index=high.index)

    result = pd.Series(index=high.index, dtype=float, name="vwap")
    for day, grp_idx in dates.groupby(dates).groups.items():
        cum_tpv = tpv.loc[grp_idx].cumsum()
        cum_vol = volume.loc[grp_idx].cumsum().replace(0, np.nan)
        result.loc[grp_idx] = cum_tpv / cum_vol

    return result


# ---------------------------------------------------------------------------
# Volume Breakout
# ---------------------------------------------------------------------------

def volume_breakout(
    volume: pd.Series,
    period: int = 20,
    multiplier: float = 1.5,
) -> pd.Series:
    avg_vol   = volume.rolling(period, min_periods=1).mean()
    threshold = avg_vol * multiplier
    return (volume > threshold).rename("vol_breakout")


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_vol = volume.rolling(period, min_periods=1).mean().replace(0, np.nan)
    return (volume / avg_vol).rename("vol_ratio")


# ---------------------------------------------------------------------------
# Convenience: enrich a full OHLCV DataFrame with all indicators
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Expects columns: open, high, low, close, volume (case-insensitive).
    Returns the same DataFrame with indicator columns appended.
    """
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    if cfg is None:
        from config import config
        cfg = config.indicators

    # RSI
    df["rsi"] = rsi(df["close"], period=cfg.rsi_period)

    # MACD
    macd_df = macd(df["close"], fast=cfg.macd_fast, slow=cfg.macd_slow,
                   signal_period=cfg.macd_signal)
    df = pd.concat([df, macd_df], axis=1)

    # EMAs
    for period in (cfg.ema_9, cfg.ema_21, cfg.ema_50, cfg.ema_200):
        df[f"ema_{period}"] = ema(df["close"], period)

    # VWAP
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])

    # Volume
    df["vol_breakout"] = volume_breakout(
        df["volume"], period=cfg.vol_ma_period,
        multiplier=cfg.vol_breakout_multiplier,
    )
    df["vol_ratio"]    = volume_ratio(df["volume"], period=cfg.vol_ma_period)
    df["vol_ma"]       = df["volume"].rolling(cfg.vol_ma_period, min_periods=1).mean()

    return df


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def macd_crossover_bullish(df: pd.DataFrame) -> pd.Series:
    """True on candles where MACD line crossed above signal line."""
    return (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    )


def macd_crossover_bearish(df: pd.DataFrame) -> pd.Series:
    """True on candles where MACD line crossed below signal line."""
    return (
        (df["macd"] < df["macd_signal"]) &
        (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    )


def price_above_vwap(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["vwap"]


def ema_bullish_alignment(df: pd.DataFrame) -> pd.Series:
    """EMA9 > EMA21 > EMA50 — short-term uptrend."""
    return (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])


def ema_bearish_alignment(df: pd.DataFrame) -> pd.Series:
    return (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])
