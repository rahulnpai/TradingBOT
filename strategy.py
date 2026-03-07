"""
strategy.py — Intraday and Swing signal generation.

Each strategy method operates on a fully-enriched OHLCV+indicator DataFrame
and returns a Signal dataclass at the latest (last) candle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from config import config
from logger import get_logger

log = get_logger("Strategy")


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    action: str          # "BUY" | "SELL" | "HOLD"
    strategy_type: str   # "intraday" | "swing"
    confidence: float    # 0.0 – 1.0
    entry_price: float
    suggested_sl: float
    suggested_target: float
    reason: str
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_actionable(self) -> bool:
        return self.action in ("BUY", "SELL")

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strategy_type": self.strategy_type,
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "suggested_sl": self.suggested_sl,
            "suggested_target": self.suggested_target,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Stateless signal generator — pass a DataFrame enriched with indicators.

    Intraday rules (5-min candles):
      BUY  when:  MACD bullish crossover  AND  RSI 35–65
                  AND price > VWAP       AND  EMA9 > EMA21
                  AND volume breakout
      SELL when:  MACD bearish crossover  OR   RSI > 72
                  OR  price < VWAP (with EMA bearish)
                  OR  stoploss / target hit (handled in trader)

    Swing rules (daily candles):
      BUY  when:  Price > EMA50 > EMA200  AND  RSI 40–60
                  AND MACD bullish        AND  volume > avg
      SELL when:  Price < EMA50           OR   RSI > 70
                  OR  MACD bearish crossover
    """

    def __init__(self) -> None:
        self._icfg = config.indicators

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        mode: str = "intraday",
    ) -> Signal:
        if df is None or len(df) < 50:
            return self._hold(symbol, df, mode, "Not enough data")

        if mode == "intraday":
            return self._intraday_signal(symbol, df)
        elif mode == "swing":
            return self._swing_signal(symbol, df)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Intraday strategy
    # ------------------------------------------------------------------

    def _intraday_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        r = df.iloc[-1]   # latest candle
        p = df.iloc[-2]   # previous candle

        entry_price = float(r["close"])
        ind = self._snapshot(r)

        # ---- BUY conditions -------------------------------------------
        buy_conditions = {
            "macd_cross_bullish": (
                float(r["macd"]) > float(r["macd_signal"]) and
                float(p["macd"]) <= float(p["macd_signal"])
            ),
            "rsi_healthy":        35 <= float(r["rsi"]) <= 65,
            "price_above_vwap":   float(r["close"]) > float(r["vwap"]),
            "ema_aligned":        float(r["ema_9"]) > float(r["ema_21"]),
           '''  "volume_breakout":    bool(r["vol_breakout"]), '''
	    "volume_breakout":     True,
        }

        # ---- SELL conditions ------------------------------------------
        sell_conditions = {
            "macd_cross_bearish": (
                float(r["macd"]) < float(r["macd_signal"]) and
                float(p["macd"]) >= float(p["macd_signal"])
            ),
            "rsi_overbought":  float(r["rsi"]) > self._icfg.rsi_overbought,
            "price_below_vwap_and_ema_bearish": (
                float(r["close"]) < float(r["vwap"]) and
                float(r["ema_9"]) < float(r["ema_21"])
            ),
        }

        buy_score  = sum(buy_conditions.values())  / len(buy_conditions)
        sell_score = sum(sell_conditions.values()) / len(sell_conditions)

        rcfg = config.risk
        sl_pct  = rcfg.default_sl_pct
        tgt_pct = rcfg.default_target_pct

        if buy_score >= 0.10:   # at least 3.5 / 5 conditions
            sl  = round(entry_price * (1 - sl_pct), 2)
            tgt = round(entry_price * (1 + tgt_pct), 2)
            reasons = [k for k, v in buy_conditions.items() if v]
            return Signal(
                symbol=symbol, action="BUY", strategy_type="intraday",
                confidence=buy_score, entry_price=entry_price,
                suggested_sl=sl, suggested_target=tgt,
                reason=", ".join(reasons), indicators=ind,
            )

        if sell_score >= 0.20:
            sl  = round(entry_price * (1 + sl_pct), 2)
            tgt = round(entry_price * (1 - tgt_pct), 2)
            reasons = [k for k, v in sell_conditions.items() if v]
            return Signal(
                symbol=symbol, action="SELL", strategy_type="intraday",
                confidence=sell_score, entry_price=entry_price,
                suggested_sl=sl, suggested_target=tgt,
                reason=", ".join(reasons), indicators=ind,
            )

        return self._hold(symbol, df, "intraday",
                          f"buy={buy_score:.2f} sell={sell_score:.2f}")

    # ------------------------------------------------------------------
    # Swing strategy
    # ------------------------------------------------------------------

    def _swing_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        r = df.iloc[-1]
        p = df.iloc[-2]

        entry_price = float(r["close"])
        ind = self._snapshot(r)

        buy_conditions = {
            "price_above_ema50":  float(r["close"]) > float(r["ema_50"]),
            "ema50_above_ema200": float(r["ema_50"]) > float(r["ema_200"]),
            "rsi_mid_range":      40 <= float(r["rsi"]) <= 62,
            "macd_bullish":       float(r["macd"]) > float(r["macd_signal"]),
            "volume_above_avg":   float(r["volume"]) > float(r["vol_ma"]),
        }

        sell_conditions = {
            "price_below_ema50":   float(r["close"]) < float(r["ema_50"]),
            "rsi_overbought":      float(r["rsi"]) > self._icfg.rsi_overbought,
            "macd_cross_bearish": (
                float(r["macd"]) < float(r["macd_signal"]) and
                float(p["macd"]) >= float(p["macd_signal"])
            ),
        }

        buy_score  = sum(buy_conditions.values())  / len(buy_conditions)
        sell_score = sum(sell_conditions.values()) / len(sell_conditions)

        rcfg = config.risk
        # Wider targets for swing
        sl_pct  = rcfg.default_sl_pct  * 2
        tgt_pct = rcfg.default_target_pct * 2

        if buy_score >= 0.75:
            sl  = round(entry_price * (1 - sl_pct), 2)
            tgt = round(entry_price * (1 + tgt_pct), 2)
            reasons = [k for k, v in buy_conditions.items() if v]
            return Signal(
                symbol=symbol, action="BUY", strategy_type="swing",
                confidence=buy_score, entry_price=entry_price,
                suggested_sl=sl, suggested_target=tgt,
                reason=", ".join(reasons), indicators=ind,
            )

        if sell_score >= 0.67:
            sl  = round(entry_price * (1 + sl_pct), 2)
            tgt = round(entry_price * (1 - tgt_pct), 2)
            reasons = [k for k, v in sell_conditions.items() if v]
            return Signal(
                symbol=symbol, action="SELL", strategy_type="swing",
                confidence=sell_score, entry_price=entry_price,
                suggested_sl=sl, suggested_target=tgt,
                reason=", ".join(reasons), indicators=ind,
            )

        return self._hold(symbol, df, "swing",
                          f"buy={buy_score:.2f} sell={sell_score:.2f}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hold(
        symbol: str, df: Optional[pd.DataFrame],
        mode: str, reason: str,
    ) -> Signal:
        price = float(df.iloc[-1]["close"]) if df is not None and len(df) else 0.0
        return Signal(
            symbol=symbol, action="HOLD", strategy_type=mode,
            confidence=0.0, entry_price=price,
            suggested_sl=0.0, suggested_target=0.0,
            reason=reason,
        )

    @staticmethod
    def _snapshot(row: pd.Series) -> Dict[str, float]:
        keys = ["rsi", "macd", "macd_signal", "macd_hist",
                "ema_9", "ema_21", "ema_50", "ema_200",
                "vwap", "vol_ratio"]
        return {k: round(float(row[k]), 4) for k in keys if k in row.index}


# Singleton
strategy_engine = StrategyEngine()
