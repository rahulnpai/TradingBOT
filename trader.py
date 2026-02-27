"""
trader.py — Trade execution layer.

PaperTrader  — simulates fills with slippage; no real orders sent.
LiveTrader   — routes orders through Zerodha KiteConnect.

The Trader class acts as a facade that delegates to whichever engine is active
based on config.trading.paper_trading.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ai_engine import AIAnalysis, AIEngine, ai_engine
from config import config
from kite_client import kite_client
from logger import TradeLogger, get_logger
from risk_manager import RiskManager, TradeOrder, risk_manager
from strategy import Signal

log = get_logger("Trader")
trade_log = TradeLogger()


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    order_id: str
    symbol: str
    action: str
    quantity: int
    fill_price: float
    stoploss: float
    target: float
    strategy_type: str
    paper: bool
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "OPEN"          # OPEN | CLOSED | REJECTED
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ""
    ai_decision: str = ""
    ai_confidence: float = 0.0
    ai_reasoning: str = ""

    def as_csv_row(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.fill_price,
            "stoploss": self.stoploss,
            "target": self.target,
            "strategy_type": self.strategy_type,
            "paper": self.paper,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "exit_reason": self.exit_reason,
            "ai_decision": self.ai_decision,
            "ai_confidence": round(self.ai_confidence, 4),
        }


# ---------------------------------------------------------------------------
# Paper trading engine
# ---------------------------------------------------------------------------

class PaperTrader:
    """Simulates order fills with configurable slippage."""

    SLIPPAGE = 0.0005   # 0.05 %

    def execute(self, order: TradeOrder) -> Optional[TradeResult]:
        if not order.valid or order.quantity < 1:
            log.warning("Paper: invalid order skipped (%s)", order.reject_reason)
            return None

        # Simulate market fill with slippage
        if order.action == "BUY":
            fill = round(order.entry_price * (1 + self.SLIPPAGE), 2)
        else:
            fill = round(order.entry_price * (1 - self.SLIPPAGE), 2)

        result = TradeResult(
            order_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            fill_price=fill,
            stoploss=order.stoploss,
            target=order.target,
            strategy_type=order.strategy_type,
            paper=True,
        )
        log.info(
            "[PAPER] %s %s x%d  filled=%.2f  sl=%.2f  tgt=%.2f",
            order.action, order.symbol, order.quantity,
            fill, order.stoploss, order.target,
        )
        return result

    def close(
        self,
        result: TradeResult,
        exit_price: float,
        reason: str = "manual",
    ) -> TradeResult:
        if result.action == "BUY":
            pnl = (exit_price - result.fill_price) * result.quantity
        else:
            pnl = (result.fill_price - exit_price) * result.quantity

        result.exit_price = exit_price
        result.pnl        = round(pnl, 2)
        result.exit_reason = reason
        result.status     = "CLOSED"
        log.info(
            "[PAPER] CLOSE %s  exit=%.2f  pnl=₹%.2f  reason=%s",
            result.symbol, exit_price, pnl, reason,
        )
        return result


# ---------------------------------------------------------------------------
# Live trading engine
# ---------------------------------------------------------------------------

class LiveTrader:
    """Routes orders to Zerodha via KiteConnect."""

    def execute(self, order: TradeOrder) -> Optional[TradeResult]:
        if not order.valid or order.quantity < 1:
            log.warning("Live: invalid order skipped (%s)", order.reject_reason)
            return None

        try:
            order_id = kite_client.place_market_order(
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                product=order.product,
            )

            # Place stoploss cover order immediately
            sl_action = "SELL" if order.action == "BUY" else "BUY"
            kite_client.place_sl_order(
                symbol=order.symbol,
                action=sl_action,
                quantity=order.quantity,
                trigger_price=order.stoploss,
                price=order.stoploss,
                product=order.product,
            )

            result = TradeResult(
                order_id=order_id,
                symbol=order.symbol,
                action=order.action,
                quantity=order.quantity,
                fill_price=order.entry_price,   # refined when order status arrives
                stoploss=order.stoploss,
                target=order.target,
                strategy_type=order.strategy_type,
                paper=False,
            )
            return result

        except Exception as e:
            log.error("Live order failed for %s: %s", order.symbol, e)
            return None

    def close(
        self,
        result: TradeResult,
        exit_price: float,
        reason: str = "manual",
    ) -> TradeResult:
        close_action = "SELL" if result.action == "BUY" else "BUY"
        try:
            kite_client.place_market_order(
                symbol=result.symbol,
                action=close_action,
                quantity=result.quantity,
                product="MIS" if result.strategy_type == "intraday" else "CNC",
            )
        except Exception as e:
            log.error("Live close failed for %s: %s", result.symbol, e)

        pnl = (
            (exit_price - result.fill_price) * result.quantity
            if result.action == "BUY"
            else (result.fill_price - exit_price) * result.quantity
        )
        result.exit_price  = exit_price
        result.pnl         = round(pnl, 2)
        result.exit_reason = reason
        result.status      = "CLOSED"
        return result


# ---------------------------------------------------------------------------
# Trader facade
# ---------------------------------------------------------------------------

class Trader:
    """
    High-level orchestrator:
      1. Calls AI engine to validate / enrich signal.
      2. Calls risk manager to size & validate the trade.
      3. Routes to paper or live execution.
      4. Persists trade events to audit log.
    """

    def __init__(self) -> None:
        self._engine: PaperTrader | LiveTrader = (
            PaperTrader() if config.trading.paper_trading else LiveTrader()
        )
        self._open_trades: Dict[str, TradeResult] = {}
        self._closed_trades: List[TradeResult] = []

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def process_signal(self, signal: Signal) -> Optional[TradeResult]:
        """Full pipeline: AI → risk check → execute."""
        if not signal.is_actionable:
            return None

        # --- 1. AI validation ----------------------------------------
        analysis: AIAnalysis = ai_engine.analyse_signal(signal)

        if analysis.is_rejected:
            log.warning(
                "AI REJECTED %s %s: %s",
                signal.action, signal.symbol, analysis.reasoning[:120],
            )
            return None

        # Apply AI-suggested SL/target if provided
        if analysis.suggested_sl:
            signal.suggested_sl = analysis.suggested_sl
        if analysis.suggested_target:
            signal.suggested_target = analysis.suggested_target

        # --- 2. Risk validation --------------------------------------
        order: TradeOrder = risk_manager.validate_trade(signal)
        if not order.valid:
            return None

        # --- 3. Execute -----------------------------------------------
        result = self._engine.execute(order)
        if result is None:
            return None

        # Attach AI metadata to result
        result.ai_decision    = analysis.decision
        result.ai_confidence  = analysis.confidence
        result.ai_reasoning   = analysis.reasoning

        # --- 4. Register position ------------------------------------
        risk_manager.open_position(order)
        self._open_trades[signal.symbol] = result

        # --- 5. Audit log --------------------------------------------
        trade_log.log_trade({
            **result.as_csv_row(),
            "ai_reasoning": analysis.reasoning[:200],
        })
        return result

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[TradeResult]:
        result = self._open_trades.pop(symbol, None)
        if result is None:
            log.warning("No open trade found for %s", symbol)
            return None

        self._engine.close(result, exit_price, reason)
        pnl = risk_manager.close_position(symbol, exit_price)
        result.pnl = round(pnl, 2)

        self._closed_trades.append(result)
        trade_log.log_trade({**result.as_csv_row(), "ai_reasoning": ""})
        return result

    def close_all(self, prices: Dict[str, float], reason: str = "eod") -> None:
        """Close every open position — used at market close."""
        for sym in list(self._open_trades.keys()):
            price = prices.get(sym, 0.0)
            if price > 0:
                self.close_trade(sym, price, reason)

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------

    def monitor_positions(self, prices: Dict[str, float]) -> None:
        """Check exits and update trailing stops for all open positions."""
        # Update trailing stops first
        risk_manager.update_trailing_stops(prices)

        # Check if any position hit SL or target
        exits = risk_manager.check_exits(prices)
        for sym in exits:
            price = prices.get(sym, 0.0)
            pos   = risk_manager.get_position(sym)
            if pos is None:
                continue
            if pos.is_stoploss_hit(price):
                self.close_trade(sym, price, "stoploss")
            elif pos.is_target_hit(price):
                self.close_trade(sym, price, "target")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def open_trades(self) -> Dict[str, TradeResult]:
        return self._open_trades

    @property
    def closed_trades(self) -> List[TradeResult]:
        return self._closed_trades

    def performance_summary(self) -> dict:
        closed = self._closed_trades
        if not closed:
            return {"message": "No closed trades yet."}

        wins   = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in closed)
        win_rate  = len(wins) / len(closed) * 100 if closed else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
            "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
            "open_positions": len(self._open_trades),
            **risk_manager.summary(),
        }


# Singleton
trader = Trader()
