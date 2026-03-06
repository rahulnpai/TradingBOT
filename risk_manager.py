"""
risk_manager.py — Position sizing, daily loss limit, trade validation.

Rules enforced:
  1. Max 2 % of capital per trade.
  2. Max open positions ≤ config.risk.max_open_positions.
  3. Daily loss limit: if realised P&L < -3 % of capital, halt trading.
  4. No duplicate positions in the same symbol.
  5. Computed quantity is always at least 1 share.
  6. Trade cooldown: 15 min no-trade window after a stop-loss exit.
  7. Max trades per symbol: default 3 entries per symbol per day.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from config import config
from logger import get_logger
from strategy import Signal

log = get_logger("RiskManager")


# ---------------------------------------------------------------------------
# TradeOrder — what gets sent to the execution layer
# ---------------------------------------------------------------------------

@dataclass
class TradeOrder:
    symbol: str
    action: str           # "BUY" | "SELL"
    quantity: int
    entry_price: float
    stoploss: float
    target: float
    product: str          # "MIS" (intraday) | "CNC" (delivery)
    strategy_type: str
    confidence: float
    reason: str
    valid: bool = True
    reject_reason: str = ""

    def capital_required(self) -> float:
        return self.quantity * self.entry_price

    def risk_amount(self) -> float:
        if self.action == "BUY":
            return self.quantity * (self.entry_price - self.stoploss)
        return self.quantity * (self.stoploss - self.entry_price)


# ---------------------------------------------------------------------------
# Open-position tracker
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    symbol: str
    action: str
    quantity: int
    entry_price: float
    stoploss: float
    target: float
    strategy_type: str
    highest_price: float = 0.0   # for trailing SL (long)
    lowest_price: float  = 0.0   # for trailing SL (short)

    def __post_init__(self) -> None:
        self.highest_price = self.entry_price
        self.lowest_price  = self.entry_price

    def unrealised_pnl(self, current_price: float) -> float:
        if self.action == "BUY":
            return self.quantity * (current_price - self.entry_price)
        return self.quantity * (self.entry_price - current_price)

    def update_trailing_sl(self, current_price: float) -> None:
        tsl_pct = config.risk.trailing_sl_pct
        if self.action == "BUY":
            if current_price > self.highest_price:
                self.highest_price = current_price
                new_sl = round(self.highest_price * (1 - tsl_pct), 2)
                if new_sl > self.stoploss:
                    old = self.stoploss
                    self.stoploss = new_sl
                    log.debug("Trailing SL raised: %s  %.2f → %.2f",
                              self.symbol, old, self.stoploss)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                new_sl = round(self.lowest_price * (1 + tsl_pct), 2)
                if new_sl < self.stoploss:
                    old = self.stoploss
                    self.stoploss = new_sl
                    log.debug("Trailing SL lowered: %s  %.2f → %.2f",
                              self.symbol, old, self.stoploss)

    def is_stoploss_hit(self, current_price: float) -> bool:
        if self.action == "BUY":
            return current_price <= self.stoploss
        return current_price >= self.stoploss

    def is_target_hit(self, current_price: float) -> bool:
        if self.action == "BUY":
            return current_price >= self.target
        return current_price <= self.target


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:

    def __init__(self) -> None:
        self._rcfg         = config.risk
        self._capital      = self._rcfg.capital
        self._daily_pnl    = 0.0
        self._realised_pnl = 0.0
        self._total_pnl    = 0.0
        self._total_trades = 0
        self._total_wins   = 0
        self._today        = date.today()
        self._positions: Dict[str, OpenPosition] = {}
        self._halted       = False

        # ---- professional risk controls ----------------------------------
        # Cooldown: symbol → datetime of last SL exit
        self._last_exit_time: Dict[str, datetime] = {}
        # Max trades per symbol per day
        self._trade_count_today: Dict[str, int] = {}

        os.makedirs(config.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_trade(self, signal: Signal) -> TradeOrder:
        """Convert a Signal into a validated TradeOrder (or a rejected one)."""
        self._reset_if_new_day()

        for check in (
            self._check_halt,
            self._check_cooldown,
            self._check_duplicate,
            self._check_max_positions,
            self._check_max_trades_symbol,
        ):
            reject = check(signal)
            if reject:
                return reject

        quantity = self._compute_quantity(signal)
        if quantity < 1:
            return self._reject(signal, "Computed quantity < 1 (capital too small?)")

        product = "MIS" if signal.strategy_type == "intraday" else "CNC"

        order = TradeOrder(
            symbol=signal.symbol,
            action=signal.action,
            quantity=quantity,
            entry_price=signal.entry_price,
            stoploss=signal.suggested_sl,
            target=signal.suggested_target,
            product=product,
            strategy_type=signal.strategy_type,
            confidence=signal.confidence,
            reason=signal.reason,
        )
        log.info(
            "Trade validated: %s %s x%d @ %.2f  SL=%.2f  TGT=%.2f  "
            "risk=₹%.0f",
            order.action, order.symbol, order.quantity,
            order.entry_price, order.stoploss, order.target,
            order.risk_amount(),
        )
        return order

    def open_position(self, order: TradeOrder) -> None:
        pos = OpenPosition(
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            entry_price=order.entry_price,
            stoploss=order.stoploss,
            target=order.target,
            strategy_type=order.strategy_type,
        )
        self._positions[order.symbol] = pos

        # Track trade count per symbol
        self._trade_count_today[order.symbol] = (
            self._trade_count_today.get(order.symbol, 0) + 1
        )
        log.info(
            "Position opened: %s %s x%d  (trades today in %s: %d)",
            order.action, order.symbol, order.quantity,
            order.symbol, self._trade_count_today[order.symbol],
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position, update P&L, return realised P&L."""
        # Inspect position BEFORE removing — needed for SL cooldown detection
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0

        sl_hit = pos.is_stoploss_hit(exit_price)

        self._positions.pop(symbol)
        pnl = pos.unrealised_pnl(exit_price)
        self._daily_pnl    += pnl
        self._realised_pnl += pnl
        self._total_pnl    += pnl
        self._total_trades += 1
        if pnl > 0:
            self._total_wins += 1

        log.info(
            "Position closed: %s %s  pnl=₹%.2f  daily_pnl=₹%.2f",
            pos.action, symbol, pnl, self._daily_pnl,
        )

        # ---- Cooldown: record exit time on SL hit ----------------------
        if sl_hit:
            self._last_exit_time[symbol] = datetime.now()
            log.info(
                "COOLDOWN SET: %s — no new trades for %d min",
                symbol, self._rcfg.cooldown_minutes,
            )

        # ---- Daily loss kill switch ------------------------------------
        limit = -(self._capital * self._rcfg.daily_loss_limit_pct)
        if self._daily_pnl < limit:
            self._halted = True
            log.critical(
                "DAILY LOSS LIMIT REACHED — TRADING HALTED. "
                "daily_pnl=₹%.2f  limit=₹%.2f",
                self._daily_pnl, limit,
            )

        # ---- Equity snapshot to CSV ------------------------------------
        self._write_equity_snapshot()

        return pnl

    def update_trailing_stops(self, prices: Dict[str, float]) -> None:
        if not config.risk.trailing_sl:
            return
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.update_trailing_sl(prices[sym])

    def check_exits(self, prices: Dict[str, float]) -> List[str]:
        """Return list of symbols that hit SL or target."""
        to_exit = []
        for sym, pos in self._positions.items():
            price = prices.get(sym, 0.0)
            if price <= 0:
                continue
            if pos.is_stoploss_hit(price):
                log.warning("STOPLOSS HIT: %s @ %.2f (sl=%.2f)", sym, price, pos.stoploss)
                to_exit.append(sym)
            elif pos.is_target_hit(price):
                log.info("TARGET HIT: %s @ %.2f (tgt=%.2f)", sym, price, pos.target)
                to_exit.append(sym)
        return to_exit

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    @property
    def open_positions(self) -> Dict[str, OpenPosition]:
        return self._positions

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def realised_pnl(self) -> float:
        return self._realised_pnl

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def available_capital(self) -> float:
        committed = sum(
            p.quantity * p.entry_price for p in self._positions.values()
        )
        return max(0.0, self._capital - committed)

    def summary(self) -> dict:
        win_rate = (self._total_wins / self._total_trades * 100
                    if self._total_trades > 0 else 0.0)
        return {
            "capital": self._capital,
            "available_capital": round(self.available_capital, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "realised_pnl": round(self._realised_pnl, 2),
            "total_pnl": round(self._total_pnl, 2),
            "total_trades": self._total_trades,
            "win_rate_pct": round(win_rate, 2),
            "open_positions": len(self._positions),
            "halted": self._halted,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_quantity(self, signal: Signal) -> int:
        max_capital = self._capital * self._rcfg.max_capital_per_trade_pct
        if signal.entry_price <= 0:
            return 0
        qty = int(max_capital // signal.entry_price)
        return max(0, qty)

    def _check_halt(self, signal: Signal) -> Optional[TradeOrder]:
        if self._halted:
            return self._reject(signal, "Trading halted — daily loss limit breached")
        return None

    def _check_cooldown(self, signal: Signal) -> Optional[TradeOrder]:
        """Block new trades in a symbol for cooldown_minutes after a SL exit."""
        last_exit = self._last_exit_time.get(signal.symbol)
        if last_exit is None:
            return None
        elapsed_min = (datetime.now() - last_exit).total_seconds() / 60.0
        remaining   = self._rcfg.cooldown_minutes - elapsed_min
        if remaining > 0:
            return self._reject(
                signal,
                f"Cooldown active: {signal.symbol} — {remaining:.0f} min remaining after SL exit",
            )
        return None

    def _check_duplicate(self, signal: Signal) -> Optional[TradeOrder]:
        if self.has_position(signal.symbol):
            return self._reject(signal, f"Already have an open position in {signal.symbol}")
        return None

    def _check_max_positions(self, signal: Signal) -> Optional[TradeOrder]:
        if len(self._positions) >= self._rcfg.max_open_positions:
            return self._reject(
                signal, f"Max open positions ({self._rcfg.max_open_positions}) reached"
            )
        return None

    def _check_max_trades_symbol(self, signal: Signal) -> Optional[TradeOrder]:
        """Enforce max trades per symbol per day."""
        count = self._trade_count_today.get(signal.symbol, 0)
        if count >= self._rcfg.max_trades_per_symbol:
            return self._reject(
                signal,
                f"Max trades per symbol ({self._rcfg.max_trades_per_symbol}) reached for {signal.symbol} today",
            )
        return None

    @staticmethod
    def _reject(signal: Signal, reason: str) -> TradeOrder:
        log.warning("Trade REJECTED: %s %s — %s", signal.action, signal.symbol, reason)
        return TradeOrder(
            symbol=signal.symbol,
            action=signal.action,
            quantity=0,
            entry_price=signal.entry_price,
            stoploss=signal.suggested_sl,
            target=signal.suggested_target,
            product="MIS",
            strategy_type=signal.strategy_type,
            confidence=signal.confidence,
            reason=signal.reason,
            valid=False,
            reject_reason=reason,
        )

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if today != self._today:
            log.info("New trading day — resetting daily counters.")
            self._daily_pnl         = 0.0
            self._halted            = False
            self._today             = today
            self._trade_count_today = {}
            self._last_exit_time    = {}

    def _write_equity_snapshot(self) -> None:
        """Append an equity summary row to data/trade_log.csv after each close."""
        win_rate = (self._total_wins / self._total_trades * 100
                    if self._total_trades > 0 else 0.0)
        row = {
            "timestamp":     datetime.now().isoformat(),
            "daily_pnl":     round(self._daily_pnl, 2),
            "total_pnl":     round(self._total_pnl, 2),
            "trade_count":   self._total_trades,
            "win_rate_pct":  round(win_rate, 2),
            "capital":       round(self._capital, 2),
        }
        path         = os.path.join(config.data_dir, "trade_log.csv")
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            log.warning("Failed to write equity snapshot: %s", e)


# Singleton
risk_manager = RiskManager()
