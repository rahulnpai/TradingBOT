"""
backtest.py — Vectorised historical backtester.

Walk-forward simulation:
  For each symbol in the watchlist:
    1. Fetch daily candles for the backtest window.
    2. Roll through candles, computing indicators on each window.
    3. Generate signals with the strategy engine.
    4. Simulate fills with slippage + brokerage.
    5. Track open positions with SL / target exits.
  Aggregate across all symbols and print performance metrics.

Run standalone:
  python backtest.py [--start 2024-01-01] [--end 2024-12-31] [--capital 100000]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config import config
from data_fetcher import DataFetcher
from indicators import add_all_indicators
from logger import get_logger
from strategy import Signal, StrategyEngine

log = get_logger("Backtest")


# ---------------------------------------------------------------------------
# Backtest trade record
# ---------------------------------------------------------------------------

@dataclass
class BTrade:
    symbol: str
    action: str
    entry_date: date
    entry_price: float
    stoploss: float
    target: float
    quantity: int
    strategy_type: str
    signal_confidence: float
    exit_date: Optional[date]  = None
    exit_price: float          = 0.0
    exit_reason: str           = ""     # "target" | "stoploss" | "eod" | "strategy"
    pnl: float                 = 0.0
    pnl_pct: float             = 0.0
    brokerage: float           = 0.0

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.brokerage * 2   # entry + exit

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str]    = None,
        end_date: Optional[str]      = None,
        initial_capital: float       = 100_000.0,
    ) -> None:
        bcfg = config.backtest
        self._symbols  = symbols or config.trading.symbols
        self._start    = pd.Timestamp(start_date or bcfg.start_date)
        self._end      = pd.Timestamp(end_date   or bcfg.end_date)
        self._capital  = initial_capital or bcfg.initial_capital
        self._brok     = bcfg.brokerage_per_trade
        self._slippage = bcfg.slippage_pct
        self._max_pos  = config.risk.max_open_positions
        self._max_trade_cap_pct = config.risk.max_capital_per_trade_pct
        self._sl_pct   = config.risk.default_sl_pct
        self._tgt_pct  = config.risk.default_target_pct

        self._fetcher  = DataFetcher()
        self._strategy = StrategyEngine()
        self._trades: List[BTrade] = []

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        log.info(
            "Backtest starting: %d symbols  %s -> %s  capital=Rs %.0f",
            len(self._symbols), self._start.date(), self._end.date(), self._capital,
        )
        all_trade_rows = []

        for symbol in self._symbols:
            log.info("Processing %s ...", symbol)
            try:
                trades = self._backtest_symbol(symbol)
                all_trade_rows.extend(trades)
            except Exception as e:
                log.error("Backtest failed for %s: %s", symbol, e)

        self._trades = all_trade_rows
        df = self._to_dataframe(all_trade_rows)
        return df

    # ------------------------------------------------------------------

    def _backtest_symbol(self, symbol: str) -> List[BTrade]:
        # Fetch daily candles; use extra days at start for indicator warm-up
        lookback_days = int((self._end - self._start).days + 300)
        raw = self._fetcher.get_historical(
            symbol, interval="day", days=lookback_days, use_cache=True
        )
        if raw.empty:
            log.warning("No data for %s", symbol)
            return []

        raw = add_all_indicators(raw)

        # Align timezone: if index is tz-aware, make start/end tz-aware too
        import pytz as _pytz
        _IST = _pytz.timezone("Asia/Kolkata")
        start = self._start
        end   = self._end
        if raw.index.tzinfo is not None:
            start = start.tz_localize(_IST) if start.tzinfo is None else start
            end   = end.tz_localize(_IST)   if end.tzinfo is None else end
        raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
        if len(raw) < 50:
            log.warning("Insufficient data for %s after filtering", symbol)
            return []

        trades: List[BTrade] = []
        open_trade: Optional[BTrade] = None
        capital     = self._capital
        daily_pnl   = 0.0
        day_of_pnl  = None

        for i in range(50, len(raw)):
            candle = raw.iloc[i]
            today  = candle.name.date() if hasattr(candle.name, "date") else candle.name
            price  = float(candle["close"])

            # Reset daily P&L tracker
            if day_of_pnl != today:
                day_of_pnl = today
                daily_pnl  = 0.0

            # ---- Monitor open trade --------------------------------
            if open_trade is not None:
                high_price = float(candle["high"])
                low_price  = float(candle["low"])

                exit_price  = None
                exit_reason = None

                if open_trade.action == "BUY":
                    if low_price  <= open_trade.stoploss:
                        exit_price, exit_reason = open_trade.stoploss, "stoploss"
                    elif high_price >= open_trade.target:
                        exit_price, exit_reason = open_trade.target, "target"
                else:
                    if high_price >= open_trade.stoploss:
                        exit_price, exit_reason = open_trade.stoploss, "stoploss"
                    elif low_price  <= open_trade.target:
                        exit_price, exit_reason = open_trade.target, "target"

                if exit_price is not None:
                    open_trade = self._close_trade(
                        open_trade, exit_price, exit_reason, today
                    )
                    daily_pnl += open_trade.net_pnl
                    capital   += open_trade.net_pnl
                    trades.append(open_trade)
                    open_trade = None

            # ---- Generate new signal (only if no open trade) -------
            if open_trade is None:
                df_window = raw.iloc[: i + 1]
                signal: Signal = self._strategy.generate_signal(
                    symbol, df_window, mode="swing"
                )

                if signal.is_actionable and signal.confidence >= 0.6:
                    qty = max(1, int(capital * self._max_trade_cap_pct / price))
                    fill = price * (
                        1 + self._slippage if signal.action == "BUY"
                        else 1 - self._slippage
                    )
                    sl_mult  = (1 - self._sl_pct)  if signal.action == "BUY" else (1 + self._sl_pct)
                    tgt_mult = (1 + self._tgt_pct) if signal.action == "BUY" else (1 - self._tgt_pct)

                    open_trade = BTrade(
                        symbol=symbol,
                        action=signal.action,
                        entry_date=today,
                        entry_price=round(fill, 2),
                        stoploss=round(fill * sl_mult,  2),
                        target=round(fill * tgt_mult, 2),
                        quantity=qty,
                        strategy_type=signal.strategy_type,
                        signal_confidence=signal.confidence,
                        brokerage=self._brok,
                    )

        # Force close any still-open trade at end of period
        if open_trade is not None:
            last_price = float(raw.iloc[-1]["close"])
            open_trade = self._close_trade(open_trade, last_price, "eod",
                                           raw.index[-1].date())
            capital   += open_trade.net_pnl
            trades.append(open_trade)

        return trades

    @staticmethod
    def _close_trade(
        t: BTrade, exit_price: float, reason: str, exit_date
    ) -> BTrade:
        t.exit_date   = exit_date
        t.exit_price  = round(exit_price, 2)
        t.exit_reason = reason

        if t.action == "BUY":
            t.pnl = (t.exit_price - t.entry_price) * t.quantity
        else:
            t.pnl = (t.entry_price - t.exit_price) * t.quantity

        t.pnl_pct = (t.pnl / (t.entry_price * t.quantity)) * 100
        return t

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def print_metrics(self) -> None:
        if not self._trades:
            log.warning("No trades to report.")
            return

        df = self._to_dataframe(self._trades)
        wins   = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] <= 0]

        total_pnl    = df["net_pnl"].sum()
        gross_profit = wins["net_pnl"].sum()
        gross_loss   = abs(losses["net_pnl"].sum())
        pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        win_rate     = len(wins) / len(df) * 100

        # Equity curve for drawdown
        eq_curve = self._capital + df.sort_values("exit_date")["net_pnl"].cumsum()
        running_max = eq_curve.cummax()
        drawdown    = (eq_curve - running_max) / running_max * 100
        max_dd      = drawdown.min()

        # Sharpe (daily returns approx)
        daily_returns = df.groupby("exit_date")["net_pnl"].sum() / self._capital
        sharpe = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 0 else 0.0
        )

        rows = [
            ["Total Trades",   len(df)],
            ["Win Rate",       f"{win_rate:.1f}%"],
            ["Total Net P&L",  f"₹{total_pnl:,.2f}"],
            ["Return on Cap",  f"{total_pnl/self._capital*100:.2f}%"],
            ["Profit Factor",  f"{pf:.3f}"],
            ["Sharpe Ratio",   f"{sharpe:.3f}"],
            ["Max Drawdown",   f"{max_dd:.2f}%"],
            ["Avg Win",        f"₹{wins['net_pnl'].mean():,.2f}" if not wins.empty else "N/A"],
            ["Avg Loss",       f"₹{losses['net_pnl'].mean():,.2f}" if not losses.empty else "N/A"],
            ["Best Trade",     f"₹{df['net_pnl'].max():,.2f}"],
            ["Worst Trade",    f"₹{df['net_pnl'].min():,.2f}"],
        ]
        print("\n" + "="*40)
        print("  BACKTEST RESULTS")
        print("="*40)
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))
        print()

        # Per-symbol breakdown
        sym_df = (
            df.groupby("symbol")
            .agg(trades=("net_pnl", "count"),
                 net_pnl=("net_pnl", "sum"),
                 win_rate=("is_win", "mean"))
            .reset_index()
        )
        sym_df["win_rate"] = (sym_df["win_rate"] * 100).round(1)
        sym_df["net_pnl"]  = sym_df["net_pnl"].round(2)
        print(tabulate(sym_df, headers="keys", tablefmt="rounded_grid", showindex=False))

    def save_report(self, out_dir: str = "data") -> None:
        if not self._trades:
            return
        os.makedirs(out_dir, exist_ok=True)
        df = self._to_dataframe(self._trades)
        path = os.path.join(out_dir, f"backtest_{date.today().isoformat()}.csv")
        df.to_csv(path, index=False)
        log.info("Backtest report saved: %s", path)
        self._plot_equity_curve(df, out_dir)

    def _plot_equity_curve(self, df: pd.DataFrame, out_dir: str) -> None:
        try:
            df_sorted = df.sort_values("exit_date")
            equity = self._capital + df_sorted["net_pnl"].cumsum()
            dates  = pd.to_datetime(df_sorted["exit_date"])

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig.suptitle("Backtest — Equity Curve & Drawdown", fontsize=14)

            axes[0].plot(dates, equity, color="steelblue", linewidth=1.5)
            axes[0].axhline(self._capital, color="grey", linestyle="--", linewidth=0.8)
            axes[0].set_ylabel("Portfolio Value (₹)")
            axes[0].grid(True, alpha=0.3)

            running_max = equity.cummax()
            dd = (equity - running_max) / running_max * 100
            axes[1].fill_between(dates, dd, 0, color="tomato", alpha=0.5)
            axes[1].set_ylabel("Drawdown (%)")
            axes[1].set_xlabel("Date")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(out_dir, f"equity_curve_{date.today().isoformat()}.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            log.info("Equity curve saved: %s", path)
        except Exception as e:
            log.warning("Could not generate equity curve plot: %s", e)

    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(trades: List[BTrade]) -> pd.DataFrame:
        rows = []
        for t in trades:
            rows.append({
                "symbol":       t.symbol,
                "action":       t.action,
                "entry_date":   t.entry_date,
                "exit_date":    t.exit_date,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "quantity":     t.quantity,
                "pnl":          round(t.pnl, 2),
                "brokerage":    round(t.brokerage * 2, 2),
                "net_pnl":      round(t.net_pnl, 2),
                "pnl_pct":      round(t.pnl_pct, 3),
                "is_win":       t.is_win,
                "exit_reason":  t.exit_reason,
                "confidence":   t.signal_confidence,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Trader backtest")
    parser.add_argument("--start",   default=config.backtest.start_date)
    parser.add_argument("--end",     default=config.backtest.end_date)
    parser.add_argument("--capital", type=float, default=config.backtest.initial_capital)
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Space-separated symbol list (default: watchlist from config)")
    args = parser.parse_args()

    bt = Backtester(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )
    trade_df = bt.run()
    bt.print_metrics()
    bt.save_report()
