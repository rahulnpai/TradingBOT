"""
main.py — AI Trader entry point and main event loop.

Scheduling:
  • Every N minutes (intraday_interval) during market hours:
      fetch → indicators → signal → AI → risk → execute
  • At market open: initialise data, print daily header.
  • At intraday exit time (15:10): force-close all MIS positions.
  • At market close: daily P&L report, persist state.

Usage:
  python main.py [--mode intraday|swing] [--paper|--live]
"""

from __future__ import annotations

import argparse
import signal as _signal
import sys
import time
from datetime import datetime, time as dtime
from typing import Dict, List

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console
from rich.table import Table
from tabulate import tabulate

from ai_engine import ai_engine
from config import config
from data_fetcher import data_fetcher
from indicators import add_all_indicators
from kite_client import kite_client
from logger import get_logger
from risk_manager import risk_manager
from strategy import strategy_engine
from trader import trader

log     = Console()
logger  = get_logger("Main", log_file=config.log_file, level=config.log_level)
IST     = pytz.timezone("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(IST)


'''
def _is_market_open() -> bool:
    now = _now_ist()
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    open_t  = dtime(config.trading.market_open_h,  config.trading.market_open_m)
    close_t = dtime(config.trading.market_close_h, config.trading.market_close_m)
    return open_t <= now.time() <= close_t
'''
def _is_market_open() -> bool:
    """
    Market hours gate.
    In PAPER mode -> always allow trading.
    In LIVE mode  -> respect NSE timings.
    """

    #Allow trading anytime in paper mode
    if config.trading.paper_trading:
        return True

    now = _now_ist()

    # Weekend check
    if now.weekday() >= 5:  # Saturday / Sunday
        return False

    open_t  = dtime(config.trading.market_open_h,  config.trading.market_open_m)
    close_t = dtime(config.trading.market_close_h, config.trading.market_close_m)

    return open_t <= now.time() <= close_t

def _is_intraday_exit_window() -> bool:
    now    = _now_ist()
    exit_t = dtime(config.trading.intraday_exit_h, config.trading.intraday_exit_m)
    close_t = dtime(config.trading.market_close_h, config.trading.market_close_m)
    return exit_t <= now.time() <= close_t


def _print_banner() -> None:
    log.print(
        "\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]"
        "\n[bold cyan]║     AI TRADER  —  Zerodha KiteConnect    ║[/bold cyan]"
        "\n[bold cyan]║          Indian Stock Market Bot         ║[/bold cyan]"
        f"\n[bold cyan]║  Mode: {'PAPER' if config.trading.paper_trading else 'LIVE ':5s}"
        f"  Strategy: {config.trading.trading_mode:10s}  ║[/bold cyan]"
        "\n[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]\n"
    )


def _print_positions() -> None:
    positions = risk_manager.open_positions
    if not positions:
        logger.info("No open positions.")
        return

    table = Table(title="Open Positions", style="cyan")
    for col in ("Symbol", "Side", "Qty", "Entry", "CMP", "SL", "Target", "Unreal P&L"):
        table.add_column(col, justify="right")

    prices = data_fetcher.get_batch_quotes(list(positions.keys()))
    for sym, pos in positions.items():
        cmp  = prices.get(sym, pos.entry_price)
        upnl = pos.unrealised_pnl(cmp)
        table.add_row(
            sym, pos.action, str(pos.quantity),
            f"₹{pos.entry_price:.2f}", f"₹{cmp:.2f}",
            f"₹{pos.stoploss:.2f}", f"₹{pos.target:.2f}",
            f"[green]₹{upnl:.2f}[/green]" if upnl >= 0 else f"[red]₹{upnl:.2f}[/red]",
        )
    log.print(table)


def _print_daily_summary() -> None:
    summary = trader.performance_summary()
    rows = [[k, v] for k, v in summary.items()]
    logger.info("Performance:\n" + tabulate(rows, headers=["Metric", "Value"],
                                            tablefmt="rounded_grid"))


# ---------------------------------------------------------------------------
# Core trading loop (called by scheduler every N minutes)
# ---------------------------------------------------------------------------

def trading_cycle() -> None:
    if not _is_market_open():
        logger.debug("Market closed — skipping cycle.")
        return

    if risk_manager.is_halted:
        logger.warning("Trading HALTED — daily loss limit reached.")
        return

    mode    = config.trading.trading_mode
    symbols = config.trading.symbols
    interval = (
        config.trading.intraday_interval if mode == "intraday"
        else config.trading.swing_interval
    )

    logger.info("=== Trading cycle | %s | %s ===", mode, _now_ist().strftime("%H:%M:%S"))

    # ---- 1. Fetch current prices for position monitoring ---------------
    prices = data_fetcher.get_batch_quotes(symbols)
    if prices:
        trader.monitor_positions(prices)

    # ---- 2. Handle intraday exit window --------------------------------
    if mode == "intraday" and _is_intraday_exit_window():
        logger.info("Intraday exit window — closing all MIS positions.")
        trader.close_all(prices, reason="eod_intraday")
        return

    # ---- 3. Scan watchlist for signals ---------------------------------
    for symbol in symbols:
        if risk_manager.has_position(symbol):
            continue   # already in this stock
        if risk_manager.is_halted:
            break

        try:
            _process_symbol(symbol, mode, interval)
        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e, exc_info=True)

    _print_positions()


def _process_symbol(symbol: str, mode: str, interval: str) -> None:
    # Fetch candles
    days = 30 if mode == "intraday" else 365
    df   = data_fetcher.get_historical(symbol, interval, days=days, use_cache=False)
    if df.empty or len(df) < 50:
        logger.debug("Insufficient data for %s", symbol)
        return

    # Calculate indicators
    df = add_all_indicators(df)

    # Generate signal
    signal = strategy_engine.generate_signal(symbol, df, mode=mode)

    if not signal.is_actionable:
        logger.info("%s: HOLD (%s)", symbol, signal.reason)
        return

    logger.info(
        "SIGNAL: %s %s  conf=%.2f  entry=%.2f  sl=%.2f  tgt=%.2f | %s",
        signal.action, symbol, signal.confidence,
        signal.entry_price, signal.suggested_sl,
        signal.suggested_target, signal.reason,
    )

    # Attempt trade
    result = trader.process_signal(signal)
    if result:
        log.print(
            f"[bold {'green' if result.action == 'BUY' else 'red'}]"
            f"✓ TRADE EXECUTED: {result.action} {result.symbol} x{result.quantity}"
            f" @ ₹{result.fill_price:.2f}  "
            f"SL=₹{result.stoploss:.2f}  TGT=₹{result.target:.2f}[/bold {'green' if result.action == 'BUY' else 'red'}]"
        )


# ---------------------------------------------------------------------------
# Market open / close callbacks
# ---------------------------------------------------------------------------

def on_market_open() -> None:
    logger.info("Market opened — initialising day.")
    _print_banner()
    # Pre-warm instrument cache
    if not config.trading.paper_trading and kite_client._kite:
        kite_client.preload_instruments()


def on_market_close() -> None:
    logger.info("Market closing — final clean-up.")
    # Force close all intraday positions
    symbols = list(trader.open_trades.keys())
    if symbols:
        prices = data_fetcher.get_batch_quotes(symbols)
        trader.close_all(prices, reason="market_close")

    _print_daily_summary()

    # Ask AI for strategy optimisation once per day
    if ai_engine.is_available():
        perf    = trader.performance_summary()
        trades  = [t.as_csv_row() for t in trader.closed_trades[-50:]]
        suggestions = ai_engine.suggest_optimisations(perf, trades)
        if suggestions:
            logger.info("AI Optimisation:\n%s", suggestions.get("overall_assessment", ""))


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------
'''
def _build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone=IST)

    # Market open (Mon–Fri 09:15)
    sched.add_job(on_market_open, CronTrigger(
        day_of_week="mon-fri", hour=9, minute=15, timezone=IST
    ), id="market_open")

    # Market close (Mon–Fri 15:30)
    sched.add_job(on_market_close, CronTrigger(
        day_of_week="mon-fri", hour=15, minute=30, timezone=IST
    ), id="market_close")

    # Intraday trading cycle — every 5 min during market hours
    if config.trading.trading_mode == "intraday":
        sched.add_job(trading_cycle, CronTrigger(
            day_of_week="mon-fri",
            hour="9-15", minute="*/5",
            timezone=IST,
        ), id="intraday_cycle")
    else:
        # Swing: run once at market open + once mid-session
        for h, m in [(9, 20), (12, 0)]:
            sched.add_job(trading_cycle, CronTrigger(
                day_of_week="mon-fri", hour=h, minute=m, timezone=IST
            ), id=f"swing_cycle_{h}{m}")

    return sched
'''
def _build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone=IST)

    # If paper mode → run every 1 minute continuously
    if config.trading.paper_trading:
        sched.add_job(
            trading_cycle,
            "interval",
            minutes=1,
            id="paper_cycle"
        )
        return sched

    # ================= REAL MARKET MODE =================

    # Market open (Mon–Fri 09:15)
    sched.add_job(on_market_open, CronTrigger(
        day_of_week="mon-fri", hour=9, minute=15, timezone=IST
    ), id="market_open")

    # Market close (Mon–Fri 15:30)
    sched.add_job(on_market_close, CronTrigger(
        day_of_week="mon-fri", hour=15, minute=30, timezone=IST
    ), id="market_close")

    # Intraday trading cycle — every 5 min during market hours
    if config.trading.trading_mode == "intraday":
        sched.add_job(trading_cycle, CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/5",
            timezone=IST,
        ), id="intraday_cycle")
    else:
        for h, m in [(9, 20), (12, 0)]:
            sched.add_job(trading_cycle, CronTrigger(
                day_of_week="mon-fri",
                hour=h,
                minute=m,
                timezone=IST
            ), id=f"swing_cycle_{h}{m}")

    return sched
# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_scheduler_ref: list = []


def _handle_sigint(signum, frame) -> None:
    logger.info("Interrupt received — shutting down gracefully.")
    if _scheduler_ref:
        _scheduler_ref[0].shutdown(wait=False)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader — Zerodha KiteConnect")
    mode_g = parser.add_mutually_exclusive_group()
    mode_g.add_argument("--paper", action="store_true",
                        help="Paper trading mode (default)")
    mode_g.add_argument("--live",  action="store_true",
                        help="Live trading (real orders)")
    parser.add_argument("--mode", choices=["intraday", "swing"], default=None,
                        help="Override trading mode from config")
    parser.add_argument("--run-now", action="store_true",
                        help="Run one trading cycle immediately (bypass scheduler)")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.live:
        config.trading.paper_trading = False
    if args.mode:
        config.trading.trading_mode = args.mode

    # Validate
    try:
        config.validate()
    except ValueError as e:
        logger.critical("Config validation failed: %s", e)
        sys.exit(1)

    _print_banner()
    logger.info("Paper trading: %s", config.trading.paper_trading)
    logger.info("Trading mode : %s", config.trading.trading_mode)
    logger.info("Symbols      : %s", ", ".join(config.trading.symbols))
    logger.info("Capital      : Rs %.0f", config.risk.capital)

    # Initialise Kite client
    if not config.trading.paper_trading or config.kite.api_key:
        try:
            kite_client.initialise()
        except Exception as e:
            if config.trading.paper_trading:
                logger.warning("Kite init failed (paper mode — continuing): %s", e)
            else:
                logger.critical("Kite init failed: %s", e)
                sys.exit(1)

    # AI health check
    if config.ai.enabled:
        available = ai_engine.is_available()
        logger.info(
            "Ollama AI (%s): %s",
            config.ai.model,
            "[green]ONLINE[/green]" if available else "[yellow]OFFLINE — technical signals only[/yellow]",
        )

    if args.run_now:
        # One-shot for testing / debugging
        logger.info("Running immediate trading cycle ...")
        on_market_open()
        trading_cycle()
        _print_daily_summary()
        return

    # Start scheduler
    sched = _build_scheduler()
    _scheduler_ref.append(sched)
    _signal.signal(_signal.SIGINT,  _handle_sigint)
    _signal.signal(_signal.SIGTERM, _handle_sigint)

    logger.info("Scheduler started. Waiting for market hours... (Ctrl+C to stop)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
