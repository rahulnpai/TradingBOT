"""
main.py — AI Trader entry point and main event loop.

Scheduling:
  • Every N minutes (intraday_interval) during market hours (sim/live):
      fetch → indicators → signal → AI → risk → execute
  • Paper mode: runs every 5 minutes, 24/7, using historical candles.
  • At market open: initialise data, print daily header.
  • At intraday exit time (15:10): force-close all MIS positions (sim/live only).
  • At market close: daily P&L report, persist state.

Usage:
  python main.py paper          ← Yahoo data, paper fills, 24/7
  python main.py sim            ← Zerodha data, paper fills, market hours
  python main.py live           ← Zerodha data, real orders, market hours

  Legacy flags still work:
  python main.py --paper
  python main.py --live
  python main.py --mode intraday|swing
  python main.py --run-now
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
from apscheduler.triggers.interval import IntervalTrigger
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


def _is_market_open() -> bool:
    now = _now_ist()
    if now.weekday() >= 5:          # Saturday / Sunday
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
    mode_label = "PAPER" if config.trading.paper_trading else "LIVE "
    provider   = config.data_provider.upper()
    log.print(
        "\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]"
        "\n[bold cyan]║     AI TRADER  —  Zerodha KiteConnect    ║[/bold cyan]"
        "\n[bold cyan]║          Indian Stock Market Bot         ║[/bold cyan]"
        f"\n[bold cyan]║  Mode: {mode_label}  Provider: {provider:8s}         ║[/bold cyan]"
        f"\n[bold cyan]║  Strategy: {config.trading.trading_mode:10s}                   ║[/bold cyan]"
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
    # Paper mode runs 24/7 on historical candles; skip market-hours gate
    if not config.trading.paper_trading and not _is_market_open():
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

    logger.info("=== Trading cycle | %s | %s | %s ===",
                mode, config.data_provider, _now_ist().strftime("%H:%M:%S"))

    # ---- 1. Fetch current prices for position monitoring ---------------
    prices = data_fetcher.get_batch_quotes(symbols)
    if prices:
        trader.monitor_positions(prices)

    # ---- 2. Handle intraday exit window (sim/live only) ----------------
    if mode == "intraday" and not config.trading.paper_trading and _is_intraday_exit_window():
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
    # Use config.history_days to ensure EMA200 / VWAP have enough data
    df = data_fetcher.get_historical(symbol, interval,
                                     days=config.history_days, use_cache=False)
    if df.empty or len(df) < 50:
        logger.debug("Insufficient data for %s", symbol)
        return

    # Calculate indicators
    df = add_all_indicators(df)

    # Generate signal
    signal = strategy_engine.generate_signal(symbol, df, mode=mode)

    if not signal.is_actionable:
        logger.debug("%s: HOLD (%s)", symbol, signal.reason)
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
        color = "green" if result.action == "BUY" else "red"
        log.print(
            f"[bold {color}]"
            f"✓ TRADE EXECUTED: {result.action} {result.symbol} x{result.quantity}"
            f" @ ₹{result.fill_price:.2f}  "
            f"SL=₹{result.stoploss:.2f}  TGT=₹{result.target:.2f}"
            f"[/bold {color}]"
        )


# ---------------------------------------------------------------------------
# Market open / close callbacks
# ---------------------------------------------------------------------------

def on_market_open() -> None:
    logger.info("Market opened — initialising day.")
    _print_banner()
    # Pre-warm instrument cache for Zerodha modes
    if config.data_provider == "zerodha" and kite_client._kite:
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

def _build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone=IST)

    # Market open (Mon–Fri 09:15) — always schedule for daily header / cache warm
    sched.add_job(on_market_open, CronTrigger(
        day_of_week="mon-fri", hour=9, minute=15, timezone=IST
    ), id="market_open")

    # Market close (Mon–Fri 15:30)
    sched.add_job(on_market_close, CronTrigger(
        day_of_week="mon-fri", hour=15, minute=30, timezone=IST
    ), id="market_close")

    if config.trading.paper_trading:
        # Paper / sim: run every 5 minutes, 24/7 (historical candles always available)
        sched.add_job(trading_cycle, IntervalTrigger(minutes=5), id="paper_cycle")
        logger.info("Scheduler: 24/7 paper cycle (every 5 min).")
    else:
        # Live: run only during market hours
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

    # Positional mode argument (new style)
    parser.add_argument(
        "mode", nargs="?",
        choices=["paper", "sim", "live"],
        default=None,
        help="Runtime mode: paper (Yahoo, no orders) | sim (Zerodha, no orders) | live (Zerodha, real orders)",
    )

    # Legacy flags kept for backward compatibility
    mode_g = parser.add_mutually_exclusive_group()
    mode_g.add_argument("--paper", action="store_true", help="Paper trading mode (legacy flag)")
    mode_g.add_argument("--live",  action="store_true", help="Live trading — real orders (legacy flag)")

    parser.add_argument("--mode-strategy", dest="mode_strategy",
                        choices=["intraday", "swing"], default=None,
                        help="Override trading strategy from config")
    parser.add_argument("--run-now", action="store_true",
                        help="Run one trading cycle immediately (bypass scheduler)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Apply mode → config mappings
    # ----------------------------------------------------------------
    if args.mode == "paper" or args.paper:
        config.data_provider          = "yahoo"
        config.trading.paper_trading  = True

    elif args.mode == "sim":
        config.data_provider          = "zerodha"
        config.trading.paper_trading  = True

    elif args.mode == "live" or args.live:
        config.data_provider          = "zerodha"
        config.trading.paper_trading  = False

    else:
        # Default: paper mode (safe default, no credentials needed)
        config.data_provider         = "yahoo"
        config.trading.paper_trading = True

    # Keep risk.capital in sync after any config change
    config.risk.capital = config.trading.capital

    if args.mode_strategy:
        config.trading.trading_mode = args.mode_strategy

    # ----------------------------------------------------------------
    # Validate config
    # ----------------------------------------------------------------
    try:
        config.validate()
    except ValueError as e:
        logger.critical("Config validation failed: %s", e)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Startup log — required by spec
    # ----------------------------------------------------------------
    runtime_mode = args.mode or ("live" if not config.trading.paper_trading else
                                 ("sim" if config.data_provider == "zerodha" else "paper"))
    _print_banner()
    logger.info("=" * 52)
    logger.info("Running mode    : %s", runtime_mode.upper())
    logger.info("Data provider   : %s", config.data_provider)
    logger.info("Paper trading   : %s", config.trading.paper_trading)
    logger.info("Capital         : Rs %.0f", config.risk.capital)
    logger.info("Symbols         : %s", ", ".join(config.trading.symbols))
    logger.info("History days    : %d", config.history_days)
    logger.info("Trading strategy: %s", config.trading.trading_mode)
    logger.info("=" * 52)

    # ----------------------------------------------------------------
    # Initialise Kite client ONLY when provider is zerodha
    # ----------------------------------------------------------------
    if config.data_provider == "zerodha":
        try:
            kite_client.initialise()
        except Exception as e:
            if config.trading.paper_trading:
                logger.warning("Kite init failed (sim mode — continuing): %s", e)
            else:
                logger.critical("Kite init failed (live mode): %s", e)
                sys.exit(1)
    else:
        logger.info("Data provider: Yahoo Finance — Kite client not initialised.")

    # ----------------------------------------------------------------
    # AI health check
    # ----------------------------------------------------------------
    if config.ai.enabled:
        available = ai_engine.is_available()
        status = "[green]ONLINE[/green]" if available else "[yellow]OFFLINE — technical signals only[/yellow]"
        logger.info("Ollama AI (%s): %s", config.ai.model, status)

    # ----------------------------------------------------------------
    # One-shot mode for testing / debugging
    # ----------------------------------------------------------------
    if args.run_now:
        logger.info("Running immediate trading cycle ...")
        on_market_open()
        trading_cycle()
        _print_daily_summary()
        return

    # ----------------------------------------------------------------
    # Start scheduler
    # ----------------------------------------------------------------
    sched = _build_scheduler()
    _scheduler_ref.append(sched)
    _signal.signal(_signal.SIGINT,  _handle_sigint)
    _signal.signal(_signal.SIGTERM, _handle_sigint)

    if config.trading.paper_trading:
        logger.info("Scheduler started. Running 24/7 in paper mode. (Ctrl+C to stop)")
    else:
        logger.info("Scheduler started. Waiting for market hours... (Ctrl+C to stop)")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
