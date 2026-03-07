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
import logging
import signal as _signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime
from typing import Deque, Dict, List, Optional, Tuple

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tabulate import tabulate

from ai_engine import ai_engine
from config import config
from data_fetcher import data_fetcher
from indicators import add_all_indicators
from kite_client import kite_client
from logger import get_logger
import market_analytics as ma
from risk_manager import risk_manager
from strategy import strategy_engine
from trader import trader

log     = Console(stderr=False)
logger  = get_logger("Main", log_file=config.log_file, level=config.log_level)
IST     = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------

_signal_log:  Deque[str] = deque(maxlen=20)   # BUY/SELL signals → left panel
_log_buffer:  Deque[str] = deque(maxlen=20)   # system messages  → right panel
_status:      str        = "Initialising..."
_dashboard_layout: Optional[Layout] = None

# Last known prices — updated in trading_cycle, used for unrealized P&L display
_last_prices: Dict[str, float] = {}


class _DashboardLogHandler(logging.Handler):
    """Routes log records into the appropriate dashboard panel buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        ts  = datetime.now().strftime("%H:%M:%S")

        if "SIGNAL:" in msg:
            color = "bold green" if " BUY " in msg else ("bold red" if " SELL " in msg else "cyan")
            _signal_log.append(f"[{color}]{ts}[/{color}] {msg[:120]}")

        elif "TRADE EXECUTED" in msg:
            color = "bold green" if "BUY" in msg else "bold red"
            _signal_log.append(f"[{color}]{ts} {msg[:120]}[/{color}]")

        elif "HALTED" in msg or "HALT" in msg:
            _signal_log.append(f"[bold yellow]{ts} {msg[:120]}[/bold yellow]")
            _log_buffer.append(f"[bold yellow]{ts}[/bold yellow] {msg[:120]}")

        else:
            level_color = {
                logging.WARNING:  "yellow",
                logging.ERROR:    "red",
                logging.CRITICAL: "bold red",
            }.get(record.levelno, "dim white")
            _log_buffer.append(f"[{level_color}]{ts}[/{level_color}] {msg[:120]}")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _make_header_panel() -> Panel:
    mode_label = (
        "LIVE"  if not config.trading.paper_trading else
        "SIM"   if config.data_provider == "zerodha" else
        "PAPER"
    )
    mode_style = "bold green" if mode_label == "LIVE" else "bold yellow"
    capital    = f"\u20b9{config.trading.capital:,.0f}"
    strategy   = config.trading.trading_mode.upper()
    provider   = config.data_provider.upper()
    now        = datetime.now(IST).strftime("%H:%M:%S IST")
    regime     = ma.get_regime()
    regime_color = {"TRENDING": "bold green", "SIDEWAYS": "bold yellow",
                    "VOLATILE": "bold red"}.get(regime, "dim white")

    header = Text(justify="center")
    header.append("  AI TRADER  ",       style="bold cyan")
    header.append("\u2502 Mode: ",        style="dim white")
    header.append(f"{mode_label}",        style=mode_style)
    header.append("  \u2502  Capital: ",  style="dim white")
    header.append(f"{capital}",           style="bold white")
    header.append("  \u2502  Strategy: ", style="dim white")
    header.append(f"{strategy}",          style="bold cyan")
    header.append("  \u2502  Provider: ", style="dim white")
    header.append(f"{provider}",          style="bold white")
    header.append("  \u2502  Regime: ",   style="dim white")
    header.append(f"{regime}",            style=regime_color)
    header.append(f"  \u2502  {now}",     style="dim white")
    header.append("  \u2502  Status: ",   style="dim white")
    header.append(f"{_status}",           style="bold green")

    return Panel(header, border_style="cyan", padding=(0, 1))


def _make_market_status_panel() -> Panel:
    """Compact one-line bar: NIFTY | BANKNIFTY | VIX."""
    status = ma.get_index_status()

    parts: List[str] = []
    for name, (price, pct) in status.items():
        color = "green" if pct >= 0 else "red"
        sign  = "+" if pct >= 0 else ""
        if name == "VIX":
            parts.append(f"[bold]{name}:[/bold] {price:.1f}  [{color}]({sign}{pct:.2f}%)[/{color}]")
        else:
            parts.append(f"[bold]{name}:[/bold] {price:,.0f}  [{color}]({sign}{pct:.2f}%)[/{color}]")

    if not parts:
        text = "[dim]Fetching market indices...[/dim]"
    else:
        text = "   \u2502   ".join(parts)

    return Panel(text, border_style="dim cyan", padding=(0, 2))


def _make_daily_pnl_panel() -> Panel:
    """One-line performance bar above the positions table."""
    summary = risk_manager.summary()

    # Unrealized P&L — compute from cached prices (no extra HTTP call)
    unrealized = 0.0
    for sym, pos in risk_manager.open_positions.items():
        cmp = _last_prices.get(sym, pos.entry_price)
        unrealized += pos.unrealised_pnl(cmp)

    trades_today = summary["total_trades"]
    win_rate     = summary["win_rate_pct"]
    realized     = summary["realised_pnl"]
    total        = realized + unrealized

    def _fmt(val: float) -> str:
        color = "green" if val >= 0 else "red"
        sign  = "+" if val >= 0 else ""
        return f"[{color}]{sign}\u20b9{val:,.2f}[/{color}]"

    text = (
        f"[bold]Trades Today:[/bold] {trades_today}"
        f"  \u2502  [bold]Win Rate:[/bold] {win_rate:.0f}%"
        f"  \u2502  [bold]Realized P&L:[/bold] {_fmt(realized)}"
        f"  \u2502  [bold]Unrealized P&L:[/bold] {_fmt(unrealized)}"
        f"  \u2502  [bold]Total P&L:[/bold] {_fmt(total)}"
    )
    return Panel(
        text,
        title="[bold yellow]\u25b6 DAILY PERFORMANCE[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    )


def _make_sector_panel() -> Panel:
    """Side panel showing all sectors sorted by today's performance."""
    perf = ma.get_sector_performance()
    strong = set(ma.get_strong_sectors())

    if not perf:
        return Panel(
            "[dim]Loading\nsector\ndata...[/dim]",
            title="[bold magenta]SECTORS[/bold magenta]",
            border_style="magenta",
        )

    sorted_s = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    lines: List[str] = []
    for i, (sector, pct) in enumerate(sorted_s, 1):
        color = "green" if pct >= 0 else "red"
        sign  = "+" if pct >= 0 else ""
        bold  = "bold " if sector in strong else ""
        medal = "\u25b2" if i == 1 else ("\u25cf" if i == 2 else ("\u25bc" if i == 3 else " "))
        lines.append(
            f"{medal} [{bold}{color}]{sector:<10}[/{bold}{color}]"
            f" [{bold}{color}]{sign}{pct:.2f}%[/{bold}{color}]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold magenta]SECTOR STRENGTH[/bold magenta]",
        border_style="magenta",
    )


def _make_signals_panel() -> Panel:
    lines = list(_signal_log)
    text  = "\n".join(lines) if lines else "[dim]No signals yet \u2014 scanning market...[/dim]"
    return Panel(text, title="[bold green]\u25b6 SIGNALS[/bold green]", border_style="green")


def _make_logs_panel() -> Panel:
    lines = list(_log_buffer)
    text  = "\n".join(lines) if lines else "[dim]Engine started[/dim]"
    return Panel(text, title="[bold blue]\u25c8 LOGS[/bold blue]", border_style="blue")


def _make_positions_panel() -> Panel:
    positions = risk_manager.open_positions
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    for col, just in [
        ("Symbol",    "left"),
        ("Side",      "center"),
        ("Qty",       "right"),
        ("Entry",     "right"),
        ("CMP",       "right"),
        ("SL",        "right"),
        ("Target",    "right"),
        ("Unreal P&L","right"),
    ]:
        table.add_column(col, justify=just)

    if positions:
        try:
            prices = data_fetcher.get_batch_quotes(list(positions.keys()))
        except Exception:
            prices = {}
        for sym, pos in positions.items():
            cmp  = prices.get(sym, pos.entry_price)
            upnl = pos.unrealised_pnl(cmp)
            color = "green" if upnl >= 0 else "red"
            table.add_row(
                sym, pos.action, str(pos.quantity),
                f"\u20b9{pos.entry_price:.2f}", f"\u20b9{cmp:.2f}",
                f"\u20b9{pos.stoploss:.2f}",   f"\u20b9{pos.target:.2f}",
                f"[{color}]\u20b9{upnl:.2f}[/{color}]",
            )
        content: object = table
    else:
        content = "[dim]No open positions[/dim]"

    pos_count = len(positions)
    title = (
        f"[bold green]OPEN POSITIONS "
        f"({pos_count}/{config.risk.max_open_positions})[/bold green]"
    )
    return Panel(content, title=title, border_style="green")


def _refresh_dashboard() -> None:
    if _dashboard_layout is None:
        return
    _dashboard_layout["header"].update(_make_header_panel())
    _dashboard_layout["market_status"].update(_make_market_status_panel())
    _dashboard_layout["signals"].update(_make_signals_panel())
    _dashboard_layout["logs"].update(_make_logs_panel())
    _dashboard_layout["sector"].update(_make_sector_panel())
    _dashboard_layout["daily_pnl"].update(_make_daily_pnl_panel())
    _dashboard_layout["positions"].update(_make_positions_panel())


def _install_dashboard_handlers(dash_handler: logging.Handler) -> None:
    """Remove stdout StreamHandlers from all cached loggers; route to dashboard."""
    import logger as _logger_module
    for lg in _logger_module._loggers.values():
        lg.handlers = [
            h for h in lg.handlers
            if not (
                isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            )
        ]
        lg.addHandler(dash_handler)


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
    """Used only before the dashboard starts (startup output)."""
    mode_label = "PAPER" if config.trading.paper_trading else "LIVE "
    provider   = config.data_provider.upper()
    log.print(
        "\n[bold cyan]\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557[/bold cyan]"
        "\n[bold cyan]\u2551     AI TRADER  \u2014  Zerodha KiteConnect    \u2551[/bold cyan]"
        "\n[bold cyan]\u2551          Indian Stock Market Bot         \u2551[/bold cyan]"
        f"\n[bold cyan]\u2551  Mode: {mode_label}  Provider: {provider:8s}         \u2551[/bold cyan]"
        f"\n[bold cyan]\u2551  Strategy: {config.trading.trading_mode:10s}                    \u2551[/bold cyan]"
        "\n[bold cyan]\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d[/bold cyan]\n"
    )


def _print_positions() -> None:
    """Used in --run-now mode only (no dashboard)."""
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
            f"\u20b9{pos.entry_price:.2f}", f"\u20b9{cmp:.2f}",
            f"\u20b9{pos.stoploss:.2f}", f"\u20b9{pos.target:.2f}",
            f"[green]\u20b9{upnl:.2f}[/green]" if upnl >= 0 else f"[red]\u20b9{upnl:.2f}[/red]",
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
    global _status

    # Paper mode runs 24/7 on historical candles; skip market-hours gate
    if not config.trading.paper_trading and not _is_market_open():
        logger.debug("Market closed — skipping cycle.")
        return

    if risk_manager.is_halted:
        logger.warning("Trading HALTED — daily loss limit reached.")
        _status = "HALTED — daily loss limit reached"
        _refresh_dashboard()
        return

    mode     = config.trading.trading_mode
    symbols  = config.trading.symbols
    interval = (
        config.trading.intraday_interval if mode == "intraday"
        else config.trading.swing_interval
    )

    _status = f"Scanning market... {_now_ist().strftime('%H:%M:%S')}"
    _refresh_dashboard()

    logger.info("Trading cycle | %s | %s | %s",
                mode, config.data_provider, _now_ist().strftime("%H:%M:%S"))

    # ---- 1. Fetch current prices for position monitoring ---------------
    prices = data_fetcher.get_batch_quotes(symbols)
    if prices:
        _last_prices.update(prices)
        trader.monitor_positions(prices)

    # ---- 2. Handle intraday exit window (sim/live only) ----------------
    if mode == "intraday" and not config.trading.paper_trading and _is_intraday_exit_window():
        logger.info("Intraday exit window — closing all MIS positions.")
        trader.close_all(prices, reason="eod_intraday")
        _status = "Intraday exit — positions closed"
        _refresh_dashboard()
        return

    # ---- 3. Scan watchlist for signals (parallel) ----------------------
    def _scan_symbol(symbol: str) -> None:
        if risk_manager.has_position(symbol) or risk_manager.is_halted:
            return
        try:
            _process_symbol(symbol, mode, interval)
        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e, exc_info=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_scan_symbol, symbols)

    _status = f"Idle \u2014 last scan {_now_ist().strftime('%H:%M:%S')}"
    _refresh_dashboard()


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

    # Apply sector confidence boost (+0.05 if in a strong sector today)
    boost = ma.sector_confidence_boost(symbol)
    if boost > 0:
        signal.confidence = min(1.0, signal.confidence + boost)
        signal.reason     = f"{signal.reason}, sector_boost"
        logger.debug("%s: sector boost applied → conf=%.2f", symbol, signal.confidence)

    logger.info(
        "SIGNAL: %s %s  conf=%.2f  entry=%.2f  sl=%.2f  tgt=%.2f | %s",
        signal.action, symbol, signal.confidence,
        signal.entry_price, signal.suggested_sl,
        signal.suggested_target, signal.reason,
    )

    # Attempt trade
    result = trader.process_signal(signal)
    if result:
        logger.info(
            "TRADE EXECUTED: %s %s x%s @ \u20b9%.2f  SL=\u20b9%.2f  TGT=\u20b9%.2f",
            result.action, result.symbol, result.quantity,
            result.fill_price, result.stoploss, result.target,
        )


# ---------------------------------------------------------------------------
# Market open / close callbacks
# ---------------------------------------------------------------------------

def on_market_open() -> None:
    global _status
    logger.info("Market opened — initialising day.")
    _status = "Market OPEN — Scanning..."
    # Pre-warm instrument cache for Zerodha modes
    if config.data_provider == "zerodha" and kite_client._kite:
        kite_client.preload_instruments()


def on_market_close() -> None:
    global _status
    logger.info("Market closing — final clean-up.")
    _status = "Market CLOSED — wrapping up"

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
            logger.info("AI Optimisation: %s",
                        suggestions.get("overall_assessment", ""))


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
        # Paper / sim: run every 1 minute, 24/7 (historical candles always available)
        sched.add_job(trading_cycle, IntervalTrigger(minutes=1), id="paper_cycle")
        logger.info("Scheduler: 24/7 paper cycle (every 1 min).")
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
    ma.stop_background_refresh()
    if _scheduler_ref:
        _scheduler_ref[0].shutdown(wait=False)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _dashboard_layout, _status

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
    # Startup log (pre-dashboard, goes to stdout)
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
        status_str = "ONLINE" if available else "OFFLINE — technical signals only"
        logger.info("Ollama AI (%s): %s", config.ai.model, status_str)

    # ----------------------------------------------------------------
    # One-shot mode for testing / debugging
    # ----------------------------------------------------------------
    if args.run_now:
        logger.info("Running immediate trading cycle ...")
        on_market_open()
        trading_cycle()
        _print_positions()
        _print_daily_summary()
        return

    # ----------------------------------------------------------------
    # Build dashboard layout
    # ----------------------------------------------------------------
    _dashboard_layout = Layout()
    _dashboard_layout.split_column(
        Layout(name="header",        size=3),
        Layout(name="market_status", size=3),
        Layout(name="middle",        ratio=1),
        Layout(name="daily_pnl",     size=3),
        Layout(name="positions",     size=14),
    )
    _dashboard_layout["middle"].split_row(
        Layout(name="signals", ratio=2),
        Layout(name="logs",    ratio=2),
        Layout(name="sector",  ratio=1),
    )

    # Pre-populate log buffer with startup info
    ts = datetime.now().strftime("%H:%M:%S")
    _log_buffer.append(f"[dim white]{ts}[/dim white] Engine started — mode: [bold]{runtime_mode.upper()}[/bold]")
    _log_buffer.append(f"[dim white]{ts}[/dim white] Capital: \u20b9{config.trading.capital:,.0f}  Symbols: {len(config.trading.symbols)}")
    if config.ai.enabled:
        ai_str = "ONLINE" if ai_engine.is_available() else "OFFLINE"
        _log_buffer.append(f"[dim white]{ts}[/dim white] AI ({config.ai.model}): {ai_str}")

    _status = "Starting scheduler..."
    _refresh_dashboard()

    # ----------------------------------------------------------------
    # Start market analytics background thread
    # ----------------------------------------------------------------
    ma.start_background_refresh()

    # ----------------------------------------------------------------
    # Start scheduler + live dashboard
    # ----------------------------------------------------------------
    sched = _build_scheduler()
    _scheduler_ref.append(sched)
    _signal.signal(_signal.SIGINT,  _handle_sigint)
    _signal.signal(_signal.SIGTERM, _handle_sigint)

    _dash_handler = _DashboardLogHandler()
    _dash_handler.setLevel(logging.INFO)

    try:
        with Live(_dashboard_layout, console=log, refresh_per_second=2, screen=True):
            # Redirect all logger output into dashboard buffers
            _install_dashboard_handlers(_dash_handler)
            _status = "Scanning market..." if config.trading.paper_trading else "Waiting for market hours..."
            _refresh_dashboard()
            sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
