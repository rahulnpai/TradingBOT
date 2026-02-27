"""
kite_client.py — Zerodha KiteConnect wrapper.

Usage:
  python kite_client.py --login      # generate access token (one-time per day)

The access token is saved to .kite_token and auto-loaded next time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytz

from config import config
from logger import get_logger

log = get_logger("KiteClient")

try:
    from kiteconnect import KiteConnect, KiteTicker
except ImportError:
    log.critical("kiteconnect not installed. Run: pip install kiteconnect")
    sys.exit(1)

IST = pytz.timezone("Asia/Kolkata")


class KiteClient:
    """Thread-safe KiteConnect wrapper with auto token persistence."""

    def __init__(self) -> None:
        self._kite: Optional[KiteConnect] = None
        self._ticker: Optional[KiteTicker] = None
        self._token_file = config.kite.token_file

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Set up KiteConnect instance, loading token from file if available."""
        self._kite = KiteConnect(api_key=config.kite.api_key)

        token = self._load_token()
        if token:
            self._kite.set_access_token(token)
            log.info("KiteConnect ready (token loaded from file).")
        elif config.kite.access_token:
            self._kite.set_access_token(config.kite.access_token)
            log.info("KiteConnect ready (token from environment).")
        elif config.trading.paper_trading:
            log.warning(
                "Paper trading mode: no live Kite token. "
                "Market data calls will use mock data."
            )
        else:
            raise RuntimeError(
                "No access token found. Run 'python kite_client.py --login'."
            )

    def login(self) -> None:
        """Interactive browser login — prints URL, waits for request_token."""
        if not config.kite.api_key:
            raise ValueError("KITE_API_KEY must be set in .env for login.")

        kite = KiteConnect(api_key=config.kite.api_key)
        login_url = kite.login_url()

        print(f"\nOpen this URL in your browser:\n{login_url}\n")
        try:
            webbrowser.open(login_url)
        except Exception:
            pass

        request_token = input(
            "After login, paste the 'request_token' from the redirect URL here:\n> "
        ).strip()

        data = kite.generate_session(request_token, api_secret=config.kite.api_secret)
        access_token: str = data["access_token"]

        self._save_token(access_token)
        print(f"\nAccess token saved to {self._token_file}")
        print("Add to .env:  KITE_ACCESS_TOKEN=" + access_token)

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    def _save_token(self, token: str) -> None:
        payload = {"token": token, "date": date.today().isoformat()}
        with open(self._token_file, "w") as f:
            json.dump(payload, f)

    def _load_token(self) -> Optional[str]:
        if not os.path.exists(self._token_file):
            return None
        try:
            with open(self._token_file) as f:
                payload = json.load(f)
            if payload.get("date") == date.today().isoformat():
                return payload["token"]
            log.info("Cached token is from a previous day — ignoring.")
        except Exception as e:
            log.warning("Failed to load token file: %s", e)
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @property
    def kite(self) -> KiteConnect:
        if self._kite is None:
            raise RuntimeError("KiteClient not initialised. Call initialise() first.")
        return self._kite

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        continuous: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch OHLCV candles for an NSE equity symbol."""
        instrument_token = self._get_instrument_token(symbol)
        data = self.kite.historical_data(
            instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=continuous,
        )
        return data

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote."""
        key = f"{config.trading.exchange}:{symbol}"
        quotes = self.kite.quote([key])
        return quotes.get(key, {})

    def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Last traded price for multiple symbols."""
        keys = [f"{config.trading.exchange}:{s}" for s in symbols]
        data = self.kite.ltp(keys)
        return {
            s: data.get(f"{config.trading.exchange}:{s}", {}).get("last_price", 0.0)
            for s in symbols
        }

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        action: str,      # "BUY" | "SELL"
        quantity: int,
        product: str = "MIS",   # MIS (intraday) | CNC (delivery) | NRML (F&O)
        tag: str = "ai_trader",
    ) -> str:
        """Place a market order; returns order_id."""
        order_id = self.kite.place_order(
            tradingsymbol=symbol,
            exchange=config.trading.exchange,
            transaction_type=action,
            quantity=quantity,
            order_type=self.kite.ORDER_TYPE_MARKET,
            product=product,
            variety=self.kite.VARIETY_REGULAR,
            tag=tag,
        )
        log.info("Market order placed: %s %s x%s → order_id=%s", action, symbol, quantity, order_id)
        return str(order_id)

    def place_limit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        product: str = "MIS",
        tag: str = "ai_trader",
    ) -> str:
        order_id = self.kite.place_order(
            tradingsymbol=symbol,
            exchange=config.trading.exchange,
            transaction_type=action,
            quantity=quantity,
            price=price,
            order_type=self.kite.ORDER_TYPE_LIMIT,
            product=product,
            variety=self.kite.VARIETY_REGULAR,
            tag=tag,
        )
        log.info("Limit order placed: %s %s x%s @ %.2f → order_id=%s",
                 action, symbol, quantity, price, order_id)
        return str(order_id)

    def place_sl_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        trigger_price: float,
        price: float,
        product: str = "MIS",
    ) -> str:
        order_id = self.kite.place_order(
            tradingsymbol=symbol,
            exchange=config.trading.exchange,
            transaction_type=action,
            quantity=quantity,
            trigger_price=round(trigger_price, 2),
            price=round(price * 0.995, 2),   # limit slightly below trigger
            order_type=self.kite.ORDER_TYPE_SLM,
            product=product,
            variety=self.kite.VARIETY_REGULAR,
        )
        return str(order_id)

    def cancel_order(self, order_id: str) -> None:
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        log.info("Order cancelled: %s", order_id)

    def get_positions(self) -> Dict[str, Any]:
        return self.kite.positions()

    def get_orders(self) -> List[Dict[str, Any]]:
        return self.kite.orders()

    def get_margins(self) -> Dict[str, Any]:
        return self.kite.margins()

    # ------------------------------------------------------------------
    # Instrument lookup
    # ------------------------------------------------------------------

    _instrument_cache: Dict[str, int] = {}

    def _get_instrument_token(self, symbol: str) -> int:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        instruments = self.kite.instruments(exchange=config.trading.exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                token = int(inst["instrument_token"])
                self._instrument_cache[symbol] = token
                return token
        raise ValueError(f"Instrument token not found for {symbol}")

    def preload_instruments(self) -> None:
        """Cache all NSE instrument tokens at startup."""
        try:
            instruments = self.kite.instruments(exchange=config.trading.exchange)
            for inst in instruments:
                self._instrument_cache[inst["tradingsymbol"]] = int(
                    inst["instrument_token"]
                )
            log.info("Loaded %d instrument tokens.", len(self._instrument_cache))
        except Exception as e:
            log.warning("Could not preload instruments: %s", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

kite_client = KiteClient()


# ---------------------------------------------------------------------------
# CLI helper for login
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kite Client utilities")
    parser.add_argument("--login", action="store_true", help="Authenticate and save access token")
    args = parser.parse_args()

    if args.login:
        kite_client.login()
    else:
        parser.print_help()
