"""
broker/base.py — Abstract broker interface.

Defines the contract every broker adapter (AlpacaClient, future MT5Client,
IBKRClient, ...) must satisfy. Higher layers (main.py, OrderExecutor,
PositionTracker, RiskManager) depend ONLY on this contract, never on
broker-specific types.

Design notes
------------
* Most return types are duck-typed: callers use a small set of attributes
  (e.g. ``account.equity``, ``position.qty``, ``clock.is_open``). Each broker
  implementation returns a type that exposes the same attribute surface.
  This keeps the abstraction lightweight and avoids forcing every broker
  through a one-size-fits-all dataclass conversion.

* The ABC enforces presence of methods. Attribute conformance is checked by
  ``tests/test_broker_abstraction.py``.

* Streaming (TradingStream / equivalent) is intentionally NOT part of the
  base contract — it varies wildly between brokers (Alpaca WebSocket,
  MT5 polling, IBKR's TWS event bus, ...). PositionTracker keeps a soft
  dependency on broker-specific stream classes and is adapted per broker.

Stable surface (every broker MUST implement)
--------------------------------------------
::

    connect / connect_with_retry / disconnect / is_connected / paper

    get_account()       -> object with .id, .status, .equity, .cash,
                            .buying_power
    get_clock()         -> object with .is_open, .next_open, .next_close
    get_buying_power(), get_portfolio_value(), get_cash() — convenience

    get_bars(symbols, timeframe, start, end=None) -> DataFrame
    get_latest_price(symbol)  -> float
    get_latest_quote(symbol)  -> dict-like  (keys: bid_price, ask_price, ...)

    get_all_positions()       -> list of objects with .symbol, .qty,
                                  .market_value, .avg_entry_price,
                                  .unrealized_pl
    get_position(symbol)      -> same shape, or None
    get_positions_as_dict()   -> {symbol: qty}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Required attribute surfaces (informational) ────────────────────────────────

#: Attributes a ``get_account()`` return value MUST expose. Used by the
#: conformance test and by any broker adapter that wraps a third-party type.
EXPECTED_ACCOUNT_ATTRS = ("id", "status", "equity", "cash", "buying_power")

#: Attributes a ``get_clock()`` return value MUST expose.
EXPECTED_CLOCK_ATTRS = ("is_open", "next_open", "next_close")

#: Attributes a position object (from ``get_all_positions()`` /
#: ``get_position(symbol)``) MUST expose.
EXPECTED_POSITION_ATTRS = ("symbol", "qty", "avg_entry_price",
                           "market_value", "unrealized_pl")


# ── Abstract base class ────────────────────────────────────────────────────────

class BaseBroker(ABC):
    """
    Abstract trading-broker interface.

    Subclasses adapt a specific broker (Alpaca, MT5, IBKR, ...) to this
    contract. The bot's main loop, OrderExecutor, PositionTracker, and
    RiskManager depend ONLY on this interface — never on broker-specific
    types — so swapping brokers reduces to writing a new subclass and
    flipping ``broker.provider`` in settings.yaml.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, skip_live_confirm: bool = False) -> None:
        """Establish connection and run a health check.

        ``skip_live_confirm`` bypasses the YES-I-UNDERSTAND-THE-RISKS prompt;
        for tests and one-off scripts only — never for live operator launches.
        """

    @abstractmethod
    def connect_with_retry(self, max_attempts: int = 3, base_delay: float = 2.0) -> None:
        """Connect with exponential-backoff retry on transient failures."""

    @abstractmethod
    def disconnect(self) -> None:
        """Best-effort close of network connections, streams, etc."""

    @property
    def is_connected(self) -> bool:
        """True after a successful ``connect()`` call.

        Default impl reads ``self._connected`` (most adapters set this in
        ``connect()``). Subclasses are free to override.
        """
        return bool(getattr(self, "_connected", False))

    # ``paper`` is expected as a regular instance attribute set in __init__.
    # We don't declare it as an abstract property because the existing
    # AlpacaClient assigns it directly (``self.paper = bool``); enforcing
    # @property here would force a redundant property accessor.
    # Conformance is checked structurally by tests/test_broker_abstraction.py.

    # ── Account / clock ───────────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> Any:
        """Snapshot of the trading account.

        Returned object must expose the attributes listed in
        ``EXPECTED_ACCOUNT_ATTRS``.
        """

    @abstractmethod
    def get_clock(self) -> Any:
        """Market clock. Returned object must expose ``EXPECTED_CLOCK_ATTRS``."""

    def is_market_open(self) -> bool:
        """Convenience wrapper. Default impl reads ``get_clock().is_open``."""
        return bool(self.get_clock().is_open)

    def get_buying_power(self) -> float:
        return float(self.get_account().buying_power)

    def get_portfolio_value(self) -> float:
        return float(self.get_account().equity)

    def get_cash(self) -> float:
        return float(self.get_account().cash)

    # ── Market data ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """OHLCV bars for ``symbols`` between ``start`` and ``end``.

        ``timeframe`` is one of ``"1Min"``, ``"5Min"``, ``"15Min"``,
        ``"1Hour"``, ``"1Day"``. Brokers that don't support a given
        timeframe should raise ``ValueError``.

        Return shape: a long-format DataFrame with a MultiIndex (symbol,
        timestamp) and columns at minimum ``[open, high, low, close, volume]``.
        """

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Most recent traded price for ``symbol``."""

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Latest bid/ask quote. Must include keys
        ``bid_price``, ``ask_price``."""

    # ── Positions ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_all_positions(self) -> List[Any]:
        """All open positions. Each element exposes ``EXPECTED_POSITION_ATTRS``."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Any]:
        """Single position for ``symbol`` or None when flat."""

    def get_positions_as_dict(self) -> Dict[str, float]:
        """Convenience wrapper. Default impl returns ``{symbol: qty}``."""
        return {p.symbol: float(p.qty) for p in self.get_all_positions()}
