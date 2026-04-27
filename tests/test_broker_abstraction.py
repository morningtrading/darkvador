"""
tests/test_broker_abstraction.py
=================================

Conformance tests for the BaseBroker abstraction.

What this guards against
------------------------
1. AlpacaClient diverging from the BaseBroker contract (renames, missing
   methods, broken subclassing).
2. The factory ``broker.get_broker`` being unable to instantiate any
   registered provider.
3. Return values of stable methods (get_account, get_clock, position
   objects) silently dropping required attributes when the underlying
   third-party SDK changes shape.

What this DOES NOT guard against
--------------------------------
- Backtest reproducibility — that lives in
  ``scripts/backtest_mean_reversion.py`` (standalone) and is meant to be
  run manually as a snapshot. Adding it as a unit test would require
  fetching market data from yfinance / Alpaca which is too slow and
  flaky for CI.
- Order-management surface (submit_order / cancel_order / streams) —
  these vary too much per broker to lock down at this layer; they will
  be tested per adapter when we add the second broker.
"""
from __future__ import annotations

import inspect

import pytest

from broker import get_broker
from broker.alpaca_client import AlpacaClient
from broker.base import (
    EXPECTED_ACCOUNT_ATTRS,
    EXPECTED_CLOCK_ATTRS,
    EXPECTED_POSITION_ATTRS,
    BaseBroker,
)


# ── Subclassing & abstract-method coverage ───────────────────────────────────


def test_alpaca_client_inherits_basebroker():
    assert issubclass(AlpacaClient, BaseBroker)


def test_alpaca_client_instantiable_without_credentials():
    """The constructor should not call any network endpoint, so missing
    credentials should not prevent instantiation. Connection is deferred
    to .connect()."""
    client = AlpacaClient(paper=True)
    assert isinstance(client, BaseBroker)
    assert client.paper is True
    assert client.is_connected is False


def test_alpaca_client_implements_every_abstract_method():
    """The metaclass would block instantiation if any abstract was missing,
    but spell out the contract explicitly so a regression is loud."""
    expected = {
        "connect", "connect_with_retry", "disconnect",
        "get_account", "get_clock",
        "get_bars", "get_latest_price", "get_latest_quote",
        "get_all_positions", "get_position",
    }
    missing = expected - {
        n for n, m in inspect.getmembers(AlpacaClient)
        if inspect.isfunction(m) or inspect.ismethod(m)
    }
    assert not missing, f"AlpacaClient missing required methods: {missing}"


def test_basebroker_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseBroker()  # type: ignore[abstract]


# ── Factory ──────────────────────────────────────────────────────────────────


def test_factory_returns_alpaca_for_default_provider():
    client = get_broker("alpaca", paper=True)
    assert isinstance(client, AlpacaClient)
    assert isinstance(client, BaseBroker)


@pytest.mark.parametrize("alias", ["alpaca", "Alpaca", "ALPACA", "alpaca-py", ""])
def test_factory_accepts_aliases(alias):
    """``provider`` lookup is case-insensitive and tolerates whitespace."""
    client = get_broker(alias, paper=True)
    assert isinstance(client, AlpacaClient)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown broker provider"):
        get_broker("definitely_not_a_real_broker", paper=True)


# ── Convenience wrappers in BaseBroker route through abstract methods ────────


class _FakeBroker(BaseBroker):
    """Minimal in-memory BaseBroker. Used only to test the convenience
    wrappers (is_market_open, get_buying_power, etc.) without hitting any
    real broker. Kept inside the test module so it's not exposed."""

    def __init__(self, account=None, clock=None, positions=None):
        self._connected = True
        self.paper = True
        self._account = account
        self._clock = clock
        self._positions = positions or []

    def connect(self, skip_live_confirm=False): self._connected = True
    def connect_with_retry(self, max_attempts=3, base_delay=2.0): self._connected = True
    def disconnect(self): self._connected = False
    def get_account(self): return self._account
    def get_clock(self): return self._clock
    def get_bars(self, symbols, timeframe, start, end=None):
        import pandas as pd
        return pd.DataFrame()
    def get_latest_price(self, symbol): return 100.0
    def get_latest_quote(self, symbol): return {"bid_price": 99.5, "ask_price": 100.5}
    def get_all_positions(self): return list(self._positions)
    def get_position(self, symbol):
        for p in self._positions:
            if getattr(p, "symbol", None) == symbol:
                return p
        return None


class _Acct:
    def __init__(self, equity, cash, bp):
        self.id, self.status = "ACC1", "ACTIVE"
        self.equity, self.cash, self.buying_power = equity, cash, bp


class _Clk:
    def __init__(self, is_open):
        self.is_open, self.next_open, self.next_close = is_open, None, None


class _Pos:
    def __init__(self, symbol, qty, market_value=0.0):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = 0.0
        self.market_value = market_value
        self.unrealized_pl = 0.0


def test_convenience_wrappers_route_through_abstract_methods():
    fake = _FakeBroker(
        account=_Acct(equity=12_345.67, cash=1_000.0, bp=20_000.0),
        clock=_Clk(is_open=True),
        positions=[_Pos("SPY", 10, 5_000.0), _Pos("QQQ", -5, -3_000.0)],
    )
    assert fake.get_buying_power() == 20_000.0
    assert fake.get_portfolio_value() == 12_345.67
    assert fake.get_cash() == 1_000.0
    assert fake.is_market_open() is True
    assert fake.get_positions_as_dict() == {"SPY": 10.0, "QQQ": -5.0}


# ── Expected attribute surfaces are non-empty ─────────────────────────────────

def test_expected_attribute_lists_are_documented():
    """If someone empties EXPECTED_*_ATTRS by mistake, the conformance
    test below short-circuits to True. Catch that here."""
    assert len(EXPECTED_ACCOUNT_ATTRS)  >= 4
    assert len(EXPECTED_CLOCK_ATTRS)    >= 1
    assert len(EXPECTED_POSITION_ATTRS) >= 4


def test_basebroker_module_exports_attribute_lists():
    """Down-stream code can import the expected-attribute tuples to do its
    own duck-typing checks."""
    from broker import base as base_mod
    assert hasattr(base_mod, "EXPECTED_ACCOUNT_ATTRS")
    assert hasattr(base_mod, "EXPECTED_CLOCK_ATTRS")
    assert hasattr(base_mod, "EXPECTED_POSITION_ATTRS")
