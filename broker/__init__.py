"""
broker — broker abstraction layer + provider implementations.

The bot's main loop, OrderExecutor, PositionTracker and RiskManager all
depend on the abstract :class:`broker.base.BaseBroker` interface.
Concrete implementations (Alpaca today, MT5 / IBKR / ... tomorrow) live in
their own modules and are selected by name through :func:`get_broker`.
"""
from broker.alpaca_client import AlpacaClient
from broker.base import BaseBroker
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker

__all__ = [
    "AlpacaClient", "BaseBroker", "OrderExecutor", "PositionTracker",
    "get_broker",
]


def get_broker(provider: str = "alpaca", *, paper: bool | None = None,
               data_feed: str | None = None) -> BaseBroker:
    """
    Instantiate a broker adapter by provider name.

    Parameters
    ----------
    provider :
        ``"alpaca"`` (default). Future values: ``"mt5"``, ``"ibkr"``,
        ``"tradier"``, ``"tradestation"``, ``"binance"``, ``"coinbase"``, ...
    paper :
        Override the paper-vs-live flag. When None the adapter resolves it
        from credentials.yaml / env vars / its own default.
    data_feed :
        Adapter-specific data-feed override (e.g. Alpaca ``"iex"`` /
        ``"sip"``). Ignored by adapters that don't expose a feed concept.

    Returns
    -------
    A :class:`BaseBroker` subclass instance, NOT yet connected. Call
    ``client.connect()`` (or ``connect_with_retry()``) on the returned object.

    Raises
    ------
    ValueError
        For unknown providers.
    """
    p = provider.lower().strip()
    if p in ("alpaca", "alpaca-py", ""):
        return AlpacaClient(paper=paper, data_feed=data_feed)
    raise ValueError(
        f"Unknown broker provider '{provider}'. "
        f"Supported: 'alpaca'. Add a new adapter in broker/ that subclasses "
        f"BaseBroker and extend this factory."
    )
