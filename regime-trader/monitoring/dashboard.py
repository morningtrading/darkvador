"""
dashboard.py -- Live terminal dashboard powered by Rich.

Renders a real-time view of portfolio state, current regime, open positions,
P&L, and a recent-events log.  Refreshes on a configurable interval using
Rich's Live context manager, running in a background thread.

Layout:
  Header  (title | time | market status)
  +------------------+---------------------------+
  | Regime Panel     | Portfolio Summary          |
  +------------------+---------------------------+
  | Positions Table                               |
  +-----------------------------------------------+
  | Event Log                                     |
  +-----------------------------------------------+
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import deque
from typing import Deque, List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from broker.position_tracker import PortfolioSnapshot
from core.signal_generator import PortfolioSignal

# Regime -> (Rich colour, label)
_REGIME_COLOURS = {
    "CRASH":       ("bright_red",    "CRASH"),
    "STRONG_BEAR": ("red",           "STRONG BEAR"),
    "BEAR":        ("red",           "BEAR"),
    "WEAK_BEAR":   ("dark_orange",   "WEAK BEAR"),
    "NEUTRAL":     ("yellow",        "NEUTRAL"),
    "WEAK_BULL":   ("chartreuse1",   "WEAK BULL"),
    "BULL":        ("green",         "BULL"),
    "STRONG_BULL": ("bright_green",  "STRONG BULL"),
    "EUPHORIA":    ("cyan",          "EUPHORIA"),
}

_MAX_EVENTS = 20


class Dashboard:
    """
    Terminal-based live trading dashboard.

    Parameters
    ----------
    refresh_seconds : How often (in seconds) to redraw.
    title           : Strategy name shown in the header.
    """

    def __init__(
        self,
        refresh_seconds: int = 5,
        title:           str = "Regime Trader",
    ) -> None:
        self.refresh_seconds = refresh_seconds
        self.title           = title

        self._console   = Console()
        self._live:       Optional[Live]    = None
        self._layout:     Optional[Layout]  = None
        self._is_running: bool              = False
        self._thread:     Optional[threading.Thread] = None

        self._latest_snapshot: Optional[PortfolioSnapshot] = None
        self._latest_signal:   Optional[PortfolioSignal]   = None
        self._recent_events:   Deque[str]                  = deque(maxlen=_MAX_EVENTS)

    # ======================================================================= #
    # Lifecycle                                                                #
    # ======================================================================= #

    def start(self) -> None:
        """Start the live dashboard in a background daemon thread."""
        if self._is_running:
            return
        self._layout     = self._build_layout()
        self._is_running = True
        self._thread     = threading.Thread(
            target = self._refresh_loop,
            daemon = True,
            name   = "dashboard",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the live dashboard and restore the terminal."""
        self._is_running = False
        if self._live:
            self._live.stop()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ======================================================================= #
    # Data updates                                                             #
    # ======================================================================= #

    def update(
        self,
        snapshot: Optional[PortfolioSnapshot] = None,
        signal:   Optional[PortfolioSignal]   = None,
        event:    Optional[str]               = None,
    ) -> None:
        """
        Push new data to the dashboard.

        Parameters
        ----------
        snapshot : Latest PortfolioSnapshot.
        signal   : Latest PortfolioSignal.
        event    : One-line event string to append to the event log.
        """
        if snapshot is not None:
            self._latest_snapshot = snapshot
        if signal is not None:
            self._latest_signal = signal
        if event is not None:
            ts  = dt.datetime.now().strftime("%H:%M:%S")
            self._recent_events.appendleft(f"[dim]{ts}[/dim]  {event}")

        # Force a redraw if live context is active
        if self._live and self._is_running:
            self._live.update(self.render())

    # ======================================================================= #
    # Rendering                                                                #
    # ======================================================================= #

    def render(self) -> Layout:
        """Build and return the full layout tree."""
        layout = self._layout or self._build_layout()
        layout["header"].update(self._render_header())
        layout["regime"].update(self._render_regime_panel())
        layout["portfolio"].update(self._render_portfolio_summary())
        layout["positions"].update(self._render_positions_table())
        layout["events"].update(self._render_event_log())
        return layout

    # ======================================================================= #
    # Private render helpers                                                   #
    # ======================================================================= #

    def _render_header(self) -> Panel:
        now    = dt.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        market = "[green]OPEN[/green]" if self._market_open() else "[red]CLOSED[/red]"
        text   = Text.assemble(
            (f"  {self.title}  ", "bold cyan"),
            (f"|  {now}  |  Market: ", "dim"),
            Text.from_markup(market),
        )
        return Panel(text, style="bold")

    def _render_regime_panel(self) -> Panel:
        sig = self._latest_signal
        if sig is None:
            return Panel("[dim]No signal yet[/dim]", title="Regime")

        colour, label = _REGIME_COLOURS.get(sig.regime, ("white", sig.regime))
        stable_icon   = "[green]STABLE[/green]" if sig.is_stable else "[yellow]PENDING[/yellow]"
        body = Text.assemble(
            (f"  {label}\n", f"bold {colour}"),
            (f"  Confidence : {sig.confidence:.1%}\n", ""),
            (f"  Stability  : ", ""),
            Text.from_markup(stable_icon + "\n"),
            (f"  Leverage   : {sig.leverage:.2f}x\n", ""),
        )
        if sig.notes:
            body.append("\n  " + "\n  ".join(sig.notes[:3]), style="dim")
        return Panel(body, title="[bold]Regime[/bold]", border_style=colour)

    def _render_portfolio_summary(self) -> Panel:
        snap = self._latest_snapshot
        if snap is None:
            return Panel("[dim]Waiting for data...[/dim]", title="Portfolio")

        dd_colour = "green" if snap.drawdown_from_peak > -0.02 else (
            "yellow" if snap.drawdown_from_peak > -0.05 else "red"
        )
        pnl_colour = "green" if snap.total_unrealized_pnl >= 0 else "red"
        pnl_sign   = "+" if snap.total_unrealized_pnl >= 0 else ""

        body = Text.assemble(
            (f"  Equity     : ${snap.total_equity:>12,.2f}\n",  "bold"),
            (f"  Cash       : ${snap.cash:>12,.2f}\n",          ""),
            (f"  Unrealised : {pnl_sign}${abs(snap.total_unrealized_pnl):>10,.2f}  "
             f"({pnl_sign}{snap.total_unrealized_pnl_pct:.2%})\n", pnl_colour),
            (f"  Exposure   : {snap.gross_exposure:.1%}\n",    ""),
            (f"  Peak DD    : {snap.drawdown_from_peak:.2%}\n", dd_colour),
            (f"  Positions  : {len(snap.positions)}\n",        ""),
        )
        return Panel(body, title="[bold]Portfolio[/bold]")

    def _render_positions_table(self) -> Table:
        table = Table(
            title="Open Positions",
            show_header=True,
            header_style="bold yellow",
            expand=True,
        )
        for col, justify in [
            ("Symbol",   "left"),
            ("Qty",      "right"),
            ("Entry",    "right"),
            ("Price",    "right"),
            ("Mkt Val",  "right"),
            ("P&L",      "right"),
            ("P&L %",    "right"),
            ("Weight",   "right"),
            ("Held",     "right"),
            ("Regime@E", "center"),
        ]:
            table.add_column(col, justify=justify)

        snap = self._latest_snapshot
        if snap is None or not snap.positions:
            table.add_row(*["--"] * 10)
            return table

        for pos in snap.positions:
            pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
            sign      = "+" if pos.unrealized_pnl >= 0 else ""
            table.add_row(
                pos.symbol,
                f"{pos.qty:.0f}",
                f"${pos.avg_entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"${pos.market_value:,.0f}",
                f"[{pnl_style}]{sign}${abs(pos.unrealized_pnl):,.0f}[/{pnl_style}]",
                f"[{pnl_style}]{sign}{pos.unrealized_pnl_pct:.2%}[/{pnl_style}]",
                f"{pos.weight:.1%}",
                f"{pos.holding_days:.1f}d",
                pos.regime_at_entry[:8],
            )
        return table

    def _render_event_log(self) -> Panel:
        if not self._recent_events:
            body = Text("[dim]No events yet[/dim]")
        else:
            body = Text.from_markup("\n".join(self._recent_events))
        return Panel(body, title="[bold]Recent Events[/bold]", border_style="dim")

    def _regime_colour(self, regime: str) -> str:
        """Return a Rich colour name for a regime label."""
        return _REGIME_COLOURS.get(regime, ("white", regime))[0]

    def _build_layout(self) -> Layout:
        """Construct the base layout skeleton."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header",    size=3),
            Layout(name="middle",    size=12),
            Layout(name="positions", size=14),
            Layout(name="events",    minimum_size=5),
        )
        layout["middle"].split_row(
            Layout(name="regime",    ratio=2),
            Layout(name="portfolio", ratio=3),
        )
        return layout

    def _market_open(self) -> bool:
        """Best-effort check -- returns False if we have no client access."""
        now = dt.datetime.now()
        # Simple heuristic: Mon-Fri, 09:30-16:00 ET
        # A real implementation would call client.is_market_open()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return dt.time(9, 30) <= t <= dt.time(16, 0)

    def _refresh_loop(self) -> None:
        """Background thread: enter Live context and redraw on each tick."""
        with Live(
            self.render(),
            console     = self._console,
            refresh_per_second = max(1, 1 // self.refresh_seconds),
            screen      = True,
        ) as live:
            self._live = live
            while self._is_running:
                live.update(self.render())
                time.sleep(self.refresh_seconds)
        self._live = None
