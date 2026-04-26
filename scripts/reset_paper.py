"""
scripts/reset_paper.py — close all Alpaca paper positions, cancel all open
orders, and remove the local state_snapshot.json so the bot starts cold on
its next run.

For a TRUE $100k cash reset (in case fills accumulated stale P&L), use the
"Reset Paper Account" button in the Alpaca dashboard — that's not exposed
via the SDK we use.

Usage:
    .venv/bin/python scripts/reset_paper.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor


def main() -> int:
    client = AlpacaClient(paper=True)
    client.connect(skip_live_confirm=True)
    executor = OrderExecutor(client)

    acct = client.get_account()
    positions = client.get_all_positions()
    print(f"Before:")
    print(f"  Equity:        ${float(acct.equity):>15,.2f}")
    print(f"  Cash:          ${float(acct.cash):>15,.2f}")
    print(f"  Positions:     {len(positions)}")
    for p in positions:
        mv = float(p.market_value)
        print(f"    {p.symbol:<6} qty={float(p.qty):>10,.2f}  market_value=${mv:>15,.2f}")

    print("\nCancelling all open orders ...")
    n_cancelled = executor.cancel_all_orders()
    print(f"  -> {n_cancelled} orders cancelled")

    print("\nClosing all positions (market orders) ...")
    results = executor.close_all_positions()
    for r in results:
        print(f"  -> {r}")

    snap = ROOT / "state_snapshot.json"
    if snap.exists():
        snap.unlink()
        print(f"\nDeleted local state: {snap}")
    else:
        print("\nNo local state_snapshot.json found (already clean).")

    print("\nDone. Restart the bot with:  .venv/bin/python main.py trade --paper")
    print("If equity is still off (residual P&L from past fills), use the")
    print("Alpaca dashboard 'Reset Paper Account' button for a true $100k reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
