"""scripts/check_alpaca.py — quick health probe of the configured Alpaca paper
account: verifies credentials, prints account status + equity + open positions.
Use it after rotating keys or resetting the paper account."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker.alpaca_client import AlpacaClient


def main() -> int:
    c = AlpacaClient(paper=True)
    c.connect(skip_live_confirm=True)
    acct = c.get_account()
    print(f"Account ID:    {acct.id}")
    print(f"Status:        {acct.status}")
    print(f"Equity:        ${float(acct.equity):,.2f}")
    print(f"Cash:          ${float(acct.cash):,.2f}")
    print(f"Buying power:  ${float(acct.buying_power):,.2f}")
    print(f"PDT flag:      {acct.pattern_day_trader}")
    positions = c.get_all_positions()
    print(f"Positions:     {len(positions)}")
    for p in positions:
        mv = float(p.market_value)
        print(f"  {p.symbol:<8} qty={float(p.qty):>10,.4f}  market_value=${mv:>15,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
