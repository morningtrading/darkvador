# Broker change — checkpoint

State of the bot at the moment we stop on Alpaca and prepare to swap brokers.
Tag: **`pre-broker-change`** (use `git checkout pre-broker-change` to roll back).

---

## Current state — what's working

- **Strategy stack**: hmm_regime (single, frozen-baseline) + mean_reversion_qqq_spy
  (just wired in) running side-by-side in live multi-strategy mode
- **Frozen-baseline backtest** (`balanced × stocks`, 2020-2026, single-strat
  hmm_regime): `+174.56% / Sharpe 1.082 / Calmar 1.678 / MaxDD -16.0%`
- **Mean-reversion backtest** (standalone): `+0.67 Sharpe, corr=-0.015 vs hmm`
- **Active config**: `balanced` set, asset_group `stocks`, HMM proxy QQQ,
  loop timeframe 5Min / HMM timeframe 1Day (decoupled), label_mode prototype
- **Paper account**: $2k equity (Alpaca won't let user reset until tomorrow)
- **Tags shipped**: `frozen-baseline-1`, `research-stop`, `pre-mr-integration`,
  `pre-broker-change` (this one)

## Alpaca surface — what needs to be abstracted

17 Python files touch Alpaca. Production-critical (must be rewritten or
adapted to BaseBroker): **7 files**.

| File | Role | Migration cost |
|---|---|---|
| `broker/alpaca_client.py` | trading + data API wrapper, account / clock / positions / orders | **major** — the entire surface lives here |
| `broker/order_executor.py` | limit-order placement, cancellation, fill tracking | **medium** — uses AlpacaClient + alpaca-py types directly |
| `broker/position_tracker.py` | TradingStream WebSocket, position sync | **medium** — tightly coupled to alpaca-py stream API |
| `main.py` | startup `[1/7] Connecting to Alpaca`, multi-strat wiring | **small** — mostly imports + factory dispatch |
| `data/market_data.py` | historical bars fetch, default benchmark "SPY" | **medium** — broker-agnostic in spirit, Alpaca-specific in code |
| `data/vix_fetcher.py` | VIX daily series (yfinance + Alpaca VXX fallback) | **small** — VIX side is yfinance, only Alpaca-fallback needs swap |
| `data/credit_spread_fetcher.py` | HYG/LQD spread series | **small** — same pattern as VIX |

Non-critical (scripts / tests / research that can be migrated later):
`scripts/probe_live_regime.py`, `scripts/check_alpaca.py`,
`scripts/reset_paper.py`, `scripts/measure_alpaca_spread.py`,
`tests/test_orders.py`, `tools/diagnose_sma_vs_hmm.py`,
`research/asset_selection/*`, `obsolete_code/*`.

## Recommended migration pattern (BaseBroker abstract)

Today: `main.py` does `from broker.alpaca_client import AlpacaClient` directly.

After migration:

```
broker/
├── base.py                   ← NEW: abstract BaseBroker class
│                                (connect, get_account, get_clock,
│                                 get_positions, get_bars, place_order,
│                                 cancel_order, stream_fills, ...)
├── alpaca_client.py          ← refactor to inherit BaseBroker
└── <new>_client.py           ← NEW: the new broker, also inherits BaseBroker
```

`main.py` reads `broker.provider` from settings.yaml (default `alpaca`)
and instantiates the matching class via a factory. Everything downstream
(OrderExecutor, PositionTracker, RiskManager) sees only the abstract
interface, so adding a 3rd broker later is one new file.

The `add-broker-adapter` skill in this Claude session is purpose-built
for exactly this scaffolding (`/skill add-broker-adapter`).

## Validation gates after the swap

Before declaring the migration done, run these and check the numbers:

1. **Frozen-baseline backtest must reproduce** (no broker dep, but uses
   data-fetch path):
   ```bash
   python main.py backtest --asset-group stocks --start 2020-01-01
   ```
   Expected: `+174.56% / Sharpe 1.082 / Calmar 1.678 / MaxDD -16.0%`.
   If it drifts, the new broker's historical-bar API returns subtly
   different OHLC than Alpaca — needs investigation.

2. **Live paper / dry run** with the new broker:
   ```bash
   python main.py trade --paper --dry-run
   ```
   Expected: dashboard renders, multi-strategy mode activates, `[8/8]`
   logs both strategies, signals appear at next bar close.

3. **Telegram message contains the new broker name + new credentials**:
   the `_header()` shows `host · ip · os · #sha`, but consider adding
   the broker name to the meta header so messages from different
   brokers are distinguishable in the chat.

4. **Reset / health probe scripts still work**:
   ```bash
   python scripts/check_alpaca.py     # rename → check_<new_broker>.py
   python scripts/reset_paper.py      # only if paper concept exists
   ```

## What to bring up at the next session

When you resume, tell me:

1. **Which broker** (Hyperliquid, MT5, IBKR, Tradestation, Tradier, Binance,
   Coinbase, ccxt-based exchange, ...). The `add-broker-adapter` skill knows
   the patterns for each.
2. **Asset class** stays equity, or shift to crypto / forex / futures?
   This may invalidate parts of the strategy stack (mean_reversion_qqq_spy
   is equity-pair specific — won't transfer to crypto).
3. **Paper / live decision** — most brokers have a paper / sandbox endpoint
   but the credentials separation we set up
   (`alpaca.paper_keys` / `alpaca.live_keys`) needs a parallel structure
   for the new one.
4. **Whether to delete `broker/alpaca_client.py`** or keep it dormant for
   side-by-side comparison runs.

## Where the system stands today (for the next session)

```
Last commit:          a317550  config: enable mean_reversion_qqq_spy
Active set:           balanced
Asset group:          stocks (SPY, QQQ, AAPL, MSFT, NVDA)
Loop timeframe:       5Min
HMM timeframe:        1Day
HMM label mode:       prototype
Strategies enabled:   hmm_regime, mean_reversion_qqq_spy
Paper account state:  $2,000 equity, 0 positions, no pending orders
Live process running: yes (paper trade --paper)
```
