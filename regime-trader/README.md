# regime-trader

HMM-based volatility regime detection and automated allocation system for US equities, wired to Alpaca for live and paper execution.

The system classifies the current market environment (low-vol, mid-vol, high-vol) using a Hidden Markov Model trained on realised volatility features, then adjusts portfolio weights, leverage, and position sizes accordingly. A multi-layer risk manager gates every order before it reaches the broker.

---

## Architecture

```
market data
    └─> FeatureEngineer      (log returns, realised vol 5d/21d)
            └─> HMMEngine    (BIC model selection, regime label + confidence)
                    └─> RegimeStrategy   (target weights per regime)
                            └─> RiskManager  (size, exposure, drawdown gates)
                                    └─> OrderExecutor  (limit orders -> Alpaca)
                                            └─> PositionTracker  (fills, P&L)
```

Supporting layer runs in parallel:
- `SignalGenerator` — single entry point that drives the pipeline each bar
- `TradeLogger` — JSON-structured rotating log
- `Dashboard` — Rich terminal live view
- `AlertManager` — rate-limited email + Slack/webhook alerts

---

## Regime framework

| Regime | Market condition | Allocation | Leverage |
|---|---|---|---|
| `low_vol` | Low realised volatility, trending | 95% of equity | 1.25× |
| `mid_vol` (trend) | Transition, trend signal present | 95% of equity | 1.0× |
| `mid_vol` (no trend) | Transition, no clear trend | 60% of equity | 1.0× |
| `high_vol` | Elevated volatility, risk-off | 60% of equity | 1.0× |

The HMM maps its internal states to these labels via BIC-selected model size (3–7 states) and rolling stability filtering (requires 3 consecutive bars in the same state before acting).

---

## Project layout

```
regime-trader/
├── config/
│   ├── settings.yaml          # All tuneable parameters (universe, risk, HMM, backtest)
│   └── credentials.yaml       # Alpaca API keys — git-ignored, never commit
│
├── core/
│   ├── hmm_engine.py          # Gaussian HMM with BIC model selection and incremental update
│   ├── regime_strategies.py   # Per-regime allocation logic (StrategyOrchestrator)
│   ├── risk_manager.py        # Circuit breakers, position sizing, drawdown limits
│   └── signal_generator.py   # Orchestrates core pipeline -> PortfolioSignal
│
├── broker/
│   ├── alpaca_client.py       # Alpaca REST + WebSocket wrapper (paper and live)
│   ├── order_executor.py      # Limit/market order placement, cancellation, tracking
│   └── position_tracker.py   # Real-time position sync via TradingStream WebSocket
│
├── data/
│   ├── market_data.py         # Historical bars, price pivot, streaming data callbacks
│   └── feature_engineering.py # Causal feature matrix (log returns, realised vol)
│
├── monitoring/
│   ├── logger.py              # JSON-structured rotating file logger (TradeLogger)
│   ├── dashboard.py           # Rich terminal live dashboard (background thread)
│   └── alerts.py              # Email (SMTP/STARTTLS) + webhook alerts with rate limiting
│
├── backtest/
│   ├── backtester.py          # Walk-forward backtester — zero look-ahead bias
│   ├── performance.py         # Sharpe, CAGR, max drawdown, regime breakdown, benchmarks
│   └── stress_test.py         # Crash / gap / vol-spike injection scenarios
│
├── tests/
│   ├── test_hmm.py            # HMM engine unit tests
│   ├── test_look_ahead.py     # Verify zero look-ahead bias in feature pipeline
│   ├── test_strategies.py     # Regime strategy allocation tests (80 tests)
│   ├── test_risk.py           # Risk manager circuit breaker and sizing tests (23 tests)
│   └── test_orders.py         # OrderExecutor limit price, sizing, submission (23 tests)
│
├── main.py                    # CLI entry point (backtest / trade / stress)
├── pytest.ini                 # Suppresses third-party deprecation warnings
└── requirements.txt
```

---

## Quick start

### 1. Prerequisites

Python 3.12 is required. `hmmlearn` has no pre-built wheels for Python 3.13+.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Credentials

Create `config/credentials.yaml` (already git-ignored):

```yaml
alpaca:
  api_key:    "YOUR_ALPACA_KEY"
  secret_key: "YOUR_ALPACA_SECRET"
  paper:      true
```

Get keys from [alpaca.markets](https://alpaca.markets) -> Paper Trading -> API Keys.

Alternatively use a `.env` file (also git-ignored):

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
```

The client loads `credentials.yaml` first and falls back to `.env` / environment variables.

### 3. Run a backtest

```bash
py -3.12 main.py backtest --symbols SPY QQQ --start 2020-01-01 --end 2024-12-31 --compare
```

`--compare` adds a buy-and-hold benchmark column to the output table.

### 4. Run stress tests

```bash
py -3.12 main.py stress
```

### 5. Start paper trading

```bash
py -3.12 main.py trade --paper
```

### 6. Run the test suite

```bash
py -3.12 -m pytest tests/ -v
```

126 tests, ~17 seconds. All pass on the current codebase.

---

## Configuration

All parameters are in [config/settings.yaml](config/settings.yaml). Key sections:

| Section | What it controls |
|---|---|
| `broker` | Trading universe (10 symbols), timeframe, paper vs live, data feed |
| `hmm` | State-count candidates (3–7), stability bars, flicker threshold, min confidence |
| `strategy` | Per-regime allocation fractions, leverage, rebalance threshold, uncertainty multiplier |
| `risk` | Per-trade risk, exposure caps, drawdown thresholds |
| `backtest` | Walk-forward windows, slippage, benchmark, commission model |
| `monitoring` | Dashboard refresh rate, alert rate limit, log level and rotation |

---

## Risk controls

Every order passes through `RiskManager.validate_signal()` — a 16-layer gate:

| Check | Default | Breach behaviour |
|---|---|---|
| Lock file present | — | All trading halted until lock deleted |
| Peak drawdown halt | 10% from peak | Full halt + CRITICAL alert + `trading_halted.lock` written |
| Weekly drawdown halt | 7% | All trading halted for rest of week |
| Daily drawdown halt | 3% | All trading halted for rest of day |
| Weekly drawdown reduce | 5% | Position sizes halved |
| Daily drawdown reduce | 2% | Position sizes halved |
| Max daily trades | 20 | Circuit breaker halt |
| Stop-loss mandatory | — | Trade rejected if no stop provided |
| Duplicate order | 60 s window | Trade rejected |
| Max concurrent positions | 5 | Trade rejected |
| 1% risk rule | 1% of equity / \|entry-stop\| | Size capped |
| Max single position | 15% | Trade rejected |
| Max gross exposure | 80% | Trade rejected |
| Overnight gap risk | 2% equity / (3× stop distance) | Size capped |
| Correlation check | 60-bar rolling | >0.70 halve size, >0.85 reject |
| Buying power | Account cash | Trade rejected |

---

## Order execution

Orders are placed as **limit orders** by default:

- BUY limit = mid-price × (1 + 0.1%) — slightly above to ensure fill
- SELL limit = mid-price × (1 − 0.1%)
- Cancelled after 30 seconds if unfilled
- Optional market-order retry on timeout (`retry_at_market=True`)

Each order carries a unique `client_order_id` in the format `{uuid8}-{SYMBOL}-{side}` so fills can be matched back to internal trade records.

---

## Strategy — entry, sizing, and exit

The strategy is **always long**. It never shorts and never moves entirely to cash. The entire edge is **drawdown avoidance through position sizing**: being 60% invested during a crash rather than 95% is what separates this from buy-and-hold.

### Entry

There is no traditional entry trigger (no crossover, no breakout). The position is opened as soon as the HMM first produces a confirmed regime. After that the portfolio is rebalanced every bar that the regime changes or the EMA filter flips — not on a fixed schedule.

### Position sizing by regime

| Regime | Condition | Portfolio deployed | Leverage |
|---|---|---|---|
| **Low vol** | Calmest HMM state | 95% | 1.25× |
| **Mid vol — trend intact** | Price above 50 EMA | 95% | 1.0× |
| **Mid vol — no trend** | Price below 50 EMA | 60% | 1.0× |
| **High vol** | Most volatile HMM state | 60% | 1.0× |

Allocation is split equally across all symbols in the universe (e.g. 95% ÷ 10 symbols = 9.5% per symbol).

### Uncertainty discount

If the HMM posterior probability is below 0.55, or the regime is flickering (too many flips in a short window), or the new state is not yet confirmed (< `stability_bars` consecutive bars), **all position sizes are halved and leverage is dropped to 1.0×**. This is a soft exit — the system stays invested but at half size until the HMM is confident again.

### Stop loss

Every signal carries a stop computed fresh from the current bar's ATR and EMA:

| Regime | Stop formula |
|---|---|
| Low vol | `max(price − 3×ATR,  50EMA − 0.5×ATR)` |
| Mid vol | `50EMA − 0.5×ATR` |
| High vol | `50EMA − 1.0×ATR` |

The stop floats upward with the EMA as the market rises — it is a trailing stop, not a fixed level.

### Exit / size reduction

A full or partial size reduction is triggered by any of:

1. **Regime transition to higher vol** — the orchestrator immediately rebalances to the new target weight (e.g. 95% → 60% when low-vol shifts to high-vol)
2. **Confidence drops below threshold** — uncertainty discount applied (size halved)
3. **Stop-loss hit** — the risk manager closes the position at the stop price
4. **Risk circuit breaker** — daily / weekly drawdown limits force a size reduction or full halt (see Risk controls section)

There is no take-profit target. Positions are held open-ended until one of the above triggers fires.

### Rebalance filter

A rebalance is skipped if the new target weight is within 10% (relative) of the current weight. This prevents constant micro-trades when the regime is stable:

```
|new_weight − current_weight| < 0.10 × new_weight  →  skip
```

---

## Walk-forward backtesting

The backtester uses a strict walk-forward methodology to prevent look-ahead bias:

- **In-sample window**: configured per run — HMM fitted and parameters selected here
- **Out-of-sample window**: never seen during tuning — strategy evaluated here
- **Step size**: equal to the OOS window (non-overlapping test periods)
- Equity is carried forward between folds (compounding)
- Slippage: 5 bps round-trip per trade

Performance metrics reported per fold and overall: CAGR, Sharpe ratio, max drawdown, annualised volatility, regime breakdown (% time, entries, average duration per regime), regime change count, transition matrix, and comparison to buy-and-hold and SMA-200 benchmarks.

---

## Parameter optimisation and validation workflow

A rigorous three-stage workflow prevents parameter snooping and produces honest out-of-sample results.

### Stage 1 — Parameter sweep (tune on 2020–2023)

```bash
py -3.12 tools/param_sweep.py --asset-group stocks --start 2020-01-01 --end 2023-12-31
```

Runs 11 parameter variants on the **tuning window only**. The 2024+ data is never touched. Outputs a comparison table; winner = config with highest Sharpe on the tuning window.

### Stage 2 — Forward test (blind hold-out 2024–today)

```bash
py -3.12 main.py backtest --asset-group stocks --start 2024-01-01 --compare
```

Single run on data that was never seen during tuning. If Sharpe and MaxDD here are close to the tuning window results, the parameters generalise. A large gap signals overfitting.

### Stage 3 — Rolling Walk-Forward Optimisation (WFO)

```bash
py -3.12 tools/rolling_wfo.py --asset-group stocks --start 2020-01-01
```

The gold-standard validation. For each fold:

```
│←────── tune 12 months ──────→│← test 3 months (blind) →│
                  │←────── tune 12 months ──────→│← test 3 months →│
                                    │←────── tune 12 months ──────→│← test →│
                                                       ...
```

- **Tune phase**: runs all parameter variants on the 12-month tune window; selects the winner by Sharpe
- **Test phase**: applies the winning params blind to the next 3 months — data never seen during tuning
- **Advances** by 3 months and repeats until today (~22 folds from 2020)
- **4:1 ratio** (12m tune / 3m test) is the professional WFO standard

#### What the WFO output tells you

| Metric | Interpretation |
|---|---|
| `% folds beat Buy & Hold` | Core robustness score. ≥ 60% = deploy, 40–60% = paper trade first, < 40% = do not deploy |
| Sharpe distribution (min/mean/max/std) | Wide std = inconsistent edge; tight std = stable strategy |
| Parameter stability table | If the same config wins 10/17 folds, it's structurally sound. If a different config wins every fold, there is no stable edge |
| Recommended live params | Winner of the **most recent fold only** — never averaged across folds |

#### Live parameter selection rule

> **Do not average parameters across folds.**
>
> Use the winner from the most recent completed fold as current live parameters.
> Retune every 3 months on a rolling basis.

```
Every 3 months:
  1. Run param_sweep on the last 6 months of data
  2. Deploy the winning config
  3. Monitor vs previous params — if Sharpe degrades >20%, retune immediately
```

#### CLI options

| Flag | Default | Description |
|---|---|---|
| `--asset-group` | stocks | Asset group to test |
| `--start` | 2020-01-01 | First fold start date |
| `--tune-months` | 12 | Tuning window length |
| `--test-months` | 3 | Blind test window length |
| `--step-months` | 3 | How far to advance each fold |
| `--show-windows` | — | Print fold schedule and bar counts without running backtests |

> **Runtime**: approximately 1.5–2 min per run × `n_variants` × `n_folds`. With default settings (9 variants, ~22 folds from 2020) expect 5–6 hours on a standard machine. Run overnight.

---

## Monitoring

**Terminal dashboard** (`monitoring/dashboard.py`):
- Refreshes every 5 seconds in a background daemon thread
- Panels: header (time, market status), regime (label, confidence, stability, leverage), portfolio (equity, cash, P&L, exposure, peak DD), open positions table (10 columns), recent events log

**Structured logger** (`monitoring/logger.py`):
- JSON lines format: `{"ts": "...", "level": "INFO", "event": "trade", ...}`
- Rotating file handler: 50 MB max, 5 backup files
- Domain helpers: `log_trade`, `log_fill`, `log_regime_change`, `log_rebalance`, `log_risk_event`, `log_error`

**Alert manager** (`monitoring/alerts.py`):
- Email via SMTP with STARTTLS (e.g. Gmail)
- Slack-compatible webhook (also works with Teams, Discord)
- Rate limiting: duplicate alerts within 15 minutes are silently dropped
- Built-in helpers: `alert_drawdown_halt`, `alert_regime_change`, `alert_order_error`

Credentials for alerts are read from the `alerts:` section of `credentials.yaml` or from environment variables (`ALERT_SMTP_HOST`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD`, `ALERT_RECIPIENT`, `ALERT_WEBHOOK_URL`).

---

## Data pipeline

Historical bars are fetched from Alpaca's `StockHistoricalDataClient` and cached in memory by `(symbols, timeframe, start, end)`. The `FeatureEngineer` computes a strictly causal feature matrix (no future data leaks):

| Feature | Description |
|---|---|
| `log_return` | Log of close[t] / close[t-1] |
| `realized_vol_5d` | Rolling 5-bar std dev of log returns |
| `realized_vol_21d` | Rolling 21-bar std dev of log returns |
| `vol_ratio` | `realized_vol_5d / realized_vol_21d` — vol regime indicator |
| `vol_of_vol` | Rolling std dev of `realized_vol_21d` — second-order vol |
| `volume_norm` | Volume / rolling mean volume — relative activity |
| `dist_sma200` | (price − SMA200) / SMA200 — distance from long-term trend |
| `sma50_slope` | Rate of change of SMA50 over a short window — trend momentum |
| `high_low_range` | (high − low) / close — intraday range normalised |
| `close_position` | (close − low) / (high − low) — close position in the bar |
| `overnight_gap` | open[t] / close[t-1] − 1 — gap risk |
| `parkinson_vol` | Parkinson high-low volatility estimator |
| `garman_klass_vol` | Garman-Klass OHLC volatility estimator |
| `amihud_illiquidity` | \|log_return\| / volume — illiquidity proxy |

All raw features are then z-scored over a rolling window before being fed to the HMM, so the model sees standardised inputs regardless of the absolute level of volatility.

Live streaming uses `StockDataStream` (market data) and `TradingStream` (order fills), both running as async coroutines in background daemon threads.

---

## Live trading

To switch from paper to live:

1. Set `paper: false` in `config/credentials.yaml`
2. Replace keys with live trading keys from Alpaca
3. At startup you will be prompted to type exactly: `YES I UNDERSTAND THE RISKS`
4. The system will refuse to start if the phrase is not entered correctly

Live trading is otherwise identical to paper trading — same risk controls, same order logic.

---

## Development notes

- **Python version**: 3.12 required (`hmmlearn` has no wheels for 3.13+)
- **Windows terminal**: `Console(force_terminal=True, legacy_windows=False)` is used to avoid `UnicodeEncodeError` on cp1252 terminals
- **Test isolation**: all broker tests use `MagicMock(spec=AlpacaClient)` — no real network calls
- **Async in sync codebase**: WebSocket streams run via `asyncio.new_event_loop()` in daemon threads, keeping the main trading loop synchronous
