# POSSIBLEUPDATES.md

Ideas worth exploring later, not blocking current work. Each one is a
self-contained improvement that doesn't impact the live bot today, listed
in rough priority order.

---

## 1. Min hold time per regime  (priority: medium)

**Problem.** `hmm.stability_bars` controls how many consecutive bars the HMM
must agree before *entering* a new regime, but nothing prevents the bot from
*leaving* a regime the very next bar. With `broker.timeframe=5Min` (intraday
loop), this can churn — switch to BULL, back to NEUTRAL one tick later, etc.

**Idea.** Add a `min_hold_bars` setting under `strategy:` that locks the active
regime for at least N bars after a transition, regardless of what the HMM says
in the meantime. Hold time would be timeframe-aware:

| Loop timeframe | Suggested `min_hold_bars` |
|---|---|
| 1Day  | 1 (no extra lock — daily already slow) |
| 1Hour | 4 (≈ half a session) |
| 5Min  | 24 (≈ 2 hours) |
| 1Min  | 60 (≈ 1 hour) |

**Where to wire it.** Inside `core/regime_strategies.py` (StrategyOrchestrator)
or `core/risk_manager.validate_signal()` — gate any "regime changed → rebalance"
signal with a check on bars-since-last-transition.

**Cost.** ~30 lines + 1 yaml key. Backtest validation on `balanced × stocks`
to confirm Sharpe doesn't drop materially.

**Origin.** Borrowed from the Regime Terminal v2.1 community update — they
ship one `min_hold_bars` per timeframe preset.

---

## 2. Per-timeframe Sharpe annualization  (priority: latent bug)

**Problem.** `backtest/performance.py` uses `trading_days_per_year=252` to
annualize Sharpe, Sortino, vol. That's correct for daily bars but **silently
wrong** for any other timeframe:

- 1Hour bars on a 6.5h US session: `~1638 bars/year`
- 5Min bars: `~19656 bars/year`
- 1Min bars: `~98280 bars/year`

A backtest on 5Min bars currently reports Sharpe values that are off by
roughly a factor of √(19656/252) ≈ 8.8× lower than reality. Today we only
backtest 1Day, so the bug is dormant. But the moment someone runs
`python main.py backtest --timeframe 5Min ...` they'll get wrong numbers
without any warning.

**Idea.** Replace the hardcoded 252 with a derived `bars_per_year` looked up
from `_N_BARS_BY_TF` (already defined in `main.py`) or from a new
`backtest/timeframe_constants.py`:

```python
BARS_PER_YEAR = {
    "1Day":  252,
    "1Hour": 252 * 6.5,
    "5Min":  252 * 78,
    "1Min":  252 * 390,
}
```

Pass the timeframe through the backtest pipeline so `PerformanceAnalyzer`
can read the right value. Add a unit test that runs the same daily strategy
at two timeframes and confirms Sharpe stays consistent up to data noise.

**Cost.** ~20 lines + 1 test. Touches `backtest/performance.py` and the
backtester wiring.

**Origin.** The Regime Terminal v2.1 update flagged this — they switched to
`bars_per_hour`-based annualization for the same reason.

---

## 3. Config sets per timeframe  (priority: low)

**Problem.** Today's three sets (`conservative`, `balanced`, `aggressive`)
trade prudence vs aggression on a single timeframe. They all assume daily
bars implicitly (e.g. `stability_bars: 7` = 7 trading days = 1.5 weeks of
confirmation, which is way too long on 5Min where 7 bars = 35 minutes).

**Idea.** Add a second dimension of presets keyed on timeframe:

```
config/sets/
├── conservative.yaml      # current (daily)
├── balanced.yaml          # current (daily)
├── aggressive.yaml        # current (daily)
├── intraday_5min.yaml     # n_candidates [3], leverage 1.0×, stability=24, min_conf=0.55
├── intraday_1hour.yaml    # n_candidates [5], leverage 1.0×, stability=8,  min_conf=0.62
└── swing_4hour.yaml       # n_candidates [5], leverage 1.5×, stability=6,  min_conf=0.62
```

Composable via a future `--timeframe-set` flag, or just by `--set
intraday_5min`. Naming convention separates the "risk profile" axis
(conservative/balanced/aggressive) from the "timeframe profile" axis.

**Cost.** Pure YAML once we have a recommended set of params per timeframe.
Calibration via the existing `bash scripts/sweep_groups.sh <set>` tooling
on each new set. No code change.

**Origin.** The Regime Terminal v2.1 update bundles four timeframe presets
(Swing King / Default / Day Driver / Scalp Pro) with all params tuned
together. Same pattern, just adapted to our naming and risk policy.

---

## What we deliberately did NOT take from v2.1

For traceability — these were considered and rejected:

- **`n_states=7`** as default. We picked `[5]` after the WEAK_BULL/STRONG_BULL
  label-stability investigation. More states means more inter-fold remapping
  noise — the very problem piste 3 just fixed.
- **`leverage 2.5×`** on 4h. Outside our risk policy. Aggressive set caps at
  1.5× and we're not increasing it without a separate risk review.
- **"Confirmations 7/8"** as opposed to consecutive-bar stability. Different
  paradigm; not clearly better. Our `stability_bars` works fine.
