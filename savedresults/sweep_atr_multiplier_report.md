# ATR Stop-Loss Multiplier Sweep

_Run: 2026-04-25 10:34_

## Setup

- **Sweep**: `factor ∈ [1.0, 1.5, 2.0, 3.0]` applied jointly
- `mid_vol_atr_mult = 0.5 × factor`
- `high_vol_atr_mult = 1.0 × factor`
- LowVol stop unchanged. EMA(50), ATR(14), enforce_stops=True, HMM, allocator, risk all default.
- Backtest: `--asset-group stocks --start 2020-01-01` (10-symbol stocks group)
- Walk-forward: IS 252 / OOS 63 / step 63 → 17 folds

## Results

| factor | mid | high | Return | CAGR | Sharpe | Sortino | Calmar | MaxDD | AnnVol | Trades | Stops | Hold (d) | Dominated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1.0 | 0.5 | 1.0 | +68.00% | +12.98% | 0.740 | 0.831 | 0.876 | -14.83% | 0.00% | 548 | 141 | 17.6 |  |
| 1.5 | 0.75 | 1.5 | +65.06% | +12.52% | 0.702 | 0.789 | 0.844 | -14.83% | 0.00% | 549 | 140 | 17.9 | yes |
| 2.0 | 1.0 | 2.0 | +64.46% | +12.42% | 0.693 | 0.778 | 0.834 | -14.89% | 0.00% | 542 | 134 | 18.0 | yes |
| 3.0 | 1.5 | 3.0 | +66.53% | +12.75% | 0.713 | 0.794 | 0.828 | -15.39% | 0.00% | 529 | 120 | 19.2 | yes |

## Per-fold STOP_OUT distribution

| fold | f=1.0 | f=1.5 | f=2.0 | f=3.0 |
|---:|---:|---:|---:|---:|
| 0 | 13 | 13 | 13 | 13 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 4 | 4 | 4 | 4 |
| 3 | 17 | 16 | 16 | 14 |
| 4 | 6 | 5 | 3 | 3 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 15 | 16 | 15 | 14 |
| 7 | 10 | 10 | 9 | 8 |
| 8 | 3 | 3 | 3 | 3 |
| 9 | 0 | 0 | 0 | 0 |
| 10 | 7 | 7 | 6 | 6 |
| 11 | 0 | 0 | 0 | 0 |
| 12 | 37 | 37 | 37 | 35 |
| 13 | 8 | 8 | 8 | 2 |
| 14 | 0 | 0 | 0 | 0 |
| 15 | 9 | 9 | 8 | 6 |
| 16 | 12 | 12 | 12 | 12 |

## Interpretation notes

- **Higher factor → wider stops → fewer stop-outs → more trend capture, but larger per-trade loss when a stop hits.**
- A row marked `Dominated` is strictly worse on Sharpe AND Calmar AND MaxDD than at least one other row — eliminate it from consideration.
- `factor=1.0` is the current default (matches BASELINE_AFTER_ENFORCE_STOPS for stocks).
- Output dirs per row recorded in script stdout; intermediate `savedresults/backtest_*` dirs preserved.

## Output dirs

- factor=1.0: `C:\Users\espac\localrepo\regime-trader\savedresults\backtest_2026-04-25_103044`
- factor=1.5: `C:\Users\espac\localrepo\regime-trader\savedresults\backtest_2026-04-25_103132`
- factor=2.0: `C:\Users\espac\localrepo\regime-trader\savedresults\backtest_2026-04-25_103236`
- factor=3.0: `C:\Users\espac\localrepo\regime-trader\savedresults\backtest_2026-04-25_103336`