# Monte Carlo Asset Selection Analysis

Research scripts to determine whether the current 5-symbol basket
[SPY, QQQ, AAPL, MSFT, NVDA] is optimal, or whether a more diversified
combination from a 40-symbol universe produces better risk-adjusted returns.

**Zero changes to the main codebase.** Scripts only read config and write to `results/`.

---

## The Question

Current basket avg pairwise correlation ≈ 0.75 (4 tech-heavy assets).
Can we find a 5–7 symbol combo with lower correlation that matches or beats Sharpe 1.123?

---

## Candidate Universe (40 symbols)

| Category | Symbols |
|----------|---------|
| Broad ETFs | SPY, QQQ, IWM |
| Sector ETFs | XLK, XLF, XLE, XLV, XLI, XLU, XLC |
| Alt / Macro | GLD, TLT, VNQ |
| Large-cap (2019 MC) | AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, PG, V, MA, UNH, HD, BAC, XOM, CVX, WMT, KO, PEP |
| Growth/Semis | NVDA, AMD, QCOM |
| Defense | LMT, RTX, NOC |
| Healthcare | PFE, ABT |
| Crypto (experimental) | BTC/USD, ETH/USD |

---

## Scripts

### Script 1 — `build_correlation_matrix.py`
Fetches daily close prices for all 40 symbols and saves a pairwise correlation matrix.

```bash
# From repo root (WSL)
source .venv/bin/activate
python research/asset_selection/build_correlation_matrix.py
```
Output: `results/corr_matrix.csv`

### Script 2 — `monte_carlo_asset_selection.py`
Two-phase Monte Carlo backtest.

```bash
python research/asset_selection/monte_carlo_asset_selection.py
```

**Phase 1** (3-year screening, 2023→today): tests ~900 low-correlation combos. ~2 hours.
**Phase 2** (6-year validation, 2020→today): tests top 20 Phase 1 finalists. ~5 min.

Output: `results/mc_results_phase1.csv`, `results/mc_results_phase2.csv`

Resume-safe — interrupted runs continue from where they left off.

### Script 3 — `asset_selection_report.py`
Reads Phase 2 results and prints a full analysis.

```bash
python research/asset_selection/asset_selection_report.py
```
Output: printed report + `results/asset_selection_summary.md`

---

## Speed Strategy

Phase 1 uses a 3-year window (2023–2026) for fast screening (~7s/backtest).
Phase 2 uses the full 6-year window (2020–2026) only for the top 20 finalists.
Total: ~2 hours instead of 3–4 hours for a naive full-window scan.

---

## Interpretation

| Finding | Meaning |
|---------|---------|
| Baseline in top-3 | Current basket is near-optimal for this universe |
| Better combo found, no hindsight bias | Consider updating asset_groups.yaml |
| Better combo found, uses recent winners | Do not adopt — selection bias |
| Crypto in top combos | Low corr but likely poor Sharpe — check MaxDD |
| Defense/TLT in top combos | Genuine diversification benefit confirmed |
