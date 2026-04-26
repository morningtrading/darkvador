"""
scripts/backtest_mean_reversion.py
====================================

Stand-alone backtest of strategies.mean_reversion_qqq_spy. Validates the
two acceptance criteria:

  1. Sharpe (annualised, daily) ≥ 0.3 standalone
  2. Correlation with hmm_regime equity curve < 0.3

Data source: yfinance (no Alpaca needed). The hmm_regime equity curve is
loaded from the most recent savedresults/backtest_*/equity_curve.csv —
the same one the dashboard already reads.

Output: prints the metrics, writes nothing to disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies.mean_reversion_qqq_spy import (
    MeanReversionQqqSpyConfig,
    MeanReversionQqqSpyStrategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def fetch_yf(symbol: str, start: str, end: str) -> pd.Series | None:
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False,
                         auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = symbol
        return s.astype(float)
    except Exception as exc:
        print(f"yfinance fetch failed for {symbol}: {exc}")
        return None


def load_hmm_equity_curve() -> pd.Series | None:
    sr = ROOT / "savedresults"
    if not sr.exists():
        return None
    dirs = sorted(sr.glob("backtest_*"), reverse=True)
    for d in dirs:
        eq = d / "equity_curve.csv"
        if eq.exists():
            df = pd.read_csv(eq, index_col=0, parse_dates=True)
            print(f"  using HMM equity curve from: {d.name}")
            return df["equity"].astype(float)
    return None


def annualised_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() < 1e-12:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def annualised_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    e = equity.dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return 0.0
    years = len(e) / periods_per_year
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


# ── Backtest ──────────────────────────────────────────────────────────────────


def backtest(
    qqq: pd.Series,
    spy: pd.Series,
    config: MeanReversionQqqSpyConfig,
    initial_equity: float = 100_000.0,
) -> dict:
    """Run the strategy day-by-day. Returns (equity_curve, daily_returns, metrics)."""
    strategy = MeanReversionQqqSpyStrategy(config)
    weights = strategy.generate_target_weights(qqq, spy)

    common = qqq.index.intersection(spy.index).intersection(weights.index)
    qqq = qqq.loc[common]
    spy = spy.loc[common]
    weights = weights.loc[common]

    # Forward (next-bar) returns: signal at t is held through t+1 close.
    qqq_ret = qqq.pct_change().shift(-1)  # return from close[t] to close[t+1]
    spy_ret = spy.pct_change().shift(-1)

    # Daily portfolio return = sum_i (weight_i_t * return_i_{t→t+1}). Cash for the
    # un-deployed fraction (1 - allocation) earns 0 — conservative.
    port_ret = (weights["weight_qqq"] * qqq_ret) + (weights["weight_spy"] * spy_ret)
    port_ret = port_ret.dropna()

    equity = (1.0 + port_ret).cumprod() * initial_equity

    metrics = {
        "n_days":        int(len(port_ret)),
        "first_day":     str(port_ret.index[0].date()) if len(port_ret) else None,
        "last_day":      str(port_ret.index[-1].date()) if len(port_ret) else None,
        "total_return":  float(equity.iloc[-1] / initial_equity - 1.0) if len(equity) else 0.0,
        "cagr":          annualised_cagr(equity),
        "ann_vol":       float(port_ret.std() * np.sqrt(252)),
        "sharpe":        annualised_sharpe(port_ret),
        "max_drawdown":  max_drawdown(equity),
        "final_equity":  float(equity.iloc[-1]) if len(equity) else initial_equity,
        "avg_qqq_weight": float(weights["weight_qqq"].mean()),
        "avg_spy_weight": float(weights["weight_spy"].mean()),
        "z_min":         float(weights["z"].min()),
        "z_max":         float(weights["z"].max()),
        "z_active_days": int((weights["z"].abs() > config.threshold).sum()),
    }
    return {"equity": equity, "returns": port_ret, "metrics": metrics, "weights": weights}


def correlate_with_hmm(strategy_returns: pd.Series, hmm_equity: pd.Series) -> float | None:
    if hmm_equity is None:
        return None
    hmm_returns = hmm_equity.pct_change().dropna()
    common = strategy_returns.index.intersection(hmm_returns.index)
    if len(common) < 60:
        return None
    a = strategy_returns.loc[common]
    b = hmm_returns.loc[common]
    return float(a.corr(b))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    start = "2020-01-01"
    end   = "2026-04-26"
    print(f"Backtesting MeanReversionQqqSpy on {start} → {end}")
    print("Fetching prices from yfinance ...")
    qqq = fetch_yf("QQQ", start, end)
    spy = fetch_yf("SPY", start, end)
    if qqq is None or spy is None:
        print("FAILED to fetch one or both series.")
        return 1
    print(f"  QQQ: {len(qqq)} bars, {qqq.index[0].date()} → {qqq.index[-1].date()}")
    print(f"  SPY: {len(spy)} bars, {spy.index[0].date()} → {spy.index[-1].date()}")

    print("\nLoading HMM-regime equity curve for correlation ...")
    hmm_eq = load_hmm_equity_curve()
    if hmm_eq is None:
        print("  no HMM equity curve found — correlation step will be skipped")

    cfg = MeanReversionQqqSpyConfig()
    print(f"\nConfig: {cfg}")

    print("\nRunning standalone backtest ...")
    res = backtest(qqq, spy, cfg)
    m = res["metrics"]

    print("\n=== Standalone results ===")
    print(f"  Period      : {m['first_day']} → {m['last_day']} ({m['n_days']} days)")
    print(f"  Total return: {m['total_return']*100:+.2f}%")
    print(f"  CAGR        : {m['cagr']*100:+.2f}%")
    print(f"  Sharpe      : {m['sharpe']:+.3f}")
    print(f"  Ann. vol    : {m['ann_vol']*100:+.2f}%")
    print(f"  MaxDD       : {m['max_drawdown']*100:+.2f}%")
    print(f"  Avg QQQ wt  : {m['avg_qqq_weight']:.3f}  (target ~{cfg.allocation/2:.3f} ± {cfg.max_tilt*cfg.allocation:.3f})")
    print(f"  Avg SPY wt  : {m['avg_spy_weight']:.3f}")
    print(f"  z range     : [{m['z_min']:+.2f}, {m['z_max']:+.2f}]")
    print(f"  Active tilts: {m['z_active_days']} days where |z| > {cfg.threshold}")
    print(f"  Final equity: ${m['final_equity']:,.0f}")

    if hmm_eq is not None:
        corr = correlate_with_hmm(res["returns"], hmm_eq)
        print("\n=== Correlation with hmm_regime ===")
        if corr is None:
            print("  (insufficient overlap)")
        else:
            print(f"  Pearson correlation of daily returns: {corr:+.3f}")
            print(f"  Acceptance threshold: < 0.3  →  {'PASS ✓' if abs(corr) < 0.3 else 'FAIL ✗'}")

    print("\n=== Acceptance check ===")
    sharpe_ok = m["sharpe"] >= 0.3
    print(f"  Sharpe ≥ 0.3       : {'PASS ✓' if sharpe_ok else 'FAIL ✗'}  ({m['sharpe']:+.3f})")
    if hmm_eq is not None:
        corr_ok = corr is not None and abs(corr) < 0.3
        print(f"  |corr_hmm| < 0.3   : {'PASS ✓' if corr_ok else 'FAIL ✗'}")
        decision_ok = sharpe_ok and corr_ok
    else:
        decision_ok = sharpe_ok
        print(f"  |corr_hmm| < 0.3   : SKIPPED (no reference)")

    # ── Combined view: hmm_regime + mean_reversion ─────────────────────────────
    if hmm_eq is not None:
        print("\n=== Combined: hmm_regime + mean_reversion ===")
        hmm_returns = hmm_eq.pct_change().dropna()
        common = res["returns"].index.intersection(hmm_returns.index)
        if len(common) >= 60:
            r_mr  = res["returns"].loc[common]
            r_hmm = hmm_returns.loc[common]
            blends = [
                ("100% hmm only",    1.00, 0.00),
                ("70% hmm / 30% mr",  0.70, 0.30),
                ("50% hmm / 50% mr",  0.50, 0.50),
                ("30% hmm / 70% mr",  0.30, 0.70),
                ("0% hmm / 100% mr",  0.00, 1.00),
            ]
            # Inverse-vol blend (the bot's default allocator).
            v_h = float(r_hmm.std() * np.sqrt(252))
            v_m = float(r_mr.std()  * np.sqrt(252))
            inv_h, inv_m = 1.0 / max(v_h, 1e-9), 1.0 / max(v_m, 1e-9)
            w_h_iv = inv_h / (inv_h + inv_m)
            w_m_iv = inv_m / (inv_h + inv_m)
            blends.append((f"inverse_vol ({w_h_iv:.0%}/{w_m_iv:.0%})", w_h_iv, w_m_iv))

            print(f"  {'Blend':<28} {'Sharpe':>8} {'AnnRet':>9} {'AnnVol':>8} {'MaxDD':>9}")
            print("  " + "-" * 68)
            for label, w_h, w_m in blends:
                rb = w_h * r_hmm + w_m * r_mr
                eq = (1.0 + rb).cumprod()
                s   = annualised_sharpe(rb)
                ar  = float(rb.mean() * 252)
                av  = float(rb.std() * np.sqrt(252))
                dd  = max_drawdown(eq)
                print(f"  {label:<28} {s:>+8.3f} {ar*100:>+8.2f}% {av*100:>+7.2f}% {dd*100:>+8.2f}%")

    print()
    if decision_ok:
        print("VERDICT: candidate qualifies. Worth wiring into the multi-strategy framework.")
        return 0
    else:
        print("VERDICT: candidate does NOT qualify. Re-tune or drop.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
