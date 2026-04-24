#!/usr/bin/env python3
"""
Script 2: monte_carlo_asset_selection.py
Two-phase Monte Carlo asset selection:

  Phase 1 (fast screening) — 3-year backtest on ~900 low-correlation combos
  Phase 2 (full validation) — 6-year backtest on top 20 Phase 1 results

Usage (from repo root, WSL):
    source .venv/bin/activate
    python research/asset_selection/monte_carlo_asset_selection.py

Resume-safe: already-completed combos are skipped on re-run.
"""
from __future__ import annotations
import os
import sys
import json
import itertools
from pathlib import Path
from datetime import date
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd
import numpy as np
import yaml

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

PHASE1_CSV = OUT / "mc_results_phase1.csv"
PHASE2_CSV = OUT / "mc_results_phase2.csv"

# ── Configuration ─────────────────────────────────────────────────────────────
UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLC",
    "GLD", "TLT", "VNQ",
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "PG",
    "V", "MA", "UNH", "HD", "BAC", "XOM", "CVX", "WMT", "KO", "PEP",
    "NVDA", "AMD", "QCOM",
    "LMT", "RTX", "NOC",
    "PFE", "ABT",
    "BTC/USD", "ETH/USD",
]

BASELINE      = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
PHASE1_START  = "2023-01-01"
PHASE2_START  = "2020-01-01"
END_DATE      = str(date.today())
TOP_PER_K     = 300   # lowest-corr combos per k fed to Phase 1
PHASE2_TOP_N  = 20    # top Phase 1 results → Phase 2
SAMPLE_K6     = 60_000
SAMPLE_K7     = 60_000
RNG_SEED      = 42

BROAD_ETFS = {"SPY", "QQQ", "IWM"}

# ── Load credentials ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

cfg_path = ROOT / "config" / "settings.yaml"
with open(cfg_path) as f:
    base_cfg = yaml.safe_load(f)

creds_path = ROOT / "config" / "credentials.yaml"
if creds_path.exists():
    with open(creds_path) as f:
        _creds = yaml.safe_load(f)
    _alpaca = _creds.get("alpaca", {})
    api_key    = _alpaca.get("api_key",    os.getenv("ALPACA_API_KEY",    ""))
    secret_key = _alpaca.get("secret_key", os.getenv("ALPACA_SECRET_KEY", ""))
else:
    api_key    = os.getenv("ALPACA_API_KEY",    "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

if not api_key or not secret_key:
    sys.exit("ERROR: Alpaca credentials not found.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_crypto(sym: str) -> bool:
    return "/" in sym

def pick_regime_proxy(combo: tuple[str, ...]) -> str:
    if "QQQ" in combo:
        return "QQQ"
    for sym in combo:
        if sym in BROAD_ETFS:
            return sym
    return combo[0]

def combo_key(combo: tuple[str, ...]) -> str:
    return json.dumps(sorted(combo))

def avg_pairwise_corr(corr_mat: pd.DataFrame, syms: list[str]) -> float:
    vals = []
    for i, a in enumerate(syms):
        for b in syms[i+1:]:
            if a in corr_mat.index and b in corr_mat.index:
                vals.append(corr_mat.loc[a, b])
    return float(np.mean(vals)) if vals else 1.0

# ── Price fetching ────────────────────────────────────────────────────────────
from alpaca.data.timeframe import TimeFrame

def fetch_daily_closes(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    stock_syms  = [s for s in symbols if not _is_crypto(s)]
    crypto_syms = [s for s in symbols if     _is_crypto(s)]
    frames = []

    if stock_syms:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        sc  = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
            adjustment=Adjustment.ALL,
        )
        df = sc.get_stock_bars(req).df
        if not df.empty and "close" in df.columns:
            close = df["close"].unstack(level="symbol")
            close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
            close.index.name = "date"
            frames.append(close)

    if crypto_syms:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        cc  = CryptoHistoricalDataClient(api_key, secret_key)
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=TimeFrame.Day,
            start=start, end=end,
        )
        df = cc.get_crypto_bars(req).df
        if not df.empty and "close" in df.columns:
            close = df["close"].unstack(level="symbol")
            close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
            close.index.name = "date"
            frames.append(close)

    if not frames:
        return pd.DataFrame()
    combined = frames[0].join(frames[1], how="outer") if len(frames) > 1 else frames[0]
    return combined.sort_index().dropna(how="all")

# ── Run one backtest ──────────────────────────────────────────────────────────
from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer

bt_cfg  = base_cfg.get("backtest", {})
strat_cfg = base_cfg.get("strategy", {})
risk_cfg  = base_cfg.get("risk", {})
PA = PerformanceAnalyzer(
    risk_free_rate=float(bt_cfg.get("risk_free_rate", 0.045))
)

def run_backtest(
    combo: tuple[str, ...],
    prices: pd.DataFrame,
) -> Optional[dict]:
    syms = [s for s in combo if s in prices.columns]
    if len(syms) < 2:
        return None
    proxy = pick_regime_proxy(tuple(syms))
    hmm_cfg = {**base_cfg.get("hmm", {}), "regime_proxy": proxy}

    bt = WalkForwardBacktester(
        symbols         = syms,
        initial_capital = float(bt_cfg.get("initial_capital", 100_000)),
        train_window    = int(bt_cfg.get("train_window", 252)),
        test_window     = int(bt_cfg.get("test_window",  63)),
        step_size       = int(bt_cfg.get("step_size",    63)),
        slippage_pct    = float(bt_cfg.get("slippage_pct", 0.0005)),
        risk_free_rate  = float(bt_cfg.get("risk_free_rate", 0.045)),
    )
    try:
        result = bt.run(
            prices          = prices[syms],
            hmm_config      = hmm_cfg,
            strategy_config = strat_cfg,
            risk_config     = risk_cfg,
        )
        report = PA.analyze(result)
        return {
            "symbols":      combo_key(tuple(syms)),
            "k":            len(syms),
            "regime_proxy": proxy,
            "sharpe":       report.sharpe_ratio,
            "max_drawdown": report.max_drawdown,
            "total_return": report.total_return,
            "cagr":         report.cagr,
            "calmar":       report.calmar_ratio,
            "n_trades":     report.total_trades,
            "win_rate":     report.win_rate,
        }
    except Exception as e:
        print(f"    ERROR {syms}: {e}")
        return None

# ── Step 1: load correlation matrix ──────────────────────────────────────────
corr_path = OUT / "corr_matrix.csv"
if not corr_path.exists():
    sys.exit("ERROR: corr_matrix.csv not found. Run build_correlation_matrix.py first.")

corr = pd.read_csv(corr_path, index_col=0)
avail = [s for s in UNIVERSE if s in corr.index]
print(f"Loaded correlation matrix: {len(avail)} / {len(UNIVERSE)} symbols available")

# ── Step 2: enumerate combinations and score by correlation ──────────────────
print("\nScoring combinations by average pairwise correlation...")
rng = np.random.default_rng(RNG_SEED)
scored: list[tuple[float, tuple[str, ...]]] = []

for k in [5, 6, 7]:
    print(f"  k={k} ...", end=" ", flush=True)
    if k == 5:
        combos = list(itertools.combinations(avail, k))
    else:
        n_total = 1
        for i in range(k):
            n_total = n_total * (len(avail) - i) // (i + 1)
        idx = rng.choice(n_total, size=min(SAMPLE_K6 if k == 6 else SAMPLE_K7, n_total), replace=False)
        idx_set = set(idx.tolist())
        combos = []
        for i, c in enumerate(itertools.combinations(avail, k)):
            if i in idx_set:
                combos.append(c)

    corr_scores = [(avg_pairwise_corr(corr, list(c)), c) for c in combos]
    corr_scores.sort(key=lambda x: x[0])
    top = corr_scores[:TOP_PER_K]
    scored.extend(top)
    print(f"{len(combos):,} evaluated → top {len(top)} kept "
          f"(corr range {top[0][0]:.3f}–{top[-1][0]:.3f})")

# Always include the baseline
baseline_tuple = tuple(sorted(BASELINE))
baseline_corr  = avg_pairwise_corr(corr, list(baseline_tuple))
scored_keys    = {combo_key(c) for _, c in scored}
if combo_key(baseline_tuple) not in scored_keys:
    scored.append((baseline_corr, baseline_tuple))
    print(f"  Baseline added (corr={baseline_corr:.3f})")

print(f"\nTotal combos for Phase 1: {len(scored)}")

# ── Step 3: Phase 1 — 3-year fast screening ───────────────────────────────────
print(f"\n{'─'*72}")
print(f"  PHASE 1 — 3-year screening  ({PHASE1_START} → {END_DATE})")
print(f"{'─'*72}")

# Load existing Phase 1 results (resume support)
done_keys: set[str] = set()
phase1_rows: list[dict] = []
if PHASE1_CSV.exists():
    existing = pd.read_csv(PHASE1_CSV)
    for _, row in existing.iterrows():
        done_keys.add(row["symbols"])
        phase1_rows.append(row.to_dict())
    print(f"  Resuming: {len(done_keys)} combos already completed")

print(f"  Fetching 3-year price data for {len(avail)} symbols...")
prices_3yr = fetch_daily_closes(avail, PHASE1_START, END_DATE)
print(f"  Got {len(prices_3yr)} bars × {prices_3yr.shape[1]} symbols")

todo = [(corr_val, combo) for corr_val, combo in scored
        if combo_key(combo) not in done_keys]
print(f"  Running {len(todo)} backtests...\n")

for i, (corr_val, combo) in enumerate(todo):
    key = combo_key(combo)
    is_baseline = (key == combo_key(baseline_tuple))
    label = " ← BASELINE" if is_baseline else ""
    print(f"  [{i+1:>4}/{len(todo)}] k={len(combo)} corr={corr_val:.3f} "
          f"{list(combo)[:4]}...{label}", end=" ", flush=True)

    row = run_backtest(combo, prices_3yr)
    if row is not None:
        row["avg_corr"]   = corr_val
        row["is_baseline"] = is_baseline
        phase1_rows.append(row)
        print(f"Sharpe={row['sharpe']:.3f}  Return={row['total_return']:+.1%}  MaxDD={row['max_drawdown']:.1%}")
    else:
        print("SKIPPED")

    if (i + 1) % 50 == 0 or (i + 1) == len(todo):
        pd.DataFrame(phase1_rows).to_csv(PHASE1_CSV, index=False)
        print(f"    → checkpoint saved ({len(phase1_rows)} rows)")

print(f"\nPhase 1 complete. Results: {PHASE1_CSV}")

# ── Step 4: Phase 2 — full 6-year validation on top-N ────────────────────────
print(f"\n{'─'*72}")
print(f"  PHASE 2 — full validation  ({PHASE2_START} → {END_DATE})")
print(f"{'─'*72}")

phase1_df = pd.read_csv(PHASE1_CSV)
phase1_df = phase1_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)

# Top N + always include baseline
top_keys   = set(phase1_df.head(PHASE2_TOP_N)["symbols"].tolist())
top_keys.add(combo_key(baseline_tuple))
finalists  = [(row["avg_corr"], json.loads(row["symbols"]))
              for _, row in phase1_df.iterrows()
              if row["symbols"] in top_keys]

print(f"  Finalists: {len(finalists)} combos (top {PHASE2_TOP_N} + baseline)")

# Load existing Phase 2 results
done2_keys: set[str] = set()
phase2_rows: list[dict] = []
if PHASE2_CSV.exists():
    existing2 = pd.read_csv(PHASE2_CSV)
    for _, row in existing2.iterrows():
        done2_keys.add(row["symbols"])
        phase2_rows.append(row.to_dict())
    print(f"  Resuming: {len(done2_keys)} already done")

print(f"  Fetching 6-year price data for {len(avail)} symbols...")
prices_6yr = fetch_daily_closes(avail, PHASE2_START, END_DATE)
print(f"  Got {len(prices_6yr)} bars × {prices_6yr.shape[1]} symbols")

todo2 = [(cv, tuple(sorted(c))) for cv, c in finalists
         if combo_key(tuple(sorted(c))) not in done2_keys]
print(f"  Running {len(todo2)} full backtests...\n")

for i, (corr_val, combo) in enumerate(todo2):
    key = combo_key(combo)
    is_baseline = (key == combo_key(baseline_tuple))
    label = " ← BASELINE" if is_baseline else ""
    print(f"  [{i+1:>3}/{len(todo2)}] k={len(combo)} corr={corr_val:.3f} "
          f"{list(combo)[:4]}...{label}", end=" ", flush=True)

    # Get Phase 1 Sharpe for reference
    p1_sharpe = phase1_df.loc[phase1_df["symbols"] == key, "sharpe"]
    p1_s = f" (3yr Sharpe={p1_sharpe.iloc[0]:.3f})" if len(p1_sharpe) else ""

    row = run_backtest(combo, prices_6yr)
    if row is not None:
        row["avg_corr"]    = corr_val
        row["is_baseline"] = is_baseline
        p1_row = phase1_df[phase1_df["symbols"] == key]
        row["sharpe_3yr"] = float(p1_row["sharpe"].iloc[0]) if len(p1_row) else float("nan")
        phase2_rows.append(row)
        print(f"Sharpe={row['sharpe']:.3f} (3yr={row['sharpe_3yr']:.3f})  "
              f"Return={row['total_return']:+.1%}  MaxDD={row['max_drawdown']:.1%}{p1_s}")
    else:
        print("SKIPPED")

pd.DataFrame(phase2_rows).to_csv(PHASE2_CSV, index=False)
print(f"\nPhase 2 complete. Results: {PHASE2_CSV}")
print(f"Next step: run asset_selection_report.py")
