"""
dashstreamlite/data_loader.py — read-only access to bot state and history.

All functions in this module read existing files written by the bot and
backtester. None of them write or modify anything outside dashstreamlite/.
"""
from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ── Paths ─────────────────────────────────────────────────────────────────────

def latest_backtest_dir() -> Optional[Path]:
    sr = ROOT / "savedresults"
    if not sr.exists():
        return None
    dirs = sorted(sr.glob("backtest_*"), reverse=True)
    return dirs[0] if dirs else None


# ── Static config ─────────────────────────────────────────────────────────────

@dataclass
class BotContext:
    bot_name:    str
    host:        str
    asset_group: str
    config_set:  str
    regime_proxy: str
    timeframe_loop: str
    timeframe_hmm:  str
    git_sha:     str


def load_bot_context() -> BotContext:
    cfg_path = ROOT / "config" / "settings.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    broker = cfg.get("broker", {}) or {}
    hmm    = cfg.get("hmm",    {}) or {}

    active_set_path = ROOT / "config" / "active_set"
    config_set = (
        active_set_path.read_text().strip()
        if active_set_path.exists() and active_set_path.read_text().strip()
        else "base"
    )

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=2,
        ).stdout.strip() or "?"
    except Exception:
        sha = "?"

    return BotContext(
        bot_name       = ROOT.name,
        host           = socket.gethostname(),
        asset_group    = broker.get("asset_group", "—"),
        config_set     = config_set,
        regime_proxy   = hmm.get("regime_proxy", ""),
        timeframe_loop = broker.get("timeframe", "1Day"),
        timeframe_hmm  = hmm.get("timeframe",    "1Day"),
        git_sha        = sha,
    )


# ── Live state snapshot ───────────────────────────────────────────────────────

def load_state_snapshot() -> Optional[dict]:
    p = ROOT / "state_snapshot.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── Latest backtest output ────────────────────────────────────────────────────

@dataclass
class BacktestSummary:
    dir_name:    str
    total_return: float
    cagr:        float
    sharpe:      float
    sortino:     float
    calmar:      float
    max_drawdown: float
    max_dd_days: int
    win_rate:    float
    trades:      int
    final_equity: float
    folds:       int
    symbols:     list
    asset_group: str
    config_set:  str
    regime_proxy: str
    start_date:  str
    end_date:    str


def load_backtest_summary(dir_path: Path) -> Optional[BacktestSummary]:
    csv  = dir_path / "performance_summary.csv"
    ctx  = dir_path / "run_context.json"
    if not csv.exists():
        return None
    s = pd.read_csv(csv, header=None, index_col=0).squeeze()
    c = json.loads(ctx.read_text()) if ctx.exists() else {}

    return BacktestSummary(
        dir_name      = dir_path.name,
        total_return  = float(s.get("total_return", 0)),
        cagr          = float(s.get("cagr", 0)),
        sharpe        = float(s.get("sharpe", 0)),
        sortino       = float(s.get("sortino", 0)),
        calmar        = float(s.get("calmar", 0)),
        max_drawdown  = float(s.get("max_drawdown", 0)),
        max_dd_days   = int(float(s.get("max_dd_days", 0))),
        win_rate      = float(s.get("win_rate", 0)),
        trades        = int(float(s.get("total_trades", 0))),
        final_equity  = float(s.get("final_equity", 0)),
        folds         = int(float(s.get("n_folds", 0))),
        symbols       = c.get("symbols", []),
        asset_group   = c.get("asset_group", "—"),
        config_set    = c.get("config_set", ""),
        regime_proxy  = c.get("regime_proxy", ""),
        start_date    = c.get("start_date", str(s.get("start", "")))[:10],
        end_date      = c.get("end_date",   str(s.get("end",   "")))[:10],
    )


def load_regime_history(dir_path: Path) -> Optional[pd.DataFrame]:
    p = dir_path / "regime_history.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.columns = ["regime"]
    return df


def load_equity_curve(dir_path: Path) -> Optional[pd.DataFrame]:
    p = dir_path / "equity_curve.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


# ── Regime segments (run-length encode) ───────────────────────────────────────

def regime_segments(regime_history: pd.DataFrame) -> list[dict]:
    """Return a list of dicts, one per contiguous regime run, ordered oldest
    to newest. Each dict has: regime, start, end, days, bars."""
    if regime_history is None or regime_history.empty:
        return []
    col = regime_history["regime"].astype(str)
    changes = col != col.shift()
    seg_id = changes.cumsum()
    out = []
    for _, group in col.groupby(seg_id):
        out.append({
            "regime": group.iloc[0],
            "start":  group.index[0],
            "end":    group.index[-1],
            "bars":   len(group),
            "days":   (group.index[-1] - group.index[0]).days,
        })
    return out


# ── Price data for the regime proxy (yfinance) ────────────────────────────────

def fetch_proxy_ohlcv(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """OHLCV DataFrame for the proxy via yfinance, with lowercased column
    names so it can be fed straight to FeatureEngineer.build_feature_matrix.
    Returns None on failure rather than raising."""
    try:
        import yfinance as yf
        df = yf.download(
            symbol, start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def compute_full_period_regimes(
    symbol: str,
    start: str,
    end: str,
    hmm_params_repr: tuple,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Train ONE HMM on the full [start, end] window for `symbol` and return
    a tuple (DataFrame[date, regime], None) on success, or (None, error_str)
    on failure. The error string is surfaced to the dashboard so the user can
    see what actually went wrong instead of a generic "training failed"
    message.

    Differences from the bot's walk-forward backtest:
      - single fold, single label mapping over the entire timeline
      - features computed from `symbol` alone (no cross-symbol blending)
      - uses yfinance OHLCV (so the dashboard has no Alpaca dep)
    """
    try:
        from core.hmm_engine import HMMEngine
        from data.feature_engineering import FeatureEngineer
    except Exception as e:
        return None, f"import error: {e}"

    bars = fetch_proxy_ohlcv(symbol, start, end)
    if bars is None or bars.empty:
        return None, f"yfinance returned no bars for {symbol} on [{start}, {end}]"
    needed = {"close", "high", "low"}
    missing = needed - set(bars.columns)
    if missing:
        return None, f"OHLCV missing columns {sorted(missing)} (got {sorted(bars.columns)})"

    params = dict(hmm_params_repr)

    fe = FeatureEngineer()
    try:
        features = fe.build_feature_matrix(
            bars,
            feature_names=list(params.get("feature_names", ["log_ret_1", "realized_vol_20"])),
        )
    except Exception as e:
        return None, f"feature build failed: {type(e).__name__}: {e}"
    features_clean = features.dropna()

    min_train = int(params.get("min_train_bars", 252))
    if len(features_clean) < min_train:
        return None, (f"only {len(features_clean)} clean feature rows "
                      f"(need {min_train}) — try a longer history window")

    engine = HMMEngine(
        n_candidates    = list(params.get("n_candidates", [5])),
        n_init          = int(params.get("n_init", 10)),
        covariance_type = str(params.get("covariance_type", "full")),
        min_train_bars  = min_train,
        stability_bars  = int(params.get("stability_bars", 7)),
        flicker_window  = int(params.get("flicker_window", 20)),
        flicker_threshold = int(params.get("flicker_threshold", 4)),
        min_confidence  = float(params.get("min_confidence", 0.62)),
    )
    try:
        engine.fit(features_clean.values)
    except Exception as e:
        return None, f"HMM fit failed: {type(e).__name__}: {e}"

    states = engine.predict_regime_filtered(features_clean.values)
    if not states:
        return None, "HMM produced zero states"

    return (
        pd.DataFrame(
            {"regime": [s.label for s in states]},
            index=features_clean.index[: len(states)],
        ),
        None,
    )


def fetch_proxy_prices(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    """Daily close-price series for the regime proxy via yfinance.
    Returns None on failure rather than raising — the dashboard degrades
    gracefully to showing only the regime overlay if prices are unavailable."""
    try:
        import yfinance as yf
        df = yf.download(
            symbol, start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None
        # yfinance returns multi-index columns sometimes; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" not in df.columns:
            return None
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = symbol
        return s
    except Exception:
        return None
