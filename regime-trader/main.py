"""
main.py — Regime Trader entry point.

    python main.py trade        — live / paper trading loop
    python main.py backtest     — walk-forward backtest
    python main.py stress       — stress-test scenario suite
    python main.py trade --dry-run     — full pipeline, no orders placed
    python main.py trade --train-only  — train HMM and exit

All tuneable parameters live in config/settings.yaml.
Credentials are read from config/credentials.yaml (git-ignored) or from
ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("main")

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).resolve().parent
_MODEL_PATH   = _ROOT / "models" / "hmm.pkl"
_SNAPSHOT_PATH = _ROOT / "state_snapshot.json"
_MODEL_MAX_AGE_DAYS = 7

# ── HMM feature helper (defined in data/feature_engineering.py) ───────────────
from data.feature_engineering import hmm_feature_names as _hmm_feature_names


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """Load the YAML settings file and return it as a nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_credentials(credentials_path: str = "config/credentials.yaml") -> Optional[Dict]:
    """
    Load credentials YAML if it exists and inject values into env vars so
    the rest of the code can use os.getenv() uniformly.
    """
    path = Path(credentials_path)
    if not path.exists():
        return None
    with path.open() as fh:
        creds = yaml.safe_load(fh) or {}

    alpaca = creds.get("alpaca", {})
    if alpaca.get("api_key") and not os.environ.get("ALPACA_API_KEY"):
        os.environ["ALPACA_API_KEY"] = alpaca["api_key"]
    if alpaca.get("secret_key") and not os.environ.get("ALPACA_SECRET_KEY"):
        os.environ["ALPACA_SECRET_KEY"] = alpaca["secret_key"]

    return creds


# ── Data fetching ──────────────────────────────────────────────────────────────

def _is_crypto(symbol: str) -> bool:
    """Return True if the symbol looks like a crypto pair (e.g. BTC/USD)."""
    return "/" in symbol


def _fetch_prices(
    symbols: List[str],
    start: str,
    end: str,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """
    Download daily close prices from Alpaca for all symbols.

    Automatically routes crypto symbols (containing '/') to the
    CryptoHistoricalDataClient and equity symbols to StockHistoricalDataClient.

    Returns wide-format DataFrame: index = date (tz-naive), columns = symbols.
    """
    from alpaca.data.timeframe import TimeFrame

    crypto_syms = [s for s in symbols if _is_crypto(s)]
    stock_syms  = [s for s in symbols if not _is_crypto(s)]
    frames: List[pd.DataFrame] = []

    # ── Equity bars ──────────────────────────────────────────────────────────
    if stock_syms:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        sc = StockHistoricalDataClient(api_key, secret_key)
        req = StockBarsRequest(
            symbol_or_symbols=stock_syms,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
        )
        df = sc.get_stock_bars(req).df
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        close.index.name = "date"
        frames.append(close)

    # ── Crypto bars ───────────────────────────────────────────────────────────
    if crypto_syms:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        # Crypto data client needs no auth for historical data
        cc = CryptoHistoricalDataClient(api_key, secret_key)
        req = CryptoBarsRequest(
            symbol_or_symbols=crypto_syms,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        df = cc.get_crypto_bars(req).df
        close = df["close"].unstack(level="symbol")
        close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
        close.index.name = "date"
        frames.append(close)

    if not frames:
        return pd.DataFrame()

    combined = frames[0] if len(frames) == 1 else frames[0].join(frames[1], how="outer")
    combined = combined.sort_index()

    available = [s for s in symbols if s in combined.columns]
    missing   = [s for s in symbols if s not in combined.columns]
    if missing:
        logger.warning("Symbols not returned by Alpaca: %s", missing)

    return combined[available].dropna(how="all")


# ── Asset group resolver ──────────────────────────────────────────────────────

def _resolve_symbols(config: Dict, asset_group: Optional[str], symbols_arg: Optional[str]) -> List[str]:
    """
    Resolve the symbol list in priority order:
      1. --symbols flag (comma-separated)
      2. --asset-group flag  → looks up config['asset_groups'][name]
      3. broker.asset_group in config → looks up config['asset_groups'][name]
      4. broker.symbols list in config
    """
    if symbols_arg:
        return [s.strip() for s in symbols_arg.split(",")]

    groups = config.get("asset_groups", {})
    group_name = asset_group or config.get("broker", {}).get("asset_group")

    if group_name:
        syms = groups.get(group_name)
        if syms:
            return list(syms)
        logger.warning("Asset group '%s' not found in config; falling back to broker.symbols", group_name)

    return config.get("broker", {}).get("symbols", ["SPY"])


# ── Rich helpers ───────────────────────────────────────────────────────────────

def _console():
    try:
        import sys
        from rich.console import Console
        # Let Rich auto-detect TTY so ANSI codes are not injected when output
        # is piped or captured (e.g. param_sweep subprocess, grep pipelines).
        return Console(legacy_windows=False, force_terminal=sys.stdout.isatty())
    except ImportError:
        return None


def _print(msg: str, console=None, style: str = "") -> None:
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


def _print_run_config(
    symbols:        List[str],
    start_date:     str,
    end_date:       str,
    config:         Dict,
    console=None,
    label:          str = "Run Configuration",
) -> None:
    """
    Print a compact recap of all parameters used in a backtest or sweep run.
    Shown before every results table so results are always self-contained.
    """
    broker_cfg   = config.get("broker",   {})
    hmm_cfg      = config.get("hmm",      {})
    strategy_cfg = config.get("strategy", {})
    risk_cfg     = config.get("risk",      {})
    bt_cfg       = config.get("backtest", {})

    lines = [
        f"\n{'─' * 62}",
        f"  {label}",
        f"{'─' * 62}",
        f"  Assets      : {', '.join(symbols)}",
        f"  Period      : {start_date}  →  {end_date}",
        f"  Frequency   : {broker_cfg.get('timeframe', '5Min')} bars",
        f"  Capital     : ${float(bt_cfg.get('initial_capital', 100_000)):,.0f}   "
        f"Slippage: {float(bt_cfg.get('slippage_pct', 0.0005)) * 10_000:.1f} bps",
        f"  Walk-fwd    : IS {bt_cfg.get('train_window', 252)} / "
        f"OOS {bt_cfg.get('test_window', 126)} bars  "
        f"step {bt_cfg.get('step_size', 126)}",
        f"",
        f"  HMM         : states {hmm_cfg.get('n_candidates', [3,4,5,6,7])}  "
        f"covariance={hmm_cfg.get('covariance_type', 'full')}  "
        f"n_init={hmm_cfg.get('n_init', 10)}",
        f"  Regime filt : stability={hmm_cfg.get('stability_bars', 3)}  "
        f"flicker_thresh={hmm_cfg.get('flicker_threshold', 4)}  "
        f"flicker_win={hmm_cfg.get('flicker_window', 20)}  "
        f"min_conf={hmm_cfg.get('min_confidence', 0.55)}",
        f"  Features    : {', '.join(hmm_cfg.get('features', []))}",
        f"",
        f"  Allocation  : low_vol={strategy_cfg.get('low_vol_allocation', 0.95)}×"
        f"{strategy_cfg.get('low_vol_leverage', 1.25)}x  "
        f"mid={strategy_cfg.get('mid_vol_allocation_trend', 0.95)}/"
        f"{strategy_cfg.get('mid_vol_allocation_no_trend', 0.60)}  "
        f"high={strategy_cfg.get('high_vol_allocation', 0.60)}",
        f"  Rebalance   : threshold={strategy_cfg.get('rebalance_threshold', 0.10)}  "
        f"trend_lookback={strategy_cfg.get('trend_lookback', 50)}",
        f"  Risk        : max_pos={risk_cfg.get('max_single_position', 0.15)}  "
        f"max_exp={risk_cfg.get('max_exposure', 0.80)}  "
        f"dd_halt={risk_cfg.get('daily_dd_halt', 0.03)}",
        f"{'─' * 62}",
    ]
    for line in lines:
        _print(line, console, style="dim")



def _print_comparison_table(
    strategy_report,
    bnh_metrics: Optional[Dict],
    sma_metrics: Optional[Dict],
    ema_cross_metrics: Optional[Dict],
    rand_metrics: Optional[Dict],
    console=None,
    n_symbols: int = 1,
) -> None:
    """Print a side-by-side benchmark comparison table."""
    rows = [
        ("Total Return",     "total_return",    "{:+.2%}"),
        ("CAGR",             "cagr",            "{:+.2%}"),
        ("Sharpe Ratio",     "sharpe_ratio",    "{:.3f}"),
        ("Sortino Ratio",    "sortino_ratio",   "{:.3f}"),
        ("Max Drawdown",     "max_drawdown",    "{:.2%}"),
        ("Calmar Ratio",     "calmar_ratio",    "{:.3f}"),
        ("Ann. Volatility",  "annualised_vol",  "{:.2%}"),
        ("Win Rate",         "win_rate",        "{:.2%}"),
    ]

    def _val(d, key, fmt):
        if d is None:
            return "N/A"
        v = d.get(key) if isinstance(d, dict) else getattr(d, key, None)
        if v is None:
            return "N/A"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    bnh_label      = "Buy & Hold (EW)" if n_symbols > 1 else "Buy & Hold"
    sma_label      = "SMA-200 (EW)"    if n_symbols > 1 else "SMA-200 Trend"
    ema_cross_label = "EMA 9/45 (EW)"  if n_symbols > 1 else "EMA 9/45 Cross"

    if console:
        try:
            from rich.table import Table
            from rich import box as rbox
            tbl = Table(title="Benchmark Comparison", box=rbox.ROUNDED,
                        header_style="bold yellow")
            tbl.add_column("Metric",          style="dim", min_width=18)
            tbl.add_column("Strategy",        justify="right")
            tbl.add_column(bnh_label,         justify="right")
            tbl.add_column(sma_label,         justify="right")
            tbl.add_column(ema_cross_label,   justify="right")
            tbl.add_column("Random (mean)",   justify="right")
            for label, key, fmt in rows:
                tbl.add_row(
                    label,
                    _val(strategy_report, key, fmt),
                    _val(bnh_metrics, key, fmt),
                    _val(sma_metrics, key, fmt),
                    _val(ema_cross_metrics, key, fmt),
                    _val(rand_metrics, key, fmt),
                )
            console.print(tbl)
            return
        except ImportError:
            pass

    print("\nBENCHMARK COMPARISON")
    print(f"{'Metric':<20} {'Strategy':>12} {'BnH':>12} {'SMA-200':>12} {'EMA 9/45':>12} {'Random':>12}")
    print("-" * 82)
    for label, key, fmt in rows:
        print(
            f"{label:<20} {_val(strategy_report, key, fmt):>12} "
            f"{_val(bnh_metrics, key, fmt):>12} "
            f"{_val(sma_metrics, key, fmt):>12} "
            f"{_val(ema_cross_metrics, key, fmt):>12} "
            f"{_val(rand_metrics, key, fmt):>12}"
        )


# ── Timeframe helpers ──────────────────────────────────────────────────────────

# Minutes per bar for each supported timeframe
_TF_MINUTES: Dict[str, int] = {
    "1Min":  1,
    "5Min":  5,
    "15Min": 15,
    "30Min": 30,
    "1Hour": 60,
    "4Hour": 240,
    "1Day":  390,   # treat daily as end-of-session (6.5 hours)
    "1Week": 1950,
}

# NYSE open/close in UTC
_MARKET_OPEN_UTC  = dt.time(14, 30)   # 09:30 ET = 14:30 UTC
_MARKET_CLOSE_UTC = dt.time(21,  0)   # 16:00 ET = 21:00 UTC


def _next_bar_close_utc(timeframe: str) -> dt.datetime:
    """
    Return the next bar-close UTC datetime for the given timeframe.
    For daily bars this is today's (or next trading day's) market close.
    For intraday bars this is the next N-minute boundary after now.
    """
    now = dt.datetime.now(dt.timezone.utc)
    minutes = _TF_MINUTES.get(timeframe, 5)

    if timeframe in ("1Day", "1Week"):
        # Align to next market close (21:00 UTC)
        today_close = now.replace(
            hour=_MARKET_CLOSE_UTC.hour,
            minute=_MARKET_CLOSE_UTC.minute,
            second=0, microsecond=0,
        )
        if now >= today_close:
            today_close += dt.timedelta(days=1)
        # Skip to Monday if landing on weekend
        while today_close.weekday() >= 5:
            today_close += dt.timedelta(days=1)
        return today_close

    # Round up to next N-minute boundary
    epoch = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    elapsed = (now - epoch).total_seconds() / 60.0
    next_boundary = (int(elapsed / minutes) + 1) * minutes
    return epoch + dt.timedelta(minutes=next_boundary)


def _seconds_until(target: dt.datetime) -> float:
    delta = (target - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return max(0.0, delta)


# ── State snapshot ─────────────────────────────────────────────────────────────

def _save_snapshot(state: Dict) -> None:
    """Persist a JSON state snapshot for crash recovery."""
    try:
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SNAPSHOT_PATH, "w") as fh:
            json.dump(state, fh, indent=2, default=str)
        logger.info("State snapshot saved to %s", _SNAPSHOT_PATH)
    except Exception as exc:
        logger.error("Failed to save state snapshot: %s", exc)


def _load_snapshot() -> Optional[Dict]:
    """Load a prior state snapshot if it exists."""
    if not _SNAPSHOT_PATH.exists():
        return None
    try:
        with open(_SNAPSHOT_PATH) as fh:
            data = json.load(fh)
        logger.info("Loaded state snapshot from %s", _SNAPSHOT_PATH)
        return data
    except Exception as exc:
        logger.warning("Could not load state snapshot: %s", exc)
        return None


# ── HMM training / loading ─────────────────────────────────────────────────────

def _hmm_needs_retrain(model_path: Path, max_age_days: int) -> bool:
    """Return True if the model file is missing or older than max_age_days."""
    if not model_path.exists():
        return True
    age = dt.datetime.now() - dt.datetime.fromtimestamp(model_path.stat().st_mtime)
    return age.days >= max_age_days


def _train_hmm(
    client,
    symbols: List[str],
    hmm_cfg: Dict,
    console=None,
    n_bars: int = 5_000,  # ~3 months of 5-min bars (78 bars/day × 65 days)
) -> "HMMEngine":
    """Fetch data, compute features, fit HMM, save to disk. Returns fitted engine."""
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import FeatureEngineer

    tf = hmm_cfg.get("timeframe", "5Min")
    _print(f"Training HMM on {tf} bars ...", console, style="dim")

    # Bars per calendar day for each timeframe (trading days only, ~252/year)
    _bars_per_day = {
        "1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13,
        "1Hour": 7,  "4Hour": 2, "1Day": 1,   "1Week": 0.2,
    }
    bars_per_day  = _bars_per_day.get(tf, 78)
    # Add 40% buffer for weekends/holidays
    lookback_days = max(10, int(n_bars / bars_per_day * 1.4 * 7 / 5) + 5)

    # Use the first symbol (benchmark) to build the feature matrix
    ref_symbol = symbols[0]
    end   = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    bars_df = client.get_bars(
        symbols=[ref_symbol],
        timeframe=tf,
        start=start,
        end=end,
    )

    if bars_df.empty:
        raise RuntimeError(f"No bars returned for {ref_symbol} during HMM training")

    # Flatten multi-index to single-symbol OHLCV
    if isinstance(bars_df.index, pd.MultiIndex):
        sym_bars = bars_df.xs(ref_symbol, level="symbol") if ref_symbol in bars_df.index.get_level_values("symbol") else bars_df.droplevel(0)
    else:
        sym_bars = bars_df

    sym_bars.index = pd.to_datetime(sym_bars.index).tz_localize(None)
    sym_bars = sym_bars.sort_index()

    # Add synthetic volume if missing
    if "volume" not in sym_bars.columns:
        sym_bars["volume"] = 1_000_000.0

    fe = FeatureEngineer()
    features_clean = fe.build_feature_matrix(
        sym_bars, feature_names=_hmm_feature_names(hmm_cfg)
    )

    if len(features_clean) < hmm_cfg.get("min_train_bars", 252):
        raise RuntimeError(
            f"Only {len(features_clean)} clean feature rows — "
            f"need {hmm_cfg.get('min_train_bars', 252)} for HMM training."
        )

    engine = HMMEngine(
        n_candidates       = hmm_cfg.get("n_candidates", [3, 4, 5]),
        n_init             = hmm_cfg.get("n_init", 10),
        stability_bars     = hmm_cfg.get("stability_bars", 3),
        flicker_window     = hmm_cfg.get("flicker_window", 20),
        flicker_threshold  = hmm_cfg.get("flicker_threshold", 4),
        min_confidence     = hmm_cfg.get("min_confidence", 0.55),
        min_train_bars     = hmm_cfg.get("min_train_bars", 252),
    )
    engine.fit(features_clean.values)
    engine.save(str(_MODEL_PATH))

    _print(
        f"  HMM trained: {engine._n_states} states  "
        f"BIC={engine._training_bic:.2f}  "
        f"bars={len(features_clean)}",
        console, style="dim",
    )
    return engine


def _load_or_train_hmm(
    client,
    symbols: List[str],
    hmm_cfg: Dict,
    force_retrain: bool = False,
    console=None,
) -> "HMMEngine":
    """Load model from disk if fresh, else retrain."""
    from core.hmm_engine import HMMEngine

    needs_train = force_retrain or _hmm_needs_retrain(_MODEL_PATH, _MODEL_MAX_AGE_DAYS)

    if not needs_train:
        try:
            engine = HMMEngine.load(str(_MODEL_PATH))
            age_days = (
                dt.datetime.now() -
                dt.datetime.fromtimestamp(_MODEL_PATH.stat().st_mtime)
            ).days
            _print(
                f"  Loaded HMM model from {_MODEL_PATH}  "
                f"(age={age_days}d  states={engine._n_states})",
                console, style="dim",
            )
            return engine
        except Exception as exc:
            logger.warning("Failed to load HMM model: %s  -- retraining", exc)

    return _train_hmm(client, symbols, hmm_cfg, console=console)


# ── Bar fetching for live loop ─────────────────────────────────────────────────

def _fetch_live_bars(
    client,
    symbols: List[str],
    timeframe: str,
    n_bars: int = 300,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch the latest n_bars OHLCV bars for each symbol.

    Returns dict of symbol -> DataFrame with columns [open, high, low, close, volume].
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import Adjustment
    from broker.alpaca_client import parse_timeframe

    tf = parse_timeframe(timeframe)
    # Determine start date from n_bars lookback
    bars_per_day = {
        "1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13,
        "1Hour": 7,  "4Hour": 2, "1Day":  1,  "1Week": 0.2,
    }.get(timeframe, 1)
    # lookback must cover the full n_bars *plus* the feature warm-up period
    # (dist_sma200 needs 200 bars; rolling z-score needs 252 more → 452 warm-up).
    # Add a 40% calendar buffer for weekends/holidays.
    warmup_bars   = n_bars + 500          # 500-bar buffer covers all warm-up windows
    lookback_days = max(15, int(warmup_bars / bars_per_day * 1.4 * 7 / 5) + 5)
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Do NOT pass a limit — let the date range control volume, then tail() selects
    # the most recent n_bars.  A hard limit would truncate to the oldest bars.
    req = StockBarsRequest(
        symbol_or_symbols = symbols,
        timeframe          = tf,
        start              = start,
        adjustment         = Adjustment.ALL,
    )
    bars = client._data_client.get_stock_bars(req)
    raw  = bars.df

    result: Dict[str, pd.DataFrame] = {}
    if raw.empty:
        return result

    # Split multi-index by symbol
    if isinstance(raw.index, pd.MultiIndex):
        # MultiIndex is (symbol, timestamp) or (timestamp, symbol)
        levels = raw.index.names
        sym_level = levels.index("symbol") if "symbol" in levels else 0
        ts_level  = 1 - sym_level
        for sym in symbols:
            try:
                sym_df = raw.xs(sym, level=sym_level)
            except KeyError:
                continue
            sym_df.index = pd.to_datetime(sym_df.index).tz_localize(None)
            # Keep n_bars + 500 so feature warm-up (dist_sma200=200, z-score=252)
            # produces valid rows for the most recent n_bars.
            sym_df = sym_df.sort_index().tail(n_bars + 500)
            if "volume" not in sym_df.columns:
                sym_df["volume"] = 1_000_000.0
            result[sym] = sym_df
    else:
        for sym in symbols:
            if sym in raw.columns:
                result[sym] = raw[[sym]].rename(columns={sym: "close"})

    return result


# ── Session summary ────────────────────────────────────────────────────────────

def _print_session_summary(
    session_start: dt.datetime,
    equity_start: float,
    equity_end: float,
    bar_count: int,
    trade_count: int,
    console=None,
) -> None:
    elapsed    = dt.datetime.now(dt.timezone.utc) - session_start
    total_pnl  = equity_end - equity_start
    pnl_pct    = total_pnl / max(equity_start, 1.0)
    sign       = "+" if total_pnl >= 0 else ""

    _print("\n" + "=" * 55, console)
    _print("[bold]Session Summary[/bold]", console)
    _print("=" * 55, console)
    _print(f"  Duration     : {str(elapsed).split('.')[0]}", console)
    _print(f"  Bars         : {bar_count}", console)
    _print(f"  Trades       : {trade_count}", console)
    _print(f"  Equity start : ${equity_start:>12,.2f}", console)
    _print(f"  Equity end   : ${equity_end:>12,.2f}", console)
    _print(
        f"  Session P&L  : {sign}${abs(total_pnl):>10,.2f}  ({sign}{pnl_pct:.2%})",
        console,
        style="green" if total_pnl >= 0 else "red",
    )
    _print("=" * 55, console)


# ── Trading session ────────────────────────────────────────────────────────────

class TradingSession:
    """
    Encapsulates the entire live trading lifecycle:
    startup -> main loop -> shutdown.
    """

    def __init__(self, config: Dict, dry_run: bool = False) -> None:
        self.config  = config
        self.dry_run = dry_run
        self.console = _console()

        # Component references (set during startup)
        self.client           = None
        self.hmm_engine       = None
        self.strategy         = None
        self.risk_manager     = None
        self.position_tracker = None
        self.dashboard        = None
        self.trade_logger     = None
        self.alert_manager    = None

        # Session state
        self._running:         bool           = False
        self._session_start:   dt.datetime    = dt.datetime.now(dt.timezone.utc)
        self._equity_at_start: float          = 0.0
        self._bar_count:       int            = 0
        self._trade_count:     int            = 0
        self._last_retrain:    dt.datetime    = dt.datetime.now(dt.timezone.utc)
        self._snapshot:        Optional[Dict] = None
        self._last_regime:     str            = "UNKNOWN"
        self._price_history:   Dict[str, pd.Series] = {}  # for risk manager

    # ======================================================================= #
    # Startup                                                                  #
    # ======================================================================= #

    def startup(self) -> None:
        """Full system startup sequence."""
        console = self.console
        broker_cfg   = self.config.get("broker",     {})
        hmm_cfg      = self.config.get("hmm",        {})
        risk_cfg     = self.config.get("risk",        {})
        monitor_cfg  = self.config.get("monitoring",  {})

        symbols   = broker_cfg.get("symbols", ["SPY"])
        timeframe = broker_cfg.get("timeframe", "1Day")
        paper     = broker_cfg.get("paper_trading", True)

        _print(
            f"\n[bold cyan]Regime Trader[/bold cyan]  "
            f"{'[yellow]DRY RUN[/yellow]' if self.dry_run else '[green]LIVE[/green]'}  "
            f"{'PAPER' if paper else '[red]LIVE MONEY[/red]'}",
            console,
        )

        # ── 1. Connect to Alpaca ──────────────────────────────────────────────
        _print("\n[1/7] Connecting to Alpaca ...", console)
        from broker.alpaca_client import AlpacaClient
        self.client = AlpacaClient(paper=paper)
        self.client.connect_with_retry(max_attempts=3, base_delay=2.0)

        account = self.client.get_account()
        equity  = float(account.equity)
        _print(
            f"      Account  : {account.account_number}  "
            f"status={account.status.value}",
            console, style="dim",
        )
        _print(f"      Equity   : ${equity:,.2f}", console, style="dim")
        _print(f"      Cash     : ${float(account.cash):,.2f}", console, style="dim")

        # ── 2. Market hours check ─────────────────────────────────────────────
        _print("\n[2/7] Checking market hours ...", console)
        clock = self.client.get_clock()
        if not clock.is_open:
            next_open = clock.next_open
            _print(
                f"      Market is CLOSED.  Next open: {next_open}",
                console, style="yellow",
            )
            _print(
                "      Continuing -- will wait for next bar close.",
                console, style="dim",
            )

        # ── 3. Load or train HMM ──────────────────────────────────────────────
        _print("\n[3/7] Loading HMM model ...", console)
        self.hmm_engine = _load_or_train_hmm(
            self.client, symbols, hmm_cfg, console=console
        )

        # Build StrategyOrchestrator from fitted regime infos
        from core.regime_strategies import StrategyOrchestrator
        self.strategy = StrategyOrchestrator(
            config        = self.config,
            regime_infos  = self.hmm_engine.get_all_regime_info(),
            min_confidence = hmm_cfg.get("min_confidence", 0.55),
            rebalance_threshold = self.config.get("strategy", {}).get(
                "rebalance_threshold", 0.10
            ),
        )
        _print(
            f"      Strategy orchestrator ready  "
            f"({self.hmm_engine._n_states} states)",
            console, style="dim",
        )

        # ── 4. Risk manager ───────────────────────────────────────────────────
        _print("\n[4/7] Initialising risk manager ...", console)
        from core.risk_manager import RiskManager
        self.risk_manager = RiskManager(
            initial_equity         = equity,
            max_risk_per_trade     = risk_cfg.get("max_risk_per_trade",   0.01),
            max_exposure           = risk_cfg.get("max_exposure",         0.80),
            max_leverage           = risk_cfg.get("max_leverage",         1.25),
            max_single_position    = risk_cfg.get("max_single_position",  0.15),
            max_concurrent         = risk_cfg.get("max_concurrent",       5),
            max_daily_trades       = risk_cfg.get("max_daily_trades",     20),
            daily_dd_reduce        = risk_cfg.get("daily_dd_reduce",      0.02),
            daily_dd_halt          = risk_cfg.get("daily_dd_halt",        0.03),
            weekly_dd_reduce       = risk_cfg.get("weekly_dd_reduce",     0.05),
            weekly_dd_halt         = risk_cfg.get("weekly_dd_halt",       0.07),
            max_dd_from_peak       = risk_cfg.get("max_dd_from_peak",     0.10),
        )
        _print(
            f"      RiskManager ready  equity=${equity:,.2f}",
            console, style="dim",
        )

        # ── 5. Position tracker ───────────────────────────────────────────────
        _print("\n[5/7] Syncing positions ...", console)
        from broker.position_tracker import PositionTracker
        self.position_tracker = PositionTracker(
            client            = self.client,
            risk_manager      = self.risk_manager,
            current_regime_fn = lambda: self._last_regime,
        )
        port_snapshot = self.position_tracker.startup_sync(
            current_regime=self._last_regime
        )
        self._equity_at_start = port_snapshot.total_equity
        _print(
            f"      {len(port_snapshot.positions)} open position(s)  "
            f"equity=${port_snapshot.total_equity:,.2f}",
            console, style="dim",
        )

        # ── 6. State snapshot recovery ────────────────────────────────────────
        _print("\n[6/7] Checking for prior session snapshot ...", console)
        self._snapshot = _load_snapshot()
        if self._snapshot:
            self._last_regime  = self._snapshot.get("regime", "UNKNOWN")
            self._bar_count    = self._snapshot.get("bar_count", 0)
            self._trade_count  = self._snapshot.get("trade_count", 0)
            last_ts = self._snapshot.get("session_start", "")
            _print(
                f"      Recovered from snapshot  "
                f"regime={self._last_regime}  "
                f"bars={self._bar_count}  "
                f"prior_session={last_ts}",
                console, style="dim",
            )
        else:
            _print("      No prior snapshot -- clean start.", console, style="dim")

        # ── 7. Start WebSocket feeds ──────────────────────────────────────────
        _print("\n[7/7] Starting WebSocket feeds ...", console)
        self.position_tracker.start_stream()
        _print("      TradingStream (fills) active.", console, style="dim")

        # ── Monitoring ────────────────────────────────────────────────────────
        from monitoring.logger import TradeLogger
        from monitoring.alerts import AlertManager
        from monitoring.dashboard import Dashboard

        self.trade_logger = TradeLogger(
            log_dir   = monitor_cfg.get("log_dir", "logs/"),
            log_level = monitor_cfg.get("log_level", "INFO"),
        )
        self.trade_logger.setup()

        self.alert_manager = AlertManager(
            rate_limit_minutes = monitor_cfg.get("alert_rate_limit_minutes", 15),
        )

        self.dashboard = Dashboard(
            refresh_seconds = monitor_cfg.get("dashboard_refresh_seconds", 5),
        )
        self.dashboard.start()

        # Populate dashboard immediately so panels don't show "Waiting..."
        from monitoring.dashboard import SystemStatus
        from core.signal_generator import PortfolioSignal
        _initial_system = SystemStatus(
            data_ok          = True,
            api_ok           = True,
            api_latency_ms   = 0.0,
            hmm_last_trained = self.hmm_engine._training_date.to_pydatetime()
                               if getattr(self.hmm_engine, "_training_date", None) is not None else None,
            mode             = "PAPER" if paper else "LIVE",
        )
        # Predict current regime from recent bars so the dashboard shows real data
        _startup_regime   = self._last_regime
        _startup_conf     = 0.0
        _startup_stable   = False
        _startup_notes    = ["Awaiting first live bar"]
        try:
            _ref_sym   = symbols[0]
            _tf        = broker_cfg.get("timeframe", "5Min")
            _pred_bars = _fetch_live_bars(self.client, [_ref_sym], _tf, n_bars=300)
            _ref_df    = _pred_bars.get(_ref_sym)
            if _ref_df is None:
                _startup_notes = [f"Startup predict: no bars returned for {_ref_sym}"]
            elif len(_ref_df) < 10:
                _startup_notes = [f"Startup predict: only {len(_ref_df)} bars for {_ref_sym} (need 10)"]
            else:
                from data.feature_engineering import FeatureEngineer
                from core.hmm_engine import RegimeState
                _fe      = FeatureEngineer()
                _feat_df = _fe.build_feature_matrix(
                    _ref_df, feature_names=_hmm_feature_names(hmm_cfg)
                )
                if len(_feat_df) < 10:
                    _startup_notes = [f"Startup predict: only {len(_feat_df)} clean feature rows (need 10)"]
                else:
                    _states  = self.hmm_engine.predict_regime_filtered(
                        _feat_df.values,
                        timestamps=[pd.Timestamp(ts) for ts in _feat_df.index],
                    )
                    _rs: RegimeState = _states[-1]
                    _startup_regime  = _rs.label
                    _startup_conf    = _rs.probability
                    _startup_stable  = _rs.is_confirmed
                    _startup_notes   = [f"Predicted from last {len(_feat_df)} bars — awaiting first live update"]
                    self._last_regime = _startup_regime
                    # Warm up the incremental HMM filter so the first live
                    # update() call has proper prior state instead of starting
                    # cold and producing a garbage regime on bar 1.
                    for _row in _feat_df.values:
                        self.hmm_engine.update(
                            new_feature_row = _row,
                            timestamp       = None,
                        )
        except Exception as _exc:
            logger.warning("Startup regime predict failed: %s", _exc)
            _startup_notes = [f"Startup predict failed: {_exc}"]

        _startup_signal = PortfolioSignal(
            timestamp       = pd.Timestamp.now(),
            regime          = _startup_regime,
            confidence      = _startup_conf,
            is_stable       = _startup_stable,
            target_weights  = {},
            delta_weights   = {},
            leverage        = 1.0,
            trading_allowed = True,
            notes           = _startup_notes,
        )
        _startup_tf = broker_cfg.get("timeframe", "5Min")
        self.dashboard.update(
            snapshot       = port_snapshot,
            signal         = _startup_signal,
            drawdown_state = self.risk_manager.get_drawdown_state(),
            system_status  = _initial_system,
            market_open    = clock.is_open,
            next_bar_dt    = _next_bar_close_utc(_startup_tf),
            next_broker_dt = _next_bar_close_utc(_startup_tf),
            next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
            timeframe      = _startup_tf,
            asset_group    = broker_cfg.get("asset_group", ""),
            symbols        = symbols,
        )

        # Register fill callback for logging
        def _on_fill(fill_event):
            self.trade_logger.log_fill(
                symbol     = fill_event.symbol,
                side       = fill_event.side,
                qty        = fill_event.qty,
                fill_price = fill_event.fill_price,
                order_id   = fill_event.order_id or "",
            )
            self._trade_count += 1
            self.dashboard.update(event=(
                f"Fill: {fill_event.side.upper()} "
                f"{fill_event.qty:.0f} {fill_event.symbol} "
                f"@ ${fill_event.fill_price:.2f}"
            ))

        self.position_tracker.register_fill_callback(_on_fill)

        # ── System online ─────────────────────────────────────────────────────
        self.trade_logger.info("System online", symbols=symbols, timeframe=timeframe)
        self._running = True
        self._session_start = dt.datetime.now(dt.timezone.utc)

        _print(
            f"\n[bold green]System online[/bold green]  "
            f"symbols={symbols}  timeframe={timeframe}",
            console,
        )

    # ======================================================================= #
    # Main loop                                                                #
    # ======================================================================= #

    def run_loop(self) -> None:
        """
        Main trading loop.  Runs until _running is set to False by shutdown().
        Blocks the calling thread.
        """
        broker_cfg = self.config.get("broker", {})
        hmm_cfg    = self.config.get("hmm",    {})
        symbols    = broker_cfg.get("symbols",   ["SPY"])
        timeframe  = broker_cfg.get("timeframe", "1Day")

        from broker.order_executor import OrderExecutor
        executor = OrderExecutor(
            client           = self.client,
            risk_manager     = self.risk_manager,
            limit_offset_pct = 0.001,
            cancel_after_sec = 30,
            retry_at_market  = True,
        )

        _print(
            f"\nEntering main loop  timeframe={timeframe}  "
            f"{'DRY RUN -- no orders will be placed' if self.dry_run else 'Orders ENABLED'}",
            self.console,
        )

        while self._running:
            # ---- sleep until next bar close ---------------------------------
            next_close = _next_bar_close_utc(timeframe)
            wait_secs  = _seconds_until(next_close)
            self.dashboard.update(
                next_bar_dt    = next_close,
                next_broker_dt = next_close,
                next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
                timeframe      = timeframe,
            )

            if wait_secs > 10:
                logger.debug(
                    "Sleeping %.0fs until next bar close (%s UTC)",
                    wait_secs,
                    next_close.strftime("%H:%M:%S"),
                )
                # Sleep in short chunks so SIGINT is responsive
                slept = 0.0
                while slept < wait_secs and self._running:
                    time.sleep(min(5.0, wait_secs - slept))
                    slept += 5.0

            if not self._running:
                break

            # ---- 1. Fetch latest bars ----------------------------------------
            try:
                bars_by_symbol = _fetch_live_bars(
                    self.client, symbols, timeframe, n_bars=300
                )
            except Exception as exc:
                logger.error("Data fetch failed: %s -- pausing signals", exc)
                self.trade_logger.log_error(exc, context="bar_fetch")
                self.dashboard.update(event=f"[red]Data feed error: {exc}[/red]")
                time.sleep(60)
                continue

            if not bars_by_symbol:
                logger.warning("No bars returned -- skipping bar")
                time.sleep(30)
                continue

            # ---- 2. Compute features (using benchmark symbol) ----------------
            ref_sym = symbols[0]
            ref_bars = bars_by_symbol.get(ref_sym)
            if ref_bars is None or len(ref_bars) < 60:
                logger.warning("Insufficient bars for %s -- skipping", ref_sym)
                continue

            try:
                from data.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                features_clean = fe.build_feature_matrix(
                    ref_bars, feature_names=_hmm_feature_names(hmm_cfg)
                )
                if len(features_clean) < 10:
                    logger.warning("Too few clean feature rows -- skipping bar")
                    continue
                feature_matrix = features_clean.values
            except Exception as exc:
                logger.error("Feature computation failed: %s", exc)
                self.trade_logger.log_error(exc, context="feature_engineering")
                continue

            # ---- 3. HMM incremental update -----------------------------------
            try:
                from core.hmm_engine import RegimeState
                timestamp = pd.Timestamp(features_clean.index[-1])
                regime_state: RegimeState = self.hmm_engine.update(
                    new_feature_row = feature_matrix[-1],
                    timestamp       = timestamp,
                )
                self._last_regime = regime_state.label
            except Exception as exc:
                logger.error(
                    "HMM update failed: %s -- holding regime=%s",
                    exc, self._last_regime,
                )
                self.trade_logger.log_error(exc, context="hmm_update")
                # Hold current regime; synthesise minimal state
                from dataclasses import dataclass
                regime_state = _make_fallback_regime(self._last_regime)

            # ---- 4. Stability / flicker checks --------------------------------
            is_flickering = self.hmm_engine.is_flickering()
            is_stable     = regime_state.is_confirmed and not is_flickering
            confidence    = regime_state.probability
            min_conf      = hmm_cfg.get("min_confidence", 0.55)
            uncertainty   = is_flickering or confidence < min_conf

            logger.info(
                "Bar %d | regime=%s | p=%.3f | stable=%s | flicker=%s",
                self._bar_count, regime_state.label, confidence,
                regime_state.is_confirmed, is_flickering,
            )

            # ---- 5. Build per-symbol price DataFrames for strategy -----------
            # Retain price history for risk-manager correlation check
            for sym, sym_bars in bars_by_symbol.items():
                if "close" in sym_bars.columns:
                    self._price_history[sym] = sym_bars["close"].tail(60)

            # ---- 6. Generate signals -----------------------------------------
            try:
                raw_signals = self.strategy.generate_signals(
                    symbols       = symbols,
                    bars          = bars_by_symbol,
                    regime_state  = regime_state,
                    is_flickering = is_flickering,
                )
            except Exception as exc:
                logger.error("Strategy signal generation failed: %s", exc)
                self.trade_logger.log_error(exc, context="signal_generation")
                raw_signals = []

            # ---- 7. Risk gate + order submission ----------------------------
            _port_snap_now = self.position_tracker.get_last_snapshot()
            positions_dict = {
                p.symbol: p.market_value
                for p in (_port_snap_now.positions if _port_snap_now else [])
            }

            from core.risk_manager import PortfolioState, TradingState
            portfolio_state = PortfolioState(
                equity          = self.risk_manager._current_equity,
                cash            = self.client.get_cash(),
                buying_power    = self.client.get_buying_power(),
                positions       = positions_dict,
                current_regime  = self._last_regime,
                price_history   = self._price_history,
            )

            for signal in raw_signals:
                if not signal.is_long:
                    continue

                decision = self.risk_manager.validate_signal(
                    signal          = signal,
                    portfolio_state = portfolio_state,
                )

                if not decision.approved:
                    logger.info(
                        "Signal REJECTED: %s  reason=%s",
                        signal.symbol, decision.rejection_reason,
                    )
                    self.trade_logger.log_risk_event(
                        event_type  = "rejected",
                        description = f"{signal.symbol}: {decision.rejection_reason}",
                        severity    = "WARNING",
                    )
                    self.dashboard.update(
                        event=f"REJECTED {signal.symbol}: {decision.rejection_reason}"
                    )
                    continue

                final_signal = decision.modified_signal
                if decision.modifications:
                    logger.info(
                        "Signal MODIFIED: %s  changes=%s",
                        signal.symbol, decision.modifications,
                    )

                # Log the trade intent
                self.trade_logger.log_trade(
                    symbol   = final_signal.symbol,
                    side     = "buy" if final_signal.is_long else "sell",
                    qty      = 0,   # actual qty determined by executor
                    price    = final_signal.entry_price,
                    extra    = {
                        "regime":  regime_state.label,
                        "dry_run": self.dry_run,
                        "size_pct": round(final_signal.position_size_pct, 4),
                    },
                )

                if self.dry_run:
                    _print(
                        f"  [DRY RUN] Would submit: {final_signal.symbol}  "
                        f"size={final_signal.position_size_pct:.1%}  "
                        f"stop={final_signal.stop_loss:.2f}",
                        self.console, style="dim",
                    )
                    continue

                # Submit with retry
                result = None
                for attempt in range(1, 4):
                    try:
                        result = executor.submit_order(final_signal)
                        break
                    except Exception as exc:
                        logger.warning(
                            "Order submission attempt %d/3 failed for %s: %s",
                            attempt, final_signal.symbol, exc,
                        )
                        if attempt < 3:
                            time.sleep(2 ** attempt)

                if result is None:
                    self.alert_manager.alert_order_error(
                        final_signal.symbol,
                        "All 3 submission attempts failed",
                    )
                    continue

                from broker.order_executor import OrderStatus
                if result.status == OrderStatus.REJECTED:
                    self.alert_manager.alert_order_error(
                        final_signal.symbol,
                        result.error_message or "REJECTED by broker",
                    )
                    self.dashboard.update(
                        event=f"Order REJECTED {final_signal.symbol}: {result.error_message}"
                    )
                else:
                    self.dashboard.update(
                        event=(
                            f"Order submitted: BUY {final_signal.symbol}  "
                            f"regime={regime_state.label}"
                        )
                    )

            # ---- 8. Update trailing stops ------------------------------------
            if not self.dry_run:
                self._update_trailing_stops(executor, regime_state)

            # ---- 9. Circuit breaker check ------------------------------------
            trading_state = self.risk_manager.get_trading_state()
            if trading_state == TradingState.HALTED:
                dd = self.risk_manager.get_drawdown_state()
                self.alert_manager.alert_drawdown_halt(
                    equity       = dd.current_equity,
                    drawdown_pct = dd.dd_from_peak,
                )
                self.trade_logger.log_risk_event(
                    event_type  = "HALT",
                    description = f"Trading halted  dd={dd.dd_from_peak:.2%}",
                    severity    = "CRITICAL",
                )
                self.dashboard.update(
                    event=f"[red]HALT: peak DD {dd.dd_from_peak:.2%}[/red]"
                )

            # ---- 10. Dashboard refresh & position update --------------------
            try:
                port_snap = self.position_tracker.update(self._last_regime)  # REST refresh
                from core.signal_generator import PortfolioSignal
                from monitoring.dashboard import SystemStatus
                portfolio_signal = PortfolioSignal(
                    timestamp       = pd.Timestamp.now(),
                    regime          = regime_state.label,
                    confidence      = confidence,
                    is_stable       = is_stable,
                    target_weights  = {},
                    delta_weights   = {},
                    leverage        = 1.0,
                    trading_allowed = trading_state != TradingState.HALTED,
                )
                _sys_status = SystemStatus(
                    data_ok          = True,
                    api_ok           = True,
                    api_latency_ms   = 0.0,
                    hmm_last_trained = self.hmm_engine._training_date.to_pydatetime()
                                       if getattr(self.hmm_engine, "_training_date", None) is not None else None,
                    mode             = "PAPER" if self.config.get("broker", {}).get("paper_trading", True) else "LIVE",
                )
                try:
                    _clock_open = self.client.get_clock().is_open
                except Exception:
                    _clock_open = None
                _next_bar = _next_bar_close_utc(timeframe)
                self.dashboard.update(
                    snapshot       = port_snap,
                    signal         = portfolio_signal,
                    drawdown_state = self.risk_manager.get_drawdown_state(),
                    system_status  = _sys_status,
                    market_open    = _clock_open,
                    next_bar_dt    = _next_bar,
                    next_broker_dt = _next_bar,
                    next_hmm_dt    = self._last_retrain + dt.timedelta(days=7),
                    timeframe      = timeframe,
                    event          = (
                        f"Bar {self._bar_count}: {regime_state.label}  "
                        f"p={confidence:.1%}  "
                        f"{'STABLE' if is_stable else 'PENDING'}"
                    ),
                )
            except Exception as exc:
                logger.warning("Dashboard update failed: %s", exc)

            # ---- 11. Weekly HMM retrain check --------------------------------
            days_since = (dt.datetime.now(dt.timezone.utc) - self._last_retrain).days
            if days_since >= 7:
                _print("Weekly HMM retrain ...", self.console, style="dim")
                try:
                    hmm_cfg = self.config.get("hmm", {})
                    self.hmm_engine = _train_hmm(
                        self.client, symbols, hmm_cfg, console=self.console
                    )
                    self.strategy = _rebuild_strategy(self.config, self.hmm_engine)
                    self._last_retrain = dt.datetime.now(dt.timezone.utc)
                    self.trade_logger.info(
                        "Weekly HMM retrain complete",
                        n_states=self.hmm_engine._n_states,
                    )
                    self.alert_manager.alert_regime_change(
                        previous_regime = self._last_regime,
                        new_regime      = self._last_regime,
                        confidence      = confidence,
                    )
                except Exception as exc:
                    logger.error("Weekly retrain failed: %s -- keeping current model", exc)
                    self.trade_logger.log_error(exc, context="weekly_retrain")

            self._bar_count += 1

    # ======================================================================= #
    # Shutdown                                                                 #
    # ======================================================================= #

    def shutdown(self) -> None:
        """
        Graceful shutdown.
        Stops WebSocket feeds, saves state snapshot, prints session summary.
        Positions are NOT closed -- stops remain in place.
        """
        self._running = False
        _print("\n\nShutting down ...", self.console)

        # Stop streams
        try:
            if self.position_tracker:
                self.position_tracker.stop_stream()
        except Exception:
            pass

        try:
            if self.dashboard:
                self.dashboard.stop()
        except Exception:
            pass

        # Save state snapshot
        try:
            equity_now = (
                self.client.get_portfolio_value() if self.client else self._equity_at_start
            )
        except Exception:
            equity_now = self._equity_at_start

        _save_snapshot({
            "session_start":    self._session_start.isoformat(),
            "session_end":      dt.datetime.now(dt.timezone.utc).isoformat(),
            "regime":           self._last_regime,
            "bar_count":        self._bar_count,
            "trade_count":      self._trade_count,
            "equity_at_start":  self._equity_at_start,
            "equity_at_end":    equity_now,
            "model_path":       str(_MODEL_PATH),
        })

        # Disconnect broker
        try:
            if self.client:
                self.client.disconnect()
        except Exception:
            pass

        # Print summary
        _print_session_summary(
            session_start = self._session_start,
            equity_start  = self._equity_at_start,
            equity_end    = equity_now,
            bar_count     = self._bar_count,
            trade_count   = self._trade_count,
            console       = self.console,
        )

        if self.trade_logger:
            self.trade_logger.info(
                "System offline",
                bars=self._bar_count,
                trades=self._trade_count,
            )

    # ======================================================================= #
    # Helpers                                                                  #
    # ======================================================================= #

    def _update_trailing_stops(self, executor, regime_state) -> None:
        """
        Tighten stops on existing positions based on current regime.
        In high-vol regimes stops are tightened more aggressively.
        """
        try:
            snap = self.position_tracker.get_last_snapshot()
            if not snap:
                return
            for pos in snap.positions:
                if pos.stop_level is None:
                    continue
                # ATR-based trail: tighten stop if price moved favorably
                current_price = pos.current_price
                current_stop  = pos.stop_level
                # Only tighten -- never widen
                if current_price > pos.avg_entry_price:
                    # Trail at 2× ATR below current price (approximate)
                    trail_pct = 0.03 if "BULL" in regime_state.label else 0.05
                    new_stop = current_price * (1.0 - trail_pct)
                    if new_stop > current_stop:
                        executor.modify_stop(pos.symbol, new_stop)
        except Exception as exc:
            logger.debug("Trailing stop update: %s", exc)


def _make_fallback_regime(label: str):
    """Return a minimal RegimeState with the given label when HMM update fails."""
    from core.hmm_engine import RegimeState
    return RegimeState(
        label             = label,
        state_id          = 0,
        probability       = 0.5,
        state_probabilities = np.array([0.5]),
        timestamp         = None,
        is_confirmed      = False,
        consecutive_bars  = 0,
    )


def _rebuild_strategy(config: Dict, hmm_engine) -> "StrategyOrchestrator":
    from core.regime_strategies import StrategyOrchestrator
    hmm_cfg = config.get("hmm", {})
    return StrategyOrchestrator(
        config           = config,
        regime_infos     = hmm_engine.get_all_regime_info(),
        min_confidence   = hmm_cfg.get("min_confidence", 0.55),
        rebalance_threshold = config.get("strategy", {}).get("rebalance_threshold", 0.10),
    )


# ── Run modes ──────────────────────────────────────────────────────────────────

def run_backtest(config: Dict, args: argparse.Namespace) -> None:
    """
    Execute the full walk-forward backtest pipeline.
    """
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceAnalyzer
    from backtest.stress_test import StressTester

    console = _console()

    symbols: List[str] = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None),
        symbols_arg=getattr(args, "symbols", None),
    )

    bt_cfg = config.get("backtest", {})
    start_date: str = args.start or "2018-01-01"
    end_date: str   = args.end   or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir      = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_capital = float(bt_cfg.get("initial_capital", 100_000))
    slippage_pct    = float(bt_cfg.get("slippage_pct",    0.0005))
    train_window    = int(bt_cfg.get("train_window",      252))
    test_window     = int(bt_cfg.get("test_window",       126))
    step_size       = int(bt_cfg.get("step_size",         126))
    risk_free_rate  = float(bt_cfg.get("risk_free_rate",  0.045))

    # Header — plain-English context about the run
    _years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    _train_mo = round(train_window / 21)
    _test_mo  = round(test_window  / 21)
    _sym_preview = ", ".join(symbols[:6])
    _sym_extra   = f" … (+{len(symbols) - 6} more)" if len(symbols) > 6 else ""

    _print(f"\n[bold cyan]Regime Trader — Walk-Forward Backtest[/bold cyan]", console)
    _print(
        f"  Symbols  : {_sym_preview}{_sym_extra}\n"
        f"  Period   : {start_date}  →  {end_date}  ({_years:.1f} years)\n"
        f"  Capital  : ${initial_capital:,.0f}\n"
        f"  Method   : Walk-forward — train {_train_mo}m in-sample, "
        f"test next {_test_mo}m out-of-sample, step forward, repeat\n"
        f"  Note     : each fold's model is never trained on its own test data",
        console,
    )

    api_key    = os.environ.get("ALPACA_API_KEY",    "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        _print(
            "[red]ERROR: Alpaca credentials not found.[/red]\n"
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or populate config/credentials.yaml.",
            console,
        )
        sys.exit(1)

    _print("\nFetching historical data from Alpaca ...", console, style="dim")
    try:
        prices = _fetch_prices(symbols, start_date, end_date, api_key, secret_key)
    except Exception as exc:
        _print(f"[red]Data fetch failed:[/red] {exc}", console)
        sys.exit(1)

    _print(
        f"  Downloaded {len(prices)} bars  "
        f"({prices.index[0].date()} -> {prices.index[-1].date()})  "
        f"symbols: {list(prices.columns)}",
        console, style="dim",
    )

    if len(prices) < train_window + test_window + 50:
        _print(
            f"[red]Insufficient data:[/red] only {len(prices)} bars. "
            f"Need >= {train_window + test_window + 50}.",
            console,
        )
        sys.exit(1)

    hmm_cfg_raw = config.get("hmm", {})
    hmm_config = {
        "n_candidates":      hmm_cfg_raw.get("n_candidates", [3, 4, 5]),
        "n_init":            hmm_cfg_raw.get("n_init", 10),
        "min_train_bars":    hmm_cfg_raw.get("min_train_bars", 120),
        "stability_bars":    hmm_cfg_raw.get("stability_bars", 3),
        "flicker_window":    hmm_cfg_raw.get("flicker_window", 20),
        "flicker_threshold": hmm_cfg_raw.get("flicker_threshold", 4),
        "min_confidence":    hmm_cfg_raw.get("min_confidence", 0.55),
    }
    strategy_config = {"strategy": config.get("strategy", {})}
    _strat = strategy_config["strategy"]

    _print(
        f"\n  [dim]HMM[/dim]\n"
        f"  [dim]  States (candidates) : {hmm_config['n_candidates']}[/dim]\n"
        f"  [dim]  Features            : {hmm_cfg_raw.get('features', ['log_return', 'realized_vol_5d', 'realized_vol_21d'])}[/dim]\n"
        f"  [dim]  Min confidence      : {hmm_config['min_confidence']}[/dim]\n"
        f"  [dim]  Stability bars      : {hmm_config['stability_bars']}  "
        f"Flicker window: {hmm_config['flicker_window']}  "
        f"Flicker threshold: {hmm_config['flicker_threshold']}[/dim]\n"
        f"\n  [dim]Strategy[/dim]\n"
        f"  [dim]  Allocation — low-vol: {_strat.get('low_vol_allocation', 0.95):.0%}  "
        f"mid-vol (trend): {_strat.get('mid_vol_allocation_trend', 0.95):.0%}  "
        f"mid-vol (no trend): {_strat.get('mid_vol_allocation_no_trend', 0.70):.0%}  "
        f"high-vol: {_strat.get('high_vol_allocation', 0.70):.0%}[/dim]\n"
        f"  [dim]  Leverage cap        : {_strat.get('low_vol_leverage', 1.25):.2f}x  "
        f"Rebalance threshold: {_strat.get('rebalance_threshold', 0.15):.0%}  "
        f"Trend lookback: {_strat.get('trend_lookback', 50)} bars[/dim]",
        console,
    )

    def _make_bt_progress(n_syms: int) -> object:
        """Return a fold-progress callback for bt.run()."""
        from rich.text import Text

        _W = 20
        _ic = initial_capital  # capture for equity colour comparison

        def _build_line(fold_id: int, n_total: int, phase: str, info: dict) -> Text:
            if phase == "training":
                pct = int(fold_id / n_total * 100)
            else:
                pct = int((fold_id + 1) / n_total * 100)

            filled = pct * _W // 100
            bar_text = Text()
            bar_text.append("#" * filled, style="cyan")
            bar_text.append("-" * (_W - filled), style="dim white")

            line = Text()
            line.append("  [")
            line.append_text(bar_text)
            line.append("] ")
            line.append(f"{pct:3d}%", style="bold white")
            line.append(f"  Fold {fold_id+1}/{n_total}  ")

            if phase == "training":
                line.append("Training ...", style="yellow")
                line.append(
                    f"  OOS {info['oos_start']} -> {info['oos_end']}", style="dim"
                )
            else:
                line.append("OK ", style="bold green")
                line.append(
                    f"{info['oos_start']} -> {info['oos_end']}", style="green"
                )
                line.append(f"  {info['n_states']} states", style="dim cyan")
                line.append(f"  {info['fold_trades']} tr", style="dim")
                eq_style = (
                    "bold green" if info["equity"] >= _ic else "bold red"
                )
                line.append(f"  ${info['equity']:>10,.0f}", style=eq_style)

            return line

        def _cb(fold_id: int, n_total: int, phase: str, info: dict) -> None:
            if console is None:
                return
            line = _build_line(fold_id, n_total, phase, info)
            end = "\r" if phase == "training" else "\n"
            console.print(line, end=end, highlight=False, no_wrap=True, overflow="crop")

        return _cb

    _print("\nRunning walk-forward backtest ...", console, style="dim")

    # Silence noisy-but-normal log output during the backtest run.
    # hmmlearn emits convergence warnings on every EM candidate that hits the
    # iteration limit — expected behaviour, not an error.
    # core.hmm_engine regime-change lines are INFO in live mode (dashboard handles
    # visibility); suppress them here so the backtest output stays readable.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    logging.getLogger("core.hmm_engine").setLevel(logging.WARNING)

    bt = WalkForwardBacktester(
        symbols         = list(prices.columns),
        initial_capital = initial_capital,
        train_window    = train_window,
        test_window     = test_window,
        step_size       = step_size,
        slippage_pct    = slippage_pct,
        risk_free_rate  = risk_free_rate,
        zscore_window       = int(config.get("backtest", {}).get("zscore_window",      60)),
        sma_long            = int(config.get("backtest", {}).get("sma_long",           200)),
        sma_trend           = int(config.get("backtest", {}).get("sma_trend",           50)),
        volume_norm_window  = int(config.get("backtest", {}).get("volume_norm_window",  50)),
    )

    try:
        result = bt.run(
            prices,
            hmm_config=hmm_config,
            strategy_config=strategy_config,
            progress_callback=_make_bt_progress(len(list(prices.columns))),
        )
    except Exception as exc:
        _print(f"[red]Backtest failed:[/red] {exc}", console)
        sys.exit(1)

    _print(
        f"  Folds completed : {result.metadata['n_folds']}\n"
        f"  OOS bars        : {len(result.combined_equity)}\n"
        f"  Total trades    : {result.metadata['total_trades']}\n"
        f"  Final equity    : ${result.final_equity:,.2f}",
        console, style="dim",
    )

    # ── Per-fold state count ──────────────────────────────────────────────────
    _fold_states = [w.n_hmm_states for w in result.windows]
    _state_counts = {}
    for n in _fold_states:
        _state_counts[n] = _state_counts.get(n, 0) + 1
    _state_summary = "  ".join(
        f"{n}-state × {c}" for n, c in sorted(_state_counts.items())
    )
    _print(
        f"  HMM states used : {_state_summary}  "
        f"(per fold: {', '.join(str(n) for n in _fold_states)})",
        console, style="dim",
    )

    # ── Regime bar counts ─────────────────────────────────────────────────────
    if len(result.combined_regimes) > 0:
        _regime_counts = result.combined_regimes.value_counts().sort_index()
        _total_bars = len(result.combined_regimes)
        _label_order = [
            "CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR",
            "NEUTRAL",
            "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA",
        ]
        # sort by the canonical bear→bull order; unknown labels go at end
        _sorted = sorted(
            _regime_counts.items(),
            key=lambda kv: (_label_order.index(kv[0]) if kv[0] in _label_order else 99),
        )
        _regime_lines = "".join(
            f"\n    {lbl:<14} {bars:>5} bars  ({bars / _total_bars:>5.1%})"
            for lbl, bars in _sorted
        )
        _print(f"  Regime counts  :{_regime_lines}", console, style="dim")

    _print("\n", console)
    pa = PerformanceAnalyzer(
        risk_free_rate=risk_free_rate,
        trading_days_per_year=252,
    )
    bm_sym    = symbols[0]
    bm_prices = prices[bm_sym] if getattr(args, "compare", False) else None
    report    = pa.analyze(result, benchmark_prices=bm_prices)
    pa.generate_report(report, print_to_console=True)

    if getattr(args, "compare", False) and bm_prices is not None:
        _print("\nComputing benchmark strategies ...", console, style="dim")

        multi = len(symbols) > 1
        if multi:
            # Build a price DataFrame aligned to the strategy equity index,
            # dropping symbols that are missing from the fetched data.
            avail_syms = [s for s in symbols if s in prices]
            price_df = pd.DataFrame(
                {s: prices[s] for s in avail_syms}
            ).reindex(result.combined_equity.index).ffill().dropna(how="all")
            bnh_equity      = pa.compute_benchmark_bnh_multi(price_df, initial_capital)
            sma_equity      = pa.compute_benchmark_sma_multi(price_df, 200, initial_capital)
            ema_cross_equity = pa.compute_benchmark_ema_cross_multi(price_df, 9, 45, initial_capital)
            rand_mean, _ = pa.compute_random_allocation_benchmark_multi(
                price_df, allocations=[0.60, 0.95], n_seeds=100,
                initial_capital=initial_capital,
            )
        else:
            bnh_equity       = pa.compute_benchmark_bnh(bm_prices, initial_capital)
            sma_equity       = pa.compute_benchmark_sma(bm_prices, 200, initial_capital)
            ema_cross_equity = pa.compute_benchmark_ema_cross(bm_prices, 9, 45, initial_capital)
            underlying_ret     = bm_prices.pct_change().dropna()
            underlying_aligned = underlying_ret.reindex(
                result.combined_returns.index
            ).dropna()
            rand_mean, _ = pa.compute_random_allocation_benchmark(
                underlying_aligned, allocations=[0.60, 0.95], n_seeds=100,
                initial_capital=initial_capital,
            )

        eq_idx       = result.combined_equity.index
        bnh_rpt      = pa.analyze_equity_curve(
            bnh_equity.reindex(eq_idx).ffill().dropna()
        )
        sma_rpt      = pa.analyze_equity_curve(
            sma_equity.reindex(eq_idx).ffill().dropna()
        )
        ema_cross_rpt = pa.analyze_equity_curve(
            ema_cross_equity.reindex(eq_idx).ffill().dropna()
        )
        rand_rpt_metrics = {
            "total_return": float(rand_mean.iloc[-1] / initial_capital - 1),
            "cagr":         pa.compute_cagr(rand_mean),
            "sharpe_ratio": pa.compute_sharpe(rand_mean.pct_change().dropna()),
            "sortino_ratio":pa.compute_sortino(rand_mean.pct_change().dropna()),
            "max_drawdown": pa.compute_max_drawdown(rand_mean)[0],
            "calmar_ratio": pa.compute_calmar(rand_mean),
            "annualised_vol":float(rand_mean.pct_change().dropna().std() * (252**0.5)),
            "win_rate":     float((rand_mean.pct_change().dropna() > 0).mean()),
        }
        _print_run_config(symbols, start_date, end_date, config, console)
        _print_comparison_table(
            report, bnh_rpt, sma_rpt, ema_cross_rpt, rand_rpt_metrics, console,
            n_symbols=len(symbols),
        )

    if getattr(args, "stress_test", False):
        _print("\nRunning stress scenarios ...", console, style="dim")
        stress = StressTester(bt, pa)
        stress_results = stress.run_stress_scenarios(
            prices, hmm_config=hmm_config, strategy_config=strategy_config,
        )
        if stress_results:
            tbl = stress.summary_table()
            _print("\n[bold]Stress Test Summary[/bold]", console)
            if console:
                try:
                    from rich.table import Table
                    from rich import box as rbox
                    rt = Table(title="Stress Test Results", box=rbox.ROUNDED,
                               header_style="bold red")
                    for col in tbl.reset_index().columns:
                        rt.add_column(col, justify="right")
                    for _, row in tbl.reset_index().iterrows():
                        rt.add_row(*[str(v) for v in row])
                    console.print(rt)
                except ImportError:
                    print(tbl.to_string())
            else:
                print(tbl.to_string())
            tbl.to_csv(output_dir / "stress_test_summary.csv")

    _print(f"\nSaving results to {output_dir}/", console, style="dim")
    result.combined_equity.rename("equity").to_csv(
        output_dir / "equity_curve.csv", header=True
    )
    result.combined_regimes.rename("regime").to_csv(
        output_dir / "regime_history.csv", header=True
    )
    all_trades = [t for w in result.windows for t in w.trades]
    if all_trades:
        pd.DataFrame(all_trades).to_csv(output_dir / "trade_log.csv", index=False)

    metrics = {
        "total_return":  report.total_return,
        "cagr":          report.cagr,
        "sharpe":        report.sharpe_ratio,
        "sortino":       report.sortino_ratio,
        "max_drawdown":  report.max_drawdown,
        "max_dd_days":   report.max_drawdown_duration_days,
        "calmar":        report.calmar_ratio,
        "win_rate":      report.win_rate,
        "profit_factor": report.profit_factor,
        "total_trades":  report.total_trades,
        "final_equity":  result.final_equity,
        "n_folds":       result.metadata["n_folds"],
        "symbols":       ",".join(result.metadata.get("symbols", symbols)),
        "start":         start_date,
        "end":           end_date,
    }
    pd.Series(metrics).to_csv(output_dir / "performance_summary.csv", header=False)

    _final_equity = result.final_equity
    _print(
        f"\n[green]Backtest complete.[/green]  "
        f"Total return: [bold]{report.total_return:+.2%}[/bold]  "
        f"Sharpe: [bold]{report.sharpe_ratio:.3f}[/bold]  "
        f"MaxDD: [bold]{report.max_drawdown:.2%}[/bold]",
        console,
    )

    # Plain-English interpretation of the key numbers
    _sharpe_grade = (
        "strong risk-adjusted return"        if report.sharpe_ratio >= 1.0  else
        "acceptable risk-adjusted return"    if report.sharpe_ratio >= 0.5  else
        "below benchmark — marginal edge"    if report.sharpe_ratio >= 0.0  else
        "negative — strategy lost money on a risk-adjusted basis"
    )
    _dd_note = (
        "within normal range for an equity strategy"  if abs(report.max_drawdown) <= 0.20 else
        "significant — account lost over 20% at its worst"
    )
    _cagr_str  = f"{report.cagr:+.1%}" if report.cagr is not None else "N/A"
    _years_str = f"{_years:.1f}"
    _print(
        f"\n  [dim]Interpretation[/dim]\n"
        f"  [dim]  ${initial_capital:,.0f}  →  ${_final_equity:,.0f}  "
        f"over {_years_str} years   (CAGR {_cagr_str} / year)[/dim]\n"
        f"  [dim]  Sharpe {report.sharpe_ratio:.3f} — {_sharpe_grade}[/dim]\n"
        f"  [dim]  Max Drawdown {report.max_drawdown:.1%} — {_dd_note}[/dim]\n"
        f"  [dim]  {result.metadata['total_trades']} trades across "
        f"{result.metadata['n_folds']} out-of-sample folds "
        f"(~{int(result.metadata['total_trades'] / max(_years, 1))} trades/year)[/dim]\n"
        f"  [dim]  All results are out-of-sample — no lookahead bias[/dim]",
        console,
    )


def run_trading(config: Dict, dry_run: bool = False) -> None:
    """
    Full live / paper trading loop.

    Handles SIGINT / SIGTERM for graceful shutdown.
    All unhandled exceptions are logged, state is saved, and an alert is fired.
    """
    session = TradingSession(config, dry_run=dry_run)

    # Register shutdown signal handlers
    def _handle_signal(signum, frame):
        logger.info("Signal %d received -- initiating shutdown", signum)
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        session.startup()
    except Exception as exc:
        _print(f"[red]Startup failed:[/red] {exc}", _console())
        logger.error("Startup failed: %s\n%s", exc, traceback.format_exc())
        try:
            session.shutdown()
        except Exception:
            pass
        sys.exit(1)

    try:
        session.run_loop()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.critical(
            "Unhandled exception in trading loop: %s\n%s",
            exc, traceback.format_exc(),
        )
        try:
            if session.trade_logger:
                session.trade_logger.log_error(exc, context="main_loop_unhandled")
            if session.alert_manager:
                session.alert_manager.alert(
                    title   = "CRITICAL: Unhandled exception in trading loop",
                    message = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    level   = "CRITICAL",
                )
        except Exception:
            pass
    finally:
        session.shutdown()


def run_train_only(config: Dict, args: Optional[argparse.Namespace] = None) -> None:
    """Train HMM on latest data and exit without entering the trading loop."""
    console = _console()
    broker_cfg = config.get("broker", {})
    hmm_cfg    = config.get("hmm",    {})
    symbols    = _resolve_symbols(
        config,
        asset_group=getattr(args, "asset_group", None) if args else None,
        symbols_arg=getattr(args, "symbols", None)     if args else None,
    )
    paper      = broker_cfg.get("paper_trading", True)

    _print("[bold cyan]Regime Trader - Train Only[/bold cyan]", console)

    from broker.alpaca_client import AlpacaClient
    client = AlpacaClient(paper=paper)
    client.connect_with_retry(max_attempts=3, base_delay=2.0)

    engine = _train_hmm(client, symbols, hmm_cfg, console=console, n_bars=756)
    _print(
        f"\n[green]Training complete.[/green]  "
        f"states={engine._n_states}  BIC={engine._training_bic:.2f}  "
        f"saved -> {_MODEL_PATH}",
        console,
    )

    client.disconnect()


def run_stress_test(config: Dict, args: argparse.Namespace) -> None:
    """Standalone stress-test entry point (delegates to run_backtest --stress-test)."""
    args.stress_test = True
    if not hasattr(args, "compare"):
        args.compare = False
    run_backtest(config, args)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-trader",
        description="HMM-based volatility regime allocation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py trade\n"
            "  python main.py trade --dry-run\n"
            "  python main.py trade --train-only\n"
            "  python main.py trade --asset-group crypto\n"
            "  python main.py backtest --asset-group stocks --start 2020-01-01 --compare\n"
            "  python main.py backtest --asset-group crypto --start 2020-01-01 --compare\n"
            "  python main.py backtest --asset-group indices --start 2020-01-01 --compare\n"
            "  python main.py backtest --symbols SPY,QQQ --start 2020-01-01\n"
            "  python main.py stress   --asset-group indices --start 2019-01-01\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── trade ─────────────────────────────────────────────────────────────────
    trade_p = sub.add_parser("trade", help="Run live or paper trading loop")
    trade_p.add_argument("--config",       default="config/settings.yaml")
    trade_p.add_argument("--paper",        action="store_true", default=None,
                          help="Force paper-trading mode")
    trade_p.add_argument("--dry-run",      action="store_true", dest="dry_run",
                          help="Full pipeline but place no real orders")
    trade_p.add_argument("--train-only",   action="store_true", dest="train_only",
                          help="Train HMM on latest data and exit")
    trade_p.add_argument("--asset-group",  default=None, dest="asset_group",
                          help="Asset group from config (stocks | crypto | indices)")
    trade_p.add_argument("--symbols",      default=None,
                          help="Comma-separated symbols — overrides asset-group")

    # ── backtest ──────────────────────────────────────────────────────────────
    bt_p = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt_p.add_argument("--config",      default="config/settings.yaml")
    bt_p.add_argument("--output",      default="results/",
                       help="Directory for output CSVs (default: results/)")
    bt_p.add_argument("--asset-group", default=None, dest="asset_group",
                       help="Asset group from config (stocks | crypto | indices)")
    bt_p.add_argument("--symbols",     default=None,
                       help="Comma-separated symbols — overrides asset-group")
    bt_p.add_argument("--start",       default=None,
                       help="Start date ISO-8601 (e.g. 2019-01-01)")
    bt_p.add_argument("--end",         default=None,
                       help="End date ISO-8601 (default: today)")
    bt_p.add_argument("--compare",     action="store_true",
                       help="Add benchmark comparison table")
    bt_p.add_argument("--stress-test", action="store_true", dest="stress_test",
                       help="Run stress scenarios after the backtest")

    # ── stress ────────────────────────────────────────────────────────────────
    stress_p = sub.add_parser("stress", help="Run stress-test scenario suite")
    stress_p.add_argument("--config",  default="config/settings.yaml")
    stress_p.add_argument("--output",  default="results/")
    stress_p.add_argument("--symbols", default=None)
    stress_p.add_argument("--start",   default=None)
    stress_p.add_argument("--end",     default=None)

    return parser


def main() -> None:
    """Parse CLI arguments, load config / credentials, and dispatch."""
    load_dotenv()

    parser = build_parser()
    args   = parser.parse_args()

    config = load_config(args.config)
    load_credentials()

    if args.command == "trade":
        if getattr(args, "paper", None):
            config["broker"]["paper_trading"] = True

        if getattr(args, "train_only", False):
            run_train_only(config, args)
            return

        run_trading(config, dry_run=getattr(args, "dry_run", False))

    elif args.command == "backtest":
        run_backtest(config, args)

    elif args.command == "stress":
        run_stress_test(config, args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
