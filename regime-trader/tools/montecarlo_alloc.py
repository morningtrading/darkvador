"""
tools/montecarlo_alloc.py — Monte Carlo sensitivity analysis for allocation parameters.

Purpose
-------
Tests whether the default allocation values (mid_vol_no_trend=0.60, high_vol=0.60)
are robust — or whether small changes collapse performance.

For each draw, two parameters are randomly sampled:
    strategy.mid_vol_allocation_no_trend   (mid-vol regime, price < EMA50)
    strategy.high_vol_allocation           (high-vol / defensive regime)

The constraint  high_vol <= mid_vol_no_trend  is always enforced.

Optionally also varies:
    strategy.mid_vol_allocation_trend      (mid-vol regime, price > EMA50)

All other settings (HMM, symbols, risk, backtest windows) are unchanged.

Usage
-----
    py -3.12 tools/montecarlo_alloc.py
    py -3.12 tools/montecarlo_alloc.py --n-draws 200 --dist normal
    py -3.12 tools/montecarlo_alloc.py --dry-run
    py -3.12 tools/montecarlo_alloc.py --vary-trend --n-draws 300 --seed 99
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for box chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Import shared primitives from rolling_wfo (same directory) ────────────────
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

from rolling_wfo import (  # noqa: E402
    ROOT, SETTINGS, PYTHON,
    _set_nested, _apply_variant,
    _run_streaming, _parse_metrics,
    _strip_ansi, _fmt_elapsed, _dump_tail,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_SYMBOLS = "SPY,QQQ,AAPL,MSFT,AMZN,GOOGL,NVDA,META,TSLA,AMD"
_DEFAULT_START   = "2020-01-01"
_DEFAULT_END     = "2024-12-31"
_DEFAULT_N       = 100
_DEFAULT_SEED    = 42

# Nominal values (defaults in settings.yaml)
_NOMINAL = {
    "mid_vol_no_trend": 0.60,
    "high_vol":         0.60,
    "mid_vol_trend":    0.95,
}

# Sampling ranges
_UNIFORM_LO  = 0.45
_UNIFORM_HI  = 0.75
_NORMAL_MU   = 0.60
_NORMAL_SIGMA= 0.05
_TREND_LO    = 0.80
_TREND_HI    = 1.00

# YAML key paths
_KEY_MID_NO_TREND = "strategy.mid_vol_allocation_no_trend"
_KEY_HIGH_VOL     = "strategy.high_vol_allocation"
_KEY_MID_TREND    = "strategy.mid_vol_allocation_trend"

RESULTS_DIR = ROOT / "results"


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_uniform(lo: float, hi: float, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)


def _sample_truncated_normal(mu: float, sigma: float, lo: float, hi: float,
                              n: int, rng: np.random.Generator) -> np.ndarray:
    """Truncated normal via rejection sampling."""
    out: List[float] = []
    while len(out) < n:
        batch = rng.normal(mu, sigma, size=(n - len(out)) * 4)
        valid = batch[(batch >= lo) & (batch <= hi)]
        out.extend(valid.tolist())
    return np.array(out[:n])


def _generate_draws(
    n: int,
    dist: str,
    vary_trend: bool,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """
    Generate n parameter dicts, enforcing high_vol <= mid_vol_no_trend.

    Each dict has keys: mid_vol_no_trend, high_vol [, mid_vol_trend].
    """
    lo, hi = _UNIFORM_LO, _UNIFORM_HI
    draws: List[Dict[str, float]] = []
    attempts = 0
    max_attempts = n * 30

    while len(draws) < n and attempts < max_attempts:
        batch = (n - len(draws)) * 4
        if dist == "normal":
            mv = _sample_truncated_normal(_NORMAL_MU, _NORMAL_SIGMA, lo, hi, batch, rng)
            hv = _sample_truncated_normal(_NORMAL_MU, _NORMAL_SIGMA, lo, hi, batch, rng)
        else:
            mv = _sample_uniform(lo, hi, batch, rng)
            hv = _sample_uniform(lo, hi, batch, rng)

        mask = hv <= mv
        for m, h in zip(mv[mask], hv[mask]):
            if len(draws) >= n:
                break
            d: Dict[str, float] = {
                "mid_vol_no_trend": round(float(m), 4),
                "high_vol":         round(float(h), 4),
            }
            if vary_trend:
                if dist == "normal":
                    t = _sample_truncated_normal(0.95, 0.03, _TREND_LO, _TREND_HI, 1, rng)[0]
                else:
                    t = float(rng.uniform(_TREND_LO, _TREND_HI))
                d["mid_vol_trend"] = round(t, 4)
            draws.append(d)
        attempts += batch

    if len(draws) < n:
        raise RuntimeError(
            f"Could not generate {n} valid draws after {attempts} attempts. "
            "Try a larger range or different distribution."
        )
    return draws


# ── Execution ─────────────────────────────────────────────────────────────────

def _run_one_draw(
    draw_idx: int,
    n_total: int,
    params: Dict[str, float],
    base_cfg: dict,
    symbols: str,
    start: str,
    end: str,
) -> Tuple[Dict[str, str], float]:
    """
    Patch settings.yaml with params, run backtest, return (metrics_dict, elapsed_s).
    """
    overrides: Dict[str, Any] = {
        _KEY_MID_NO_TREND: params["mid_vol_no_trend"],
        _KEY_HIGH_VOL:     params["high_vol"],
    }
    if "mid_vol_trend" in params:
        overrides[_KEY_MID_TREND] = params["mid_vol_trend"]

    cfg = _apply_variant(base_cfg, overrides)
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    cmd = [
        PYTHON, str(ROOT / "main.py"), "backtest",
        "--symbols", symbols,
        "--start", start,
        "--end",   end,
    ]

    t0 = time.monotonic()
    output = _run_streaming(cmd, indent=f"  [{draw_idx}/{n_total}] ")
    elapsed = time.monotonic() - t0

    return _parse_metrics(output), elapsed, output


# ── Display helpers ───────────────────────────────────────────────────────────

_COL_TREND  = 12
_COL_PARAM  = 10
_COL_METRIC = 9
_COL_TIME   = 8
_COL_ETA    = 9


def _progress_header(vary_trend: bool) -> None:
    params_hdr = f"  {'#':>4}  {'mid_no_trend':>12}  {'high_vol':>8}"
    if vary_trend:
        params_hdr += f"  {'mid_trend':>9}"
    params_hdr += (
        f"  {'Return':>9}  {'Sharpe':>7}  {'MaxDD':>9}"
        f"  {'Trades':>7}  {'Time':>7}  {'ETA':>9}"
    )
    sep = "  " + "-" * (len(params_hdr) - 2)
    print()
    print(params_hdr)
    print(sep)


def _progress_row(
    idx: int,
    n: int,
    params: Dict[str, float],
    metrics: Dict[str, str],
    elapsed: float,
    elapsed_list: List[float],
    vary_trend: bool,
) -> None:
    eta_s = statistics.mean(elapsed_list) * (n - idx) if elapsed_list else 0.0
    eta   = _fmt_elapsed(eta_s) if eta_s > 0 else "—"
    time_s = _fmt_elapsed(elapsed)

    row = (
        f"  {idx:>4}  {params['mid_vol_no_trend']:>12.4f}"
        f"  {params['high_vol']:>8.4f}"
    )
    if vary_trend:
        row += f"  {params.get('mid_vol_trend', 0.0):>9.4f}"

    row += (
        f"  {metrics['total_return']:>9}"
        f"  {metrics['sharpe']:>7}"
        f"  {metrics['max_dd']:>9}"
        f"  {metrics['trades']:>7}"
        f"  {time_s:>7}"
        f"  ~{eta:>8}"
    )
    print(row, flush=True)


# ── Analytics ─────────────────────────────────────────────────────────────────

def _to_float(s: str) -> Optional[float]:
    """Parse '+12.34%' or '1.234' to float. Returns None on ERR."""
    try:
        return float(s.replace("%", "").replace("+", ""))
    except (ValueError, AttributeError, TypeError):
        return None


def _compute_summary(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean, std, p5, p95 for each metric across valid draws."""
    out: Dict[str, Dict[str, float]] = {}
    for key in ("sharpe", "total_return", "max_dd"):
        vals = [_to_float(r[key]) for r in results if _to_float(r[key]) is not None]
        if not vals:
            continue
        arr = np.array(vals)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "p5":   float(np.percentile(arr, 5)),
            "p95":  float(np.percentile(arr, 95)),
            "n":    float(len(arr)),
        }
    return out


def _compute_correlations(results: List[Dict], vary_trend: bool) -> Dict[str, float]:
    """Pearson correlation of each sampled param with Sharpe."""
    sharpes = np.array([
        _to_float(r["sharpe"]) for r in results
        if _to_float(r["sharpe"]) is not None
    ])
    valid_idx = [
        i for i, r in enumerate(results)
        if _to_float(r["sharpe"]) is not None
    ]
    corrs: Dict[str, float] = {}
    for param in ["mid_vol_no_trend", "high_vol"] + (["mid_vol_trend"] if vary_trend else []):
        vals = np.array([results[i][param] for i in valid_idx])
        if len(vals) < 3:
            corrs[param] = float("nan")
            continue
        r_mat = np.corrcoef(vals, sharpes)
        corrs[param] = float(r_mat[0, 1])
    return corrs


def _compute_fragility(
    results: List[Dict],
    param: str,
    center: float,
    pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Sharpe std for draws where param is within ±pct of center.
    Tells you how variable performance is when the parameter is near its nominal value.
    """
    lo, hi = center * (1 - pct), center * (1 + pct)
    nearby = [
        r for r in results
        if lo <= r.get(param, -1) <= hi and _to_float(r["sharpe"]) is not None
    ]
    if len(nearby) < 3:
        return {"std": float("nan"), "n": len(nearby), "lo": lo, "hi": hi}
    sharpes = [_to_float(r["sharpe"]) for r in nearby]  # type: ignore[arg-type]
    return {
        "std": statistics.stdev(sharpes),  # type: ignore[arg-type]
        "n":   len(sharpes),
        "lo":  lo,
        "hi":  hi,
    }


def _build_heatmap(
    results: List[Dict],
    n_bins: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin mid_vol_no_trend × high_vol → mean Sharpe.

    Returns (mv_edges, hv_edges, grid) where grid[i,j] = mean Sharpe
    for results with mv in bin i and hv in bin j.  NaN = no draws.
    """
    mv_edges = np.linspace(_UNIFORM_LO, _UNIFORM_HI, n_bins + 1)
    hv_edges = np.linspace(_UNIFORM_LO, _UNIFORM_HI, n_bins + 1)
    counts   = np.zeros((n_bins, n_bins))
    totals   = np.zeros((n_bins, n_bins))

    for r in results:
        sh = _to_float(r["sharpe"])
        if sh is None:
            continue
        mv = r["mid_vol_no_trend"]
        hv = r["high_vol"]
        mi = min(int((mv - _UNIFORM_LO) / (_UNIFORM_HI - _UNIFORM_LO) * n_bins), n_bins - 1)
        hi_ = min(int((hv - _UNIFORM_LO) / (_UNIFORM_HI - _UNIFORM_LO) * n_bins), n_bins - 1)
        counts[mi, hi_] += 1
        totals[mi, hi_] += sh

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(counts > 0, totals / counts, np.nan)
    return mv_edges, hv_edges, grid


def _print_heatmap(mv_edges: np.ndarray, hv_edges: np.ndarray,
                   grid: np.ndarray) -> None:
    n = grid.shape[0]
    col_w = 7

    # Column headers (high_vol bin centres)
    hv_centres = [(hv_edges[j] + hv_edges[j + 1]) / 2 for j in range(n)]
    header = f"  {'mid\\high':>10}" + "".join(f"{c:>{col_w}.2f}" for c in hv_centres)
    print(header)
    print("  " + "-" * (10 + col_w * n))

    mv_centres = [(mv_edges[i] + mv_edges[i + 1]) / 2 for i in range(n)]
    for i in range(n):
        row_lbl = f"  {mv_centres[i]:>10.2f}"
        row_vals = ""
        for j in range(n):
            val = grid[i, j]
            if np.isnan(val):
                row_vals += f"{'--':>{col_w}}"
            else:
                row_vals += f"{val:>{col_w}.3f}"
        print(row_lbl + row_vals)

    print("  (rows = mid_vol_no_trend, cols = high_vol, values = mean Sharpe)")
    print("  (-- = no draws in that bin due to high_vol <= mid_vol constraint)")


def _print_summary(
    results: List[Dict],
    vary_trend: bool,
    n_requested: int,
) -> None:
    valid = sum(1 for r in results if _to_float(r["sharpe"]) is not None)
    errors = len(results) - valid

    print()
    print("=" * 68)
    print(f"  MONTE CARLO RESULTS  —  {valid}/{n_requested} valid draws"
          + (f"  ({errors} parse errors)" if errors else ""))
    print("=" * 68)

    # ── Summary statistics ─────────────────────────────────────────────────
    summary = _compute_summary(results)
    labels = {
        "sharpe":       "Sharpe",
        "total_return": "Total Return",
        "max_dd":       "Max Drawdown",
    }
    pct_metrics = {"total_return", "max_dd"}

    print(f"\n  {'Metric':<16} {'Mean':>9} {'Std':>9} {'P5':>9} {'P95':>9}")
    print("  " + "-" * 54)
    for key, label in labels.items():
        if key not in summary:
            continue
        s = summary[key]
        is_pct = key in pct_metrics
        if is_pct:
            mean_s = f"{s['mean']:+.2f}%"
            std_s  = f"{s['std']:.2f}%"
            p5_s   = f"{s['p5']:+.2f}%"
            p95_s  = f"{s['p95']:+.2f}%"
        else:
            mean_s = f"{s['mean']:.3f}"
            std_s  = f"{s['std']:.3f}"
            p5_s   = f"{s['p5']:.3f}"
            p95_s  = f"{s['p95']:.3f}"
        print(f"  {label:<16} {mean_s:>9} {std_s:>9} {p5_s:>9} {p95_s:>9}")

    # ── Sensitivity ────────────────────────────────────────────────────────
    corrs = _compute_correlations(results, vary_trend)
    print()
    print("  Sensitivity (Pearson correlation with Sharpe):")
    print("  " + "-" * 54)
    interp = {
        (0.7,  1.0): "strong positive",
        (0.4,  0.7): "moderate positive",
        (0.1,  0.4): "weak positive",
        (-0.1, 0.1): "negligible",
        (-0.4,-0.1): "weak negative",
        (-0.7,-0.4): "moderate negative",
        (-1.0,-0.7): "strong negative",
    }
    for param, r in corrs.items():
        if np.isnan(r):
            label_str = "insufficient data"
        else:
            label_str = next(
                (v for (lo, hi), v in interp.items() if lo <= r <= hi),
                "—"
            )
        print(f"    {param:<24} :  r={r:+.3f}   ({label_str})")

    # ── Fragility ──────────────────────────────────────────────────────────
    print()
    print("  Fragility (Sharpe std for draws within ±10% of nominal 0.60):")
    print("  " + "-" * 54)
    for param in ["mid_vol_no_trend", "high_vol"] + (["mid_vol_trend"] if vary_trend else []):
        nominal = _NOMINAL.get(param, 0.60)
        frag = _compute_fragility(results, param, nominal)
        if np.isnan(frag["std"]):
            frag_str = f"n={frag['n']} draws — too few"
        else:
            frag_str = f"Sharpe std={frag['std']:.3f}   (n={frag['n']} draws)"
        print(f"    {param:<24}  [{frag['lo']:.2f}, {frag['hi']:.2f}]:  {frag_str}")

    # ── Heatmap ────────────────────────────────────────────────────────────
    print()
    print("  Sharpe landscape (mean Sharpe per allocation bin):")
    print("  " + "-" * 54)
    mv_edges, hv_edges, grid = _build_heatmap(results)
    _print_heatmap(mv_edges, hv_edges, grid)


# ── CSV output ────────────────────────────────────────────────────────────────

def _open_csv(path: Path, vary_trend: bool):
    """Open CSV writer; return (file_handle, writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    fieldnames = ["draw", "mid_vol_no_trend", "high_vol"]
    if vary_trend:
        fieldnames.append("mid_vol_trend")
    fieldnames += ["total_return", "sharpe", "max_dd", "trades", "elapsed_s"]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def _write_csv_row(
    writer,
    idx: int,
    params: Dict[str, float],
    metrics: Dict[str, str],
    elapsed: float,
    vary_trend: bool,
) -> None:
    row: Dict[str, Any] = {
        "draw":              idx,
        "mid_vol_no_trend":  params["mid_vol_no_trend"],
        "high_vol":          params["high_vol"],
        "total_return":      metrics["total_return"],
        "sharpe":            metrics["sharpe"],
        "max_dd":            metrics["max_dd"],
        "trades":            metrics["trades"],
        "elapsed_s":         round(elapsed, 1),
    }
    if vary_trend:
        row["mid_vol_trend"] = params.get("mid_vol_trend", "")
    writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Monte Carlo sensitivity analysis for allocation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  py -3.12 tools/montecarlo_alloc.py\n"
            "  py -3.12 tools/montecarlo_alloc.py --n-draws 200 --dist normal\n"
            "  py -3.12 tools/montecarlo_alloc.py --vary-trend --dry-run\n"
            "  py -3.12 tools/montecarlo_alloc.py --start 2022-01-01 --end 2024-12-31\n"
        ),
    )
    ap.add_argument("--n-draws",    default=_DEFAULT_N,       type=int,  dest="n_draws",
                    help=f"Number of random draws (default: {_DEFAULT_N})")
    ap.add_argument("--dist",       default="uniform", choices=["uniform", "normal"],
                    help="Sampling distribution (default: uniform)")
    ap.add_argument("--start",      default=_DEFAULT_START,
                    help=f"Backtest start date (default: {_DEFAULT_START})")
    ap.add_argument("--end",        default=_DEFAULT_END,
                    help=f"Backtest end date (default: {_DEFAULT_END})")
    ap.add_argument("--symbols",    default=_DEFAULT_SYMBOLS,
                    help="Comma-separated symbols (default: 10-stock universe)")
    ap.add_argument("--vary-trend", action="store_true", dest="vary_trend",
                    help="Also vary mid_vol_allocation_trend in Uniform(0.80, 1.00)")
    ap.add_argument("--seed",       default=_DEFAULT_SEED,    type=int,
                    help=f"RNG seed for reproducibility (default: {_DEFAULT_SEED})")
    ap.add_argument("--output-csv", default=None,             dest="output_csv",
                    help="CSV output path (default: results/montecarlo_<timestamp>.csv)")
    ap.add_argument("--dry-run",    action="store_true",      dest="dry_run",
                    help="Print sampled draws without running any backtests")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Generate draws ────────────────────────────────────────────────────
    print(f"\n  Monte Carlo Allocation Sensitivity")
    print(f"  {'-' * 50}")
    print(f"  Draws       : {args.n_draws}")
    print(f"  Distribution: {args.dist}  (mid_vol_no_trend, high_vol ∈ [{_UNIFORM_LO}, {_UNIFORM_HI}])")
    if args.dist == "normal":
        print(f"                Normal(μ={_NORMAL_MU}, σ={_NORMAL_SIGMA}) truncated")
    print(f"  Constraint  : high_vol ≤ mid_vol_no_trend  (always enforced)")
    print(f"  Vary trend  : {'yes  (mid_vol_trend ∈ [0.80, 1.00])' if args.vary_trend else 'no'}")
    print(f"  Seed        : {args.seed}")
    print(f"  Period      : {args.start} → {args.end}")
    print(f"  Symbols     : {args.symbols}")

    draws = _generate_draws(args.n_draws, args.dist, args.vary_trend, rng)

    if args.dry_run:
        print(f"\n  DRY RUN — {len(draws)} draws would be executed:\n")
        hdr = f"  {'#':>4}  {'mid_no_trend':>12}  {'high_vol':>8}"
        if args.vary_trend:
            hdr += f"  {'mid_trend':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, d in enumerate(draws, 1):
            row = f"  {i:>4}  {d['mid_vol_no_trend']:>12.4f}  {d['high_vol']:>8.4f}"
            if args.vary_trend:
                row += f"  {d.get('mid_vol_trend', 0.0):>9.4f}"
            print(row)
        print(f"\n  No backtests run (--dry-run).")
        return

    # ── Load base config ──────────────────────────────────────────────────
    with open(SETTINGS, encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    # Reset critical backtest windows to production defaults.
    # Guards against WFO-modified settings.yaml (train_window=106, test_window=49,
    # sma_long=20, etc.) being in place when this tool runs — those values cause
    # "Insufficient data" errors in every draw.
    _set_nested(base_cfg, "backtest.train_window",      252)
    _set_nested(base_cfg, "backtest.test_window",       126)
    _set_nested(base_cfg, "backtest.step_size",         126)
    _set_nested(base_cfg, "backtest.sma_long",          200)
    _set_nested(base_cfg, "backtest.sma_trend",         50)
    _set_nested(base_cfg, "backtest.volume_norm_window", 50)
    _set_nested(base_cfg, "backtest.zscore_window",     60)

    # ── Backup settings ───────────────────────────────────────────────────
    backup = SETTINGS.with_suffix(".yaml.mc_bak")
    shutil.copy2(SETTINGS, backup)

    # ── Prepare CSV ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output_csv) if args.output_csv else RESULTS_DIR / f"montecarlo_{ts}.csv"
    csv_fh, csv_writer = _open_csv(csv_path, args.vary_trend)

    results: List[Dict] = []
    elapsed_list: List[float] = []
    mc_start = time.monotonic()

    _progress_header(args.vary_trend)

    _first_err_output: Optional[str] = None

    try:
        for idx, params in enumerate(draws, 1):
            metrics, elapsed, raw_output = _run_one_draw(
                idx, args.n_draws,
                params, base_cfg,
                args.symbols, args.start, args.end,
            )
            elapsed_list.append(elapsed)
            record = {**params, **metrics, "elapsed_s": elapsed}
            results.append(record)

            if _to_float(metrics["sharpe"]) is None and _first_err_output is None:
                _first_err_output = raw_output

            _progress_row(idx, args.n_draws, params, metrics, elapsed,
                          elapsed_list, args.vary_trend)
            _write_csv_row(csv_writer, idx, params, metrics, elapsed, args.vary_trend)
            csv_fh.flush()

            # ── Early abort if first 3 draws all ERR ──────────────────────
            if idx == 3 and all(_to_float(r["sharpe"]) is None for r in results):
                print(
                    "\n  [ABORT] First 3 draws all returned ERR — "
                    "likely a systemic failure."
                )
                if _first_err_output is not None:
                    _dump_tail(_first_err_output, n=30, label="draw 1 raw output")
                print(
                    "\n  Verify standalone: "
                    f"py -3.12 main.py backtest --symbols {args.symbols} "
                    f"--start {args.start} --end {args.end}"
                )
                break

    except KeyboardInterrupt:
        print(f"\n  Interrupted at draw {len(results)}/{args.n_draws} — partial results follow.")

    finally:
        csv_fh.close()
        shutil.copy2(backup, SETTINGS)
        backup.unlink(missing_ok=True)
        print("\n  Settings restored.")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.monotonic() - mc_start
    print(f"\n  Total time: {_fmt_elapsed(total_time)}")

    if results:
        _print_summary(results, args.vary_trend, args.n_draws)
        print(f"\n  Raw results saved: {csv_path}")
    else:
        print("  No results to summarise.")


if __name__ == "__main__":
    main()
