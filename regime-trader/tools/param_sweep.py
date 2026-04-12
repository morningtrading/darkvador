"""
tools/param_sweep.py — Regime stability & trade-frequency parameter sweep.

Runs the backtest for each config variant defined in VARIANTS, streams live
progress, captures key metrics, restores settings.yaml, and prints a
comparison table.

No HMM retraining between variants — stability_bars, min_confidence,
flicker_threshold, and rebalance_threshold are all applied at backtest time
from the config, so the saved model file is reused as-is.

Usage:
    py -3.12 tools/param_sweep.py
    py -3.12 tools/param_sweep.py --symbols SPY,QQQ --start 2020-01-01
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "config" / "settings.yaml"
PYTHON   = sys.executable

# ── Parameter variants to sweep ───────────────────────────────────────────────
VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("Baseline",                    {}),
    ("stability_bars=5",            {"hmm.stability_bars": 5}),
    ("stability_bars=7",            {"hmm.stability_bars": 7}),
    ("min_confidence=0.65",         {"hmm.min_confidence": 0.65}),
    ("min_confidence=0.70",         {"hmm.min_confidence": 0.70}),
    ("flicker_threshold=2",         {"hmm.flicker_threshold": 2}),
    ("rebalance=0.15",              {"strategy.rebalance_threshold": 0.15}),
    ("rebalance=0.20",              {"strategy.rebalance_threshold": 0.20}),
    ("trend_lookback=200",          {"strategy.trend_lookback": 200}),
    ("combined (stab=5 conf=0.70 reb=0.15)", {
        "hmm.stability_bars":           5,
        "hmm.min_confidence":           0.70,
        "strategy.rebalance_threshold": 0.15,
    }),
    ("flicker=2 + conf=0.65", {
        "hmm.flicker_threshold": 2,
        "hmm.min_confidence":    0.65,
    }),
]

# ── Lines to surface from subprocess output ───────────────────────────────────
# Patterns whose matching lines are printed indented during the run.
_STREAM_PATTERNS = [
    re.compile(r"Fold\s+\d+", re.IGNORECASE),
    re.compile(r"Folds completed"),
    re.compile(r"Total trades"),
    re.compile(r"Final equity"),
    re.compile(r"Backtest complete"),
    re.compile(r"ERROR|CRITICAL", re.IGNORECASE),
]

# ── ANSI stripping ────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ── Metric extraction ──────────────────────────────────────────────────────────
_SUMMARY_RE = re.compile(
    r"Total return:\s*([\+\-\d\.]+%)\s+Sharpe:\s*([\d\.]+)\s+MaxDD:\s*([\+\-\d\.]+%)"
)
_TRADES_RE  = re.compile(r"Total trades\s*:\s*(\d+)")
# Rich table uses │ (U+2502) as column separator; also handle plain | and :
_CAGR_RE    = re.compile(r"CAGR\s*[:\|\u2502]\s*([\+\-\d\.]+%)")


def _parse_metrics(output: str) -> Dict[str, str]:
    clean = _strip_ansi(output)
    m = _SUMMARY_RE.search(clean)
    t = _TRADES_RE.search(clean)
    c = _CAGR_RE.search(clean)
    return {
        "total_return": m.group(1) if m else "ERR",
        "sharpe":       m.group(2) if m else "ERR",
        "max_dd":       m.group(3) if m else "ERR",
        "trades":       t.group(1) if t else "—",
        "cagr":         c.group(1) if c else "—",
    }


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _apply_variant(base: dict, overrides: Dict[str, Any]) -> dict:
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        _set_nested(cfg, k, v)
    return cfg


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s"


def _eta(elapsed_per_variant: List[float], remaining: int) -> str:
    if not elapsed_per_variant:
        return "—"
    avg = sum(elapsed_per_variant) / len(elapsed_per_variant)
    return _fmt_elapsed(avg * remaining)


# ── Run one backtest (streaming) ───────────────────────────────────────────────

def _run_backtest_streaming(symbols: str, start: str, end: Optional[str] = None) -> str:
    """
    Run the backtest subprocess, stream interesting lines to stdout in real
    time, and return the full combined output for metric parsing.
    """
    cmd = [
        PYTHON, str(ROOT / "main.py"),
        "backtest",
        "--symbols", symbols,
        "--start",   start,
    ]
    if end:
        cmd += ["--end", end]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )

    captured: List[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        captured.append(line)
        # Print lines that match progress patterns, indented and ANSI-clean
        plain = _strip_ansi(line)
        if any(p.search(plain) for p in _STREAM_PATTERNS):
            # Strip log-level prefixes for cleaner output
            clean = re.sub(r"^\s*\d{2}:\d{2}:\d{2}\s+\w+\s+\S+\s+", "", plain).strip()
            print(f"    {clean}", flush=True)

    proc.wait()
    return "\n".join(captured)


# ── Table printing ─────────────────────────────────────────────────────────────

def _print_table(rows: List[Dict]) -> None:
    cols = ["Label", "Total Return", "CAGR", "Sharpe", "MaxDD", "Trades", "Time"]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]

    def _row(vals: List[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    sep = "  ".join("-" * w for w in widths)
    print()
    print(_row(cols))
    print(sep)
    for r in rows:
        print(_row([r.get(c, "—") for c in cols]))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regime parameter sweep",
        epilog=(
            "Proper workflow:\n"
            "  1. Tune  : param_sweep.py --start 2020-01-01 --end 2023-12-31\n"
            "  2. Verify: apply best params, then backtest --start 2024-01-01 (forward test)\n"
        ),
    )
    ap.add_argument("--asset-group", default=None, dest="asset_group",
                    help="Asset group name from config (stocks | crypto | indices). "
                         "Overrides --symbols.")
    ap.add_argument("--symbols",     default=None,
                    help="Comma-separated symbols — overrides --asset-group")
    ap.add_argument("--start",       default="2020-01-01",
                    help="Tuning window start date (default: 2020-01-01)")
    ap.add_argument("--end",         default="2023-12-31",
                    help="Tuning window end date (default: 2023-12-31 — keep 2024+ as hold-out)")
    args = ap.parse_args()

    # Resolve symbols from group or explicit list
    with open(SETTINGS, "r", encoding="utf-8") as fh:
        _cfg_for_syms = yaml.safe_load(fh)
    if args.symbols:
        resolved_symbols = args.symbols
    elif args.asset_group:
        group_syms = _cfg_for_syms.get("asset_groups", {}).get(args.asset_group)
        if not group_syms:
            print(f"ERROR: asset group '{args.asset_group}' not found in config.")
            print(f"  Available: {list(_cfg_for_syms.get('asset_groups', {}).keys())}")
            sys.exit(1)
        resolved_symbols = ",".join(group_syms)
    else:
        # Default to active group in config
        active = _cfg_for_syms.get("broker", {}).get("asset_group", "stocks")
        group_syms = _cfg_for_syms.get("asset_groups", {}).get(active,
                     _cfg_for_syms.get("broker", {}).get("symbols", ["SPY"]))
        resolved_symbols = ",".join(group_syms)

    with open(SETTINGS, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    backup = SETTINGS.with_suffix(".yaml.sweep_bak")
    shutil.copy2(SETTINGS, backup)

    results:          List[Dict]  = []
    elapsed_per_run:  List[float] = []
    total = len(VARIANTS)

    holdout_year = str(int(args.end[:4]) + 1)
    group_label  = args.asset_group or _cfg_for_syms.get("broker", {}).get("asset_group", "stocks")
    print(f"\n  Sweep : {total} variants")
    print(f"  Group : {group_label}")
    print(f"  Tune  : {args.start}  →  {args.end}  (hold-out: {holdout_year}+ never touched)")
    print(f"  Syms  : {resolved_symbols}")
    print("  " + "─" * 60)

    try:
        for i, (label, overrides) in enumerate(VARIANTS, 1):
            remaining = total - i
            eta_str   = _eta(elapsed_per_run, remaining)
            print(
                f"\n[{i}/{total}]  {label}"
                + (f"  |  ETA: ~{eta_str}" if elapsed_per_run else ""),
                flush=True,
            )
            if overrides:
                for k, v in overrides.items():
                    print(f"    {k} = {v}", flush=True)

            cfg = _apply_variant(base_cfg, overrides)
            with open(SETTINGS, "w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

            t0     = time.monotonic()
            output = _run_backtest_streaming(resolved_symbols, args.start, args.end)
            elapsed = time.monotonic() - t0
            elapsed_per_run.append(elapsed)

            metrics = _parse_metrics(output)
            results.append({
                "Label":        label,
                "Total Return": metrics["total_return"],
                "CAGR":         metrics["cagr"],
                "Sharpe":       metrics["sharpe"],
                "MaxDD":        metrics["max_dd"],
                "Trades":       metrics["trades"],
                "Time":         _fmt_elapsed(elapsed),
            })
            print(
                f"    → {metrics['total_return']}  Sharpe {metrics['sharpe']}"
                f"  MaxDD {metrics['max_dd']}  Trades {metrics['trades']}"
                f"  ({_fmt_elapsed(elapsed)})",
                flush=True,
            )

    finally:
        shutil.copy2(backup, SETTINGS)
        backup.unlink(missing_ok=True)
        print("\n  Settings restored.")

    total_time = sum(elapsed_per_run)
    print(f"\n  Total sweep time: {_fmt_elapsed(total_time)}")

    # Re-read the restored config for the recap block
    with open(SETTINGS, "r", encoding="utf-8") as fh:
        final_cfg = yaml.safe_load(fh)

    broker_cfg   = final_cfg.get("broker",   {})
    hmm_cfg      = final_cfg.get("hmm",      {})
    strategy_cfg = final_cfg.get("strategy", {})
    risk_cfg     = final_cfg.get("risk",      {})
    bt_cfg       = final_cfg.get("backtest", {})

    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP RESULTS")
    print("=" * 70)
    print(f"  Group       : {group_label}")
    print(f"  Assets      : {resolved_symbols}")
    print(f"  Tune period : {args.start}  →  {args.end}  "
          f"(hold-out: {holdout_year}+)")
    print(f"  Frequency   : {broker_cfg.get('timeframe', '5Min')} bars")
    print(f"  Capital     : ${float(bt_cfg.get('initial_capital', 100_000)):,.0f}   "
          f"Slippage: {float(bt_cfg.get('slippage_pct', 0.0005)) * 10_000:.1f} bps")
    print(f"  Walk-fwd    : IS {bt_cfg.get('train_window', 252)} / "
          f"OOS {bt_cfg.get('test_window', 126)} bars  "
          f"step {bt_cfg.get('step_size', 126)}")
    print(f"  HMM         : states {hmm_cfg.get('n_candidates')}  "
          f"n_init={hmm_cfg.get('n_init')}  "
          f"cov={hmm_cfg.get('covariance_type')}")
    print(f"  Baseline    : stability={hmm_cfg.get('stability_bars')}  "
          f"flicker_thresh={hmm_cfg.get('flicker_threshold')}  "
          f"flicker_win={hmm_cfg.get('flicker_window')}  "
          f"min_conf={hmm_cfg.get('min_confidence')}")
    print(f"  Allocation  : low={strategy_cfg.get('low_vol_allocation')}×"
          f"{strategy_cfg.get('low_vol_leverage')}x  "
          f"mid={strategy_cfg.get('mid_vol_allocation_trend')}/"
          f"{strategy_cfg.get('mid_vol_allocation_no_trend')}  "
          f"high={strategy_cfg.get('high_vol_allocation')}")
    print(f"  Rebalance   : {strategy_cfg.get('rebalance_threshold')}  "
          f"trend_lb={strategy_cfg.get('trend_lookback')}")
    print(f"  Risk        : max_pos={risk_cfg.get('max_single_position')}  "
          f"max_exp={risk_cfg.get('max_exposure')}  "
          f"dd_halt={risk_cfg.get('daily_dd_halt')}")
    print("─" * 70)
    _print_table(results)

    valid = [r for r in results if r["Sharpe"] not in ("ERR", "—")]
    if valid:
        best = max(valid, key=lambda r: float(r["Sharpe"]))
        print(f"  Best Sharpe     : {best['Label']}  ({best['Sharpe']})")

    valid_dd = [r for r in results if r["MaxDD"] not in ("ERR", "—")]
    if valid_dd:
        # MaxDD values are negative; max() picks the least negative = smallest drawdown
        best_dd = max(
            valid_dd,
            key=lambda r: float(r["MaxDD"].replace("%", "").replace("+", "")),
        )
        print(f"  Lowest MaxDD    : {best_dd['Label']}  ({best_dd['MaxDD']})")

    valid_ret = [r for r in results if r["Total Return"] not in ("ERR", "—")]
    if valid_ret:
        best_ret = max(
            valid_ret,
            key=lambda r: float(r["Total Return"].replace("%", "").replace("+", "")),
        )
        print(f"  Best Total Ret  : {best_ret['Label']}  ({best_ret['Total Return']})")
    print()


if __name__ == "__main__":
    main()
