"""
tools/rolling_wfo.py — Rolling Walk-Forward Optimisation (WFO).

Purpose
-------
Validates that the strategy's parameter choices are robust across many
different market regimes — not just the single period used for tuning.

This is a VALIDATION tool, not a live-param selector.  It answers:
  "Is this strategy concept robust enough to trust in live trading at all?"

How it works
------------
For each fold:
  1. TUNE  window  : run param_sweep on [fold_start, fold_start + tune_months]
  2. TEST  window  : backtest the winning params on [tune_end, tune_end + test_months]
                     (this data was NEVER seen during tuning)
  3. Shift by step_months and repeat until today

At the end, aggregate across all folds:
  - % of folds where strategy beats buy-and-hold
  - Distribution of Sharpe ratios (min, mean, max, std)
  - Best params per fold — stability check (does the same config win?)
  - Recommended live params = winner in most recent fold

Live param selection
--------------------
Do NOT average parameters across folds.  Use the winner from the most
recent completed fold as the current live parameters.

Usage
-----
    py -3.12 tools/rolling_wfo.py
    py -3.12 tools/rolling_wfo.py --asset-group crypto
    py -3.12 tools/rolling_wfo.py --tune-months 6 --test-months 3 --step-months 3
    py -3.12 tools/rolling_wfo.py --start 2020-01-01 --asset-group indices
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "config" / "settings.yaml"
PYTHON   = sys.executable

# ── Parameter variants (same as param_sweep) ─────────────────────────────────
VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("Baseline",                    {}),
    ("stability_bars=5",            {"hmm.stability_bars": 5}),
    ("stability_bars=7",            {"hmm.stability_bars": 7}),
    ("min_confidence=0.65",         {"hmm.min_confidence": 0.65}),
    ("min_confidence=0.70",         {"hmm.min_confidence": 0.70}),
    ("flicker_threshold=2",         {"hmm.flicker_threshold": 2}),
    ("rebalance=0.15",              {"strategy.rebalance_threshold": 0.15}),
    ("rebalance=0.20",              {"strategy.rebalance_threshold": 0.20}),
    ("combined (stab=5 conf=0.70 reb=0.15)", {
        "hmm.stability_bars":           5,
        "hmm.min_confidence":           0.70,
        "strategy.rebalance_threshold": 0.15,
    }),
]

# ── ANSI / metric helpers (shared with param_sweep) ──────────────────────────
_ANSI_RE    = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY_RE = re.compile(
    r"Total return:\s*([\+\-\d\.]+%)\s+Sharpe:\s*([\d\.]+)\s+MaxDD:\s*([\+\-\d\.]+%)"
)
_TRADES_RE  = re.compile(r"Total trades\s*:\s*(\d+)")

_STREAM_PATTERNS = [
    re.compile(r"Folds completed"),
    re.compile(r"Total trades"),
    re.compile(r"Backtest complete"),
    re.compile(r"ERROR|CRITICAL", re.IGNORECASE),
]


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _parse_metrics(output: str) -> Dict[str, str]:
    clean = _strip_ansi(output)
    m = _SUMMARY_RE.search(clean)
    t = _TRADES_RE.search(clean)
    return {
        "total_return": m.group(1) if m else "ERR",
        "sharpe":       m.group(2) if m else "ERR",
        "max_dd":       m.group(3) if m else "ERR",
        "trades":       t.group(1) if t else "—",
    }


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s"


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


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_streaming(cmd: List[str], indent: str = "      ") -> str:
    """Run a subprocess, stream matching lines, return full output."""
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
        plain = _strip_ansi(line)
        if any(p.search(plain) for p in _STREAM_PATTERNS):
            clean = re.sub(r"^\s*\d{2}:\d{2}:\d{2}\s+\w+\s+\S+\s+", "", plain).strip()
            print(f"{indent}{clean}", flush=True)
    proc.wait()
    return "\n".join(captured)


def _dump_tail(output: str, n: int = 30, label: str = "") -> None:
    """Print the last n lines of subprocess output for debugging."""
    lines = [_strip_ansi(l) for l in output.splitlines() if _strip_ansi(l).strip()]
    tail = lines[-n:] if len(lines) > n else lines
    print(f"\n  [DEBUG{' ' + label if label else ''}] Last {len(tail)} lines of output:")
    for l in tail:
        print(f"    | {l}")
    print(flush=True)


def _run_sweep(symbols: str, start: str, end: str, overrides: Dict[str, Any],
               base_cfg: dict) -> str:
    """Apply overrides to settings, run backtest, restore settings."""
    cfg = _apply_variant(base_cfg, overrides)
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    cmd = [PYTHON, str(ROOT / "main.py"), "backtest",
           "--symbols", symbols, "--start", start, "--end", end]
    return _run_streaming(cmd)


def _run_test(symbols: str, start: str, end: str) -> str:
    """Backtest with current settings.yaml (best params already written)."""
    cmd = [PYTHON, str(ROOT / "main.py"), "backtest",
           "--symbols", symbols, "--start", start, "--end", end, "--compare"]
    return _run_streaming(cmd, indent="    ")


# ── Table helpers ─────────────────────────────────────────────────────────────

def _print_fold_table(fold_results: List[Dict]) -> None:
    cols = ["Fold", "Tune period", "Test period", "Best params",
            "Test Return", "Test Sharpe", "Test MaxDD", "Beats B&H"]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in fold_results))
              for c in cols]

    def _row(vals: List[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    sep = "  ".join("-" * w for w in widths)
    print()
    print(_row(cols))
    print(sep)
    for r in fold_results:
        print(_row([r.get(c, "—") for c in cols]))
    print()


def _param_stability(fold_results: List[Dict]) -> Dict[str, int]:
    """Count how many folds each param config won."""
    counts: Dict[str, int] = {}
    for r in fold_results:
        key = r.get("Best params", "—")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ── Date helpers ──────────────────────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    return d + relativedelta(months=months)


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rolling Walk-Forward Optimisation",
        epilog=(
            "Workflow:\n"
            "  1. WFO validates the strategy concept across many market regimes\n"
            "  2. If robust (>60%% folds beat B&H), use most-recent-fold winner as live params\n"
            "  3. Retune every 3 months on a rolling basis\n"
        ),
    )
    ap.add_argument("--asset-group",  default=None, dest="asset_group",
                    help="Asset group (stocks | crypto | indices)")
    ap.add_argument("--symbols",      default=None,
                    help="Comma-separated symbols — overrides asset-group")
    ap.add_argument("--start",        default="2020-01-01",
                    help="First fold start date (default: 2020-01-01)")
    ap.add_argument("--tune-months",  default=12, type=int, dest="tune_months",
                    help="Tuning window length in months (default: 12)")
    ap.add_argument("--test-months",  default=3, type=int, dest="test_months",
                    help="Test window length in months (default: 3)")
    ap.add_argument("--step-months",  default=3, type=int, dest="step_months",
                    help="How far to advance each fold in months (default: 3)")
    ap.add_argument("--show-windows", action="store_true", dest="show_windows",
                    help="Print fold schedule and window sizes then exit — no backtests run")
    args = ap.parse_args()

    with open(SETTINGS, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    # Resolve symbols
    if args.symbols:
        resolved_symbols = args.symbols
    elif args.asset_group:
        group_syms = base_cfg.get("asset_groups", {}).get(args.asset_group)
        if not group_syms:
            print(f"ERROR: asset group '{args.asset_group}' not found.")
            sys.exit(1)
        resolved_symbols = ",".join(group_syms)
    else:
        active = base_cfg.get("broker", {}).get("asset_group", "stocks")
        group_syms = base_cfg.get("asset_groups", {}).get(
            active, base_cfg.get("broker", {}).get("symbols", ["SPY"])
        )
        resolved_symbols = ",".join(group_syms)

    group_label = args.asset_group or base_cfg.get("broker", {}).get("asset_group", "stocks")

    # Build fold schedule
    today      = date.today()
    fold_start = date.fromisoformat(args.start)
    folds: List[Tuple[date, date, date, date]] = []  # tune_start, tune_end, test_start, test_end

    while True:
        tune_end  = _add_months(fold_start, args.tune_months)
        test_end  = _add_months(tune_end,   args.test_months)
        if tune_end >= today:
            break   # not enough data for a full tune window
        test_end = min(test_end, today)
        folds.append((fold_start, tune_end, tune_end, test_end))
        fold_start = _add_months(fold_start, args.step_months)
        if test_end >= today:
            break

    if not folds:
        print("ERROR: not enough data for even one fold. Try an earlier --start date.")
        sys.exit(1)

    # ── Scale walk-forward windows to fit inside the tune/test periods ───────
    # The backtest walk-forward needs train_window + test_window + buffer bars.
    # A 6-month tune window ≈ 126 daily bars — far less than the default 252+126.
    # We scale down so every tune/test run has at least one complete WF fold.
    TRADING_DAYS_PER_MONTH = 21
    tune_bars   = args.tune_months * TRADING_DAYS_PER_MONTH
    test_bars   = args.test_months * TRADING_DAYS_PER_MONTH
    # Use 55% for IS training, 25% for OOS test, leaving 20% buffer
    wf_train    = max(42, int(tune_bars * 0.55))
    wf_test     = max(21, int(tune_bars * 0.25))
    wf_step     = wf_test
    # Overrides applied to EVERY tune variant run
    _TUNE_BT_OVERRIDES = {
        "backtest.train_window": wf_train,
        "backtest.test_window":  wf_test,
        "backtest.step_size":    wf_step,
        "hmm.min_train_bars":    max(20, wf_train - 5),   # must be < train_window
    }
    # Scale test-phase BT windows to fit the test period
    test_wf_train = max(42, int(test_bars * 0.55))
    test_wf_test  = max(21, int(test_bars * 0.25))
    _TEST_BT_OVERRIDES = {
        "backtest.train_window": test_wf_train,
        "backtest.test_window":  test_wf_test,
        "backtest.step_size":    test_wf_test,
        "hmm.min_train_bars":    max(20, test_wf_train - 5),  # must be < train_window
    }

    # ── Window / fold summary (shared between --show-windows and normal run) ──
    tune_min_needed = wf_train + wf_test
    tune_buffer     = tune_bars - tune_min_needed
    test_min_needed = test_wf_train + test_wf_test
    test_buffer     = test_bars - test_min_needed

    def _print_windows() -> None:
        print(f"\n{'=' * 70}")
        print(f"  WFO WINDOW PLAN")
        print(f"{'=' * 70}")
        print(f"  Group       : {group_label}")
        print(f"  Start date  : {args.start}")
        print(f"  Tune months : {args.tune_months}  (~{tune_bars} trading days)")
        print(f"  Test months : {args.test_months}  (~{test_bars} trading days)")
        print(f"  Step months : {args.step_months}")
        print(f"  Folds       : {len(folds)}")
        print(f"  Variants    : {len(VARIANTS)} per fold")
        total_runs = len(folds) * len(VARIANTS)
        print(f"  Total runs  : {total_runs}  (~{total_runs * 1.5:.0f}–{total_runs * 2:.0f} min estimate)")
        print(f"{'─' * 70}")
        tune_hmm_min = max(20, wf_train - 5)
        test_hmm_min = max(20, test_wf_train - 5)
        print(f"  TUNE phase backtest windows (applied to each variant):")
        print(f"    train_window   : {wf_train} bars  (55% of {tune_bars})")
        print(f"    test_window    : {wf_test} bars  (25% of {tune_bars})")
        print(f"    step_size      : {wf_step} bars")
        print(f"    min_train_bars : {tune_hmm_min} bars  (train_window - 5)")
        print(f"    min needed     : {tune_min_needed} bars   available: {tune_bars}   buffer: {tune_buffer} bars"
              + ("  ✓" if tune_buffer >= 0 else "  ✗ TOO SHORT"))
        print(f"{'─' * 70}")
        print(f"  TEST phase backtest windows (applied to winning params):")
        print(f"    train_window   : {test_wf_train} bars  (55% of {test_bars})")
        print(f"    test_window    : {test_wf_test} bars  (25% of {test_bars})")
        print(f"    step_size      : {test_wf_test} bars")
        print(f"    min_train_bars : {test_hmm_min} bars  (train_window - 5)")
        print(f"    min needed     : {test_min_needed} bars   available: {test_bars}   buffer: {test_buffer} bars"
              + ("  ✓" if test_buffer >= 0 else "  ✗ TOO SHORT"))
        print(f"{'─' * 70}")
        print(f"  {'Fold':<4}  {'Tune window':<25}  {'Test window':<25}  Notes")
        print(f"  {'─'*4}  {'─'*25}  {'─'*25}  {'─'*20}")
        for i, (ts, te, ts2, te2) in enumerate(folds, 1):
            note = "(partial)" if te2 >= date.today() else ""
            print(f"  {i:<4}  {_fmt(ts)} → {_fmt(te)}    {_fmt(ts2)} → {_fmt(te2)}    {note}")
        print()

    _print_windows()

    if args.show_windows:
        return

    # Print plan
    print(f"\n{'=' * 70}")
    print(f"  ROLLING WALK-FORWARD OPTIMISATION")
    print(f"{'=' * 70}")
    print(f"  Group       : {group_label}")
    print(f"  Assets      : {resolved_symbols}")
    print(f"  Tune window : {args.tune_months} months (~{tune_bars} bars)")
    print(f"  Test window : {args.test_months} months")
    print(f"  Step        : {args.step_months} months")
    print(f"  Tune WF     : IS={wf_train} / OOS={wf_test} / step={wf_step} bars (scaled to fit tune)")
    print(f"  Test WF     : IS={test_wf_train} / OOS={test_wf_test} / step={test_wf_test} bars (scaled to fit test)")
    print(f"  Folds       : {len(folds)}")
    print(f"  Variants    : {len(VARIANTS)} per fold")
    total_runs = len(folds) * len(VARIANTS)
    print(f"  Total runs  : {total_runs}  (expect ~{total_runs * 1.5:.0f}–{total_runs * 2:.0f} min)")
    print(f"{'─' * 70}")

    backup = SETTINGS.with_suffix(".yaml.wfo_bak")
    shutil.copy2(SETTINGS, backup)

    fold_results: List[Dict] = []
    wfo_start_time = time.monotonic()

    try:
        for fi, (tune_start, tune_end, test_start, test_end) in enumerate(folds, 1):
            print(f"\n{'━' * 70}")
            print(f"  FOLD {fi}/{len(folds)}"
                  f"  │  Tune: {_fmt(tune_start)} → {_fmt(tune_end)}"
                  f"  │  Test: {_fmt(test_start)} → {_fmt(test_end)}")
            print(f"{'━' * 70}")

            # ── TUNE: find best variant on tune window ──────────────────────
            print(f"\n  [TUNE] Running {len(VARIANTS)} variants ...", flush=True)
            best_label   = "Baseline"
            best_sharpe  = -999.0
            best_overrides: Dict[str, Any] = {}

            _debug_dumped = False   # only dump once per fold
            for vi, (label, overrides) in enumerate(VARIANTS, 1):
                print(f"    [{vi}/{len(VARIANTS)}] {label}", flush=True)
                # Merge tune-window BT sizing with variant-specific overrides
                merged = {**_TUNE_BT_OVERRIDES, **overrides}
                output  = _run_sweep(
                    resolved_symbols,
                    _fmt(tune_start), _fmt(tune_end),
                    merged, base_cfg,
                )
                metrics = _parse_metrics(output)
                sharpe_str = metrics["sharpe"]
                try:
                    sharpe = float(sharpe_str)
                except ValueError:
                    sharpe = -999.0
                    # First ERR in this fold — dump raw output to diagnose
                    if not _debug_dumped and fi == 1:
                        _dump_tail(output, n=40, label=f"fold {fi} variant {vi}")
                        _debug_dumped = True

                print(f"         → Sharpe {sharpe_str}  "
                      f"Return {metrics['total_return']}  "
                      f"MaxDD {metrics['max_dd']}", flush=True)

                if sharpe > best_sharpe:
                    best_sharpe   = sharpe
                    best_label    = label
                    best_overrides = overrides

            print(f"\n  [TUNE WINNER] {best_label}  (Sharpe {best_sharpe:.3f})", flush=True)

            # ── Apply best params + test-period BT window scaling ────────────
            # best_overrides contains only HMM/strategy params.
            # We merge _TEST_BT_OVERRIDES so the test period (e.g. 3 months)
            # has appropriately scaled walk-forward windows.
            test_overrides = {**best_overrides, **_TEST_BT_OVERRIDES}
            best_cfg = _apply_variant(base_cfg, test_overrides)
            with open(SETTINGS, "w", encoding="utf-8") as fh:
                yaml.dump(best_cfg, fh, default_flow_style=False, sort_keys=False)

            # ── TEST: blind run on held-out window ───────────────────────────
            print(f"\n  [TEST] Blind test on {_fmt(test_start)} → {_fmt(test_end)} ...",
                  flush=True)
            t0          = time.monotonic()
            test_output = _run_test(resolved_symbols, _fmt(test_start), _fmt(test_end))
            test_time   = time.monotonic() - t0
            test_metrics = _parse_metrics(test_output)

            # Check if strategy beats buy-and-hold
            # Buy-and-hold Sharpe is in the comparison table — parse it
            bnh_sharpe_m = re.search(
                r"Buy & Hold.*?Sharpe.*?([\d\.]+)", _strip_ansi(test_output)
            )
            bnh_sharpe = float(bnh_sharpe_m.group(1)) if bnh_sharpe_m else None
            try:
                test_sharpe = float(test_metrics["sharpe"])
                beats_bnh   = "YES" if (bnh_sharpe and test_sharpe > bnh_sharpe) else "NO"
            except ValueError:
                beats_bnh = "ERR"

            print(f"  [TEST RESULT] Return {test_metrics['total_return']}"
                  f"  Sharpe {test_metrics['sharpe']}"
                  f"  MaxDD {test_metrics['max_dd']}"
                  f"  Beats B&H: {beats_bnh}"
                  f"  ({_fmt_elapsed(test_time)})", flush=True)

            fold_results.append({
                "Fold":        str(fi),
                "Tune period": f"{_fmt(tune_start)} → {_fmt(tune_end)}",
                "Test period": f"{_fmt(test_start)} → {_fmt(test_end)}",
                "Best params": best_label,
                "Test Return": test_metrics["total_return"],
                "Test Sharpe": test_metrics["sharpe"],
                "Test MaxDD":  test_metrics["max_dd"],
                "Beats B&H":   beats_bnh,
            })

    finally:
        shutil.copy2(backup, SETTINGS)
        backup.unlink(missing_ok=True)
        print("\n  Settings restored.")

    # ── Aggregate results ─────────────────────────────────────────────────────
    total_time = time.monotonic() - wfo_start_time
    print(f"\n  Total WFO time: {_fmt_elapsed(total_time)}")
    print(f"\n{'=' * 70}")
    print(f"  ROLLING WFO RESULTS — {group_label.upper()}")
    print(f"{'=' * 70}")

    _print_fold_table(fold_results)

    # Summary statistics
    valid_sharpes = []
    for r in fold_results:
        try:
            valid_sharpes.append(float(r["Test Sharpe"]))
        except ValueError:
            pass

    beats_count = sum(1 for r in fold_results if r["Beats B&H"] == "YES")
    total_folds = len(fold_results)

    print(f"  Robustness    : {beats_count}/{total_folds} folds beat Buy & Hold"
          f"  ({100 * beats_count / total_folds:.0f}%)")

    if valid_sharpes:
        import statistics
        print(f"  Sharpe dist   : "
              f"min={min(valid_sharpes):.3f}  "
              f"mean={statistics.mean(valid_sharpes):.3f}  "
              f"max={max(valid_sharpes):.3f}  "
              f"std={statistics.stdev(valid_sharpes):.3f}" if len(valid_sharpes) > 1
              else f"min={min(valid_sharpes):.3f}  mean={valid_sharpes[0]:.3f}  max={max(valid_sharpes):.3f}")

    # Parameter stability
    stability = _param_stability(fold_results)
    print(f"\n  Parameter stability (wins per config):")
    for label, count in stability.items():
        bar = "█" * count
        print(f"    {bar} {count:2d}x  {label}")

    # Live param recommendation = most recent fold winner
    if fold_results:
        latest = fold_results[-1]
        print(f"\n  Recommended live params : {latest['Best params']}")
        print(f"    (Winner of most recent fold: {latest['Tune period']})")
        if beats_count / total_folds >= 0.6:
            print(f"  Strategy verdict : ROBUST — deploy with confidence")
        elif beats_count / total_folds >= 0.4:
            print(f"  Strategy verdict : MARGINAL — paper trade before going live")
        else:
            print(f"  Strategy verdict : NOT ROBUST — do not deploy")

    print()


if __name__ == "__main__":
    main()
