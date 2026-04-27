"""scripts/compare_sweep_levers.py — print a comparison table for the
config sets backtested via scripts/sweep_levers.sh."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Column shown for each set in the table.
SET_LABELS = [
    ("balanced",          "BASELINE balanced"),
    ("exp_high_conf",     "min_conf 0.62 -> 0.75"),
    ("exp_no_vix",        "drop VIX feature"),
    ("exp_tight_midvol",  "mid_vol_no_trend 0.70 -> 0.50"),
]


def latest_sweep() -> Path:
    sweeps = sorted((ROOT / "savedresults").glob("sweep_levers_*"))
    if not sweeps:
        sys.exit("No sweep_levers_* directory found.")
    return sweeps[-1]


def parse(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for row in csv.reader(path.open()):
        if len(row) >= 2:
            out[row[0]] = row[1]
    return out


def main() -> int:
    sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_sweep()
    print(f"Sweep: {sweep}\n")

    rows = []
    for name, label in SET_LABELS:
        d = parse(sweep / f"{name}.csv")
        if not d:
            rows.append((name, label, None))
            continue
        rows.append((name, label, {
            "ret":    float(d["total_return"]) * 100,
            "cagr":   float(d["cagr"]) * 100,
            "sharpe": float(d["sharpe"]),
            "calmar": float(d["calmar"]),
            "dd":     float(d["max_drawdown"]) * 100,
            "trades": int(float(d["total_trades"])),
            "win":    float(d["win_rate"]) * 100,
        }))

    base = next((m for n, _, m in rows if n == "balanced"), None)

    print(f"  {'Set':<32} {'Return':>8} {'CAGR':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print("  " + "-" * 88)
    for name, label, m in rows:
        if m is None:
            print(f"  {label:<32}  (no result)")
            continue
        print(f"  {label:<32} "
              f"{m['ret']:>+7.2f}% {m['cagr']:>+5.2f}% "
              f"{m['sharpe']:>+6.3f}  {m['calmar']:>+6.3f}  "
              f"{m['dd']:>+7.2f}% {m['trades']:>7}  {m['win']:>5.1f}%")

    if base is not None:
        print()
        print("  Delta vs baseline (positive = improvement):")
        print(f"  {'Set':<32} {'Δ Sharpe':>9} {'Δ Calmar':>10} "
              f"{'Δ MaxDD pp':>11} {'Δ Return pp':>12}")
        print("  " + "-" * 78)
        for name, label, m in rows:
            if m is None or name == "balanced":
                continue
            d_sharpe = m["sharpe"] - base["sharpe"]
            d_calmar = m["calmar"] - base["calmar"]
            d_dd     = m["dd"]     - base["dd"]    # higher (less negative) = better
            d_ret    = m["ret"]    - base["ret"]
            print(f"  {label:<32} "
                  f"{d_sharpe:>+8.3f}  {d_calmar:>+9.3f}  "
                  f"{d_dd:>+10.2f}  {d_ret:>+11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
