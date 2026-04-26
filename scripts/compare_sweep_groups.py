"""
scripts/compare_sweep_groups.py — aggregate every savedresults/sweep_groups_*
directory and print:
  1. Per-set ranking (one mini-table per set)
  2. Overall top across all (set × basket) combos
  3. Per-basket: which set wins

Composite score: 0.4*Sharpe + 0.4*Calmar + 0.2*(1+MaxDD)
"""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASKETS = ["stocks", "stocks2", "stocks3", "stocks4", "stocks_gold", "indices"]


def parse_csv(path: Path) -> dict | None:
    if not path.exists():
        return None
    out = {}
    for row in csv.reader(path.open()):
        if len(row) >= 2:
            out[row[0]] = row[1]
    return out


def metrics(d: dict) -> dict:
    ret    = float(d["total_return"]) * 100
    cagr   = float(d["cagr"]) * 100
    sharpe = float(d["sharpe"])
    sortino = float(d["sortino"])
    calmar = float(d["calmar"])
    dd_raw = float(d["max_drawdown"])
    dd     = dd_raw * 100
    dd_d   = int(float(d["max_dd_days"]))
    trades = int(float(d["total_trades"]))
    win    = float(d["win_rate"]) * 100
    score  = 0.4 * sharpe + 0.4 * calmar + 0.2 * (1.0 + dd_raw)
    return dict(ret=ret, cagr=cagr, sharpe=sharpe, sortino=sortino,
                calmar=calmar, dd=dd, dd_d=dd_d, trades=trades,
                win=win, score=score)


def collect() -> dict:
    """Return {set_name: {basket: metrics_dict}}."""
    out: dict[str, dict[str, dict]] = {}
    for sweep in sorted((ROOT / "savedresults").glob("sweep_groups_*")):
        m = re.match(r"sweep_groups_(.+?)_\d{8}_\d{6}$", sweep.name)
        if not m:
            continue
        set_name = m.group(1)
        # Newer sweep wins if name collides
        out.setdefault(set_name, {})
        for b in BASKETS:
            d = parse_csv(sweep / f"{b}.csv")
            if d is not None:
                out[set_name][b] = metrics(d)
    return out


def print_set_table(set_name: str, basket_to_m: dict) -> None:
    rows = sorted(basket_to_m.items(), key=lambda kv: -kv[1]["score"])
    print(f"\n  ── set: {set_name} ──")
    print(f"  {'Rank':>4}  {'Basket':<12} {'Return':>8} {'CAGR':>7} "
          f"{'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8} {'Score':>7}")
    print("  " + "-" * 72)
    for i, (b, m) in enumerate(rows, 1):
        marker = ["#1", "#2", "#3"][i - 1] if i <= 3 else f"#{i}"
        print(f"  {marker:>4}  {b:<12} {m['ret']:>+7.1f}%  {m['cagr']:>+5.1f}% "
              f"{m['sharpe']:>+7.2f}  {m['calmar']:>+7.2f}  "
              f"{m['dd']:>+7.1f}%  {m['score']:>+7.3f}")


def main() -> None:
    data = collect()
    if not data:
        print("No sweep_groups_* directories found.")
        return

    set_order = [s for s in ["conservative", "balanced", "aggressive"] if s in data]
    set_order += [s for s in data if s not in set_order]

    # 1. Per-set tables
    print("=" * 80)
    print("  Per-set rankings")
    print("=" * 80)
    for s in set_order:
        print_set_table(s, data[s])

    # 2. Overall top-N across all combos
    flat = []
    for s, baskets in data.items():
        for b, m in baskets.items():
            flat.append((s, b, m))
    flat.sort(key=lambda r: -r[2]["score"])

    print("\n" + "=" * 80)
    print("  Overall top — all (set × basket) combos by score")
    print("=" * 80)
    print(f"  {'Rank':>4}  {'Set':<13} {'Basket':<12} {'Return':>8} "
          f"{'Sharpe':>7} {'Calmar':>7} {'MaxDD':>8} {'Score':>7}")
    print("  " + "-" * 76)
    for i, (s, b, m) in enumerate(flat, 1):
        marker = ["#1", "#2", "#3"][i - 1] if i <= 3 else f"#{i}"
        print(f"  {marker:>4}  {s:<13} {b:<12} {m['ret']:>+7.1f}% "
              f"{m['sharpe']:>+7.2f}  {m['calmar']:>+7.2f}  "
              f"{m['dd']:>+7.1f}%  {m['score']:>+7.3f}")

    # 3. Per-basket: which set wins?
    print("\n" + "=" * 80)
    print("  Per-basket — best set on score")
    print("=" * 80)
    print(f"  {'Basket':<12} | " + " | ".join(f"{s:^17}" for s in set_order)
          + " | best set")
    print("  " + "-" * (12 + 3 + (17 + 3) * len(set_order) + 11))
    for b in BASKETS:
        cells, best, best_score = [], None, -1e9
        for s in set_order:
            m = data.get(s, {}).get(b)
            if m is None:
                cells.append(" " * 17)
                continue
            cell = f"{m['sharpe']:+.2f}/{m['calmar']:+.2f}/{m['dd']:+.0f}%"
            cells.append(f"{cell:^17}")
            if m["score"] > best_score:
                best_score = m["score"]
                best = s
        print(f"  {b:<12} | " + " | ".join(cells)
              + f" | {best or '?':<13}")
    print("  (cells = Sharpe/Calmar/MaxDD)")


if __name__ == "__main__":
    main()
