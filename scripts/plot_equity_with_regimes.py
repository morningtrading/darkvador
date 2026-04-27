"""
scripts/plot_equity_with_regimes.py
====================================

Produce an equity-curve-with-regime-backdrop chart for the latest backtest:

  - top panel: equity line, with vertical coloured bands per regime segment
  - bottom panel: horizontal ribbon, one row per regime, filled where active

Reads:
  savedresults/backtest_*/equity_curve.csv     (date, equity)
  savedresults/backtest_*/regime_history.csv   (date, regime)

Writes:
  savedresults/backtest_*/equity_with_regimes.png

Usage:
  .venv/bin/python scripts/plot_equity_with_regimes.py            # latest backtest
  .venv/bin/python scripts/plot_equity_with_regimes.py <dir>      # specific dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Regime visual identity — same palette as the dashstreamlite + Telegram views.
REGIME_ORDER = ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"]
REGIME_COLOURS = {
    "CRASH":    "#7f1d1d",   # dark red
    "BEAR":     "#f4a8a8",   # light red
    "NEUTRAL":  "#cbd5e1",   # gray
    "BULL":     "#86efac",   # light green
    "EUPHORIA": "#16a34a",   # dark green
    # Tolerate the legacy 7-state set if we ever come across one.
    "STRONG_BEAR": "#dc2626",
    "WEAK_BEAR":   "#fdba74",
    "WEAK_BULL":   "#bbf7d0",
    "STRONG_BULL": "#22c55e",
}


def latest_backtest_dir() -> Path | None:
    sr = ROOT / "savedresults"
    if not sr.exists():
        return None
    dirs = sorted(sr.glob("backtest_*"), reverse=True)
    return dirs[0] if dirs else None


def regime_segments(rh: pd.DataFrame) -> list[dict]:
    """Run-length encode the regime column into contiguous segments."""
    col = rh["regime"].astype(str)
    changes = col != col.shift()
    seg_id = changes.cumsum()
    out = []
    for _, group in col.groupby(seg_id):
        out.append({
            "regime": group.iloc[0],
            "start":  group.index[0],
            "end":    group.index[-1],
        })
    return out


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_backtest_dir()
    if target is None or not target.exists():
        print(f"No backtest directory available (looked in {ROOT/'savedresults'})")
        return 1

    eq_path  = target / "equity_curve.csv"
    rh_path  = target / "regime_history.csv"
    if not eq_path.exists() or not rh_path.exists():
        print(f"Missing equity_curve.csv or regime_history.csv in {target}")
        return 1

    # ── load data ──────────────────────────────────────────────────────────────
    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    rh = pd.read_csv(rh_path, index_col=0, parse_dates=True)
    rh.columns = ["regime"]
    segs = regime_segments(rh)
    period_label = (
        f"{rh.index.min().strftime('%Y-%m')}  →  "
        f"{rh.index.max().strftime('%Y-%m')}"
    )

    # Try to read the proxy symbol from run_context for the title.
    proxy = "QQQ"
    try:
        import json
        ctx = target / "run_context.json"
        if ctx.exists():
            proxy = json.loads(ctx.read_text()).get("regime_proxy", proxy) or proxy
    except Exception:
        pass

    # ── plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib is not installed in this venv. Install with:")
        print("    .venv/bin/pip install matplotlib")
        return 1

    fig, (ax_eq, ax_rb) = plt.subplots(
        nrows=2, sharex=True,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── top panel: equity + regime bands ───────────────────────────────────────
    for s in segs:
        ax_eq.axvspan(
            s["start"], s["end"],
            facecolor=REGIME_COLOURS.get(s["regime"], "#94a3b8"),
            alpha=0.30, linewidth=0,
        )
    ax_eq.plot(
        eq.index, eq["equity"].values,
        color="#0f172a", linewidth=1.2, label="equity",
    )

    ax_eq.set_ylabel("Equity ($)")
    ax_eq.set_title(
        f"Equity curve with regime backdrop — HMM Regime Trader  "
        f"(darkvador, {proxy} proxy, {period_label})"
    )
    ax_eq.grid(True, axis="y", alpha=0.3)
    ax_eq.set_axisbelow(True)

    # Legend (one chip per regime that actually appears in the data).
    seen = []
    for s in segs:
        if s["regime"] not in seen:
            seen.append(s["regime"])
    ordered = [r for r in REGIME_ORDER if r in seen] + \
              [r for r in seen if r not in REGIME_ORDER]
    handles = [
        Patch(facecolor=REGIME_COLOURS.get(r, "#94a3b8"), alpha=0.6, label=r)
        for r in ordered
    ]
    ax_eq.legend(
        handles=handles, loc="upper left", ncol=len(ordered),
        frameon=True, fancybox=False, framealpha=0.9, fontsize=9,
    )

    # ── bottom panel: regime ribbon ────────────────────────────────────────────
    y_pos = {r: i for i, r in enumerate(REGIME_ORDER)}
    for s in segs:
        y = y_pos.get(s["regime"])
        if y is None:
            continue
        ax_rb.barh(
            y=y,
            width=(s["end"] - s["start"]),
            left=s["start"],
            height=0.7,
            color=REGIME_COLOURS.get(s["regime"], "#94a3b8"),
            alpha=0.85,
        )
    ax_rb.set_yticks(list(y_pos.values()))
    ax_rb.set_yticklabels(list(y_pos.keys()), fontsize=9)
    ax_rb.set_ylabel("Regime")
    ax_rb.set_xlabel("Date")
    ax_rb.grid(False)
    ax_rb.set_axisbelow(True)

    # X-axis formatting
    ax_rb.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_rb.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45, ha="right")

    plt.tight_layout()
    out_path = target / "equity_with_regimes.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
