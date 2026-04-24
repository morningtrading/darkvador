"""
telegram/formatter.py — Format trading data into compact Telegram messages.

All functions return a ready-to-send HTML string (4-6 lines max).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

def _pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"

def _latest_backtest_dir() -> Optional[Path]:
    sr = ROOT / "savedresults"
    if not sr.exists():
        return None
    dirs = sorted(sr.glob("backtest_*"), reverse=True)
    return dirs[0] if dirs else None


# ── Message formatters ────────────────────────────────────────────────────────

def format_test() -> str:
    """Simple connectivity test message."""
    import socket
    return (
        f"✅ <b>Regime Trader — connexion OK</b>\n"
        f"Machine : <code>{socket.gethostname()}</code>\n"
        f"Heure   : {_now()}"
    )


def format_backtest_summary() -> str:
    """
    4-line summary from the latest savedresults/backtest_*/performance_summary.csv
    """
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat de backtest trouvé dans savedresults/"

    csv = d / "performance_summary.csv"
    ctx = d / "run_context.json"

    if not csv.exists():
        return f"❌ performance_summary.csv introuvable dans {d.name}"

    import pandas as pd
    s = pd.read_csv(csv, header=None, index_col=0).squeeze()

    # Asset group + symbols
    group = "—"
    symbols = str(s.get("symbols", "—"))
    if ctx.exists():
        try:
            c = json.loads(ctx.read_text())
            group   = c.get("asset_group", "—")
            symbols = ", ".join(c.get("symbols", []))
        except Exception:
            pass

    ret    = float(s.get("total_return", 0))
    sharpe = float(s.get("sharpe", 0))
    dd     = float(s.get("max_drawdown", 0))
    calmar = float(s.get("calmar", 0))
    start  = str(s.get("start", ""))
    end    = str(s.get("end", ""))
    folds  = int(float(s.get("n_folds", 0)))

    return (
        f"📊 <b>Backtest — {group}</b>\n"
        f"<code>{symbols}</code>\n"
        f"Période  : {start} → {end}  ({folds} folds)\n"
        f"Retour   : <b>{_pct(ret)}</b>   Sharpe <b>{sharpe:.2f}</b>\n"
        f"MaxDD    : {_pct(dd)}   Calmar {calmar:.2f}\n"
        f"<i>{_now()}</i>"
    )


def format_latest_trades(n: int = 5) -> str:
    """
    Last N trades from the latest backtest trade_log.csv
    """
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat trouvé."

    tlog = d / "trade_log.csv"
    if not tlog.exists():
        return "❌ trade_log.csv introuvable."

    import pandas as pd
    df = pd.read_csv(tlog)
    if df.empty:
        return "ℹ️ Aucun trade enregistré."

    # Detect return column
    ret_col = next((c for c in ["pnl_pct", "return", "pct_return", "trade_return"] if c in df.columns), None)
    sym_col = next((c for c in ["symbol", "ticker", "sym"] if c in df.columns), None)
    date_col = next((c for c in ["exit_date", "date", "entry_date"] if c in df.columns), None)

    ctx = d / "run_context.json"
    group = "—"
    if ctx.exists():
        try:
            group = json.loads(ctx.read_text()).get("asset_group", "—")
        except Exception:
            pass

    lines = [f"📈 <b>Derniers trades — {group}</b>"]
    for _, row in df.tail(n).iterrows():
        sym  = str(row[sym_col])  if sym_col  else "?"
        date = str(row[date_col])[:10] if date_col else "?"
        if ret_col:
            r = float(row[ret_col])
            icon = "🟢" if r >= 0 else "🔴"
            lines.append(f"  {icon} {sym:<6} {_pct(r)}   {date}")
        else:
            lines.append(f"  • {sym}   {date}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_stress_summary() -> str:
    """
    Summary from latest stress_test_summary.csv
    """
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat trouvé."

    stress = d / "stress_test_summary.csv"
    if not stress.exists():
        # Try finding any stress file
        all_stress = sorted(ROOT.glob("savedresults/backtest_*/stress_test_summary.csv"), reverse=True)
        if not all_stress:
            return "❌ Aucun stress test trouvé. Lancez d'abord: python main.py stress"
        stress = all_stress[0]

    import pandas as pd
    df = pd.read_csv(stress, index_col=0)

    ctx = (stress.parent) / "run_context.json"
    group = "—"
    if ctx.exists():
        try:
            group = json.loads(ctx.read_text()).get("asset_group", "—")
        except Exception:
            pass

    lines = [f"⚡ <b>Stress Test — {group}</b>"]
    for scenario, row in df.iterrows():
        sharpe = row.get("sharpe", "?")
        dd     = row.get("max_drawdown", "?")
        delta  = row.get("vs_baseline_equity", "?")
        icon   = "✅" if float(str(sharpe).replace(",", ".")) > 0 else "❌"
        lines.append(f"  {icon} <code>{scenario:<22}</code> Sharpe {sharpe}  DD {dd}  Δeq {delta}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_regime_status() -> str:
    """
    Current regime from latest regime_history.csv (last bar)
    """
    d = _latest_backtest_dir()
    if d is None:
        return "❌ Aucun résultat trouvé."

    rh = d / "regime_history.csv"
    if not rh.exists():
        return "❌ regime_history.csv introuvable."

    import pandas as pd
    df = pd.read_csv(rh, index_col=0)
    if df.empty:
        return "ℹ️ Historique de régime vide."

    last_date   = str(df.index[-1])[:10]
    last_regime = str(df.iloc[-1, 0])

    regime_icons = {
        "BULL":     "🟢", "EUPHORIA": "🚀",
        "BEAR":     "🔴", "CRASH":    "💥",
        "NEUTRAL":  "⚪", "UNKNOWN":  "❓",
    }
    icon = regime_icons.get(last_regime.upper(), "📊")

    # Count last 10 bars
    recent = df.iloc[-10:, 0].tolist()
    recent_str = " ".join(r[:3] for r in recent)

    return (
        f"{icon} <b>Régime actuel : {last_regime}</b>\n"
        f"Date     : {last_date}\n"
        f"Récent   : <code>{recent_str}</code>\n"
        f"<i>{_now()}</i>"
    )
