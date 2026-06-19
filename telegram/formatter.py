"""
telegram/formatter.py - Compact Telegram messages for Regime Trader.

Objective
---------
Build short Telegram-safe HTML messages with an explicit account-mode label.

Rationale
---------
Trading notifications must always say whether they refer to paper trading or
live cash trading. Ambiguous alerts are dangerous when the same user may operate
paper and live environments.

Dependencies
------------
Reads config/settings.yaml for broker.paper_trading and savedresults/ for
backtest summaries.

Expected output
---------------
Small HTML-formatted Telegram messages.

How to test
-----------
python -m py_compile telegram/formatter.py
python telegram/hooks.py test
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


ROOT = Path(__file__).resolve().parent.parent
_HOST = socket.gethostname()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")


def _pct(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{v * 100:.2f}%"


def account_label() -> str:
    settings_path = ROOT / "config" / "settings.yaml"
    if not settings_path.exists():
        return "ACCOUNT MODE UNKNOWN"
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    paper_trading = raw.get("broker", {}).get("paper_trading")
    if paper_trading is True:
        return "PAPER ACCOUNT - not live cash"
    if paper_trading is False:
        return "LIVE CASH ACCOUNT"
    return "ACCOUNT MODE UNKNOWN"


def _account_line() -> str:
    return f"Account: <b>{account_label()}</b>"


def _latest_backtest_dir() -> Optional[Path]:
    savedresults = ROOT / "savedresults"
    if not savedresults.exists():
        return None
    dirs = sorted(savedresults.glob("backtest_*"), reverse=True)
    return dirs[0] if dirs else None


def format_test() -> str:
    return (
        "<b>Regime Trader - connection OK</b>\n"
        f"{_account_line()}\n"
        f"Machine: <code>{_HOST}</code> - {_now()}"
    )


def format_backtest_summary() -> str:
    directory = _latest_backtest_dir()
    if directory is None:
        return f"No backtest result found.\n{_account_line()}"

    csv_path = directory / "performance_summary.csv"
    ctx_path = directory / "run_context.json"
    if not csv_path.exists():
        return f"performance_summary.csv not found in {directory.name}\n{_account_line()}"

    import pandas as pd

    summary = pd.read_csv(csv_path, header=None, index_col=0).squeeze()
    group = "-"
    symbols = str(summary.get("symbols", "-"))
    cfg_set = ""
    if ctx_path.exists():
        try:
            context = json.loads(ctx_path.read_text(encoding="utf-8"))
            group = context.get("asset_group", "-")
            symbols = ", ".join(context.get("symbols", []))
            cfg_set = context.get("config_set", "")
        except Exception:
            pass

    total_return = float(summary.get("total_return", 0))
    cagr = float(summary.get("cagr", 0))
    sharpe = float(summary.get("sharpe", 0))
    max_drawdown = float(summary.get("max_drawdown", 0))
    calmar = float(summary.get("calmar", 0))
    trades = int(float(summary.get("total_trades", 0)))
    win_rate = float(summary.get("win_rate", 0))
    folds = int(float(summary.get("n_folds", 0)))
    start = str(summary.get("start", ""))[:10]
    end = str(summary.get("end", ""))[:10]

    set_str = f" [{cfg_set}]" if cfg_set else ""
    return (
        f"<b>Backtest - {group}</b>{set_str} <code>{_HOST}</code>\n"
        f"{_account_line()}\n"
        f"<code>{symbols}</code> - {start}->{end} ({folds} folds)\n"
        f"<b>{_pct(total_return)}</b> CAGR {_pct(cagr)} - Sharpe <b>{sharpe:.2f}</b> "
        f"Calmar {calmar:.2f} MaxDD {_pct(max_drawdown)}\n"
        f"{trades} trades - {win_rate * 100:.1f}% win - <i>{_now()}</i>"
    )


def format_latest_trades(n: int = 5) -> str:
    directory = _latest_backtest_dir()
    if directory is None:
        return f"No result found.\n{_account_line()}"

    trade_log = directory / "trade_log.csv"
    if not trade_log.exists():
        return f"trade_log.csv not found.\n{_account_line()}"

    import pandas as pd

    df = pd.read_csv(trade_log)
    if df.empty:
        return f"No trade recorded.\n{_account_line()}"

    ret_col = next((c for c in ["pnl_pct", "return", "pct_return", "trade_return"] if c in df.columns), None)
    sym_col = next((c for c in ["symbol", "ticker", "sym"] if c in df.columns), None)
    date_col = next((c for c in ["exit_date", "date", "entry_date"] if c in df.columns), None)

    ctx_path = directory / "run_context.json"
    group = "-"
    if ctx_path.exists():
        try:
            group = json.loads(ctx_path.read_text(encoding="utf-8")).get("asset_group", "-")
        except Exception:
            pass

    lines = [
        f"<b>Latest trades - {group}</b> <code>{_HOST}</code>",
        _account_line(),
    ]
    for _, row in df.tail(n).iterrows():
        symbol = str(row[sym_col]) if sym_col else "?"
        date = str(row[date_col])[:10] if date_col else "?"
        if ret_col:
            trade_return = float(row[ret_col])
            sign = "WIN" if trade_return >= 0 else "LOSS"
            lines.append(f"{sign} <code>{symbol}</code> {_pct(trade_return)} {date}")
        else:
            lines.append(f"<code>{symbol}</code> {date}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_stress_summary() -> str:
    directory = _latest_backtest_dir()
    stress = None
    if directory:
        candidate = directory / "stress_test_summary.csv"
        if candidate.exists():
            stress = candidate
    if stress is None:
        all_stress = sorted(ROOT.glob("savedresults/backtest_*/stress_test_summary.csv"), reverse=True)
        stress = all_stress[0] if all_stress else None
    if stress is None:
        return f"No stress test found.\n{_account_line()}"

    import pandas as pd

    df = pd.read_csv(stress, index_col=0)
    ctx_path = stress.parent / "run_context.json"
    group = "-"
    if ctx_path.exists():
        try:
            group = json.loads(ctx_path.read_text(encoding="utf-8")).get("asset_group", "-")
        except Exception:
            pass

    lines = [
        f"<b>Stress Test - {group}</b> <code>{_HOST}</code>",
        _account_line(),
    ]
    for scenario, row in df.iterrows():
        sharpe = row.get("sharpe", "?")
        max_drawdown = row.get("max_drawdown", "?")
        status = "OK" if float(str(sharpe).replace(",", ".")) > 0 else "FAIL"
        lines.append(f"{status} <code>{scenario}</code> Sh {sharpe} DD {max_drawdown}")

    lines.append(f"<i>{_now()}</i>")
    return "\n".join(lines)


def format_regime_status() -> str:
    directory = _latest_backtest_dir()
    if directory is None:
        return f"No result found.\n{_account_line()}"

    regime_history = directory / "regime_history.csv"
    if not regime_history.exists():
        return f"regime_history.csv not found.\n{_account_line()}"

    import pandas as pd

    df = pd.read_csv(regime_history, index_col=0)
    if df.empty:
        return f"Regime history is empty.\n{_account_line()}"

    last_date = str(df.index[-1])[:10]
    last_regime = str(df.iloc[-1, 0])
    recent = " ".join(str(r)[:3] for r in df.iloc[-8:, 0].tolist())
    return (
        f"<b>Regime: {last_regime}</b> {last_date} <code>{_HOST}</code>\n"
        f"{_account_line()}\n"
        f"Recent: <code>{recent}</code> - <i>{_now()}</i>"
    )
