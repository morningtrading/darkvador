"""
telegram/notifier.py - Central notification dispatcher for Regime Trader.

Objective
---------
Route notification events to Telegram while keeping account mode explicit.

Rationale
---------
Every trading alert must state whether it belongs to paper trading or live cash
trading. The notifier never raises to trading code; failures are logged.

Dependencies
------------
telegram.bot.send and telegram.formatter helpers.

Expected output
---------------
Short Telegram-safe HTML messages.

How to test
-----------
python -m py_compile telegram/notifier.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_cfg: Optional[Dict] = None
_cli_override: Optional[bool] = None


def configure(enabled: Optional[bool] = None) -> None:
    global _cli_override
    _cli_override = enabled


def _load_cfg() -> Dict:
    global _cfg
    if _cfg is not None:
        return _cfg
    try:
        from pathlib import Path

        import yaml

        settings = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        if settings.exists():
            raw = yaml.safe_load(settings.read_text(encoding="utf-8")) or {}
            _cfg = raw.get("notifications", {}).get("telegram", {})
        else:
            _cfg = {}
    except Exception:
        _cfg = {}
    return _cfg


def _is_enabled(event: str) -> bool:
    if _cli_override is not None:
        return _cli_override
    cfg = _load_cfg()
    if not cfg.get("enabled", False):
        return False
    return cfg.get(f"on_{event}", True)


def notify(event: str, data: Optional[Dict[str, Any]] = None) -> None:
    if not _is_enabled(event):
        return
    try:
        text = _build_message(event, data or {})
        if text:
            from telegram.bot import send

            ok = send(text)
            if not ok:
                logger.warning("Telegram notification failed for event '%s'", event)
    except Exception as exc:
        logger.warning("Telegram notifier error for event '%s': %s", event, exc)


def _build_message(event: str, data: Dict[str, Any]) -> str:
    if event == "backtest":
        from telegram.formatter import format_backtest_summary

        return format_backtest_summary()

    if event == "stress":
        from telegram.formatter import format_stress_summary

        return format_stress_summary()

    if event == "regime_change":
        from telegram.formatter import _HOST, _now, account_label

        from_regime = data.get("from_regime", "?")
        to_regime = data.get("to_regime", "?")
        group = data.get("asset_group", "-")
        equity = data.get("equity")
        eq_str = f"${equity:,.0f}" if equity else "-"
        return (
            f"<b>Regime change: {from_regime} -> {to_regime}</b> {group} <code>{_HOST}</code>\n"
            f"Account: <b>{account_label()}</b>\n"
            f"Equity: {eq_str} - <i>{_now()}</i>"
        )

    if event == "trade":
        from telegram.formatter import _HOST, _now, _pct, account_label

        symbol = data.get("symbol", "?")
        side = str(data.get("side", "?")).upper()
        pnl_pct = data.get("pnl_pct")
        equity = data.get("equity")
        group = data.get("asset_group", "-")
        regime = data.get("regime", "?")
        pnl_str = f" {_pct(pnl_pct)}" if pnl_pct is not None else ""
        eq_str = f"${equity:,.0f}" if equity else "-"
        return (
            f"<b>TRADE {symbol} {side}</b>{pnl_str} {group}/{regime} <code>{_HOST}</code>\n"
            f"Account: <b>{account_label()}</b>\n"
            f"Equity: {eq_str} - <i>{_now()}</i>"
        )

    logger.warning("Unknown telegram event: '%s'", event)
    return ""
