"""SCA_regime_momentum_01_paper_live.py

Objective
---------
Run one paper-trading rebalance cycle for the optimized Regime Momentum 01
strategy against Alpaca paper.

Rationale
---------
The backtest engine is research-only. This script bridges the optimized
walk-forward strategy to paper trading by computing current target weights from
real Alpaca SIP adjusted bars, reconciling Alpaca paper positions, and sending
orders only when explicitly launched with `--execute`.

Dependencies
------------
Alpaca paper credentials in config/credentials.yaml, project settings in
config/settings.yaml, and the existing virtual environment.

Expected output
---------------
Console summary plus CSV logs under savedresults/SCA_regime_momentum_01_live/.

How to test
-----------
.venv/bin/python scratchpad/SCA_regime_momentum_01_paper_live.py --once
.venv/bin/python scratchpad/SCA_regime_momentum_01_paper_live.py --once --execute
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.regime_momentum_01 import HMM_FEATURES, RegimeMomentumBacktester, RegimeMomentumConfig
from broker.alpaca_client import AlpacaClient
from core.hmm_engine import HMMEngine
from data.feature_engineering import FeatureEngineer


OUTPUT_DIR = ROOT / "savedresults" / "SCA_regime_momentum_01_live"
_STOP_REQUESTED = False
_CYCLE_COUNT = 0


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _symbols_from_group(group_name: str) -> list[str]:
    groups = _load_yaml(ROOT / "config" / "asset_groups.yaml").get("groups", {})
    group = groups.get(group_name)
    if not group:
        raise ValueError(f"Unknown asset group: {group_name}")
    return list(group["symbols"])


def _strategy_config() -> RegimeMomentumConfig:
    settings = _load_yaml(ROOT / "config" / "settings.yaml")
    return RegimeMomentumConfig.from_dict(settings.get("regime_momentum_01", {}))


def _feature_matrix(cfg: RegimeMomentumConfig, proxy_bars: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer(
        zscore_window=60,
        vol_window=20,
        sma_long=cfg.trend_ma,
        sma_trend=50,
        volume_norm_window=50,
    )
    return engineer.build_feature_matrix(proxy_bars, feature_names=HMM_FEATURES)


def _current_targets(
    cfg: RegimeMomentumConfig,
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
) -> tuple[pd.Timestamp, dict[str, float], str, float]:
    backtester = RegimeMomentumBacktester(cfg)
    selected_symbols = list(dict.fromkeys([cfg.regime_proxy] + symbols))
    close = backtester._close_matrix(bars_by_symbol, selected_symbols)
    features = _feature_matrix(cfg, bars_by_symbol[cfg.regime_proxy])

    common_index = close.index.intersection(features.index)
    close = close.reindex(common_index).dropna(how="any")
    features = features.reindex(close.index).dropna()
    common_index = close.index.intersection(features.index)
    close = close.reindex(common_index)
    features = features.reindex(common_index)

    if len(features) < cfg.train_window:
        raise RuntimeError(f"Need {cfg.train_window} clean bars, got {len(features)}")

    train = features.iloc[-cfg.train_window:]
    engine = HMMEngine(
        n_candidates=cfg.n_candidates,
        n_init=cfg.n_init,
        min_train_bars=cfg.min_train_bars,
        stability_bars=cfg.stability_bars,
        flicker_window=cfg.flicker_window,
        flicker_threshold=cfg.flicker_threshold,
        min_confidence=cfg.min_confidence,
    )
    engine.fit(train.values)
    regime_state = engine.predict_regime_filtered(
        train.values,
        timestamps=list(train.index),
    )[-1]
    vol_ranks = backtester._regime_vol_ranks(engine)
    ts = close.index[-1]
    weights = backtester._target_weights(
        close=close,
        ts=ts,
        symbols=selected_symbols,
        regime_state=regime_state,
        vol_ranks=vol_ranks,
    )
    return ts, weights, regime_state.label, float(regime_state.probability)


def _position_values(client: AlpacaClient) -> dict[str, float]:
    return {pos.symbol: float(pos.market_value) for pos in client.get_all_positions()}


def _position_quantities(client: AlpacaClient) -> dict[str, float]:
    return {pos.symbol: float(pos.qty) for pos in client.get_all_positions()}


def _submit_market_order(client: AlpacaClient, symbol: str, side: str, qty: int) -> str:
    if client._trading_client is None:
        raise RuntimeError("Alpaca trading client is not connected")
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = client._trading_client.submit_order(req)
    return str(order.id)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _telegram_enabled(cfg: RegimeMomentumConfig) -> bool:
    settings = _load_yaml(ROOT / "config" / "settings.yaml")
    master = settings.get("notifications", {}).get("telegram", {}).get("enabled", False)
    return bool(master and cfg.live_telegram_enabled)


def _telegram_account_label() -> str:
    return "ALPACA PAPER ACCOUNT - not live cash"


def _send_telegram(text: str, cfg: RegimeMomentumConfig) -> None:
    if not _telegram_enabled(cfg):
        return
    try:
        from telegram.bot import send

        if not send(text):
            _append_csv(
                OUTPUT_DIR / "SCA_paper_live_errors.csv",
                [
                    {
                        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "mode": "TELEGRAM",
                        "error": "telegram_send_returned_false",
                    }
                ],
            )
    except Exception as exc:
        _append_csv(
            OUTPUT_DIR / "SCA_paper_live_errors.csv",
            [
                {
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "mode": "TELEGRAM",
                    "error": repr(exc),
                }
            ],
        )


def _status_message(summary: dict[str, Any], cfg: RegimeMomentumConfig) -> str:
    icon = "OK" if summary["orders"] == 0 else "TRADE"
    return (
        f"<b>SCA Regime Momentum paper</b> {icon}\n"
        f"Account: <b>{_telegram_account_label()}</b>\n"
        f"Mode: <code>{summary['mode']}</code> | Market: <code>{summary['market_open']}</code>\n"
        f"Regime: <b>{summary['regime']}</b> p={summary['regime_probability']:.2f} "
        f"| Target gross {summary['gross_target_weight']:.0%}\n"
        f"Equity: ${summary['equity']:,.2f} | Orders: {summary['orders']} "
        f"| Every {cfg.live_loop_interval_seconds}s"
    )


def _orders_message(orders: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "<b>SCA paper orders submitted</b>",
        f"Account: <b>{_telegram_account_label()}</b>",
        f"Regime: <b>{summary['regime']}</b> | Equity: ${summary['equity']:,.2f}",
    ]
    for order in orders[:8]:
        lines.append(
            f"<code>{order['symbol']}</code> {order['side'].upper()} "
            f"{order['qty']} @ {order['latest_price']:.2f}"
        )
    if len(orders) > 8:
        lines.append(f"... +{len(orders) - 8} more")
    return "\n".join(lines)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    global _CYCLE_COUNT
    logging.getLogger("core.hmm_engine").setLevel(logging.ERROR)
    logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _strategy_config()
    cfg.data_feed = args.data_feed or cfg.live_data_feed
    cfg.adjustment = args.adjustment or cfg.adjustment
    history_days = args.history_days or cfg.live_history_days
    min_notional = args.min_notional if args.min_notional is not None else cfg.live_min_notional
    symbols = args.symbols.split(",") if args.symbols else _symbols_from_group(args.asset_group)
    symbols = [sym.strip().upper() for sym in symbols if sym.strip()]
    fetch_symbols = list(dict.fromkeys([cfg.regime_proxy] + symbols))

    client = AlpacaClient(paper=True, data_feed=cfg.data_feed)
    client.connect()
    try:
        account = client.get_account()
        clock = client.get_clock()
        is_open = bool(clock.is_open)

        end = dt.date.today().isoformat()
        start = (dt.date.today() - dt.timedelta(days=history_days)).isoformat()
        bars = RegimeMomentumBacktester(cfg).fetch_bars(
            fetch_symbols,
            start=start,
            end=end,
            paper=True,
            data_feed=cfg.data_feed,
        )
        ts, target_weights, regime, probability = _current_targets(cfg, bars, symbols)
        equity = float(account.equity)
        current_values = _position_values(client)
        current_qty = _position_quantities(client)

        orders = []
        for symbol in symbols:
            target_weight = float(target_weights.get(symbol, 0.0))
            target_value = equity * target_weight
            current_value = float(current_values.get(symbol, 0.0))
            delta_value = target_value - current_value
            if abs(delta_value) < min_notional:
                continue
            price = client.get_latest_price(symbol)
            if price <= 0:
                continue
            side = "buy" if delta_value > 0 else "sell"
            qty = int(abs(delta_value) / price)
            if side == "sell":
                qty = min(qty, int(abs(current_qty.get(symbol, 0.0))))
            if qty < 1:
                continue
            order_id = ""
            if args.execute and is_open:
                order_id = _submit_market_order(client, symbol, side, qty)
            orders.append(
                {
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "bar_timestamp": str(ts),
                    "execute": bool(args.execute),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "latest_price": price,
                    "target_weight": target_weight,
                    "current_value": current_value,
                    "target_value": target_value,
                    "delta_value": delta_value,
                    "regime": regime,
                    "regime_probability": probability,
                    "alpaca_order_id": order_id,
                }
            )

        summary = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "EXECUTE" if args.execute else "DRY_RUN",
            "market_open": is_open,
            "equity": equity,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "regime": regime,
            "regime_probability": probability,
            "bar_timestamp": str(ts),
            "orders": len(orders),
            "gross_target_weight": sum(float(target_weights.get(sym, 0.0)) for sym in symbols),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }
        print(summary)
        _append_csv(OUTPUT_DIR / "SCA_paper_live_cycles.csv", [summary])
        _CYCLE_COUNT += 1
        if orders:
            df = pd.DataFrame(orders)
            print(df[["symbol", "side", "qty", "latest_price", "target_weight", "delta_value", "alpaca_order_id"]].to_string(index=False))
            log_path = OUTPUT_DIR / "SCA_paper_live_orders.csv"
            _append_csv(log_path, orders)
            print(f"Order log: {log_path}")
            _send_telegram(_orders_message(orders, summary), cfg)
        else:
            print("No rebalance orders needed.")
        if args.execute and not is_open:
            print("Market is closed. Cycle logged; no paper orders submitted.")
        every = max(1, int(cfg.live_telegram_status_every_cycles))
        if _CYCLE_COUNT % every == 0:
            _send_telegram(_status_message(summary, cfg), cfg)
        return summary
    finally:
        client.disconnect()


def run_loop(args: argparse.Namespace) -> None:
    cfg = _strategy_config()
    interval = args.interval_seconds or cfg.live_loop_interval_seconds
    if interval < 60:
        raise ValueError("interval_seconds must be at least 60")
    print(
        f"Starting Regime Momentum 01 paper loop: interval={interval}s "
        f"execute={args.execute}"
    )
    while not _STOP_REQUESTED:
        try:
            run_once(args)
        except Exception as exc:
            row = {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "mode": "EXECUTE" if args.execute else "DRY_RUN",
                "error": repr(exc),
            }
            _append_csv(OUTPUT_DIR / "SCA_paper_live_errors.csv", [row])
            print(f"Cycle error: {exc!r}")
            _send_telegram(
                f"<b>SCA paper bot error</b>\n"
                f"Account: <b>{_telegram_account_label()}</b>\n"
                f"<code>{repr(exc)}</code>",
                cfg,
            )
        slept = 0
        while slept < interval and not _STOP_REQUESTED:
            step = min(5, interval - slept)
            time.sleep(step)
            slept += step
    print("Stop requested. Paper loop exited cleanly.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one rebalance cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--execute", action="store_true", help="Submit paper orders")
    parser.add_argument("--asset-group", default="alpaca_momentum10")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--data-feed", default=None, choices=["sip", "iex"])
    parser.add_argument("--adjustment", default=None, choices=["all", "raw", "split", "dividend"])
    parser.add_argument("--history-days", type=int, default=None)
    parser.add_argument("--min-notional", type=float, default=None)
    parser.add_argument("--interval-seconds", type=int, default=None)
    return parser


def main() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    args = build_parser().parse_args()
    if args.loop:
        run_loop(args)
        return
    if args.once:
        run_once(args)
        return
    raise SystemExit("Use --once or --loop for this guarded paper-live runner.")


if __name__ == "__main__":
    main()
