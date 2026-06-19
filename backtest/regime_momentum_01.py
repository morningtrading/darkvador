"""
regime_momentum_01.py -- Intraday regime-gated momentum backtester.

Objective
---------
Test a tradable intraday strategy where the HMM controls risk budget and
momentum/SMA filters control entries and exits.

Rationale
---------
Regime detection alone only sizes risk. This module adds a concrete signal:
rank liquid Alpaca assets by risk-adjusted momentum, hold the top names only
when trend filters are positive, and reduce exposure in high-volatility regimes.

Dependencies
------------
AlpacaClient for real historical bars, FeatureEngineer for causal features,
and HMMEngine for walk-forward filtered regime inference.

Expected output
---------------
CSV files under savedresults/regime_momentum_01_<timestamp>/ containing equity,
weights, trades, fold metadata, and summary metrics.

How to test
-----------
python main.py regime-momentum --asset-group alpaca_top20 --start 2024-01-01
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from broker.alpaca_client import AlpacaClient
from core.hmm_engine import HMMEngine, RegimeState
from data.feature_engineering import FeatureEngineer


HMM_FEATURES = [
    "log_ret_1",
    "realized_vol_20",
    "vol_ratio",
    "adx_14",
    "dist_sma200",
]


@dataclass
class RegimeMomentumConfig:
    strategy_id: str = "regime_momentum_01"
    timeframe: str = "1Hour"
    data_feed: str = "sip"
    adjustment: str = "all"
    regime_proxy: str = "QQQ"
    initial_capital: float = 100_000.0
    top_n: int = 4
    trend_ma: int = 200
    momentum_fast: int = 24
    momentum_slow: int = 120
    vol_window: int = 120
    atr_window: int = 14
    rebalance_interval: int = 4
    slippage_pct: float = 0.0005
    train_window: int = 520
    test_window: int = 130
    step_size: int = 130
    min_train_bars: int = 390
    max_single_position: float = 0.15
    cash_rate: float = 0.0
    n_candidates: List[int] = field(default_factory=lambda: [5])
    n_init: int = 5
    stability_bars: int = 6
    flicker_window: int = 20
    flicker_threshold: int = 5
    min_confidence: float = 0.60
    low_vol_allocation: float = 0.90
    mid_vol_allocation: float = 0.55
    high_vol_allocation: float = 0.15
    unconfirmed_allocation_mult: float = 0.50
    live_loop_interval_seconds: int = 900
    live_history_days: int = 540
    live_min_notional: float = 250.0
    live_data_feed: str = "iex"
    live_telegram_enabled: bool = True
    live_telegram_status_every_cycles: int = 1

    @classmethod
    def from_dict(cls, raw: Optional[Dict]) -> "RegimeMomentumConfig":
        data = dict(raw or {})
        if "n_candidates" in data and isinstance(data["n_candidates"], tuple):
            data["n_candidates"] = list(data["n_candidates"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RegimeMomentumResult:
    output_dir: Path
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    trades: pd.DataFrame
    folds: pd.DataFrame
    summary: Dict[str, float]


class RegimeMomentumBacktester:
    """Walk-forward intraday backtester for regime-gated momentum."""

    def __init__(self, config: RegimeMomentumConfig) -> None:
        self.config = config

    def fetch_bars(
        self,
        symbols: List[str],
        start: str,
        end: Optional[str],
        paper: bool = True,
        data_feed: str = "iex",
    ) -> Dict[str, pd.DataFrame]:
        client = AlpacaClient(paper=paper, data_feed=data_feed)
        client.connect()
        try:
            raw = client.get_bars(
                symbols=symbols,
                timeframe=self.config.timeframe,
                start=start,
                end=end,
                adjustment=self.config.adjustment,
            )
        finally:
            client.disconnect()

        if raw.empty:
            raise RuntimeError("Alpaca returned no bars for regime_momentum_01.")
        return self._split_bars(raw, symbols)

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        symbols: List[str],
        output_root: Path,
    ) -> RegimeMomentumResult:
        cfg = self.config
        selected_symbols = [s for s in symbols if s in bars_by_symbol]
        if cfg.regime_proxy not in selected_symbols:
            selected_symbols = [cfg.regime_proxy] + selected_symbols
        selected_symbols = list(dict.fromkeys(selected_symbols))

        close = self._close_matrix(bars_by_symbol, selected_symbols)
        if close.empty:
            raise RuntimeError("No common close-price matrix for selected symbols.")

        proxy_bars = bars_by_symbol.get(cfg.regime_proxy)
        if proxy_bars is None:
            raise RuntimeError(f"Missing regime proxy bars: {cfg.regime_proxy}")

        feature_engineer = FeatureEngineer(
            zscore_window=60,
            vol_window=20,
            sma_long=cfg.trend_ma,
            sma_trend=50,
            volume_norm_window=50,
        )
        features = feature_engineer.build_feature_matrix(
            proxy_bars,
            feature_names=HMM_FEATURES,
        )

        common_index = close.index.intersection(features.index)
        close = close.reindex(common_index).dropna(how="any")
        features = features.reindex(close.index).dropna()
        common_index = close.index.intersection(features.index)
        close = close.reindex(common_index)
        features = features.reindex(common_index)

        if len(features) < cfg.train_window + cfg.test_window:
            raise RuntimeError(
                "Insufficient clean intraday bars for walk-forward. "
                f"Need at least {cfg.train_window + cfg.test_window}; got {len(features)}."
            )

        equity_parts: List[pd.Series] = []
        regime_parts: List[pd.Series] = []
        weight_rows: List[Dict] = []
        trade_rows: List[Dict] = []
        fold_rows: List[Dict] = []

        cash = cfg.initial_capital
        shares = {sym: 0.0 for sym in selected_symbols}
        last_weights = {sym: 0.0 for sym in selected_symbols}
        fold_id = 0
        start_i = 0

        while start_i + cfg.train_window + cfg.test_window <= len(features):
            fold_id += 1
            train_slice = features.iloc[start_i:start_i + cfg.train_window]
            test_start = start_i + cfg.train_window
            test_end = test_start + cfg.test_window
            context_slice = features.iloc[start_i:test_end]
            test_index = features.index[test_start:test_end]

            engine = HMMEngine(
                n_candidates=cfg.n_candidates,
                n_init=cfg.n_init,
                min_train_bars=cfg.min_train_bars,
                stability_bars=cfg.stability_bars,
                flicker_window=cfg.flicker_window,
                flicker_threshold=cfg.flicker_threshold,
                min_confidence=cfg.min_confidence,
            )
            engine.fit(train_slice.values)
            regime_states = engine.predict_regime_filtered(
                context_slice.values,
                timestamps=list(context_slice.index),
            )[-len(test_index):]
            vol_ranks = self._regime_vol_ranks(engine)

            fold_equity: Dict[pd.Timestamp, float] = {}
            fold_regimes: Dict[pd.Timestamp, str] = {}

            for bar_number, ts in enumerate(test_index):
                prices = close.loc[ts]
                equity = cash + sum(shares[sym] * float(prices[sym]) for sym in selected_symbols)
                regime_state = regime_states[bar_number]
                target_weights = self._target_weights(
                    close=close,
                    ts=ts,
                    symbols=selected_symbols,
                    regime_state=regime_state,
                    vol_ranks=vol_ranks,
                )

                if bar_number % cfg.rebalance_interval == 0:
                    cash, shares, fills = self._rebalance(
                        cash=cash,
                        shares=shares,
                        prices=prices,
                        target_weights=target_weights,
                        equity=equity,
                    )
                    for fill in fills:
                        fill.update(
                            {
                                "timestamp": ts,
                                "fold_id": fold_id,
                                "regime": regime_state.label,
                            }
                        )
                        trade_rows.append(fill)
                    last_weights = target_weights

                equity = cash + sum(shares[sym] * float(prices[sym]) for sym in selected_symbols)
                fold_equity[ts] = equity
                fold_regimes[ts] = regime_state.label
                row = {
                    "timestamp": ts,
                    "fold_id": fold_id,
                    "regime": regime_state.label,
                    "regime_probability": regime_state.probability,
                    "equity": equity,
                }
                row.update({f"weight_{sym}": last_weights.get(sym, 0.0) for sym in selected_symbols})
                weight_rows.append(row)

            equity_parts.append(pd.Series(fold_equity, name="equity"))
            regime_parts.append(pd.Series(fold_regimes, name="regime"))
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_slice.index[0],
                    "train_end": train_slice.index[-1],
                    "test_start": test_index[0],
                    "test_end": test_index[-1],
                    "n_states": engine._n_states,
                    "bars": len(test_index),
                }
            )
            start_i += cfg.step_size

        equity = pd.concat(equity_parts).sort_index()
        regimes = pd.concat(regime_parts).sort_index()
        returns = equity.pct_change().fillna(0.0)
        weights = pd.DataFrame(weight_rows).set_index("timestamp").sort_index()
        trades = pd.DataFrame(trade_rows)
        folds = pd.DataFrame(fold_rows)
        summary = self._summary(equity, returns, trades)

        output_dir = self._save_outputs(
            output_root=output_root,
            equity=equity,
            returns=returns,
            weights=weights,
            regimes=regimes,
            trades=trades,
            folds=folds,
            summary=summary,
            symbols=selected_symbols,
        )
        return RegimeMomentumResult(
            output_dir=output_dir,
            equity=equity,
            returns=returns,
            weights=weights,
            regimes=regimes,
            trades=trades,
            folds=folds,
            summary=summary,
        )

    def _target_weights(
        self,
        close: pd.DataFrame,
        ts: pd.Timestamp,
        symbols: List[str],
        regime_state: RegimeState,
        vol_ranks: Dict[int, float],
    ) -> Dict[str, float]:
        cfg = self.config
        history = close.loc[:ts]
        if len(history) < max(cfg.trend_ma, cfg.momentum_slow, cfg.vol_window) + 1:
            return {sym: 0.0 for sym in symbols}

        budget = self._regime_budget(regime_state, vol_ranks)
        scores = []
        for sym in symbols:
            series = history[sym].dropna()
            if len(series) < max(cfg.trend_ma, cfg.momentum_slow, cfg.vol_window) + 1:
                continue
            price = float(series.iloc[-1])
            trend_ma = float(series.iloc[-cfg.trend_ma:].mean())
            fast_mom = price / float(series.iloc[-cfg.momentum_fast]) - 1.0
            slow_mom = price / float(series.iloc[-cfg.momentum_slow]) - 1.0
            vol = float(series.pct_change().iloc[-cfg.vol_window:].std())
            if price <= trend_ma or fast_mom <= 0.0 or slow_mom <= 0.0 or vol <= 0.0:
                continue
            score = slow_mom / vol
            scores.append((score, sym))

        scores.sort(reverse=True)
        selected = [sym for _, sym in scores[:cfg.top_n]]
        weights = {sym: 0.0 for sym in symbols}
        if not selected or budget <= 0.0:
            return weights
        per_symbol = min(cfg.max_single_position, budget / len(selected))
        for sym in selected:
            weights[sym] = per_symbol
        return weights

    def _regime_budget(
        self,
        regime_state: RegimeState,
        vol_ranks: Dict[int, float],
    ) -> float:
        cfg = self.config
        rank = vol_ranks.get(regime_state.state_id, 1.0)
        if rank <= 0.33:
            budget = cfg.low_vol_allocation
        elif rank >= 0.67:
            budget = cfg.high_vol_allocation
        else:
            budget = cfg.mid_vol_allocation
        if (
            regime_state.probability < cfg.min_confidence
            or not regime_state.is_confirmed
        ):
            budget *= cfg.unconfirmed_allocation_mult
        return float(max(0.0, min(1.0, budget)))

    def _rebalance(
        self,
        cash: float,
        shares: Dict[str, float],
        prices: pd.Series,
        target_weights: Dict[str, float],
        equity: float,
    ) -> tuple[float, Dict[str, float], List[Dict]]:
        cfg = self.config
        fills: List[Dict] = []
        new_shares = dict(shares)
        for sym, target_weight in target_weights.items():
            price = float(prices[sym])
            target_value = equity * target_weight
            current_value = new_shares.get(sym, 0.0) * price
            delta_value = target_value - current_value
            if abs(delta_value) < max(10.0, equity * 0.0001):
                continue
            side = "buy" if delta_value > 0 else "sell"
            fill_price = price * (1.0 + cfg.slippage_pct if side == "buy" else 1.0 - cfg.slippage_pct)
            delta_shares = delta_value / fill_price
            cash -= delta_shares * fill_price
            new_shares[sym] = new_shares.get(sym, 0.0) + delta_shares
            fills.append(
                {
                    "symbol": sym,
                    "side": side,
                    "price": fill_price,
                    "shares": delta_shares,
                    "target_weight": target_weight,
                    "notional": delta_shares * fill_price,
                }
            )
        return cash, new_shares, fills

    @staticmethod
    def _split_bars(raw: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if not isinstance(raw.index, pd.MultiIndex):
            raise RuntimeError("Expected Alpaca bars with MultiIndex(symbol, timestamp).")
        names = list(raw.index.names)
        sym_level = names.index("symbol") if "symbol" in names else 0
        for sym in symbols:
            try:
                df = raw.xs(sym, level=sym_level).sort_index()
            except KeyError:
                continue
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(cols) == 5:
                out[sym] = df[cols].astype(float)
        return out

    @staticmethod
    def _close_matrix(
        bars_by_symbol: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> pd.DataFrame:
        series = {
            sym: bars_by_symbol[sym]["close"]
            for sym in symbols
            if sym in bars_by_symbol and "close" in bars_by_symbol[sym]
        }
        return pd.DataFrame(series).sort_index().dropna(how="any")

    @staticmethod
    def _regime_vol_ranks(engine: HMMEngine) -> Dict[int, float]:
        infos = list(engine._regime_info.values())
        ordered = sorted(infos, key=lambda item: item.expected_volatility)
        n = len(ordered)
        return {
            info.regime_id: (idx / (n - 1) if n > 1 else 0.5)
            for idx, info in enumerate(ordered)
        }

    def _summary(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        bars_per_year = self._bars_per_year(self.config.timeframe)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        years = max(len(returns) / bars_per_year, 1e-9)
        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
        vol = float(returns.std(ddof=1) * np.sqrt(bars_per_year))
        sharpe = float((returns.mean() * bars_per_year) / vol) if vol > 0 else 0.0
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min())
        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": float(equity.iloc[-1]),
            "total_return": total_return,
            "cagr": cagr,
            "annualized_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trade_count": float(len(trades)),
            "bars": float(len(equity)),
        }

    @staticmethod
    def _bars_per_year(timeframe: str) -> float:
        mapping = {
            "15Min": 252.0 * 26.0,
            "1Hour": 252.0 * 6.5,
            "1Day": 252.0,
        }
        return mapping.get(timeframe, 252.0)

    def _save_outputs(
        self,
        output_root: Path,
        equity: pd.Series,
        returns: pd.Series,
        weights: pd.DataFrame,
        regimes: pd.Series,
        trades: pd.DataFrame,
        folds: pd.DataFrame,
        summary: Dict[str, float],
        symbols: List[str],
    ) -> Path:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = output_root / f"{self.config.strategy_id}_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": equity, "returns": returns}).to_csv(output_dir / "equity_curve.csv")
        weights.to_csv(output_dir / "weights.csv")
        regimes.to_frame("regime").to_csv(output_dir / "regimes.csv")
        trades.to_csv(output_dir / "trades.csv", index=False)
        folds.to_csv(output_dir / "folds.csv", index=False)
        pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
        return output_dir
