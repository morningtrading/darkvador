"""
strategies/mean_reversion_orchestrator.py
==========================================

Drop-in replacement for ``core.regime_strategies.StrategyOrchestrator`` that
serves the mean-reversion QQQ/SPY pair strategy. Implements the same public
interface (``generate_signals``, ``update_weights``) so ``main.py`` can swap
it in without touching the rest of the live trading loop.

Why this is separate from StrategyOrchestrator
----------------------------------------------
StrategyOrchestrator is HMM-regime-driven and single-symbol-per-call inside.
Mean-reversion on a pair needs paired bars at signal time, no regime input,
and emits coordinated signals for both legs at once. The two control flows
don't fit the same orchestrator cleanly, so this is its own class.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

# Local imports kept lazy — let main.py import this orchestrator without
# pulling the entire regime_strategies module just for the Signal dataclass.

logger = logging.getLogger(__name__)


class MeanReversionOrchestrator:
    """
    Wraps a MeanReversionQqqSpyStrategy and exposes the
    StrategyOrchestrator interface used by TradingSession.run_loop.

    Parameters
    ----------
    config :
        Full bot config dict (merged settings.yaml + active set).
    allocation :
        Total portfolio fraction this orchestrator deploys across QQQ + SPY.
    lookback / drift_lookback / threshold / max_tilt :
        Forwarded to MeanReversionQqqSpyConfig.
    """

    def __init__(
        self,
        config: Dict,
        allocation:     float = 0.30,
        lookback:       int   = 30,
        drift_lookback: int   = 252,
        threshold:      float = 1.5,
        max_tilt:       float = 0.40,
    ) -> None:
        from strategies.mean_reversion_qqq_spy import (
            MeanReversionQqqSpyConfig,
            MeanReversionQqqSpyStrategy,
        )
        cfg = MeanReversionQqqSpyConfig(
            allocation     = allocation,
            lookback       = lookback,
            drift_lookback = drift_lookback,
            threshold      = threshold,
            max_tilt       = max_tilt,
        )
        self._strategy = MeanReversionQqqSpyStrategy(cfg)
        self._config   = config or {}
        self._current_weights: Dict[str, float] = {}

    # ── Public interface (matches StrategyOrchestrator) ───────────────────────

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Receive the latest live weights from the position tracker.
        Stored for rebalance-threshold filtering only."""
        self._current_weights = dict(weights or {})

    def generate_signals(
        self,
        symbols: List[str],
        bars: Dict[str, pd.DataFrame],
        regime_state=None,
        is_flickering: bool = False,
    ) -> List:
        """
        Build per-symbol Signals for the QQQ/SPY pair based on the current
        z-score of log(QQQ/SPY).

        ``regime_state`` and ``is_flickering`` are accepted for interface
        parity with StrategyOrchestrator but ignored — this strategy is
        regime-independent by design.
        """
        # Lazy import to avoid pulling Signal/Direction at module load
        from core.regime_strategies import Direction, Signal, _atr

        qqq = bars.get("QQQ")
        spy = bars.get("SPY")
        if qqq is None or spy is None:
            logger.debug("MR orchestrator: missing QQQ or SPY bars — no signals")
            return []
        # Need enough history for the drift_lookback warmup
        warmup = self._strategy.cfg.drift_lookback + self._strategy.cfg.lookback
        if len(qqq) < warmup or len(spy) < warmup:
            logger.debug(
                "MR orchestrator: only %d/%d bars — still warming up",
                min(len(qqq), len(spy)), warmup,
            )
            return []

        try:
            weights_df = self._strategy.generate_target_weights(
                qqq["close"], spy["close"],
            )
        except Exception as exc:
            logger.warning("MR orchestrator: weight calc failed: %s", exc)
            return []
        if weights_df.empty or pd.isna(weights_df["z"].iloc[-1]):
            return []

        last       = weights_df.iloc[-1]
        target_qqq = float(last["weight_qqq"])
        target_spy = float(last["weight_spy"])
        z          = float(last["z"])
        ts         = pd.Timestamp(qqq.index[-1])

        signals: List = []
        for sym, target in (("QQQ", target_qqq), ("SPY", target_spy)):
            if sym not in symbols:
                continue
            sym_bars = bars[sym]
            close = float(sym_bars["close"].iloc[-1])

            # ATR-based stop so the RiskManager accepts the signal.
            try:
                atr_val = _atr(
                    sym_bars["high"], sym_bars["low"], sym_bars["close"], period=14,
                )
            except Exception:
                atr_val = close * 0.02  # fallback 2 % stop

            stop = max(close * 0.90, close - 3.0 * atr_val)

            direction = Direction.LONG if target > 1e-4 else Direction.FLAT
            confidence = min(0.99, 0.50 + abs(z) / 4.0)
            regime_id   = getattr(regime_state, "state_id", 0)    if regime_state else 0
            regime_name = getattr(regime_state, "label", "N/A")    if regime_state else "N/A"
            regime_prob = getattr(regime_state, "probability", 0.0) if regime_state else 0.0

            signals.append(Signal(
                symbol             = sym,
                direction          = direction,
                confidence         = confidence,
                entry_price        = close,
                stop_loss          = stop,
                take_profit        = None,
                position_size_pct  = max(0.0, target),
                leverage           = 1.0,
                regime_id          = regime_id,
                regime_name        = regime_name,
                regime_probability = regime_prob,
                timestamp          = ts,
                reasoning          = (
                    f"MR pair (regime-independent): z={z:+.2f}, "
                    f"target={target:.2%} on {sym}"
                ),
                strategy_name      = "MeanReversionQqqSpy",
                metadata           = {
                    "z_score":    round(z, 4),
                    "weight_qqq": round(target_qqq, 4),
                    "weight_spy": round(target_spy, 4),
                },
            ))
        return signals
