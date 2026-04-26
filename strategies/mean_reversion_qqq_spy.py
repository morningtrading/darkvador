"""
strategies/mean_reversion_qqq_spy.py
====================================

Mean-reversion pair strategy on the QQQ/SPY ratio. Long-only. No leverage.

The idea
--------
The log(QQQ/SPY) ratio drifts upward over time (tech outperforms broad market)
but mean-reverts around that drift on a multi-week horizon. We compute the
residual against a long-window drift, z-score the residual on a short window,
and tilt the QQQ/SPY allocation accordingly:

    z > +threshold : QQQ overpriced vs drift → tilt toward SPY
    z < -threshold : QQQ underpriced vs drift → tilt toward QQQ
    -threshold ≤ z ≤ +threshold : neutral 50/50

Total deployed capital = ``allocation`` (e.g. 0.30 of NAV); the split between
QQQ and SPY moves with the z-score. Zero short, leverage cap 1.0×.

Validation goals (from POSSIBLEUPDATES + the strategy proposal):
- Sharpe ≥ 0.3 standalone on 2020-2026
- Correlation < 0.3 with the hmm_regime equity curve
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MeanReversionQqqSpyConfig:
    """All knobs of the strategy in one place."""
    allocation:      float = 0.30   # total capital deployed (sum of QQQ + SPY weights)
    lookback:        int   = 30     # z-score window in trading days
    drift_lookback:  int   = 252    # long-window drift removal (1 year)
    threshold:       float = 1.5    # |z| above which we tilt
    max_tilt:        float = 0.40   # max single-leg deviation from 0.5 (so weights ∈ [0.10, 0.90])


class MeanReversionQqqSpyStrategy:
    """
    Stand-alone implementation, decoupled from the regime_strategies framework.

    Use generate_target_weights() in a backtest to obtain a per-day
    DataFrame[date, weight_qqq, weight_spy].

    Deliberately not a BaseStrategy subclass yet — the existing framework's
    generate_signal(symbol, bars, regime_state) interface is single-symbol,
    which doesn't fit a paired-asset signal cleanly. We integrate into the
    multi-strategy framework only once the standalone numbers justify it.
    """

    def __init__(self, config: Optional[MeanReversionQqqSpyConfig] = None) -> None:
        self.cfg = config or MeanReversionQqqSpyConfig()

    # ── Core signal ────────────────────────────────────────────────────────────

    def compute_z_score(
        self,
        qqq_close: pd.Series,
        spy_close: pd.Series,
    ) -> pd.Series:
        """Rolling z-score of log(QQQ/SPY) detrended by a long-window mean.
        Aligns the two series on their shared index first."""
        common = qqq_close.index.intersection(spy_close.index)
        qqq = qqq_close.loc[common].astype(float)
        spy = spy_close.loc[common].astype(float)
        ratio = np.log(qqq / spy)
        drift = ratio.rolling(self.cfg.drift_lookback, min_periods=self.cfg.drift_lookback).mean()
        residual = ratio - drift
        mu = residual.rolling(self.cfg.lookback, min_periods=self.cfg.lookback).mean()
        sd = residual.rolling(self.cfg.lookback, min_periods=self.cfg.lookback).std()
        z = (residual - mu) / sd
        z.name = "z_score"
        return z

    def _z_to_weights(self, z: float) -> tuple[float, float]:
        """Map a single z-score value to (qqq_weight, spy_weight)."""
        if not np.isfinite(z):
            return 0.0, 0.0
        # Soft-clip z to [-2, +2], then compute tilt direction.
        # tilt > 0 means QQQ is expensive → reduce QQQ weight.
        clipped = max(-2.0, min(2.0, z))
        tilt = clipped / 2.0  # in [-1, +1]
        # 50/50 ± max_tilt depending on z direction.
        qqq_weight = self.cfg.allocation * (0.5 - self.cfg.max_tilt * tilt)
        spy_weight = self.cfg.allocation * (0.5 + self.cfg.max_tilt * tilt)
        # Guarantee non-negativity (paranoia — math above never produces negatives
        # given allocation >= 0 and max_tilt <= 0.5).
        return max(0.0, qqq_weight), max(0.0, spy_weight)

    # ── Backtest API ────────────────────────────────────────────────────────────

    def generate_target_weights(
        self,
        qqq_close: pd.Series,
        spy_close: pd.Series,
    ) -> pd.DataFrame:
        """
        Per-day target weights for the QQQ + SPY pair.

        Returns
        -------
        DataFrame indexed by date with columns ['weight_qqq', 'weight_spy', 'z'].
        Rows where the z-score isn't yet defined (warmup period) get zero weights.
        """
        z = self.compute_z_score(qqq_close, spy_close)
        rows = []
        for ts, z_val in z.items():
            wq, ws = self._z_to_weights(z_val)
            rows.append({"weight_qqq": wq, "weight_spy": ws, "z": z_val})
        df = pd.DataFrame(rows, index=z.index)
        return df
