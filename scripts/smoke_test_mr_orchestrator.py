"""scripts/smoke_test_mr_orchestrator.py — instantiate the
MeanReversionOrchestrator with synthetic bars and verify it returns
well-formed signals."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.mean_reversion_orchestrator import MeanReversionOrchestrator


def main() -> int:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    np.random.seed(42)
    qqq = 400 + np.cumsum(np.random.normal(0.05, 1.0, 400))
    spy = 500 + np.cumsum(np.random.normal(0.04, 0.8, 400))

    def mk(close):
        return pd.DataFrame({
            "open":   close,
            "high":   close * 1.01,
            "low":    close * 0.99,
            "close":  close,
            "volume": [1e6] * len(close),
        }, index=dates)

    bars = {"QQQ": mk(qqq), "SPY": mk(spy)}

    orch = MeanReversionOrchestrator(config={}, allocation=0.30)
    sigs = orch.generate_signals(
        symbols=["QQQ", "SPY"], bars=bars, regime_state=None,
    )
    print(f"Got {len(sigs)} signals:")
    for s in sigs:
        print(
            f"  {s.symbol:<5} "
            f"dir={s.direction.value:<4} "
            f"size={s.position_size_pct:.2%}  "
            f"stop=${s.stop_loss:.2f}  "
            f"conf={s.confidence:.2f}  "
            f"reasoning: {s.reasoning}"
        )
    return 0 if len(sigs) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
