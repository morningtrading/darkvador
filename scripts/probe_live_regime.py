"""scripts/probe_live_regime.py — exercise the LIVE HMM training path
(same code as TradingSession.startup) and print the resulting regime
on the most recent training bar. Quick way to verify timeframe parity
with the backtest."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import (
    load_config, _resolve_symbols, _train_hmm,
)
from broker.alpaca_client import AlpacaClient


def main() -> int:
    cfg         = load_config()
    broker_cfg  = cfg["broker"]
    hmm_cfg     = cfg["hmm"]
    timeframe   = broker_cfg.get("timeframe", "1Day")
    symbols     = _resolve_symbols(cfg, asset_group=None, symbols_arg=None)
    proxy       = hmm_cfg.get("regime_proxy") or symbols[0]

    print(f"timeframe   : {timeframe}")
    print(f"symbols     : {symbols}")
    print(f"regime_proxy: {proxy}")
    print(f"n_candidates: {hmm_cfg.get('n_candidates')}")
    print()

    client = AlpacaClient(paper=True)
    client.connect(skip_live_confirm=True)

    print("Training HMM (will retrain — pickle deleted) ...")
    hmm_cfg_eff = {**hmm_cfg, "timeframe": timeframe}
    engine = _train_hmm(client, symbols, hmm_cfg_eff)
    print(f"  trained on {timeframe} bars: {engine._n_states} states, "
          f"BIC={engine._training_bic:.2f}")
    print()

    # Use the engine's own state-sequence on the training data:
    if not hasattr(engine, "_train_state_seq") or engine._train_state_seq is None:
        # fall back: predict on engine's stored features
        print("(no cached state sequence — running predict on training features)")
        states = engine.predict_regime_filtered(engine._train_features)
    else:
        states = engine._train_state_seq

    # Last 10 unique regime segments for context
    print("Last 5 training-data regime labels:")
    if hasattr(engine, "_train_features") and engine._train_features is not None:
        proba = engine.predict_regime_proba(engine._train_features)
        import numpy as np
        labels = [engine.get_state_label(int(np.argmax(p))) for p in proba[-5:]]
        for i, lbl in enumerate(labels, 1):
            print(f"  -{6-i}: {lbl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
