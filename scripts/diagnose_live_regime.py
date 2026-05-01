"""scripts/diagnose_live_regime.py — explain why the live HMM is calling
the regime it does.

Loads the trained HMM (`models/hmm.pkl`), replays the same feature
pipeline the bot uses at startup (FeatureEngineer.build_feature_matrix
on the regime_proxy + VIX series), and prints:

  1. The latest 6-dim feature vector being fed to the model.
  2. Per-state training means (so you can compare).
  3. Per-state log-likelihood at the latest bar (which state best fits
     the live feature vector?).
  4. Per-feature contribution = (x - mu_state)² for each (feature, state)
     so we can pinpoint which dimension is firing the winning state.
  5. Forward-algorithm posterior over the last N bars — i.e. the actual
     reading the live bot would produce.

Run from repo root:
    .venv/bin/python scripts/diagnose_live_regime.py
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.feature_engineering import FeatureEngineer, hmm_feature_names  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df


def _maybe_fetch_vix(idx: pd.DatetimeIndex) -> pd.Series | None:
    try:
        v = yf.download("^VIX",
                        start=str(idx.min().date()),
                        end=str((idx.max() + pd.Timedelta(days=1)).date()),
                        progress=False, auto_adjust=True)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.droplevel(1)
        return v["Close"].reindex(idx).ffill()
    except Exception as e:
        print(f"  WARNING: VIX fetch failed ({e}) — feeding NaN")
        return None


def _bar(label: str, val: float, lo: float, hi: float, width: int = 30) -> str:
    """Return a tiny ascii bar showing where ``val`` sits in [lo, hi]."""
    if hi <= lo:
        return ""
    pos = int(round((val - lo) / (hi - lo) * (width - 1)))
    pos = max(0, min(width - 1, pos))
    cells = [" "] * width
    cells[pos] = "█"
    return "".join(cells)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    # 1. Load model + config
    pkl = ROOT / "models" / "hmm.pkl"
    if not pkl.exists():
        print(f"FAIL: {pkl} not found — run [1] Train HMM first.")
        return 1

    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    state_to_label: dict[int, str] = saved["state_to_label"]
    n_features: int = saved["n_features"]
    label_mode_train: str = saved["config"].get("label_mode", "sort")

    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text()) or {}
    hmm_cfg = cfg.get("hmm", {})
    feat_names = hmm_feature_names(hmm_cfg)
    proxy = hmm_cfg.get("regime_proxy") or "QQQ"

    print("=" * 78)
    print("  LIVE REGIME DIAGNOSTIC")
    print("=" * 78)
    print(f"  Model            : {pkl}")
    print(f"  Trained          : {saved['training_date']}")
    print(f"  Training BIC     : {saved['training_bic']:.1f}")
    print(f"  Label mode       : trained={label_mode_train!r} · "
          f"settings.yaml={hmm_cfg.get('label_mode')!r}")
    print(f"  N features       : {n_features}  · settings list: {feat_names}")
    print(f"  Regime proxy     : {proxy}")
    print(f"  State→label map  : {state_to_label}")
    print()

    if len(feat_names) != n_features:
        print(f"  ⚠ FEATURE COUNT MISMATCH: model expects {n_features}, "
              f"settings.yaml gives {len(feat_names)}. The live pipeline will "
              f"feed wrong data — that alone explains a bogus regime call.")
        print()

    # 2. Fetch live data: ~2y of QQQ + VIX so the rolling features warm up
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=730)
    print(f"  Fetching {proxy} OHLCV  {start.date()} → {end.date()} ...")
    ohlcv = _fetch_ohlcv(proxy, str(start.date()),
                         str((end + pd.Timedelta(days=1)).date()))
    if ohlcv.empty:
        print(f"  FAIL: yfinance returned no bars for {proxy}.")
        return 2
    print(f"  Got {len(ohlcv):,} bars · last close ${ohlcv['close'].iloc[-1]:.2f}"
          f"  ({ohlcv.index[-1].date()})")

    vix = None
    if hmm_cfg.get("use_vix_features", False):
        print(f"  Fetching ^VIX ...")
        vix = _maybe_fetch_vix(ohlcv.index)
        print(f"  VIX bars: {0 if vix is None else int(vix.notna().sum())}")
    print()

    # 3. Build the feature matrix exactly as the bot does
    fe = FeatureEngineer()
    feat_df = fe.build_feature_matrix(
        ohlcv, feature_names=feat_names, vix_series=vix,
    )
    if feat_df.empty:
        print("  FAIL: build_feature_matrix returned empty DataFrame.")
        return 3
    print(f"  Feature matrix built: {len(feat_df):,} clean rows × "
          f"{feat_df.shape[1]} cols")
    print()

    # ── 4. Latest feature vector ──────────────────────────────────────────
    last_row = feat_df.iloc[-1].values.astype(float)
    last_ts = feat_df.index[-1]
    print("─" * 78)
    print(f"  LATEST FEATURE VECTOR  (bar = {last_ts.date()})")
    print("─" * 78)
    for i, (name, val) in enumerate(zip(feat_df.columns, last_row)):
        recent = feat_df[name].tail(20)
        col_min, col_max = float(recent.min()), float(recent.max())
        col_mu = float(feat_df[name].mean())
        col_sd = float(feat_df[name].std())
        z = (val - col_mu) / col_sd if col_sd > 0 else 0.0
        print(f"  f{i}  {name:<22} = {val:+.4f}  "
              f"(z vs full window: {z:+.2f}σ, "
              f"20d range [{col_min:+.3f}, {col_max:+.3f}])")
    print()

    # ── 5. State means + likelihood breakdown ────────────────────────────
    means = model.means_                # (K, n_features)
    cov = model.covars_                  # (K, n_features, n_features)
    K = saved["n_states"]
    label_order = ["EUPHORIA", "BULL", "NEUTRAL", "BEAR", "CRASH"]
    state_order = [s for s, _ in sorted(state_to_label.items(),
                   key=lambda kv: label_order.index(kv[1])
                   if kv[1] in label_order else 99)]

    print("─" * 78)
    print("  PER-STATE MEAN (training z-scores) vs LIVE OBSERVATION")
    print("─" * 78)
    header = f"  {'feature':<22}  {'LIVE':>8}  " + "  ".join(
        f"{state_to_label[s]:>10}" for s in state_order
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(feat_df.columns):
        row_means = [means[s, i] for s in state_order]
        line = f"  {name:<22}  {last_row[i]:>+8.3f}  "
        for s_idx, mu in zip(state_order, row_means):
            # Highlight if this state is closest to live on this feature.
            diff = abs(last_row[i] - mu)
            best_diff = min(abs(last_row[i] - means[s, i]) for s in state_order)
            mark = "*" if diff == best_diff else " "
            line += f"{mu:>+9.3f}{mark} "
        print(line)
    print("  (the * marks which state mean is closest to LIVE on each feature)")
    print()

    # ── 6. Per-feature squared distance per state ────────────────────────
    print("─" * 78)
    print("  PER-FEATURE SQUARED DISTANCE  (live - state_mean)²"
          " — bigger = worse fit")
    print("─" * 78)
    print(f"  {'feature':<22}  " + "  ".join(
        f"{state_to_label[s]:>10}" for s in state_order
    ))
    print("  " + "-" * 76)
    sq_total = {s: 0.0 for s in state_order}
    for i, name in enumerate(feat_df.columns):
        line = f"  {name:<22}  "
        for s in state_order:
            d2 = (last_row[i] - means[s, i]) ** 2
            sq_total[s] += d2
            line += f"{d2:>10.3f} "
        print(line)
    print("  " + "-" * 76)
    line = f"  {'TOTAL Σ(Δ²)':<22}  "
    for s in state_order:
        line += f"{sq_total[s]:>10.3f} "
    print(line)
    print()

    # ── 7. Real per-state log-likelihood (uses full covariance) ──────────
    print("─" * 78)
    print("  PER-STATE LOG-LIKELIHOOD AT LATEST BAR  (full covariance)")
    print("─" * 78)
    from scipy.stats import multivariate_normal as mvn
    logp = []
    for s in range(K):
        try:
            lp = mvn.logpdf(last_row, mean=means[s],
                            cov=cov[s], allow_singular=True)
        except Exception as e:
            lp = float("-inf")
            print(f"  warning: state {s} logpdf failed: {e}")
        logp.append(lp)
    logp = np.array(logp)
    # Soft-max to a "if we ignored transitions" likelihood-only posterior
    p = np.exp(logp - logp.max())
    p /= p.sum()
    print(f"  {'state':<12}  {'log-likelihood':>16}  "
          f"{'likelihood-only posterior':>26}")
    print("  " + "-" * 60)
    lp_lo, lp_hi = float(logp.min()), float(logp.max())
    for s in state_order:
        bar = _bar("", logp[s], lp_lo, lp_hi, width=24)
        print(f"  {state_to_label[s]:<12}  {logp[s]:>16.3f}  "
              f"{p[s]*100:>7.2f}%   |{bar}|")
    print()
    print("  ↑ this ignores the transition matrix (no path memory).  "
          "If the winner here disagrees with the LIVE bot reading, "
          "the bot is being held in its current state by the "
          "stability_bars / forward-filter latch.")
    print()

    # ── 8. Forward-algorithm posterior (the actual live reading) ────────
    print("─" * 78)
    print("  FORWARD-ALGORITHM POSTERIOR  (matches the live bot)")
    print("─" * 78)
    from core.hmm_engine import HMMEngine, RegimeState  # noqa: E402
    eng = HMMEngine.__new__(HMMEngine)
    # Hydrate the engine from the saved dict like the bot does.
    eng._model = model
    eng._n_states = K
    eng._n_features = n_features
    eng._state_to_label = state_to_label
    eng._regime_info = saved["regime_info"]
    eng._is_fitted = True
    eng.stability_bars = int(hmm_cfg.get("stability_bars", 7))
    eng.flicker_window = int(hmm_cfg.get("flicker_window", 20))
    eng.flicker_threshold = int(hmm_cfg.get("flicker_threshold", 4))
    eng.min_confidence = float(hmm_cfg.get("min_confidence", 0.62))
    # min_covar is read inside _log_emission_probs to regularise covariance.
    eng.min_covar = float(saved["config"].get("min_covar", 1e-3))
    # The forward filter also touches a few internal buffers — initialise them.
    eng._last_state_distribution = None
    eng._regime_history: list = []
    eng._last_log_alpha = None
    eng._log_transmat = np.log(model.transmat_ + 1e-300)

    # Use the last ~350 bars, like the bot's startup snapshot.
    n_bars = min(350, len(feat_df))
    window = feat_df.iloc[-n_bars:]
    states: list[RegimeState] = eng.predict_regime_filtered(
        window.values,
        timestamps=[pd.Timestamp(ts) for ts in window.index],
    )
    rs_last = states[-1]
    print(f"  Window           : last {n_bars} bars "
          f"({window.index[0].date()} → {window.index[-1].date()})")
    print(f"  Live regime      : {rs_last.label}  "
          f"({rs_last.probability*100:.1f}%)")
    print(f"  Confirmed/stable : {rs_last.is_confirmed}")
    print(f"  All-state probs  :")
    for s in state_order:
        print(f"    {state_to_label[s]:<10}  "
              f"{rs_last.state_probabilities[s]*100:>6.2f}%")
    print()

    # Last 10 bars: how stable has the regime call been?
    print("  Last 10 bars (regime trajectory):")
    print(f"    {'date':<12}  {'label':<10}  {'P(label)':>9}")
    for st in states[-10:]:
        print(f"    {st.timestamp.date()}  {st.label:<10}  "
              f"{st.probability*100:>7.2f}%")
    print()

    # ── 9. Verdict ───────────────────────────────────────────────────────
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    # Compare likelihood-only winner vs forward-filter winner.
    ll_winner = state_to_label[int(np.argmax(logp))]
    fwd_winner = rs_last.label
    if ll_winner == fwd_winner:
        print(f"  Likelihood-only and forward-filter agree on: {fwd_winner}.")
        print(f"  → The model genuinely thinks the latest bar matches "
              f"this state.  Inspect the LATEST FEATURE VECTOR section above "
              f"to see which dimensions are firing it.")
    else:
        print(f"  DISAGREEMENT: likelihood-only says {ll_winner}, "
              f"but the forward filter is locked on {fwd_winner}.")
        print(f"  → The transition matrix / persistence is overriding the "
              f"observation.  This typically means the filter latched on "
              f"a state earlier in the window and stability_bars / "
              f"min_confidence are keeping it there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
