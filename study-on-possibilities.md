# Study on possibilities — beyond the frozen baseline

Context and motivation
----------------------

The current `frozen-baseline-1` (balanced × stocks, prototype labels, single-strat
hmm_regime path) sits at **+174.56% / Sharpe 1.082 / Calmar 1.678 / MaxDD -16.0%**
on 2020–2026. Three trivial config-level levers were tested and **all failed
to improve the baseline** (cf. `scripts/sweep_levers.sh` and the comparison
table at commit `83dc6fe`):

- raising `min_confidence` 0.62 → 0.75 had near-zero effect (HMM is already
  highly confident on its calls)
- dropping the VIX feature **hurt** by -23pp return / -0.13 Sharpe (VIX
  contributes real signal)
- tightening `mid_vol_no_trend` allocation 0.70 → 0.50 cost return without
  reducing MaxDD (the worst drawdowns happen in `high_vol`, not `mid_vol`)

**Conclusion**: the baseline is on a local optimum for the current config
surface. Further gains require structural change. This document scopes
**four** directions, in plain language, with numbered execution plans, effort
estimates, risk levels, and expected gains.

---

## 1. Different model (XGBoost / transformer-based)

### Plain language

Today we use a **Hidden Markov Model (HMM)**. It assumes the market is in one
of 5 hidden states, switches between them with fixed transition probabilities,
and that each state emits observations (returns, volatility) drawn from a
Gaussian distribution. It is **generative** and **Markov**: today's state
depends only on yesterday's.

Limitations of HMM in this context:
- **Gaussian emissions**: real returns have fat tails (large moves are far
  more frequent than a normal distribution predicts).
- **Short memory**: state at *t* depends only on state at *t-1*. A crash
  after six months of bull doesn't have the same meaning as a crash after
  six months of bear, but the HMM cannot distinguish them.
- **Linear in features**: cannot capture multiplicative interactions like
  "high vol AND broken trend AND elevated VIX" as a compound signal.

**Alternatives**:

- **XGBoost classifier** — gradient-boosted decision trees that predict the
  regime label from features. Non-linear, captures interactions, robust to
  fat tails. But **supervised**, so it requires labelled training data
  ("what was the right regime at each historical bar?"). Defining the
  ground truth is the central paradox.
- **Transformer-based** (small TFT / temporal-attention model) — learns
  long-range temporal dependencies. Powerful but heavy to train, many
  hyperparameters, easy to overfit on six years of daily data.
- **HSMM (Hidden Semi-Markov Model)** — relaxes the geometric-duration
  assumption of HMM. States have explicit duration distributions. Better
  fit for real markets where regimes persist 20–100 days.

### Numbered plan

1. **Define the ground truth.** XGBoost needs labels. Two viable options:
   (a) take the HMM's labels as targets and train XGBoost to reproduce them
   with less per-fold remapping noise; (b) define labels retrospectively
   from forward-looking measures ("monthly annualised vol > 30% AND
   monthly return < -5% → CRASH").
2. **Build the feature pipeline.** Reuse the six existing HMM features and
   add a few candidates: RSI, momentum 60d, current drawdown depth.
3. **Train and validate XGBoost** in walk-forward (252 IS / 63 OOS).
   Standard hyperparameter sweep on `max_depth`, `learning_rate`,
   `n_estimators`, `subsample`.
4. **Compare regime sequences** between HMM and XGBoost: agreement
   percentage, periods of disagreement, transition counts.
5. **Backtest with the same allocation logic.** If XGBoost gives more
   stable transitions, the bot's Sharpe should rise even with identical
   downstream code.
6. **Integrate behind a feature flag** (`hmm.model_type: hmm | xgboost`)
   in `settings.yaml` if the validation is positive.

### Cost / risk / gain

- **Effort**: 2–3 weeks (data prep, tuning, rigorous validation).
- **Risk**: HIGH. The label ground-truth question is genuinely unsolved
  in regime detection; an XGBoost trained to mimic an imperfect HMM
  inherits the same imperfections.
- **Expected gain**: speculative. Academic studies show ML models
  sometimes beat HMM by 0.1–0.2 Sharpe for regime detection, sometimes
  underperform (overfit). HMM remains a structurally well-suited and
  hard-to-beat baseline for this specific problem.

---

## 2. Different features (microstructure / options-implied / sentiment)

### Plain language

Current features are all **derived from price and volume** (returns, realized
vol, ADX, distance from SMA-200, VIX z-score). They are **photographs of the
past**. The HMM looks at where we have been, not where the market expects
to go.

**Three new feature families**:

- **Microstructure** — bid-ask spread, order-book imbalance, volume profile,
  opening range. Mostly **intraday**. When the spread widens or the book
  becomes severely imbalanced, that is often a precursor to a move. But it
  requires **tick-level data**, which Alpaca paper does not provide cleanly.

- **Options-implied volatility** — VIX (which we already use), plus
  **skew** (the asymmetry of the implied distribution: when puts trade
  richer than calls, the market fears a crash), **term structure** (1-month
  IV vs 3-month IV; when the front month is above the back, traders expect
  imminent stress), **put/call ratio**, **gamma exposure**. These features
  are **forward-looking**: they reflect what options traders think about
  the future, not the past.

- **Sentiment** — news-sentiment scores, Twitter/Reddit/X mentions, AAII
  retail-sentiment survey, Google Trends on terms like "recession" or
  "stock crash". Noisy on average but often **uncorrelated** with
  price-based features → genuine diversification of signal.

### Numbered plan

1. **Audit the data sources we can pull from.** Free tier: yfinance for
   ^VIX, ^VVIX, ^SKEW; FRED for macro; Alpaca for prices; Hugging Face
   models for news-NLP. Paid: Polygon for options chains, IBKR for
   real-time options.
2. **Choose ONE feature class to start with.** Recommendation:
   **options-implied vol**. More predictable than sentiment, more
   accessible than microstructure, and the existing VIX feature is
   already from this family — so we know the direction works.
3. **Build the historical feature time series** for 2020–2026: pull
   options-skew and term-structure history, compute z-scores.
4. **Feature-importance analysis**: rolling correlation of each new
   feature with future realised vol; mutual information with the
   existing HMM regime labels.
5. **Add to the HMM features list** via an experimental config set
   `exp_options_features.yaml` overriding `features_override`.
6. **Backtest** with the new feature set, compare against `balanced × stocks`.

### Cost / risk / gain

- **Effort**: 2–4 weeks per feature class. Sentiment is the longest
  (NLP pipeline, label review, source reliability).
- **Risk**: MEDIUM. The VIX feature we already have is in this family,
  and it materially helps (the "drop VIX" lever cost 23pp). Direction
  is empirically validated; the question is whether *more* of the same
  family adds incremental value.
- **Expected gain**: +0.1 to +0.3 Sharpe in published studies when done
  well. With skew + term structure on top of VIX, this is likely **the
  most promising direction of the four**.

---

## 3. Different universe (sector rotation)

### Plain language

Today we trade a **fixed basket** of five mega-caps: SPY, QQQ, AAPL, MSFT,
NVDA. The bot adjusts the **size** of the exposure (how long we are) but
not the **composition** (which names we hold).

**Sector rotation** replaces this with the eleven sector ETFs (XLK tech,
XLV health, XLF financials, XLE energy, XLP staples, XLU utilities, XLY
consumer discretionary, XLI industrials, XLB materials, XLRE real estate,
XLC communication) plus **safe havens** (GLD gold, TLT long-duration
treasuries).

**The idea**: at each cycle stage some sectors **lead** while others **lag**.
The bot identifies the top 3–4 sectors by momentum and allocates among
them. As the macro regime changes (BULL → BEAR), it rotates toward
defensives.

**Regime × sector mapping (illustrative)**:
- **BULL** → cyclicals (XLK, XLY, XLI)
- **NEUTRAL** → balanced mix
- **BEAR** → defensives (XLP, XLU, XLV)
- **CRASH** → cash + GLD or TLT

### Numbered plan

1. **Build a `sector_rotation` asset_group** in `config/asset_groups.yaml`
   listing the eleven sector ETFs plus GLD and TLT.
2. **Naïve baseline**: backtest top-3-by-60d-momentum, equal-weighted,
   weekly rebalance, no regime filter. This is the line a regime-aware
   rotation has to beat.
3. **Compute sector features**: 30/60/120-day momentum, relative
   strength vs SPY, 20-day vol of each ETF.
4. **Sector-selection layer conditional on the HMM regime**: simple
   rules (`if BULL: top-3 cyclicals by momentum / if BEAR: defensives /
   if CRASH: 100% TLT + GLD`).
5. **Intra-sector allocation** via `inverse_vol` (already implemented).
6. **Walk-forward backtest 2020–2026**, compare three runs:
   - Pure momentum (no regime filter).
   - Regime + momentum (the proposal).
   - Current baseline (`balanced × stocks` fixed basket).
7. **Measure MaxDD per macro regime.** The real value of rotation is
   reducing drawdown during BEAR/CRASH, not necessarily lifting absolute
   return.

### Cost / risk / gain

- **Effort**: 1–2 weeks. The infrastructure (asset_groups registry,
  walk-forward backtester) already exists.
- **Risk**: MEDIUM-LOW. Sector rotation has been a tough environment in
  2023–2024 (tech crushed everything else, so rotations toward defensives
  cost return), but **2022 and early 2025 were strong years** for
  rotation: well-tuned models reduced MaxDD by ~30% versus buy-and-hold
  in those windows.
- **Expected gain**: +0.0 to +0.3 Sharpe. **The headline gain is on
  MaxDD reduction, not on absolute return** — typically -30% on MaxDD
  versus a fixed basket in historical backtests. That makes the bot
  investable at larger size.

---

## 4. Different time horizon (intraday with microstructure)

### Plain language

Today the bot's bars are **1 day** (HMM macro), with an intraday 5-min
main loop only for risk monitoring. The bot makes one macro decision per
day, effectively.

**Going intraday** means making decisions on a shorter timeframe (15-min
or 1-hour bars), capturing intra-day patterns: the **opening-range
breakout**, the **lunchtime drift** (US markets often weaken between 12:00
and 14:00 ET), the **close auction**.

**Microstructure features**: bid-ask spread, volume profile (how much
volume traded at each price level), VWAP, order-book imbalance, gap from
prior close.

**Main difficulty — costs and slippage become material**.
At 1Day cadence we take ~85 trades/year with 5bp slippage per side, for
~85bp annual drag. At 15-min cadence we could take 20× that = 1700
trades/year = ~17pp annual drag. **The intraday edge has to clear that
hurdle**, which is a high bar.

### Numbered plan

1. **Pick the timeframe.** Rule out 5-min (too noisy for HMM-style regime
   detection). Choose between **1H** (more robust) and **15-min** (more
   signal, more cost).
2. **Two-layer architecture**: keep the **macro HMM on 1Day** as the
   strategic filter; build a **separate intraday model** that decides
   *when* in the day to enter / exit. Macro filters, intraday times.
3. **Build intraday features**: opening-range high-low (first 30 min),
   volume profile, distance from VWAP, time-of-day dummies, gap-fill
   probability.
4. **Simple intraday signal model**: deterministic rule or logistic
   regression that outputs "trade / skip" at each 15-min bar.
5. **Walk-forward backtest with REALISTIC slippage**: 5–10bp per side
   minimum on liquid ETFs like SPY/QQQ, more for single names. Optionally
   model **market impact** (orders move the price).
6. **Compare net-of-slippage Sharpe** against the daily baseline.
7. **If net Sharpe is positive**, integrate as an overlay (the daily bot
   makes the macro call, the intraday module optimises the entry timing).

### Cost / risk / gain

- **Effort**: 3–6 weeks. The heaviest of the four. Requires tick data or
  at least clean 1-min bars.
- **Risk**: VERY HIGH. Intraday strategies are notoriously fragile.
  Many show in-sample Sharpe of 2.0 that collapses to 0.5 OOS. The
  1-day baseline +174% / Sharpe 1.08 is a high bar to beat **net of fees**.
- **Expected gain**: very speculative. **Significant probability that
  the net gain is zero or negative** once realistic slippage is applied.
  Least promising of the four directions.

---

## Synthesis

| Direction | Effort | Risk | Realistic gain | Verdict |
|---|---|---|---|---|
| **2. Features (options-implied)** | 2–4 weeks | Medium | +0.1 to +0.3 Sharpe | ⭐ **best ROI** |
| **3. Sector rotation** | 1–2 weeks | Medium-low | -30% MaxDD, return ≈ | 🥈 strong second |
| **1. Different model (XGBoost)** | 2–3 weeks | High | speculative | 🤔 only out of curiosity |
| **4. Intraday** | 3–6 weeks | Very high | often net-negative | ❌ avoid unless specific edge |

### Recommended attack order if you stay on darkvador

1. **First #3 (sector rotation)** — the infrastructure already exists,
   the experimentation is fast, and the headline benefit (MaxDD
   reduction) is what makes the bot investable at larger size.
2. **Then #2 (options features)** — the natural next step, lifts Sharpe.
3. **Then #1 (XGBoost)** if there is appetite for an open-ended
   modelling experiment.
4. **#4 (intraday)** only if a specific intraday edge is identified;
   otherwise the effort/return ratio is poor.

If you move on to the new project as planned, keep this document as a
reference for when you eventually revisit darkvador. None of the four
directions is time-sensitive — they will still be valid in three or
six months.

---

*Document generated 2026-04-28 after the lever-sweep experiment
(`scripts/sweep_levers.sh`) confirmed that no trivial config tweak
improves the frozen-baseline-1 reference numbers.*
