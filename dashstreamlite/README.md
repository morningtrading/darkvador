# dashstreamlite

A read-only web dashboard for the regime-trader bot. Renders the same data the
terminal dashboard uses, plus a chart of the HMM proxy symbol with regime
bands overlaid.

## Install

```bash
.venv/bin/pip install -r dashstreamlite/requirements.txt
```

(streamlit + plotly only; nothing in the main `requirements.txt` is touched.)

## Run

```bash
.venv/bin/streamlit run dashstreamlite/app.py
```

Open `http://localhost:8501` in a browser. Add `--server.port 8502` if 8501
is taken (the bot doesn't bind 8501; this is just for parallel dashboards).

## What it shows

- **Header**: asset group · `[active set]` · HMM proxy, plus host/SHA/timeframes
- **Top metrics**: current regime, days since regime start, backtest return,
  Sharpe / Calmar, equity (live if `state_snapshot.json` exists, else backtest)
- **Main chart**: regime-proxy daily close (via yfinance) with coloured
  background bands per regime segment and dotted lines at transitions
- **Last 10 regime segments**: same data as the Telegram regime status, in a
  sortable table
- **Active configuration**: read straight from `config/settings.yaml`,
  `config/active_set`, and `run_context.json` of the latest backtest

## What it does NOT do

- Place orders, modify state, or call the Alpaca trading API
- Persist anything outside dashstreamlite/
- Change any code under `core/`, `broker/`, `monitoring/`, etc.

It is a pure visualisation layer over files the bot already writes.

## Notes

- Price data comes from yfinance (already a project dep), not Alpaca, so the
  dashboard works even when the bot's keys are between rotations
- `Auto-refresh (60s)` toggle in the sidebar uses an HTML meta-refresh
- History range is configurable from 1 to 10 years via the sidebar slider
