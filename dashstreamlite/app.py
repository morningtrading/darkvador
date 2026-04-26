"""
dashstreamlite/app.py — clean, web-based dashboard for the regime-trader bot.

Run:
    streamlit run dashstreamlite/app.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Streamlit runs app.py as a top-level script (not as part of a package),
# so `dashstreamlite/` is already on sys.path[0]. Import the sibling module
# directly to keep this working whether you launch from the repo root or
# from inside dashstreamlite/.
from data_loader import (
    BotContext,
    fetch_proxy_prices,
    latest_backtest_dir,
    load_backtest_summary,
    load_bot_context,
    load_equity_curve,
    load_regime_history,
    load_state_snapshot,
    regime_segments,
)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regime Trader — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Visual theme ──────────────────────────────────────────────────────────────

REGIME_COLOURS = {
    "EUPHORIA":    "#16a34a",
    "STRONG_BULL": "#22c55e",
    "BULL":        "#4ade80",
    "WEAK_BULL":   "#86efac",
    "NEUTRAL":     "#94a3b8",
    "WEAK_BEAR":   "#fdba74",
    "BEAR":        "#f97316",
    "STRONG_BEAR": "#dc2626",
    "CRASH":       "#7f1d1d",
}
REGIME_BG = {k: v + "33" for k, v in REGIME_COLOURS.items()}  # ~20% alpha

REGIME_ICONS = {
    "EUPHORIA":    "🚀",
    "STRONG_BULL": "🟢",
    "BULL":        "🟢",
    "WEAK_BULL":   "🟢",
    "NEUTRAL":     "⚪",
    "WEAK_BEAR":   "🟠",
    "BEAR":        "🔴",
    "STRONG_BEAR": "🔴",
    "CRASH":       "💥",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px; }
      h1 { font-size: 1.5rem !important; margin-bottom: 0.2rem !important; }
      .meta-row { font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.75rem;
                  color: #6b7280; margin-bottom: 1rem; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
      div[data-testid="stMetricLabel"]  { color: #6b7280; }
      .regime-card { padding: 0.6rem 0.9rem; border-radius: 8px;
                     border-left: 4px solid; background: #1e293b15;
                     font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar (controls) ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    history_years = st.slider("Chart history (years)", 1, 10, 6)
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

if auto_refresh:
    # Streamlit native auto-rerun every 60s
    st.markdown(
        """
        <meta http-equiv="refresh" content="60">
        """,
        unsafe_allow_html=True,
    )


# ── Load all data ─────────────────────────────────────────────────────────────

ctx: BotContext = load_bot_context()
snap          = load_state_snapshot()
bt_dir        = latest_backtest_dir()
summary       = load_backtest_summary(bt_dir) if bt_dir else None
regime_hist   = load_regime_history(bt_dir)   if bt_dir else None


# ── Header ────────────────────────────────────────────────────────────────────

now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

st.markdown(
    f"# 📊 Regime Trader — `{ctx.asset_group}` · "
    f"<span style='color:#eab308'>[{ctx.config_set}]</span> · "
    f"<span style='color:#06b6d4'>HMM: {ctx.regime_proxy}</span>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='meta-row'>"
    f"{ctx.bot_name} · {ctx.host} · "
    f"loop={ctx.timeframe_loop} · hmm={ctx.timeframe_hmm} · #{ctx.git_sha} · {now_utc}"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Helper: current regime stats ──────────────────────────────────────────────

segments = regime_segments(regime_hist) if regime_hist is not None else []
current_seg = segments[-1] if segments else None


# ── Top metric row ────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if current_seg:
        regime  = current_seg["regime"]
        icon    = REGIME_ICONS.get(regime, "📊")
        st.markdown(
            f"<div class='regime-card' style='border-color:{REGIME_COLOURS.get(regime,'#94a3b8')}'>"
            f"<div style='font-size:0.75rem;color:#6b7280;margin-bottom:0.2rem;'>REGIME</div>"
            f"<div style='font-size:1.4rem;'>{icon} {regime}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.metric("Regime", "—")

with c2:
    days = current_seg["days"] if current_seg else 0
    st.metric("In regime since", f"{days}j",
              delta=current_seg["start"].strftime("%Y-%m-%d") if current_seg else None,
              delta_color="off")

with c3:
    if summary:
        ret_pct = summary.total_return * 100
        st.metric("Backtest return", f"{ret_pct:+.1f}%",
                  delta=f"CAGR {summary.cagr*100:+.1f}%", delta_color="off")
    else:
        st.metric("Backtest return", "—")

with c4:
    if summary:
        st.metric("Sharpe / Calmar",
                  f"{summary.sharpe:.2f} / {summary.calmar:.2f}",
                  delta=f"MaxDD {summary.max_drawdown*100:+.1f}%",
                  delta_color="off")
    else:
        st.metric("Sharpe / Calmar", "—")

with c5:
    live_eq = float(snap.get("equity", 0.0)) if snap else None
    if live_eq:
        regime_live = snap.get("regime", "—") if snap else "—"
        st.metric("Live equity", f"${live_eq:,.0f}",
                  delta=f"Live regime: {regime_live}",
                  delta_color="off")
    elif summary:
        st.metric("Backtest final equity", f"${summary.final_equity:,.0f}")
    else:
        st.metric("Equity", "—")


st.divider()


# ── Main chart: HMM proxy price + regime overlay ──────────────────────────────

st.subheader(f"📈 {ctx.regime_proxy} (HMM proxy) — price & regimes")

if not ctx.regime_proxy:
    st.warning("No `hmm.regime_proxy` configured in settings.yaml.")
elif regime_hist is None or regime_hist.empty:
    st.warning("No regime_history.csv found in latest backtest directory.")
else:
    # Restrict to the requested history window.
    cutoff = regime_hist.index.max() - pd.Timedelta(days=int(history_years * 365.25))
    hist_view = regime_hist[regime_hist.index >= cutoff]
    segs_view = [s for s in segments if s["end"] >= cutoff]

    # Fetch proxy price series for the same window via yfinance (does not
    # touch Alpaca / bot credentials).
    start_str = hist_view.index.min().strftime("%Y-%m-%d")
    end_str   = (hist_view.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with st.spinner(f"Fetching {ctx.regime_proxy} prices from yfinance..."):
        prices = fetch_proxy_prices(ctx.regime_proxy, start_str, end_str)

    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True,
        specs=[[{"secondary_y": False}]],
    )

    # Coloured background bands per regime segment
    for seg in segs_view:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor=REGIME_BG.get(seg["regime"], "#94a3b833"),
            line_width=0,
            layer="below",
            annotation_text="",
        )

    # Vertical markers at regime transitions (start of each new segment)
    if len(segs_view) > 1:
        for seg in segs_view[1:]:
            fig.add_vline(
                x=seg["start"],
                line=dict(color="#cbd5e1", width=0.5, dash="dot"),
                layer="below",
            )

    if prices is not None and not prices.empty:
        prices_view = prices[prices.index >= cutoff]
        fig.add_trace(go.Scatter(
            x=prices_view.index,
            y=prices_view.values,
            mode="lines",
            name=ctx.regime_proxy,
            line=dict(color="#0f172a", width=1.5),
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>"
                f"{ctx.regime_proxy} close: $%{{y:.2f}}<extra></extra>"
            ),
        ))
    else:
        # No price line → regime bands still rendered, with a placeholder note.
        fig.add_annotation(
            x=hist_view.index.min(),
            y=0.5, xref="x", yref="paper",
            text=f"(yfinance returned no data for {ctx.regime_proxy} — bands only)",
            showarrow=False, font=dict(color="#9ca3af"),
        )

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Legend strip below the chart
    legend_chunks = []
    seen = []
    for seg in segs_view:
        if seg["regime"] not in seen:
            seen.append(seg["regime"])
    for r in seen:
        c = REGIME_COLOURS.get(r, "#94a3b8")
        legend_chunks.append(
            f"<span style='display:inline-block;width:0.7rem;height:0.7rem;"
            f"background:{c};border-radius:2px;margin-right:0.3rem;"
            f"vertical-align:middle;'></span>"
            f"<span style='margin-right:1rem;font-size:0.85rem;'>{r}</span>"
        )
    st.markdown(
        "<div style='margin-top:-0.5rem;'>" + "".join(legend_chunks) + "</div>",
        unsafe_allow_html=True,
    )


st.divider()


# ── Bottom row: regime segments table + active config ─────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("🔄 Last 10 regime segments")
    if segments:
        last10 = list(reversed(segments[-10:]))
        rows = []
        for s in last10:
            rows.append({
                "Regime":   f"{REGIME_ICONS.get(s['regime'],'📊')}  {s['regime']}",
                "Start":    s["start"].strftime("%Y-%m-%d"),
                "End":      "en cours" if s is segments[-1] else s["end"].strftime("%Y-%m-%d"),
                "Days":     s["days"],
                "Bars":     s["bars"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True, use_container_width=True,
            column_config={
                "Regime": st.column_config.TextColumn(width="medium"),
                "Days":   st.column_config.NumberColumn(width="small"),
                "Bars":   st.column_config.NumberColumn(width="small"),
            },
        )
    else:
        st.info("No regime history available.")

with right:
    st.subheader("⚙️ Active configuration")
    st.markdown(
        f"""
        - **Bot name** &nbsp;`{ctx.bot_name}`
        - **Host** &nbsp;`{ctx.host}`
        - **Asset group** &nbsp;`{ctx.asset_group}`
        - **Config set** &nbsp;`{ctx.config_set}`
        - **HMM proxy** &nbsp;`{ctx.regime_proxy}`
        - **Loop timeframe** &nbsp;`{ctx.timeframe_loop}`
        - **HMM timeframe** &nbsp;`{ctx.timeframe_hmm}`
        - **Git** &nbsp;`#{ctx.git_sha}`
        """
    )
    if summary:
        st.markdown(
            f"""
            ---
            **Latest backtest** &nbsp;<span style='font-family:monospace;color:#6b7280;'>{summary.dir_name}</span>

            - {len(summary.symbols)} symbols: `{', '.join(summary.symbols)}`
            - Period: `{summary.start_date}` → `{summary.end_date}` ({summary.folds} folds)
            - Trades: **{summary.trades}** · Win rate: **{summary.win_rate*100:.1f}%**
            """,
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='text-align:right;color:#9ca3af;font-size:0.75rem;margin-top:1.5rem;'>"
    f"Read-only view. Bot state is at <code>state_snapshot.json</code> and "
    f"<code>{bt_dir.name if bt_dir else '—'}</code>. "
    f"This dashboard does not modify anything."
    f"</div>",
    unsafe_allow_html=True,
)
