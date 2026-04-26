"""
dashstreamlite/app.py — clean, web-based dashboard for the regime-trader bot.

Run:
    streamlit run dashstreamlite/app.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    compute_full_period_regimes,
    fetch_proxy_prices,
    latest_backtest_dir,
    load_backtest_summary,
    load_bot_context,
    load_equity_curve,
    load_hmm_state_stats,
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


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """Convert a #RRGGBB hex string to plotly's rgba(r,g,b,a) form.
    Plotly's fillcolor validator rejects 8-digit hex (#RRGGBBAA)."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


REGIME_BG = {k: _hex_to_rgba(v, 0.35) for k, v in REGIME_COLOURS.items()}

# Distinct plotly symbol per regime — picked so bullish entries point up,
# bearish point down, and extreme states get a special glyph.
REGIME_SYMBOL = {
    "EUPHORIA":    "star",
    "STRONG_BULL": "triangle-up",
    "BULL":        "triangle-up",
    "WEAK_BULL":   "triangle-up-open",
    "NEUTRAL":     "circle",
    "WEAK_BEAR":   "triangle-down-open",
    "BEAR":        "triangle-down",
    "STRONG_BEAR": "triangle-down",
    "CRASH":       "x",
}

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


# ── Chart-building helpers ────────────────────────────────────────────────────


def _build_regime_figure(prices_view, segs_view, proxy_label):
    """Return a Plotly Figure for `proxy_label` price overlaid with the
    coloured regime bands, transition vlines, and regime markers from
    `segs_view`. `prices_view` is a pd.Series of close prices; may be empty
    if yfinance failed."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        specs=[[{"secondary_y": False}]])

    # Coloured background bands per regime segment.
    for seg in segs_view:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor=REGIME_BG.get(seg["regime"], "rgba(148,163,184,0.20)"),
            line_width=0, layer="below",
        )

    # Dotted vertical lines at regime transitions.
    if len(segs_view) > 1:
        for seg in segs_view[1:]:
            fig.add_vline(
                x=seg["start"],
                line=dict(color="#cbd5e1", width=0.5, dash="dot"),
                layer="below",
            )

    if prices_view is not None and not prices_view.empty:
        fig.add_trace(go.Scatter(
            x=prices_view.index,
            y=prices_view.values,
            mode="lines", name=proxy_label,
            line=dict(color="#0f172a", width=1.5),
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br>"
                f"{proxy_label} close: $%{{y:.2f}}<extra></extra>"
            ),
            showlegend=False,
        ))

        # Regime-change markers — one Scatter per regime so the legend auto-builds.
        per_regime: dict = {}
        for seg in segs_view[1:]:
            try:
                idx = prices_view.index.get_indexer([seg["start"]], method="ffill")[0]
                if idx == -1:
                    continue
                px = float(prices_view.iloc[idx])
            except Exception:
                continue
            per_regime.setdefault(seg["regime"], {"x": [], "y": [], "hover": []})
            per_regime[seg["regime"]]["x"].append(seg["start"])
            per_regime[seg["regime"]]["y"].append(px)
            per_regime[seg["regime"]]["hover"].append(
                f"<b>{REGIME_ICONS.get(seg['regime'],'')} {seg['regime']}</b><br>"
                f"start: {seg['start'].strftime('%Y-%m-%d')}<br>"
                f"end: {seg['end'].strftime('%Y-%m-%d')}<br>"
                f"duration: {seg['days']}j ({seg['bars']} bars)"
            )
        for regime, data in per_regime.items():
            fig.add_trace(go.Scatter(
                x=data["x"], y=data["y"],
                mode="markers", name=regime,
                marker=dict(
                    symbol=REGIME_SYMBOL.get(regime, "circle"),
                    color=REGIME_COLOURS.get(regime, "#94a3b8"),
                    size=14,
                    line=dict(color="#0f172a", width=1.0),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=data["hover"],
                showlegend=True,
            ))
    else:
        fig.add_annotation(
            x=segs_view[0]["start"] if segs_view else None,
            y=0.5, xref="x", yref="paper",
            text=f"(yfinance returned no data for {proxy_label} — bands only)",
            showarrow=False, font=dict(color="#9ca3af"),
        )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="#e5e7eb", borderwidth=1,
        ),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
        hovermode="closest",
    )
    return fig


def _regime_legend_strip(segs_view) -> str:
    """HTML strip showing one coloured swatch per regime present in segs_view."""
    seen = []
    for seg in segs_view:
        if seg["regime"] not in seen:
            seen.append(seg["regime"])
    chunks = []
    for r in seen:
        c = REGIME_COLOURS.get(r, "#94a3b8")
        chunks.append(
            f"<span style='display:inline-block;width:0.7rem;height:0.7rem;"
            f"background:{c};border-radius:2px;margin-right:0.3rem;"
            f"vertical-align:middle;'></span>"
            f"<span style='margin-right:1rem;font-size:0.85rem;'>{r}</span>"
        )
    return "<div style='margin-top:-0.5rem;'>" + "".join(chunks) + "</div>"


# ── Main charts: walk-forward (backtest) vs single-fold (full period) ────────

if not ctx.regime_proxy:
    st.warning("No `hmm.regime_proxy` configured in settings.yaml.")
elif regime_hist is None or regime_hist.empty:
    st.warning("No regime_history.csv found in latest backtest directory.")
else:
    # Common: history window cutoff and proxy price series.
    cutoff = regime_hist.index.max() - pd.Timedelta(days=int(history_years * 365.25))
    start_str = (regime_hist.index.min()).strftime("%Y-%m-%d")
    end_str   = (regime_hist.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with st.spinner(f"Fetching {ctx.regime_proxy} prices from yfinance..."):
        prices_full = fetch_proxy_prices(ctx.regime_proxy, start_str, end_str)
    prices_view = prices_full[prices_full.index >= cutoff] if prices_full is not None else None

    # ── Panel 1: walk-forward (backtest) regimes ─────────────────────────────
    st.subheader(
        f"📈 {ctx.regime_proxy} — walk-forward regimes "
        f"(17 folds × 252-bar IS / 63-bar OOS)"
    )
    st.caption(
        "Source: latest `savedresults/backtest_*/regime_history.csv`. "
        "Each segment was predicted by a different HMM trained on the prior "
        "12 months — so the `CRASH`/`BEAR`/etc. labels are **re-mapped per "
        "fold** and not directly comparable across the timeline."
    )
    segs_bt_view = [s for s in segments if s["end"] >= cutoff]
    fig_bt = _build_regime_figure(prices_view, segs_bt_view, ctx.regime_proxy)
    st.plotly_chart(fig_bt, use_container_width=True, theme=None,
                    key="chart_walk_forward")
    st.markdown(_regime_legend_strip(segs_bt_view), unsafe_allow_html=True)

    st.write("")  # vertical spacer

    # ── Panel 2: single-fold (full-period) regimes ──────────────────────────
    st.subheader(
        f"🧠 {ctx.regime_proxy} — single-fold regimes "
        f"(one HMM trained on the full period)"
    )
    st.caption(
        "Same HMM hyper-parameters as the backtest, but trained **once** on "
        f"all {start_str} → {end_str} bars. Labels stay consistent across "
        "the timeline so a `CRASH` here means the same statistical state "
        "everywhere. Quick check for whether the noise above came from "
        "per-fold re-mapping. **First load takes ~90s** while the HMM fits."
    )

    # Hashable representation of the HMM params relevant to the fit.
    bt_hmm_cfg = {}
    if summary:
        # Re-read the run_context for full hmm config that was applied.
        try:
            import json as _json
            ctx_path = (latest_backtest_dir() or Path()) / "run_context.json"
            if ctx_path.exists():
                bt_hmm_cfg = _json.loads(ctx_path.read_text()) or {}
        except Exception:
            pass
    # Use bot's actual hmm config from settings.yaml (loaded via load_bot_context
    # only as a starting point — re-read to get full feature names list, etc.).
    import yaml as _yaml
    cfg_full = _yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "settings.yaml").read_text()) or {}
    hmm_full = cfg_full.get("hmm", {}) or {}
    hmm_params_repr = (
        ("feature_names",     tuple(hmm_full.get("features", ["log_ret_1", "realized_vol_20"]))),
        ("n_candidates",      tuple(hmm_full.get("n_candidates", [5]))),
        ("n_init",            int(hmm_full.get("n_init", 10))),
        ("min_train_bars",    int(hmm_full.get("min_train_bars", 252))),
        ("stability_bars",    int(hmm_full.get("stability_bars", 7))),
        ("flicker_window",    int(hmm_full.get("flicker_window", 20))),
        ("flicker_threshold", int(hmm_full.get("flicker_threshold", 4))),
        ("min_confidence",    float(hmm_full.get("min_confidence", 0.62))),
        ("covariance_type",   str(hmm_full.get("covariance_type", "full"))),
    )

    @st.cache_data(show_spinner="Training single-fold HMM on full period (~90s on first run)...")
    def _cached_full_period_regimes(symbol, start, end, params_repr):
        df, err = compute_full_period_regimes(symbol, start, end, params_repr)
        # Cache only successes — caching None would persist transient yfinance
        # failures across reruns. Raising on failure makes Streamlit skip the
        # cache write and re-attempt next time.
        if df is None:
            raise RuntimeError(err or "unknown error")
        return df

    sf_df = None
    sf_err = None
    try:
        sf_df = _cached_full_period_regimes(
            ctx.regime_proxy, start_str, end_str, hmm_params_repr,
        )
    except Exception as exc:
        sf_err = str(exc)

    if sf_df is None or sf_df.empty:
        st.error(
            f"Single-fold HMM training failed: **{sf_err or 'no data'}**. "
            "Click *Refresh now* in the sidebar to retry — transient "
            "yfinance failures often clear on a second attempt."
        )
    else:
        sf_segments = regime_segments(sf_df)
        sf_segs_view = [s for s in sf_segments if s["end"] >= cutoff]
        fig_sf = _build_regime_figure(prices_view, sf_segs_view, ctx.regime_proxy)
        st.plotly_chart(fig_sf, use_container_width=True, theme=None,
                        key="chart_single_fold")
        st.markdown(_regime_legend_strip(sf_segs_view), unsafe_allow_html=True)

        # Side-by-side stat: how often do the two views agree?
        try:
            common = regime_hist.join(sf_df.rename(columns={"regime": "regime_sf"}),
                                      how="inner")
            agree_pct = (common["regime"] == common["regime_sf"]).mean() * 100
            n_changes_bt = sum(1 for _ in segs_bt_view) - 1
            n_changes_sf = sum(1 for _ in sf_segs_view) - 1
            st.caption(
                f"**Agreement** between the two regime views on overlapping "
                f"dates: **{agree_pct:.1f}%**. "
                f"Walk-forward: {n_changes_bt} transitions in window. "
                f"Single-fold: {n_changes_sf} transitions in window."
            )
        except Exception:
            pass


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


st.divider()


# ── Pareto: HMM state distribution in (vol_z, ret_z) space ────────────────────

st.subheader("📍 HMM state distribution — Pareto in (vol_z, ret_z) space")
st.caption(
    "Each point is one of the HMM's discrete states, plotted by its emission "
    "mean on `realized_vol_20` (x-axis, z-scored) and `log_ret_1` (y-axis, "
    "z-scored). The colour is the **current** label assigned to that state. "
    "Crosses (`✕`) are the prototype targets piste 3 will use to assign "
    "labels — a state should sit close to its prototype if the labelling is "
    "coherent."
)

state_stats = load_hmm_state_stats()

if state_stats is None or state_stats.empty:
    st.info("No HMM model found at `models/hmm.pkl` (train the bot once first).")
else:
    # Prototype targets in z-score space — same values piste 3 will use.
    PROTOTYPES = {
        "CRASH":    (+2.0, -2.0),   # high vol, very negative return
        "BEAR":     (+1.0, -1.0),
        "NEUTRAL":  ( 0.0,  0.0),
        "BULL":     (-0.5, +1.0),   # low vol, positive return
        "EUPHORIA": (-1.0, +2.0),   # very low vol, very positive return
    }

    fig_p = go.Figure()

    # Plot each state as a labelled marker, coloured by its current label.
    for _, row in state_stats.iterrows():
        lbl = row["label"]
        c   = REGIME_COLOURS.get(lbl, "#94a3b8")
        sym = REGIME_SYMBOL.get(lbl, "circle")
        occ = row.get("occupancy_pct", 0.0)
        size = max(18, min(60, 18 + occ * 1.5))  # bigger marker for higher occupancy
        fig_p.add_trace(go.Scatter(
            x=[row["vol_z"]],
            y=[row["ret_z"]],
            mode="markers+text",
            marker=dict(
                size=size,
                symbol=sym,
                color=c,
                line=dict(color="#0f172a", width=1.5),
            ),
            text=[f"S{int(row['state_id'])}: {lbl}"],
            textposition="top center",
            textfont=dict(size=11, color="#0f172a"),
            hovertemplate=(
                f"<b>State {int(row['state_id'])}</b><br>"
                f"label: {lbl}<br>"
                f"vol_z: {row['vol_z']:+.3f}<br>"
                f"ret_z: {row['ret_z']:+.3f}<br>"
                f"occupancy: {occ:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ))

    # Plot the prototype targets as light reference markers.
    for name, (vol_p, ret_p) in PROTOTYPES.items():
        fig_p.add_trace(go.Scatter(
            x=[vol_p], y=[ret_p],
            mode="markers+text",
            marker=dict(
                size=14, symbol="x-thin",
                color=REGIME_COLOURS.get(name, "#94a3b8"),
                line=dict(color=REGIME_COLOURS.get(name, "#94a3b8"), width=2.5),
            ),
            text=[f"  {name} target"],
            textposition="middle right",
            textfont=dict(size=9, color="#64748b"),
            opacity=0.55,
            hovertemplate=f"prototype: {name}<br>vol_z={vol_p:+.1f}<br>ret_z={ret_p:+.1f}<extra></extra>",
            showlegend=False,
        ))

    # Reference cross at (0, 0) — the average market regime.
    fig_p.add_hline(y=0, line=dict(color="#e2e8f0", width=1, dash="solid"), layer="below")
    fig_p.add_vline(x=0, line=dict(color="#e2e8f0", width=1, dash="solid"), layer="below")

    fig_p.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(
            title="realized_vol_20  (z-score)  →  more volatile",
            zeroline=True, zerolinecolor="#e2e8f0",
            showgrid=True, gridcolor="#f1f5f9",
            range=[-2.5, 3.0],
        ),
        yaxis=dict(
            title="log_ret_1  (z-score)  →  higher mean return",
            zeroline=True, zerolinecolor="#e2e8f0",
            showgrid=True, gridcolor="#f1f5f9",
            range=[-2.5, 2.5],
        ),
        showlegend=False,
        hovermode="closest",
    )
    st.plotly_chart(fig_p, use_container_width=True, theme=None,
                    key="chart_pareto")

    # Stats table below
    show = state_stats[["state_id", "label", "ret_z", "vol_z", "occupancy_pct"]].copy()
    show.columns = ["State", "Label", "Mean log_ret (z)", "Mean vol_20 (z)", "Occupancy (%)"]
    show["Mean log_ret (z)"] = show["Mean log_ret (z)"].map(lambda v: f"{v:+.3f}")
    show["Mean vol_20 (z)"]  = show["Mean vol_20 (z)"].map(lambda v: f"{v:+.3f}")
    show["Occupancy (%)"]    = show["Occupancy (%)"].map(lambda v: f"{v:.1f}")
    st.dataframe(show, hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='text-align:right;color:#9ca3af;font-size:0.75rem;margin-top:1.5rem;'>"
    f"Read-only view. Bot state is at <code>state_snapshot.json</code> and "
    f"<code>{bt_dir.name if bt_dir else '—'}</code>. "
    f"This dashboard does not modify anything."
    f"</div>",
    unsafe_allow_html=True,
)
