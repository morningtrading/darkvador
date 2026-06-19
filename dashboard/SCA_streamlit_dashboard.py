"""SCA_streamlit_dashboard.py

Objective
---------
Browser dashboard for the SCA Regime Momentum 01 paper-live bot.

Rationale
---------
The optimized paper runner writes cycle, order, and error CSV files. This app
turns those files into a browser view that can be opened from a phone or local
desktop without touching trading logic.

Dependencies
------------
Streamlit, pandas, and the paper-live CSV files under
savedresults/SCA_regime_momentum_01_live/.

Expected output
---------------
An auto-refreshing browser dashboard on http://localhost:8501.

How to test
-----------
.venv/bin/streamlit run dashboard/SCA_streamlit_dashboard.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from html import escape
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "savedresults" / "SCA_regime_momentum_01_live"
CYCLES_CSV = LIVE_DIR / "SCA_paper_live_cycles.csv"
ORDERS_CSV = LIVE_DIR / "SCA_paper_live_orders.csv"
ERRORS_CSV = LIVE_DIR / "SCA_paper_live_errors.csv"
PID_FILE = LIVE_DIR / "SCA_paper_live.pid"
STDOUT_LOG = LIVE_DIR / "SCA_paper_live_stdout.log"
STDERR_LOG = LIVE_DIR / "SCA_paper_live_stderr.log"
BOT_SCRIPT = ROOT / "scratchpad" / "SCA_regime_momentum_01_paper_live.py"
BOT_PATTERN = "SCA_regime_momentum_01_paper_live.py"
DEFAULT_ASSET_GROUP = "alpaca_momentum10"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["timestamp", "bar_timestamp", "next_open", "next_close"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _strategy_settings() -> dict:
    settings = _load_yaml(ROOT / "config" / "settings.yaml")
    return dict(settings.get("regime_momentum_01", {}))


def _asset_group_symbols(group_name: str = DEFAULT_ASSET_GROUP) -> list[str]:
    groups = _load_yaml(ROOT / "config" / "asset_groups.yaml").get("groups", {})
    group = groups.get(group_name, {})
    return [str(symbol).upper() for symbol in group.get("symbols", [])]


def _scan_universe() -> tuple[str, list[str]]:
    cfg = _strategy_settings()
    symbols = _asset_group_symbols(DEFAULT_ASSET_GROUP)
    proxy = str(cfg.get("regime_proxy", "QQQ")).upper()
    universe = list(dict.fromkeys([proxy] + symbols))
    return proxy, universe


def _minutes_until_next_scan(last_cycle: pd.Series | None, interval_seconds: int) -> float | None:
    if last_cycle is None:
        return None
    timestamp = pd.to_datetime(last_cycle.get("timestamp"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    next_scan = timestamp + pd.Timedelta(seconds=interval_seconds)
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (next_scan - now).total_seconds() / 60.0)


def _bot_process() -> tuple[bool, str]:
    pids = _matching_bot_pids()
    if not pids:
        return False, "No matching process found."
    result = subprocess.run(
        ["ps", "-o", "pid=,ppid=,stat=,etime=,cmd=", "-p", ",".join(str(pid) for pid in pids)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    return True, result.stdout.strip()


def _matching_bot_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-af", BOT_PATTERN],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    pids = []
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        pid = int(parts[0])
        command = parts[1] if len(parts) > 1 else ""
        if pid == current_pid or "streamlit" in command or "pgrep" in command:
            continue
        if command.startswith(("bash ", "sh ", "timeout ")):
            continue
        if "python" not in command or "--loop" not in command or "--execute" not in command:
            continue
        pids.append(pid)
    return pids


def _start_bot() -> str:
    running_pids = _matching_bot_pids()
    if running_pids:
        return f"Bot already running: {running_pids}"

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    stdout_fh = STDOUT_LOG.open("ab")
    stderr_fh = STDERR_LOG.open("ab")
    try:
        process = subprocess.Popen(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(BOT_SCRIPT),
                "--loop",
                "--execute",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
        )
    finally:
        stdout_fh.close()
        stderr_fh.close()
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    time.sleep(1)
    confirmed_pids = _matching_bot_pids()
    if process.pid not in confirmed_pids:
        raise RuntimeError(f"Bot start failed; pid {process.pid} is not running.")
    return f"Bot started: pid {process.pid}"


def _stop_bot() -> str:
    initial_pids = _matching_bot_pids()
    if not initial_pids:
        return "No matching bot process found."
    for pid in initial_pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    remaining = _matching_bot_pids()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.25)
        remaining = _matching_bot_pids()
    killed = []
    for pid in remaining:
        os.kill(pid, signal.SIGKILL)
        killed.append(pid)
    final_remaining = _matching_bot_pids()
    if final_remaining:
        raise RuntimeError(f"Bot stop failed; still running: {final_remaining}")
    PID_FILE.unlink(missing_ok=True)
    if killed:
        return f"Bot stopped. SIGTERM sent to {initial_pids}; SIGKILL needed for {killed}."
    return f"Bot stopped cleanly. SIGTERM sent to {initial_pids}."


def _reset_statistics() -> str:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = LIVE_DIR / f"SCA_reset_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved_files = []
    for path in [CYCLES_CSV, ORDERS_CSV, ERRORS_CSV]:
        if path.exists():
            path.rename(archive_dir / path.name)
            moved_files.append(path.name)
    if not moved_files:
        archive_dir.rmdir()
        return "No statistics files found to reset."
    return f"Archived {len(moved_files)} file(s) to {archive_dir}"


def _clear_errors() -> str:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not ERRORS_CSV.exists() or ERRORS_CSV.stat().st_size == 0:
        return "No error log found to clear."
    archive_dir = LIVE_DIR / f"SCA_error_archive_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    ERRORS_CSV.rename(archive_dir / ERRORS_CSV.name)
    return f"Error log archived to {archive_dir}"


def _fmt_money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "-"


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except Exception:
        return "-"


def _status_badge(is_running: bool, last_cycle: pd.Series | None) -> str:
    if not is_running:
        return "STOPPED"
    if last_cycle is None:
        return "RUNNING - WAITING FOR DATA"
    if int(last_cycle.get("orders", 0)) > 0:
        return "RUNNING - ORDERS SENT"
    return "RUNNING - MONITORING"


def _process_conclusion(is_running: bool, last_cycle: pd.Series | None, errors: pd.DataFrame) -> tuple[str, str]:
    if not is_running:
        return (
            "Bot is not running",
            "The dashboard is online, but the paper-live bot process is not active.",
        )
    if last_cycle is None:
        return (
            "Bot is running, waiting for first cycle",
            "The process is alive, but no live cycle has been logged yet.",
        )
    if not errors.empty:
        return (
            "Bot is running with logged errors",
            "The process is alive, but the Errors tab contains entries to review.",
        )
    return (
        "Everything is running normally",
        "The dashboard is online, the paper-live bot process is active, and no errors are logged.",
    )


def _health_banner(is_running: bool, last_cycle: pd.Series | None, errors: pd.DataFrame) -> tuple[str, str, str]:
    if not is_running:
        return (
            "health-error",
            "ERROR DETECTED - ACTION NEEDED",
            "The dashboard is online, but the paper-live bot process is stopped.",
        )
    if not errors.empty:
        return (
            "health-error",
            "ERROR DETECTED - ACTION NEEDED",
            f"The bot is running, but {len(errors)} error log entry(s) need review.",
        )
    if last_cycle is None:
        return (
            "health-warning",
            "BOT STARTING - WAITING FOR DATA",
            "The bot process is running, but no live cycle has been logged yet.",
        )
    return (
        "health-ok",
        "ALL IS OKAY",
        "The bot is running, live cycles are logging, and no errors are currently recorded.",
    )


def main() -> None:
    st.set_page_config(
        page_title="SCA Paper Live",
        page_icon="SCA",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 1.5rem;}
        div[data-testid="stMetric"] {border: 1px solid #d8dee9; padding: 10px 12px; border-radius: 6px;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        h1, h2, h3 {letter-spacing: 0;}
        .health-banner {
            border-radius: 8px;
            color: #f8fafc;
            margin: 0.9rem 0 1rem;
            padding: 1.1rem 1.3rem;
            text-align: center;
        }
        .health-banner-title {
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: 0;
            line-height: 1.1;
            margin: 0;
            text-transform: uppercase;
        }
        .health-banner-body {
            font-size: 1rem;
            font-weight: 650;
            margin-top: 0.35rem;
        }
        .health-ok {
            background: linear-gradient(135deg, #047857 0%, #059669 55%, #065f46 100%);
            border: 1px solid rgba(110, 231, 183, 0.72);
            box-shadow: 0 0 26px rgba(5, 150, 105, 0.26);
        }
        .health-warning {
            background: linear-gradient(135deg, #92400e 0%, #d97706 55%, #78350f 100%);
            border: 1px solid rgba(253, 186, 116, 0.72);
            box-shadow: 0 0 26px rgba(217, 119, 6, 0.26);
        }
        .health-error {
            background: linear-gradient(135deg, #991b1b 0%, #dc2626 55%, #7f1d1d 100%);
            border: 1px solid rgba(252, 165, 165, 0.76);
            box-shadow: 0 0 28px rgba(220, 38, 38, 0.30);
        }
        .control-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.25rem 0 0.2rem;
        }
        .control-subtitle {
            color: #5a6577;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .control-kicker {
            color: #6b7280;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .control-card-title {
            color: #e9eef7;
            font-size: 1.03rem;
            font-weight: 800;
            margin: 0;
        }
        .control-card-text {
            color: #aab6c8;
            font-size: 0.85rem;
            height: 4.3rem;
            line-height: 1.35;
            margin: 0.45rem 0 0.7rem;
            overflow: hidden;
        }
        .control-chip {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            padding: 0.22rem 0.55rem;
            text-transform: uppercase;
        }
        .chip-start {background: rgba(17, 245, 173, 0.12); color: #35f5c6;}
        .chip-stop {background: rgba(255, 77, 121, 0.13); color: #ff7193;}
        .chip-reset {background: rgba(103, 166, 255, 0.14); color: #8bbcff;}
        .st-key-control_panel {
            background: linear-gradient(135deg, #101521 0%, #171c29 56%, #10151d 100%);
            border: 1px solid rgba(133, 151, 180, 0.26);
            border-radius: 8px;
            box-shadow: 0 18px 42px rgba(14, 20, 31, 0.24);
            padding: 1.05rem;
        }
        .st-key-control_tile_start,
        .st-key-control_tile_stop,
        .st-key-control_tile_reset {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            min-height: 15rem;
            padding: 1rem 1rem 0.85rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }
        .st-key-control_tile_start div[data-testid="stCheckbox"],
        .st-key-control_tile_stop div[data-testid="stCheckbox"],
        .st-key-control_tile_reset div[data-testid="stCheckbox"] {
            align-items: center;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 6px;
            display: flex;
            height: 2.75rem;
            margin: 0.2rem 0 0.75rem;
            padding: 0 0.65rem;
        }
        .st-key-control_tile_start div[data-testid="stCheckbox"] label,
        .st-key-control_tile_stop div[data-testid="stCheckbox"] label,
        .st-key-control_tile_reset div[data-testid="stCheckbox"] label {
            margin-bottom: 0;
        }
        .st-key-control_tile_start {
            background: radial-gradient(circle at 15% 0%, rgba(17, 245, 173, 0.18), transparent 30%), #111827;
        }
        .st-key-control_tile_stop {
            background: radial-gradient(circle at 15% 0%, rgba(255, 77, 121, 0.18), transparent 30%), #111827;
        }
        .st-key-control_tile_reset {
            background: radial-gradient(circle at 15% 0%, rgba(103, 166, 255, 0.20), transparent 30%), #111827;
        }
        .st-key-control_tile_start:hover,
        .st-key-control_tile_stop:hover,
        .st-key-control_tile_reset:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #071426 0%, #0b2344 58%, #06111f 100%);
            border-color: rgba(96, 165, 250, 0.62);
            box-shadow: 0 16px 34px rgba(2, 8, 23, 0.34), 0 0 18px rgba(37, 99, 235, 0.22);
        }
        .st-key-control_00_start_bot button,
        .st-key-control_02_stop_bot button,
        .st-key-control_04_reset_stats button {
            border-radius: 6px;
            font-weight: 850;
            letter-spacing: 0;
            min-height: 2.85rem;
            text-transform: uppercase;
            transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease, filter 160ms ease;
        }
        .st-key-control_00_start_bot button {
            background: #0ce6a8;
            border: 1px solid #89ffe0;
            box-shadow: 0 0 0 1px rgba(12, 230, 168, 0.24), 0 0 18px rgba(12, 230, 168, 0.38);
            color: #05130f;
        }
        .st-key-control_02_stop_bot button {
            background: #ff477d;
            border: 1px solid #ff9cb7;
            box-shadow: 0 0 0 1px rgba(255, 71, 125, 0.22), 0 0 18px rgba(255, 71, 125, 0.34);
            color: #20030b;
        }
        .st-key-control_04_reset_stats button {
            background: #5aa6ff;
            border: 1px solid #b4d6ff;
            box-shadow: 0 0 0 1px rgba(90, 166, 255, 0.22), 0 0 18px rgba(90, 166, 255, 0.34);
            color: #06111f;
        }
        .st-key-control_00_start_bot button:hover,
        .st-key-control_02_stop_bot button:hover,
        .st-key-control_04_reset_stats button:hover {
            background: #0b2344;
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.30), 0 0 22px rgba(37, 99, 235, 0.42);
            color: #dbeafe;
            filter: none;
            transform: translateY(-1px);
        }
        .st-key-control_00_start_bot button:disabled,
        .st-key-control_02_stop_bot button:disabled,
        .st-key-control_04_reset_stats button:disabled {
            background: #263040;
            border-color: rgba(148, 163, 184, 0.22);
            box-shadow: none;
            color: #778296;
        }
        .control-result {
            background: #07111f;
            border: 1px solid rgba(125, 211, 252, 0.34);
            border-radius: 8px;
            color: #c8f2ff;
            margin-bottom: 1rem;
            padding: 0.8rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cycles = _read_csv(CYCLES_CSV)
    orders = _read_csv(ORDERS_CSV)
    errors = _read_csv(ERRORS_CSV)
    is_running, process_text = _bot_process()
    last_cycle = cycles.iloc[-1] if not cycles.empty else None
    strategy_cfg = _strategy_settings()
    scan_interval_seconds = int(strategy_cfg.get("live_loop_interval_seconds", 900))
    scan_interval_minutes = scan_interval_seconds / 60.0
    next_scan_minutes = _minutes_until_next_scan(last_cycle, scan_interval_seconds)
    regime_proxy, scan_universe = _scan_universe()

    st.title("SCA Regime Momentum 01")
    st.caption("Alpaca paper-live monitoring dashboard")

    health_class, health_title, health_body = _health_banner(is_running, last_cycle, errors)
    st.markdown(
        f"""
        <div class="health-banner {health_class}">
            <div class="health-banner-title">{escape(health_title)}</div>
            <div class="health-banner-body">{escape(health_body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = _status_badge(is_running, last_cycle)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bot", status)
    c2.metric("Regime", "-" if last_cycle is None else str(last_cycle.get("regime", "-")))
    c3.metric("Target", "-" if last_cycle is None else _fmt_pct(last_cycle.get("gross_target_weight")))
    c4.metric("Equity", "-" if last_cycle is None else _fmt_money(last_cycle.get("equity")))
    c5.metric("Orders", 0 if orders.empty else len(orders))

    if last_cycle is not None:
        st.write(
            f"Last cycle: `{last_cycle.get('timestamp')}` | "
            f"Market open: `{last_cycle.get('market_open')}` | "
            f"Regime probability: `{float(last_cycle.get('regime_probability', 0.0)):.3f}`"
        )

    tab_status, tab_scan, tab_cycles, tab_orders, tab_errors, tab_control, tab_process = st.tabs(
        ["Status", "Scan", "Cycles", "Orders", "Errors", "Control", "Process"]
    )

    with tab_status:
        left, right = st.columns([2, 1])
        with left:
            if cycles.empty:
                st.info("No live cycles logged yet.")
            else:
                chart_df = cycles.copy()
                chart_df = chart_df.dropna(subset=["timestamp"])
                if not chart_df.empty:
                    chart_df = chart_df.set_index("timestamp")
                    st.line_chart(chart_df[["equity", "gross_target_weight"]])
                st.dataframe(cycles.tail(20), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Latest Values")
            if last_cycle is None:
                st.write("No cycle yet.")
            else:
                fields = [
                    "mode", "market_open", "cash", "buying_power", "bar_timestamp",
                    "next_open", "next_close",
                ]
                for field in fields:
                    st.write(f"**{field}**: `{last_cycle.get(field, '-')}`")

    with tab_scan:
        first_cycle = cycles.iloc[0] if not cycles.empty else None
        last_scan_at = None if last_cycle is None else last_cycle.get("timestamp")
        first_scan_at = None if first_cycle is None else first_cycle.get("timestamp")
        bar_timestamp = None if last_cycle is None else last_cycle.get("bar_timestamp")
        next_scan_label = "-" if next_scan_minutes is None else f"{next_scan_minutes:.1f} min"

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Assets", len(scan_universe))
        s2.metric("Interval", f"{scan_interval_minutes:.0f} min")
        s3.metric("Next Scan", next_scan_label)
        s4.metric("Regime Proxy", regime_proxy)

        st.write(
            f"Scan history from `{first_scan_at or '-'}` | "
            f"Last scan `{last_scan_at or '-'}` | "
            f"Last market bar `{bar_timestamp or '-'}`"
        )

        scan_rows = []
        for index, symbol in enumerate(scan_universe, start=1):
            scan_rows.append(
                {
                    "id": f"asset_{index:03d}",
                    "symbol": symbol,
                    "role": "regime_proxy + tradable" if symbol == regime_proxy else "tradable",
                    "asset_group": DEFAULT_ASSET_GROUP,
                    "last_scanned_at": last_scan_at,
                    "market_bar_timestamp": bar_timestamp,
                    "next_scan_in_minutes": None if next_scan_minutes is None else round(next_scan_minutes, 1),
                    "scan_interval_minutes": round(scan_interval_minutes, 1),
                }
            )
        st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)

    with tab_cycles:
        st.dataframe(cycles.tail(200), use_container_width=True, hide_index=True)

    with tab_orders:
        if orders.empty:
            st.info("No paper orders logged by this runner yet.")
        else:
            st.dataframe(orders.tail(200), use_container_width=True, hide_index=True)

    with tab_errors:
        if st.session_state.get("errors_03_last_result"):
            st.code(st.session_state["errors_03_last_result"], language="text")
        clear_col, spacer_col = st.columns([1, 3])
        with clear_col:
            confirm_clear_errors = st.checkbox("Confirm clear errors", key="errors_01_confirm_clear")
            if st.button(
                "Clear Errors",
                disabled=not confirm_clear_errors or errors.empty,
                use_container_width=True,
                key="errors_02_clear_errors",
            ):
                st.session_state["errors_03_last_result"] = _clear_errors()
                st.rerun()
        if errors.empty:
            st.success("No errors logged.")
        else:
            st.dataframe(errors.tail(200), use_container_width=True, hide_index=True)

    with tab_control:
        with st.container(key="control_panel"):
            st.markdown(
                """
                <div class="control-kicker">WSL paper-live command deck</div>
                <div class="control-title">Bot Control</div>
                <div class="control-subtitle">
                    Start, stop, or reset the paper-live telemetry without leaving the dashboard.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.get("control_05_last_result"):
                safe_result = escape(st.session_state["control_05_last_result"])
                st.markdown(
                    f"<div class='control-result'>{safe_result}</div>",
                    unsafe_allow_html=True,
                )

            start_col, stop_col, reset_col = st.columns(3, gap="medium")

            with start_col:
                with st.container(key="control_tile_start"):
                    start_state = "Already running" if is_running else "Ready to launch"
                    start_chip = "Running" if is_running else "Ready"
                    st.markdown(
                        f"""
                        <span class="control-chip chip-start">{start_chip}</span>
                        <p class="control-card-title">Start Bot</p>
                        <p class="control-card-text">
                            {start_state}. Launches the WSL paper-live loop only when no bot process is active.
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    confirm_start = st.checkbox("Confirm start", key="control_06_confirm_start")
                    if st.button(
                        "Start Bot",
                        type="primary",
                        use_container_width=True,
                        key="control_00_start_bot",
                        disabled=is_running or not confirm_start,
                    ):
                        st.session_state["control_05_last_result"] = _start_bot()
                        st.rerun()

            with stop_col:
                with st.container(key="control_tile_stop"):
                    st.markdown(
                        """
                        <span class="control-chip chip-stop">Guarded</span>
                        <p class="control-card-title">Stop Bot</p>
                        <p class="control-card-text">
                            Sends a stop signal to the paper-live process. Requires confirmation.
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    confirm_stop = st.checkbox("Confirm stop", key="control_01_confirm_stop")
                    if st.button(
                        "Stop Bot",
                        disabled=not confirm_stop or not is_running,
                        use_container_width=True,
                        key="control_02_stop_bot",
                    ):
                        st.session_state["control_05_last_result"] = _stop_bot()
                        st.rerun()

            with reset_col:
                with st.container(key="control_tile_reset"):
                    st.markdown(
                        """
                        <span class="control-chip chip-reset">Archive</span>
                        <p class="control-card-title">Reset Statistics</p>
                        <p class="control-card-text">
                            Moves current CSV telemetry into a timestamped archive folder.
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    confirm_reset = st.checkbox("Confirm reset", key="control_03_confirm_reset")
                    if st.button(
                        "Reset Statistics",
                        disabled=not confirm_reset,
                        use_container_width=True,
                        key="control_04_reset_stats",
                    ):
                        st.session_state["control_05_last_result"] = _reset_statistics()
                        st.rerun()

            st.caption(
                "Reset archives files under "
                "`savedresults/SCA_regime_momentum_01_live/SCA_reset_archive_*`."
            )

    with tab_process:
        conclusion_title, conclusion_body = _process_conclusion(is_running, last_cycle, errors)
        if is_running and errors.empty:
            st.success(f"{conclusion_title}. {conclusion_body}")
        elif is_running:
            st.warning(f"{conclusion_title}. {conclusion_body}")
        else:
            st.error(f"{conclusion_title}. {conclusion_body}")

        p1, p2, p3 = st.columns(3)
        p1.metric("Bot process", "Running" if is_running else "Stopped")
        p2.metric("Logged errors", 0 if errors.empty else len(errors))
        p3.metric("Last cycle", "-" if last_cycle is None else str(last_cycle.get("timestamp")))

        st.write("Running:", is_running)
        st.code(process_text or "No matching process found.", language="text")
        st.write("Log files:")
        st.code(
            "\n".join(
                [
                    str(CYCLES_CSV),
                    str(ORDERS_CSV),
                    str(ERRORS_CSV),
                ]
            ),
            language="text",
        )

    st.caption("Refresh the browser to update. Streamlit also reruns when you interact with the page.")


if __name__ == "__main__":
    main()
