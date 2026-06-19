#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/regime-trader-work"
cd "$ROOT"

exec .venv/bin/streamlit run dashboard/SCA_streamlit_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --browser.gatherUsageStats false
