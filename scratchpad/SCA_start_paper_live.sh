#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/regime-trader-work"
LIVE_DIR="$ROOT/savedresults/SCA_regime_momentum_01_live"
PID_FILE="$LIVE_DIR/SCA_paper_live.pid"
STDOUT_LOG="$LIVE_DIR/SCA_paper_live_stdout.log"

mkdir -p "$LIVE_DIR"
cd "$ROOT"

if [[ -s "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "already_running pid=$old_pid"
    exit 0
  fi
fi

nohup .venv/bin/python scratchpad/SCA_regime_momentum_01_paper_live.py --loop --execute \
  > "$STDOUT_LOG" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"
echo "started pid=$pid"
