#!/usr/bin/env bash
# scripts/sweep_groups.sh — run a backtest for each asset group and aggregate
# performance_summary.csv + run_context.json into one sweep dir for comparison.
#
# Usage:
#   bash scripts/sweep_groups.sh                # uses active_set
#   bash scripts/sweep_groups.sh balanced       # forces --set balanced
#   bash scripts/sweep_groups.sh aggressive
set -euo pipefail
cd "$(dirname "$0")/.."

SET_NAME="${1:-}"
SET_TAG="${SET_NAME:-active}"
SWEEP_DIR="savedresults/sweep_groups_${SET_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR  (set=$SET_TAG)"

BASKETS=(stocks stocks2 stocks3 stocks4 stocks_gold indices)

SET_ARGS=()
if [ -n "$SET_NAME" ]; then
    SET_ARGS=(--set "$SET_NAME")
fi

for g in "${BASKETS[@]}"; do
    echo "[$(date +%H:%M:%S)] $g ..."
    if .venv/bin/python main.py backtest \
            --asset-group "$g" \
            --start 2020-01-01 \
            --end   2026-04-26 \
            "${SET_ARGS[@]}" \
            --compare > "$SWEEP_DIR/$g.log" 2>&1
    then
        LATEST=$(ls -td savedresults/backtest_* 2>/dev/null | head -1)
        cp "$LATEST/performance_summary.csv" "$SWEEP_DIR/$g.csv" 2>/dev/null || true
        cp "$LATEST/run_context.json"        "$SWEEP_DIR/$g.json" 2>/dev/null || true
        echo "  -> $LATEST"
    else
        echo "  FAILED — log at $SWEEP_DIR/$g.log"
    fi
done

echo "[$(date +%H:%M:%S)] DONE → $SWEEP_DIR"
