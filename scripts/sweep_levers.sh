#!/usr/bin/env bash
# scripts/sweep_levers.sh — backtest balanced (baseline) + 3 experimental
# sets, then aggregate results in one table for comparison.
#
# Each set is a pure config override. Zero touch to production code.
# Roll back: just delete the config/sets/exp_*.yaml files.
set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP_DIR="savedresults/sweep_levers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR"
echo

SETS=(balanced exp_high_conf exp_no_vix exp_tight_midvol)

for s in "${SETS[@]}"; do
    echo "[$(date +%H:%M:%S)] Running --set $s ..."
    if .venv/bin/python main.py backtest \
            --asset-group stocks \
            --start 2020-01-01 \
            --end   2026-04-26 \
            --set   "$s" > "$SWEEP_DIR/$s.log" 2>&1
    then
        LATEST=$(ls -td savedresults/backtest_* 2>/dev/null | head -1)
        cp "$LATEST/performance_summary.csv" "$SWEEP_DIR/$s.csv"
        cp "$LATEST/run_context.json"        "$SWEEP_DIR/$s.json"
        echo "  -> $LATEST"
    else
        echo "  FAILED — log at $SWEEP_DIR/$s.log"
    fi
done

echo
echo "[$(date +%H:%M:%S)] All backtests done. Results in $SWEEP_DIR"
