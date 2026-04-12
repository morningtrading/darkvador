#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Regime Trader — launcher menu
#  Run from the repo root:  bash menu/regime_trader.sh
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT" || exit 1

# ── colours ──────────────────────────────────────────────────
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
RED='\033[1;31m'
DIM='\033[2m'
RESET='\033[0m'

# ── active asset group (default: stocks) ─────────────────────
ASSET_GROUP="stocks"

print_header() {
    clear
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║         REGIME  TRADER               ║"
    echo "  ║   HMM Volatility Regime System       ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${RESET}"
    echo -e "  ${DIM}Working directory: $ROOT${RESET}"
    echo -e "  ${YELLOW}Active group : ${CYAN}${ASSET_GROUP}${RESET}  ${DIM}(change with g)${RESET}"
    echo ""
}

print_menu() {
    echo -e "  ${YELLOW}── Trading ─────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[1]${RESET}  Train HMM        ${DIM}(fetch latest bars, fit model, save)${RESET}"
    echo -e "  ${GREEN}[2]${RESET}  Dry Run          ${DIM}(full pipeline — no orders placed)${RESET}"
    echo -e "  ${GREEN}[3]${RESET}  Live / Paper     ${DIM}(start trading loop)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Backtesting ─────────────────────────────${RESET}"
    echo -e "  ${GREEN}[4]${RESET}  Backtest Quick   ${DIM}(active group, 2020-now, no benchmark)${RESET}"
    echo -e "  ${GREEN}[5]${RESET}  Backtest Group   ${DIM}(active group, 2020-now, benchmark)${RESET}"
    echo -e "  ${GREEN}[6]${RESET}  Forward Test     ${DIM}(hold-out 2024-today, out-of-sample)${RESET}"
    echo ""
    echo -e "  ${GREEN}[8]${RESET}  Train + Backtest ${DIM}(retrain HMM then full benchmark, active group)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Optimisation ────────────────────────────${RESET}"
    echo -e "  ${GREEN}[7]${RESET}  Param Sweep      ${DIM}(tune on 2020-2023, active group)${RESET}"
    echo -e "  ${GREEN}[9]${RESET}  Rolling WFO      ${DIM}(12m tune / 3m test, 4:1 ratio, robustness check)${RESET}"
    echo -e "  ${GREEN}[w]${RESET}  WFO Windows      ${DIM}(show fold schedule and bar counts — no backtests run)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Asset Groups ────────────────────────────${RESET}"
    echo -e "  ${BLUE}[g]${RESET}  Change Group     ${DIM}(stocks | crypto | indices)${RESET}"
    echo ""
    echo -e "  ${RED}[q]${RESET}  Quit"
    echo ""
    echo -e "  ${DIM}─────────────────────────────────────────${RESET}"
}

select_group() {
    echo ""
    echo -e "  ${YELLOW}Select asset group:${RESET}"
    echo -e "  ${BLUE}[1]${RESET}  stocks   ${DIM}(SPY QQQ AAPL MSFT AMZN GOOGL NVDA META TSLA AMD)${RESET}"
    echo -e "  ${BLUE}[2]${RESET}  crypto   ${DIM}(BTC ETH SOL AVAX DOGE LTC LINK UNI)${RESET}"
    echo -e "  ${BLUE}[3]${RESET}  indices  ${DIM}(SPY QQQ DIA IWM GLD TLT EFA EEM VNQ USO)${RESET}"
    echo ""
    read -rp "  Your choice: " gchoice
    case "$gchoice" in
        1) ASSET_GROUP="stocks"  ;;
        2) ASSET_GROUP="crypto"  ;;
        3) ASSET_GROUP="indices" ;;
        *) echo -e "  ${RED}Invalid — keeping '$ASSET_GROUP'${RESET}" ; sleep 1 ;;
    esac
}

run_command() {
    local label="$1"
    local cmd="$2"
    echo ""
    echo -e "  ${CYAN}>> $label${RESET}"
    echo -e "  ${DIM}$cmd${RESET}"
    echo -e "  ${DIM}─────────────────────────────────────────${RESET}"
    echo ""
    eval "$cmd"
    local exit_code=$?
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "  ${GREEN}Done (exit 0)${RESET}"
    else
        echo -e "  ${RED}Exited with code $exit_code${RESET}"
    fi
    echo ""
    read -rp "  Press Enter to return to menu..."
}

while true; do
    print_header
    print_menu
    read -rp "  Your choice: " choice
    case "$choice" in
        1)
            run_command \
                "Train HMM — group: $ASSET_GROUP" \
                "py -3.12 main.py trade --train-only --asset-group $ASSET_GROUP"
            ;;
        2)
            run_command \
                "Dry Run — group: $ASSET_GROUP" \
                "py -3.12 main.py trade --dry-run --asset-group $ASSET_GROUP"
            ;;
        3)
            run_command \
                "Live / Paper Trade — group: $ASSET_GROUP" \
                "py -3.12 main.py trade --asset-group $ASSET_GROUP"
            ;;
        4)
            run_command \
                "Backtest group: $ASSET_GROUP  2020-now (no benchmark, fast)" \
                "py -3.12 main.py backtest --asset-group $ASSET_GROUP --start 2020-01-01"
            ;;
        5)
            run_command \
                "Backtest group: $ASSET_GROUP  2020-now (benchmark)" \
                "py -3.12 main.py backtest --asset-group $ASSET_GROUP --start 2020-01-01 --compare"
            ;;
        6)
            run_command \
                "Forward Test — group: $ASSET_GROUP  2024-today (hold-out)" \
                "py -3.12 main.py backtest --asset-group $ASSET_GROUP --start 2024-01-01 --compare"
            ;;
        7)
            run_command \
                "Parameter Sweep — group: $ASSET_GROUP  tune 2020-2023" \
                "py -3.12 tools/param_sweep.py --asset-group $ASSET_GROUP --start 2020-01-01 --end 2023-12-31"
            ;;
        8)
            run_command \
                "Train HMM + Backtest — group: $ASSET_GROUP" \
                "py -3.12 main.py trade --train-only --asset-group $ASSET_GROUP && py -3.12 main.py backtest --asset-group $ASSET_GROUP --start 2020-01-01 --compare"
            ;;
        9)
            run_command \
                "Rolling WFO — group: $ASSET_GROUP  (6m tune / 3m test)" \
                "py -3.12 tools/rolling_wfo.py --asset-group $ASSET_GROUP --start 2020-01-01 --tune-months 12 --test-months 3 --step-months 3"
            ;;
        g|G)
            select_group
            ;;
        w|W)
            run_command \
                "WFO Window Plan — group: $ASSET_GROUP" \
                "py -3.12 tools/rolling_wfo.py --asset-group $ASSET_GROUP --start 2020-01-01 --tune-months 12 --test-months 3 --step-months 3 --show-windows"
            ;;
        q|Q)
            echo ""
            echo -e "  ${DIM}Goodbye.${RESET}"
            echo ""
            exit 0
            ;;
        *)
            echo -e "  ${RED}Invalid option '$choice' — try again.${RESET}"
            sleep 1
            ;;
    esac
done
