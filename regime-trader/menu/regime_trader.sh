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

# ── active asset group (loaded from registry) ────────────────
# Registry: config/asset_groups.yaml (managed via `py -3.12 main.py groups`)
AVAILABLE_GROUPS=()
load_groups() {
    # mapfile from Python, fall back to "stocks" if Python fails
    mapfile -t AVAILABLE_GROUPS < <(py -3.12 main.py groups list --names-only 2>/dev/null)
    if [ ${#AVAILABLE_GROUPS[@]} -eq 0 ]; then
        AVAILABLE_GROUPS=("stocks")
    fi
}
load_groups
DEFAULT_GROUP="$(py -3.12 main.py groups default 2>/dev/null | head -n1)"
ASSET_GROUP="${DEFAULT_GROUP:-${AVAILABLE_GROUPS[0]}}"

# ── active config set ─────────────────────────────────────────
ACTIVE_SET_FILE="$ROOT/config/active_set"
get_active_set() {
    if [ -f "$ACTIVE_SET_FILE" ]; then
        cat "$ACTIVE_SET_FILE" | tr -d '[:space:]'
    else
        echo "base"
    fi
}

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
    echo -e "  ${YELLOW}Config set   : ${MAGENTA}$(get_active_set)${RESET}  ${DIM}(change with c)${RESET}"
    echo ""
}

print_menu() {
    echo -e "  ${YELLOW}── Trading ─────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[1]${RESET}  Train HMM        ${DIM}(fetch latest bars, fit model, save)${RESET}"
    echo -e "  ${GREEN}[2]${RESET}  Dry Run          ${DIM}(full pipeline — no orders placed)${RESET}"
    echo -e "  ${GREEN}[3]${RESET}  Live / Paper     ${DIM}(start trading loop)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Backtesting ─────────────────────────────${RESET}"
    echo -e "  ${GREEN}[0]${RESET}  Full Cycle       ${DIM}(HMM + backtest ALL 3 groups, summary table)${RESET}"
    echo -e "  ${GREEN}[4]${RESET}  Backtest Quick   ${DIM}(active group, 2020-now, no benchmark)${RESET}"
    echo -e "  ${GREEN}[5]${RESET}  Backtest Group   ${DIM}(active group, 2020-now, benchmark)${RESET}"
    echo -e "  ${GREEN}[6]${RESET}  Forward Test     ${DIM}(hold-out 2024-today, out-of-sample)${RESET}"
    echo ""
    echo -e "  ${GREEN}[7]${RESET}  Stress Test      ${DIM}(crash / gap / vol-spike scenarios, active group)${RESET}"
    echo -e "  ${GREEN}[8]${RESET}  Train + Backtest ${DIM}(retrain HMM then full benchmark, active group)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Analysis ────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[p]${RESET}  Interval Sweep   ${DIM}(find best min_rebalance_interval, active group)${RESET}"
    echo -e "  ${GREEN}[r]${RESET}  Conf×Stab Sweep  ${DIM}(2-D grid: min_confidence × stability_bars)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Config Sets ─────────────────────────────${RESET}"
    echo -e "  ${MAGENTA}[c]${RESET}  Config Set       ${DIM}(conservative | balanced | aggressive)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Asset Groups ────────────────────────────${RESET}"
    echo -e "  ${BLUE}[g]${RESET}  Change Group     ${DIM}(dynamic list from config/asset_groups.yaml)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Source Control ──────────────────────────${RESET}"
    echo -e "  ${MAGENTA}[s]${RESET}  Save & Push      ${DIM}(git commit all changes + push to GitHub)${RESET}"
    echo ""
    echo -e "  ${RED}[q]${RESET}  Quit"
    echo ""
    echo -e "  ${DIM}─────────────────────────────────────────${RESET}"
}

git_save() {
    echo ""
    echo -e "  ${CYAN}Current git status:${RESET}"
    echo ""
    git -C "$ROOT" status --short | sed 's/^/    /'
    echo ""

    # ── nothing to commit? ──────────────────────────────────────
    local has_changes
    has_changes=$(git -C "$ROOT" status --porcelain)
    if [ -z "$has_changes" ]; then
        echo -e "  ${GREEN}Working tree clean — nothing to commit.${RESET}"
        echo ""
        read -rp "  Push existing commits to GitHub anyway? [y/N]: " push_yn
        if [[ "$push_yn" =~ ^[Yy]$ ]]; then
            echo ""
            git -C "$ROOT" push origin main 2>&1 | sed 's/^/    /'
        fi
        return
    fi

    # ── prompt for commit message ───────────────────────────────
    local default_msg
    default_msg="update: $(date '+%Y-%m-%d %H:%M')"
    echo -e "  ${YELLOW}Commit message${RESET}  ${DIM}(press Enter to use: \"$default_msg\")${RESET}"
    read -rp "  > " commit_msg
    [ -z "$commit_msg" ] && commit_msg="$default_msg"

    echo ""
    echo -e "  ${DIM}Staging all changes...${RESET}"
    git -C "$ROOT" add -A

    echo -e "  ${DIM}Committing...${RESET}"
    git -C "$ROOT" commit -m "$commit_msg"
    local commit_rc=$?
    echo ""

    if [ $commit_rc -ne 0 ]; then
        echo -e "  ${RED}Commit failed (exit $commit_rc) — not pushing.${RESET}"
        return
    fi

    echo -e "  ${GREEN}Committed.${RESET}"
    echo ""

    # ── confirm push ────────────────────────────────────────────
    read -rp "  Push to GitHub (origin/main)? [Y/n]: " push_yn
    if [[ ! "$push_yn" =~ ^[Nn]$ ]]; then
        echo ""
        echo -e "  ${DIM}Pushing to origin/main...${RESET}"
        git -C "$ROOT" push origin main 2>&1 | sed 's/^/    /'
        local push_rc=${PIPESTATUS[0]}
        echo ""
        if [ $push_rc -eq 0 ]; then
            echo -e "  ${GREEN}Pushed successfully to origin/main.${RESET}"
        else
            echo -e "  ${RED}Push failed (exit $push_rc) — check credentials or network.${RESET}"
        fi
    fi
}

select_set() {
    echo ""
    echo -e "  ${YELLOW}Select config set:${RESET}"
    echo -e "  ${MAGENTA}[1]${RESET}  conservative  ${DIM}(capital preservation, no leverage, low churn)${RESET}"
    echo -e "  ${MAGENTA}[2]${RESET}  balanced      ${DIM}(fixes churn issues, realistic slippage — recommended)${RESET}"
    echo -e "  ${MAGENTA}[3]${RESET}  aggressive    ${DIM}(max deployment, 1.5x leverage, responsive)${RESET}"
    echo -e "  ${MAGENTA}[4]${RESET}  base          ${DIM}(raw settings.yaml, no overrides)${RESET}"
    echo ""
    read -rp "  Your choice: " schoice
    case "$schoice" in
        1) NEW_SET="conservative" ;;
        2) NEW_SET="balanced"     ;;
        3) NEW_SET="aggressive"   ;;
        4) NEW_SET="base"         ;;
        *) echo -e "  ${RED}Invalid — keeping '$(get_active_set)'${RESET}" ; sleep 1 ; return ;;
    esac
    if [ "$NEW_SET" = "base" ]; then
        echo -n "" > "$ACTIVE_SET_FILE"
    else
        echo "$NEW_SET" > "$ACTIVE_SET_FILE"
    fi
    echo -e "  ${GREEN}Config set switched to: $NEW_SET${RESET}"
    sleep 1
}

select_group() {
    load_groups
    echo ""
    echo -e "  ${YELLOW}Select asset group:${RESET}"
    local i=1
    for name in "${AVAILABLE_GROUPS[@]}"; do
        local syms
        syms="$(py -3.12 main.py groups show "$name" --json 2>/dev/null \
                 | py -3.12 -c "import sys,json; d=json.load(sys.stdin); g=list(d.values())[0]; print(' '.join(g['symbols'][:8]) + (' ...' if len(g['symbols'])>8 else ''))" 2>/dev/null)"
        [ -z "$syms" ] && syms=""
        echo -e "  ${BLUE}[$i]${RESET}  ${name}   ${DIM}(${syms})${RESET}"
        i=$((i+1))
    done
    echo -e "  ${DIM}(tip: add a group with: py -3.12 main.py groups add <name> --symbols A,B,C)${RESET}"
    echo ""
    read -rp "  Your choice: " gchoice
    if [[ "$gchoice" =~ ^[0-9]+$ ]]; then
        local idx=$((gchoice - 1))
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#AVAILABLE_GROUPS[@]}" ]; then
            ASSET_GROUP="${AVAILABLE_GROUPS[$idx]}"
            return
        fi
    fi
    echo -e "  ${RED}Invalid — keeping '$ASSET_GROUP'${RESET}" ; sleep 1
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
        0)
            run_command \
                "Full Cycle — HMM + Backtest for ALL 3 groups" \
                "py -3.12 main.py full-cycle --start 2020-01-01"
            ;;
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
                "Stress Test — group: $ASSET_GROUP  2020-now" \
                "PYTHONIOENCODING=utf-8 py -3.12 main.py stress --asset-group $ASSET_GROUP --start 2020-01-01"
            ;;
        8)
            run_command \
                "Train HMM + Backtest — group: $ASSET_GROUP" \
                "py -3.12 main.py trade --train-only --asset-group $ASSET_GROUP && py -3.12 main.py backtest --asset-group $ASSET_GROUP --start 2020-01-01 --compare"
            ;;
        p|P)
            run_command \
                "Interval Sweep — group: $ASSET_GROUP  2020-now" \
                "py -3.12 main.py sweep --asset-group $ASSET_GROUP --start 2020-01-01"
            ;;
        r|R)
            run_command \
                "Conf×Stab Grid Sweep — group: $ASSET_GROUP  2020-now" \
                "py -3.12 main.py cs-sweep --asset-group $ASSET_GROUP --start 2020-01-01"
            ;;
        s|S)
            git_save
            echo ""
            read -rp "  Press Enter to return to menu..."
            ;;
        c|C)
            select_set
            ;;
        g|G)
            select_group
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
