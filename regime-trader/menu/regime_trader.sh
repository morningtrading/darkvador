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
declare -A GROUP_PREVIEW   # name -> "SYM1 SYM2 ... (N symbols)"
load_groups() {
    mapfile -t AVAILABLE_GROUPS < <(py -3.12 main.py groups list --names-only 2>/dev/null)
    if [ ${#AVAILABLE_GROUPS[@]} -eq 0 ]; then
        AVAILABLE_GROUPS=("stocks")
    fi
    # Build preview map in one shot (one python call, not one per group)
    GROUP_PREVIEW=()
    while IFS=$'\t' read -r _gname _gprev; do
        [ -n "$_gname" ] && GROUP_PREVIEW["$_gname"]="$_gprev"
    done < <(py -3.12 main.py groups list --json 2>/dev/null | py -3.12 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for name, g in d.get('groups', {}).items():
        s = g.get('symbols', [])
        preview = ' '.join(s[:6]) + (' ...' if len(s) > 6 else '')
        print(f'{name}\t{preview} ({len(s)} symbols)')
except Exception:
    pass
")
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
    echo -e "  ${YELLOW}── Help ────────────────────────────────────${RESET}"
    echo -e "  ${CYAN}[h]${RESET}  Detailed Help    ${DIM}(all commands + live config values)${RESET}"
    echo ""
    echo -e "  ${RED}[q]${RESET}  Quit"
    echo ""
    echo -e "  ${DIM}─────────────────────────────────────────${RESET}"
}

show_help() {
    # Dynamically builds a help page from:
    #   - config/asset_groups.yaml  (via `main.py groups list --json`)
    #   - config/settings.yaml      (all sections + key params)
    #   - argparse `--help`         (flags per subcommand)
    local active_set
    active_set="$(get_active_set)"
    clear
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║              REGIME TRADER — DETAILED HELP                   ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo -e "  ${DIM}Config set: ${MAGENTA}${active_set}${RESET}${DIM}   │   Active group: ${CYAN}${ASSET_GROUP}${RESET}"
    echo ""

    # ── Menu commands ─────────────────────────────────────────────
    echo -e "${YELLOW}══ MENU COMMANDS ═══════════════════════════════════════════════${RESET}"
    echo ""
    local groups_csv paper_flag rebal_iv
    groups_csv="$(py -3.12 main.py groups list --names-only 2>/dev/null | paste -sd, - | sed 's/,/, /g')"
    paper_flag="$(py -3.12 -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['broker']['paper_trading'])" 2>/dev/null)"
    rebal_iv="$(py -3.12 -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['backtest']['min_rebalance_interval'])" 2>/dev/null)"

    echo -e "  ${GREEN}[0]${RESET} Full Cycle      → py -3.12 main.py full-cycle --start 2020-01-01"
    echo -e "       Trains HMM and backtests ${CYAN}every${RESET} registered asset group,"
    echo -e "       then prints a consolidated summary table."
    echo -e "       Iterates: ${groups_csv}"
    echo ""
    echo -e "  ${GREEN}[1]${RESET} Train HMM       → py -3.12 main.py trade --train-only --asset-group ${ASSET_GROUP}"
    echo -e "       Fetches latest bars, fits the HMM, saves to models/hmm.pkl, exits."
    echo ""
    echo -e "  ${GREEN}[2]${RESET} Dry Run         → py -3.12 main.py trade --dry-run --asset-group ${ASSET_GROUP}"
    echo -e "       Full live pipeline (signals, sizing, risk) but places ${RED}no orders${RESET}."
    echo ""
    echo -e "  ${GREEN}[3]${RESET} Live / Paper    → py -3.12 main.py trade --asset-group ${ASSET_GROUP}"
    echo -e "       Real trading loop. broker.paper_trading = ${paper_flag}."
    echo ""
    echo -e "  ${GREEN}[4]${RESET} Backtest Quick  → py -3.12 main.py backtest --asset-group ${ASSET_GROUP} --start 2020-01-01"
    echo -e "       Walk-forward backtest, no benchmark (faster)."
    echo ""
    echo -e "  ${GREEN}[5]${RESET} Backtest Group  → py -3.12 main.py backtest --asset-group ${ASSET_GROUP} --start 2020-01-01 --compare"
    echo -e "       Walk-forward backtest + benchmark comparison table."
    echo ""
    echo -e "  ${GREEN}[6]${RESET} Forward Test    → py -3.12 main.py backtest --asset-group ${ASSET_GROUP} --start 2024-01-01 --compare"
    echo -e "       Hold-out out-of-sample test from 2024."
    echo ""
    echo -e "  ${GREEN}[7]${RESET} Stress Test     → py -3.12 main.py stress --asset-group ${ASSET_GROUP} --start 2020-01-01"
    echo -e "       Crash / gap / vol-spike scenario suite."
    echo ""
    echo -e "  ${GREEN}[8]${RESET} Train+Backtest  → retrains HMM then runs Backtest (with benchmark)."
    echo ""
    # Pull current HMM params + default grids from argparse
    local cur_conf cur_stab sweep_default_grid cs_conf_grid cs_stab_grid
    cur_conf="$(py -3.12 -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['hmm']['min_confidence'])" 2>/dev/null)"
    cur_stab="$(py -3.12 -c "import yaml; print(yaml.safe_load(open('config/settings.yaml'))['hmm']['stability_bars'])" 2>/dev/null)"
    sweep_default_grid="$(py -3.12 main.py sweep --help 2>/dev/null | awk -F'default:' '/--values/ || /default:/{print $2}' | tr -d ')' | xargs | head -c80)"
    cs_conf_grid="$(py -3.12 main.py cs-sweep --help 2>/dev/null | awk '/--conf/{flag=1} flag && /default:/{print; exit}' | sed -E 's/.*default: ?//' | tr -d ')' | xargs)"
    cs_stab_grid="$(py -3.12 main.py cs-sweep --help 2>/dev/null | awk '/--stab/{flag=1} flag && /default:/{print; exit}' | sed -E 's/.*default: ?//' | tr -d ')' | xargs)"

    echo -e "  ${GREEN}[p]${RESET} Interval Sweep  → py -3.12 main.py sweep --asset-group ${ASSET_GROUP} --start 2020-01-01"
    echo -e "       ${DIM}Purpose:${RESET} find the best ${CYAN}min_rebalance_interval${RESET} — the number of bars"
    echo -e "       to lock out further rebalances after one fires. 0 = off (rebalance any time);"
    echo -e "       higher values suppress churn but delay reaction to regime changes."
    echo -e "       ${DIM}Currently in settings.yaml:${RESET} backtest.min_rebalance_interval = ${YELLOW}${rebal_iv}${RESET}"
    echo -e "       ${DIM}Default grid tested:${RESET} 0, 1, 2, 3, 5, 7, 10, 15, 20 bars"
    echo -e "       ${DIM}Override:${RESET} --values 0,3,5,8,12"
    echo -e "       ${DIM}Output:${RESET} ranked table (Sharpe / return / #trades / max DD per interval)"
    echo -e "       ${DIM}When to run:${RESET} after changing symbols / HMM params — the optimum drifts."
    echo ""
    echo -e "  ${GREEN}[r]${RESET} Conf×Stab Sweep → py -3.12 main.py cs-sweep --asset-group ${ASSET_GROUP} --start 2020-01-01"
    echo -e "       ${DIM}Purpose:${RESET} 2-D grid search over two HMM gating thresholds:"
    echo -e "         ${CYAN}min_confidence${RESET}  — posterior-probability floor to trust a regime call"
    echo -e "                           (currently: ${YELLOW}${cur_conf}${RESET})"
    echo -e "         ${CYAN}stability_bars${RESET} — consecutive bars the regime must persist before acting"
    echo -e "                           (currently: ${YELLOW}${cur_stab}${RESET})"
    echo -e "       ${DIM}Default grids:${RESET}"
    echo -e "         conf = ${cs_conf_grid:-0.55, 0.60, 0.65, 0.70, 0.75}"
    echo -e "         stab = ${cs_stab_grid:-3, 5, 7, 9, 12}"
    echo -e "       ${DIM}Override:${RESET} --conf 0.55,0.65,0.75  --stab 3,5,7"
    echo -e "       ${DIM}Trains once per fold${RESET} and evaluates the full grid on the same walk-forward"
    echo -e "       slices — much faster than N × M independent backtests."
    echo -e "       ${DIM}Output:${RESET} CSV heatmap sorted by Sharpe desc; pick the (conf, stab) cell"
    echo -e "       with the best Sharpe ${DIM}and${RESET} trade-count (avoid cells that trade < 10 times)."
    echo -e "       ${DIM}When to run:${RESET} after [p] — these three params interact."
    echo ""
    echo -e "  ${MAGENTA}[c]${RESET} Config Set      → switches preset (conservative | balanced | aggressive)."
    echo -e "  ${BLUE}[g]${RESET} Change Group    → picks active asset group (dynamic list)."
    echo -e "  ${MAGENTA}[s]${RESET} Save & Push     → git add -A, commit, push to origin/main."
    echo ""

    # ── Asset groups (live) ──────────────────────────────────────
    echo -e "${YELLOW}══ ASSET GROUPS (config/asset_groups.yaml) ═════════════════════${RESET}"
    py -3.12 main.py groups list 2>/dev/null | sed 's/^/  /'
    echo ""
    echo -e "  ${DIM}Manage: py -3.12 main.py groups <list|show|add|remove|edit|rename|set-default|validate|export|import>${RESET}"
    echo ""

    # ── Parameters (live, from settings.yaml) ────────────────────
    echo -e "${YELLOW}══ PARAMETERS (config/settings.yaml, set: ${MAGENTA}${active_set}${YELLOW}) ═══════════════${RESET}"
    py -3.12 -c "
import yaml
cfg = yaml.safe_load(open('config/settings.yaml'))
for section, body in cfg.items():
    if not isinstance(body, dict):
        print(f'  {section}: {body}')
        continue
    print(f'  [{section}]')
    for k, v in body.items():
        if isinstance(v, list) and len(v) > 6:
            shown = ', '.join(str(x) for x in v[:6]) + f' ... ({len(v)} items)'
        elif isinstance(v, list):
            shown = ', '.join(str(x) for x in v)
        else:
            shown = v
        print(f'    {k:<30} = {shown}')
    print()
" 2>/dev/null

    # ── CLI flags per subcommand (live, from argparse) ───────────
    echo -e "${YELLOW}══ CLI FLAGS PER SUBCOMMAND (py -3.12 main.py <cmd> --help) ════${RESET}"
    echo ""
    for cmd in trade backtest stress full-cycle sweep cs-sweep groups; do
        echo -e "  ${GREEN}▸ ${cmd}${RESET}"
        py -3.12 main.py "$cmd" --help 2>/dev/null \
            | awk '/^  -/{print "    " $0}' \
            | sed 's/^    --/    --/'
        echo ""
    done

    echo -e "  ${DIM}─────────────────────────────────────────${RESET}"
    read -rp "  Press Enter to return to menu..."
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

    local idx name num preview
    for idx in "${!AVAILABLE_GROUPS[@]}"; do
        name="${AVAILABLE_GROUPS[$idx]}"
        num=$((idx + 1))
        preview="${GROUP_PREVIEW[$name]:-}"
        echo -e "  ${BLUE}[${num}]${RESET}  ${name}   ${DIM}${preview}${RESET}"
    done
    echo -e "  ${DIM}(tip: add a group with: py -3.12 main.py groups add <name> --symbols A,B,C)${RESET}"
    echo ""
    read -rp "  Your choice: " gchoice
    if [[ "$gchoice" =~ ^[0-9]+$ ]]; then
        local choose_idx=$((gchoice - 1))
        if [ "$choose_idx" -ge 0 ] && [ "$choose_idx" -lt "${#AVAILABLE_GROUPS[@]}" ]; then
            ASSET_GROUP="${AVAILABLE_GROUPS[$choose_idx]}"
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
        h|H)
            show_help
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
