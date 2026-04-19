#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Tailscale File Transfer — interactive menu
#  Run from anywhere:  bash menu/tailscale_transfer.sh
#
#  Features
#   • Live status table of all tailnet peers (online / offline / ping)
#   • Taildrop send (preferred)
#   • SCP send (fallback)
#   • Ephemeral HTTP server send (most reliable, bypasses Taildrop)
#   • Tailscale SSH (interactive shell on peer)
#   • Receive via `tailscale file get`
#   • Diagnostics (ping, status, version)
#   • Rename this machine on the tailnet
# ─────────────────────────────────────────────────────────────

set -u

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
GREY='\033[0;37m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# ── locate tailscale binary (Win/Linux/Mac) ──────────────────
TS_BIN=""
find_ts() {
    if command -v tailscale >/dev/null 2>&1; then
        TS_BIN="tailscale"
        return 0
    fi
    local candidates=(
        "/c/Program Files/Tailscale/tailscale.exe"
        "/mnt/c/Program Files/Tailscale/tailscale.exe"
        "C:/Program Files/Tailscale/tailscale.exe"
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
        "/usr/bin/tailscale"
        "/usr/local/bin/tailscale"
    )
    for p in "${candidates[@]}"; do
        if [ -x "$p" ] || [ -f "$p" ]; then
            TS_BIN="$p"
            return 0
        fi
    done
    return 1
}
if ! find_ts; then
    echo -e "${RED}ERROR:${RESET} tailscale binary not found. Install Tailscale first."
    exit 1
fi

ts() { "$TS_BIN" "$@"; }

# ── status cache ─────────────────────────────────────────────
declare -a PEER_IP=()
declare -a PEER_NAME=()
declare -a PEER_USER=()
declare -a PEER_OS=()
declare -a PEER_STATUS=()   # online | offline | self
SELF_NAME=""
SELF_IP=""

refresh_status() {
    PEER_IP=(); PEER_NAME=(); PEER_USER=(); PEER_OS=(); PEER_STATUS=()
    SELF_NAME=""; SELF_IP=""

    local raw
    raw="$(ts status 2>/dev/null)"
    if [ -z "$raw" ]; then
        return 1
    fi

    local first=1
    while IFS= read -r line; do
        # skip empty / comment / health lines
        [ -z "$line" ] && continue
        case "$line" in
            \#*) continue;;
        esac
        # tailscale status columns: ip name user os status...
        local ip name user os status_raw
        ip="$(awk '{print $1}'  <<<"$line")"
        name="$(awk '{print $2}' <<<"$line")"
        user="$(awk '{print $3}' <<<"$line")"
        os="$(awk '{print $4}'   <<<"$line")"
        status_raw="$(awk '{for(i=5;i<=NF;i++) printf "%s ", $i; print ""}' <<<"$line" | sed 's/ *$//')"

        # sanity
        case "$ip" in 100.*) ;; *) continue;; esac

        local st="online"
        if echo "$status_raw" | grep -qi "offline"; then
            st="offline"
        fi
        if [ "$first" -eq 1 ]; then
            st="self"
            SELF_NAME="$name"
            SELF_IP="$ip"
            first=0
        fi

        PEER_IP+=("$ip")
        PEER_NAME+=("$name")
        PEER_USER+=("$user")
        PEER_OS+=("$os")
        PEER_STATUS+=("$st")
    done <<<"$raw"
}

# quick ping (returns 0 if pong within ~2 s)
quick_ping() {
    local target="$1"
    timeout 3 "$TS_BIN" ping --c=1 --until-direct=false "$target" >/dev/null 2>&1
}

print_header() {
    clear
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║    TAILSCALE  FILE  TRANSFER  MENU       ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo -e "  ${DIM}repo: $ROOT${RESET}"
    echo -e "  ${DIM}ts  : $TS_BIN${RESET}"
    echo ""
}

print_status_table() {
    echo -e "  ${YELLOW}── Tailnet status ──────────────────────────────────${RESET}"
    printf "  ${BOLD}%-4s %-16s %-16s %-8s %-10s${RESET}\n" "#" "Machine" "IP" "OS" "Status"
    printf "  ${DIM}%-4s %-16s %-16s %-8s %-10s${RESET}\n" "---" "----------------" "----------------" "--------" "----------"
    local i
    for i in "${!PEER_NAME[@]}"; do
        local idx="$((i+1))"
        local st="${PEER_STATUS[$i]}"
        local badge
        case "$st" in
            self)    badge="${CYAN}● SELF${RESET}"   ;;
            online)  badge="${GREEN}● online${RESET}" ;;
            offline) badge="${RED}○ offline${RESET}" ;;
            *)       badge="${GREY}? $st${RESET}"   ;;
        esac
        printf "  %-4s %-16s %-16s %-8s ${badge}\n" \
            "[$idx]" "${PEER_NAME[$i]}" "${PEER_IP[$i]}" "${PEER_OS[$i]}"
    done
    echo ""
}

pick_peer() {
    # $1 = prompt ; sets PICKED_NAME / PICKED_IP ; returns 1 if cancelled
    local prompt="$1"
    PICKED_NAME=""; PICKED_IP=""
    echo -ne "  ${YELLOW}${prompt}${RESET} "
    read -r raw
    [ -z "$raw" ] && return 1
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        local idx="$((raw-1))"
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#PEER_NAME[@]}" ]; then
            PICKED_NAME="${PEER_NAME[$idx]}"
            PICKED_IP="${PEER_IP[$idx]}"
            return 0
        fi
    else
        # lookup by name
        local i
        for i in "${!PEER_NAME[@]}"; do
            if [ "${PEER_NAME[$i]}" = "$raw" ]; then
                PICKED_NAME="$raw"
                PICKED_IP="${PEER_IP[$i]}"
                return 0
            fi
        done
    fi
    echo -e "  ${RED}Invalid selection.${RESET}"
    return 1
}

prompt_path() {
    # $1 = prompt, $2 = default (optional) ; sets PICKED_PATH
    local prompt="$1" default="${2:-}"
    PICKED_PATH=""
    if [ -n "$default" ]; then
        echo -ne "  ${YELLOW}${prompt}${RESET} ${DIM}[${default}]${RESET} "
    else
        echo -ne "  ${YELLOW}${prompt}${RESET} "
    fi
    read -r raw
    PICKED_PATH="${raw:-$default}"
    [ -n "$PICKED_PATH" ]
}

pause_return() {
    echo ""
    echo -ne "  ${DIM}Press [Enter] to return to menu...${RESET} "
    read -r _
}

# ── actions ──────────────────────────────────────────────────

action_taildrop_send() {
    print_header
    print_status_table
    prompt_path "Local file to send:" || { pause_return; return; }
    local f="$PICKED_PATH"
    if [ ! -f "$f" ]; then
        echo -e "  ${RED}File not found: $f${RESET}"; pause_return; return
    fi
    pick_peer "Destination peer (# or name):" || { pause_return; return; }
    if [ "$PICKED_NAME" = "$SELF_NAME" ]; then
        echo -e "  ${RED}Cannot send to self.${RESET}"; pause_return; return
    fi
    echo -e "  ${DIM}Sending via Taildrop → ${PICKED_NAME}${RESET}"
    ts file cp "$f" "${PICKED_NAME}:"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}✓ Sent.${RESET} On ${PICKED_NAME} run: ${CYAN}sudo tailscale file get ~/${RESET}"
    else
        echo -e "  ${RED}✗ Failed (exit $rc).${RESET} Try option 3 (HTTP) as fallback."
    fi
    pause_return
}

action_scp_send() {
    print_header
    print_status_table
    prompt_path "Local file to send:" || { pause_return; return; }
    local f="$PICKED_PATH"
    [ ! -f "$f" ] && { echo -e "  ${RED}File not found.${RESET}"; pause_return; return; }
    pick_peer "Destination peer (# or name):" || { pause_return; return; }
    echo -ne "  ${YELLOW}Remote user:${RESET} ${DIM}[morningtrading]${RESET} "
    read -r user; user="${user:-morningtrading}"
    prompt_path "Remote path:" "~/$(basename "$f")" || { pause_return; return; }
    local dst="$PICKED_PATH"
    echo -e "  ${DIM}scp → ${user}@${PICKED_NAME}:${dst}${RESET}"
    scp "$f" "${user}@${PICKED_NAME}:${dst}"
    local rc=$?
    [ $rc -eq 0 ] && echo -e "  ${GREEN}✓ Sent.${RESET}" || echo -e "  ${RED}✗ Failed.${RESET} Check SSH on ${PICKED_NAME}."
    pause_return
}

action_http_send() {
    print_header
    print_status_table
    prompt_path "Local file or directory to serve:" || { pause_return; return; }
    local src="$PICKED_PATH"
    [ ! -e "$src" ] && { echo -e "  ${RED}Path not found.${RESET}"; pause_return; return; }

    local serve_dir serve_name
    if [ -d "$src" ]; then
        serve_dir="$src"
        serve_name=""
    else
        serve_dir="$(dirname "$src")"
        serve_name="$(basename "$src")"
    fi

    local port=8765
    echo -ne "  ${YELLOW}Port${RESET} ${DIM}[${port}]${RESET} "
    read -r p; [ -n "$p" ] && port="$p"

    echo ""
    echo -e "  ${GREEN}HTTP server ready.${RESET}"
    echo -e "  On the peer, run:"
    if [ -n "$serve_name" ]; then
        echo -e "     ${CYAN}curl -O http://${SELF_NAME}:${port}/${serve_name}${RESET}"
        echo -e "     ${CYAN}curl -O http://${SELF_IP}:${port}/${serve_name}${RESET}"
    else
        echo -e "     ${CYAN}curl -O http://${SELF_NAME}:${port}/<filename>${RESET}"
        echo -e "     Index: ${CYAN}http://${SELF_IP}:${port}/${RESET}"
    fi
    echo -e "  ${DIM}Ctrl+C to stop the server.${RESET}"
    echo ""
    ( cd "$serve_dir" && py -3.12 -m http.server "$port" 2>&1 ) || \
        ( cd "$serve_dir" && python3 -m http.server "$port" 2>&1 )
    echo -e "  ${DIM}Server stopped.${RESET}"
    pause_return
}

action_taildrop_receive() {
    print_header
    echo -e "  ${YELLOW}── Taildrop receive (pull queued files) ────────────${RESET}"
    prompt_path "Destination directory:" "$HOME" || { pause_return; return; }
    local dst="$PICKED_PATH"
    mkdir -p "$dst"
    ts file get "$dst"
    local rc=$?
    [ $rc -eq 0 ] && echo -e "  ${GREEN}✓ Received (check $dst).${RESET}" || \
        echo -e "  ${RED}✗ Nothing to receive / error.${RESET}"
    ls -la "$dst" | tail -20
    pause_return
}

action_ssh_open() {
    print_header
    print_status_table
    pick_peer "Peer to SSH into (# or name):" || { pause_return; return; }
    echo -ne "  ${YELLOW}Remote user:${RESET} ${DIM}[morningtrading]${RESET} "
    read -r user; user="${user:-morningtrading}"
    echo -e "  ${DIM}Launching tailscale ssh ${user}@${PICKED_NAME}...${RESET}"
    ts ssh "${user}@${PICKED_NAME}"
    pause_return
}

action_ping() {
    print_header
    print_status_table
    pick_peer "Peer to ping (# or name):" || { pause_return; return; }
    echo -e "  ${DIM}tailscale ping ${PICKED_NAME} ...${RESET}"
    ts ping --c=4 "$PICKED_NAME"
    pause_return
}

action_full_status() {
    print_header
    echo -e "  ${YELLOW}── Full tailscale status ───────────────────────────${RESET}"
    ts status
    echo ""
    echo -e "  ${YELLOW}── Version & IP ───────────────────────────────────${RESET}"
    ts version | head -3
    echo ""
    echo -e "  ${YELLOW}── DNS / netcheck ──────────────────────────────────${RESET}"
    ts netcheck 2>&1 | head -30
    pause_return
}

action_connectivity_matrix() {
    print_header
    print_status_table
    echo -e "  ${YELLOW}── Connectivity matrix (ping every online peer) ───${RESET}"
    printf "  ${BOLD}%-16s %-10s %s${RESET}\n" "Peer" "Result" "Path"
    printf "  ${DIM}%-16s %-10s %s${RESET}\n" "----------------" "----------" "-----------------"
    local i
    for i in "${!PEER_NAME[@]}"; do
        local st="${PEER_STATUS[$i]}" name="${PEER_NAME[$i]}"
        [ "$st" = "self" ]    && { printf "  %-16s ${CYAN}%-10s${RESET} %s\n" "$name" "SELF" "-" ; continue; }
        [ "$st" = "offline" ] && { printf "  %-16s ${RED}%-10s${RESET} %s\n" "$name" "offline" "-" ; continue; }
        local out path rc
        out="$(timeout 5 "$TS_BIN" ping --c=1 --timeout=3s "$name" 2>&1)"
        rc=$?
        if [ $rc -eq 0 ]; then
            path="$(echo "$out" | grep -oE 'via [A-Za-z0-9()-]+' | head -1)"
            [ -z "$path" ] && path="direct"
            printf "  %-16s ${GREEN}%-10s${RESET} %s\n" "$name" "✓ reach" "$path"
        else
            printf "  %-16s ${RED}%-10s${RESET} %s\n" "$name" "✗ fail" "timeout"
        fi
    done
    pause_return
}

action_rename_self() {
    print_header
    echo -e "  ${YELLOW}── Rename this machine on the tailnet ──────────────${RESET}"
    echo -e "  Current name: ${CYAN}${SELF_NAME}${RESET}"
    echo -ne "  ${YELLOW}New hostname (lowercase, no spaces):${RESET} "
    read -r newname
    [ -z "$newname" ] && { echo -e "  ${DIM}cancelled.${RESET}"; pause_return; return; }
    ts set --hostname="$newname"
    local rc=$?
    [ $rc -eq 0 ] && echo -e "  ${GREEN}✓ Hostname set to '$newname'.${RESET} (may require reconnect)" || \
        echo -e "  ${RED}✗ Failed.${RESET}"
    pause_return
}

action_quick_send_credentials() {
    # Preset : push config/credentials.yaml + .env (if present) to a peer
    print_header
    print_status_table
    echo -e "  ${YELLOW}── Quick send: regime-trader secrets ───────────────${RESET}"
    pick_peer "Destination peer (# or name):" || { pause_return; return; }
    local sent=0
    for f in "config/credentials.yaml" ".env" "config/active_set"; do
        if [ -f "$ROOT/$f" ]; then
            echo -e "  ${DIM}→ $f${RESET}"
            ts file cp "$ROOT/$f" "${PICKED_NAME}:"
            if [ $? -eq 0 ]; then
                echo -e "    ${GREEN}✓ sent${RESET}"
                sent=$((sent+1))
            else
                echo -e "    ${RED}✗ failed${RESET}"
            fi
        else
            echo -e "  ${DIM}· skip $f (not present)${RESET}"
        fi
    done
    echo ""
    echo -e "  ${GREEN}${sent} file(s) queued.${RESET}"
    echo -e "  On ${PICKED_NAME}: ${CYAN}sudo tailscale file get ~/${RESET}"
    pause_return
}

# ── main loop ────────────────────────────────────────────────
while true; do
    refresh_status
    print_header
    print_status_table

    echo -e "  ${YELLOW}── Send ────────────────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[1]${RESET}  Taildrop send      ${DIM}(pick any file → peer)${RESET}"
    echo -e "  ${GREEN}[2]${RESET}  SCP send           ${DIM}(requires SSH on peer)${RESET}"
    echo -e "  ${GREEN}[3]${RESET}  HTTP server send   ${DIM}(most reliable, peer pulls)${RESET}"
    echo -e "  ${GREEN}[4]${RESET}  Quick secrets push ${DIM}(.env + credentials.yaml + active_set)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Receive / Interact ──────────────────────────────${RESET}"
    echo -e "  ${GREEN}[5]${RESET}  Taildrop receive   ${DIM}(pull queued files here)${RESET}"
    echo -e "  ${GREEN}[6]${RESET}  Tailscale SSH      ${DIM}(open shell on peer)${RESET}"
    echo ""
    echo -e "  ${YELLOW}── Diagnostics ─────────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[7]${RESET}  Ping a peer"
    echo -e "  ${GREEN}[8]${RESET}  Connectivity matrix ${DIM}(ping all online peers)${RESET}"
    echo -e "  ${GREEN}[9]${RESET}  Full status + netcheck"
    echo ""
    echo -e "  ${YELLOW}── Settings ────────────────────────────────────────${RESET}"
    echo -e "  ${GREEN}[r]${RESET}  Rename this machine (${CYAN}${SELF_NAME}${RESET})"
    echo -e "  ${GREEN}[R]${RESET}  Refresh status"
    echo -e "  ${GREEN}[q]${RESET}  Quit"
    echo ""
    echo -ne "  ${YELLOW}Choice:${RESET} "
    read -r choice
    case "$choice" in
        1) action_taildrop_send ;;
        2) action_scp_send ;;
        3) action_http_send ;;
        4) action_quick_send_credentials ;;
        5) action_taildrop_receive ;;
        6) action_ssh_open ;;
        7) action_ping ;;
        8) action_connectivity_matrix ;;
        9) action_full_status ;;
        r|rename) action_rename_self ;;
        R|refresh) ;; # loop re-refreshes automatically
        q|Q|quit|exit) echo -e "  ${DIM}bye.${RESET}"; exit 0 ;;
        *) echo -e "  ${RED}Unknown choice.${RESET}"; sleep 1 ;;
    esac
done
