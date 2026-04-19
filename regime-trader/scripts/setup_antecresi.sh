#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Bootstrap script for Linux tailnet nodes (antecresi, permanent…)
#  Installs: repo clone/pull + `transfert` bash alias
#  Safe to re-run (idempotent).
# ─────────────────────────────────────────────────────────────
set -e

REPO_URL="${REPO_URL:-https://github.com/morningtrading/regime-trader}"
REPO_DIR="${REPO_DIR:-$HOME/regime-trader}"

echo "▶ Cloning / updating $REPO_URL → $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "▶ Installing 'transfert' alias in ~/.bashrc"
ALIAS_LINE="alias transfert='cd $REPO_DIR && bash menu/tailscale_transfer.sh'"
if grep -qF "bash menu/tailscale_transfer.sh" "$HOME/.bashrc" 2>/dev/null; then
    # Replace existing line
    sed -i "\|bash menu/tailscale_transfer.sh|c\\$ALIAS_LINE" "$HOME/.bashrc"
    echo "  (existing alias line replaced)"
else
    echo "$ALIAS_LINE" >> "$HOME/.bashrc"
    echo "  (alias appended)"
fi

echo ""
echo "✓ Done."
echo "  Run:  source ~/.bashrc  &&  transfert"
