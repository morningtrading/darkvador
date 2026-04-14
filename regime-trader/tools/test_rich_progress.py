"""
Quick sanity-check for the Rich-based fold progress display.
Run with:  py -3.12 tools/test_rich_progress.py
"""
import sys
import time

from rich.console import Console
from rich.text import Text

# Use ASCII-safe bar chars so cp1252 / legacy terminals don't blow up.
_BAR_FULL  = "#"
_BAR_EMPTY = "-"
_CHECK     = "OK"


def _progress_line(fold_id: int, n_total: int, phase: str, info: dict) -> Text:
    _W = 20
    if phase == "training":
        pct = int(fold_id / n_total * 100)
    else:
        pct = int((fold_id + 1) / n_total * 100)

    filled = pct * _W // 100
    bar_text = Text()
    bar_text.append(_BAR_FULL * filled, style="cyan")
    bar_text.append(_BAR_EMPTY * (_W - filled), style="dim white")

    line = Text()
    line.append("  [")
    line.append_text(bar_text)
    line.append("] ")
    line.append(f"{pct:3d}%", style="bold white")
    line.append(f"  Fold {fold_id+1}/{n_total}  ")

    if phase == "training":
        line.append("Training ...", style="yellow")
        line.append(f"  OOS {info['oos_start']} -> {info['oos_end']}", style="dim")
    else:
        line.append(f"{_CHECK} ", style="bold green")
        line.append(f"{info['oos_start']} -> {info['oos_end']}", style="green")
        line.append(f"  {info['n_states']} states", style="dim cyan")
        line.append(f"  {info['fold_trades']} tr", style="dim")
        equity_style = "bold green" if info["equity"] >= info["start_equity"] else "bold red"
        line.append(f"  ${info['equity']:>10,.0f}", style=equity_style)

    return line


def main():
    console = Console(force_terminal=True, legacy_windows=False, width=130)
    n_folds = 5
    start_equity = 100_000

    fake_folds = [
        {"oos_start": "2020-01-02", "oos_end": "2020-06-30", "n_states": 4, "fold_trades": 72},
        {"oos_start": "2020-07-01", "oos_end": "2020-12-31", "n_states": 3, "fold_trades": 58},
        {"oos_start": "2021-01-04", "oos_end": "2021-06-30", "n_states": 5, "fold_trades": 91},
        {"oos_start": "2021-07-01", "oos_end": "2021-12-31", "n_states": 4, "fold_trades": 63},
        {"oos_start": "2022-01-03", "oos_end": "2022-06-30", "n_states": 4, "fold_trades": 47},
    ]
    equities = [107_432, 103_218, 119_874, 115_201, 98_340]

    print()
    for fold_id, fold in enumerate(fake_folds):
        # Show "training" line (overwrites itself with \r)
        info_train = {**fold, "equity": start_equity if fold_id == 0 else equities[fold_id - 1]}
        line = _progress_line(fold_id, n_folds, "training", info_train)
        console.print(line, end="\r", highlight=False, no_wrap=True, overflow="crop")
        time.sleep(0.4)

        # Show "complete" line (persists with newline)
        info_done = {**fold, "equity": equities[fold_id], "start_equity": start_equity}
        line = _progress_line(fold_id, n_folds, "complete", info_done)
        console.print(line, highlight=False, no_wrap=True, overflow="crop")
        time.sleep(0.2)

    print()


if __name__ == "__main__":
    main()
