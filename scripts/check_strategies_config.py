"""Quick sanity check: print which strategies are enabled in settings.yaml
after the merge with the active config set."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_config


def main() -> int:
    cfg = load_config()
    strats = cfg.get("strategies", {}) or {}
    enabled = [n for n, s in strats.items() if s.get("enabled", False)]

    print("All strategies in settings.yaml (post-active-set merge):")
    for name, scfg in strats.items():
        flag = "ENABLED" if scfg.get("enabled", False) else "disabled"
        print(f"  {name:<30}  {flag}")

    print(f"\nEnabled count: {len(enabled)}")
    print(f"  >= 2 → multi-strategy mode would activate at next live launch.")

    if "mean_reversion_qqq_spy" in strats:
        mr = strats["mean_reversion_qqq_spy"]
        print("\nmean_reversion_qqq_spy entry:")
        for k, v in mr.items():
            print(f"  {k:<16} {v}")
    else:
        print("\nmean_reversion_qqq_spy NOT in strategies — entry missing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
