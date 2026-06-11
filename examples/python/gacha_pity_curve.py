"""Analytical gacha pity curve using Talos-XII tensors (no simulator API).

Models cumulative P(at least one 6★ within n paid pulls) under a simplified
soft-pity ramp. Useful as a standalone probability toy; not wired to
data/config.json pool rates.

Run:
  cargo run --features python -- python examples/python/gacha_pity_curve.py
  cargo run --features python -- python examples/python/gacha_pity_curve.py -- 90 0.008 0.5
"""

import functools
import sys

import talos_xii as tx

print = functools.partial(print, flush=True)


def per_pull_rate(pulls, base_rate, soft_start, soft_bonus):
    """Piecewise rate: base until soft_start, then linear ramp to soft_bonus."""
    idx = tx.arange(0, pulls, 1.0)  # 0 .. pulls-1
    t = tx.clamp((idx - float(soft_start)) / max(pulls - soft_start, 1), 0.0, 1.0)
    return base_rate + soft_bonus * t


def survival_curve(rates):
    """P(no 6 star yet after each pull) via cumulative product of (1 - rate_i)."""
    survival = 1.0
    hit_by = []
    for rate in rates.to_list():
        survival *= 1.0 - rate
        hit_by.append(1.0 - survival)
    return hit_by


def main():
    pulls = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    base_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.008
    soft_bonus = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    soft_start = 74

    print(f"talos_xii={tx.version()}")
    print(f"pulls={pulls} base={base_rate} soft_start={soft_start} soft_bonus={soft_bonus}")

    rates = per_pull_rate(pulls, base_rate, soft_start, soft_bonus)
    hit_by = survival_curve(rates)

    milestones = [1, 30, 60, soft_start, pulls - 1, pulls - 1]
    seen = set()
    print("P(at least one 6 star by pull n):")
    for n in milestones:
        if n < 0 or n >= pulls or n in seen:
            continue
        seen.add(n)
        print(f"  pull {n + 1:3d}: {hit_by[n] * 100:.2f}%")

    expected = rates.sum().item()
    print(f"sum of per-pull rates (rough intensity): {expected:.3f}")


if __name__ == "__main__":
    main()
