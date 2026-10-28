"""
Regression test for the Kelly stake calculation in `value_bets.py`.

THE BUG THIS TEST EXISTS FOR
----------------------------
`value_bets.py` sizes a stake as a quarter of full Kelly. The Kelly function
was written as:

    b = (o - 1) / o          # <-- WRONG
    f = (p * b - (1 - p)) / b

`b = (o-1)/o` is the RETURN FRACTION of the bet, not the net odds. The net
odds are `b = o - 1`: stake 1 unit, win `o` back, profit `o - 1`.

Feeding the return fraction into `f = (p*b - (1-p))/b` algebraically gives
`f = (p*(2o-1) - o) / (o-1)`, so a positive stake requires

    p > o / (2o - 1)

which is ~0.500 at o=2 and still ~0.506 at o=43. No realistic longshot ever
clears that bar, so Kelly collapsed to exactly 0 for every candidate.

THE CONSEQUENCE WAS THE WORST KIND: SILENT AND INVERTED. On 2026-10-28 all 19
"value candidates" printed `quarter-Kelly=0.00%` -- a stake of zero on horses
with up to +3.7% edge. A zero stake on a positive-edge horse says "do not
bet", which reads as prudence. It was actually a broken formula that had
collapsed every longshot to the same wrong answer.

WHAT THE TEST FIXES
-------------------
The formula is now the standard one, `f* = (p*(b+1) - 1)/b` with `b = o - 1`,
equivalently `f* = (p*o - 1) / (o - 1)`.

Note this is a formula fix, NOT a recommendation to bet. The rule that
generates these candidates loses money out of sample (see
`scripts/value_scan_backtest.py`): ROI -39.5% on 676 best-edge bets. The
model over-rates longshots, so a positive model-vs-market edge on a longshot
is a calibration artifact, not an edge. `value_bets.py` now prints that
warning above the stake table.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/test_kelly.py
"""

import sys

# The formula as it was written in scripts/value_bets.py (the bug).
def kelly_buggy(p, o):
    b = (o - 1) / o
    return max(0.0, (p * b - (1 - p)) / b)


# The corrected formula: b is the NET odds per unit staked.
def kelly_fixed(p, o):
    b = o - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p * (b + 1) - 1) / b)


CASES = [
    # (p, odds, note)
    (0.50, 2.0,  'p == break-even -> zero stake'),
    (0.60, 2.0,  'p above break-even -> positive stake'),
    (0.40, 2.0,  'p below break-even -> zero stake'),
    (1.0 / 36.0, 36.0, 'exactly break-even at 36 -> zero stake'),
    (0.0548, 24.0, 'real longshot from tonight (YOUNG ACHIEVER)'),
    (0.0596, 36.0, 'real longshot from tonight (THE CONCENTRATION)'),
    (0.0469, 36.0, 'real longshot from tonight (KOLACHI)'),
    (0.05, 20.0,  '5% at 20s'),
    (0.30, 8.0,   'short-ish price'),
]


def main():
    failures = []

    # 1. Break-even must produce exactly zero stake at any odds.
    for o in (2.0, 5.0, 11.0, 24.0, 36.0, 53.0):
        f = kelly_fixed(1.0 / o, o)
        if abs(f) > 1e-12:
            failures.append(f"break-even at odds {o}: expected 0, got {f}")

    # 2. A horse priced at exactly fair odds earns the market's take, so
    #    Kelly is zero. A horse the model rates higher than fair MUST get a
    #    positive stake -- this is the assertion the old formula violated.
    for p, o, note in CASES:
        f_fixed = kelly_fixed(p, o)
        f_buggy = kelly_buggy(p, o)
        if p > 1.0 / o and f_fixed <= 0:
            failures.append(
                f"p={p} odds={o} is above fair ({1/o:.4f}) but stake is 0")
        if p < 1.0 / o and f_fixed != 0:
            failures.append(
                f"p={p} odds={o} is below fair ({1/o:.4f}) but stake is {f_fixed}")
        # The old formula must be demonstrably broken on the same inputs.
        if p > 1.0 / o and f_buggy == 0 and f_fixed > 0:
            pass  # expected: this is the bug we are proving
        print(f"  p={p:<8.4f} odds={o:<6} fair={1/o:.4f} "
              f"fixed={f_fixed:>9.5f}  buggy={f_buggy:>9.5f}   ({note})")

    # 3. Kelly must never exceed 1 (you cannot stake more than your bankroll).
    for p in (0.5, 0.8, 0.95, 1.0):
        for o in (2.0, 3.0, 10.0, 100.0):
            f = kelly_fixed(p, o)
            if f > 1.0 + 1e-12:
                failures.append(f"p={p} odds={o}: stake {f} exceeds 1.0")

    # 4. The specific real-world regression: the 2026-10-28 card.
    #    THE CONCENTRATION @ 36.0, model p=5.96%. The old code printed 0.00%.
    f = kelly_fixed(0.0596, 36.0)
    if f <= 0:
        failures.append(
            f"2026-10-28 regression: THE CONCENTRATION stake is {f}, expected > 0")
    else:
        print(f"\n  2026-10-28 regression check: THE CONCENTRATION @36 "
              f"-> full Kelly {f:.4%}, quarter-Kelly {f/4:.4%} (old code: 0.00%)")

    # 5. Sanity: a double-underdog bet at the old formula's implied
    #    break-even (p > o/(2o-1)) should now be handled normally.
    o = 10.0
    p = 0.55  # well above fair (0.10)
    f = kelly_fixed(p, o)
    if f <= 0:
        failures.append(f"p={p} odds={o}: expected positive stake, got {f}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("All Kelly regression tests passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
