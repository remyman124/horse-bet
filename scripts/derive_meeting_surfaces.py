"""Derive the meeting-level surface from PHYSICAL constraints, not from the
per-race `Course :` cell.

WHY THIS EXISTS
---------------
The per-race sweep (sweep_per_race_course.py) read HKJC's `Course :` cell for
all 13,471 races. The readings looked self-consistent: ZERO distance
violations. But that check was TOO WEAK. At Sha Tin, 1200m is a valid trip on
BOTH the turf course and the all-weather track, so a spurious AWT label on a
1200m race passes the check. The check only bites on 1000/1400/1600/2000m.

And the per-race readings produced 383 "mixed" meetings -- 25-30% of Sha Tin
meetings in EVERY year including 2024, 2025 and 2026. HKJC runs a whole
meeting on ONE surface. That is not real, so the per-race cell is not fully
reliable on its own.

THE ACTUAL CONSTRAINT
---------------------
The race DISTANCE determines which surface a meeting must be on:

  Sha Tin turf only : 1000, 1400, 1600, 2000   (+ 1200, 1800, 2200, 2400)
  Sha Tin AWT only  : 1650                      (+ 1200, 1800, 2200, 2400)
  Happy Valley      : always turf, 1200/1650/1800/2200/2400

So:
  * a meeting containing ANY turf-only distance is a Turf meeting
  * a meeting containing 1650m and no turf-only distance is an AWT meeting
  * a meeting with only 1200/1800/2200/2400 is ambiguous on distance alone,
    and the MAJORITY of the per-race `Course :` cells breaks the tie

This derives the surface from the meeting's own distance profile -- which
comes from HKJC's per-race page and was independently confirmed against the
serving-path card (11/11 agreement for 2026-10-19 ST).

It also matches what the SERVING path produces: the racecard page header is
meeting-consistent (all 9 races AWT for 2026-10-28 ST, all 11 Turf for
2026-10-19 ST), so training labels must be meeting-consistent too.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/derive_meeting_surfaces.py
"""
import collections
import json
import sqlite3

DB = '/home/work/horse-bet/data/horse.db'
SWEEP = '/home/work/.hermes-isak/cache/scratch/per_race_sweep.jsonl'
OUT = '/home/work/.hermes-isak/cache/scratch/meeting_surfaces.json'

# turf-only trips and AWT-only trips, by venue
TURF_ONLY = {'ST': {1000, 1400, 1600, 2000}, 'HV': set()}
AWT_ONLY = {'ST': {1650}, 'HV': set()}
# races whose distance works on either surface
NEUTRAL = {1200, 1800, 2200, 2400}


def load_sweep():
    cells = {}
    for line in open(SWEEP):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get('surface'):
            cells[rec['race_id']] = rec['surface']
    return cells


def main():
    cells = load_sweep()
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "select r.id, rd.race_date, rd.venue, r.race_no, r.track, r.distance "
        "from races r join race_days rd on r.race_day_id=rd.id").fetchall()
    conn.close()

    meetings = collections.defaultdict(list)
    for rid, date, venue, race_no, track, dist in rows:
        meetings[(date, venue)].append((rid, race_no, track, dist))

    derived = {}
    reasons = collections.Counter()
    mixed_by_cell = 0
    cell_vs_derived = collections.Counter()

    for key, races in meetings.items():
        date, venue = key
        dists = {int(d) for _, _, _, d in races if d}
        t_only = dists & TURF_ONLY.get(venue, set())
        a_only = dists & AWT_ONLY.get(venue, set())
        cell_counts = collections.Counter(cells[rid] for rid, _, _, _ in races if rid in cells)
        cell_awt = cell_counts.get('AWT', 0)
        cell_turf = cell_counts.get('Turf', 0)
        if len(cell_counts) > 1:
            mixed_by_cell += 1

        if venue == 'HV':
            surf, why = 'Turf', 'HV is always turf'
        elif t_only:
            surf, why = 'Turf', f'turf-only distance {sorted(t_only)}'
        elif a_only:
            surf, why = 'AWT', f'AWT-only distance {sorted(a_only)}'
        else:
            surf = 'AWT' if cell_awt > cell_turf else 'Turf'
            why = f'ambiguous distance, cell majority ({cell_awt} AWT / {cell_turf} Turf)'
        derived[key] = surf
        reasons[why.split(' ')[0] if surf == 'Turf' else 'AWT'] += 1

        for rid, race_no, track, dist in races:
            c = cells.get(rid)
            if c:
                cell_vs_derived[(c, surf)] += 1

    print('=' * 76)
    print('A) MEETING SURFACE DERIVED FROM DISTANCE PROFILE')
    print('=' * 76)
    dist = collections.Counter(derived.values())
    for k, v in dist.most_common():
        print(f'  {k:<6} meetings: {v}')

    print('\n  meetings where the per-race cells DISAGREE with each other: '
          f'{mixed_by_cell} of {len(meetings)}')

    print('\n' + '=' * 76)
    print('B) HOW THE PER-RACE CELL COMPARES TO THE DERIVED MEETING SURFACE')
    print('=' * 76)
    for (cell, derived_s), n in cell_vs_derived.most_common():
        tag = 'agrees' if cell == derived_s else 'DISAGREES'
        print(f'  cell={cell:<5} derived={derived_s:<5} n={n:<6} {tag}')

    total = sum(cell_vs_derived.values())
    agree = sum(n for (c, d), n in cell_vs_derived.items() if c == d)
    print(f'\n  agreement: {agree}/{total} = {agree/total*100:.2f}%')
    print(f'  disagreements: {total - agree} '
          f'({(total-agree)/total*100:.2f}%)')

    print('\n' + '=' * 76)
    print('C) THE MIXED MEETINGS, RESOLVED')
    print('=' * 76)
    shown = 0
    for key, races in sorted(meetings.items()):
        cc = collections.Counter(cells[rid] for rid, _, _, _ in races if rid in cells)
        if len(cc) < 2:
            continue
        dists = sorted({int(d) for _, _, _, d in races if d})
        if shown < 6:
            print(f'  {key[0]} {key[1]}: cells={dict(cc)} distances={dists}')
            print(f'      -> DERIVED: {derived[key]}')
            shown += 1

    print('\n' + '=' * 76)
    print('D) WHAT THE REPAIR WOULD CHANGE (meeting-level surface)')
    print('=' * 76)
    plan = []
    for key, races in meetings.items():
        want = derived[key]
        for rid, race_no, track, dist in races:
            if track != want:
                plan.append((rid, track, want))
    moves = collections.Counter((o, n) for _, o, n in plan)
    for (o, n), c in moves.most_common():
        print(f'  {str(o):<6} -> {n:<6} : {c}')
    print(f'\n  races to change: {len(plan)} of {len(rows)}')

    with open(OUT, 'w') as f:
        json.dump({f'{d}|{v}': s for (d, v), s in derived.items()}, f, indent=1)
    print(f'\n  meeting surfaces -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
