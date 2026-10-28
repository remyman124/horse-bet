"""Repair `races.track` from HKJC's per-race `Course :` cell.

EVIDENCE BASE
-------------
`sweep_per_race_course.py` fetches every race's own HKJC results page and
records that race's own `Course :` cell. Per-race is the correct granularity:

  * Each page carries THAT race's header (`RACE n (idx) Class x - DISTANCE`),
    its race title and its horse list. Verified: R2's page contains the horse
    the DB stores as R2, and not R1's horse.
  * A meeting does NOT have one course. On 2026-10-11 ST, R1 reports
    ALL WEATHER TRACK while R2 and R6 report TURF. Reading only race 1 and
    copying it across the meeting would corrupt 791 races.
  * The serving path (`build_racecard.py`) also reads per-race headers, so
    per-race labels are the only train/serve-consistent choice.
  * Across the AWT-labelled races, none carried a distance that cannot exist
    on the all-weather track, and the 7 turf-labelled anomalies are all
    real dual-surface trips (HV 1650m, ST 2200m/2400m).

WHY THE OLD LABELS ARE WRONG
----------------------------
They came from sniffing for the bare token 'AWT' anywhere in the page. HKJC
never writes that abbreviation on the results page -- it writes the surface
out in full ('ALL WEATHER TRACK'). Verified independently: 0 occurrences of
the token 'AWT' on sampled results pages, including pages whose course
really was all-weather. So every all-weather race was labelled 'Turf'.

`track_enc` is a model feature (features.py FORM_FEATURES), so this was
train/serve skew in the training data itself.

SAFETY RULES
------------
  * A label is applied ONLY where the race's own page stated a course.
  * Races whose page could not be read keep their current value.
  * The write is transactional and a backup is taken first.
  * A verification pass reads the DB back and reports disagreements.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/repair_track_labels.py          # dry run
    cd /home/work/horse-bet && .venv/bin/python scripts/repair_track_labels.py --apply  # write
"""
import argparse, collections, json, shutil, sqlite3, sys, time

DB = '/home/work/horse-bet/data/horse.db'
SWEEP = '/home/work/.hermes-isak/cache/scratch/per_race_sweep.jsonl'


def load_sweep():
    """race_id -> surface, for races where HKJC actually stated a course."""
    labels = {}
    with open(SWEEP) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            surf = rec.get('surface')
            if surf:                      # only real readings count
                labels[rec['race_id']] = surf
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    labels = load_sweep()
    print(f"per-race readings with a stated course: {len(labels)}")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    plan = []
    for rid, want in labels.items():
        row = conn.execute("select track from races where id=?", (rid,)).fetchone()
        if row is None:
            continue
        if row['track'] != want:
            plan.append((rid, row['track'], want))

    before = collections.Counter(r[0] for r in conn.execute("select track from races"))
    after = collections.Counter(before)
    for _, old, new in plan:
        after[old] -= 1
        after[new] += 1

    print(f"\nraces to change: {len(plan)}")
    for (o, n), c in collections.Counter((o, n) for _, o, n in plan).most_common():
        print(f"  {str(o):<6} -> {n:<6} : {c}")

    total = conn.execute("select count(*) from races").fetchone()[0]
    print(f"\n{'surface':<10}{'before':>9}{'after':>9}")
    for k in ('Turf', 'AWT', 'Dirt', None):
        if before[k] or after[k]:
            print(f"{str(k):<10}{before[k]:>9}{after[k]:>9}")
    print(f"{'TOTAL':<10}{total:>9}{total:>9}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    bak = f"{DB}.bak.track-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(DB, bak)
    print(f"\nbackup -> {bak}")

    conn.execute("BEGIN")
    conn.executemany("update races set track=? where id=?",
                     [(n, rid) for rid, _, n in plan])
    conn.commit()

    bad = 0
    for rid, want in labels.items():
        got = conn.execute("select track from races where id=?", (rid,)).fetchone()
        if got and got[0] != want:
            bad += 1
    print(f"\nverification: {len(labels)} labelled races, {bad} still disagreeing")
    print("final distribution:")
    for t, n in conn.execute("select track, count(*) from races group by track order by 2 desc"):
        print(f"  {t!r}: {n}")
    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
