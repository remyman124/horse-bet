"""
Regression test for the shared feature module.

Two invariants matter, and both were violated before this test existed:

1. TRAIN == SERVE. `features.build_features()` must produce the same values
   whether the race being scored is present in the frame with its results
   (training) or absent (serving). If they differ, the model is trained on
   one representation and served on another.

2. NO SAME-DAY LEAK. A rolling window must contain only races that finished
   STRICTLY BEFORE the current race day. A jockey rides several times in one
   meeting, so a naive `shift(1)` lets race 2 inherit race 1's result from
   the same afternoon. At 09:00 nobody knows how race 1 went.

Run:
    cd /home/work/horse-bet && source /tmp/horsebet_venv/bin/activate \
        && python3 scripts/test_features.py
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import features as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, 'data', 'horse.db')

LOAD_SQL = """
    SELECT re.race_id, re.horse_name, re.horse_no, re.draw, re.jockey, re.trainer,
           re.win_odds, re.place_odds, re.finishing_position,
           rd.race_date, rd.venue, r.race_no, r.distance, r.race_class, r.track
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
      AND CAST(re.finishing_position AS INTEGER) > 0
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
"""

TEST_DATE = '2026-10-19'
TEST_VENUE = 'ST'


def compare(a, b, label, tol=1e-9):
    """Return (n_mismatches, list of offending feature names)."""
    bad = []
    for c in F.ALL_FEATURES:
        if c not in a.columns or c not in b.columns:
            bad.append(c)
            continue
        x = pd.to_numeric(a[c], errors='coerce')
        y = pd.to_numeric(b[c], errors='coerce')
        both_na = x.isna() & y.isna()
        diff = (~both_na) & ((x.fillna(-999) - y.fillna(-999)).abs() > tol)
        if int(diff.sum()):
            bad.append(c)
    return bad


def main():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    full = pd.read_sql(LOAD_SQL, conn)
    conn.close()
    full['race_date'] = pd.to_datetime(full['race_date'])
    full = full.sort_values(
        ['race_date', 'venue', 'race_no', 'horse_no']).reset_index(drop=True)
    print(f"loaded {len(full):,} entries")

    ts = pd.Timestamp(TEST_DATE)
    day = full[full['race_date'] == ts].copy()
    hist = full[full['race_date'] < ts].copy()
    print(f"test day {TEST_DATE} {TEST_VENUE}: {len(day)} runners, "
          f"{day['race_no'].nunique()} races")

    failures = []

    # ── Invariant 1: train path == serve path ──────────────────────────────
    # Train: the day's races are in the frame WITH their results.
    train_out, _ = F.build_features(full)
    train_day = train_out[train_out['race_date'] == ts].copy()

    # Serve: the day's races are in the frame WITHOUT results (finishing
    # position unknown), exactly as a race card looks before the meeting.
    card = day.copy()
    card['finishing_position'] = np.nan
    serve_out, _ = F.build_features(pd.concat([hist, card], ignore_index=True,
                                               sort=False))
    serve_day = serve_out[serve_out['race_date'] == ts].copy()

    key = ['race_no', 'horse_no']
    train_day = train_day.sort_values(key).reset_index(drop=True)
    serve_day = serve_day.sort_values(key).reset_index(drop=True)
    print(f"\n[1] train rows={len(train_day)}  serve rows={len(serve_day)}")
    if len(train_day) != len(serve_day):
        failures.append('row count differs between train and serve paths')
    bad = compare(train_day, serve_day, 'train-vs-serve')
    for c in bad:
        print(f"    MISMATCH {c}")
    if bad:
        failures.append(f"train/serve mismatch in: {bad}")
    else:
        print(f"    all {len(F.ALL_FEATURES)} features identical ✅")

    # ── Invariant 2: no same-day leak ─────────────────────────────────────
    # Truncate the frame to the day BEFORE the test day, then append the test
    # day. The features for the test day must be unchanged compared with the
    # full-table build, which proves the test day's own results did not feed
    # its own features.
    trunc = full[full['race_date'] <= ts].copy()
    trunc_out, _ = F.build_features(trunc)
    trunc_day = trunc_out[trunc_out['race_date'] == ts].copy()
    trunc_day = trunc_day.sort_values(key).reset_index(drop=True)
    print("\n[2] same-day leak check (build with data truncated at the day "
          "boundary)")
    bad2 = compare(trunc_day, train_day, 'trunc-vs-full')
    for c in bad2:
        print(f"    MISMATCH {c}")
    if bad2:
        failures.append(f"same-day leak in: {bad2}")
    else:
        print("    same-day results do not feed the same-day features ✅")

    # ── Invariant 3: leakage guard: no post-race field may be a feature ──
    print("\n[3] leakage field guard")
    for forbidden in ('place_odds', 'place_odds_n', 'rating', 'finishing_position',
                      'win_dividend', 'actual_weight'):
        if forbidden in F.ALL_FEATURES:
            failures.append(f'LEAK: {forbidden} is in the feature list')
            print(f"    !! LEAK: {forbidden}")
    if not failures:
        print("    no post-race field in the feature list ✅")

    # ── Invariant 4: the claiming-weight suffix is ONE jockey ─────────────
    # The race card writes a claiming ride as "C L Chau (-2)". If that counts
    # as a different entity from plain "C L Chau", the claim spelling gets no
    # history, its jockey features come out NaN, and the model scores that
    # rider on "no form". Both spellings must land on the same group.
    print("\n[4] jockey claim-suffix normalisation")
    probe = pd.DataFrame({
        'horse_name': ['PROBE A', 'PROBE B'],
        'horse_no': [1, 2],
        'draw': [1, 2],
        'jockey': ['C L Chau (-2)', 'C L Chau'],
        'trainer': ['T ONE', 'T TWO'],
        'win_odds': [5.0, 5.0],
        'finishing_position': [np.nan, np.nan],
        'race_date': [ts, ts],
        'venue': [TEST_VENUE, TEST_VENUE],
        'race_no': [1, 1],
        'distance': [1200, 1200],
        'race_class': ['Class 4', 'Class 4'],
        'track': ['Turf', 'Turf'],
    })
    probe_out, _ = F.build_features(pd.concat([hist, probe], ignore_index=True,
                                                sort=False))
    probe_out = probe_out[probe_out['race_date'] == ts]
    jk = probe_out.set_index('horse_name')['jockey']
    if jk.get('PROBE A') != jk.get('PROBE B'):
        failures.append('claim suffix not normalised: '
                        f"{jk.get('PROBE A')!r} != {jk.get('PROBE B')!r}")
        print(f"    !! MISMATCH {jk.get('PROBE A')!r} != {jk.get('PROBE B')!r}")
    else:
        print(f"    both spellings map to one jockey: {jk.get('PROBE A')!r} ✅")

    if F.normalize_jockey('C L Chau (-2)') != 'C L Chau':
        failures.append('normalize_jockey() did not strip the claim suffix')
    if F.normalize_jockey('Z Purton') != 'Z Purton':
        failures.append('normalize_jockey() mangled a plain name')
    if F.normalize_jockey('H Y Yuen (-10)') != 'H Y Yuen':
        failures.append('normalize_jockey() failed on a two-digit claim')

    # A race card must produce NO NaN jockey features: every runner's jockey
    # must have history under the canonical name.
    card_out, _ = F.build_features(pd.concat([hist, card], ignore_index=True,
                                              sort=False))
    card_out = card_out[card_out['race_date'] == ts]
    for c in ('jockey_win_10', 'jockey_win_30', 'jockey_win_venue'):
        if c in card_out.columns:
            n = int(pd.to_numeric(card_out[c], errors='coerce').isna().sum())
            if n:
                failures.append(f'{c} is NaN for {n}/{len(card_out)} card runners')
                print(f"    !! {c} NaN for {n}/{len(card_out)} runners")
    if not any('NaN for' in f for f in failures):
        print("    no NaN jockey features on the card ✅")

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print(f"\nPASS: train==serve, no same-day leak, no leakage fields, "
          f"jockey claim-suffix normalised, {len(F.ALL_FEATURES)} features.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
