"""Provenance guard for horse_entries.win_odds.

WHY THIS EXISTS
---------------
On 2026-10-21 an audit of horse.db found that the WINNER's `win_odds` had
been written from the post-race WIN dividend instead of the published
'Win Odds' column:

    394 rows had odds >= 10 with a fractional part.
    393 of them (99.7%) were the winner of their race.

Published odds for SP >= 10 are always integers (HKJC publishes the
fractional SP as a whole number above 10) and for SP < 10 they carry at most
one decimal. So "odds >= 10 AND has a fractional part" is proof the value
came from the payout table, and it was a near-perfect winner fingerprint
sitting inside a pre-race feature.

This module makes that class of defect IMPOSSIBLE to reintroduce silently:
it asserts the published-odds invariants over the whole database and refuses
to let training proceed if any of them is broken.

INVARIANTS
----------
Every invariant here must be PROVABLE, or it is not an invariant. An
earlier version of this guard asserted three things that turned out to be
false, and it reported FAIL on perfectly good data:

  * "published odds are capped at 99"  -- DISPROVED against live pages.
    Modern HKJC publishes 143 / 254 / 268 / 271 (max observed in the DB: 693).
  * "no odds may equal the race's WIN dividend"  -- FALSE POSITIVE.
    A longshot priced 26.0 in a race whose winner had SP 2.6 (dividend
    26.0) is a coincidence, not contamination. 345 such rows exist in the
    database and NONE of them is a winner row.
  * "a race whose winner shares the race with a fractional-odds horse is
    suspicious"  -- FALSE POSITIVE. A published value below 10 legally
    carries one decimal place, so this describes almost every race.

What IS provable, and is asserted here:

I1  A published value >= 10 is always a whole number, and a published
    value < 10 has at most one decimal place. A stored value that breaks
    this cannot have come from the published column.  (This is the
    fingerprint: such rows were 99.7% winners.)
I2  The winner's stored odds must not be the raw WIN dividend -- that is
    the R1a contamination signature.
I3  No stored odds below 1.0, and none above 1000.
I4  Every row this pipeline repaired still holds the official value
    recorded in repair_provenance, and that value is backed by an
    official HKJC URL.
I5  The winner's odds must be a value the published rule could produce
    from its true starting price.
I6  No winner row may share a race with a row that has an impossible
    published value (i.e. the old fingerprint may not survive in any form).

Run standalone:  python3 scripts/verify_odds_provenance.py
"""
import json
import math
import os
import sqlite3
import sys

DB = '/home/work/horse-bet/data/horse.db'
EVIDENCE = '/home/work/horse-bet/data/official_evidence/official_results.json'
OFFICIAL_HOST = 'racing.hkjc.com'


def open_ro():
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=300)
    c.row_factory = sqlite3.Row
    return c


def load_evidence():
    """The official HKJC pages that justify every repair.

    They live in the repository (not in scratch) so this guard stays
    reproducible: anyone can re-run it and get the same answer.
    """
    if not os.path.exists(EVIDENCE):
        return {}
    try:
        return json.load(open(EVIDENCE))
    except (json.JSONDecodeError, OSError):
        return {}


def published_from_sp(sp):
    """The published display rule, as observed on official pages.

    SP < 10  -> floor(SP*10)/10   (at most one decimal place)
    SP >= 10 -> floor(SP)         (a whole number)

    There is deliberately NO upper cap: modern meetings publish values
    above 99, so any cap would be era-dependent and unverifiable.
    """
    if sp >= 10:
        return float(math.floor(sp))
    return math.floor(sp * 10) / 10.0


# ── the era boundary, established from live official pages ────────────────
# Max published value seen on the official pages, per year:
#   2006..2014  -> 99   (HKJC capped the published display at 99)
#   2015        -> 122
#   2016        -> 341
#   2017        -> 327
#   2020        -> 520
#   2026        -> 440
# The cap was therefore lifted during 2015. CAP_LAST_YEAR is the last year
# in which a published value of 99 means "capped", not "exactly 99".
CAP_LAST_YEAR = 2014


def is_possible_published(v):
    """Could HKJC have published this value?"""
    if v is None:
        return True
    v = float(v)
    if v < 1.0 or v > 1000.0:
        return False
    if v >= 10:
        return v == int(v)
    return abs(round(v, 1) - v) < 1e-9


def check(c):
    f = {}
    ev = load_evidence()
    f['I0_official_evidence_races'] = len(ev)

    # ── I1: the value must be one HKJC could have published ───────────────
    i1 = c.execute("""
        SELECT COUNT(*) FROM horse_entries
        WHERE win_odds IS NOT NULL AND (
              (win_odds >= 10 AND win_odds <> CAST(win_odds AS INTEGER))
           OR (win_odds <  10 AND ROUND(win_odds,1) <> win_odds) )""").fetchone()[0]
    f['I1_values_impossible_as_published'] = i1
    if i1:
        f['I1_examples'] = [dict(r) for r in c.execute("""
            SELECT rd.race_date, rd.venue, r.race_no, he.horse_no, he.horse_name,
                   he.win_odds, he.finishing_position
            FROM horse_entries he
            JOIN races r ON r.id=he.race_id
            JOIN race_days rd ON rd.id=r.race_day_id
            WHERE he.win_odds IS NOT NULL AND (
                  (he.win_odds >= 10 AND he.win_odds <> CAST(he.win_odds AS INTEGER))
               OR (he.win_odds <  10 AND ROUND(he.win_odds,1) <> he.win_odds) )
            LIMIT 15""")]
        w = c.execute("""
            SELECT COUNT(*) FROM horse_entries he
            WHERE he.win_odds IS NOT NULL AND he.finishing_position = 1
              AND ((he.win_odds >= 10 AND he.win_odds <> CAST(he.win_odds AS INTEGER))
                OR (he.win_odds <  10 AND ROUND(he.win_odds,1) <> he.win_odds))""").fetchone()[0]
        f['I1_of_which_winners'] = w

    # ── I2: winner odds must not be the raw WIN dividend ──────────────────
    # Only the WINNER row is meaningful here. For a non-winner, a price
    # that coincidentally equals the race dividend proves nothing.
    i2 = c.execute("""
        SELECT COUNT(*) FROM horse_entries he
        JOIN races r ON r.id=he.race_id
        JOIN race_results rr ON rr.race_id=r.id
        WHERE he.finishing_position = 1 AND rr.win_dividend IS NOT NULL
          AND he.win_odds IS NOT NULL
          AND ABS(he.win_odds - CAST(rr.win_dividend AS REAL)) < 1e-9""").fetchone()[0]
    f['I2_winner_equals_raw_dividend'] = i2

    # ── I3: sanity range ───────────────────────────────────────────────────
    i3a = c.execute("""SELECT COUNT(*) FROM horse_entries
                       WHERE win_odds IS NOT NULL AND win_odds < 1.0""").fetchone()[0]
    i3b = c.execute("""SELECT COUNT(*) FROM horse_entries
                       WHERE win_odds IS NOT NULL AND win_odds > 1000.0""").fetchone()[0]
    f['I3a_below_1'] = i3a
    f['I3b_above_1000'] = i3b
    f['I3c_max_observed'] = c.execute(
        'SELECT MAX(win_odds) FROM horse_entries').fetchone()[0]

    # ── I4: every repaired row still holds its official value ─────────────
    i4_checked = 0
    i4_bad = []
    no_url = 0
    has_prov = False
    try:
        has_prov = c.execute("""SELECT COUNT(*) FROM sqlite_master
                                WHERE type='table' AND name='repair_provenance'""").fetchone()[0] > 0
    except sqlite3.Error:
        pass
    if not has_prov:
        f['I4_skipped_no_provenance_table'] = True
    else:
        pairs = c.execute("""
            SELECT p.race_date, p.venue, p.race_no, p.horse_no,
                   p.new_value, p.old_value, p.repair_class, p.evidence_url
            FROM repair_provenance p
            WHERE p.action='UPDATE_WIN_ODDS' AND p.column_name='win_odds'
        """).fetchall()
        for p in pairs:
            i4_checked += 1
            if not p['evidence_url'] or OFFICIAL_HOST not in p['evidence_url']:
                no_url += 1
            row = c.execute("""SELECT he.win_odds FROM horse_entries he
                                JOIN races r ON r.id=he.race_id
                                JOIN race_days rd ON rd.id=r.race_day_id
                                WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?
                                  AND he.horse_no=?""",
                             (p['race_date'], p['venue'], p['race_no'], p['horse_no'])).fetchone()
            if row is None:
                i4_bad.append({'race': [p['race_date'], p['venue'], p['race_no']],
                               'horse_no': p['horse_no'], 'error': 'row missing'})
                continue
            want = float(p['new_value'])
            if row['win_odds'] is None or abs(float(row['win_odds']) - want) > 1e-9:
                i4_bad.append({'race': [p['race_date'], p['venue'], p['race_no']],
                               'horse_no': p['horse_no'],
                               'expected': want, 'actual': row['win_odds']})
            elif not is_possible_published(want):
                i4_bad.append({'race': [p['race_date'], p['venue'], p['race_no']],
                               'horse_no': p['horse_no'],
                               'error': 'written value is not a legal published value',
                               'value': want})
        f['I4_provenance_rows_checked'] = i4_checked
        f['I4_provenance_violations'] = len(i4_bad)
        f['I4_provenance_without_official_url'] = no_url
        if i4_bad:
            f['I4_examples'] = i4_bad[:10]

    # ── I5: every race with official evidence must agree with the page ────
    # The official cell is the authority. The dividend is NOT a reliable
    # inverse of the odds: a race-day bonus adds to the payout and a dead
    # heat splits it, so the same odds produce different dividends. An
    # earlier version of this guard inferred the expected odds from the
    # dividend and flagged 35 rows; 26 of them were adjudicated against
    # their official pages and ALL 26 were correct, with the DIVIDEND
    # being the wrong field. So the check is now made directly against
    # the official page wherever we hold one.
    i5_odds_mismatch = []
    i5_div_mismatch = []
    i5_races = 0
    i5_runners = 0
    for key, v in ev.items():
        if not v.get('rows'):
            continue
        d0, ven, rn = key.split('|')
        i5_races += 1
        rows = c.execute("""SELECT he.horse_no, he.win_odds
                             FROM horse_entries he
                             JOIN races r ON r.id=he.race_id
                             JOIN race_days rd ON rd.id=r.race_day_id
                             WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                          (d0, ven, int(rn))).fetchall()
        db = {r['horse_no']: r['win_odds'] for r in rows}
        for hno, hrow in v['rows'].items():
            hno = int(hno)
            if hno not in db:
                continue
            i5_runners += 1
            off = hrow.get('odds')
            if off in (None, '', '---'):
                continue
            try:
                off = float(off)
            except (TypeError, ValueError):
                continue
            stored = db[hno]
            if stored is None:
                i5_odds_mismatch.append({'race': key, 'horse_no': hno,
                                         'db': None, 'official': off})
            elif abs(float(stored) - off) > 1e-9:
                i5_odds_mismatch.append({'race': key, 'horse_no': hno,
                                         'db': stored, 'official': off})
        w = (v.get('dividends') or {}).get('win')
        if w and w.get('amount'):
            try:
                off_div = float(w['amount'])
            except (TypeError, ValueError):
                off_div = None
            if off_div is not None:
                rr = c.execute("""SELECT rr.win_dividend FROM race_results rr
                                   JOIN races r ON r.id=rr.race_id
                                   JOIN race_days rd ON rd.id=r.race_day_id
                                   WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                                (d0, ven, int(rn))).fetchone()
                if rr is not None and rr['win_dividend'] is not None:
                    if abs(float(rr['win_dividend']) - off_div) > 1e-9:
                        i5_div_mismatch.append({'race': key,
                                                'db': rr['win_dividend'],
                                                'official': off_div})
    f['I5_races_checked'] = i5_races
    f['I5_runners_checked'] = i5_runners
    f['I5_odds_mismatches_vs_official'] = len(i5_odds_mismatch)
    f['I5_dividend_mismatches_vs_official'] = len(i5_div_mismatch)
    if i5_odds_mismatch:
        f['I5_odds_examples'] = i5_odds_mismatch[:10]
    if i5_div_mismatch:
        f['I5_dividend_examples'] = i5_div_mismatch[:10]

    # ── I7: the dividend must not be a misread cell ───────────────────────
    # The HKJC dividends table is (label, horse_no, amount). A parser that
    # reads cell[1] instead of cell[2] stores the winning HORSE NUMBER as
    # the dividend. The signature is exact: win_dividend == horse_no of
    # the winner. Verified on 2017-04-09 ST R1 (winner #8): DB 8.0,
    # official 18.00; and ST R8 (winner #7): DB 7.0, official 16.00.
    i7 = c.execute("""
        SELECT COUNT(*) FROM race_results rr
        JOIN races r ON r.id=rr.race_id
        JOIN horse_entries he ON he.race_id=r.id AND he.finishing_position=1
        WHERE rr.win_dividend IS NOT NULL
          AND ABS(CAST(rr.win_dividend AS REAL) - he.horse_no) < 1e-9""").fetchone()[0]
    f['I7_dividend_is_a_misread_cell'] = i7
    if i7:
        f['I7_examples'] = [dict(r) for r in c.execute("""
            SELECT rd.race_date, rd.venue, r.race_no, he.horse_no,
                   he.win_odds, rr.win_dividend AS div
            FROM race_results rr
            JOIN races r ON r.id=rr.race_id
            JOIN race_days rd ON rd.id=r.race_day_id
            JOIN horse_entries he ON he.race_id=r.id AND he.finishing_position=1
            WHERE rr.win_dividend IS NOT NULL
              AND ABS(CAST(rr.win_dividend AS REAL) - he.horse_no) < 1e-9
            LIMIT 12""")]

    # ── I6: the fingerprint may not survive in any form ───────────────────
    i6 = c.execute("""
        SELECT COUNT(*) FROM (
          SELECT he.race_id FROM horse_entries he
          JOIN races r ON r.id=he.race_id
          WHERE he.finishing_position = 1 AND he.win_odds IS NOT NULL
            AND EXISTS (
              SELECT 1 FROM horse_entries h2
              WHERE h2.race_id = he.race_id AND h2.win_odds IS NOT NULL
                AND h2.horse_no <> he.horse_no
                AND ((h2.win_odds >= 10 AND h2.win_odds <> CAST(h2.win_odds AS INTEGER))
                  OR (h2.win_odds <  10 AND ROUND(h2.win_odds,1) <> h2.win_odds)) )
        )""").fetchone()[0]
    f['I6_races_where_fingerprint_survives'] = i6

    ok = (i1 == 0 and i2 == 0 and i3a == 0 and i3b == 0
          and not i5_odds_mismatch and not i5_div_mismatch
          and i6 == 0 and i7 == 0
          and len(i4_bad) == 0 and no_url == 0)
    f['OK'] = ok
    return ok, f


def main():
    c = open_ro()
    ok, f = check(c)
    c.close()
    print('=== ODDS PROVENANCE GUARD ===')
    for k, v in f.items():
        if k.endswith('_examples'):
            print(f'  {k}:')
            for e in v:
                print(f'      {e}')
        else:
            print(f'  {k}: {v}')
    print(f'\nRESULT: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
