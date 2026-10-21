"""Test suite for the odds pipeline.

Run:  python3 scripts/test_odds_pipeline.py

Covers the four things that actually broke:

T1  Provenance: every stored odds value is a value HKJC could have
    published, never a payout-table number.
T2  No winner fingerprint: no row's odds may encode that it won.
T3  Train -> serve equivalence: a race card scored through the shared
    feature module must produce the same features as the same race built
    from the DB (this is the check that was missing on 2026-10-20).
T4  The published-odds display convention holds on real official pages.
"""
import json
import os
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import features as F

DB = '/home/work/horse-bet/data/horse.db'
PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  -- {detail}' if detail else ''))


def ro():
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=300)
    c.row_factory = sqlite3.Row
    return c


# ── T1: provenance ──────────────────────────────────────────────────────────
def t1_provenance():
    print('\nT1  odds provenance')
    c = ro()
    n = c.execute("""SELECT COUNT(*) FROM horse_entries
                     WHERE win_odds IS NOT NULL AND win_odds >= 10
                       AND win_odds <> CAST(win_odds AS INTEGER)""").fetchone()[0]
    check('no odds >= 10 carries a fractional part (impossible for a published value)',
          n == 0, f'{n} rows')

    # The two assertions below used to compare EVERY runner's odds against
    # the race's WIN dividend. That is a false-positive generator:
    #   * a longshot priced 26.0 in a race whose winner had SP 2.6 has a
    #     dividend of 26.0 too -- 345 such coincidences exist and NONE of
    #     them is a winner row;
    #   * a race-day bonus changes the payout, and a dead heat splits it,
    #     so the dividend is not a reliable inverse of the odds.
    # The contamination signature is the WINNER's odds equalling the raw
    # dividend, so the check is restricted to the winner row.
    n = c.execute("""SELECT COUNT(*) FROM horse_entries he
                      JOIN races r ON r.id=he.race_id
                      JOIN race_results rr ON rr.race_id=r.id
                      WHERE rr.win_dividend IS NOT NULL AND he.win_odds IS NOT NULL
                        AND he.finishing_position = 1
                        AND ABS(he.win_odds - CAST(rr.win_dividend AS REAL)) < 1e-9""").fetchone()[0]
    check("the winner's odds never equals the race's raw WIN dividend",
          n == 0, f'{n} rows')

    n = c.execute("""SELECT COUNT(*) FROM horse_entries he
                      JOIN races r ON r.id=he.race_id
                      JOIN race_results rr ON rr.race_id=r.id
                      WHERE rr.win_dividend IS NOT NULL AND he.win_odds IS NOT NULL
                        AND he.finishing_position = 1 AND he.win_odds >= 10
                        AND ABS(he.win_odds - CAST(rr.win_dividend AS REAL)/10.0) < 1e-9
                        AND he.win_odds <> CAST(he.win_odds AS INTEGER)""").fetchone()[0]
    check("the winner's odds is never a non-integer derived from the dividend",
          n == 0, f'{n} rows')

    n = c.execute("""SELECT COUNT(*) FROM horse_entries
                     WHERE win_odds IS NOT NULL AND win_odds < 1""").fetchone()[0]
    check('no odds below 1', n == 0, f'{n} rows')
    c.close()


# ── T2: no winner fingerprint ───────────────────────────────────────────────
def t2_no_fingerprint():
    print('\nT2  no outcome fingerprint in the odds feature')
    c = ro()
    rows = c.execute("""SELECT he.win_odds, he.finishing_position FROM horse_entries he
                         WHERE he.win_odds IS NOT NULL""").fetchall()
    od = np.array([float(r['win_odds']) for r in rows], dtype=float)

    def _pos(v):
        # finishing_position is an INTEGER in the DB, but older rows stored
        # the official finish string ('WV', '3 DH', ...). Only a real
        # integer 1 means "won".
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    pos = np.array([_pos(r['finishing_position']) for r in rows])
    winner = pos == 1
    check('winner labels are readable', winner.sum() > 1000,
          f'{int(winner.sum())} winners, base rate {winner.mean():.1%}')

    # the old sentinel: fractional odds at >= 10 was 99.7% winners
    sent = (od >= 10) & (od % 1 != 0)
    if sent.sum():
        rate = winner[sent].mean()
        check('fractional odds >= 10 is not a winner marker', rate < 0.5,
              f'{int(sent.sum())} rows, {rate:.1%} winners')
    else:
        check('fractional odds >= 10 sentinel absent', True, '0 rows')

    # general check: no threshold on win_odds may separate winners sharply
    worst, worst_thr = 0.0, None
    base = winner.mean()
    for thr in np.arange(1.0, 120.0, 0.5):
        hi = od >= thr
        if hi.sum() < 30:
            continue
        r = winner[hi].mean()
        if abs(r - base) > abs(worst - base):
            worst, worst_thr = r, thr
    check('no odds threshold separates winners from the field',
          abs(worst - base) < 0.12,
          f'worst threshold {worst_thr}: {worst:.1%} winners vs base {base:.1%}')

    # a NULL-odds row must not be a winner marker
    n_null = c.execute('SELECT COUNT(*) FROM horse_entries WHERE win_odds IS NULL').fetchone()[0]
    n_null_win = c.execute("""SELECT COUNT(*) FROM horse_entries
                               WHERE win_odds IS NULL AND finishing_position=1""").fetchone()[0]
    if n_null:
        rate = n_null_win / n_null
        check('NULL odds is not a winner marker', rate < 0.5,
              f'{n_null} NULL rows, {rate:.1%} winners')
    c.close()


# ── T3: train -> serve equivalence ───────────────────────────────────────────
def t3_equivalence():
    print('\nT3  train -> serve equivalence')
    c = ro()
    q = """SELECT he.horse_no, he.horse_name, he.draw, he.jockey, he.trainer,
                  he.win_odds, he.finishing_position, rd.race_date, rd.venue,
                  r.race_no, r.distance, r.race_class, r.track
           FROM horse_entries he
           JOIN races r ON r.id=he.race_id
           JOIN race_days rd ON rd.id=r.race_day_id"""
    df = pd.read_sql_query(q, c, parse_dates=['race_date'])
    c.close()

    # pick a recent meeting: the features for it must be identical whether
    # we build them in one batch or race by race.
    last_day = df['race_date'].max()
    day = df[df['race_date'] == last_day]
    if len(day) == 0:
        check('have a meeting to test', False, 'no rows')
        return

    # The serving path (scripts/predict_racecard.py) scores a card as:
    #     combined = concat([history, card])
    #     build_features(combined)
    # where `card` has NO results yet. So the equivalence to test is:
    #
    #   reference : build_features(full history incl. the race's results)
    #   serving   : build_features(history up to the race, with the race's
    #               own results removed)
    #
    # and the served rows must equal the reference rows for the same
    # runners. An earlier version of this test compared the full build
    # against a build of ONE MEETING with no history, which can never
    # match: the rolling form windows have nothing to roll over.
    Xb, feats = F.build_features(df)
    ref = Xb[Xb['race_date'] == last_day].copy()

    hist = df[df['race_date'] < last_day].copy()
    card = day.copy()
    card['finishing_position'] = np.nan
    combined = pd.concat([hist, card], ignore_index=True, sort=False)
    Xs, _ = F.build_features(combined)
    served = Xs[Xs['race_date'] == last_day].copy()

    if len(ref) != len(served):
        check('features identical when built in one batch vs via the serving path',
              False, f'reference {len(ref)} rows, served {len(served)} rows')
        return

    def keyed(frame):
        k = frame[['race_no', 'horse_no']].astype(str).agg('|'.join, axis=1)
        return frame.set_index(k)

    ref_k = keyed(ref)
    srv_k = keyed(served)
    common = [k for k in ref_k.index if k in srv_k.index]
    mism = []
    for col in feats:
        a = pd.to_numeric(ref_k.loc[common, col], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(srv_k.loc[common, col], errors='coerce').to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.abs(np.nan_to_num(a - b))
        if not np.allclose(a[~both_nan], b[~both_nan], atol=1e-9, equal_nan=True):
            mism.append((col, f'max diff {diff.max():.3e}'))
    check('features identical when built in one batch vs via the serving path',
          not mism, f'{len(day)} rows, mismatches: {mism[:4]}')

    # the served card must produce usable form features, not a row of zeros
    zero_cols = [c_ for c_ in feats
                 if pd.to_numeric(srv_k.loc[common, c_], errors='coerce').fillna(0).abs().sum() == 0]
    check('served card produces non-degenerate feature values',
          len(zero_cols) <= 2, f'all-zero features: {zero_cols}')


# ── T4: the published display convention ─────────────────────────────────────
def t4_convention():
    print('\nT4  published-odds display convention')
    import math

    def published_from_dividend(div):
        # NO cap. The 99 cap was real but only until 2014: modern meetings
        # publish 143, 254, 268 and 271, and the DB's maximum published
        # value is 693. A cap here would fail on the live samples below.
        sp = float(div) / 10.0
        if sp >= 10:
            return float(math.floor(sp))
        return math.floor(sp * 10) / 10.0

    c = ro()
    # live-verified samples: (race_date, venue, race_no, official published
    # value of the winner, official WIN dividend)
    live = [
        ('2006-01-01', 'ST', 5, 33.0, '338.00'),
        ('2006-03-01', 'HV', 5, 10.0, '107.00'),
        ('2015-06-14', 'ST', 1, 6.6, '66.50'),
        ('2020-11-01', 'ST', 1, 9.7, '97.00'),
        ('2021-05-02', 'ST', 1, 108.0, '1089.00'),
        ('2023-01-15', 'ST', 4, 143.0, '1439.50'),
        ('2026-04-26', 'ST', 1, 3.3, '33.50'),
    ]
    bad = []
    for d0, ven, rn, official, div in live:
        got = published_from_dividend(div)
        if abs(got - official) > 1e-9:
            bad.append((d0, ven, rn, official, got))
    check('derived published value matches the official cell on live samples',
          not bad, f'disagreements: {bad}')

    # NOTE: the 99 cap is ERA-DEPENDENT and was removed from the derivation
    # above; modern HKJC publishes values above 99 (143, 254, 268, 271).
    # The derivation is only used to CROSS-CHECK official values, never to
    # write them. Repair values always come from the official page.
    c.close()


# ── T5: the DB agrees with the official pages we hold ───────────────────────
def t5_against_official_pages():
    """The strongest check available: compare the stored odds against the
    official 'Win Odds' cell of every race we have evidence for."""
    print('\nT5  stored odds agree with the official pages')
    ev_path = '/home/work/horse-bet/data/official_evidence/official_results.json'
    if not os.path.exists(ev_path):
        check('official evidence file is present', False, ev_path)
        return
    ev = json.load(open(ev_path))
    c = ro()
    mism = []
    runners = 0
    races = 0
    for key, v in ev.items():
        if not v.get('rows'):
            continue
        d0, ven, rn = key.split('|')
        races += 1
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
            off = hrow.get('odds')
            if off in (None, '', '---'):
                continue
            try:
                off = float(off)
            except (TypeError, ValueError):
                continue
            runners += 1
            if db[hno] is None or abs(float(db[hno]) - off) > 1e-9:
                mism.append((key, hno, db[hno], off))
    c.close()
    check(f'stored odds match the official cell ({runners} runners, {races} races)',
          not mism, f'{len(mism)} mismatches: {mism[:5]}')


def main():
    print('=== ODDS PIPELINE TESTS ===')
    t1_provenance()
    t2_no_fingerprint()
    t3_equivalence()
    t4_convention()
    t5_against_official_pages()
    print(f'\n=== {len(PASS)} passed, {len(FAIL)} failed ===')
    if FAIL:
        for f in FAIL:
            print(f'  FAILED: {f}')
        sys.exit(1)


if __name__ == '__main__':
    main()
