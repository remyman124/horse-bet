"""Build a reproducible repair set for horse.db from official HKJC evidence.

READ-ONLY. Emits /tmp/repair_plan.json: one entry per DB mutation, each with
the official URL, the official value, and the current DB value, so every
change can be audited before a single write happens.

Repair classes
--------------
R1  WINNER_ODDS_PROVENANCE
    The winner's win_odds was written from the post-race WIN dividend instead
    of the published 'Win Odds' column. Two sub-cases:
      R1a  stored == raw dividend (e.g. 10.5)   -> catastrophic
      R1b  stored == dividend/10 (true SP, e.g. 33.8)
    Both are outcome-derived. Fix: write the published 'Win Odds' value.

    IMPORTANT: the repair value is ALWAYS read from the official page's
    'Win Odds' cell. It is never derived from the dividend.

    Published display convention, as observed on the official pages
    (cross-check only, never used to write a value):
        SP < 10  -> floor(SP*10)/10   (1 dp, e.g. SP 1.05 -> '1', SP 6.2 -> '6.2')
        SP >= 10 -> floor(SP)         (whole,  e.g. SP 33.8 -> '33')
    There is NO global upper cap. An earlier working assumption that
    published odds were capped at 99 was DISPROVED against live pages:
    modern HKJC publishes values above 99 (143, 254, 268, 271), while old
    meetings (2006-2015) top out at 99. Any cap is era-dependent, so no
    cap rule is used anywhere in this pipeline.

    Verified across 500+ official pages: a published value >= 10 is ALWAYS
    a whole number, and a published value < 10 has at most 1 decimal place.
    That invariant is used to DETECT suspect rows, not to derive values.

    NOTE: published '1' is a REAL published value for SP 1.05; the DB simply
    never had a row with win_odds == 1.0 before, so the feature gate
    `win_odds > 1` would silently drop these horses. That is a SEPARATE
    feature-layer decision, handled in features.py, not here.

R2  DEAD_HEAT_PARTNER_ROWS
    HKJC renders a dead heat as TWO rows both at position 'N DH'. The old
    parser dropped one of each pair (and then renumbered), so 746 runner rows
    are missing and 369 races have position sets that do not match official.
    Fix: insert the missing partner rows with the official position and
    finish_type 'DH'. Existing positions are left alone (they are correct
    per-horse); only the missing rows are added.

R3  MISSING_RESULTS
    20 races have entries but no finishing positions, though official results
    exist. Fix: write the official finishing position for every horse.

R4  POSITION_ZERO
    Races where horses carry finishing_position = 0 (never written) while
    official results exist. Fix: write official positions.

R5  PHANTOM_RACE
    Race days whose official pages carry no race at all ('No information')
    but which exist in the DB. Fix: delete the race + its entries + results.
    Requires official evidence of a non-existent race.
"""
import json, os, sqlite3, re
from collections import Counter, defaultdict

DB = '/home/work/horse-bet/data/horse.db'
LIVE = '/tmp/live_all.json'
# Additional official-evidence files, merged in. The checkpoint is written
# incrementally by the paced fetcher, so a partial run is still useful.
LIVE_EXTRA = ('/tmp/live_all.ckpt.json', '/tmp/live_null.json', '/tmp/live_2026.json')
OUT = '/tmp/repair_plan.json'

MONTHS = ('January February March April May June July August September '
          'October November December').split()


def norm_pos(p):
    """Official 'Pla.' cell -> (int position, finish_type).

    '3 DH'      -> (3, 'DH')      dead heat
    '5'         -> (5, None)
    'WV'        -> (None, 'WV')   withdrawn
    'DNF'/'DISQ'-> (None, 'DNF'/'DISQ')
    """
    if p is None:
        return None, None
    s = str(p).strip()
    m = re.match(r'^(\d+)\s*DH$', s, re.I)
    if m:
        return int(m.group(1)), 'DH'
    if s.isdigit():
        return int(s), None
    return None, s.upper() or None


def name_key(n):
    """'SKY FIELD (D216)' -> 'SKY FIELD'"""
    return re.sub(r'\s*\([A-Z]\d+\)\s*$', '', str(n or '')).strip().upper()


def main():
    live = json.load(open(LIVE)) if os.path.exists(LIVE) else {}
    for extra in LIVE_EXTRA:
        if os.path.exists(extra):
            live.update(json.load(open(extra)))
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=180)
    c.row_factory = sqlite3.Row

    plan = []
    stats = Counter()
    unresolved = []

    # ── index the live evidence ────────────────────────────────────────────
    by_race = {}
    for k, v in live.items():
        by_race[(v['date'], v['venue'], int(v['race_no']))] = v

    # ── R1: winner odds provenance ─────────────────────────────────────────
    # Official published odds per race, straight from the live 'Win Odds' cell.
    for key, lv in by_race.items():
        if lv['status'] != 'OK' or not lv['rows']:
            continue
        d0, ven, rn = key
        rows = list(c.execute("""
            SELECT he.id, he.horse_no, he.horse_name, he.win_odds, he.finishing_position,
                   rr.win_dividend
            FROM horse_entries he
            JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
            LEFT JOIN race_results rr ON rr.race_id=r.id
            WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?
              AND he.finishing_position=1""", key))
        if len(rows) != 1:
            continue
        row = rows[0]
        lrow = lv['rows'].get(str(row['horse_no'])) or lv['rows'].get(row['horse_no'])
        if not lrow:
            continue
        pub = lrow['odds']
        if pub in (None, '', '---'):
            continue
        try:
            pubf = float(pub)
        except ValueError:
            continue
        stored = row['win_odds']
        if stored is not None and abs(float(stored) - pubf) < 1e-9:
            continue
        div = row['win_dividend']
        try:
            divf = float(div)
        except (TypeError, ValueError):
            divf = None
        if divf is not None and abs(float(stored or -1) - divf) < 1e-9:
            cls = 'R1a_raw_dividend'
        elif divf is not None and stored is not None and abs(float(stored) - divf / 10.0) < 1e-9:
            cls = 'R1b_true_sp'
        else:
            cls = 'R1c_other'
        plan.append({
            'action': 'UPDATE_WIN_ODDS', 'class': cls,
            'race': [d0, ven, rn], 'horse_no': row['horse_no'],
            'entry_id': row['id'], 'horse_name': row['horse_name'],
            'db_value': stored, 'official_value': pubf,
            'official_raw_cell': pub, 'win_dividend': div,
            'evidence_url': lv['url'], 'evidence_page_date': lv['page_date'],
        })
        stats[cls] += 1

    # ── R2/R3/R4: runner rows vs official rows ─────────────────────────────
    for key, lv in by_race.items():
        if lv['status'] != 'OK' or not lv['rows']:
            continue
        d0, ven, rn = key
        dbrows = list(c.execute("""
            SELECT he.id, he.horse_no, he.horse_name, he.win_odds, he.finishing_position,
                   he.finish_type, he.draw, he.jockey, he.trainer
            FROM horse_entries he
            JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
            WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""", key))
        if not dbrows:
            continue
        db_by_no = {r['horse_no']: r for r in dbrows}
        live_by_no = {int(k): v for k, v in lv['rows'].items()}

        # R2: official runner absent from the DB
        for hno, lrow in sorted(live_by_no.items()):
            if hno in db_by_no:
                continue
            pos, ftype = norm_pos(lrow['pos'])
            if pos is None and ftype is None:
                continue
            plan.append({
                'action': 'INSERT_MISSING_RUNNER', 'class': 'R2_dead_heat_partner',
                'race': [d0, ven, rn], 'horse_no': hno,
                'horse_name': name_key(lrow['name']),
                'official_name_raw': lrow['name'],
                'official_pos': pos, 'official_finish_type': ftype,
                'official_odds': lrow['odds'], 'official_draw': lrow['draw'],
                'official_jockey': lrow['jockey'], 'official_trainer': lrow['trainer'],
                'evidence_url': lv['url'],
            })
            stats['R2_insert_runner'] += 1

        # R3/R4: DB row has no/zero position but official does
        for hno, drow in db_by_no.items():
            lrow = live_by_no.get(hno)
            if not lrow:
                continue
            pos, ftype = norm_pos(lrow['pos'])
            if pos is None:
                continue
            dbpos = drow['finishing_position']
            if dbpos is not None and int(dbpos) == pos:
                continue
            plan.append({
                'action': 'UPDATE_FINISHING_POSITION', 'class':
                    'R3_missing_results' if dbpos is None else 'R4_position_zero',
                'race': [d0, ven, rn], 'horse_no': hno, 'entry_id': drow['id'],
                'horse_name': drow['horse_name'],
                'db_position': dbpos, 'official_position': pos,
                'official_finish_type': ftype, 'official_odds': lrow['odds'],
                'evidence_url': lv['url'],
            })
            stats['R3R4_update_position'] += 1

    # ── R5: phantom races (official says the race does not exist) ─────────
    phantom = []
    for key, lv in by_race.items():
        if lv['status'] != 'NO_INFORMATION':
            continue
        d0, ven, rn = key
        n = c.execute("""SELECT COUNT(*) FROM horse_entries he
            JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
            WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""", key).fetchone()[0]
        if n:
            phantom.append({'race': [d0, ven, rn], 'db_rows': n,
                            'evidence_url': lv['url'], 'page_date': lv['page_date']})
    for p in phantom:
        plan.append({'action': 'DELETE_PHANTOM_RACE', 'class': 'R5_phantom_race',
                     **p})
    stats['R5_phantom_race'] = len(phantom)

    # ── R6: races where official odds differ from stored odds (non-winner) ─
    # R1 already covers the winner row of every race. Skip any row R1 has
    # already planned, so each entry_id appears at most once in the plan.
    r1_covered = {a['entry_id'] for a in plan
                  if a['action'] == 'UPDATE_WIN_ODDS' and 'entry_id' in a}
    for key, lv in by_race.items():
        if lv['status'] != 'OK' or not lv['rows']:
            continue
        d0, ven, rn = key
        for hno_s, lrow in lv['rows'].items():
            hno = int(hno_s)
            if lrow['odds'] in (None, '', '---'):
                continue
            try:
                pubf = float(lrow['odds'])
            except ValueError:
                continue
            drow = c.execute("""
                SELECT he.id, he.win_odds, he.finishing_position FROM horse_entries he
                JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
                WHERE rd.race_date=? AND rd.venue=? AND r.race_no=? AND he.horse_no=?""",
                (d0, ven, rn, hno)).fetchone()
            if drow is None:
                continue
            if drow['id'] in r1_covered:
                continue
            if drow['win_odds'] is not None and abs(float(drow['win_odds']) - pubf) < 1e-9:
                continue
            plan.append({
                'action': 'UPDATE_WIN_ODDS', 'class': 'R6_odds_drift',
                'race': [d0, ven, rn], 'horse_no': hno, 'entry_id': drow['id'],
                'db_value': drow['win_odds'], 'official_value': pubf,
                'official_raw_cell': lrow['odds'],
                'is_winner': drow['finishing_position'] == 1,
                'evidence_url': lv['url'],
            })
            stats['R6_odds_drift'] += 1

    # ── races in the DB that we could NOT verify (no live evidence) ────────
    q = """SELECT rd.race_date, rd.venue, r.race_no, COUNT(he.id) AS n,
                 SUM(CASE WHEN he.finishing_position IS NOT NULL AND he.finishing_position>0
                     THEN 1 ELSE 0 END) AS withpos
          FROM races r JOIN race_days rd ON rd.id=r.race_day_id
          LEFT JOIN horse_entries he ON he.race_id=r.id
          GROUP BY r.id"""
    unver = []
    for d0, ven, rn, n, withpos in c.execute(q):
        if (d0, ven, rn) in by_race:
            continue
        if n and (withpos is None or withpos == 0):
            unver.append([d0, ven, rn, n])
    unresolved = unver

    out = {'plan': plan, 'stats': dict(stats),
           'unverified_races': unver,
           'live_status_counts': dict(Counter(v['status'] for v in live.values())),
           'live_races': len(live)}
    json.dump(out, open(OUT, 'w'), indent=1, default=str)
    print('=== REPAIR PLAN ===')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')
    print(f'  total actions: {len(plan)}')
    print(f'  races with NO live evidence (need separate handling): {len(unver)}')
    for u in unver[:40]:
        print('    ', u)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
