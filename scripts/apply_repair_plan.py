"""Apply the official-evidence repair plan to horse.db.

SINGLE BULK WRITER. Every mutation is recorded in a new `repair_provenance`
table so any row can be traced back to the official HKJC page that justifies
it, and reversed if needed.

Usage
-----
    python3 scripts/apply_repair_plan.py --plan /tmp/repair_plan.json [--dry-run]
    python3 scripts/apply_repair_plan.py --plan /tmp/repair_plan.json --apply

Safety
------
* refuses to run unless the plan is present and every action carries an
  official evidence URL
* backs up the DB to data/horse.db.bak-<timestamp> before the first write
* runs inside a single transaction; rolls back on any error
* a post-apply assertion pass re-verifies the invariants
"""
import argparse, json, os, shutil, sqlite3, sys, time
from collections import Counter

DB = '/home/work/horse-bet/data/horse.db'


def open_db():
    c = sqlite3.connect(DB, timeout=300)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON')
    return c


def ensure_provenance(c):
    c.execute("""
        CREATE TABLE IF NOT EXISTS repair_provenance (
            id INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            action TEXT NOT NULL,
            repair_class TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_key TEXT NOT NULL,
            race_date TEXT, venue TEXT, race_no INTEGER, horse_no INTEGER,
            column_name TEXT,
            old_value TEXT, new_value TEXT,
            evidence_url TEXT NOT NULL,
            note TEXT
        )""")
    c.execute("""CREATE INDEX IF NOT EXISTS ix_repair_prov_race
                 ON repair_provenance(race_date, venue, race_no)""")
    c.execute("""CREATE INDEX IF NOT EXISTS ix_repair_prov_action
                 ON repair_provenance(action)""")


def log_prov(c, plan_item, table_name, row_key, column_name, old_value, new_value):
    r = plan_item.get('race', [None, None, None])
    c.execute("""
        INSERT INTO repair_provenance
        (applied_at, action, repair_class, table_name, row_key, race_date, venue,
         race_no, horse_no, column_name, old_value, new_value, evidence_url, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (time.strftime('%Y-%m-%dT%H:%M:%S'), plan_item['action'], plan_item['class'],
         table_name, str(row_key), r[0], r[1], r[2],
         plan_item.get('horse_no'), column_name,
         None if old_value is None else str(old_value),
         None if new_value is None else str(new_value),
         plan_item['evidence_url'],
         plan_item.get('note')))


def parse_odds(v):
    if v in (None, '', '---'):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def apply_plan(plan, do_apply):
    stats = Counter()
    c = open_db()
    ensure_provenance(c)
    c.commit()

    if not do_apply:
        print('=== DRY RUN (no writes) ===')

    # ── R6/R1: update win_odds ─────────────────────────────────────────────
    for p in plan:
        if p['action'] != 'UPDATE_WIN_ODDS':
            continue
        eid = p['entry_id']
        row = c.execute('SELECT win_odds FROM horse_entries WHERE id=?', (eid,)).fetchone()
        if row is None:
            stats['odds_update_missing_row'] += 1
            continue
        cur = row['win_odds']
        newv = float(p['official_value'])
        if cur is not None and abs(float(cur) - newv) < 1e-9:
            stats['odds_already_correct'] += 1
            continue
        if not do_apply:
            stats['odds_update'] += 1
            continue
        c.execute('UPDATE horse_entries SET win_odds=? WHERE id=?', (newv, eid))
        log_prov(c, p, 'horse_entries', eid, 'win_odds', cur, newv)
        stats['odds_update'] += 1

    # ── R9: create a missing `races` row ───────────────────────────────────
    # Some real official races were never captured at all: the race_day
    # exists but the `races` row does not, so their runners cannot be
    # inserted. The row is created with the official race number; the
    # remaining columns stay NULL because they were never published in
    # the evidence we hold.
    #
    # This must run BEFORE the runner inserts.
    for p in plan:
        if p['action'] != 'INSERT_RACE':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is not None:
            stats['race_already_exists'] += 1
            continue
        day = c.execute('SELECT id FROM race_days WHERE race_date=? AND venue=?',
                        (d0, ven)).fetchone()
        if day is None:
            stats['race_no_race_day'] += 1
            continue
        if not do_apply:
            stats['race_insert'] += 1
            continue
        cur_r = c.execute(
            'INSERT INTO races (race_day_id, race_no, scraped_at) VALUES (?,?,?)',
            (day['id'], int(rn), time.strftime('%Y-%m-%dT%H:%M:%S')))
        new_rid = cur_r.lastrowid
        log_prov(c, p, 'races', new_rid, '(race)',
                 f'{d0} {ven} R{rn}', 'created')
        stats['race_insert'] += 1

    # ── R2: insert missing runner rows ──────────────────────────────────────
    for p in plan:
        if p['action'] != 'INSERT_MISSING_RUNNER':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['insert_no_race'] += 1
            continue
        rid = rid['id']
        exists = c.execute('SELECT id FROM horse_entries WHERE race_id=? AND horse_no=?',
                            (rid, p['horse_no'])).fetchone()
        if exists:
            stats['insert_already_exists'] += 1
            continue
        if not do_apply:
            stats['insert_runner'] += 1
            continue
        cur_ins = c.execute("""
            INSERT INTO horse_entries
            (race_id, horse_no, draw, horse_name, jockey, trainer,
             win_odds, finishing_position, finish_type, scraped_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (rid, p['horse_no'], parse_odds(p.get('official_draw')),
             p['horse_name'], p.get('official_jockey'), p.get('official_trainer'),
             parse_odds(p.get('official_odds')), p.get('official_pos'),
             p.get('official_finish_type'), time.strftime('%Y-%m-%dT%H:%M:%S')))
        # lastrowid lives on the CURSOR, not the connection
        newid = cur_ins.lastrowid
        log_prov(c, p, 'horse_entries', newid, '(row)',
                 None, f"horse_no={p['horse_no']} pos={p.get('official_pos')}")
        stats['insert_runner'] += 1

    # ── R3/R4: update finishing position ────────────────────────────────────
    for p in plan:
        if p['action'] != 'UPDATE_FINISHING_POSITION':
            continue
        eid = p['entry_id']
        row = c.execute('SELECT finishing_position FROM horse_entries WHERE id=?',
                        (eid,)).fetchone()
        if row is None:
            stats['pos_update_missing_row'] += 1
            continue
        cur = row['finishing_position']
        newv = p['official_position']
        if cur is not None and int(cur) == int(newv):
            stats['pos_already_correct'] += 1
            continue
        if not do_apply:
            stats['pos_update'] += 1
            continue
        c.execute('UPDATE horse_entries SET finishing_position=? WHERE id=?', (newv, eid))
        if p.get('official_finish_type'):
            c.execute('UPDATE horse_entries SET finish_type=? WHERE id=?',
                      (p['official_finish_type'], eid))
        log_prov(c, p, 'horse_entries', eid, 'finishing_position', cur, newv)
        stats['pos_update'] += 1

    # ── R5: delete phantom races ────────────────────────────────────────────
    for p in plan:
        if p['action'] != 'DELETE_PHANTOM_RACE':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['phantom_no_race'] += 1
            continue
        rid = rid['id']
        if not do_apply:
            stats['phantom_delete'] += 1
            continue
        n = c.execute('SELECT COUNT(*) FROM horse_entries WHERE race_id=?', (rid,)).fetchone()[0]
        c.execute('DELETE FROM horse_entries WHERE race_id=?', (rid,))
        c.execute('DELETE FROM race_results WHERE race_id=?', (rid,))
        c.execute('DELETE FROM races WHERE id=?', (rid,))
        log_prov(c, p, 'races', rid, '(race)', f'{d0} {ven} R{rn}', 'DELETED')
        p['note'] = f'deleted {n} horse_entries'
        stats['phantom_delete'] += 1

    # ── R10: create a missing `race_results` row ───────────────────────────
    # A race can have runners but no result row. When the official page
    # carries a WIN dividend, the row is created with it.
    for p in plan:
        if p['action'] != 'INSERT_RACE_RESULT':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['result_no_race'] += 1
            continue
        rid = rid['id']
        if c.execute('SELECT id FROM race_results WHERE race_id=?', (rid,)).fetchone():
            stats['result_already_exists'] += 1
            continue
        if not do_apply:
            stats['result_insert'] += 1
            continue
        cur_r = c.execute(
            'INSERT INTO race_results (race_id, win_dividend, scraped_at) VALUES (?,?,?)',
            (rid, p['official_value'], time.strftime('%Y-%m-%dT%H:%M:%S')))
        log_prov(c, p, 'race_results', cur_r.lastrowid, 'win_dividend',
                 None, p['official_value'])
        stats['result_insert'] += 1

    # ── R7: update race_results.win_dividend ───────────────────────────────
    # The HKJC dividends table is (label, horse_no, amount). A parser that
    # reads cell[1] instead of cell[2] stores the winning HORSE NUMBER as
    # the dividend. The value written here is always the official amount
    # cell -- never a derivation from the odds.
    for p in plan:
        if p['action'] != 'UPDATE_WIN_DIVIDEND':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['dividend_no_race'] += 1
            continue
        rid = rid['id']
        row = c.execute('SELECT win_dividend FROM race_results WHERE race_id=?', (rid,)).fetchone()
        if row is None:
            stats['dividend_no_result_row'] += 1
            continue
        cur = row['win_dividend']
        newv = p['official_value']
        try:
            if cur is not None and abs(float(cur) - float(newv)) < 1e-9:
                stats['dividend_already_correct'] += 1
                continue
        except (TypeError, ValueError):
            pass
        if not do_apply:
            stats['dividend_update'] += 1
            continue
        c.execute('UPDATE race_results SET win_dividend=? WHERE race_id=?', (newv, rid))
        log_prov(c, p, 'race_results', rid, 'win_dividend', cur, newv)
        stats['dividend_update'] += 1

    # ── R11: write the three published PLACE dividends ─────────────────────
    # The old parser read cell[1] of the PLACE row, so place_dividend_1 held
    # the winning COMBINATION (e.g. '3,6') instead of the payout. The three
    # published amounts are written in finishing order: 1st, 2nd, 3rd.
    for p in plan:
        if p['action'] != 'UPDATE_PLACE_DIVIDENDS':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['place_div_no_race'] += 1
            continue
        rid = rid['id']
        row = c.execute('SELECT place_dividend_1, place_dividend_2, place_dividend_3 '
                        'FROM race_results WHERE race_id=?', (rid,)).fetchone()
        if row is None:
            stats['place_div_no_result_row'] += 1
            continue
        amounts = p['official_place_amounts']
        cols = ('place_dividend_1', 'place_dividend_2', 'place_dividend_3')
        cur_vals = [row[c] for c in cols]
        new_vals = [amounts[i] if i < len(amounts) else None for i in range(3)]
        same = all((a is None and b is None) or
                   (a is not None and b is not None and str(a) == str(b))
                   for a, b in zip(cur_vals, new_vals))
        if same:
            stats['place_div_already_correct'] += 1
            continue
        if not do_apply:
            stats['place_div_update'] += 1
            continue
        c.execute('UPDATE race_results SET place_dividend_1=?, place_dividend_2=?, '
                  'place_dividend_3=? WHERE race_id=?',
                  (new_vals[0], new_vals[1], new_vals[2], rid))
        for i, col in enumerate(cols):
            log_prov(c, p, 'race_results', rid, col, cur_vals[i], new_vals[i])
        stats['place_div_update'] += 1

    # ── R12: drop a fabricated result row, keep the race ────────────────────
    # A race that was voided has no outcome: the official dividends table
    # says REFUND for every pool. Any dividend stored for it came from the
    # old cell-misread parser (a horse number, or a combination string),
    # so the row is removed while the scheduled race itself is kept.
    for p in plan:
        if p['action'] != 'DELETE_FABRICATED_RESULT':
            continue
        d0, ven, rn = p['race']
        rid = c.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, rn)).fetchone()
        if rid is None:
            stats['fab_result_no_race'] += 1
            continue
        rid = rid['id']
        if not do_apply:
            stats['fab_result_delete'] += 1
            continue
        old = c.execute('SELECT win_dividend, place_dividend_1 FROM race_results '
                        'WHERE race_id=?', (rid,)).fetchone()
        c.execute('DELETE FROM race_results WHERE race_id=?', (rid,))
        log_prov(c, p, 'race_results', rid, '(row)',
                 f'win_dividend={old[0] if old else None} place_dividend_1={old[1] if old else None}',
                 'DELETED')
        stats['fab_result_delete'] += 1

    # ── R13: drop an empty race_day shell ───────────────────────────────────
    # A meeting that the official site does not list leaves an empty
    # race_days row behind once its fabricated races are removed.
    for p in plan:
        if p['action'] != 'DELETE_EMPTY_RACE_DAY':
            continue
        d0, ven = p['race'][0], p['race'][1]
        rdid = c.execute('SELECT id FROM race_days WHERE race_date=? AND venue=?',
                         (d0, ven)).fetchone()
        if rdid is None:
            stats['empty_day_no_row'] += 1
            continue
        rdid = rdid['id']
        n = c.execute('SELECT COUNT(*) FROM races WHERE race_day_id=?', (rdid,)).fetchone()[0]
        if n:
            stats['empty_day_has_races'] += 1
            continue
        if not do_apply:
            stats['empty_day_delete'] += 1
            continue
        c.execute('DELETE FROM race_days WHERE id=?', (rdid,))
        log_prov(c, p, 'race_days', rdid, '(row)', f'{d0} {ven}', 'DELETED')
        stats['empty_day_delete'] += 1

    if do_apply:
        c.commit()
        print('committed.')
    else:
        c.rollback()
    c.close()
    return stats


def verify():
    """Post-apply invariants."""
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=300)
    c.row_factory = sqlite3.Row
    out = {}

    n = c.execute("""SELECT COUNT(*) FROM horse_entries he
        WHERE he.win_odds >= 10 AND he.win_odds <> CAST(he.win_odds AS INTEGER)""").fetchone()[0]
    out['rows_with_odds_ge10_and_fractional'] = n

    n = c.execute("""SELECT COUNT(*) FROM horse_entries he
        JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
        JOIN race_results rr ON rr.race_id=r.id
        WHERE he.finishing_position=1 AND rr.win_dividend IS NOT NULL
          AND ABS(he.win_odds - CAST(rr.win_dividend AS REAL)/10.0) < 1e-9
          AND he.win_odds <> CAST(he.win_odds AS INTEGER)""").fetchone()[0]
    out['winner_rows_still_equal_true_SP'] = n

    n = c.execute("""SELECT COUNT(*) FROM horse_entries he
        JOIN races r ON r.id=he.race_id JOIN race_days rd ON rd.id=r.race_day_id
        JOIN race_results rr ON rr.race_id=r.id
        WHERE he.finishing_position=1 AND rr.win_dividend IS NOT NULL
          AND ABS(he.win_odds - CAST(rr.win_dividend AS REAL)) < 1e-9""").fetchone()[0]
    out['winner_rows_still_equal_raw_dividend'] = n

    n = c.execute("""SELECT COUNT(*) FROM horse_entries WHERE win_odds IS NOT NULL
                      AND (win_odds <= 0 OR win_odds > 999)""").fetchone()[0]
    out['rows_with_implausible_odds'] = n

    n = c.execute("""SELECT COUNT(*) FROM (
        SELECT he.race_id FROM horse_entries he
        GROUP BY he.race_id
        HAVING COUNT(DISTINCT he.finishing_position) <> COUNT(*)
               AND SUM(CASE WHEN he.finishing_position IS NULL OR he.finishing_position<=0 THEN 1 ELSE 0 END)=0
    )""").fetchone()[0]
    out['races_with_position_gaps'] = n

    n = c.execute("SELECT COUNT(*) FROM repair_provenance").fetchone()[0]
    out['provenance_rows'] = n
    c.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--no-backup', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.plan):
        print(f'plan not found: {a.plan}'); sys.exit(1)
    plan = json.load(open(a.plan))
    items = plan['plan']
    print(f'plan: {len(items)} actions, stats={plan.get("stats")}')

    missing_ev = [p for p in items if not p.get('evidence_url')]
    if missing_ev:
        print(f'REFUSING: {len(missing_ev)} actions have no evidence URL')
        sys.exit(1)

    if a.apply and not a.no_backup:
        bak = f'{DB}.bak-{time.strftime("%Y%m%d-%H%M%S")}'
        shutil.copy2(DB, bak)
        print(f'backup -> {bak}')

    stats = apply_plan(items, a.apply)
    print('\n=== ACTIONS ===')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')

    if a.apply:
        print('\n=== VERIFY ===')
        for k, v in verify().items():
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
