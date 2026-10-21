"""
Build a dividend repair plan from the official HKJC evidence only.

Scope
-----
The official results page publishes, in the dividends table:

    ['WIN',  '<winning horse no>', '<dividend per $10 stake>']
    ['<other winner>', '<dividend>']      <- dead heat, extra winners
    ['PLACE','<horse no>', '<dividend>']  ... etc

The OLD parser grabbed cell[1] of a row instead of cell[2], so the winner's
HORSE NUMBER ended up in `race_results.win_dividend` (e.g. 5.0 instead of
180.00). The repair value is ALWAYS the official amount cell -- never a
derivation from the odds, never the dividend divided by ten.

This builder only ever emits two actions:
    UPDATE_WIN_DIVIDEND  - the row exists but holds the wrong value
    INSERT_RACE_RESULT   - the race has runners but no result row at all

The input is the persisted evidence file (data/official_evidence/
official_results.json), so the plan is reproducible.
"""
import json
import os
import sqlite3
import sys

ROOT = '/home/work/horse-bet'
DB = os.path.join(ROOT, 'data', 'horse.db')
EV = os.path.join(ROOT, 'data', 'official_evidence', 'official_results.json')
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dividend_plan_v4.json'

ev = json.load(open(EV))
conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=300)
conn.row_factory = sqlite3.Row

plan = []
stats = {'UPDATE_WIN_DIVIDEND': 0, 'INSERT_RACE_RESULT': 0,
         'already_correct': 0, 'no_race': 0, 'no_amount': 0}

for key, v in ev.items():
    if not v.get('rows'):
        continue
    w = (v.get('dividends') or {}).get('win')
    if not w or not w.get('amount'):
        stats['no_amount'] += 1
        continue
    try:
        amount = float(w['amount'])
    except (TypeError, ValueError):
        stats['no_amount'] += 1
        continue
    d0, ven, rn = key.split('|')
    race = conn.execute("""SELECT r.id FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                           WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?""",
                        (d0, ven, int(rn))).fetchone()
    if race is None:
        stats['no_race'] += 1
        continue
    rid = race['id']
    rr = conn.execute('SELECT id, win_dividend FROM race_results WHERE race_id=?',
                      (rid,)).fetchone()
    if rr is None:
        plan.append({'action': 'INSERT_RACE_RESULT', 'class': 'R10_missing_result_row',
                     'race': [d0, ven, rn], 'official_value': amount,
                     'evidence_url': v.get('url'),
                     'note': 'official WIN dividend cell'})
        stats['INSERT_RACE_RESULT'] += 1
        continue
    cur = rr['win_dividend']
    try:
        if cur is not None and abs(float(cur) - amount) < 1e-9:
            stats['already_correct'] += 1
            continue
    except (TypeError, ValueError):
        pass
    plan.append({'action': 'UPDATE_WIN_DIVIDEND', 'class': 'R7_dividend_cell_misread',
                 'race': [d0, ven, rn], 'horse_no': int(w['combo']) if str(w.get('combo', '')).isdigit() else None,
                 'db_value': cur, 'official_value': amount,
                 'official_raw_cell': w['amount'],
                 'evidence_url': v.get('url'),
                 'note': 'official WIN dividend cell (not derived from odds)'})
    stats['UPDATE_WIN_DIVIDEND'] += 1

conn.close()
json.dump({'plan': plan, 'stats': stats}, open(OUT, 'w'), indent=1)
print('=== DIVIDEND PLAN (official cells only) ===')
for k, n in sorted(stats.items()):
    print(f'  {k}: {n}')
print(f'  total actions: {len(plan)}')
for p in plan[:12]:
    print(f"   {p['action']:22s} {p['race']} db={p.get('db_value')} -> official={p['official_value']}")
print(f'wrote {OUT}')
