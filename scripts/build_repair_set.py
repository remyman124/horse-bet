#!/usr/bin/env python3
"""
REPAIR SET BUILDER — derives an explicit, reproducible per-race repair list from
OFFICIAL HKJC data.  Every action is justified by a live fetch of the official
results page, never by heuristics alone.

Defect classes handled:
  A. RAW_DIVIDEND_CONTAMINATION
     DB stores win_odds = raw HK$ dividend (10.5) for the WINNER, while the
     official 'Win Odds' column publishes the real odds.  Root cause: an old
     scraper wrote the dividend instead of the odds.
     Fix: win_odds := the officially published 'Win Odds' value for that horse.
     NOTE: when the published value is <= 1 (HKJC prints '1' for a 1.05 shot)
     the odds are NOT recoverable from the Win Odds column -> set NULL, do NOT
     invent a value from the dividend.

  B. PHANTOM_RACE_DAY
     A race_day/races with entries but the official page says 'No information'
     -> the meeting never happened.  Fix: delete the fabricated rows.

  C. MISSING_ENTRIES
     DB has fewer runners than the official page.  Fix: INSERT the missing
     horse_entries rows (with official odds + finishing position).

  D. MISSING_RESULTS
     Races with entries but no finishing positions (racecard loaded, results
     never scraped).  Fix: populate finishing_position from the official page.

  E. WINNER_ROW_MISSING
     Race has entries and positions, but the position-1 row is absent.
     Fix: INSERT the missing winner row.

Usage:
    python3 scripts/build_repair_set.py --out /tmp/repair_set.json [--jobs 6]
    python3 scripts/apply_repair_set.py --set /tmp/repair_set.json --dry-run
    python3 scripts/apply_repair_set.py --set /tmp/repair_set.json
"""
import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from bs4 import BeautifulSoup

DB_PATH = '/home/work/horse-bet/data/horse.db'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36'}
RESULTS_URL = ('https://racing.hkjc.com/en-us/local/information/localresults'
               '?racedate={date}&Racecourse={venue}&RaceNo={race_no}')


def _num(s):
    s = (s or '').strip().replace(',', '').replace('$', '')
    try:
        return float(s)
    except ValueError:
        return None


def parse_results_page(txt):
    """Header-driven parse of an official HKJC results page.

    Returns None when the page carries no results ('No information').
    Returns dict with 'horses' {horse_no: {...}} and dividends.
    """
    if not txt or 'No information' in txt:
        return None
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')

    res_table = None
    hdr_map = []
    for t in tables:
        rows = t.find_all('tr')
        if not rows:
            continue
        hdr = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
        if 'Pla.' in hdr and 'Horse No.' in hdr:
            res_table, hdr_map = t, hdr
            break
    if res_table is None:
        return None

    def idx(*names, default=None):
        for n in names:
            if n in hdr_map:
                return hdr_map.index(n)
        return default

    i_pos = idx('Pla.', default=0)
    i_hno = idx('Horse No.', 'No.', default=1)
    i_name = idx('Horse', default=2)
    i_jk = idx('Jockey', default=3)
    i_tr = idx('Trainer', default=4)
    i_awt = idx('Act. Wt.', default=5)
    i_dwt = idx('Declar. Horse Wt.', 'Declar. Wt.')
    i_draw = idx('Dr.', 'Draw', default=7)
    i_odds = idx('Win Odds', 'Odds', 'Odds(HK$)', 'Win Odds(HK$)')

    horses = {}
    for row in res_table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) <= i_hno:
            continue
        hn = cells[i_hno].get_text(strip=True)
        if not hn.isdigit():
            continue
        pos_raw = cells[i_pos].get_text(strip=True)
        odds = None
        if i_odds is not None and len(cells) > i_odds:
            odds = _num(cells[i_odds].get_text())
            if odds is not None and odds <= 1:
                # HKJC prints '1' for a ~1.05 favourite: the published column
                # truncates, so the real odds are NOT recoverable from it.
                odds = None
        name = cells[i_name].get_text(strip=True) if len(cells) > i_name else ''
        import re
        name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
        name = re.sub(r'\s+', ' ', name)
        jk = cells[i_jk].get_text(strip=True) if len(cells) > i_jk else ''
        tr = cells[i_tr].get_text(strip=True) if len(cells) > i_tr else ''
        awt = cells[i_awt].get_text(strip=True) if len(cells) > i_awt else ''
        dwt = cells[i_dwt].get_text(strip=True) if i_dwt is not None and len(cells) > i_dwt else ''
        dr = cells[i_draw].get_text(strip=True) if len(cells) > i_draw else ''
        horses[int(hn)] = {
            'pos': int(pos_raw) if pos_raw.isdigit() else pos_raw,
            'name': name,
            'jockey': jk,
            'trainer': tr,
            'act_wt': int(awt) if awt.isdigit() else None,
            'decl_wt': int(dwt) if dwt.isdigit() else None,
            'draw': int(dr) if dr.isdigit() else None,
            'odds': odds,
        }
    if not horses:
        return None

    win_div = None
    win_horse_no = None
    place_divs = []
    for t in tables:
        found = False
        for row in t.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 3:
                continue
            key = cells[0].get_text(strip=True)
            amount = _num(cells[2].get_text())
            if amount is None:
                continue
            combo = cells[1].get_text(strip=True)
            if key == 'WIN':
                win_div = amount
                if combo.isdigit():
                    win_horse_no = int(combo)
                found = True
            elif key == 'PLACE':
                if combo.isdigit():
                    place_divs.append((int(combo), amount))
                found = True
            elif key == '' and place_divs and combo.isdigit():
                place_divs.append((int(combo), amount))
        if found:
            break

    return {'horses': horses, 'win_dividend': win_div,
            'win_horse_no': win_horse_no, 'place_dividends': place_divs}


def fetch(client, race_date, venue, race_no):
    ds = str(race_date).replace('-', '/')
    url = RESULTS_URL.format(date=ds, venue=venue, race_no=race_no)
    try:
        r = client.get(url, timeout=40.0)
        if r.status_code != 200:
            return None, 'http_%d' % r.status_code
        return parse_results_page(r.text), 'ok'
    except Exception as e:                      # noqa: BLE001
        return None, 'err_%s' % type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/repair_set.json')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True, timeout=60)
    cur = conn.cursor()

    # --- candidate races: anything with entries, plus every race_day/race
    cur.execute("""
        SELECT rd.race_date, rd.venue, r.race_no, r.id,
               (SELECT COUNT(*) FROM horse_entries he WHERE he.race_id = r.id)
        FROM races r JOIN race_days rd ON r.race_day_id = rd.id
        ORDER BY rd.race_date, r.race_no
    """)
    targets = cur.fetchall()
    if args.limit:
        targets = targets[:args.limit]
    print(f'candidate races: {len(targets)}', flush=True)

    # DB snapshot per race (read once, in the main thread)
    db_rows = {}
    for rdate, venue, race_no, rid, n in targets:
        rows = cur.execute(
            'SELECT horse_no, horse_name, finishing_position, win_odds, jockey, trainer,'
            ' draw, finish_type FROM horse_entries WHERE race_id=? ORDER BY horse_no', (rid,)
        ).fetchall()
        db_rows[rid] = rows

    out = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'source': 'official HKJC localresults page (header-driven parse)',
           'repair': {}, 'detail': []}

    def classify(rdate, venue, race_no, rid, rows):
        """Return list of actions for this race given the official parse."""
        acts = []
        if parsed is None:
            if rows:
                acts.append({'action': 'DELETE_RACE_ENTRIES', 'reason': 'official page has no result (meeting not held)'})
                acts.append({'action': 'DELETE_RACE', 'reason': 'official page has no result (meeting not held)'})
                acts.append({'action': 'DELETE_RACE_RESULT', 'reason': 'official page has no result (meeting not held)'})
            return acts
        db_map = {r[0]: r for r in rows}
        live = parsed['horses']
        # 1) missing entries
        for hno, h in sorted(live.items()):
            if hno not in db_map:
                acts.append({'action': 'INSERT_ENTRY', 'horse_no': hno, 'data': h,
                             'reason': 'official results page lists this runner; DB has no row'})
        # 2) odds contamination / mismatch
        for hno, h in sorted(live.items()):
            if hno not in db_map:
                continue
            row = db_map[hno]
            db_odds = row[3]
            live_odds = h['odds']
            if live_odds is None and db_odds is not None:
                acts.append({'action': 'SET_ODDS_NULL', 'horse_no': hno,
                             'db_odds': db_odds, 'live_odds': None,
                             'reason': 'published Win Odds column is truncated (<=1); odds not recoverable -> keep missing'})
            elif live_odds is not None and (db_odds is None or abs(db_odds - live_odds) > 1e-9):
                acts.append({'action': 'SET_ODDS', 'horse_no': hno,
                             'db_odds': db_odds, 'live_odds': live_odds,
                             'reason': 'DB odds differ from officially published Win Odds column'})
        # 3) missing results (finishing position absent) for existing rows
        for hno, h in sorted(live.items()):
            if hno not in db_map:
                continue
            row = db_map[hno]
            if row[2] is None:
                acts.append({'action': 'SET_POSITION', 'horse_no': hno, 'live_pos': h['pos'],
                             'reason': 'official results page has finishing position; DB has none'})
        # 4) winner row missing
        if rows and not any(r[2] == 1 for r in rows) and any(h['pos'] == 1 for h in live.values()):
            acts.append({'action': 'INSERT_WINNER_ROW', 'reason': 'race has no finishing_position=1 row'})
        return acts

    parsed = None   # closure variable set by the worker
    results = {}

    def work(item):
        rdate, venue, race_no, rid, n = item
        with httpx.Client(headers=HEADERS, follow_redirects=True) as cl:
            p, status = fetch(cl, rdate, venue, race_no)
        return (rdate, venue, race_no, rid, p, status)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for rdate, venue, race_no, rid, p, status in ex.map(work, targets):
            done += 1
            if done % 1000 == 0:
                print(f'  fetched {done}/{len(targets)} ({time.time()-t0:.0f}s)', flush=True)
            rows = db_rows[rid]
            if p is None:
                if rows:
                    out['detail'].append({
                        'race_date': str(rdate), 'venue': venue, 'race_no': race_no,
                        'db_runners': len(rows), 'live': 'NO_DATA',
                        'actions': [{'action': 'DELETE_RACE_ENTRIES',
                                     'reason': 'official page has no result (meeting not held)'},
                                    {'action': 'DELETE_RACE',
                                     'reason': 'official page has no result (meeting not held)'},
                                    {'action': 'DELETE_RACE_RESULT',
                                     'reason': 'official page has no result (meeting not held)'}]})
                    out['repair'].setdefault('phantom_race', []).append(
                        {'race_date': str(rdate), 'venue': venue, 'race_no': race_no,
                         'db_runners': len(rows), 'live': 'NO_DATA'})
                continue
            parsed = p
            acts = classify(rdate, venue, race_no, rid, rows)
            if acts:
                out['detail'].append({
                    'race_date': str(rdate), 'venue': venue, 'race_no': race_no,
                    'db_runners': len(rows), 'live_runners': len(p['horses']),
                    'actions': acts})

    # summarise
    summary = {}
    for d in out['detail']:
        for a in d['actions']:
            summary[a['action']] = summary.get(a['action'], 0) + 1
    out['summary'] = summary
    out['races_affected'] = len(out['detail'])
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'\nwrote {args.out}: {len(out["detail"])} races affected, actions={summary}', flush=True)
    conn.close()


if __name__ == '__main__':
    sys.exit(main())
