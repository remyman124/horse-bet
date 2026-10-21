#!/usr/bin/env python3
"""
Build a repair plan for races where the DB disagrees with the official
runners table, using the official PLACE evidence to find them.

How these races are found
-------------------------
`scripts/fetch_place_dividends.py` cross-checks the official PLACE
dividends against the DB's top-3 finishers on the same page. A mismatch
means one of two things:

  * the official page shows a DEAD HEAT for 3rd place, so there are four
    PLACE dividends (1st, 2nd, 3rd-DH, 3rd-DH) while the DB only knows
    three placers -- the DB is missing a runner; or
  * the DB is missing a runner or has the wrong finishing position.

In every case the authoritative source is the official runners table on
the same page, which also carries the pre-race published Win Odds -- so
the missing runner can be inserted with the odds the market actually
quoted, not with anything derived.

Every action carries the official evidence URL, and the applier records
it in repair_provenance.

Usage
-----
    python3 scripts/build_dh_gap_plan.py --ckpt /tmp/place_div.ckpt.json
    python3 scripts/build_dh_gap_plan.py --apply
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import bs4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, 'data', 'horse.db')
OUT = '/tmp/dh_gap_plan.json'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}
_lock = threading.Lock()
_last = [0.0]


def get(url, tries=4, timeout=60):
    for a in range(tries):
        try:
            with _lock:
                gap = 1.2 - (time.time() - _last[0])
                if gap > 0:
                    time.sleep(gap)
                _last[0] = time.time()
            req = urllib.request.Request(url, headers=HEADERS)
            html = urllib.request.urlopen(req, timeout=timeout).read().decode('utf-8', 'ignore')
            if len(html) > 20000:
                return html
        except Exception:
            if a == tries - 1:
                return None
        time.sleep(2 + 2 * a)
    return None


def norm_pos(p):
    """Official 'Pla.' cell -> (int position, finish_type).

    '3 DH' -> (3, 'DH')   '5' -> (5, None)
    'WV'   -> (None, 'WV')   'DNF' -> (None, 'DNF')
    """
    if p is None:
        return None, None
    s = str(p).strip()
    m = re.match(r'^(\d+)\s*DH$', s, re.I)
    if m:
        return int(m.group(1)), 'DH'
    m = re.match(r'^(\d+)$', s)
    if m:
        return int(m.group(1)), None
    if not s:
        return None, None
    return None, s.upper()


def name_key(n):
    """'SKY FIELD (D216)' -> 'SKY FIELD'"""
    return re.sub(r'\s*\([A-Z]{0,3}\d+\)\s*$', '', str(n or '')).strip().upper()


def parse_runners(html):
    """-> {horse_no: {'pos','name','odds','draw','jockey','trainer'}}"""
    soup = bs4.BeautifulSoup(html, 'html.parser')
    for t in soup.find_all('table'):
        rows = t.find_all('tr')
        if not rows:
            continue
        hdr = [x.get_text(' ', strip=True) for x in rows[0].find_all(['td', 'th'])]
        low = [h.lower() for h in hdr]
        if 'horse no.' not in low or not any('pla' in h for h in low):
            continue
        idx = {}
        for want in ('pla.', 'horse no.', 'horse', 'jockey', 'trainer', 'dr.', 'win odds'):
            for i, h in enumerate(low):
                if h == want or h.rstrip('.') == want.rstrip('.'):
                    idx[want] = i
                    break
        if 'horse no.' not in idx:
            continue
        out = {}
        for r in rows[1:]:
            cs = [x.get_text(' ', strip=True) for x in r.find_all(['td', 'th'])]
            if len(cs) <= idx['horse no.']:
                continue
            hn = cs[idx['horse no.']].strip()
            if not hn.isdigit():
                continue
            def cell(name):
                i = idx.get(name)
                return cs[i].strip() if i is not None and i < len(cs) else ''
            out[int(hn)] = {
                'pos': cell('pla.'),
                'name': cell('horse'),
                'odds': cell('win odds'),
                'draw': cell('dr.'),
                'jockey': cell('jockey'),
                'trainer': cell('trainer'),
            }
        if out:
            return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/tmp/place_div.ckpt.json')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--mode', default='invalid', choices=['invalid', 'empty', 'both'],
                    help="invalid: races whose official PLACE disagrees with the DB; "
                         "empty: DB races that have a result row but no runners at all; "
                         "both: the union")
    a = ap.parse_args()

    raw = json.load(open(a.ckpt))
    targets = [k for k, v in raw.items()
               if isinstance(v, dict) and v.get('status') == 'INVALID']
    print(f'INVALID place races to reconcile: {len(targets)}')

    # races that carry a result row but no runners at all. The old scraper
    # looked for a 'No.' column while the modern results pages use
    # 'Horse No.', so every horse of those races was dropped.
    empty = []
    if a.mode in ('empty', 'both'):
        conn0 = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=180)
        for r in conn0.execute('''SELECT rd.race_date, rd.venue, r.race_no
                                   FROM races r JOIN race_days rd ON rd.id=r.race_day_id
                                   WHERE (SELECT COUNT(*) FROM horse_entries he WHERE he.race_id=r.id)=0
                                     AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id=r.id)>0
                                   ORDER BY rd.race_date'''):
            empty.append(f'{r[0]}|{r[1]}|{r[2]}')
        conn0.close()
        print(f'races with a result row but zero runners: {len(empty)}')
    if a.mode == 'both':
        targets = sorted(set(targets) | set(empty))
    elif a.mode == 'empty':
        targets = empty
    print(f'target races: {len(targets)}')

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=180)
    conn.row_factory = sqlite3.Row

    def db_rows(d0, ven, rn):
        return {r['horse_no']: r for r in conn.execute('''
            SELECT he.id, he.horse_no, he.horse_name, he.win_odds,
                   he.finishing_position, he.finish_type, he.draw
            FROM horse_entries he
            JOIN races r ON r.id = he.race_id
            JOIN race_days rd ON rd.id = r.race_day_id
            WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?
        ''', (d0, ven, int(rn)))}

    cache = {}
    cache_path = '/tmp/runners_cache.json'
    if os.path.exists(cache_path):
        raw_cache = json.load(open(cache_path))
        # JSON turns the int horse numbers into strings; put them back, or
        # every cached horse looks missing from the DB
        cache = {k: {int(h): v for h, v in rows.items()}
                 for k, rows in raw_cache.items()}
        print(f'runners cache: {len(cache)} races already parsed')

    def work(key):
        if key in cache:
            return key, cache[key]
        d0, ven, rn = key.split('|')
        url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
               f'?racedate={d0.replace("-", "/")}&Racecourse={ven}&RaceNo={rn}')
        html = get(url)
        if not html:
            return key, None
        return key, parse_runners(html)

    fetched = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for key, runners in ex.map(work, targets):
            if runners:
                fetched[key] = runners
    cache.update(fetched)
    json.dump(cache, open(cache_path, 'w'))
    print(f'official runners table parsed for {len(fetched)} races '
          f'(cache now {len(cache)})')

    plan = []
    stats = {'INSERT_MISSING_RUNNER': 0, 'UPDATE_FINISHING_POSITION': 0,
             'UPDATE_WIN_ODDS': 0, 'UPDATE_PLACE_DIVIDENDS': 0,
             'ok': 0, 'no_fetch': 0}
    for key, runners in fetched.items():
        d0, ven, rn = key.split('|')
        url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
               f'?racedate={d0.replace("-", "/")}&Racecourse={ven}&RaceNo={rn}')
        db = db_rows(d0, ven, rn)
        changed = False
        for hno, o in sorted(runners.items()):
            pos, ftype = norm_pos(o['pos'])
            if pos is None:
                # withdrawn / did not finish: the DB's convention is to
                # omit these, so do not invent a row for them
                continue
            if hno not in db:
                plan.append({
                    'action': 'INSERT_MISSING_RUNNER',
                    'class': 'R2_dead_heat_partner',
                    'race': [d0, ven, rn], 'horse_no': hno,
                    'horse_name': name_key(o['name']),
                    'official_name_raw': o['name'],
                    'official_pos': pos, 'official_finish_type': ftype,
                    'official_odds': o['odds'], 'official_draw': o['draw'],
                    'official_jockey': o['jockey'], 'official_trainer': o['trainer'],
                    'evidence_url': url,
                })
                stats['INSERT_MISSING_RUNNER'] += 1
                changed = True
                continue
            drow = db[hno]
            dbpos = drow['finishing_position']
            if pos is not None and (dbpos is None or int(dbpos) != pos):
                plan.append({
                    'action': 'UPDATE_FINISHING_POSITION',
                    'class': 'R3_missing_results' if dbpos is None else 'R4_position_zero',
                    'race': [d0, ven, rn], 'horse_no': hno, 'entry_id': drow['id'],
                    'official_pos': pos, 'official_finish_type': ftype,
                    'official_odds': o['odds'],
                    'evidence_url': url,
                })
                stats['UPDATE_FINISHING_POSITION'] += 1
                changed = True
            # published odds must match the official cell
            try:
                oo = float(str(o['odds']).replace(',', ''))
            except (TypeError, ValueError):
                oo = None
            try:
                do = float(drow['win_odds'])
            except (TypeError, ValueError):
                do = None
            if oo is not None and do is not None and abs(oo - do) > 1e-9:
                plan.append({
                    'action': 'UPDATE_WIN_ODDS',
                    'class': 'R1_odds_mismatch',
                    'race': [d0, ven, rn], 'horse_no': hno, 'entry_id': drow['id'],
                    'db_value': do, 'official_value': oo,
                    'official_raw_cell': o['odds'],
                    'evidence_url': url,
                })
                stats['UPDATE_WIN_ODDS'] += 1
                changed = True
        # the PLACE dividends the old scraper wrote are a combination
        # string such as '3,6', not the published amounts
        pv = raw.get(key, {})
        pl = pv.get('place') if isinstance(pv, dict) else None
        if pl and len(pl) >= 2:
            amounts = [str(x.get('amount', '')) for x in pl][:3]
            if all(a and a.replace('.', '', 1).isdigit() for a in amounts):
                plan.append({
                    'action': 'UPDATE_PLACE_DIVIDENDS',
                    'class': 'R11_place_dividend_cell_misread',
                    'race': [d0, ven, rn], 'official_place_amounts': amounts,
                    'evidence_url': url,
                })
                stats['UPDATE_PLACE_DIVIDENDS'] += 1
                changed = True
        if not changed:
            stats['ok'] += 1
    conn.close()

    json.dump({'plan': plan, 'stats': stats}, open(a.out, 'w'), indent=1)
    print('\n=== DH / GAP PLAN (official runners table only) ===')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')
    print(f'  total actions: {len(plan)}')
    for p in plan[:10]:
        print(f"   {p['action']:24s} {p['race']} #{p.get('horse_no','-')} "
              f"{p.get('horse_name','')} pos={p.get('official_pos')} "
              f"odds={p.get('official_odds', p.get('official_value', p.get('official_place_amounts')))}")
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
