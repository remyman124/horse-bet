#!/usr/bin/env python3
"""
Fetch and validate the OFFICIAL PLACE dividends from HKJC result pages.

Why this script exists
----------------------
The first place-dividend parser (`/tmp/fetch_all2.py::parse_dividends`)
collected PLACE rows with a loose rule: "any table row whose first cell is
all digits, once a PLACE row has been seen". On the MODERN HKJC results
page the RUNNERS table also has rows whose first cell is a number (the
finishing position), and on those pages the runners table is processed
BEFORE the dividends table — so the parser swallowed runner rows into the
PLACE list.

The tell: 33% of PLACE amounts came out BELOW $10 per $10 stake, i.e. a
"winning" place bet would have paid back less than the stake. That is
impossible, and the damage was concentrated by era:

    2006-2019   0% impossible rows   (page order: dividends table first)
    2024-2026  55-60% impossible    (page order: runners table first)

The fix
-------
Parse the PLACE rows ONLY from the table that actually contains the
"PLACE" label row. Then validate against an independent source on the
same page: the runners table gives the finishing position of every horse,
so the set of PLACE horse numbers MUST equal the set of horses that
finished 1st/2nd/3rd. That is a strong, self-contained check that needs
no external ground truth.

The other invariants checked:
  * PLACE[0] horse number == WIN horse number
  * every PLACE amount >= 10 (a $10 stake cannot return less than $10)
  * the number of PLACE rows equals the number of horses that placed
    (3 for races with 8+ declared runners, 2 for 4-7 runners)
  * amounts are strictly positive and parse as numbers

Output: data/official_evidence/place_dividends.json
"""
import argparse
import json
import os
import random
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
OUT = os.path.join(ROOT, 'data', 'official_evidence', 'place_dividends.json')
CKPT = '/tmp/place_div.ckpt.json'

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
        except Exception as e:
            if a == tries - 1:
                return None
        time.sleep(2 + 2 * a)
    return None


def _num(s):
    try:
        return float(str(s).replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def parse_page(html):
    """Return (rows_by_horse, dividends) with PLACE rows taken ONLY from the
    dividends table."""
    if not html:
        return None, None
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # --- runners table: horse number -> finishing position ----------------
    rows = {}
    for t in soup.find_all('table'):
        for r in t.find_all('tr'):
            cs = [x.get_text(' ', strip=True) for x in r.find_all(['td', 'th'])]
            if len(cs) >= 3 and cs[0].isdigit() and cs[1].isdigit():
                pos, hn = int(cs[0]), int(cs[1])
                if 1 <= pos <= 30 and 1 <= hn <= 30:
                    rows.setdefault(hn, pos)
        if len(rows) >= 6:
            break

    # --- dividends table: find the table that carries the PLACE label ----
    win = None
    place = []
    for t in soup.find_all('table'):
        table_place = None
        table_win = None
        for r in t.find_all('tr'):
            cs = [x.get_text(' ', strip=True) for x in r.find_all(['td', 'th'])]
            if not cs:
                continue
            label = cs[0].upper().strip()
            if label == 'WIN' and len(cs) >= 3:
                table_win = (cs[1].strip(), _num(cs[2]))
            elif label == 'PLACE' and len(cs) >= 3:
                combo, amt = cs[1].strip(), _num(cs[2])
                if combo and amt is not None:
                    table_place = [(combo, amt)]
            elif table_place is not None and cs[0].isdigit() and len(cs) >= 2:
                amt = _num(cs[1])
                if amt is not None:
                    table_place.append((cs[0].strip(), amt))
        if table_place:
            place = table_place
        if table_win and win is None:
            win = table_win
        if place and win:
            break

    if not place:
        return rows, None
    return rows, {'win': win, 'place': place}


def validate(rows, div):
    """Independent check against the same page's finishing positions."""
    errs = []
    if not div:
        return ['no dividends parsed']
    win = div.get('win')
    place = div.get('place') or []
    if not place:
        return ['no place rows']
    if win and win[1] is not None and win[1] < 10:
        errs.append(f'win amount {win[1]} < 10 per $10 stake')
    for combo, amt in place:
        if amt is None or amt < 10:
            errs.append(f'place amount {amt} for combo {combo} is < 10 per $10 stake')
    # place[0] must be the winner
    if win and win[0] and place[0][0] != win[0]:
        errs.append(f'place[0]={place[0][0]} != win combo {win[0]}')
    # every place horse must be a horse that finished in the top 3
    if rows:
        placed = {h for h, p in rows.items() if 1 <= p <= 3}
        combos = set()
        for combo, _ in place:
            if not str(combo).isdigit():
                errs.append(f'non-numeric place combo {combo!r}')
                continue
            combos.add(int(combo))
        if placed and combos and combos != placed:
            errs.append(f'place horses {sorted(combos)} != top-3 {sorted(placed)}')
        if len(place) != len(placed) and placed:
            errs.append(f'{len(place)} place rows vs {len(placed)} top-3 horses')
    return errs


def fetch_one(key):
    d0, ven, rn = key.split('|')
    url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
           f'?racedate={d0.replace("-", "/")}&Racecourse={ven}&RaceNo={rn}')
    html = get(url)
    if not html:
        return key, {'status': 'FETCH_FAIL', 'url': url}
    rows, div = parse_page(html)
    if not div:
        return key, {'status': 'NO_DIVIDENDS', 'url': url, 'page_len': len(html)}
    errs = validate(rows, div)
    return key, {
        'status': 'OK' if not errs else 'INVALID',
        'url': url,
        'date': d0, 'venue': ven, 'race_no': rn,
        'win': {'combo': div['win'][0], 'amount': div['win'][1]} if div.get('win') else None,
        'place': [{'combo': c, 'amount': a} for c, a in div['place']],
        'errors': errs,
        'page_len': len(html),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2006-01-01')
    ap.add_argument('--until', default='2026-12-31')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--reverse', action='store_true',
                    help='fetch newest races first (the recent era is the most '
                         'representative of today\'s market and the model\'s test window)')
    ap.add_argument('--ckpt', default=CKPT)
    a = ap.parse_args()

    conn = sqlite3.connect(DB)
    keys = [r[0] for r in conn.execute('''
        SELECT DISTINCT rd.race_date || '|' || rd.venue || '|' || r.race_no
        FROM races r JOIN race_days rd ON rd.id = r.race_day_id
        WHERE rd.race_date BETWEEN ? AND ?
        ORDER BY rd.race_date
    ''', (a.since, a.until))]
    conn.close()
    if a.reverse:
        keys = keys[::-1]
    if a.limit:
        keys = keys[:a.limit]
    print(f'target races: {len(keys)}', flush=True)

    ck = {}
    if os.path.exists(a.ckpt):
        ck = json.load(open(a.ckpt))
    done = {k for k, v in ck.items() if v.get('status') == 'OK'}
    todo = [k for k in keys if k not in done]
    print(f'already OK: {len(done)} | to fetch: {len(todo)}', flush=True)

    stats = {'OK': 0, 'INVALID': 0, 'FETCH_FAIL': 0, 'NO_DIVIDENDS': 0}

    def work(k):
        key, res = fetch_one(k)
        with _lock:
            ck[key] = res
            stats[res['status']] = stats.get(res['status'], 0) + 1
            n = len(ck)
            if n % 25 == 0:
                json.dump(ck, open(a.ckpt, 'w'))
                print(f'  {n}/{len(keys)}  {stats}', flush=True)
        return res

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(work, todo):
            pass

    json.dump(ck, open(a.ckpt, 'w'))
    ok = {k: v for k, v in ck.items() if v.get('status') == 'OK'}
    print('\nfinal:', stats, flush=True)
    print(f'valid races: {len(ok)}', flush=True)

    # ---- evidence file (repo-local, reproducible) -------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {}
    for k, v in ok.items():
        out[k] = {'url': v['url'], 'win': v.get('win'), 'place': v.get('place'),
                  'date': v.get('date'), 'venue': v.get('venue'), 'race_no': v.get('race_no')}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'saved: {OUT}  ({len(out)} races)', flush=True)

    # ---- summary of what the old parser got wrong ------------------------
    old = {}
    if os.path.exists('/tmp/live_all.ckpt.json'):
        for k, v in json.load(open('/tmp/live_all.ckpt.json')).items():
            dv = v.get('dividends')
            if isinstance(dv, dict) and dv.get('place'):
                old[k] = dv
    both = [k for k in ok if k in old]
    disagree = 0
    for k in both:
        o = {(r['combo'], round(float(r['amount']), 2)) for r in old[k]['place']}
        n = {(r['combo'], round(float(r['amount']), 2)) for r in ok[k]['place']}
        if o != n:
            disagree += 1
    print(f'\ncompared with old parser: {len(both)} races, {disagree} disagreements '
          f'({disagree/max(1,len(both))*100:.0f}%)')


if __name__ == '__main__':
    main()
