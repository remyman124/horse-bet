#!/usr/bin/env python3
"""
Classify DB races whose result row exists but which have no runners.

A race that carries a result row but not a single horse entry cannot have
been scraped correctly -- the runners and the results come from the same
page. Two things are possible:

  * the race really exists and the runners are simply missing -> the
    official page carries the runners table and the gap is repaired by
    INSERT_MISSING_RUNNER;
  * the race never existed and the row is an artefact of the old
    scraper -> the official page for that exact (date, venue, race no)
    returns the explicit "No information" page, and the row is removed.

The decision is made ONLY from the official page. Nothing here is
inferred from the stored values.

Usage
-----
    python3 scripts/classify_empty_races.py --out /tmp/empty_races.json
"""

import argparse
import json
import os
import sqlite3
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, 'data', 'horse.db')

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
                gap = 1.3 - (time.time() - _last[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/empty_races.json')
    a = ap.parse_args()

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=180)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute('''
        SELECT rd.race_date AS d0, rd.venue AS ven, r.race_no AS rn, r.id AS rid
        FROM races r JOIN race_days rd ON rd.id = r.race_day_id
        WHERE (SELECT COUNT(*) FROM horse_entries he WHERE he.race_id = r.id) = 0
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) > 0
        ORDER BY rd.race_date, rd.venue, r.race_no'''))
    conn.close()
    print(f'races with a result row but zero runners: {len(rows)}')

    def work(r):
        d0, ven, rn = r['d0'], r['ven'], r['rn']
        url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
               f'?racedate={d0.replace("-", "/")}&Racecourse={ven}&RaceNo={rn}')
        html = get(url)
        if not html:
            return dict(r, status='FETCH_FAIL', url=url)
        has_table = ('Horse No.' in html) or ('Winning Combination' in html)
        if not has_table:
            return dict(r, status='NO_INFORMATION', url=url, bytes=len(html))
        return dict(r, status='HAS_DATA', url=url, bytes=len(html))

    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(work, rows):
            out.append(r)
    counts = {}
    for r in out:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print('official page status:', counts)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}')
    for r in out:
        if r['status'] != 'NO_INFORMATION':
            print(f"   {r['status']:12s} {r['d0']} {r['ven']} R{r['rn']}")


if __name__ == '__main__':
    main()
