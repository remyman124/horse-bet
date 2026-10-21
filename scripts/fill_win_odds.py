#!/usr/bin/env python3
"""
Fill in win_odds for 2026 races where the scraper missed it.

Context: eval_1019.py and backfill_2026_slow.py used a simplified parser that
did not extract the "Win Odds" column from the HKJC results table. The results
page header is:
  Pla. | Horse No. | Horse | Jockey | Trainer | Act. Wt. | Declar. Horse Wt.
  | Dr. | LBW | RunningPosition | Finish Time | Win Odds
So Win Odds is the LAST cell (index 11) when present.

Also used to backfill odds for any race where win_odds is NULL but the race
exists on HKJC.

Usage: python3 scripts/fill_win_odds.py [--year 2026]
"""
import sys, re, asyncio, time, argparse
from datetime import date
from pathlib import Path
import sqlite3

import httpx
from bs4 import BeautifulSoup

DB_PATH = '/home/work/horse-bet/data/horse.db'
LOG_FILE = Path('/tmp/fill_win_odds.log')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def parse_odds(txt):
    """Return {horse_no: win_odds} plus {horse_no: (jockey, draw, act_wt)}."""
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 3:
        return None, None
    t = tables[2]
    rows = t.find_all('tr')
    if not rows:
        return None, None
    hdr = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
    if 'Win Odds' not in hdr:
        return None, None
    i_odds = hdr.index('Win Odds')
    i_hno = hdr.index('Horse No.') if 'Horse No.' in hdr else 1
    i_jk = hdr.index('Jockey') if 'Jockey' in hdr else 3
    i_tr = hdr.index('Trainer') if 'Trainer' in hdr else 4
    i_draw = hdr.index('Dr.') if 'Dr.' in hdr else 7
    i_awt = hdr.index('Act. Wt.') if 'Act. Wt.' in hdr else 5

    odds = {}
    extra = {}
    for row in rows[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) <= i_odds:
            continue
        hn = cells[i_hno].get_text(strip=True)
        if not hn.isdigit():
            continue
        ov = cells[i_odds].get_text(strip=True)
        if not re.match(r'^\d+(?:\.\d+)?$', ov):
            continue
        odds[int(hn)] = float(ov)
        extra[int(hn)] = {
            'jockey': cells[i_jk].get_text(strip=True) if len(cells) > i_jk else '',
            'trainer': cells[i_tr].get_text(strip=True) if len(cells) > i_tr else '',
            'draw': int(cells[i_draw].get_text(strip=True)) if len(cells) > i_draw
                    and cells[i_draw].get_text(strip=True).isdigit() else None,
            'act_wt': int(cells[i_awt].get_text(strip=True)) if len(cells) > i_awt
                      and cells[i_awt].get_text(strip=True).isdigit() else None,
        }
    return odds, extra


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2026)
    args = ap.parse_args()

    LOG_FILE.write_text('')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT rd.race_date, rd.venue, r.race_no, r.id
        FROM races r
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE strftime('%Y', rd.race_date) = ?
          AND EXISTS (SELECT 1 FROM horse_entries he WHERE he.race_id = r.id)
        ORDER BY rd.race_date, r.race_no
    """, (str(args.year),))
    targets = cur.fetchall()
    log(f"Target races ({args.year}): {len(targets)}")

    filled = 0
    t0 = time.time()
    async with httpx.AsyncClient(headers=HEADERS, limits=httpx.Limits(max_connections=6)) as client:
        for i, (rdate, venue, race_no, r_id) in enumerate(targets):
            ds = (rdate if isinstance(rdate, str) else rdate.isoformat()).replace('-', '/')
            url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
                   f'?racedate={ds}&Racecourse={venue}&RaceNo={race_no}')
            try:
                resp = await client.get(url, timeout=20.0)
                if resp.status_code != 200 or 'No information' in resp.text:
                    await asyncio.sleep(0.35)
                    continue
                odds, extra = parse_odds(resp.text)
                if not odds:
                    await asyncio.sleep(0.35)
                    continue
                for hno, ov in odds.items():
                    cur.execute(
                        "UPDATE horse_entries SET win_odds=COALESCE(win_odds, ?) WHERE race_id=? AND horse_no=?",
                        (ov, r_id, hno))
                    filled += cur.rowcount if cur.rowcount > 0 else 0
                conn.commit()
            except Exception:
                pass
            if (i + 1) % 25 == 0:
                log(f"  {i+1}/{len(targets)} filled_rows={filled} "
                    f"({(i+1)/(time.time()-t0):.1f}/s)")
            await asyncio.sleep(0.35)

    log(f"DONE: filled={filled} elapsed={time.time()-t0:.0f}s")
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
