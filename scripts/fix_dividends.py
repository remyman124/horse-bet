#!/usr/bin/env python3
"""
FIX: win_dividend / place_dividend stored the WINNER'S HORSE NUMBER instead of
the actual dividend amount.

Root cause (all scrapers 2006-2026):
    dividends['win'] = f"{horse_no}\t{amount}"     # e.g. "6\t51.00"
    ...
    for p in (divs.get('win') or '').split('\t'):
        if re.match(r'^\d+(?:\.\d+)?$', p): win_v = float(p); break
        # ^ takes the FIRST number = horse_no, never the amount

Also fixes: win_odds not captured for races where the odds cell sits in a
different column position, and the race_info class regex misses Group races
on some pages.

Strategy: re-scrape the dividends table for every race we already have, using
the CORRECT column (index 2 = Dividend (HK$)), and update race_results.
No need to re-scrape the horse table.

Usage:
    python3 scripts/fix_dividends.py [--year 2026] [--all]
"""
import sys, re, asyncio, time, argparse
from datetime import date
from pathlib import Path
import sqlite3

import httpx
from bs4 import BeautifulSoup

DB_PATH = '/home/work/horse-bet/data/horse.db'
LOG_FILE = Path('/tmp/fix_dividends.log')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def parse_dividends(txt):
    """Return (win_dividend, place_dividends[list]) using column index 2 = amount."""
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return None
    div_table = tables[3]

    win_amt = None
    place_amts = []
    for row in div_table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        key = cells[0].get_text(strip=True)
        combo = cells[1].get_text(strip=True)
        amount_raw = cells[2].get_text(strip=True).replace(',', '').replace('$', '').strip()
        if not re.match(r'^\d+(?:\.\d+)?$', amount_raw):
            continue
        amount = float(amount_raw)
        if key == 'WIN':
            win_amt = amount
        elif key == 'PLACE':
            place_amts.append(amount)
        elif key == '' and re.match(r'^\d+$', combo) and place_amts:
            # continuation rows of the PLACE pool: 2nd/3rd place dividends
            place_amts.append(amount)
    return win_amt, place_amts


async def fetch_dividends(client, date_str, venue, race_no):
    url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
           f'?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}')
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code != 200 or 'No information' in resp.text:
            return None
        return parse_dividends(resp.text)
    except Exception:
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, help='only fix this year')
    ap.add_argument('--all', action='store_true', help='fix every race in DB')
    args = ap.parse_args()

    LOG_FILE.write_text('')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    where = ''
    params = []
    if args.year:
        where = "WHERE strftime('%Y', rd.race_date) = ?"
        params = [str(args.year)]
    elif not args.all:
        where = ("WHERE rd.race_date >= '2026-01-01'")
        print("No --year/--all given -> defaulting to 2026 only. Use --all for full history.")

    cur.execute(f"""
        SELECT rd.race_date, rd.venue, r.race_no, rr.id
        FROM races r
        JOIN race_days rd ON r.race_day_id = rd.id
        JOIN race_results rr ON rr.race_id = r.id
        {where}
        ORDER BY rd.race_date, r.race_no
    """, params)
    targets = cur.fetchall()
    log(f"Target races: {len(targets)}")
    if not targets:
        return

    fixed = skipped = failed = 0
    t0 = time.time()
    async with httpx.AsyncClient(headers=HEADERS, limits=httpx.Limits(max_connections=6)) as client:
        for i, (rdate, venue, race_no, rr_id) in enumerate(targets):
            # rdate is a datetime.date -> str() gives ISO 'YYYY-MM-DD'
            ds = (rdate if isinstance(rdate, str) else rdate.isoformat()).replace('-', '/')
            res = await fetch_dividends(client, ds, venue, race_no)
            if res is None:
                # "No information" -> race does not exist on HKJC (phantom race in DB)
                skipped += 1
            else:
                win_amt, place_amts = res
                if win_amt is None:
                    failed += 1
                else:
                    p1 = place_amts[0] if place_amts else None
                    p2 = place_amts[1] if len(place_amts) > 1 else None
                    p3 = place_amts[2] if len(place_amts) > 2 else None
                    cur.execute("""
                        UPDATE race_results SET
                          win_dividend = ?,
                          place_dividend_1 = COALESCE(?, place_dividend_1),
                          place_dividend_2 = COALESCE(?, place_dividend_2),
                          place_dividend_3 = COALESCE(?, place_dividend_3)
                        WHERE id = ?
                    """, (win_amt, p1, p2, p3, rr_id))
                    conn.commit()
                    fixed += 1
            if (i + 1) % 25 == 0:
                log(f"  {i+1}/{len(targets)} fixed={fixed} skipped={skipped} "
                    f"failed={failed} ({(i+1)/(time.time()-t0):.1f}/s)")
            await asyncio.sleep(0.35)

    log(f"DONE: fixed={fixed} skipped={skipped} failed={failed} "
        f"elapsed={time.time()-t0:.0f}s")
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
