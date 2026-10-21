#!/usr/bin/env python3
"""
CORRECTED results parser + backfill for win_odds and dividends.

Two root-cause bugs fixed here:

BUG 1 — win_odds grabbed the WRONG COLUMN for older races.
  The old parser searched parts[10:] for the first bare number > 1.
  HKJC results tables have TWO layouts:
    - 12 cols: Pla.|Horse No.|Horse|Jockey|Trainer|Act.Wt.|Declar.Wt.|Dr.|LBW|
               RunningPosition|Finish Time|Win Odds   (index 11 = Win Odds)
    - 11 cols: Pla.|Horse No.|Horse|Jockey|Trainer|Act.Wt.|Declar.Wt.|Dr.|LBW|
               Finish Time|Win Odds                          (index 10 = Win Odds)
  On the 11-col layout parts[10] is 'Win Odds' (correct), but the loop also
  matched the "Declar. Horse Wt." / margin values and the LAST match won,
  so odds ended up being the place dividend of the winner, not the odds.
  => old data had WINNERS WITH HIGH 'odds' and FAVOURITES WITH LOW 'odds'
     (inverted), which is why implied_prob had NEGATIVE correlation with wins.

BUG 2 — win_dividend stored the WINNER'S HORSE NUMBER.
  dividends['win'] = f"{horse_no}\t{amount}" then the save code did
  split('\t') and took the FIRST number = horse_no.
  Correct value: amount = odds * 10 (HKJC $10 unit).

This script re-scrapes every race in the DB using header-driven column
lookup and repairs both fields.

Usage:
    python3 scripts/fix_odds_and_dividends.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                                               [--dry-run] [--limit N]
"""
import sys, re, asyncio, time, argparse
from pathlib import Path
import sqlite3

import httpx
from bs4 import BeautifulSoup

DB_PATH = '/home/work/horse-bet/data/horse.db'
LOG_FILE = Path('/tmp/fix_odds_div.log')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
UNIT = 10.0  # HKJC standard stake


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def _num(s):
    s = (s or '').strip().replace(',', '').replace('$', '')
    return float(s) if re.match(r'^\d+(?:\.\d+)?$', s) else None


def parse_results_page(txt):
    """
    Header-driven parse. Returns:
      horses: {horse_no: {pos, name, jockey, trainer, draw, odds, act_wt}}
      win_dividend, place_dividends[list]
    """
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 3:
        return None

    # --- horses table (find the one with a 'Pla.' header) ---
    res_table = None
    for t in tables:
        rows = t.find_all('tr')
        if not rows:
            continue
        hdr = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
        if 'Pla.' in hdr and 'Horse No.' in hdr:
            res_table = t
            hdr_map = hdr
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
            odds = None
        name = cells[i_name].get_text(strip=True) if len(cells) > i_name else ''
        name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
        name = re.sub(r'\s+', ' ', name)
        horses[int(hn)] = {
            'pos': int(pos_raw) if pos_raw.isdigit() else pos_raw,
            'name': name,
            'jockey': cells[i_jk].get_text(strip=True) if len(cells) > i_jk else '',
            'trainer': cells[i_tr].get_text(strip=True) if len(cells) > i_tr else '',
            'draw': int(cells[i_draw].get_text(strip=True)) if len(cells) > i_draw
                    and cells[i_draw].get_text(strip=True).isdigit() else None,
            'act_wt': int(cells[i_awt].get_text(strip=True)) if len(cells) > i_awt
                      and cells[i_awt].get_text(strip=True).isdigit() else None,
            'odds': odds,
        }

    if not horses:
        return None

    # --- dividends table (find the one with a 'Dividend' or WIN row) ---
    # WIN row:  ['WIN',  '<winner horse_no>', '<amount>']
    # PLACE rows: ['PLACE','<horse_no>','<amount>'] then two continuation rows
    #             with an empty pool label for 2nd and 3rd.
    # NOTE: on older race cards the "Win Odds" column is TRUNCATED to an integer
    # (e.g. displayed 14 for a true 14.85) while the dividend keeps the exact
    # value (148.50). The dividend is authoritative, so we derive precise odds
    # from it: odds = dividend / 10 (HKJC $10 unit).
    win_div = None
    win_horse_no = None
    place_divs = []          # list of (horse_no, amount)
    for t in tables:
        found_win = False
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
                found_win = True
            elif key == 'PLACE':
                if combo.isdigit():
                    place_divs.append((int(combo), amount))
                found_win = True
            elif key == '' and place_divs:
                if combo.isdigit():
                    place_divs.append((int(combo), amount))
        if found_win:
            break

    return {'horses': horses, 'win_dividend': win_div, 'win_horse_no': win_horse_no,
            'place_dividend_list': place_divs,
            'place_dividends': [a for _, a in place_divs]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='date_from', default=None)
    ap.add_argument('--to', dest='date_to', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    LOG_FILE.write_text('')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    where, params = '', []
    if args.date_from:
        where = 'AND rd.race_date >= ?'
        params.append(args.date_from)
    if args.date_to:
        where += (' AND ' if where else 'WHERE ') + 'rd.race_date <= ?'
        params.append(args.date_to)

    cur.execute(f"""
        SELECT rd.race_date, rd.venue, r.race_no, r.id
        FROM races r
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE EXISTS (SELECT 1 FROM horse_entries he WHERE he.race_id = r.id)
        {where}
        ORDER BY rd.race_date, r.race_no
    """, params)
    targets = cur.fetchall()
    if args.limit:
        targets = targets[:args.limit]
    log(f"Target races: {len(targets)}  dry_run={args.dry_run}")

    fixed_odds = fixed_div = ok = skipped = failed = 0
    t0 = time.time()

    async with httpx.AsyncClient(headers=HEADERS, limits=httpx.Limits(max_connections=6),
                                 follow_redirects=True) as client:
        for i, (rdate, venue, race_no, r_id) in enumerate(targets):
            ds = (rdate if isinstance(rdate, str) else rdate.isoformat()).replace('-', '/')
            url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
                   f'?racedate={ds}&Racecourse={venue}&RaceNo={race_no}')
            try:
                resp = await client.get(url, timeout=25.0)
                if resp.status_code != 200 or 'No information' in resp.text:
                    skipped += 1
                else:
                    parsed = parse_results_page(resp.text)
                    if not parsed:
                        failed += 1
                    else:
                        ok += 1
                        if not args.dry_run:
                            for hno, h in parsed['horses'].items():
                                if h['odds'] is None:
                                    continue
                                cur.execute(
                                    'UPDATE horse_entries SET win_odds=? WHERE race_id=? AND horse_no=?',
                                    (h['odds'], r_id, hno))
                                fixed_odds += cur.rowcount if cur.rowcount > 0 else 0
                            win_div = parsed['win_dividend']
                            if win_div is None:
                                # derive from winner odds * 10
                                winners = [h for h in parsed['horses'].values() if h['pos'] == 1]
                                if winners and winners[0]['odds']:
                                    win_div = winners[0]['odds'] * UNIT
                            if win_div:
                                pd = parsed['place_dividends']
                                cur.execute("""
                                    UPDATE race_results SET
                                      win_dividend=?,
                                      place_dividend_1=COALESCE(?, place_dividend_1),
                                      place_dividend_2=COALESCE(?, place_dividend_2),
                                      place_dividend_3=COALESCE(?, place_dividend_3)
                                    WHERE race_id=?
                                """, (win_div, pd[0] if len(pd) > 0 else None,
                                      pd[1] if len(pd) > 1 else None,
                                      pd[2] if len(pd) > 2 else None, r_id))
                                fixed_div += 1
                            conn.commit()
            except Exception as e:
                failed += 1
            if (i + 1) % 50 == 0:
                log(f"  {i+1}/{len(targets)} ok={ok} skipped={skipped} failed={failed} "
                    f"odds_rows={fixed_odds} div_races={fixed_div} "
                    f"({(i+1)/(time.time()-t0):.1f}/s)")
            await asyncio.sleep(0.4)

    log(f"DONE: ok={ok} skipped={skipped} failed={failed} "
        f"odds_rows_fixed={fixed_odds} div_races_fixed={fixed_div} "
        f"elapsed={time.time()-t0:.0f}s")
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
