#!/usr/bin/env python3
"""
Backfill HKJC race results for 2026-05-07 to 2026-10-15.
Closes the 5-month data gap in horse.db.
Uses proven 3-phase pattern: probe -> parallel scrape -> sequential DB save.
"""
import sys, asyncio, re, html, time
from datetime import date, timedelta
from pathlib import Path
import sqlite3
sqlite3.register_adapter(date, lambda d: d.isoformat())

import httpx
from bs4 import BeautifulSoup

MAX_CONCURRENT = 10  # Conservative for HKJC rate limiting
REQUEST_DELAY = 0.4
PROBE_RACES = list(range(1, 13))
LOG_FILE = Path('/tmp/backfill_2026_progress.log')
RESULTS_FILE = Path('/tmp/backfill_2026_results.json')

START_DATE = date(2026, 5, 7)
END_DATE = date(2026, 10, 15)
DB_PATH = '/home/work/horse-bet/data/horse.db'


def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def generate_days_in_range(start: date, end: date):
    """Generate candidate race days in range (Wed=HV, Sat=ST, Sun=ST+HV)."""
    days = []
    d = start
    while d <= end:
        if d.weekday() == 2:  # Wed
            days.append((d, 'HV'))
        elif d.weekday() == 5:  # Sat
            days.append((d, 'ST'))
        elif d.weekday() == 6:  # Sun
            days.append((d, 'ST'))
            days.append((d, 'HV'))
        d += timedelta(days=1)
    return days


def parse_race_page(txt: str):
    """Parse localresults HTML page for results + dividends + race info."""
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return None

    result_table = tables[2]
    div_table = tables[3]

    header_rows = result_table.find_all('tr')
    if not header_rows:
        return None
    first_cells = header_rows[0].find_all(['td', 'th'])
    if not any(c.get_text(strip=True) == 'Pla.' for c in first_cells):
        return None

    # Dividends
    dividends = {}
    for row in div_table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 3:
            key = cells[0].get_text(strip=True)
            if key in ('WIN', 'PLACE'):
                horse_no = cells[1].get_text(strip=True)
                amount = cells[2].get_text(strip=True)
                dividends[key.lower()] = f"{horse_no}\t{amount}"

    # Horses
    horses = []
    rows = result_table.find_all('tr')
    for i in range(1, len(rows)):
        cells = rows[i].find_all(['td', 'th'])
        if not cells:
            continue
        cell_texts = [c.get_text() for c in cells]
        row_str = '\t'.join(cell_texts)
        parts = row_str.split('\t')
        if len(parts) < 8:
            continue

        pos = parts[0].strip()
        if not (pos.isdigit() or pos in ('DH', 'WV', 'DSQ', 'TEMP', 'PU', 'UR', 'FE', 'AB', 'NP')):
            continue

        try:
            horse_no = int(parts[1].strip()) if parts[1].strip().isdigit() else None
            raw_name = parts[2].strip()
            horse_name = html.unescape(raw_name)
            horse_name = re.sub(r'\s*\([^)]+\)\s*$', '', horse_name).strip()
            horse_name = re.sub(r'\s+', ' ', horse_name)
            jockey  = parts[3].strip()
            trainer = parts[4].strip()
            draw    = int(parts[7].strip()) if len(parts) > 7 and parts[7].strip().isdigit() else None

            finish_time, odds = '', None
            for j in range(10, len(parts)):
                p = parts[j].strip()
                if re.match(r'^\d+:\d+\.\d+$', p):
                    finish_time = p
                elif re.match(r'^\d+(?:\.\d+)?$', p) and p and float(p) > 1:
                    odds = float(p)

            position = int(pos) if pos.isdigit() else pos
            horses.append({
                'horse_no': horse_no, 'horse_name': horse_name, 'position': position,
                'jockey': jockey, 'trainer': trainer, 'draw': draw,
                'finish_time': finish_time, 'win_odds': odds,
            })
        except (ValueError, IndexError):
            pass

    # Race metadata
    race_info = {'race_class': None, 'distance': None, 'track': 'Turf'}
    full_text = soup.get_text()
    cm = re.search(r'Class\s+(\d+)', full_text)
    dm = re.search(r'(\d{4})[Mm](?:\s|$)', full_text)
    gm = re.search(r'Group\s*(One|Two|Three)', full_text, re.I)
    if gm:
        race_info['race_class'] = f"Group-{gm.group(1).title()}"
    elif cm:
        race_info['race_class'] = f"Class {cm.group(1)}"
    if dm:
        race_info['distance'] = dm.group(1)
    if 'AWT' in full_text:
        race_info['track'] = 'AWT'

    return {'horses': horses, 'dividends': dividends, 'race_info': race_info}


async def probe_day(client, d, venue):
    """Probe R1..R12. Returns (d, venue, race_nos) or None."""
    date_str = d.strftime('%Y/%m/%d')
    race_nos = []
    for rn in PROBE_RACES:
        url = f'https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={rn}'
        try:
            resp = await client.get(url, timeout=12.0)
            if resp.status_code != 200:
                break
            parsed = parse_race_page(resp.text)
            if parsed and parsed['horses'] and parsed.get('dividends', {}).get('win'):
                race_nos.append(rn)
            else:
                break
        except Exception:
            break
    return (d, venue, race_nos) if race_nos else None


async def scrape_race(client, d, venue, race_no):
    date_str = d.strftime('%Y/%m/%d')
    url = f'https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}'
    try:
        resp = await client.get(url, timeout=15.0)
        if resp.status_code != 200:
            return None
        parsed = parse_race_page(resp.text)
        if not parsed or not parsed['horses']:
            return None
        return {
            'race_date': date_str, 'venue': venue, 'race_no': race_no,
            'race_info': parsed['race_info'], 'horses': parsed['horses'],
            'dividends': parsed['dividends'],
        }
    except Exception:
        return None


def save_to_db(race_data, conn):
    cur = conn.cursor()
    dp = race_data['race_date'].split('/')
    py_date = date(int(dp[0]), int(dp[1]), int(dp[2]))
    venue, race_no = race_data['venue'], race_data['race_no']
    info = race_data.get('race_info', {})
    dist_str = info.get('distance') or ''
    dist = int(re.search(r'\d+', dist_str).group()) if re.search(r'\d+', dist_str) else None

    cur.execute("""
        INSERT INTO race_days (race_date, venue, total_races)
        VALUES (?, ?, 0)
        ON CONFLICT(race_date, venue) DO UPDATE SET total_races=excluded.total_races
    """, (py_date, venue))
    rd_id = cur.execute(
        "SELECT id FROM race_days WHERE race_date=? AND venue=?",
        (py_date, venue)
    ).fetchone()[0]

    cur.execute("""
        INSERT INTO races (race_day_id, race_no, distance, race_class, track)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(race_day_id, race_no) DO UPDATE SET
            distance=excluded.distance, race_class=excluded.race_class,
            track=COALESCE(excluded.track, track)
    """, (rd_id, race_no, dist, info.get('race_class') or '', info.get('track', 'Turf')))
    r_id = cur.execute(
        "SELECT id FROM races WHERE race_day_id=? AND race_no=?",
        (rd_id, race_no)
    ).fetchone()[0]

    for h in race_data.get('horses', []):
        if not h.get('horse_no'):
            continue
        # Use COALESCE to preserve pre-existing data (e.g., from racecard scrape)
        cur.execute("""
            INSERT INTO horse_entries
            (race_id, horse_no, horse_name, draw, jockey, trainer, win_odds, finishing_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id, horse_no) DO UPDATE SET
                draw=COALESCE(excluded.draw, horse_entries.draw),
                horse_name=excluded.horse_name,
                jockey=COALESCE(excluded.jockey, horse_entries.jockey),
                trainer=COALESCE(excluded.trainer, horse_entries.trainer),
                win_odds=COALESCE(excluded.win_odds, horse_entries.win_odds),
                finishing_position=COALESCE(excluded.finishing_position, horse_entries.finishing_position)
        """, (r_id, h['horse_no'], h['horse_name'], h.get('draw'),
              h.get('jockey', ''), h.get('trainer', ''),
              h.get('win_odds'), h.get('position')))

    divs = race_data.get('dividends', {})
    if divs:
        win_v = place_v = None
        for p in (divs.get('win') or '').split('\t'):
            p = p.strip()
            if re.match(r'^\d+(?:\.\d+)?$', p):
                win_v = float(p)
                break
        for p in (divs.get('place') or '').split('\t'):
            p = p.strip()
            if re.match(r'^\d+(?:\.\d+)?$', p):
                place_v = float(p)
                break
        cur.execute("""
            INSERT INTO race_results (race_id, win_dividend, place_dividend_1)
            VALUES (?, ?, ?)
            ON CONFLICT(race_id) DO UPDATE SET
                win_dividend=excluded.win_dividend, place_dividend_1=excluded.place_dividend_1
        """, (r_id, win_v, place_v))

    conn.commit()
    return len(race_data.get('horses', []))


async def main():
    LOG_FILE.write_text('')
    log(f"Backfill {START_DATE} -> {END_DATE}")

    candidate_days = generate_days_in_range(START_DATE, END_DATE)
    log(f"Candidate days: {len(candidate_days)}")

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    conn = sqlite3.connect(DB_PATH)

    # Phase 1: Probe
    log(f"Phase 1: Probing race days...")
    t0 = time.time()
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=MAX_CONCURRENT)) as client:
        tasks = [probe_day(client, d, v) for d, v in candidate_days]
        results = []
        for i in range(0, len(tasks), 20):
            chunk = tasks[i:i+20]
            r = await asyncio.gather(*chunk)
            results.extend(r)
            log(f"  probed {min(i+20, len(tasks))}/{len(tasks)}")

    race_days_found = [r for r in results if r is not None]
    total_races = sum(len(r[2]) for r in race_days_found)
    log(f"Phase 1 done: {len(race_days_found)} race days, {total_races} races in {time.time()-t0:.0f}s")
    for d, v, rnos in race_days_found:
        log(f"  FOUND  {d} {v}: R{rnos[0]}-{rnos[-1]} ({len(rnos)} races)")

    # Phase 2: Scrape
    log(f"Phase 2: Scraping {total_races} races...")
    t1 = time.time()
    races_to_scrape = [(d, v, rn) for d, v, rnos in race_days_found for rn in rnos]
    all_results = []
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=MAX_CONCURRENT)) as client:
        for i in range(0, len(races_to_scrape), 30):
            chunk = races_to_scrape[i:i+30]
            tasks = [scrape_race(client, d, v, rn) for d, v, rn in chunk]
            r = await asyncio.gather(*tasks)
            all_results.extend(r)
            await asyncio.sleep(REQUEST_DELAY)
            log(f"  fetched {min(i+30, len(races_to_scrape))}/{len(races_to_scrape)}")

    log(f"Phase 2 done in {time.time()-t1:.0f}s")

    # Phase 3: Save
    log("Phase 3: Saving to DB...")
    saved = skipped = 0
    for rd in all_results:
        if rd and rd.get('horses') and rd.get('dividends', {}).get('win'):
            try:
                n = save_to_db(rd, conn)
                saved += 1
                if saved % 20 == 0:
                    log(f"  saved {saved} races so far")
            except Exception as e:
                log(f"  ERROR {rd.get('race_date')} {rd.get('venue')} R{rd.get('race_no')}: {e}")
                skipped += 1
        else:
            skipped += 1

    log(f"DONE: {saved} saved, {skipped} skipped")
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
