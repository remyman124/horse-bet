#!/usr/bin/env python3
"""
Conservative backfill for May-Aug 2026.
HKJC rate-limits aggressively so we go slow and serial for missed dates.
"""
import sys, asyncio, re, html, time
from datetime import date, timedelta
from pathlib import Path
import sqlite3
sqlite3.register_adapter(date, lambda d: d.isoformat())

import httpx
from bs4 import BeautifulSoup

# Conservative: serial-ish
MAX_CONCURRENT = 3
REQUEST_DELAY = 1.5
PROBE_RACES = list(range(1, 13))
LOG_FILE = Path('/tmp/backfill_2026b_progress.log')
DB_PATH = '/home/work/horse-bet/data/horse.db'

START_DATE = date(2026, 5, 7)
END_DATE = date(2026, 8, 31)


def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def generate_days_in_range(start, end):
    days = []
    d = start
    while d <= end:
        if d.weekday() == 2:
            days.append((d, 'HV'))
        elif d.weekday() == 5:
            days.append((d, 'ST'))
        elif d.weekday() == 6:
            days.append((d, 'ST'))
            days.append((d, 'HV'))
        d += timedelta(days=1)
    return days


def parse_race_page(txt):
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

    dividends = {}
    for row in div_table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 3:
            key = cells[0].get_text(strip=True)
            if key in ('WIN', 'PLACE'):
                dividends[key.lower()] = f"{cells[1].get_text(strip=True)}\t{cells[2].get_text(strip=True)}"

    horses = []
    for row in result_table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
        cell_texts = [c.get_text() for c in cells]
        parts = '\t'.join(cell_texts).split('\t')
        if len(parts) < 8:
            continue
        pos = parts[0].strip()
        if not (pos.isdigit() or pos in ('DH', 'WV', 'DSQ', 'TEMP', 'PU', 'UR', 'FE', 'AB', 'NP')):
            continue
        try:
            horse_no = int(parts[1].strip()) if parts[1].strip().isdigit() else None
            horse_name = re.sub(r'\s*\([^)]+\)\s*$', '', html.unescape(parts[2].strip())).strip()
            horse_name = re.sub(r'\s+', ' ', horse_name)
            horses.append({
                'horse_no': horse_no, 'horse_name': horse_name,
                'position': int(pos) if pos.isdigit() else pos,
                'jockey': parts[3].strip(), 'trainer': parts[4].strip(),
                'draw': int(parts[7].strip()) if len(parts) > 7 and parts[7].strip().isdigit() else None,
            })
        except (ValueError, IndexError):
            pass

    race_info = {'race_class': None, 'distance': None, 'track': 'Turf'}
    full_text = soup.get_text()
    cm = re.search(r'Class\s+(\d+)', full_text)
    dm = re.search(r'(\d{4})[Mm](?:\s|$)', full_text)
    gm = re.search(r'Group\s*(One|Two|Three)', full_text, re.I)
    if gm: race_info['race_class'] = f"Group-{gm.group(1).title()}"
    elif cm: race_info['race_class'] = f"Class {cm.group(1)}"
    if dm: race_info['distance'] = dm.group(1)
    if 'AWT' in full_text: race_info['track'] = 'AWT'

    return {'horses': horses, 'dividends': dividends, 'race_info': race_info}


async def probe_one(client, d, venue):
    date_str = d.strftime('%Y/%m/%d')
    race_nos = []
    for rn in PROBE_RACES:
        url = f'https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={rn}'
        try:
            resp = await client.get(url, timeout=20.0, follow_redirects=True)
            if resp.status_code != 200:
                break
            # Check if it's a "No information" page
            if 'No information' in resp.text or 'errorContainer' in resp.text:
                break
            parsed = parse_race_page(resp.text)
            if parsed and parsed['horses']:
                race_nos.append(rn)
            else:
                break
        except Exception as e:
            log(f"  probe err {date_str} {venue} R{rn}: {e}")
            break
    return (d, venue, race_nos) if race_nos else None


async def scrape_one(client, d, venue, race_no):
    date_str = d.strftime('%Y/%m/%d')
    url = f'https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}'
    try:
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
        if resp.status_code != 200 or 'No information' in resp.text:
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
    info = race_data.get('race_info', {})
    dist_str = info.get('distance') or ''
    dist = int(re.search(r'\d+', dist_str).group()) if re.search(r'\d+', dist_str) else None

    cur.execute("""
        INSERT INTO race_days (race_date, venue, total_races) VALUES (?, ?, 0)
        ON CONFLICT(race_date, venue) DO UPDATE SET total_races=excluded.total_races
    """, (py_date, race_data['venue']))
    rd_id = cur.execute(
        "SELECT id FROM race_days WHERE race_date=? AND venue=?",
        (py_date, race_data['venue'])
    ).fetchone()[0]

    cur.execute("""
        INSERT INTO races (race_day_id, race_no, distance, race_class, track)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(race_day_id, race_no) DO UPDATE SET
            distance=excluded.distance, race_class=excluded.race_class,
            track=COALESCE(excluded.track, track)
    """, (rd_id, race_data['race_no'], dist, info.get('race_class') or '', info.get('track', 'Turf')))
    r_id = cur.execute("SELECT id FROM races WHERE race_day_id=? AND race_no=?",
                       (rd_id, race_data['race_no'])).fetchone()[0]

    for h in race_data.get('horses', []):
        if not h.get('horse_no'):
            continue
        cur.execute("""
            INSERT INTO horse_entries
            (race_id, horse_no, horse_name, draw, jockey, trainer, finishing_position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id, horse_no) DO UPDATE SET
                draw=COALESCE(excluded.draw, horse_entries.draw),
                horse_name=excluded.horse_name,
                jockey=COALESCE(excluded.jockey, horse_entries.jockey),
                trainer=COALESCE(excluded.trainer, horse_entries.trainer),
                finishing_position=COALESCE(excluded.finishing_position, horse_entries.finishing_position)
        """, (r_id, h['horse_no'], h['horse_name'], h.get('draw'),
              h.get('jockey', ''), h.get('trainer', ''), h.get('position')))

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
        if win_v or place_v:
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
    log(f"Conservative backfill {START_DATE} -> {END_DATE}")
    candidate_days = generate_days_in_range(START_DATE, END_DATE)
    log(f"Candidate days: {len(candidate_days)}")

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    conn = sqlite3.connect(DB_PATH)

    # Serial probe with delay
    log(f"Phase 1: Probing {len(candidate_days)} race days (serial, slow)...")
    t0 = time.time()
    race_days_found = []
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=MAX_CONCURRENT)) as client:
        for d, v in candidate_days:
            r = await probe_one(client, d, v)
            if r:
                race_days_found.append(r)
                log(f"  FOUND  {d} {v}: R{r[2][0]}-{r[2][-1]} ({len(r[2])} races)")
            await asyncio.sleep(REQUEST_DELAY)

    total_races = sum(len(r[2]) for r in race_days_found)
    log(f"Phase 1 done: {len(race_days_found)} race days, {total_races} races in {time.time()-t0:.0f}s")

    if not race_days_found:
        log("No races found - HKJC may not have published older results, or all rate-limited")
        conn.close()
        return

    # Scrape
    log(f"Phase 2: Scraping {total_races} races...")
    t1 = time.time()
    races_to_scrape = [(d, v, rn) for d, v, rnos in race_days_found for rn in rnos]
    all_results = []
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=MAX_CONCURRENT)) as client:
        for d, v, rn in races_to_scrape:
            r = await scrape_one(client, d, v, rn)
            all_results.append(r)
            await asyncio.sleep(REQUEST_DELAY / 2)

    log(f"Phase 2 done in {time.time()-t1:.0f}s")

    # Save
    saved = skipped = 0
    for rd in all_results:
        if rd and rd.get('horses'):
            try:
                n = save_to_db(rd, conn)
                saved += 1
            except Exception as e:
                log(f"  ERROR {rd.get('race_date')} {rd.get('venue')} R{rd.get('race_no')}: {e}")
                skipped += 1
        else:
            skipped += 1
    log(f"DONE: {saved} saved, {skipped} skipped")
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
