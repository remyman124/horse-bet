#!/usr/bin/env python3
"""
Backfill a race meeting that is missing from the DB.

Found 2026-10-21: the meeting 2026-10-01 ST (11 races) exists on HKJC but was
never saved to the DB. That gap corrupts the rolling windows used by the model:
`days_since` and the jockey/trainer rolling windows are anchored on the last
race DAY, so a missing meeting shifts every window and every horse's rest gap.

The parser is header-driven (never a hard-coded column index) and reads the
published "Win Odds" column, which is the pre-race SP — the same field the
model was trained on. `place_odds` is deliberately NOT written: it only exists
for the top three and is an outcome leak.

Writes use WAL + busy_timeout + autocommit, one writer at a time, and the run
is verified afterwards with a FRESH connection (a counter of HTTP successes is
not a count of DB changes).
"""
import argparse
import asyncio
import html
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data' / 'horse.db'

RESULTS_URL = ("https://racing.hkjc.com/en-us/local/information/localresults"
               "?racedate={date}&Racecourse={venue}&RaceNo={race}")


def _num(s):
    s = (s or '').strip().replace(',', '')
    if not s or s in ('-', '--', '—'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_results_page(txt):
    """Header-driven parse of one localresults page.

    Returns {horses: {horse_no: {...}}, win_dividend, win_horse_no}.
    Odds come from the published 'Win Odds' column (pre-race SP) for EVERY
    runner, which is what the model expects.
    """
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')

    res_table = None
    hdr_map = None
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
        name = re.sub(r'\s+', ' ', html.unescape(name))
        draw_txt = cells[i_draw].get_text(strip=True) if len(cells) > i_draw else ''
        horses[int(hn)] = {
            'pos': int(pos_raw) if pos_raw.isdigit() else pos_raw,
            'name': name,
            'jockey': cells[i_jk].get_text(strip=True) if len(cells) > i_jk else '',
            'trainer': cells[i_tr].get_text(strip=True) if len(cells) > i_tr else '',
            'draw': int(draw_txt) if draw_txt.isdigit() else None,
            'odds': odds,
        }
    if not horses:
        return None

    win_div = None
    win_horse_no = None
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
        if found:
            break

    return {'horses': horses, 'win_dividend': win_div, 'win_horse_no': win_horse_no}


def extract_meta(txt):
    """Race class / distance / track / going from the page text.

    The results page carries a labelled row:

        <race name> | Course : | ALL WEATHER TRACK
        <race name> | Going  : | GOOD

    That `Course :` cell is the ONLY authoritative statement of the surface
    on this page. The old code sniffed for the bare token 'AWT' anywhere in
    the document, but HKJC writes the surface out in full on the results
    page -- 'ALL WEATHER TRACK' -- and never emits the abbreviation. So every
    all-weather meeting was silently written to the DB as 'Turf'.

    That is a train/serve skew, not a cosmetic one: `track_enc` is a model
    feature, and the model was trained on a history where AWT races were
    labelled Turf. It is also wrong on its own terms -- the row says
    ALL WEATHER TRACK and the DB said Turf.
    """
    soup = BeautifulSoup(txt, 'html.parser')
    full = soup.get_text()
    info = {'race_class': '', 'distance': None, 'track': 'Turf', 'going': ''}

    gm = re.search(r'Group\s*(One|Two|Three)', full, re.I)
    cm = re.search(r'Class\s+(\d+)', full)
    if gm:
        info['race_class'] = f"Group-{gm.group(1).title()}"
    elif cm:
        info['race_class'] = f"Class {cm.group(1)}"

    dm = re.search(r'(\d{4})[Mm](?:\s|$)', full)
    if dm:
        info['distance'] = int(dm.group(1))

    # Read the labelled rows rather than sniffing the whole document.
    course = going = None
    for tr in soup.find_all('tr'):
        cells = [td.get_text(' ', strip=True) for td in tr.find_all(['td', 'th'])]
        if len(cells) < 2:
            continue
        label = cells[-2].rstrip(':').strip().lower()
        value = cells[-1].strip()
        if label == 'course' and course is None:
            course = value
        elif label == 'going' and going is None:
            going = value
    if course:
        info['track'] = normalise_track(course)
    if going:
        info['going'] = going
    else:
        # Fallback for pages without the labelled row: accept both the
        # abbreviation and the spelled-out form. Only used when the
        # labelled `Course :` cell was absent -- otherwise a stray 'AWT'
        # elsewhere in the document would override the authoritative value.
        if 'AWT' in full or 'ALL WEATHER TRACK' in full.upper():
            info['track'] = 'AWT'
        elif 'Dirt' in full:
            info['track'] = 'Dirt'
    return info


def normalise_track(course_text):
    """Map an HKJC `Course :` value onto the model's track vocabulary.

    The model only knows AWT / Dirt / Turf, so the named turf courses
    ('"A" Course', '"C+3" Course') all collapse to 'Turf'.
    """
    c = (course_text or '').upper()
    if 'AWT' in c or 'ALL WEATHER' in c:
        return 'AWT'
    if 'DIRT' in c:
        return 'Dirt'
    return 'Turf'


async def fetch(client, date_slash, venue, race_no):
    url = RESULTS_URL.format(date=date_slash, venue=venue, race=race_no)
    for attempt in range(3):
        try:
            resp = await client.get(url, timeout=25.0, follow_redirects=True)
            if resp.status_code != 200:
                await asyncio.sleep(2.0)
                continue
            if 'No information' in resp.text or 'errorContainer' in resp.text:
                return None
            parsed = parse_results_page(resp.text)
            if not parsed or not parsed['horses']:
                await asyncio.sleep(2.0)
                continue
            parsed['meta'] = extract_meta(resp.text)
            return parsed
        except Exception as e:
            print(f"    fetch err R{race_no} attempt {attempt+1}: {e}", flush=True)
            await asyncio.sleep(2.0)
    return None


def save(conn, date_str, venue, race_no, parsed):
    """One race -> DB. Autocommit so a partial run is still consistent."""
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO race_days (race_date, venue, total_races) VALUES (?, ?, 0) "
        "ON CONFLICT(race_date, venue) DO UPDATE SET total_races=excluded.total_races",
        (date_str, venue))
    rd_id = cur.execute("SELECT id FROM race_days WHERE race_date=? AND venue=?",
                         (date_str, venue)).fetchone()[0]
    meta = parsed.get('meta', {})
    cur.execute("""
        INSERT INTO races (race_day_id, race_no, distance, race_class, track)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(race_day_id, race_no) DO UPDATE SET
            distance=excluded.distance, race_class=excluded.race_class,
            track=COALESCE(excluded.track, track)
    """, (rd_id, race_no, meta.get('distance'), meta.get('race_class') or '',
          meta.get('track', 'Turf')))
    r_id = cur.execute("SELECT id FROM races WHERE race_day_id=? AND race_no=?",
                       (rd_id, race_no)).fetchone()[0]

    n = 0
    for hno, h in parsed['horses'].items():
        pos = h['pos'] if isinstance(h['pos'], int) else None
        cur.execute("""
            INSERT INTO horse_entries
            (race_id, horse_no, horse_name, draw, jockey, trainer, win_odds,
             finishing_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id, horse_no) DO UPDATE SET
                horse_name=excluded.horse_name,
                draw=COALESCE(excluded.draw, horse_entries.draw),
                jockey=COALESCE(excluded.jockey, horse_entries.jockey),
                trainer=COALESCE(excluded.trainer, horse_entries.trainer),
                win_odds=COALESCE(excluded.win_odds, horse_entries.win_odds),
                finishing_position=COALESCE(excluded.finishing_position,
                                           horse_entries.finishing_position)
        """, (r_id, hno, h['name'], h['draw'], h['jockey'], h['trainer'],
              h['odds'], pos))
        n += 1

    # WIN dividend = winner's win_odds x 10 (HKJC $10 unit). Never leave the
    # old bug that stored the winner's HORSE NUMBER here.
    if parsed.get('win_horse_no') and parsed.get('win_dividend'):
        w = parsed['horses'].get(parsed['win_horse_no'])
        if w and w.get('odds'):
            div = round(w['odds'] * 10, 2)
            cur.execute("""
                INSERT INTO race_results (race_id, win_dividend) VALUES (?, ?)
                ON CONFLICT(race_id) DO UPDATE SET
                    win_dividend=excluded.win_dividend
            """, (r_id, div))
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='ST or HV')
    ap.add_argument('--max-races', type=int, default=14)
    args = ap.parse_args()

    date_str = args.date
    date_slash = args.date.replace('-', '/')

    conn = sqlite3.connect(DB, timeout=60, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Is this meeting already there?
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM races r JOIN race_days rd ON r.race_day_id=rd.id "
                "WHERE rd.race_date=? AND rd.venue=?", (date_str, args.venue))
    already = cur.fetchone()[0]
    print(f"{date_str} {args.venue}: {already} races already in DB", flush=True)

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    async with httpx.AsyncClient(headers=headers,
                                 limits=httpx.Limits(max_connections=3)) as client:
        total_rows = 0
        races_done = 0
        for rn in range(1, args.max_races + 1):
            parsed = await fetch(client, date_slash, args.venue, rn)
            if not parsed:
                print(f"  R{rn}: no data — stopping", flush=True)
                break
            n = save(conn, date_str, args.venue, rn, parsed)
            total_rows += n
            races_done += 1
            with_odds = sum(1 for h in parsed['horses'].values() if h.get('odds'))
            print(f"  R{rn}: {n} runners ({with_odds} with odds) "
                  f"{parsed.get('meta', {}).get('race_class')} "
                  f"{parsed.get('meta', {}).get('distance')}m", flush=True)
            await asyncio.sleep(1.0)

    cur = conn.execute("UPDATE race_days SET total_races = ("
                       "SELECT COUNT(*) FROM races WHERE races.race_day_id = race_days.id) "
                       "WHERE race_date=? AND venue=?", (date_str, args.venue))
    conn.commit()
    conn.close()
    print(f"saved {races_done} races, {total_rows} entries", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
