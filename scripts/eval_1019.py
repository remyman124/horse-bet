#!/usr/bin/env python3
"""Scrape 10/19 ST full results into DB + evaluate ML picks."""
import sys, re, html, time, sqlite3
from datetime import date
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

DB_PATH = '/home/work/horse-bet/data/horse.db'
RACE_DATE = '2026/10/19'
VENUE = 'ST'
LOG = Path('/tmp/eval_1019.log')

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def parse_race_page(txt):
    soup = BeautifulSoup(txt, 'html.parser')
    tables = soup.find_all('table')
    if len(tables) < 4:
        return None
    result_table, div_table = tables[2], tables[3]
    hdr = result_table.find_all('tr')
    if not hdr:
        return None
    if not any(c.get_text(strip=True) == 'Pla.' for c in hdr[0].find_all(['td', 'th'])):
        return None

    dividends = {}
    for row in div_table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 3:
            k = cells[0].get_text(strip=True)
            if k in ('WIN', 'PLACE'):
                dividends[k.lower()] = f"{cells[1].get_text(strip=True)}\t{cells[2].get_text(strip=True)}"

    horses = []
    for row in result_table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
        parts = '\t'.join(c.get_text() for c in cells).split('\t')
        if len(parts) < 8:
            continue
        pos = parts[0].strip()
        if not (pos.isdigit() or pos in ('DH', 'WV', 'DSQ', 'TEMP', 'PU', 'UR', 'FE', 'AB', 'NP')):
            continue
        try:
            hno = int(parts[1].strip()) if parts[1].strip().isdigit() else None
            hname = re.sub(r'\s*\([^)]+\)\s*$', '', html.unescape(parts[2].strip())).strip()
            hname = re.sub(r'\s+', ' ', hname)
            draw = int(parts[7].strip()) if len(parts) > 7 and parts[7].strip().isdigit() else None
            horses.append({
                'horse_no': hno, 'horse_name': hname,
                'position': int(pos) if pos.isdigit() else pos,
                'jockey': parts[3].strip(), 'trainer': parts[4].strip(),
                'draw': draw,
            })
        except (ValueError, IndexError):
            pass

    race_info = {'race_class': None, 'distance': None, 'track': 'Turf'}
    full = soup.get_text()
    cm = re.search(r'Class\s+(\d+)', full)
    dm = re.search(r'(\d{4})[Mm](?:\s|$)', full)
    gm = re.search(r'Group\s*(One|Two|Three)', full, re.I)
    if gm: race_info['race_class'] = f"Group-{gm.group(1).title()}"
    elif cm: race_info['race_class'] = f"Class {cm.group(1)}"
    if dm: race_info['distance'] = dm.group(1)
    if 'AWT' in full: race_info['track'] = 'AWT'
    return {'horses': horses, 'dividends': dividends, 'race_info': race_info}


def save_to_db(rd, conn):
    cur = conn.cursor()
    dp = rd['race_date'].split('/')
    pyd = date(int(dp[0]), int(dp[1]), int(dp[2]))
    info = rd.get('race_info', {})
    ds = info.get('distance') or ''
    dist = int(re.search(r'\d+', ds).group()) if re.search(r'\d+', ds) else None
    cur.execute("""INSERT INTO race_days (race_date, venue, total_races) VALUES (?,?,0)
        ON CONFLICT(race_date, venue) DO UPDATE SET total_races=excluded.total_races""", (pyd, VENUE))
    rd_id = cur.execute("SELECT id FROM race_days WHERE race_date=? AND venue=?", (pyd, VENUE)).fetchone()[0]
    cur.execute("""INSERT INTO races (race_day_id, race_no, distance, race_class, track)
        VALUES (?,?,?,?,?)
        ON CONFLICT(race_day_id, race_no) DO UPDATE SET
        distance=excluded.distance, race_class=excluded.race_class, track=COALESCE(excluded.track, track)""",
        (rd_id, rd['race_no'], dist, info.get('race_class') or '', info.get('track', 'Turf')))
    r_id = cur.execute("SELECT id FROM races WHERE race_day_id=? AND race_no=?", (rd_id, rd['race_no'])).fetchone()[0]
    for h in rd['horses']:
        if not h.get('horse_no'):
            continue
        cur.execute("""INSERT INTO horse_entries
            (race_id, horse_no, horse_name, draw, jockey, trainer, finishing_position)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(race_id, horse_no) DO UPDATE SET
            draw=COALESCE(excluded.draw, horse_entries.draw),
            horse_name=excluded.horse_name,
            jockey=COALESCE(excluded.jockey, horse_entries.jockey),
            trainer=COALESCE(excluded.trainer, horse_entries.trainer),
            finishing_position=COALESCE(excluded.finishing_position, horse_entries.finishing_position)""",
            (r_id, h['horse_no'], h['horse_name'], h.get('draw'), h.get('jockey', ''), h.get('trainer', ''), h.get('position')))
    divs = rd.get('dividends', {})
    wv = pv = None
    for p in (divs.get('win') or '').split('\t'):
        p = p.strip()
        if re.match(r'^\d+(?:\.\d+)?$', p): wv = float(p); break
    for p in (divs.get('place') or '').split('\t'):
        p = p.strip()
        if re.match(r'^\d+(?:\.\d+)?$', p): pv = float(p); break
    if wv or pv:
        cur.execute("""INSERT INTO race_results (race_id, win_dividend, place_dividend_1) VALUES (?,?,?)
            ON CONFLICT(race_id) DO UPDATE SET win_dividend=excluded.win_dividend,
            place_dividend_1=excluded.place_dividend_1""", (r_id, wv, pv))
    conn.commit()
    return len(rd['horses'])


def main():
    LOG.write_text('')
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    conn = sqlite3.connect(DB_PATH)
    all_results = {}
    with httpx.Client(headers=headers, timeout=20.0) as client:
        for rn in range(1, 12):
            url = f'https://racing.hkjc.com/en-us/local/information/localresults?racedate={RACE_DATE}&Racecourse={VENUE}&RaceNo={rn}'
            try:
                r = client.get(url)
                if r.status_code != 200 or 'No information' in r.text:
                    log(f'R{rn}: no data (status {r.status_code})')
                    break
                p = parse_race_page(r.text)
                if not p or not p['horses']:
                    log(f'R{rn}: parse failed')
                    break
                all_results[rn] = p
                n = save_to_db({'race_date': RACE_DATE, 'venue': VENUE, 'race_no': rn, **p}, conn)
                log(f'R{rn}: {p["race_info"].get("race_class")} {p["race_info"].get("distance")}m - {n} horses saved')
            except Exception as e:
                log(f'R{rn}: error {e}')
                break
            time.sleep(1.2)

    cur = conn.cursor()
    cur.execute("""UPDATE race_days SET total_races = (SELECT COUNT(*) FROM races WHERE races.race_day_id = race_days.id)
                    WHERE total_races = 0""")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM horse_entries WHERE race_id IN (SELECT r.id FROM races r JOIN race_days rd ON r.race_day_id=rd.id WHERE rd.race_date='2026-10-19')")
    log(f'Total entries for 10/19: {cur.fetchone()[0]}')
    conn.close()


if __name__ == '__main__':
    main()
