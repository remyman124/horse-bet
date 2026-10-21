#!/usr/bin/env python3
"""
Scrape HKJC win odds and place odds from resultsall pages
for race days 2006-01-01 to 2012-12-31.
Updates horse_entries.win_odds and horse_entries.place_odds.
"""
import sqlite3
import subprocess
import time
import json
import os
from datetime import datetime

DB_PATH = '/home/work/horse-bet/data/horse.db'
LOG_FILE = '/tmp/scrape_odds_2013_2019.log'
RESULTS_FILE = '/tmp/scrape_odds_results_2013_2019.json'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def fetch_page(date, venue):
    date_slash = date.replace('-', '/')
    url = f"https://racing.hkjc.com/en-us/local/information/resultsall?racedate={date_slash}&Racecourse={venue}"
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '15', url],
            capture_output=True, text=True, timeout=20
        )
        return result.stdout
    except Exception as e:
        log(f"  ERROR fetching {date} {venue}: {e}")
        return None

def parse_dividends(html, num_races):
    from bs4 import BeautifulSoup
    if not html or len(html) < 100:
        return {}
    soup = BeautifulSoup(html, 'html.parser')
    div_tables = soup.find_all('table', class_='dividends')
    results = {}
    for i, dt in enumerate(div_tables):
        if i >= num_races:
            break
        race_no = i + 1
        rows = dt.find_all('tr')
        win_data = None
        place_data = []
        in_place_section = False
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) < 2:
                continue
            pool = cells[0].get_text(strip=True)
            if pool == 'WIN':
                in_place_section = False
                if len(cells) >= 3:
                    try:
                        horse_no = int(cells[1].get_text(strip=True))
                        dividend = float(cells[2].get_text(strip=True))
                        win_data = (horse_no, dividend)
                    except:
                        pass
            elif pool == 'PLACE':
                in_place_section = True
                if len(cells) >= 3:
                    try:
                        horse_no = int(cells[1].get_text(strip=True))
                        dividend = float(cells[2].get_text(strip=True))
                        place_data.append((horse_no, dividend))
                    except:
                        pass
            elif in_place_section and len(cells) == 2:
                try:
                    pool_check = cells[0].get_text(strip=True)
                    horse_no = int(pool_check)
                    dividend = float(cells[1].get_text(strip=True))
                    place_data.append((horse_no, dividend))
                except:
                    in_place_section = False
            else:
                in_place_section = False
        if win_data or place_data:
            results[race_no] = {'win': win_data, 'place': place_data}
    return results

def get_db_race_info(date, venue):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.race_no, r.id, he.horse_no, he.horse_name
        FROM races r
        JOIN race_days rd ON r.race_day_id = rd.id
        JOIN horse_entries he ON he.race_id = r.id
        WHERE rd.race_date = ? AND rd.venue = ?
        ORDER BY r.race_no, he.horse_no
    """, (date, venue))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return {}
    race_info = {}
    for race_no, race_id, horse_no, horse_name in rows:
        if race_no not in race_info:
            race_info[race_no] = {'race_id': race_id, 'horses': {}}
        if horse_no:
            race_info[race_no]['horses'][horse_no] = horse_name
    return race_info

def update_odds(date, venue, dividends):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    updated = 0
    for race_no, data in dividends.items():
        race_info_map = get_db_race_info(date, venue)
        if race_no not in race_info_map:
            continue
        race_info = race_info_map[race_no]
        if data.get('win'):
            horse_no, dividend = data['win']
            if horse_no in race_info['horses']:
                win_odds = dividend
                cur.execute("""
                    UPDATE horse_entries
                    SET win_odds = ?
                    WHERE race_id = ? AND horse_no = ?
                """, (win_odds, race_info['race_id'], horse_no))
                updated += 1
        if data.get('place'):
            for horse_no, dividend in data['place']:
                if horse_no in race_info['horses']:
                    cur.execute("""
                        UPDATE horse_entries
                        SET place_odds = ?
                        WHERE race_id = ? AND horse_no = ?
                    """, (dividend, race_info['race_id'], horse_no))
                    updated += 1
    conn.commit()
    conn.close()
    return updated

def main():
    log("="*60)
    log("STARTING: Scrape HKJC odds 2013-2019")
    log("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT rd.race_date, rd.venue
        FROM race_days rd
        JOIN races r ON r.race_day_id = rd.id
        JOIN horse_entries he ON he.race_id = r.id
        WHERE rd.race_date >= '2013-01-01' AND rd.race_date <= '2019-12-31'
        ORDER BY rd.race_date
    """)
    dates = [(r[0], r[1]) for r in cur.fetchall()]
    conn.close()
    
    log(f"Found {len(dates)} race days to process")
    
    results = []
    total_updated = 0
    errors = 0
    
    for i, (date, venue) in enumerate(dates):
        log(f"[{i+1}/{len(dates)}] {date} {venue}...")
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT r.race_no)
            FROM races r
            JOIN race_days rd ON r.race_day_id = rd.id
            JOIN horse_entries he ON he.race_id = r.id
            WHERE rd.race_date = ? AND rd.venue = ?
        """, (date, venue))
        num_races = cur.fetchone()[0]
        conn.close()
        
        if num_races == 0:
            log(f"  SKIP: no entries in DB")
            continue
        
        html = fetch_page(date, venue)
        if not html:
            errors += 1
            continue
        
        dividends = parse_dividends(html, num_races)
        
        if not dividends:
            log(f"  WARN: no dividends parsed from page")
            errors += 1
            continue
        
        updated = update_odds(date, venue, dividends)
        total_updated += updated
        
        races_with_data = len(dividends)
        log(f"  OK: {races_with_data} races, {updated} entries updated")
        
        results.append({
            'date': date, 'venue': venue,
            'races': races_with_data, 'updated': updated
        })
        
        time.sleep(0.5)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': results, 'total_updated': total_updated, 'errors': errors}, f, indent=2)
    
    log("="*60)
    log(f"DONE: {len(results)} days processed, {total_updated} entries updated, {errors} errors")
    log(f"Results saved to {RESULTS_FILE}")
    log("="*60)

if __name__ == '__main__':
    main()
