#!/usr/bin/env python3
"""Scrape live odds from bet.hkjc.com for 2026-05-06 ST all 9 races."""
from playwright.sync_api import sync_playwright
import time, re

BASE = "https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{raceNo}?b_cid=SPLDSPA_hkjc-home_MegaMenu"

def parse_race(content):
    """Parse a race page content, extracting horse entries with odds."""
    # Use regex to find all table rows with horse data
    # Pattern: horse_no, horse_name, draw, weight, jockey, trainer, win_odds, place_odds
    
    entries = []
    
    # Find the racecard table (Table 0) - contains horse_no, horse_name, draw, weight, jockey, trainer, win_odds
    # We parse the HTML tables directly
    
    # Split by table
    table_pattern = r'<table[^>]*>(.*?)</table>'
    tables = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if len(tables) < 2:
        return entries
    
    # Table 0 is the racecard with: 馬號, 綵衣, 馬名, 檔位, 負磅, 騎師, 練馬師, 獨贏
    card_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tables[0], re.DOTALL | re.IGNORECASE)
    
    # Table 1 has: 馬號, 馬名, 獨贏, 位置, 獨贏及位置
    odds_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tables[1], re.DOTALL | re.IGNORECASE)
    
    # Parse Table 1 for odds (has win + place odds, cleaner)
    # Format: horse_no, horse_name (with jockey/trainer appended), win_odds, place_odds
    odds_map = {}
    for row in odds_rows[1:]:  # Skip header
        cells = re.findall(r'<t[dh][^>]*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cells) >= 4:
            try:
                horse_no = int(cells[0])
                # cells[1] is horse_name + jockey + trainer
                win_odds = float(cells[2]) if cells[2].replace('.','').isdigit() else None
                place_odds = float(cells[3]) if cells[3].replace('.','').isdigit() else None
                
                # Extract horse name from cells[1] - it's "Name騎Jockey練Trainer"
                name_part = cells[1]
                horse_name = name_part  # Will clean below
                
                odds_map[horse_no] = {'win': win_odds, 'place': place_odds, 'name_raw': name_part}
            except (ValueError, IndexError):
                pass
    
    # Parse Table 0 for horse details (no, name, draw, weight, jockey, trainer)
    for row in card_rows[1:]:  # Skip header
        cells = re.findall(r'<t[dh][^>]*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cells) >= 8:
            try:
                horse_no = int(cells[0])
                horse_name = cells[2].strip()
                draw = int(cells[3]) if cells[3].isdigit() else None
                weight = cells[4].strip()
                jockey = cells[5].strip()
                trainer = cells[6].strip()
                
                odds = odds_map.get(horse_no, {})
                win_odds = odds.get('win')
                place_odds = odds.get('place')
                
                entries.append({
                    'horse_no': horse_no,
                    'horse_name': horse_name,
                    'draw': draw,
                    'weight': weight,
                    'jockey': jockey,
                    'trainer': trainer,
                    'win_odds': win_odds,
                    'place_odds': place_odds,
                })
            except (ValueError, IndexError):
                pass
    
    return entries

def scrape_race(page, date, venue, race_no):
    url = BASE.format(date=date, venue=venue, raceNo=race_no)
    page.goto(url, timeout=30000)
    page.wait_for_load_state("networkidle", timeout=20000)
    time.sleep(2)
    
    content = page.content()
    entries = parse_race(content)
    return race_no, entries

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    page = context.new_page()
    
    all_entries = {}
    for race_no in range(1, 10):
        print(f"Scraping Race {race_no}...", end=" ")
        _, entries = scrape_race(page, "2026-05-06", "ST", race_no)
        print(f"{len(entries)} horses")
        all_entries[race_no] = entries
        time.sleep(1)
    
    browser.close()

# Print summary
print("\n" + "="*75)
print("2026-05-06 SHA TIN — Live Odds")
print("="*75)
for race_no in sorted(all_entries.keys()):
    entries = all_entries[race_no]
    if entries:
        print(f"\nRace {race_no}: {len(entries)} horses")
        for e in entries:
            wo = f"{e['win_odds']}" if e['win_odds'] else "N/A"
            po = f"{e['place_odds']}" if e['place_odds'] else "N/A"
            print(f"  #{e['horse_no']:2d} {e['horse_name']:<16} Draw:{e['draw']:2d}  Win:{wo:>5}  Place:{po:>4}")
