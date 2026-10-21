#!/usr/bin/env python3
"""Scrape odds + predict for 2026-05-06 ST using live odds from bet.hkjc.com."""
from playwright.sync_api import sync_playwright
import time, re, pickle, sqlite3, pandas as pd, numpy as np

DB = 'data/horse.db'
TODAY = '2026-05-06'
BASE = "https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{raceNo}?b_cid=SPLDSPA_hkjc-home_MegaMenu"

# ── 1. SCRAPE LIVE ODDS ─────────────────────────────────────────────────────
def parse_race(content):
    tables = re.findall(r'<table[^>]*>(.*?)</table>', content, re.DOTALL | re.IGNORECASE)
    if len(tables) < 2:
        return {}
    # Table 1: horse_no, name+jockey+trainer, win, place
    odds_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tables[1], re.DOTALL | re.IGNORECASE)
    odds_map = {}
    for row in odds_rows[1:]:
        cells = re.findall(r'<t[dh][^>]*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cells) >= 4:
            try:
                horse_no = int(cells[0])
                win_odds = float(cells[2]) if cells[2].replace('.','').replace(' ','').isdigit() else None
                place_odds = float(cells[3]) if cells[3].replace('.','').replace(' ','').isdigit() else None
                if win_odds:
                    odds_map[horse_no] = {'win': win_odds, 'place': place_odds}
            except:
                pass
    # Table 0: horse_no, name, draw, weight, jockey, trainer
    card_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tables[0], re.DOTALL | re.IGNORECASE)
    entries = {}
    for row in card_rows[1:]:
        cells = re.findall(r'<t[dh][^>]*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if len(cells) >= 8:
            try:
                horse_no = int(cells[0])
                entries[horse_no] = {
                    'horse_name': cells[2].strip(),
                    'draw': int(cells[3]) if cells[3].isdigit() else None,
                    'jockey': cells[5].strip(),
                    'trainer': cells[6].strip(),
                }
                if horse_no in odds_map:
                    entries[horse_no].update(odds_map[horse_no])
            except:
                pass
    return entries

print("Scraping live odds...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page()
    all_entries = {}
    for race_no in range(1, 10):
        url = BASE.format(date=TODAY, venue="ST", raceNo=race_no)
        page.goto(url, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(2)
        content = page.content()
        entries = parse_race(content)
        all_entries[race_no] = entries
        print(f"  Race {race_no}: {len(entries)} horses")
        time.sleep(1)
    browser.close()

# ── 2. UPDATE DB WITH LIVE ODDS ─────────────────────────────────────────────
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Get race_day_id
cur.execute("SELECT id FROM race_days WHERE race_date = ? AND venue = 'ST'", (TODAY,))
rd_row = cur.fetchone()
if not rd_row:
    print("ERROR: race day not found in DB")
    exit(1)
rd_id = rd_row[0]

updated = 0
for race_no, entries in all_entries.items():
    # Get race_id
    cur.execute("SELECT id FROM races WHERE race_day_id = ? AND race_no = ?", (rd_id, race_no))
    r_row = cur.fetchone()
    if not r_row:
        print(f"  Race {race_no}: not found in DB")
        continue
    race_id = r_row[0]
    
    for horse_no, data in entries.items():
        win_odds = data.get('win')
        place_odds = data.get('place')
        if win_odds:
            cur.execute("""
                UPDATE horse_entries SET win_odds = ?, place_odds = ?
                WHERE race_id = ? AND horse_no = ?
            """, (win_odds, place_odds, race_id, horse_no))
            updated += 1

conn.commit()
conn.close()
print(f"\nUpdated {updated} entries with live odds")

# ── 3. LOAD MODEL ────────────────────────────────────────────────────────────
with open('models/model_v2.pkl', 'rb') as f:
    pkg = pickle.load(f)
models = pkg['models']
encoders = pkg['encoders']
FEATURE_COLS = pkg['feature_cols']
global_imp = pkg['global_implied_prob_mean']

# ── 4. FETCH TODAY'S DATA WITH LIVE ODDS ────────────────────────────────────
conn = sqlite3.connect(DB)
up = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.win_odds, re.place_odds,
           re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = ? AND rd.venue = 'ST'
    ORDER BY r.race_no, re.horse_no
''', conn, params=(TODAY,))
conn.close()

up['win_odds'] = pd.to_numeric(up['win_odds'], errors='coerce')
up['place_odds'] = pd.to_numeric(up['place_odds'], errors='coerce')
up['draw'] = pd.to_numeric(up['draw'], errors='coerce')
up['finishing_position'] = 0

print(f"\nToday's entries: {len(up)} ({up['race_no'].nunique()} races)")
print(f"With live odds: {(up['win_odds'] > 0).sum()}/{len(up)}")

# ── 5. FETCH HISTORICAL DATA ────────────────────────────────────────────────
conn = sqlite3.connect(DB)
hist = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.win_odds, re.place_odds,
           re.finishing_position, re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no,
           rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < ? AND re.finishing_position IS NOT NULL
      AND CAST(re.finishing_position AS INTEGER) > 0
    ORDER BY rd.race_date, rd.venue, r.race_no
''', conn, params=(TODAY,))
hist['win_odds'] = pd.to_numeric(hist['win_odds'], errors='coerce')
hist['draw'] = pd.to_numeric(hist['draw'], errors='coerce')
hist['finishing_position'] = pd.to_numeric(hist['finishing_position'], errors='coerce').fillna(0).astype(int)
hist['race_date'] = pd.to_datetime(hist['race_date'])
conn.close()

# ── 6. COMBINE & ENGINEER FEATURES ──────────────────────────────────────────
df = pd.concat([hist, up], ignore_index=True)
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# implied_prob — using ACTUAL live odds for today!
df['implied_prob'] = 1 / df['win_odds'].replace(0, np.nan)
df['implied_prob'] = df['implied_prob'].fillna(global_imp)

# Horse rolling features
for col, g in [
    ('horse_avg_fin', 'horse_name'),
    ('horse_win_rate_5', 'horse_name'),
    ('horse_place_rate_5', 'horse_name'),
    ('horse_top4_rate_5', 'horse_name'),
]:
    df[col] = df.groupby(g)['finishing_position'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean() if col == 'horse_avg_fin'
        else x.shift(1).eq(1).rolling(5, min_periods=1).mean() if 'win' in col
        else x.shift(1).isin([1,2,3]).rolling(5, min_periods=1).mean() if 'place' in col
        else x.shift(1).isin([1,2,3,4]).rolling(5, min_periods=1).mean()
    )

for col, g in [('horse_win_rate_10', 'horse_name'), ('horse_place_rate_10', 'horse_name')]:
    df[col] = df.groupby(g)['finishing_position'].transform(
        lambda x: x.shift(1).eq(1).rolling(10, min_periods=1).mean() if 'win' in col
        else x.shift(1).isin([1,2,3]).rolling(10, min_periods=1).mean()
    )

df['days_since_last'] = df.groupby('horse_name')['race_date'].transform(
    lambda x: (x - x.shift(1)).dt.days)
df['days_since_last'] = df['days_since_last'].fillna(df['days_since_last'].median())

# Jockey
df = df.sort_values(['jockey', 'race_date', 'venue']).reset_index(drop=True)
for col in ['jockey_win_rate', 'jockey_place_rate', 'jockey_top4_rate', 'jockey_race_count']:
    df[col] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean() if 'win' in col
        else x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean() if 'place' in col
        else x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean() if 'top4' in col
        else x.shift(1).rolling(20, min_periods=1).count()
    )

# Trainer
df = df.sort_values(['trainer', 'race_date', 'venue']).reset_index(drop=True)
for col in ['trainer_win_rate', 'trainer_place_rate', 'trainer_top4_rate', 'trainer_race_count']:
    df[col] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean() if 'win' in col
        else x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean() if 'place' in col
        else x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean() if 'top4' in col
        else x.shift(1).rolling(20, min_periods=1).count()
    )

# Jockey x Trainer
df = df.sort_values(['jockey', 'trainer', 'race_date']).reset_index(drop=True)
df['jt_combo_win_rate'] = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(30, min_periods=1).mean())
df['jt_combo_place_rate'] = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(30, min_periods=1).mean())

# Horse x distance
df = df.sort_values(['horse_name', 'distance', 'race_date']).reset_index(drop=True)
df['horse_dist_win_rate'] = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_dist_count'] = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

# Horse x track
df = df.sort_values(['horse_name', 'track', 'race_date']).reset_index(drop=True)
df['horse_track_win_rate'] = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_track_count'] = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

# Class drop
class_order = {'': 0, 'Class 5': 1, 'Class 4': 2, 'Class 3': 3, 'Class 2': 4, 'Class 1': 5,
                'Group-Three': 6, 'Group-Two': 7, 'Group-One': 8, 'Listed': 5, 'Pattern Race': 6}
df['race_class_std'] = df['race_class'].fillna('').replace(
    {'Group-One': 'Pattern Race', 'Group-Two': 'Pattern Race', 'Group-Three': 'Pattern Race', 'Listed': 'Pattern Race'})
df['race_class_level'] = df['race_class_std'].map(class_order).fillna(0)
df['prev_class_level'] = df.groupby('horse_name')['race_class_level'].shift(1)
df['class_drop'] = ((df['prev_class_level'] > df['race_class_level']) & df['prev_class_level'].notna()).astype(int)

# Race context
df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
df['draw_normalized'] = df['draw'] / df['field_size'].replace(0, 1)
df['prev_distance'] = df.groupby('horse_name')['distance'].shift(1)
df['is_different_distance'] = (df['distance'] != df['prev_distance']).astype(int)
df['is_first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

# Encode categoricals
df['race_class_enc'] = df['race_class_std'].apply(
    lambda x: encoders['race_class'].transform([x])[0] if x in encoders['race_class'].classes_ else 0)
df['track_enc'] = df['track'].fillna('Turf').apply(
    lambda x: encoders['track'].transform([x])[0] if x in encoders['track'].classes_ else 0)
df['venue_enc'] = df['venue'].fillna('ST').apply(
    lambda x: encoders['venue'].transform([x])[0] if x in encoders['venue'].classes_ else 0)
df['jockey_enc'] = df['jockey'].fillna('').apply(
    lambda x: encoders['jockey'].transform([x])[0] if x in encoders['jockey'].classes_ else 0)
df['trainer_enc'] = df['trainer'].fillna('').apply(
    lambda x: encoders['trainer'].transform([x])[0] if x in encoders['trainer'].classes_ else 0)

# ── 7. EXTRACT TODAY'S FEATURES ─────────────────────────────────────────────
up_mask = df['race_date'] == pd.Timestamp(TODAY)
up_feat = df[up_mask].copy().fillna(0)

for col in FEATURE_COLS:
    if col not in up_feat.columns:
        up_feat[col] = 0

X = up_feat[FEATURE_COLS].astype(float).fillna(0)
print(f"Feature matrix: {X.shape}, NaN: {X.isnull().sum().sum()}")

# ── 8. PREDICT ──────────────────────────────────────────────────────────────
probs = {}
for name, model in models.items():
    probs[f'prob_{name}'] = model.predict_proba(X)[:, 1]

for k, v in probs.items():
    up_feat[k] = v

# Normalize within race
for col in ['prob_win', 'prob_place', 'prob_top4']:
    total = up_feat.groupby('race_no')[col].transform('sum')
    up_feat[col] = up_feat[col] / total.replace(0, 1)

up_feat = up_feat.sort_values(['race_no', 'prob_win'], ascending=[True, False])

# ── 9. PRINT RESULTS ────────────────────────────────────────────────────────
# Merge back live odds for display
up_feat = up_feat.merge(
    up[['horse_no', 'race_no', 'win_odds', 'place_odds']],
    on=['horse_no', 'race_no'],
    how='left', suffixes=('', '_live')
)

print()
print("=" * 75)
print("🏇  2026-05-06  SHA TIN  —  XGBoost Model v2  (Win AUC 0.78)")
print("    ✅ LIVE odds from bet.hkjc.com — implied_prob = 1/odds")
print("=" * 75)

for race_no, grp in up_feat.groupby('race_no'):
    first = grp.iloc[0]
    dc = str(int(first['distance'])) if pd.notna(first['distance']) and first['distance'] > 0 else '?'
    rc = first['race_class'] or '?'
    print(f"\n  Race {race_no:2d}  |  {rc:<12}  |  {dc}M")
    print(f"  {'#':<3} {'Horse':<16} {'Draw':<5} {'Odds':<6} {'Jockey':<12} {'Win%':<6} {'Place%':<7} {'Top4%':<6}")
    print(f"  {'-'*3} {'-'*16} {'-'*5} {'-'*6} {'-'*12} {'-'*6} {'-'*7} {'-'*6}")
    for _, r in grp.iterrows():
        win   = f"{float(r['prob_win'])*100:.1f}%"
        place = f"{float(r['prob_place'])*100:.1f}%"
        top4  = f"{float(r['prob_top4'])*100:.1f}%"
        draw  = str(int(r['draw'])) if pd.notna(r.get('draw')) and r.get('draw', 0) > 0 else '-'
        name  = (r['horse_name'] or '?')[:16]
        jockey = (r['jockey'] or '')[:12]
        wo = f"{r['win_odds']:.1f}" if pd.notna(r.get('win_odds')) and r.get('win_odds', 0) > 0 else '-'
        print(f"  {int(r['horse_no']):<3} {name:<16} {draw:<5} {wo:<6} {jockey:<12} {win:<6} {place:<7} {top4:<6}")

print("\n✅ Done — predictions with live odds!")
