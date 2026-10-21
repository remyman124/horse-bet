#!/usr/bin/env python3
"""SQL-fast feature extraction + predict with LIVE odds for 2026-05-06 ST."""
import pickle, sqlite3, pandas as pd, numpy as np

DB = 'data/horse.db'
TODAY = '2026-05-06'

# ── Load model ──────────────────────────────────────────────────────────────
with open('models/model_v2.pkl', 'rb') as f:
    pkg = pickle.load(f)
models = pkg['models']
encoders = pkg['encoders']
FEATURE_COLS = pkg['feature_cols']
global_imp = pkg['global_implied_prob_mean']

conn = sqlite3.connect(DB)

# ── Fetch today's entries WITH LIVE ODDS ────────────────────────────────────
up = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.win_odds, re.place_odds,
           re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = ? AND rd.venue = 'ST'
    ORDER BY r.race_no, re.horse_no
''', conn, params=(TODAY,))
up['win_odds'] = pd.to_numeric(up['win_odds'], errors='coerce')
up['place_odds'] = pd.to_numeric(up['place_odds'], errors='coerce')
up['draw'] = pd.to_numeric(up['draw'], errors='coerce')
print(f"Today: {len(up)} entries, {(up['win_odds']>0).sum()} with live odds")

# ── SQL: Horse last-5 features ──────────────────────────────────────────────
h5 = pd.read_sql('''
    WITH ranked AS (
        SELECT re.horse_name, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.horse_name ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
    )
    SELECT horse_name,
           COUNT(*) as n5,
           AVG(CAST(finishing_position AS REAL)) as avg_fin5,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as win_rate5,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1.0 ELSE 0.0 END) as place_rate5,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=4 THEN 1.0 ELSE 0.0 END) as top4_rate5
    FROM ranked WHERE rn<=5 GROUP BY horse_name
''', conn, params=(TODAY,))

h10 = pd.read_sql('''
    WITH ranked AS (
        SELECT re.horse_name, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.horse_name ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
    )
    SELECT horse_name,
           COUNT(*) as n10,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as win_rate10,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1.0 ELSE 0.0 END) as place_rate10
    FROM ranked WHERE rn<=10 GROUP BY horse_name
''', conn, params=(TODAY,))

hd = pd.read_sql('''
    WITH ranked AS (
        SELECT re.horse_name, re.finishing_position, rd.race_date, r.distance,
               ROW_NUMBER() OVER (PARTITION BY re.horse_name, r.distance ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
    )
    SELECT horse_name, distance,
           COUNT(*) as nd,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as dist_win_rate
    FROM ranked WHERE rn<=5 GROUP BY horse_name, distance
''', conn, params=(TODAY,))

ht = pd.read_sql('''
    WITH ranked AS (
        SELECT re.horse_name, re.finishing_position, rd.race_date, r.track,
               ROW_NUMBER() OVER (PARTITION BY re.horse_name, r.track ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
    )
    SELECT horse_name, track,
           COUNT(*) as nt,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as track_win_rate
    FROM ranked WHERE rn<=5 GROUP BY horse_name, track
''', conn, params=(TODAY,))

jk = pd.read_sql('''
    WITH ranked AS (
        SELECT re.jockey, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.jockey ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
          AND re.jockey IS NOT NULL AND re.jockey != ''
    )
    SELECT jockey,
           COUNT(*) as nj,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as jk_win_rate,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1.0 ELSE 0.0 END) as jk_place_rate,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=4 THEN 1.0 ELSE 0.0 END) as jk_top4_rate
    FROM ranked WHERE rn<=20 GROUP BY jockey
''', conn, params=(TODAY,))

tr = pd.read_sql('''
    WITH ranked AS (
        SELECT re.trainer, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.trainer ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
          AND re.trainer IS NOT NULL AND re.trainer != ''
    )
    SELECT trainer,
           COUNT(*) as ntr,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as tr_win_rate,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1.0 ELSE 0.0 END) as tr_place_rate,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=4 THEN 1.0 ELSE 0.0 END) as tr_top4_rate
    FROM ranked WHERE rn<=20 GROUP BY trainer
''', conn, params=(TODAY,))

jt = pd.read_sql('''
    WITH ranked AS (
        SELECT re.jockey, re.trainer, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.jockey, re.trainer ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0
          AND re.jockey IS NOT NULL AND re.jockey != ''
          AND re.trainer IS NOT NULL AND re.trainer != ''
    )
    SELECT jockey, trainer,
           COUNT(*) as n_jt,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)=1 THEN 1.0 ELSE 0.0 END) as jt_win_rate,
           AVG(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1.0 ELSE 0.0 END) as jt_place_rate
    FROM ranked WHERE rn<=30 GROUP BY jockey, trainer
''', conn, params=(TODAY,))

ds = pd.read_sql('''
    WITH last_run AS (
        SELECT re.horse_name, MAX(rd.race_date) as last_date
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ?
        GROUP BY re.horse_name
    )
    SELECT horse_name,
           julianday(?) - julianday(last_date) as days_since
    FROM last_run
''', conn, params=(TODAY, TODAY))

conn.close()

# ── Merge features ─────────────────────────────────────────────────────────
up = up.merge(h5, on='horse_name', how='left')
up = up.merge(h10, on='horse_name', how='left')
up = up.merge(hd, on=['horse_name', 'distance'], how='left')
up = up.merge(ht, on=['horse_name', 'track'], how='left')
up = up.merge(jk, on='jockey', how='left')
up = up.merge(tr, on='trainer', how='left')
up = up.merge(jt, on=['jockey', 'trainer'], how='left')
up = up.merge(ds[['horse_name', 'days_since']], on='horse_name', how='left')

# ── Field size & draw ──────────────────────────────────────────────────────
fs = up.groupby('race_no')['horse_no'].transform('count')
up['field_size'] = fs
up['draw_normalized'] = up['draw'] / fs.replace(0, 1)

# ── implied_prob from LIVE odds ─────────────────────────────────────────────
median_days = ds['days_since'].median() if len(ds) > 0 else 30
up['implied_prob'] = 1 / up['win_odds'].replace(0, np.nan)
up['implied_prob'] = up['implied_prob'].fillna(global_imp)

# ── Fill missing ─────────────────────────────────────────────────────────────
fill = {
    'n5': 0, 'n10': 0, 'nd': 0, 'nt': 0, 'nj': 0, 'ntr': 0, 'n_jt': 0,
    'avg_fin5': 5.0,
    'win_rate5': 0, 'place_rate5': 0.3, 'top4_rate5': 0.4,
    'win_rate10': 0, 'place_rate10': 0.3,
    'dist_win_rate': 0, 'track_win_rate': 0,
    'jk_win_rate': 0, 'jk_place_rate': 0.3, 'jk_top4_rate': 0.4,
    'tr_win_rate': 0, 'tr_place_rate': 0.3, 'tr_top4_rate': 0.4,
    'jt_win_rate': 0, 'jt_place_rate': 0.3,
    'days_since': median_days,
    'class_drop': 0, 'is_different_distance': 0,
}
for k, v in fill.items():
    if k in up.columns:
        up[k] = up[k].fillna(v)
up['is_first_run'] = (up.get('n5', 0) == 0).astype(int)

# ── Encode categoricals ──────────────────────────────────────────────────────
def le_encode(col, enc):
    vals = col.fillna('').astype(str)
    classes = set(enc.classes_)
    return vals.map(lambda x: {v: i for i, v in enumerate(enc.classes_)}.get(x, 0)).astype(int)

up['race_class_enc'] = up['race_class'].fillna('').replace(
    {'Group-One': 'Pattern Race', 'Group-Two': 'Pattern Race',
     'Group-Three': 'Pattern Race', 'Listed': 'Pattern Race'}).apply(
    lambda x: encoders['race_class'].transform([x])[0] if x in encoders['race_class'].classes_ else 0)
up['track_enc'] = up['track'].fillna('Turf').apply(
    lambda x: encoders['track'].transform([x])[0] if x in encoders['track'].classes_ else 0)
up['venue_enc'] = up['venue'].fillna('ST').apply(
    lambda x: encoders['venue'].transform([x])[0] if x in encoders['venue'].classes_ else 0)
up['jockey_enc'] = le_encode(up['jockey'], encoders['jockey'])
up['trainer_enc'] = le_encode(up['trainer'], encoders['trainer'])

# ── Rename to match FEATURE_COLS ────────────────────────────────────────────
rename = {
    'avg_fin5': 'horse_avg_fin',
    'win_rate5': 'horse_win_rate_5',
    'place_rate5': 'horse_place_rate_5',
    'top4_rate5': 'horse_top4_rate_5',
    'win_rate10': 'horse_win_rate_10',
    'place_rate10': 'horse_place_rate_10',
    'jk_win_rate': 'jockey_win_rate',
    'jk_place_rate': 'jockey_place_rate',
    'jk_top4_rate': 'jockey_top4_rate',
    'nj': 'jockey_race_count',
    'tr_win_rate': 'trainer_win_rate',
    'tr_place_rate': 'trainer_place_rate',
    'tr_top4_rate': 'trainer_top4_rate',
    'ntr': 'trainer_race_count',
    'jt_win_rate': 'jt_combo_win_rate',
    'jt_place_rate': 'jt_combo_place_rate',
    'dist_win_rate': 'horse_dist_win_rate',
    'nd': 'horse_dist_count',
    'track_win_rate': 'horse_track_win_rate',
}
up = up.rename(columns=rename)

for col in FEATURE_COLS:
    if col not in up.columns:
        up[col] = 0

X = up[FEATURE_COLS].astype(float).fillna(0)
print(f"X: {X.shape}, NaN: {X.isnull().sum().sum()}")

# ── Predict ─────────────────────────────────────────────────────────────────
probs = {}
for name, model in models.items():
    probs[f'prob_{name}'] = model.predict_proba(X)[:, 1]
for k, v in probs.items():
    up[k] = v

for col in ['prob_win', 'prob_place', 'prob_top4']:
    total = up.groupby('race_no')[col].transform('sum')
    up[col] = up[col] / total.replace(0, 1)

up = up.sort_values(['race_no', 'prob_win'], ascending=[True, False])

# ── Print ─────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("🏇  2026-05-06  SHA TIN  —  XGBoost Model v2  (Win AUC 0.78)")
print("    ✅ LIVE odds from bet.hkjc.com — implied_prob = 1/odds")
print("=" * 75)

for race_no, grp in up.groupby('race_no'):
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

print("\n✅ Done!")
