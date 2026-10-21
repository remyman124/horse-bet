#!/usr/bin/env python3
"""Fast prediction for 2026-05-06 ST using SQL-precomputed features."""
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

# ── Fetch today's entries ───────────────────────────────────────────────────
conn = sqlite3.connect(DB)
up = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw,
           re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = ? AND rd.venue = 'ST'
    ORDER BY r.race_no, re.horse_no
''', conn, params=(TODAY,))

print(f"Today's races: {up['race_no'].nunique()}, entries: {len(up)}")

# ── Precompute all features via SQL ─────────────────────────────────────────
# Horse rolling stats (last 5 runs)
horse_5 = pd.read_sql('''
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

# Horse rolling stats (last 10 runs)
horse_10 = pd.read_sql('''
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

# Horse x distance win rate
horse_dist = pd.read_sql('''
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
           COUNT(*) as nd, SUM(CASE WHEN finishing_position=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) as dist_win_rate
    FROM ranked WHERE rn<=5 GROUP BY horse_name, distance
''', conn, params=(TODAY,))

# Horse x track win rate
horse_track = pd.read_sql('''
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
           COUNT(*) as nt, SUM(CASE WHEN finishing_position=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) as track_win_rate
    FROM ranked WHERE rn<=5 GROUP BY horse_name, track
''', conn, params=(TODAY,))

# Jockey stats
jockey_stats = pd.read_sql('''
    WITH ranked AS (
        SELECT re.jockey, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.jockey ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0 AND re.jockey IS NOT NULL AND re.jockey != ''
    )
    SELECT jockey,
           COUNT(*) as nj, 
           SUM(CASE WHEN finishing_position=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) as jk_win_rate,
           SUM(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1 ELSE 0 END)*1.0/COUNT(*) as jk_place_rate,
           SUM(CASE WHEN CAST(finishing_position AS INTEGER)<=4 THEN 1 ELSE 0 END)*1.0/COUNT(*) as jk_top4_rate
    FROM ranked WHERE rn<=20 GROUP BY jockey
''', conn, params=(TODAY,))

# Trainer stats
trainer_stats = pd.read_sql('''
    WITH ranked AS (
        SELECT re.trainer, re.finishing_position, rd.race_date,
               ROW_NUMBER() OVER (PARTITION BY re.trainer ORDER BY rd.race_date DESC) as rn
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ? AND re.finishing_position NOT NULL
          AND CAST(re.finishing_position AS INTEGER) > 0 AND re.trainer IS NOT NULL AND re.trainer != ''
    )
    SELECT trainer,
           COUNT(*) as nt, 
           SUM(CASE WHEN finishing_position=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) as tr_win_rate,
           SUM(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1 ELSE 0 END)*1.0/COUNT(*) as tr_place_rate,
           SUM(CASE WHEN CAST(finishing_position AS INTEGER)<=4 THEN 1 ELSE 0 END)*1.0/COUNT(*) as tr_top4_rate
    FROM ranked WHERE rn<=20 GROUP BY trainer
''', conn, params=(TODAY,))

# Jockey x Trainer combo
jt_stats = pd.read_sql('''
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
           SUM(CASE WHEN finishing_position=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) as jt_win_rate,
           SUM(CASE WHEN CAST(finishing_position AS INTEGER)<=3 THEN 1 ELSE 0 END)*1.0/COUNT(*) as jt_place_rate
    FROM ranked WHERE rn<=30 GROUP BY jockey, trainer
''', conn, params=(TODAY,))

# Days since last run
days_since = pd.read_sql('''
    WITH last_run AS (
        SELECT re.horse_name, MAX(rd.race_date) as last_date
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date < ?
        GROUP BY re.horse_name
    )
    SELECT horse_name, last_date,
           julianday(?) - julianday(last_date) as days_since
    FROM last_run
''', conn, params=(TODAY, TODAY))

conn.close()

# ── Merge features into today's entries ─────────────────────────────────────
up = up.merge(horse_5, on='horse_name', how='left')
up = up.merge(horse_10, on='horse_name', how='left')
up = up.merge(horse_dist, on=['horse_name', 'distance'], how='left')
up = up.merge(horse_track, on=['horse_name', 'track'], how='left')
up = up.merge(jockey_stats, on='jockey', how='left')
up = up.merge(trainer_stats, on='trainer', how='left')
up = up.merge(jt_stats, on=['jockey', 'trainer'], how='left')
up = up.merge(days_since[['horse_name', 'days_since']], on='horse_name', how='left')

# ── Field size ─────────────────────────────────────────────────────────────
field_size = up.groupby('race_no')['horse_no'].transform('count')
up['field_size'] = field_size
up['draw_normalized'] = up['draw'] / field_size.replace(0, 1)

# ── Global defaults for missing ─────────────────────────────────────────────
median_days = days_since['days_since'].median() if len(days_since) > 0 else 30
up['days_since'] = up['days_since'].fillna(median_days)
up['implied_prob'] = global_imp
up['avg_fin5'] = up['avg_fin5'].fillna(5.0)
up['win_rate5'] = up['win_rate5'].fillna(0)
up['place_rate5'] = up['place_rate5'].fillna(0.3)
up['top4_rate5'] = up['top4_rate5'].fillna(0.4)
up['win_rate10'] = up['win_rate10'].fillna(0)
up['place_rate10'] = up['place_rate10'].fillna(0.3)
up['dist_win_rate'] = up['dist_win_rate'].fillna(0)
up['nd'] = up['nd'].fillna(0)
up['track_win_rate'] = up['track_win_rate'].fillna(0)
up['jk_win_rate'] = up['jk_win_rate'].fillna(0)
up['jk_place_rate'] = up['jk_place_rate'].fillna(0)
up['jk_top4_rate'] = up['jk_top4_rate'].fillna(0.4)
up['tr_win_rate'] = up['tr_win_rate'].fillna(0)
up['tr_place_rate'] = up['tr_place_rate'].fillna(0.3)
up['tr_top4_rate'] = up['tr_top4_rate'].fillna(0.4)
up['jt_win_rate'] = up['jt_win_rate'].fillna(0)
up['jt_place_rate'] = up['jt_place_rate'].fillna(0.3)
up['n_jt'] = up['n_jt'].fillna(0)

# Rename + fill race count cols
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
    'nt': 'trainer_race_count',
    'jt_win_rate': 'jt_combo_win_rate',
    'jt_place_rate': 'jt_combo_place_rate',
    'dist_win_rate': 'horse_dist_win_rate',
    'nd': 'horse_dist_count',
    'track_win_rate': 'horse_track_win_rate',
}
up = up.rename(columns=rename)

up['n5'] = up['n5'].fillna(0)
up['n10'] = up['n10'].fillna(0)
up['jockey_race_count'] = up['jockey_race_count'].fillna(0)
up['trainer_race_count'] = up['trainer_race_count'].fillna(0)

for col in FEATURE_COLS:
    if col not in up.columns:
        up[col] = 0

X = up[FEATURE_COLS].astype(float).fillna(0)
print(f"X shape: {X.shape}")
print(f"NaN: {X.isnull().sum().sum()}")

# ── Predict ─────────────────────────────────────────────────────────────────
probs = {}
for name, model in models.items():
    probs[f'prob_{name}'] = model.predict_proba(X)[:, 1]

for k, v in probs.items():
    up[k] = v

# Normalize within race
for col in ['prob_win', 'prob_place', 'prob_top4']:
    total = up.groupby('race_no')[col].transform('sum')
    up[col] = up[col] / total.replace(0, 1)

up = up.sort_values(['race_no', 'prob_win'], ascending=[True, False])

# ── Print ─────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("🏇  2026-05-06  SHA TIN  —  XGBoost Model v2  (Win AUC 0.78)")
print("    ⚠️  No LIVE odds — implied_prob = historical mean (0.084)")
print("=" * 75)

for rn, grp in up.groupby('race_no'):
    first = grp.iloc[0]
    dc = str(int(first['distance'])) if pd.notna(first['distance']) and first['distance'] > 0 else '?'
    rc = first['race_class'] or '?'
    print(f"\n  Race {rn:2d}  |  {rc:<12}  |  {dc}M")
    print(f"  {'#':<3} {'Horse':<22} {'Draw':<5} {'Jockey':<12} {'Trainer':<12} {'Win%':<6} {'Place%':<7} {'Top4%':<6}")
    print(f"  {'-'*3} {'-'*22} {'-'*5} {'-'*12} {'-'*12} {'-'*6} {'-'*7} {'-'*6}")
    for _, r in grp.iterrows():
        win   = f"{float(r['prob_win'])*100:.1f}%"
        place = f"{float(r['prob_place'])*100:.1f}%"
        top4  = f"{float(r['prob_top4'])*100:.1f}%"
        draw  = str(int(r['draw'])) if pd.notna(r.get('draw')) and r.get('draw', 0) > 0 else '-'
        name  = (r['horse_name'] or '?')[:22]
        jockey = (r['jockey'] or '')[:12]
        trainer = (r['trainer'] or '')[:12]
        print(f"  {int(r['horse_no']):<3} {name:<22} {draw:<5} {jockey:<12} {trainer:<12} {win:<6} {place:<7} {top4:<6}")

print("\n✅ Done")
