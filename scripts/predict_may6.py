#!/usr/bin/env python3
"""Predict for 2026-05-06 ST using model_v2"""
import pickle, sqlite3, pandas as pd, numpy as np
from pathlib import Path

# ── Load model_v2 ───────────────────────────────────────────────────────────
with open('models/model_v2.pkl', 'rb') as f:
    pkg = pickle.load(f)

models      = pkg['models']
calibrators = pkg['calibrators']
encoders    = pkg['encoders']
FEATURE_COLS = pkg['feature_cols']
global_imp   = pkg['global_implied_prob_mean']  # 0.0840

# ── Load historical + today data ───────────────────────────────────────────
conn = sqlite3.connect('data/horse.db')

hist_df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.win_odds, re.place_odds,
           re.finishing_position, re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no,
           rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < '2026-05-06'
      AND re.finishing_position IS NOT NULL
      AND CAST(re.finishing_position AS INTEGER) > 0
    ORDER BY rd.race_date, rd.venue, r.race_no
''', conn)
hist_df['race_date'] = pd.to_datetime(hist_df['race_date'])
hist_df['distance']  = pd.to_numeric(hist_df['distance'], errors='coerce')
hist_df['draw']      = pd.to_numeric(hist_df['draw'], errors='coerce')
hist_df['win_odds']  = pd.to_numeric(hist_df['win_odds'], errors='coerce')
hist_df['finishing_position'] = pd.to_numeric(hist_df['finishing_position'], errors='coerce').fillna(0).astype(int)

up_df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw,
           re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no,
           rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = '2026-05-06' AND rd.venue = 'ST'
    ORDER BY r.race_no, re.horse_no
''', conn)
up_df['race_date'] = pd.to_datetime(up_df['race_date'])
up_df['distance']  = pd.to_numeric(up_df['distance'], errors='coerce')
up_df['draw']      = pd.to_numeric(up_df['draw'], errors='coerce')
up_df['win_odds']  = np.nan
up_df['finishing_position'] = 0

conn.close()

# ── Combine & sort ─────────────────────────────────────────────────────────
df = pd.concat([hist_df, up_df], ignore_index=True)
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# ── Feature engineering ──────────────────────────────────────────────────────
df['implied_prob'] = 1 / df['win_odds'].replace(0, np.nan)
df['implied_prob'] = df['implied_prob'].fillna(global_imp)

df['horse_avg_fin']       = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['horse_win_rate_5']    = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_place_rate_5']  = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(5, min_periods=1).mean())
df['horse_top4_rate_5']   = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(5, min_periods=1).mean())
df['horse_win_rate_10']   = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(10, min_periods=1).mean())
df['horse_place_rate_10'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(10, min_periods=1).mean())

df['days_since_last'] = df.groupby('horse_name')['race_date'].transform(
    lambda x: (x - x.shift(1)).dt.days)
median_days = df['days_since_last'].median()
df['days_since_last'] = df['days_since_last'].fillna(median_days)

df = df.sort_values(['jockey', 'race_date', 'venue']).reset_index(drop=True)
df['jockey_win_rate']   = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean())
df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean())
df['jockey_top4_rate']  = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean())
df['jockey_race_count'] = df.groupby('jockey')['race_date'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).count())

df = df.sort_values(['trainer', 'race_date', 'venue']).reset_index(drop=True)
df['trainer_win_rate']   = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean())
df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean())
df['trainer_top4_rate']  = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean())
df['trainer_race_count'] = df.groupby('trainer')['race_date'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).count())

df = df.sort_values(['jockey', 'trainer', 'race_date']).reset_index(drop=True)
df['jt_combo_win_rate']  = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(30, min_periods=1).mean())
df['jt_combo_place_rate'] = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(30, min_periods=1).mean())

df = df.sort_values(['horse_name', 'distance', 'race_date']).reset_index(drop=True)
df['horse_dist_win_rate'] = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_dist_count']    = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

df = df.sort_values(['horse_name', 'track', 'race_date']).reset_index(drop=True)
df['horse_track_win_rate'] = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_track_count']   = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

class_order = {'': 0, 'Class 5': 1, 'Class 4': 2, 'Class 3': 3, 'Class 2': 4, 'Class 1': 5,
                'Group-Three': 6, 'Group-Two': 7, 'Group-One': 8, 'Listed': 5, 'Pattern Race': 6}
df['race_class_std'] = df['race_class'].fillna('').replace(
    {'Group-One': 'Pattern Race', 'Group-Two': 'Pattern Race', 'Group-Three': 'Pattern Race', 'Listed': 'Pattern Race'})
df['race_class_level'] = df['race_class_std'].map(class_order).fillna(0)
df['prev_class_level'] = df.groupby('horse_name')['race_class_level'].shift(1)
df['class_drop'] = ((df['prev_class_level'] > df['race_class_level']) & df['prev_class_level'].notna()).astype(int)

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

# ── Extract today's features ───────────────────────────────────────────────
up_mask = df['race_date'] == pd.Timestamp('2026-05-06')
up_feat = df[up_mask].copy().fillna(0)

X_up = up_feat[FEATURE_COLS].astype(float).fillna(0)
print(f"Feature matrix: {X_up.shape}, NaN: {X_up.isnull().sum().sum()}")

# ── Predict ────────────────────────────────────────────────────────────────
probs = {}
for name, model in models.items():
    p = model.predict_proba(X_up)[:, 1]
    probs[f'prob_{name}'] = p

out = up_feat[['race_no', 'horse_no', 'horse_name', 'draw', 'jockey', 'trainer',
               'distance', 'race_class', 'track']].copy().reset_index(drop=True)
for k, v in probs.items():
    out[k] = v

# Normalize within each race
for col in ['prob_win', 'prob_place', 'prob_top4']:
    total = out.groupby('race_no')[col].transform('sum')
    out[col] = out[col] / total.replace(0, 1)

out = out.sort_values(['race_no', 'prob_win'], ascending=[True, False])

# ── Print ─────────────────────────────────────────────────────────────────
print()
print("=" * 75)
print("🏇  2026-05-06  SHA TIN  —  XGBoost Model v2  (Win AUC 0.78)")
print("    ⚠️  No LIVE odds — implied_prob = historical mean (0.084)")
print("=" * 75)

for rn, grp in out.groupby('race_no'):
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
