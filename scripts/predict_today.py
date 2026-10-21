#!/usr/bin/env python3
"""
Simple prediction script for 2026-04-26 ST — no scraping needed, data already in DB.
Uses new retrained model (2024-2026 data, XGBoost).
"""
import sqlite3, pandas as pd, numpy as np, pickle, json, logging, sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('predict')

sqlite3.register_adapter(date, lambda d: d.isoformat())
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.models import engineer_features

# ── Load model ─────────────────────────────────────────────────────────────────
model_path = Path('models/model.pkl')
log.info(f"Loading model from {model_path}...")
with open(model_path, 'rb') as f:
    pkg = pickle.load(f)

# pkg is {'models': {...}, 'encoders': {...}, 'feature_cols': [...]}
models = pkg['models']
encoders = pkg['encoders']
FEATURE_COLS = pkg['feature_cols']
log.info(f"Loaded: models={list(models.keys())}, features={FEATURE_COLS}")

# ── Load DB data ────────────────────────────────────────────────────────────────
conn = sqlite3.connect('data/horse.db')

# Historical (before 2026/04/26)
hist_df = pd.read_sql("""
    SELECT re.horse_name, re.horse_no, re.draw, re.rating, re.win_odds,
           re.finishing_position, re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no,
           rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < '2026-04-26'
      AND re.finishing_position IS NOT NULL
      AND re.finishing_position > 0
    ORDER BY rd.race_date, rd.venue, r.race_no
""", conn)
hist_df['race_date'] = pd.to_datetime(hist_df['race_date'])
hist_df['distance'] = pd.to_numeric(hist_df['distance'], errors='coerce').fillna(0).astype(int)
hist_df['rating'] = pd.to_numeric(hist_df['rating'], errors='coerce')
hist_df['draw'] = pd.to_numeric(hist_df['draw'], errors='coerce')
hist_df['jockey'] = hist_df['jockey'].fillna('')
hist_df['trainer'] = hist_df['trainer'].fillna('')
log.info(f"Historical entries: {len(hist_df)}")

# Upcoming (2026/04/26)
up_df = pd.read_sql("""
    SELECT re.horse_name, re.horse_no, re.draw, re.rating, re.win_odds,
           CAST(NULL AS INTEGER) AS finishing_position,
           re.jockey, re.trainer,
           r.distance, r.race_class, r.track, r.race_no,
           rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = '2026-04-26' AND rd.venue = 'ST'
    ORDER BY r.race_no, re.horse_no
""", conn)
up_df['race_date'] = pd.to_datetime(up_df['race_date'])
up_df['distance'] = pd.to_numeric(up_df['distance'], errors='coerce').fillna(0).astype(int)
up_df['rating'] = pd.to_numeric(up_df['rating'], errors='coerce')
up_df['draw'] = pd.to_numeric(up_df['draw'], errors='coerce')
up_df['jockey'] = up_df['jockey'].fillna('')
up_df['trainer'] = up_df['trainer'].fillna('')
log.info(f"Upcoming entries: {len(up_df)} ({up_df['race_no'].nunique()} races)")
log.info(f"  race_classes: {up_df['race_class'].unique().tolist()}")
log.info(f"  distances: {sorted(up_df['distance'].unique().tolist())}")

# ── Feature engineering ────────────────────────────────────────────────────────
combined = pd.concat([hist_df, up_df], ignore_index=True)
combined = combined.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
combined = engineer_features(combined)

# Extract upcoming features
up_date = up_df['race_date'].iloc[0]
up_mask = combined['race_date'] == up_date
up_feat = combined[up_mask].copy()
log.info(f"Upcoming features extracted: {len(up_feat)} rows")

# ── Encode categoricals ───────────────────────────────────────────────────────
race_class_map = {'Group-One': '', 'Group-Two': '', 'Group-Three': '',
                  'Listed': '', 'Pattern': ''}
fallback = {'track': 'Turf', 'venue': 'ST', 'race_class': ''}

if 'race_class' in up_feat.columns:
    up_feat['race_class'] = up_feat['race_class'].fillna('').replace(race_class_map)

enc_col_map = {}
for col in ['venue', 'race_class', 'track']:
    if col in up_feat.columns and col in encoders:
        enc = encoders[col]
        fb = fallback.get(col, '')
        vals = up_feat[col].fillna('').astype(str).replace({'nan': fb, 'None': fb, '': fb})
        classes = enc.classes_
        mapping = {v: i for i, v in enumerate(classes)}
        enc_col_map[col] = vals.map(lambda x: mapping.get(x, mapping.get(fb, 0)))

# ── Build feature matrix ──────────────────────────────────────────────────────
X_up = up_feat[FEATURE_COLS].astype(float).fillna(0)

# Override encoded cols
for col, enc_series in enc_col_map.items():
    if col in X_up.columns:
        X_up[col] = enc_series.values

log.info(f"Feature matrix: {X_up.shape}")
log.info(f"Columns: {list(X_up.columns)}")

# Check for NaN/inf
if X_up.isnull().any().any():
    log.warning(f"NaN in features:\n{X_up.isnull().sum()[X_up.isnull().sum() > 0]}")
if np.isinf(X_up.values).any():
    log.warning("Inf in features!")

# ── Predict ───────────────────────────────────────────────────────────────────
log.info("Predicting...")
probs = {}
for name, m in models.items():
    p = m.predict_proba(X_up)[:, 1]
    probs[f'prob_{name}'] = p

# Assemble output
out = up_feat[['race_date', 'venue', 'race_no', 'horse_no', 'horse_name',
               'draw', 'rating', 'race_class', 'distance', 'track',
               'jockey', 'trainer', 'win_odds']].copy().reset_index(drop=True)
for k, v in probs.items():
    out[k] = v

# Normalize within each race
for col in ['prob_win', 'prob_place', 'prob_top4']:
    total = out.groupby(['race_no'])[col].transform('sum')
    out[col] = out[col] / total.replace(0, 1)

out = out.sort_values(['race_no', 'prob_win'], ascending=[True, False])

# ── Print ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🏇 2026-04-26 SHA TIN — XGBoost Predictions (Win AUC 0.74)")
print("=" * 70)

for rn, grp in out.groupby('race_no'):
    first = grp.iloc[0]
    dc = str(int(first['distance'])) if pd.notna(first['distance']) and first['distance'] > 0 else '?'
    rc = first['race_class'] or '?'
    print(f"\n  Race {rn:2d}  |  {rc:<12}  |  {dc}M")
    print(f"  {'#':<3} {'Horse':<22} {'Draw':<5} {'Rt':<4} {'Win%':<6} {'Place%':<7} {'Top4%':<6}")
    print(f"  {'-'*3} {'-'*22} {'-'*5} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
    for _, r in grp.iterrows():
        win  = f"{float(r['prob_win'])*100:.1f}%"
        place= f"{float(r['prob_place'])*100:.1f}%"
        top4 = f"{float(r['prob_top4'])*100:.1f}%"
        draw = str(int(r['draw'])) if pd.notna(r.get('draw')) and r.get('draw', 0) > 0 else '-'
        rt   = str(int(r['rating'])) if pd.notna(r.get('rating')) and r.get('rating', 0) > 0 else '-'
        name = (r['horse_name'] or '?')[:22]
        print(f"  {int(r['horse_no']):<3} {name:<22} {draw:<5} {rt:<4} {win:<6} {place:<7} {top4:<6}")

print("\n✅ Done")
