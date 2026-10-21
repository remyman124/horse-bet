#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 3 (FAST)
==========================================
Changes from v2:
- REMOVED implied_prob (negative correlation in this DB!)
- Simplified feature engineering for speed
- Lighter XGBoost with early stopping
"""
import pickle, sqlite3, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('retrain_v3')

DB  = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model_v3.pkl'

# ── Load data ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw,
           re.jockey, re.trainer,
           re.win_odds, re.place_odds,
           re.finishing_position,
           rd.race_date, rd.venue,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
      AND CAST(re.finishing_position AS INTEGER) > 0
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
''', conn)
conn.close()

df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce').fillna(0).astype(int)
df['draw']               = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['win_odds']           = pd.to_numeric(df['win_odds'], errors='coerce').fillna(0)
df['place_odds']         = pd.to_numeric(df['place_odds'], errors='coerce').fillna(0)
df['distance']           = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['race_date']          = pd.to_datetime(df['race_date'])

log.info(f"Loaded {len(df):,} entries, {df['race_date'].min().date()} – {df['race_date'].max().date()}")

# ── Simple, fast feature engineering ──────────────────────────────────────────
# Sort once for all rolling calculations
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# 1. Horse rolling win rate (last 5 races) — simple shift+groupby
df['prev_fp'] = df.groupby('horse_name')['finishing_position'].shift(1)
df['horse_win_5'] = df.groupby('horse_name')['prev_fp'].transform(
    lambda x: x.eq(1).rolling(5, min_periods=1).mean())
df['horse_place_5'] = df.groupby('horse_name')['prev_fp'].transform(
    lambda x: x.isin([1,2,3]).rolling(5, min_periods=1).mean())
df['horse_avg_fp_5'] = df.groupby('horse_name')['prev_fp'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())
df.drop('prev_fp', axis=1, inplace=True)

# 2. Days since last run
df['prev_date'] = df.groupby('horse_name')['race_date'].shift(1)
df['days_since'] = (df['race_date'] - df['prev_date']).dt.days
median_days = df['days_since'].median()
df['days_since'] = df['days_since'].fillna(median_days)
df.drop('prev_date', axis=1, inplace=True)

# 3. Jockey win rate (last 30 rides) — fast
df = df.sort_values(['jockey', 'race_date']).reset_index(drop=True)
df['jockey_prev_fp'] = df.groupby('jockey')['finishing_position'].shift(1)
df['jockey_win'] = df.groupby('jockey')['jockey_prev_fp'].transform(
    lambda x: x.eq(1).rolling(30, min_periods=1).mean())
df['jockey_place'] = df.groupby('jockey')['jockey_prev_fp'].transform(
    lambda x: x.isin([1,2,3]).rolling(30, min_periods=1).mean())
df.drop('jockey_prev_fp', axis=1, inplace=True)

# 4. Trainer win rate (last 30 runs) — fast
df = df.sort_values(['trainer', 'race_date']).reset_index(drop=True)
df['trainer_prev_fp'] = df.groupby('trainer')['finishing_position'].shift(1)
df['trainer_win'] = df.groupby('trainer')['trainer_prev_fp'].transform(
    lambda x: x.eq(1).rolling(30, min_periods=1).mean())
df['trainer_place'] = df.groupby('trainer')['trainer_prev_fp'].transform(
    lambda x: x.isin([1,2,3]).rolling(30, min_periods=1).mean())
df.drop('trainer_prev_fp', axis=1, inplace=True)

# 5. Field size, draw normalized
df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
df['draw_norm'] = df['draw'] / df['field_size'].replace(0, 1)

# 6. First run
df['first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

# 7. Rating placeholder
df['rating'] = 0.0

# Targets
df['target_win']   = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1,2,3]).astype(int)

# ── Encode categoricals ───────────────────────────────────────────────────────
class_map = {'Class 5': 1, 'Class 4': 2, 'Class 3': 3, 'Class 2': 4, 'Class 1': 5, 'Pattern Race': 6}
df['class_enc'] = df['race_class'].fillna('').map(class_map).fillna(0)

track_enc = LabelEncoder()
track_enc.fit(['Turf', 'Dirt', 'AWT'])
df['track_enc'] = track_enc.transform(df['track'].fillna('Turf'))

ven_enc = LabelEncoder()
ven_enc.fit(['HV', 'ST'])
df['venue_enc'] = ven_enc.transform(df['venue'].fillna('ST'))

jock_enc = LabelEncoder()
jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))

trainer_enc = LabelEncoder()
trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

# ── Feature columns (NO implied_prob!) ─────────────────────────────────────────
FEATURE_COLS = [
    'draw', 'draw_norm', 'rating',
    'horse_win_5', 'horse_place_5', 'horse_avg_fp_5',
    'days_since',
    'jockey_win', 'jockey_place',
    'trainer_win', 'trainer_place',
    'field_size', 'first_run',
    'class_enc', 'track_enc', 'venue_enc', 'distance',
]

log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# ── Temporal split ─────────────────────────────────────────────────────────────
cutoff_train = pd.Timestamp('2024-04-01')
cutoff_test  = pd.Timestamp('2026-01-01')
df_train = df[df['race_date'] <= cutoff_train].copy()
df_test  = df[(df['race_date'] > cutoff_train) & (df['race_date'] <= cutoff_test)].copy()

log.info(f"Train: {cutoff_train.date()}–{df_train.race_date.max().date()} ({len(df_train):,} rows)  "
         f"Test: {cutoff_train.date()}–{df_test.race_date.max().date()} ({len(df_test):,} rows)")

X_train = df_train[FEATURE_COLS].astype(float).values
X_test  = df_test[FEATURE_COLS].astype(float).values

# ── Train XGBoost ─────────────────────────────────────────────────────────────
log.info("Training XGBoost...")

models = {}
calibrators = {}

for target_name, y_col in [('win', 'target_win'), ('place', 'target_place')]:
    y_train = df_train[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    raw_preds = model.predict_proba(X_test)[:, 1]
    train_raw = model.predict_proba(X_train)[:, 1]

    # Isotonic calibration
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(train_raw, y_train)
    calibrated_preds = iso_reg.transform(raw_preds)

    raw_auc  = roc_auc_score(y_test, raw_preds)
    cal_auc  = roc_auc_score(y_test, calibrated_preds)

    log.info(f"  {target_name}: raw AUC={raw_auc:.4f}  cal AUC={cal_auc:.4f}")

    models[target_name] = model
    calibrators[target_name] = iso_reg

# ── Save ─────────────────────────────────────────────────────────────────────
encoders = {
    'venue':    ven_enc,
    'track':    track_enc,
    'jockey':   jock_enc,
    'trainer':  trainer_enc,
}

package = {
    'models':       models,
    'calibrators':  calibrators,
    'encoders':     encoders,
    'feature_cols': FEATURE_COLS,
}

with open(OUT, 'wb') as f:
    pickle.dump(package, f)

meta = {
    'version':       'v3',
    'trained_on':    f"{df_train.race_date.min().date()} – {df_train.race_date.max().date()}",
    'train_cutoff':  str(cutoff_train.date()),
    'test_cutoff':   str(cutoff_test.date()),
    'train_rows':    len(df_train),
    'test_rows':     len(df_test),
    'feature_count': len(FEATURE_COLS),
    'features':      FEATURE_COLS,
    'metrics': {}
}
for name in ['win', 'place']:
    y_t = df_test[f'target_{name}'].values.astype(int)
    raw_p = models[name].predict_proba(X_test)[:, 1]
    cal_p = calibrators[name].transform(raw_p)
    meta['metrics'][name] = {
        'raw_auc': round(roc_auc_score(y_t, raw_p), 4),
        'cal_auc': round(roc_auc_score(y_t, cal_p), 4),
    }

meta_path = OUT.with_suffix('.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)

log.info(f"Saved to {OUT}")
