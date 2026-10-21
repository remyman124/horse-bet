#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 5 (FAST)
==========================================
Only essential features, optimized for speed.
Uses rating + form features only.
"""
import pickle, sqlite3, json, logging, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('retrain_v5f')

DB   = Path('data/horse.db')
OUT  = Path('models/model_v5.pkl')

conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.rating,
           re.jockey, re.trainer,
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
df['draw'] = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['race_date'] = pd.to_datetime(df['race_date'])

log.info(f"Loaded {len(df):,} entries")

# ── Fast feature engineering ──────────────────────────────────────────────────
df = df.sort_values(['horse_name','race_date']).reset_index(drop=True)

# Horse prev finishing position
df['prev_fp'] = df.groupby('horse_name')['finishing_position'].shift(1)
# Horse win rate last 5
df['horse_win5'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(5, min_periods=1).mean())
df['horse_place5'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(5, min_periods=1).mean())
df.drop('prev_fp', axis=1, inplace=True)

# Days since
df['prev_date'] = df.groupby('horse_name')['race_date'].shift(1)
df['days_since'] = (df['race_date'] - df['prev_date']).dt.days.fillna(90)
df.drop('prev_date', axis=1, inplace=True)

# Jockey
df = df.sort_values(['jockey','race_date']).reset_index(drop=True)
df['jp'] = df.groupby('jockey')['finishing_position'].shift(1)
df['jockey_win'] = df.groupby('jockey')['jp'].transform(lambda x: x.eq(1).rolling(20, min_periods=1).mean())
df.drop('jp', axis=1, inplace=True)

# Trainer
df = df.sort_values(['trainer','race_date']).reset_index(drop=True)
df['tp'] = df.groupby('trainer')['finishing_position'].shift(1)
df['trainer_win'] = df.groupby('trainer')['tp'].transform(lambda x: x.eq(1).rolling(20, min_periods=1).mean())
df.drop('tp', axis=1, inplace=True)

# Field size
df['field_size'] = df.groupby(['race_date','venue','race_no'])['horse_no'].transform('count')
df['draw_norm'] = df['draw'] / df['field_size'].replace(0, 1)

# First run
df['first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

# Target
df['target_win'] = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1,2,3]).astype(int)

# Encode
from sklearn.preprocessing import LabelEncoder
class_map = {'Class 5':1,'Class 4':2,'Class 3':3,'Class 2':4,'Class 1':5,'Pattern Race':6,'':0}
df['class_enc'] = df['race_class'].fillna('').map(class_map).fillna(0)
track_enc = LabelEncoder(); track_enc.fit(['Turf','Dirt','AWT'])
df['track_enc'] = track_enc.transform(df['track'].fillna('Turf'))
venue_enc = LabelEncoder(); venue_enc.fit(['HV','ST'])
df['venue_enc'] = venue_enc.transform(df['venue'].fillna('ST'))
jock_enc = LabelEncoder(); jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))
trainer_enc = LabelEncoder(); trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

FEATURE_COLS = [
    'draw', 'draw_norm', 'rating',
    'horse_win5', 'horse_place5', 'days_since',
    'jockey_win', 'trainer_win',
    'field_size', 'first_run',
    'class_enc', 'track_enc', 'venue_enc', 'distance',
]

df = df.fillna(0)
log.info(f"Features: {FEATURE_COLS}")

# Split
cutoff_train = pd.Timestamp('2025-06-01')
cutoff_val   = pd.Timestamp('2025-10-01')
df_train = df[df['race_date'] <= cutoff_train].copy()
df_val   = df[(df['race_date'] > cutoff_train) & (df['race_date'] <= cutoff_val)].copy()
df_test  = df[df['race_date'] > cutoff_val].copy()

log.info(f"Train: {len(df_train):,} Val: {len(df_val):,} Test: {len(df_test):,}")

X_train = df_train[FEATURE_COLS].astype(float).values
X_val   = df_val[FEATURE_COLS].astype(float).values
X_test  = df_test[FEATURE_COLS].astype(float).values

# Train XGBoost (win only - fastest)
log.info("Training XGBoost win model...")
y_train = df_train['target_win'].values.astype(int)
y_val   = df_val['target_win'].values.astype(int)
y_test  = df_test['target_win'].values.astype(int)

xgb_model = xgb.XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', early_stopping_rounds=15,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val  = xgb_model.predict_proba(X_val)[:,1]
xgb_test = xgb_model.predict_proba(X_test)[:,1]
log.info(f"  XGB win: val={roc_auc_score(y_val, xgb_val):.4f} test={roc_auc_score(y_test, xgb_test):.4f}")

# Train RF
log.info("Training RandomForest...")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_val  = rf_model.predict_proba(X_val)[:,1]
rf_test = rf_model.predict_proba(X_test)[:,1]
log.info(f"  RF win: val={roc_auc_score(y_val, rf_val):.4f} test={roc_auc_score(y_test, rf_test):.4f}")

# Ensemble
ens_test = (xgb_test + rf_test) / 2
log.info(f"  Ensemble: test AUC={roc_auc_score(y_test, ens_test):.4f}")

# Save
package = {
    'xgb_model': xgb_model, 'rf_model': rf_model,
    'feature_cols': FEATURE_COLS,
    'encoders': {'track': track_enc, 'venue': venue_enc, 'jockey': jock_enc, 'trainer': trainer_enc}
}
with open(OUT, 'wb') as f:
    pickle.dump(package, f)

log.info(f"Saved to {OUT}")
