#!/usr/bin/env python3
"""
Retrain XGBoost model WITHOUT win_odds/implied_prob features.
Usage: python scripts/retrain_no_odds.py
"""
import pickle, sqlite3, sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.models import NoOddsModel

# ── Config ──────────────────────────────────────────────────────────────────
DB  = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model.pkl'

# Features to USE (excludes win_odds, implied_prob)
NEW_FEATURE_COLS = [
    'draw', 'rating',
    'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
    'jockey_win_rate', 'jockey_place_rate',
    'trainer_win_rate', 'trainer_place_rate',
    'field_size', 'is_different_distance', 'distance',
    'race_class_enc', 'track_enc', 'jockey_enc', 'trainer_enc',
]

# ── Load data ────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.rating,
           re.jockey, re.trainer, re.finishing_position,
           rd.race_date, rd.venue,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE re.finishing_position IS NOT NULL
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
''', conn)
conn.close()

log.info(f"Loaded {len(df):,} entries from DB")
df['race_date'] = pd.to_datetime(df['race_date'])
df['distance']   = pd.to_numeric(df['distance'], errors='coerce')

# ── Engineer features ────────────────────────────────────────────────────────
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# Horse rolling features
df['horse_avg_finishing_position'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['horse_win_rate']  = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean()
)
df['horse_place_rate'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1, 2, 3]).rolling(5, min_periods=1).mean()
)

# Days since last run
df['days_since_last'] = df.groupby('horse_name')['race_date'].transform(
    lambda x: (x - x.shift(1)).dt.days
)

# Jockey features
df = df.sort_values(['jockey', 'race_date', 'venue']).reset_index(drop=True)
df['jockey_win_rate']  = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean()
)
df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1, 2, 3]).rolling(20, min_periods=1).mean()
)

# Trainer features
df = df.sort_values(['trainer', 'race_date', 'venue']).reset_index(drop=True)
df['trainer_win_rate']  = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean()
)
df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1, 2, 3]).rolling(20, min_periods=1).mean()
)

# Race context
df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
df['prev_distance'] = df.groupby('horse_name')['distance'].shift(1)
df['is_different_distance'] = (df['distance'] != df['prev_distance']).astype(int)

# Target variables
df['target_win']   = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1, 2]).astype(int)
df['target_top4']  = df['finishing_position'].isin([1, 2, 3, 4]).astype(int)

# Encode race_class: Group/Listed → ''
from sklearn.preprocessing import LabelEncoder
race_class_map = {
    'Group-One': '', 'Group-Two': '', 'Group-Three': '',
    'Listed': '', 'Pattern Race': '',
}
df['race_class_clean'] = df['race_class'].map(race_class_map).fillna('')
rc_enc = LabelEncoder()
rc_enc.fit(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', ''])
df['race_class_enc'] = rc_enc.transform(df['race_class_clean'])

# Encode track
df['track_clean'] = df['track'].fillna('Turf')
trk_enc = LabelEncoder()
trk_enc.fit(['Turf', 'Dirt', 'AWT'])
df['track_enc'] = trk_enc.transform(df['track_clean'])

# Encode venue
ven_enc = LabelEncoder()
ven_enc.fit(['HV', 'ST'])
df['venue_enc'] = ven_enc.transform(df['venue'].fillna('ST'))

# Encode jockey / trainer
jock_enc = LabelEncoder()
jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))

trainer_enc = LabelEncoder()
trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

log.info(f"Features engineered. Rating non-null: {df['rating'].notna().sum():,} / {len(df):,}")

# ── Temporal train/test split ────────────────────────────────────────────────
df = df.sort_values('race_date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
df_train  = df.iloc[:split_idx].copy()
df_test   = df.iloc[split_idx:].copy()
split_date = df_train['race_date'].max().date()
log.info(f"Train: {len(df_train):,} | Test: {len(df_test):,} | Split date: {split_date}")

X_train = df_train[NEW_FEATURE_COLS].values
X_test  = df_test[NEW_FEATURE_COLS].values

# ── Train XGBoost multi-output ──────────────────────────────────────────────
log.info("Training XGBoost models (no odds)...")

models = {}
for target_name, y_col in [('win', 'target_win'), ('place', 'target_place'), ('top4', 'target_top4')]:
    y_train = df_train[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    log.info(f"  {target_name}: ROC AUC = {auc:.4f}")
    models[target_name] = model

# ── Save ─────────────────────────────────────────────────────────────────────
encoders = {
    'venue': ven_enc,
    'race_class': rc_enc,
    'track': trk_enc,
    'jockey': jock_enc,
    'trainer': trainer_enc,
}

horse_model = NoOddsModel(models, encoders)
with open(OUT, 'wb') as f:
    pickle.dump(horse_model, f)

log.info(f"Model saved to {OUT}")
log.info(f"Feature cols ({len(NEW_FEATURE_COLS)}): {NEW_FEATURE_COLS}")
