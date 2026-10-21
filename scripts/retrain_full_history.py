#!/usr/bin/env python3
"""
Retrain XGBoost model on ALL available historical data (2006–2026).
Uses NoOddsModel (no win_odds features) for predicting upcoming races.
Saves to models/model.pkl
"""
import pickle, sqlite3, sys, logging, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('retrain')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.models import NoOddsModel

DB  = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model.pkl'

# ── Load ALL historical data ──────────────────────────────────────────────────
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
      AND re.finishing_position GLOB '[0-9]*'
      AND CAST(re.finishing_position AS INTEGER) > 0
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
''', conn)
conn.close()

# Convert numeric columns
df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce').fillna(0).astype(int)
df['draw'] = pd.to_numeric(df['draw'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

log.info(f"Loaded {len(df):,} entries from DB")
df['race_date'] = pd.to_datetime(df['race_date'])

# ── Engineer features ──────────────────────────────────────────────────────────
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# Horse rolling features (last 5 races)
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

# Jockey features (last 20 rides)
df = df.sort_values(['jockey', 'race_date', 'venue']).reset_index(drop=True)
df['jockey_win_rate']   = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean()
)
df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1, 2, 3]).rolling(20, min_periods=1).mean()
)

# Trainer features (last 20 runs)
df = df.sort_values(['trainer', 'race_date', 'venue']).reset_index(drop=True)
df['trainer_win_rate']   = df.groupby('trainer')['finishing_position'].transform(
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
df['target_place'] = df['finishing_position'].isin([1, 2, 3]).astype(int)
df['target_top4']  = df['finishing_position'].isin([1, 2, 3, 4]).astype(int)

# ── Encode categoricals ────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

# race_class: normalize Group/Listed
race_class_map = {
    'Group-One': '', 'Group-Two': '', 'Group-Three': '',
    'Listed': '', 'Pattern Race': '',
}
df['race_class_clean'] = df['race_class'].map(race_class_map).fillna('')
rc_enc = LabelEncoder()
rc_enc.fit(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', ''])
df['race_class_enc'] = rc_enc.transform(df['race_class_clean'])

# track
df['track_clean'] = df['track'].fillna('Turf')
trk_enc = LabelEncoder()
trk_enc.fit(['Turf', 'Dirt', 'AWT'])
df['track_enc'] = trk_enc.transform(df['track_clean'])

# venue
ven_enc = LabelEncoder()
ven_enc.fit(['HV', 'ST'])
df['venue_enc'] = ven_enc.transform(df['venue'].fillna('ST'))

# jockey / trainer
jock_enc = LabelEncoder()
jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))

trainer_enc = LabelEncoder()
trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

# Fill NaN in numeric cols
for col in ['draw', 'rating', 'days_since_last', 'horse_avg_finishing_position',
            'horse_win_rate', 'horse_place_rate', 'jockey_win_rate', 'jockey_place_rate',
            'trainer_win_rate', 'trainer_place_rate']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

log.info(f"Features engineered. Shape: {df.shape}")
log.info(f"Rating non-null: {df['rating'].notna().sum():,} / {len(df):,}")
log.info(f"Draw non-null:   {df['draw'].notna().sum():,} / {len(df):,}")

# ── Temporal train/test split (80/20 by date) ──────────────────────────────────
df = df.sort_values('race_date').reset_index(drop=True)
cutoff = df['race_date'].quantile(0.80)
df_train = df[df['race_date'] < cutoff].copy()
df_test  = df[df['race_date'] >= cutoff].copy()

log.info(f"Split date: {pd.Timestamp(cutoff).date()}  "
         f"Train: {len(df_train):,} ({df_train.race_date.min().date()} – {df_train.race_date.max().date()})  "
         f"Test: {len(df_test):,} ({df_test.race_date.min().date()} – {df_test.race_date.max().date()})")

# ── Feature columns ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'draw',
    'rating',
    'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
    'jockey_win_rate', 'jockey_place_rate',
    'trainer_win_rate', 'trainer_place_rate',
    'field_size', 'is_different_distance', 'distance',
    'race_class_enc', 'track_enc', 'jockey_enc', 'trainer_enc',
]
log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X_train = df_train[FEATURE_COLS].astype(float).fillna(0).values
X_test  = df_test[FEATURE_COLS].astype(float).fillna(0).values

# ── Train XGBoost ──────────────────────────────────────────────────────────────
log.info("Training XGBoost models...")

models = {}
for target_name, y_col in [('win', 'target_win'), ('place', 'target_place'), ('top4', 'target_top4')]:
    y_train = df_train[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
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
    brier = brier_score_loss(y_test, preds)
    log.info(f"  {target_name}: ROC AUC = {auc:.4f}  Brier = {brier:.4f}")
    models[target_name] = model

# ── Save ──────────────────────────────────────────────────────────────────────
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

# Also save metadata JSON
meta = {
    'trained_on': '2006-2026',
    'split_date': str(pd.Timestamp(cutoff).date()),
    'train_rows': len(df_train),
    'test_rows': len(df_test),
    'feature_cols': FEATURE_COLS,
    'metrics': {}
}
for name, m in models.items():
    preds = m.predict_proba(X_test)[:, 1]
    y_t = df_test[f'target_{name}'].values.astype(int)
    meta['metrics'][name] = {
        'roc_auc': round(roc_auc_score(y_t, preds), 4),
        'brier': round(brier_score_loss(y_t, preds), 4),
    }

meta_path = Path(__file__).parent.parent / 'models' / 'model_meta.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)

log.info(f"Model saved to {OUT}")
log.info(f"Meta saved to {meta_path}")
