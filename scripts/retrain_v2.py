#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 2
====================================
Improvements over v1:
- win_odds → implied_prob as a feature (available for 98% of entries)
  For upcoming races without odds, falls back to historical mean (smoothed)
- days_since_last_run (freshness)
- jockey×trainer combo win rate
- horse×distance win rate (specialist at this trip)
- horse×track win rate (Turf/Dirt specialist)
- class_drop indicator (moved down in class = easier field)
- larger model capacity (max_depth=8, 500 trees)
- probability calibration via isotonic regression
- temporal split: train ≤ 2022-03-06, test ≥ 2022-03-09
"""
import pickle, sqlite3, sys, logging, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('retrain_v2')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.models import NoOddsModel

DB  = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model_v2.pkl'

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
df['draw']               = pd.to_numeric(df['draw'], errors='coerce')
df['win_odds']           = pd.to_numeric(df['win_odds'], errors='coerce')
df['place_odds']         = pd.to_numeric(df['place_odds'], errors='coerce')
df['distance']           = pd.to_numeric(df['distance'], errors='coerce')
df['race_date']          = pd.to_datetime(df['race_date'])

log.info(f"Loaded {len(df):,} entries, {df['race_date'].min().date()} – {df['race_date'].max().date()}")

# ── Engineer features ─────────────────────────────────────────────────────────
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# 1. win_odds → implied probability
df['implied_prob'] = 1 / df['win_odds'].replace(0, np.nan)
global_implied_mean = df['implied_prob'].mean()
log.info(f"Global implied_prob mean: {global_implied_mean:.4f}")
df['implied_prob'] = df['implied_prob'].fillna(global_implied_mean)

# 2. Horse rolling features (last 5 races)
df['horse_avg_fin']      = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['horse_win_rate_5']  = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_place_rate_5'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(5, min_periods=1).mean())
df['horse_top4_rate_5'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(5, min_periods=1).mean())

# 3. Horse rolling features (last 10 races — more stable)
df['horse_win_rate_10']  = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(10, min_periods=1).mean())
df['horse_place_rate_10'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(10, min_periods=1).mean())

# 4. Days since last run
df['days_since_last'] = df.groupby('horse_name')['race_date'].transform(
    lambda x: (x - x.shift(1)).dt.days
)
# Fill first-time runners with median
median_days = df['days_since_last'].median()
df['days_since_last'] = df['days_since_last'].fillna(median_days)

# 5. Jockey features (last 20 rides)
df = df.sort_values(['jockey', 'race_date', 'venue']).reset_index(drop=True)
df['jockey_win_rate']   = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean())
df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean())
df['jockey_top4_rate']  = df.groupby('jockey')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean())
# Jockey races this season (more rides = active)
df['jockey_race_count'] = df.groupby('jockey')['race_date'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).count())

# 6. Trainer features (last 20 runs)
df = df.sort_values(['trainer', 'race_date', 'venue']).reset_index(drop=True)
df['trainer_win_rate']   = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(20, min_periods=1).mean())
df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(20, min_periods=1).mean())
df['trainer_top4_rate']  = df.groupby('trainer')['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3,4]).rolling(20, min_periods=1).mean())
df['trainer_race_count'] = df.groupby('trainer')['race_date'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).count())

# 7. Jockey × Trainer combo win rate (stored as horse×jockey×trainer combo)
log.info("Computing jockey×trainer combo rates...")
df = df.sort_values(['jockey', 'trainer', 'race_date']).reset_index(drop=True)
df['jt_combo_win_rate'] = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(30, min_periods=1).mean())
df['jt_combo_place_rate'] = df.groupby(['jockey', 'trainer'])['finishing_position'].transform(
    lambda x: x.shift(1).isin([1,2,3]).rolling(30, min_periods=1).mean())

# 8. Horse × Distance specialist win rate
df = df.sort_values(['horse_name', 'distance', 'race_date']).reset_index(drop=True)
df['horse_dist_win_rate'] = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_dist_count']    = df.groupby(['horse_name', 'distance'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

# 9. Horse × Track specialist
df = df.sort_values(['horse_name', 'track', 'race_date']).reset_index(drop=True)
df['horse_track_win_rate'] = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).eq(1).rolling(5, min_periods=1).mean())
df['horse_track_count']   = df.groupby(['horse_name', 'track'])['finishing_position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).count())

# 10. Class drop: horse previously in higher class
df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
class_order = {'': 0, 'Class 5': 1, 'Class 4': 2, 'Class 3': 3, 'Class 2': 4, 'Class 1': 5,
                'Group-Three': 6, 'Group-Two': 7, 'Group-One': 8, 'Listed': 5, 'Pattern Race': 6}
race_class_map = {
    'Group-One': 'Pattern Race', 'Group-Two': 'Pattern Race', 'Group-Three': 'Pattern Race',
    'Listed': 'Pattern Race',
}
df['race_class_std'] = df['race_class'].fillna('').replace(race_class_map)
df['race_class_level'] = df['race_class_std'].map(class_order).fillna(0)
df['prev_class_level'] = df.groupby('horse_name')['race_class_level'].shift(1)
df['class_drop'] = ((df['prev_class_level'] > df['race_class_level']) &
                     df['prev_class_level'].notna()).astype(int)
# Positive class_drop = moved to easier class

# 11. Race context
df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
df['draw_normalized'] = df['draw'] / df['field_size'].replace(0, 1)  # draw 1 of 14 vs draw 1 of 8
df['prev_distance'] = df.groupby('horse_name')['distance'].shift(1)
df['is_different_distance'] = (df['distance'] != df['prev_distance']).astype(int)

# 12. First-time runner indicator
df['is_first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

# 13. Recent win/loss streak
df['horse_last_result'] = df.groupby('horse_name')['finishing_position'].transform(
    lambda x: x.shift(1).fillna(0).astype(int))
df['horse_streak'] = df.groupby('horse_name')['horse_last_result'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
)
# Note: streak direction depends on whether last result was win or loss
# We'll encode as: positive = consecutive places, negative = consecutive losses

# Targets
df['target_win']   = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1,2,3]).astype(int)
df['target_top4']  = df['finishing_position'].isin([1,2,3,4]).astype(int)

# ── Encode categoricals ───────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

# race_class
rc_enc = LabelEncoder()
rc_enc.fit(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Pattern Race', ''])
df['race_class_enc'] = rc_enc.transform(df['race_class_std'])

# track
trk_enc = LabelEncoder()
trk_enc.fit(['Turf', 'Dirt', 'AWT'])
df['track_enc'] = trk_enc.transform(df['track'].fillna('Turf'))

# venue
ven_enc = LabelEncoder()
ven_enc.fit(['HV', 'ST'])
df['venue_enc'] = ven_enc.transform(df['venue'].fillna('ST'))

# jockey
jock_enc = LabelEncoder()
jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))

# trainer
trainer_enc = LabelEncoder()
trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

# ── Feature columns ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'draw', 'draw_normalized',
    'implied_prob',              # key new feature — strong predictor
    'horse_avg_fin', 'horse_win_rate_5', 'horse_place_rate_5', 'horse_top4_rate_5',
    'horse_win_rate_10', 'horse_place_rate_10',
    'days_since_last',
    'jockey_win_rate', 'jockey_place_rate', 'jockey_top4_rate', 'jockey_race_count',
    'trainer_win_rate', 'trainer_place_rate', 'trainer_top4_rate', 'trainer_race_count',
    'jt_combo_win_rate', 'jt_combo_place_rate',
    'horse_dist_win_rate', 'horse_dist_count',
    'horse_track_win_rate', 'horse_track_count',
    'class_drop',
    'field_size', 'is_different_distance', 'distance',
    'race_class_enc', 'track_enc', 'venue_enc',
    'is_first_run',
]

log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# Fill NaN
for col in FEATURE_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# ── Temporal train/test split ─────────────────────────────────────────────────
cutoff = pd.Timestamp('2022-03-06')
df_train = df[df['race_date'] <= cutoff].copy()
df_test  = df[df['race_date'] >  cutoff].copy()

log.info(f"Cutoff: {cutoff.date()}  "
         f"Train: {len(df_train):,} ({df_train.race_date.min().date()} – {df_train.race_date.max().date()})  "
         f"Test:  {len(df_test):,} ({df_test.race_date.min().date()} – {df_test.race_date.max().date()})")

X_train = df_train[FEATURE_COLS].astype(float).values
X_test  = df_test[FEATURE_COLS].astype(float).values

# ── Train XGBoost (larger capacity) ───────────────────────────────────────────
log.info("Training XGBoost models...")

models = {}
calibrators = {}  # for isotonic calibration

for target_name, y_col in [('win', 'target_win'), ('place', 'target_place'), ('top4', 'target_top4')]:
    y_train = df_train[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    # XGBoost — larger model
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Raw predictions
    raw_preds = model.predict_proba(X_test)[:, 1]

    # Isotonic calibration (fits on train predictions, transforms test)
    # Fit calibrator on train set predictions
    train_raw = model.predict_proba(X_train)[:, 1]
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(train_raw, y_train)

    # Apply calibration
    calibrated_preds = iso_reg.transform(raw_preds)

    raw_auc  = roc_auc_score(y_test, raw_preds)
    cal_auc  = roc_auc_score(y_test, calibrated_preds)
    raw_brier = brier_score_loss(y_test, raw_preds)
    cal_brier = brier_score_loss(y_test, calibrated_preds)

    log.info(f"  {target_name}: raw AUC={raw_auc:.4f} Brier={raw_brier:.4f}  "
             f"cal AUC={cal_auc:.4f} Brier={cal_brier:.4f}")

    models[target_name] = model
    calibrators[target_name] = iso_reg

# ── Save ─────────────────────────────────────────────────────────────────────
encoders = {
    'venue':    ven_enc,
    'race_class': rc_enc,
    'track':    trk_enc,
    'jockey':   jock_enc,
    'trainer':  trainer_enc,
}

# Pack everything
package = {
    'models':       models,
    'calibrators':  calibrators,
    'encoders':     encoders,
    'feature_cols': FEATURE_COLS,
    'global_implied_prob_mean': global_implied_mean,
    'cutoff':       str(cutoff.date()),
}

with open(OUT, 'wb') as f:
    pickle.dump(package, f)

# Metadata
meta = {
    'version':      'v2',
    'trained_on':  f"{df_train.race_date.min().date()} – {df_train.race_date.max().date()}",
    'test_from':    str(cutoff.date()),
    'train_rows':   len(df_train),
    'test_rows':    len(df_test),
    'feature_count': len(FEATURE_COLS),
    'features':     FEATURE_COLS,
    'metrics': {}
}
for name in ['win', 'place', 'top4']:
    y_t = df_test[f'target_{name}'].values.astype(int)
    raw_p  = models[name].predict_proba(X_test)[:, 1]
    cal_p  = calibrators[name].transform(raw_p)
    meta['metrics'][name] = {
        'raw_auc':   round(roc_auc_score(y_t, raw_p), 4),
        'raw_brier': round(brier_score_loss(y_t, raw_p), 4),
        'cal_auc':   round(roc_auc_score(y_t, cal_p), 4),
        'cal_brier': round(brier_score_loss(y_t, cal_p), 4),
    }

meta_path = OUT.with_suffix('.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)

log.info(f"Model saved to {OUT}")
log.info(f"Metadata saved to {meta_path}")
