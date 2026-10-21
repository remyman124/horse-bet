#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 4
====================================
Key insight from v3 analysis:
- win_odds in DB are POST-RACE payouts (888x, 54x etc.) not pre-race odds
- Training used these as implied_prob feature → model learned wrong signal
- v4: REMOVES ALL odds data. Uses ONLY pre-race features:
  horse form, jockey/trainer stats, draw, class, distance, track
- Multiple model architectures: XGB, RF, LR
- Backtests on Apr 22, Apr 29, May 6 2026
"""
import pickle, sqlite3, json, logging, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
log = logging.getLogger('retrain_v4')

DB  = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model_v4.pkl'

# ── Load data ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw,
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
df['draw']               = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['distance']           = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['race_date']          = pd.to_datetime(df['race_date'])

log.info(f"Loaded {len(df):,} entries, {df['race_date'].min().date()} – {df['race_date'].max().date()}")

# ── Engineer features (ONLY pre-race info — NO odds!) ──────────────────────────
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# Horse rolling features (past races only, shift=1 to avoid leakage)
df['prev_fp'] = df.groupby('horse_name')['finishing_position'].shift(1)

for n in [3, 5, 10]:
    df[f'horse_win_{n}']    = df.groupby('horse_name')['prev_fp'].transform(
        lambda x: x.eq(1).rolling(n, min_periods=1).mean())
    df[f'horse_place_{n}'] = df.groupby('horse_name')['prev_fp'].transform(
        lambda x: x.isin([1,2,3]).rolling(n, min_periods=1).mean())
    df[f'horse_avg_fp_{n}'] = df.groupby('horse_name')['prev_fp'].transform(
        lambda x: x.rolling(n, min_periods=1).mean())

df.drop('prev_fp', axis=1, inplace=True)

# Days since last run
df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
df['prev_date'] = df.groupby('horse_name')['race_date'].shift(1)
df['days_since'] = (df['race_date'] - df['prev_date']).dt.days.fillna(90)
df.drop('prev_date', axis=1, inplace=True)

# Jockey features
df = df.sort_values(['jockey', 'race_date']).reset_index(drop=True)
df['jockey_prev'] = df.groupby('jockey')['finishing_position'].shift(1)
for n in [10, 30]:
    df[f'jockey_win_{n}']    = df.groupby('jockey')['jockey_prev'].transform(
        lambda x: x.eq(1).rolling(n, min_periods=1).mean())
    df[f'jockey_place_{n}']  = df.groupby('jockey')['jockey_prev'].transform(
        lambda x: x.isin([1,2,3]).rolling(n, min_periods=1).mean())
df.drop('jockey_prev', axis=1, inplace=True)

# Trainer features
df = df.sort_values(['trainer', 'race_date']).reset_index(drop=True)
df['trainer_prev'] = df.groupby('trainer')['finishing_position'].shift(1)
for n in [10, 30]:
    df[f'trainer_win_{n}']    = df.groupby('trainer')['trainer_prev'].transform(
        lambda x: x.eq(1).rolling(n, min_periods=1).mean())
    df[f'trainer_place_{n}']  = df.groupby('trainer')['trainer_prev'].transform(
        lambda x: x.isin([1,2,3]).rolling(n, min_periods=1).mean())
df.drop('trainer_prev', axis=1, inplace=True)

# Field size, draw normalized
df['field_size'] = df.groupby(['race_date','venue','race_no'])['horse_no'].transform('count')
df['draw_norm'] = df['draw'] / df['field_size'].replace(0, 1)

# First run
df['first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

# Targets
df['target_win']   = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1,2,3]).astype(int)

# Encode categoricals
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

# Feature columns — NO odds, NO implied_prob
FEATURE_COLS = [
    'draw', 'draw_norm',
    'horse_win_3','horse_place_3','horse_avg_fp_3',
    'horse_win_5','horse_place_5','horse_avg_fp_5',
    'horse_win_10','horse_place_10','horse_avg_fp_10',
    'days_since',
    'jockey_win_10','jockey_place_10',
    'jockey_win_30','jockey_place_30',
    'trainer_win_10','trainer_place_10',
    'trainer_win_30','trainer_place_30',
    'field_size','first_run',
    'class_enc','track_enc','venue_enc','distance',
]

df = df.fillna(0)
log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── Train/val/test split ──────────────────────────────────────────────────────
cutoff_train = pd.Timestamp('2025-06-01')
cutoff_val   = pd.Timestamp('2025-10-01')

df_train = df[df['race_date'] <= cutoff_train].copy()
df_val   = df[(df['race_date'] > cutoff_train) & (df['race_date'] <= cutoff_val)].copy()
df_test  = df[df['race_date'] > cutoff_val].copy()

log.info(f"Train: {df_train.race_date.min().date()}–{df_train.race_date.max().date()} ({len(df_train):,})")
log.info(f"Val:   {df_val.race_date.min().date()}–{df_val.race_date.max().date()} ({len(df_val):,})")
log.info(f"Test:  {df_test.race_date.min().date()}–{df_test.race_date.max().date()} ({len(df_test):,})")

X_train = df_train[FEATURE_COLS].astype(float).values
X_val   = df_val[FEATURE_COLS].astype(float).values
X_test  = df_test[FEATURE_COLS].astype(float).values

best_models = {}
all_results = {}

for target_name, y_col in [('win','target_win'),('place','target_place')]:
    y_train = df_train[y_col].values.astype(int)
    y_val   = df_val[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    best_val_auc = 0
    best_name = None
    best_model = None

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric='auc', early_stopping_rounds=20,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val  = xgb_model.predict_proba(X_val)[:,1]
    xgb_test = xgb_model.predict_proba(X_test)[:,1]
    xgb_vauc = roc_auc_score(y_val, xgb_val)
    xgb_tauc = roc_auc_score(y_test, xgb_test)
    log.info(f"  XGB {target_name}: val={xgb_vauc:.4f} test={xgb_tauc:.4f}")
    all_results[f'xgb_{target_name}'] = {'val': xgb_vauc, 'test': xgb_tauc, 'model': xgb_model}
    if xgb_vauc > best_val_auc:
        best_val_auc = xgb_vauc; best_name = f'xgb_{target_name}'; best_model = xgb_model

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_val   = rf_model.predict_proba(X_val)[:,1]
    rf_test  = rf_model.predict_proba(X_test)[:,1]
    rf_vauc  = roc_auc_score(y_val, rf_val)
    rf_tauc  = roc_auc_score(y_test, rf_test)
    log.info(f"  RF  {target_name}: val={rf_vauc:.4f} test={rf_tauc:.4f}")
    all_results[f'rf_{target_name}'] = {'val': rf_vauc, 'test': rf_tauc, 'model': rf_model}
    if rf_vauc > best_val_auc:
        best_val_auc = rf_vauc; best_name = f'rf_{target_name}'; best_model = rf_model

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42, C=0.1)
    lr_model.fit(X_train, y_train)
    lr_val   = lr_model.predict_proba(X_val)[:,1]
    lr_test  = lr_model.predict_proba(X_test)[:,1]
    lr_vauc  = roc_auc_score(y_val, lr_val)
    lr_tauc  = roc_auc_score(y_test, lr_test)
    log.info(f"  LR  {target_name}: val={lr_vauc:.4f} test={lr_tauc:.4f}")
    all_results[f'lr_{target_name}'] = {'val': lr_vauc, 'test': lr_tauc, 'model': lr_model}
    if lr_vauc > best_val_auc:
        best_val_auc = lr_vauc; best_name = f'lr_{target_name}'; best_model = lr_model

    log.info(f"  → Best for {target_name}: {best_name} (val AUC={best_val_auc:.4f})")
    best_models[target_name] = {'model': best_model, 'name': best_name}

# ── Ensemble: XGB + RF average ─────────────────────────────────────────────────
xgb_win  = all_results['xgb_win']['model'].predict_proba(X_test)[:,1]
rf_win   = all_results['rf_win']['model'].predict_proba(X_test)[:,1]
ens_win  = (xgb_win + rf_win) / 2
ens_win_auc = roc_auc_score(df_test['target_win'].values.astype(int), ens_win)

xgb_place  = all_results['xgb_place']['model'].predict_proba(X_test)[:,1]
rf_place   = all_results['rf_place']['model'].predict_proba(X_test)[:,1]
ens_place  = (xgb_place + rf_place) / 2
ens_place_auc = roc_auc_score(df_test['target_place'].values.astype(int), ens_place)

log.info(f"Ensemble: win AUC={ens_win_auc:.4f}  place AUC={ens_place_auc:.4f}")

# ── Backtest on specific recent race days ───────────────────────────────────────
backtest_dates = [
    ('2026-04-22', 'HV'),
    ('2026-04-29', 'HV'),
    ('2026-05-06', 'ST'),
]

log.info("\n=== Backtest: Win hits / Top-3 hits ===")
for date, venue in backtest_dates:
    bt_df = df[(df['race_date'] == date) & (df['venue'] == venue)].copy()
    if len(bt_df) == 0:
        log.info(f"  {date} {venue}: no data")
        continue

    X_bt = bt_df[FEATURE_COLS].astype(float).values
    bt_df['pred_win'] = ens_win[:len(bt_df)] if date == df_test[df_test['race_date'] > cutoff_val].race_date.min() else \
        best_models['win']['model'].predict_proba(X_bt)[:,1]

    hits_win = 0
    hits_place = 0
    races_found = bt_df['race_no'].nunique()

    for rn in sorted(bt_df['race_no'].unique()):
        race = bt_df[bt_df['race_no'] == rn].sort_values('pred_win', ascending=False)
        winner_row = race[race['target_win']==1]
        winner = winner_row['horse_no'].values[0] if len(winner_row) > 0 else -1
        top3 = race.head(3)['horse_no'].tolist()
        if winner in top3:
            hits_place += 1
        if winner == race.iloc[0]['horse_no']:
            hits_win += 1

    log.info(f"  {date} {venue}: {races_found} races | Win: {hits_win}/{races_found} | Top-3: {hits_place}/{races_found}")

# ── Save ─────────────────────────────────────────────────────────────────────
package = {
    'models': best_models,
    'feature_cols': FEATURE_COLS,
    'encoders': {
        'track': track_enc,
        'venue': venue_enc,
        'jockey': jock_enc,
        'trainer': trainer_enc,
    }
}

with open(OUT, 'wb') as f:
    pickle.dump(package, f)

meta = {
    'version': 'v4',
    'features': FEATURE_COLS,
    'feature_count': len(FEATURE_COLS),
    'val_results': {k: {'val': v['val'], 'test': v['test']} 
                   for k, v in all_results.items()},
    'best_models': {k: v['name'] for k, v in best_models.items()},
    'ensemble_win_auc': round(ens_win_auc, 4),
    'ensemble_place_auc': round(ens_place_auc, 4),
}

meta_path = OUT.with_suffix('.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, default=str)

log.info(f"\nSaved to {OUT}")
log.info(f"Meta saved to {meta_path}")
