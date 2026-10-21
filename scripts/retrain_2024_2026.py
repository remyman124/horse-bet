#!/usr/bin/env python3
"""
Retrain XGBoost model on 2024-2026 data only.
Saves to models/model.pkl (full pickle) + models/model.json + *_xgb.json
"""
import sqlite3, pandas as pd, numpy as np, pickle, json, logging, sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('retrain')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.models import engineer_features

sqlite3.register_adapter(date, lambda d: d.isoformat())
conn = sqlite3.connect('data/horse.db')

# ── Load 2024-2026 data ──────────────────────────────────────────────────────
query = """
    SELECT
        re.horse_name, re.horse_no, re.draw, re.rating, re.win_odds,
        re.finishing_position, re.jockey, re.trainer,
        r.distance, r.race_class, r.track, r.race_no,
        rd.race_date, rd.venue
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date >= '2024-01-01'
      AND re.finishing_position IS NOT NULL
      AND re.finishing_position > 0
"""
df = pd.read_sql(query, conn)
df['race_date'] = pd.to_datetime(df['race_date'])
df['distance']  = pd.to_numeric(df['distance'], errors='coerce').fillna(0).astype(int)
df['rating']     = pd.to_numeric(df['rating'], errors='coerce')
df['draw']       = pd.to_numeric(df['draw'], errors='coerce')
df['jockey']     = df['jockey'].fillna('')
df['trainer']    = df['trainer'].fillna('')

log.info(f"Raw rows: {len(df)}")
log.info(f"  Win odds NULL: {df['win_odds'].isna().sum()} / {len(df)}")
log.info(f"  Rating NULL:  {df['rating'].isna().sum()} / {len(df)}")
log.info(f"  Draw NULL:    {df['draw'].isna().sum()} / {len(df)}")

# NOTE: Rating is 0% for 2024-2025. We train WITHOUT rating feature.
# Future: re-add when enough rated historical data accumulates.
USE_RATING = False

df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
df = engineer_features(df)

# ── Train/Test split by TIME ──────────────────────────────────────────────────
df = df.sort_values('race_date')
cutoff = df['race_date'].quantile(0.80)
log.info(f"Cutoff: {cutoff.date()}  Train: {sum(df.race_date < cutoff)}  Test: {sum(df.race_date >= cutoff)}")

train_df = df[df['race_date'] < cutoff].copy()
test_df  = df[df['race_date'] >= cutoff].copy()

# ── Encode categoricals ────────────────────────────────────────────────────────
race_class_map = {
    'Group-One': '', 'Group-Two': '', 'Group-Three': '',
    'Listed': '', 'Pattern': '',
}
fallback = {'track': 'Turf', 'venue': 'ST', 'race_class': ''}

from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in ['venue', 'race_class', 'track']:
    encoders[col] = LabelEncoder()
    train_df[col] = train_df[col].fillna('Unknown').astype(str)
    encoders[col].fit(train_df[col])

FEATURE_COLS = [
    'draw', 'rating' if USE_RATING else 'draw',
    'win_odds',
    'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
    'jockey_win_rate', 'jockey_place_rate',
    'trainer_win_rate', 'trainer_place_rate',
    'field_size', 'is_different_distance', 'distance',
]
# Remove duplicate 'draw' if rating disabled
if not USE_RATING:
    FEATURE_COLS = [c for c in FEATURE_COLS if c != 'draw' or FEATURE_COLS.count('draw') == 1]
# Actually just hardcode:
FEATURE_COLS = [
    'draw',
    'win_odds',
    'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
    'jockey_win_rate', 'jockey_place_rate',
    'trainer_win_rate', 'trainer_place_rate',
    'field_size', 'is_different_distance', 'distance',
]
log.info(f"Features: {FEATURE_COLS}")

def prepare_X(df_, enc):
    df_ = df_.copy()
    if 'race_class' in df_.columns:
        df_['race_class'] = df_['race_class'].fillna('').replace(race_class_map)
    for col in ['venue', 'race_class', 'track']:
        if col in df_.columns and col in enc:
            vals = df_[col].fillna('').astype(str)
            known = set(enc[col].classes_)
            fb = fallback.get(col, '')
            vals = vals.apply(lambda x: x if x in known else fb)
            df_[col] = enc[col].transform(vals)
    df_ = df_.fillna(0)
    return df_[FEATURE_COLS].astype(float)

X_train = prepare_X(train_df, encoders)
X_test  = prepare_X(test_df,  encoders)

y_win_tr  = train_df['target_win'].astype(int)
y_place_tr = train_df['target_place'].astype(int)
y_top4_tr  = train_df['target_top4'].astype(int)
y_win_te   = test_df['target_win'].astype(int)
y_place_te = test_df['target_place'].astype(int)
y_top4_te  = test_df['target_top4'].astype(int)

# ── Train ─────────────────────────────────────────────────────────────────────
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

models = {}
for name, X_tr, y_tr, X_te, y_te in [
    ('win',   X_train, y_win_tr,   X_test, y_win_te),
    ('place', X_train, y_place_tr, X_test, y_place_te),
    ('top4',  X_train, y_top4_tr,  X_test, y_top4_te),
]:
    log.info(f"Training {name}...")
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42,
    )
    m.fit(X_tr, y_tr)
    prob = m.predict_proba(X_te)[:, 1]
    log.info(f"  {name}: AUC={roc_auc_score(y_te, prob):.4f}  Brier={brier_score_loss(y_te, prob):.4f}")
    models[name] = m

# ── Save ──────────────────────────────────────────────────────────────────────
mdir = Path('models'); mdir.mkdir(exist_ok=True)

for name, m in models.items():
    m.save_model(str(mdir / f'{name}_xgb.json'))

artifacts = {
    'encoders': {k: v.classes_.tolist() for k, v in encoders.items()},
    'feature_cols': FEATURE_COLS,
}
with open(mdir / 'model.json', 'w') as f:
    json.dump(artifacts, f, default=str)

# Full pickle (backward compat with batch_predict.py)
pkg = {'models': models, 'encoders': encoders, 'feature_cols': FEATURE_COLS}
with open(mdir / 'model.pkl', 'wb') as f:
    pickle.dump(pkg, f)

log.info(f"Saved to models/")
log.info(f"  model.pkl + model.json + win_xgb.json / place_xgb.json / top4_xgb.json")
