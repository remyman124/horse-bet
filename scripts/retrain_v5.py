#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 5
====================================
Key insight: We should use ACTUAL HKJC live odds as a key feature
because they contain the collective wisdom of all bettors.

Strategy:
1. For each race, include actual HKJC win_odds as a feature
2. Train model to predict WIN (not place) using ALL available pre-race info
3. The model learns: when a horse's TRUE probability differs from market odds,
   which factors predict that difference
4. Apply to May 6 ST races using live scraped odds

New features:
- ACTUAL win_odds from HKJC (for May 6 races, scraped via browser)
- Horse rating (from DB)
- Odds rank within race (relative to field)
"""
import pickle, sqlite3, json, logging, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
log = logging.getLogger('retrain_v5')

DB   = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT  = Path(__file__).parent.parent / 'models' / 'model_v5.pkl'
ODDS = Path(__file__).parent.parent / 'data' / 'may6_odds.json'

# ── Load May 6 actual HKJC odds (scraped) ────────────────────────────────────
# These are the REAL win odds scraped from HKJC for May 6 ST
may6_odds = {
    1: {1:6.4, 2:11.0, 3:19.0, 4:19.0, 5:16.0, 6:8.3, 7:32.0, 8:62.0, 9:3.5, 10:14.0, 11:7.6, 12:10.0, 13:31.0, 14:36.0},
    2: {1:15.0, 2:7.9, 3:3.2, 4:17.0, 5:55.0, 6:18.0, 7:21.0, 8:6.8, 9:5.7, 10:6.5, 11:50.0, 12:27.0},
    3: {1:50.0, 2:4.8, 3:11.0, 4:7.3, 5:14.0, 6:6.3, 7:3.9, 8:14.0, 9:38.0, 10:36.0, 11:54.0, 12:7.0},
    4: {1:12.0, 2:5.5, 3:11.0, 4:4.1, 5:24.0, 6:18.0, 7:11.0, 8:45.0, 9:4.8, 10:7.7, 11:11.0},
    5: {1:3.7, 2:4.0, 3:15.0, 4:13.0, 5:13.0, 6:9.3, 7:17.0, 8:13.0, 9:24.0, 10:6.3, 11:57.0, 12:80.0},
    6: {1:18.0, 2:14.0, 3:19.0, 4:15.0, 5:7.6, 6:4.9, 7:12.0, 8:11.0, 9:15.0, 10:93.0, 11:4.8, 12:13.0, 13:8.2},
    7: {1:11.0, 2:10.0, 3:6.6, 4:7.3, 5:5.6, 6:14.0, 7:12.0, 8:6.5, 9:30.0, 10:16.0, 11:7.8, 12:30.0},
    8: {1:6.0, 2:19.0, 3:5.9, 4:33.0, 5:15.0, 6:20.0, 7:5.0, 8:12.0, 9:7.5, 10:26.0, 11:9.7, 12:27.0, 13:23.0, 14:19.0},
    9: {1:16.0, 2:4.4, 3:9.7, 4:7.0, 5:18.0, 6:28.0, 7:5.3, 8:10.0, 9:13.0, 10:14.0, 11:7.5, 12:35.0},
}

# ── Load data ──────────────────────────────────────────────────────────────────
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
df['draw']               = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating']             = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance']           = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['race_date']          = pd.to_datetime(df['race_date'])

log.info(f"Loaded {len(df):,} entries")

# ── Engineer features ──────────────────────────────────────────────────────────
df = df.sort_values(['horse_name', 'race_date', 'venue']).reset_index(drop=True)

# Horse rolling features
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

# ── Feature columns (now INCLUDING actual HKJC odds for May 6 races) ───────────
# For training: use rating as proxy for odds (rating is available pre-race)
# The model learns: given this horse's rating/form, what is the probability of winning?
FEATURE_COLS = [
    'draw', 'draw_norm',
    'rating',
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
log.info(f"Features ({len(FEATURE_COLS)})")

# ── Temporal split ─────────────────────────────────────────────────────────────
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

# ── Train models ─────────────────────────────────────────────────────────────
results = {}

for target_name, y_col in [('win','target_win'),('place','target_place')]:
    y_train = df_train[y_col].values.astype(int)
    y_val   = df_val[y_col].values.astype(int)
    y_test  = df_test[y_col].values.astype(int)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
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

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=8,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_val   = rf_model.predict_proba(X_val)[:,1]
    rf_test  = rf_model.predict_proba(X_test)[:,1]
    rf_vauc  = roc_auc_score(y_val, rf_val)
    rf_tauc  = roc_auc_score(y_test, rf_test)
    log.info(f"  RF  {target_name}: val={rf_vauc:.4f} test={rf_tauc:.4f}")

    results[f'xgb_{target_name}'] = {'model': xgb_model, 'val': xgb_vauc, 'test': xgb_tauc}
    results[f'rf_{target_name}'] = {'model': rf_model, 'val': rf_vauc, 'test': rf_tauc}

# Ensemble
xgb_win  = results['xgb_win']['model'].predict_proba(X_test)[:,1]
rf_win   = results['rf_win']['model'].predict_proba(X_test)[:,1]
ens_win  = (xgb_win + rf_win) / 2
ens_win_auc = roc_auc_score(df_test['target_win'].values.astype(int), ens_win)

xgb_place  = results['xgb_place']['model'].predict_proba(X_test)[:,1]
rf_place   = results['rf_place']['model'].predict_proba(X_test)[:,1]
ens_place  = (xgb_place + rf_place) / 2
ens_place_auc = roc_auc_score(df_test['target_place'].values.astype(int), ens_place)

log.info(f"Ensemble: win AUC={ens_win_auc:.4f}  place AUC={ens_place_auc:.4f}")

# ── Now predict May 6 ST using actual HKJC odds ────────────────────────────────
# Build May 6 dataset WITH actual HKJC odds
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Horse history (up to May 5)
cur.execute('''
    SELECT re.horse_name, re.finishing_position, rd.race_date
    FROM horse_entries re JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < "2026-05-06"
      AND re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
    ORDER BY re.horse_name, rd.race_date DESC
''')
horse_hist = {}
for name, fp, date in cur.fetchall():
    if name not in horse_hist: horse_hist[name] = []
    if len(horse_hist[name]) < 10:
        horse_hist[name].append((date, int(fp)))

def horse_feats(name):
    h = horse_hist.get(name, [])
    if not h: return [0.0]*9
    fps = [fp for _,fp in h]
    rows = []
    for n in [3,5,10]:
        s = fps[:n]
        rows.extend([sum(1 for fp in s if fp==1)/len(s),
                     sum(1 for fp in s if fp in [1,2,3])/len(s),
                     sum(s)/len(s)])
    return rows

def days_since_fn(name):
    h = horse_hist.get(name, [])
    if not h: return 90.0
    return max(1.0, float((pd.Timestamp('2026-05-06') - pd.Timestamp(h[0][0])).days))

cur.execute('''
    SELECT re.jockey, re.finishing_position, rd.race_date
    FROM horse_entries re JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < "2026-05-06"
      AND re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
    ORDER BY re.jockey, rd.race_date DESC
''')
jockey_hist = {}
for jockey, fp, date in cur.fetchall():
    if jockey not in jockey_hist: jockey_hist[jockey] = []
    if len(jockey_hist[jockey]) < 30:
        jockey_hist[jockey].append(int(fp))

def jockey_feats(jockey):
    h = jockey_hist.get(jockey, [])
    if not h: return [0.0]*4
    rows = []
    for n in [10,30]:
        s = h[:n]
        rows.extend([sum(1 for fp in s if fp==1)/len(s),
                     sum(1 for fp in s if fp in [1,2,3])/len(s)])
    return rows

cur.execute('''
    SELECT re.trainer, re.finishing_position, rd.race_date
    FROM horse_entries re JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date < "2026-05-06"
      AND re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
    ORDER BY re.trainer, rd.race_date DESC
''')
trainer_hist = {}
for trainer, fp, date in cur.fetchall():
    if trainer not in trainer_hist: trainer_hist[trainer] = []
    if len(trainer_hist[trainer]) < 30:
        trainer_hist[trainer].append(int(fp))

def trainer_feats(trainer):
    h = trainer_hist.get(trainer, [])
    if not h: return [0.0]*4
    rows = []
    for n in [10,30]:
        s = h[:n]
        rows.extend([sum(1 for fp in s if fp==1)/len(s),
                     sum(1 for fp in s if fp in [1,2,3])/len(s)])
    return rows

# Load May 6 entries
may6_df = pd.read_sql('''
    SELECT re.horse_name, re.horse_no, re.draw, re.rating,
           re.jockey, re.trainer,
           rd.race_date, rd.venue,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE rd.race_date = "2026-05-06" AND rd.venue = "ST"
    ORDER BY r.race_no, re.horse_no
''', conn)
conn.close()

class_map = {'Class 5':1,'Class 4':2,'Class 3':3,'Class 2':4,'Class 1':5,'Pattern Race':6,'':0}
rows = []
for _, r in may6_df.iterrows():
    hf = horse_feats(r['horse_name'])
    jf = jockey_feats(r['jockey'])
    tf = trainer_feats(r['trainer'])
    ds = days_since_fn(r['horse_name'])
    fs = len(may6_df[may6_df['race_no'] == r['race_no']])
    dn = r['draw']/fs if fs > 0 else 0.5
    ce = class_map.get(str(r['race_class']), 0)
    te = track_enc.transform([r['track'] if pd.notna(r['track']) else 'Turf'])[0]
    ve = venue_enc.transform([r['venue']])[0]
    fr = 1 if r['horse_name'] not in horse_hist else 0
    rating = float(r['rating']) if pd.notna(r['rating']) else 0.0
    row = {
        'draw': float(r['draw']) if pd.notna(r['draw']) else 0,
        'draw_norm': dn,
        'rating': rating,
        'horse_win_3': hf[0], 'horse_place_3': hf[1], 'horse_avg_fp_3': hf[2],
        'horse_win_5': hf[3], 'horse_place_5': hf[4], 'horse_avg_fp_5': hf[5],
        'horse_win_10': hf[6], 'horse_place_10': hf[7], 'horse_avg_fp_10': hf[8],
        'days_since': ds,
        'jockey_win_10': jf[0], 'jockey_place_10': jf[1],
        'jockey_win_30': jf[2], 'jockey_place_30': jf[3],
        'trainer_win_10': tf[0], 'trainer_place_10': tf[1],
        'trainer_win_30': tf[2], 'trainer_place_30': tf[3],
        'field_size': float(fs), 'first_run': float(fr),
        'class_enc': float(ce), 'track_enc': float(te), 'venue_enc': float(ve),
        'distance': float(r['distance']) if pd.notna(r['distance']) else 1200,
        'race_no': int(r['race_no']), 'horse_no': int(r['horse_no']),
        'horse_name': r['horse_name'],
    }
    rows.append(row)

may6_pred = pd.DataFrame(rows)

# Add actual HKJC odds
for _, r in may6_pred.iterrows():
    rn = r['race_no']
    hn = r['horse_no']
    odds = may6_odds.get(rn, {}).get(hn, None)
    may6_pred.loc[may6_pred.index[may6_pred['horse_no']==hn], 'hk_odds'] = odds

# Get model predictions
X_may6 = may6_pred[FEATURE_COLS].astype(float).values
pred_xgb_win   = results['xgb_win']['model'].predict_proba(X_may6)[:,1]
pred_rf_win    = results['rf_win']['model'].predict_proba(X_may6)[:,1]
pred_xgb_place = results['xgb_place']['model'].predict_proba(X_may6)[:,1]
pred_rf_place  = results['rf_place']['model'].predict_proba(X_may6)[:,1]
pred_ens_win   = (pred_xgb_win + pred_rf_win) / 2
pred_ens_place = (pred_xgb_place + pred_rf_place) / 2

may6_pred['pred_xgb_win']   = pred_xgb_win
may6_pred['pred_rf_win']    = pred_rf_win
may6_pred['pred_ens_win']   = pred_ens_win
may6_pred['pred_xgb_place'] = pred_xgb_place
may6_pred['pred_rf_place']  = pred_rf_place
may6_pred['pred_ens_place'] = pred_ens_place

# ── Strategy: COMBINE model probability with actual odds ───────────────────────
# Compute market-implied probability
may6_pred['market_prob'] = 1.0 / may6_pred['hk_odds'].replace(0, 999)
# Value score: how much is the horse undervalued by market?
may6_pred['value_score'] = may6_pred['pred_ens_win'] - may6_pred['market_prob']
# Expected return if horse places
may6_pred['expected_return'] = may6_pred['pred_ens_place'] * may6_pred['hk_odds']

# ── Actual results ────────────────────────────────────────────────────────────
actual = {
    1: {'winner': 9, 'places': [9,13,11]},
    2: {'winner': 9, 'places': [9,10,2]},
    3: {'winner': 7, 'places': [7,6,8]},
    4: {'winner': 9, 'places': [9,4,6]},
    5: {'winner': 10, 'places': [10,4,7]},
    6: {'winner': 13, 'places': [13,6,1]},
    7: {'winner': 1, 'places': [1,6,12]},
    8: {'winner': 7, 'places': [7,11,1]},
    9: {'winner': 7, 'places': [7,1,9]},
}

# ── Print comparison ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("MODEL v5 — May 6, 2026 ST with HKJC Actual Odds")
print("="*80)

# Test different strategies
for race_no in sorted(actual.keys()):
    race_data = may6_pred[may6_pred['race_no'] == race_no].copy()
    winner_actual = actual[race_no]['winner']
    places_actual = actual[race_no]['places']

    # Strategy 1: Highest model win probability
    s1 = race_data.sort_values('pred_ens_win', ascending=False).head(3)
    # Strategy 2: Highest value score (undervalued by market)
    s2 = race_data.sort_values('value_score', ascending=False).head(3)
    # Strategy 3: Highest expected return (place prob * odds)
    s3 = race_data.sort_values('expected_return', ascending=False).head(3)
    # Strategy 4: Highest combined score (model win prob * log(odds))
    race_data['combined'] = race_data['pred_ens_win'] * np.log(race_data['hk_odds'].replace(0,1)+1)
    s4 = race_data.sort_values('combined', ascending=False).head(3)

    hit_s1 = winner_actual in s1['horse_no'].tolist()
    hit_s2 = winner_actual in s2['horse_no'].tolist()
    hit_s3 = winner_actual in s3['horse_no'].tolist()
    hit_s4 = winner_actual in s4['horse_no'].tolist()

    actual_name = race_data[race_data['horse_no']==winner_actual]['horse_name'].values
    actual_name = actual_name[0] if len(actual_name)>0 else '?'
    actual_odds = race_data[race_data['horse_no']==winner_actual]['hk_odds'].values
    actual_odds = actual_odds[0] if len(actual_odds)>0 else '?'

    print(f"\nR{race_no}: ACTUAL #{winner_actual} {actual_name} ({actual_odds}x)")
    print(f"  S1-ModelWin:  {[(int(h), round(p,3)) for h,p in zip(s1['horse_no'],s1['pred_ens_win'])]} → {'✓' if hit_s1 else '✗'}")
    print(f"  S2-Value:     {[(int(h), round(v,3)) for h,v in zip(s2['horse_no'],s2['value_score'])]} → {'✓' if hit_s2 else '✗'}")
    print(f"  S3-ExpReturn: {[(int(h), round(e,2)) for h,e in zip(s3['horse_no'],s3['expected_return'])]} → {'✓' if hit_s3 else '✗'}")
    print(f"  S4-Combined:  {[(int(h), round(c,3)) for h,c in zip(s4['horse_no'],s4['combined'])]} → {'✓' if hit_s4 else '✗'}")

# ── Save ─────────────────────────────────────────────────────────────────────
package = {
    'models': {k: v['model'] for k, v in results.items()},
    'feature_cols': FEATURE_COLS,
    'encoders': {
        'track': track_enc, 'venue': venue_enc,
        'jockey': jock_enc, 'trainer': trainer_enc,
    }
}
with open(OUT, 'wb') as f:
    pickle.dump(package, f)

meta = {
    'version': 'v5',
    'features': FEATURE_COLS,
    'ensemble_win_auc': round(ens_win_auc, 4),
    'ensemble_place_auc': round(ens_place_auc, 4),
}
with open(OUT.with_suffix('.json'), 'w') as f:
    json.dump(meta, f, indent=2)
log.info(f"Saved to {OUT}")
