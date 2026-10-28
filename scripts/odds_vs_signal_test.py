"""Is model_v6 actually a model, or a noisy copy of the market?

Feature importance says the odds features carry 83% of the signal
(implied_prob 0.316 + prob_norm 0.293 + log_odds 0.220 in XGBoost), while
track_enc carries 0.9% and every non-odds feature is under 1%. The model
predicts from the odds it is handed and adds a little noise on top.

That would explain everything observed so far:
  * model AUC 0.78232 < market AUC 0.78584
  * no blend of model and market beats the market alone
  * the "edge" is a longshot miscalibration artifact

This script tests that directly by training the SAME architecture on
non-odds features only, and on odds-only, and comparing all of them plus the
market on AUC and on top-1 hit rate. If the non-odds model is far behind the
market, then the honest product of this project is the market, and the right
thing to tell Peter is that the model should not be used to pick horses.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/odds_vs_signal_test.py
"""
import pickle, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
import features as F

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'
MODEL = ROOT / 'models' / 'model_v6.pkl'

conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.race_id, re.horse_name, re.horse_no, re.draw, re.rating,
           re.jockey, re.trainer, re.win_odds, re.place_odds,
           re.finishing_position,
           rd.race_date, rd.venue,
           r.distance, r.race_class, r.track, r.race_no
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
      AND CAST(re.finishing_position AS INTEGER) > 0
''', conn)
conn.close()

df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce').fillna(0).astype(int)
df['draw'] = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
df['race_date'] = pd.to_datetime(df['race_date'])

df, _ = F.build_features(df)
pkg = pickle.load(open(MODEL, 'rb'))
COLS = pkg['feature_cols']
RK = ['race_date', 'venue', 'race_no']

ODDS_FEATS = {'implied_prob', 'prob_norm', 'log_odds', 'odds_rank', 'odds_ratio'}
NON_ODDS = [c for c in COLS if c not in ODDS_FEATS]
ODDS_ONLY = [c for c in COLS if c in ODDS_FEATS]

data = df[df['win_odds'].notna() & (df['win_odds'] > 0)].copy()
data['won'] = (data['finishing_position'] == 1).astype(int)
print(f'data: {len(data):,} runners, {data.groupby(RK).ngroups:,} races')

# temporal split so nothing leaks
cut1 = data['race_date'].quantile(0.70)
cut2 = data['race_date'].quantile(0.85)
train = data[data['race_date'] < cut1]
valid = data[(data['race_date'] >= cut1) & (data['race_date'] < cut2)]
test = data[data['race_date'] >= cut2]
print(f'split: train {len(train):,} / valid {len(valid):,} / test {len(test):,}\n')


def fit(cols, seed=42):
    Xtr = train[cols].astype(float).fillna(0).values
    Xva = valid[cols].astype(float).fillna(0).values
    xgbm = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, eval_metric='logloss',
                              n_jobs=4, verbosity=0)
    xgbm.fit(Xtr, train['won'], eval_set=[(Xva, valid['won'])], verbose=False)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                random_state=seed, n_jobs=4)
    rf.fit(Xtr, train['won'])
    return xgbm, rf


def evaluate(name, cols, seed=42):
    xgbm, rf = fit(cols, seed)
    Xte = test[cols].astype(float).fillna(0).values
    p = (xgbm.predict_proba(Xte)[:, 1] * 0.5 + rf.predict_proba(Xte)[:, 1] * 0.5)
    t = test.copy()
    t['p'] = p
    t['p_market_raw'] = 1.0 / t['win_odds']
    t['p_market'] = t.groupby(RK)['p_market_raw'].transform(lambda s: s / s.sum())
    t['p_model'] = t.groupby(RK)['p'].transform(lambda s: s / s.sum())

    auc_m = roc_auc_score(t['won'], t['p_market'])
    auc_s = roc_auc_score(t['won'], t['p_model'])

    def top1(col):
        hit = 0
        for _, g in t.groupby(RK, sort=False):
            w = g[g['won'] == 1]
            if len(w) == 0:
                continue
            pick = g.sort_values(col, ascending=False).head(1)
            if len(pick) and pick['horse_name'].iloc[0] == w['horse_name'].iloc[0]:
                hit += 1
        return hit, t.groupby(RK).ngroups

    h_m, n = top1('p_market')
    h_s, _ = top1('p_model')
    print(f'  {name}')
    print(f'    AUC  market={auc_m:.5f}   model={auc_s:.5f}   diff={auc_s - auc_m:+.5f}')
    print(f'    top1 market={h_m}/{n} ({h_m/n*100:.1f}%)   model={h_s}/{n} ({h_s/n*100:.1f}%)')
    return auc_m, auc_s, h_m, h_s, n


print('=' * 74)
print('TRAIN THE SAME ARCHITECTURE ON DIFFERENT FEATURE SETS')
print('=' * 74)
evaluate('ALL features (as shipped)', COLS)
evaluate('NON-ODDS features only (the real "signal")', NON_ODDS)
evaluate('ODDS features only', ODDS_ONLY)

# pure market on the same test set
t = test.copy()
t['p_market_raw'] = 1.0 / t['win_odds']
t['p_market'] = t.groupby(RK)['p_market_raw'].transform(lambda s: s / s.sum())
auc = roc_auc_score(t['won'], t['p_market'])
hit = 0
for _, g in t.groupby(RK, sort=False):
    w = g[g['won'] == 1]
    if len(w) == 0:
        continue
    pick = g.sort_values('p_market', ascending=False).head(1)
    if len(pick) and pick['horse_name'].iloc[0] == w['horse_name'].iloc[0]:
        hit += 1
print('=' * 74)
print('  MARKET ONLY (no model at all)')
print(f'    AUC  market={auc:.5f}')
print(f'    top1 market={hit}/{t.groupby(RK).ngroups} ({hit/t.groupby(RK).ngroups*100:.1f}%)')

print('\n' + '=' * 74)
print('INTERPRETATION')
print('=' * 74)
print('If "NON-ODDS features only" is far below the market on AUC and top-1,')
print('then the model contains no independent racing signal: it is the odds')
print('echoed back with noise. The correct product is then the market, and')
print('the honest recommendation to Peter is NOT to bet, plus a clear')
print('statement that the model cannot be used to select horses.')
