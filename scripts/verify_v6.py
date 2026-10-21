#!/usr/bin/env python3
"""
Independent verification of model_v6 (2026-10-21).

Purpose
-------
1. REPRODUCE the retrain_v6.py backtest from the SAVED model artefacts
   (models/model_v6.pkl) rather than trusting the training log.
2. Run a MODEL_V5 HEAD-TO-HEAD on the IDENTICAL race set and window, so the
   v5-vs-v6 comparison is apples-to-apples (the 10/19 ST live figure of
   2/11 top-1, 5/11 top-3 is 11 races; this is 500+ races of the same kind).
3. Statistical significance of v6 vs the favourite baseline (bootstrap CI on
   the paired difference, plus McNemar on discordant races).
4. Flat-stake ROI, because hit-rate without price is not a betting strategy.

Everything here is READ-ONLY against data/horse.db and models/.
"""
import pickle, sqlite3, json, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('verify_v6')

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'
CUT_VAL = pd.Timestamp('2025-10-01')

conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.race_id,
           re.horse_name, re.horse_no, re.draw, re.rating,
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
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
''', conn)
conn.close()

df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce').fillna(0).astype(int)
df['draw']   = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
df['race_date'] = pd.to_datetime(df['race_date'])
df['year'] = df['race_date'].dt.year

# ── v6 feature engineering (mirrors retrain_v6.py) ──────────────────────────
df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
df['prev_fp'] = df.groupby('horse_name')['finishing_position'].shift(1)
df['horse_win5']    = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(5, min_periods=1).mean())
df['horse_place5'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(5, min_periods=1).mean())
df['horse_win10']   = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(10, min_periods=1).mean())
df['horse_place10'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(10, min_periods=1).mean())
df['prev_date'] = df.groupby('horse_name')['race_date'].shift(1)
df['days_since'] = (df['race_date'] - df['prev_date']).dt.days.fillna(90)
df.drop(['prev_fp', 'prev_date'], axis=1, inplace=True)

df = df.sort_values(['jockey', 'race_date']).reset_index(drop=True)
df['jp'] = df.groupby('jockey')['finishing_position'].shift(1)
for n in (10, 30):
    df[f'jockey_win_{n}']   = df.groupby('jockey')['jp'].transform(lambda x: x.eq(1).rolling(n, min_periods=1).mean())
    df[f'jockey_place_{n}'] = df.groupby('jockey')['jp'].transform(lambda x: x.isin([1,2,3]).rolling(n, min_periods=1).mean())
df.drop('jp', axis=1, inplace=True)

df = df.sort_values(['jockey', 'venue', 'race_date']).reset_index(drop=True)
df['jpv'] = df.groupby(['jockey', 'venue'])['finishing_position'].shift(1)
df['jockey_win_venue'] = df.groupby(['jockey', 'venue'])['jpv'].transform(lambda x: x.eq(1).rolling(20, min_periods=1).mean())
df.drop('jpv', axis=1, inplace=True)

df = df.sort_values(['trainer', 'race_date']).reset_index(drop=True)
df['tp'] = df.groupby('trainer')['finishing_position'].shift(1)
for n in (10, 30):
    df[f'trainer_win_{n}']   = df.groupby('trainer')['tp'].transform(lambda x: x.eq(1).rolling(n, min_periods=1).mean())
    df[f'trainer_place_{n}'] = df.groupby('trainer')['tp'].transform(lambda x: x.isin([1,2,3]).rolling(n, min_periods=1).mean())
df.drop('tp', axis=1, inplace=True)

df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
df['draw_norm']  = df['draw'] / df['field_size'].replace(0, 1)
df['first_run']  = df.groupby('horse_name').cumcount().eq(0).astype(int)

df['has_odds']    = df['win_odds'].notna() & (df['win_odds'] > 1)
df['implied_prob'] = np.where(df['has_odds'], 1.0 / df['win_odds'], np.nan)
df['log_odds']     = np.where(df['has_odds'], np.log(df['win_odds']), np.nan)
grp = df.groupby(['race_date', 'venue', 'race_no'])
df['odds_rank'] = grp['win_odds'].rank(ascending=False, method='min')
df['odds_min']  = grp['win_odds'].transform('min')
df['odds_max']  = grp['win_odds'].transform('max')
df['odds_ratio'] = df['win_odds'] / df['odds_min'].replace(0, np.nan)
df['field_odds_sum'] = grp['implied_prob'].transform('sum')
df['prob_norm'] = df['implied_prob'] / df['field_odds_sum']

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
df['target_win'] = (df['finishing_position'] == 1).astype(int)

# ── v5 feature engineering (mirrors retrain_v5_fast.py) ─────────────────────
v5 = df.copy()
v5 = v5.sort_values(['horse_name','race_date']).reset_index(drop=True)
v5['prev_fp'] = v5.groupby('horse_name')['finishing_position'].shift(1)
v5['horse_win5']  = v5.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(5, min_periods=1).mean())
v5['horse_place5'] = v5.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(5, min_periods=1).mean())
v5.drop('prev_fp', axis=1, inplace=True)
v5['prev_date'] = v5.groupby('horse_name')['race_date'].shift(1)
v5['days_since'] = (v5['race_date'] - v5['prev_date']).dt.days.fillna(90)
v5.drop('prev_date', axis=1, inplace=True)
v5 = v5.sort_values(['jockey','race_date']).reset_index(drop=True)
v5['jp'] = v5.groupby('jockey')['finishing_position'].shift(1)
v5['jockey_win'] = v5.groupby('jockey')['jp'].transform(lambda x: x.eq(1).rolling(20, min_periods=1).mean())
v5.drop('jp', axis=1, inplace=True)
v5 = v5.sort_values(['trainer','race_date']).reset_index(drop=True)
v5['tp'] = v5.groupby('trainer')['finishing_position'].shift(1)
v5['trainer_win'] = v5.groupby('trainer')['tp'].transform(lambda x: x.eq(1).rolling(20, min_periods=1).mean())
v5.drop('tp', axis=1, inplace=True)
v5['field_size'] = v5.groupby(['race_date','venue','race_no'])['horse_no'].transform('count')
v5['draw_norm']  = v5['draw'] / v5['field_size'].replace(0, 1)
v5['first_run']  = v5.groupby('horse_name').cumcount().eq(0).astype(int)
V5_FEATURES = ['draw','draw_norm','rating','horse_win5','horse_place5','days_since',
               'jockey_win','trainer_win','field_size','first_run',
               'class_enc','track_enc','venue_enc','distance']

df_v5 = v5.fillna(0)
df_v6 = df.fillna(0)

# ── load saved artefacts ────────────────────────────────────────────────────
with open(ROOT/'models'/'model_v6.pkl','rb') as f:
    pkg6 = pickle.load(f)
with open(ROOT/'models'/'model_v5.pkl','rb') as f:
    pkg5 = pickle.load(f)

FEATURE_COLS = pkg6['feature_cols']
assert pkg6['odds_feature_cols'] == ['implied_prob','log_odds','odds_rank','odds_ratio','prob_norm']
assert 'rating' not in FEATURE_COLS, 'rating must stay excluded'

test6 = df_v6[(df_v6['race_date'] > CUT_VAL) & df_v6['has_odds']].copy()
test5 = df_v5[df_v5['race_date'] > CUT_VAL].copy()
# align v5 predictions onto the SAME race rows as v6
test5 = test5.set_index(['race_id','horse_no']).loc[
    pd.MultiIndex.from_arrays([test6['race_id'], test6['horse_no']])]
test5 = test5.reset_index()

p6 = (pkg6['xgb_model'].predict_proba(test6[FEATURE_COLS].astype(float).values)[:,1]*0.5 +
      pkg6['rf_model'].predict_proba(test6[FEATURE_COLS].astype(float).values)[:,1]*0.5)
test6['pred_v6'] = p6
p5 = (pkg5['xgb_model'].predict_proba(test5[V5_FEATURES].astype(float).values)[:,1]*0.5 +
      pkg5['rf_model'].predict_proba(test5[V5_FEATURES].astype(float).values)[:,1]*0.5)
test5['pred_v5'] = p5
test6['pred_v5'] = np.asarray(p5).ravel()

test6['pred_fav'] = -test6['win_odds']
test6['pred_rating'] = test6['rating']
test6['pred_blend'] = test6['prob_norm']*0.6 + test6['pred_v6']*0.4

log.info(f"Test window (odds-bearing): {test6['race_date'].min().date()}–{test6['race_date'].max().date()}"
         f"  rows={len(test6):,}  AUC(v6)={roc_auc_score(test6['target_win'], test6['pred_v6']):.4f}"
         f"  AUC(v5)={roc_auc_score(test6['target_win'], test6['pred_v5']):.4f}")

RACE_KEYS = ['race_date','venue','race_no']

def per_race(df_):
    """One row per race: winner + candidate scores."""
    out = []
    for key, g in df_.groupby(RACE_KEYS):
        w = g[g['finishing_position'] == 1]
        if len(w) != 1 or len(g) < 2:
            continue
        out.append((key, w.iloc[0]['horse_name'], g))
    return out

def hits(df_, score_col):
    """Return per-race booleans: top1_hit, top3_hit, plus the picked horse's odds."""
    res = []
    for key, wname, g in per_race(df_):
        s = g.sort_values(score_col, ascending=False)
        t1 = s.head(1)['horse_name'].tolist()
        t3 = s.head(3)['horse_name'].tolist()
        res.append({'race': key, 'year': int(pd.Timestamp(key[0]).year),
                    'top1': bool(t1 and t1[0] == wname),
                    'top3': bool(wname in t3),
                    'pick_odds': float(s.head(1)['win_odds'].iloc[0]),
                    'win_odds': float(g[g['horse_name']==wname]['win_odds'].iloc[0]),
                    'field': len(g)})
    return pd.DataFrame(res)

STRATS = {'model_v6 (form+odds)':'pred_v6', 'model_v5 (legacy)':'pred_v5',
          'BASELINE favourite':'pred_fav', 'BASELINE rating':'pred_rating',
          'BLEND 60mkt/40ml':'pred_blend'}

results = {}
for name, col in STRATS.items():
    h = hits(test6, col)
    results[name] = h
    # ROI: flat $1 on the top-1 pick at its published SP
    roi = (h['pick_odds'] * h['top1']).sum() - len(h)
    log.info(f"{name:24s} n={len(h):>4}  top1={h['top1'].sum():>4} ({h['top1'].mean()*100:5.2f}%)"
             f"  top3={h['top3'].sum():>4} ({h['top3'].mean()*100:5.2f}%)"
             f"  flatROI(top1)={roi/len(h)*100:+6.2f}%")

# ── significance: v6 vs favourite ───────────────────────────────────────────
def mcnemar(a, b):
    """Exact-ish McNemar on paired booleans. a=model, b=baseline."""
    n01 = int(((a) & (~b)).sum()); n10 = int(((~a) & (b)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    from math import comb
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k+1)) / (2**n) * 2
    return n01, n10, min(1.0, p)

def boot_ci(a, b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(d.mean())

print()
print("=" * 78)
print("HEAD-TO-HEAD: model_v6 vs BASELINE favourite (paired, same races)")
print("=" * 78)
for metric in ['top1', 'top3']:
    a = results['model_v6 (form+odds)'][metric].values
    b = results['BASELINE favourite'][metric].values
    lo, hi, mean = boot_ci(a, b)
    n01, n10, p = mcnemar(a, b)
    print(f"  {metric}: v6={a.sum()}  fav={b.sum()}  diff={a.sum()-b.sum():+d}"
          f"  mean_diff={mean*100:+.2f}pp  95%CI=[{lo*100:+.2f}, {hi*100:+.2f}]pp  McNemar p={p:.3f}"
          f"  (v6-only={n01}, fav-only={n10})")

print()
print("=" * 78)
print("HEAD-TO-HEAD: model_v6 vs model_v5 (paired, same races)")
print("=" * 78)
for metric in ['top1', 'top3']:
    a = results['model_v6 (form+odds)'][metric].values
    b = results['model_v5 (legacy)'][metric].values
    lo, hi, mean = boot_ci(a, b)
    n01, n10, p = mcnemar(a, b)
    print(f"  {metric}: v6={a.sum()}  v5={b.sum()}  diff={a.sum()-b.sum():+d}"
          f"  mean_diff={mean*100:+.2f}pp  95%CI=[{lo*100:+.2f}, {hi*100:+.2f}]pp  McNemar p={p:.3f}")

# ── does the model beat the market where the market is confident? ───────────
print()
print("=" * 78)
print("VALUE CHECK: flat $1 on top-1 pick, split by favourite's SP bucket")
print("=" * 78)
h6 = results['model_v6 (form+odds)'].copy()
h6['fav_sp'] = h6['pick_odds']
buckets = [(1,2.5),(2.5,4),(4,8),(8,1000)]
for lo_, hi_ in buckets:
    m = (h6['fav_sp']>=lo_) & (h6['fav_sp']<hi_)
    sub = h6[m]
    if len(sub)==0: continue
    roi = (sub['pick_odds']*sub['top1']).sum() - len(sub)
    print(f"  SP [{lo_}, {hi_}): n={len(sub):>4}  v6 top1={sub['top1'].sum():>4} ({sub['top1'].mean()*100:5.1f}%)"
          f"  ROI={roi/len(sub)*100:+7.2f}%")

# ── per-year stability of v6 vs favourite (is it consistent?) ──────────────
print()
print("=" * 78)
print("PER-YEAR: v6 vs favourite top-3 (consistency check)")
print("=" * 78)
years = results['model_v6 (form+odds)']['year'].values
yr_rows = []
for yr in sorted(set(years)):
    idx = years == yr
    a = results['model_v6 (form+odds)']['top3'].values[idx]
    b = results['BASELINE favourite']['top3'].values[idx]
    c = results['model_v5 (legacy)']['top3'].values[idx]
    n = len(a)
    yr_rows.append((yr, n, a.sum(), b.sum(), c.sum()))
    print(f"  {yr}: races={n:>4}  v6 top3={a.sum():>4} ({a.mean()*100:5.1f}%)"
          f"  fav top3={b.sum():>4} ({b.mean()*100:5.1f}%)"
          f"  v5 top3={c.sum():>4} ({c.mean()*100:5.1f}%)")

# ── write results ───────────────────────────────────────────────────────────
out = {
    'window': {'from': str(test6['race_date'].min().date()), 'to': str(test6['race_date'].max().date())},
    'races': int(len(results['model_v6 (form+odds)'])),
    'auc': {
        'v6_form_odds': round(float(roc_auc_score(test6['target_win'], test6['pred_v6'])), 4),
        'v5_legacy':    round(float(roc_auc_score(test6['target_win'], test6['pred_v5'])), 4),
        'favourite':    round(float(roc_auc_score(test6['target_win'], 1.0/test6['win_odds'])), 4),
        'rating':       round(float(roc_auc_score(test6['target_win'], test6['pred_rating'])), 4),
    },
    'strategies': {k: {'top1': int(v['top1'].sum()), 'top1_pct': round(float(v['top1'].mean()*100),2),
                        'top3': int(v['top3'].sum()), 'top3_pct': round(float(v['top3'].mean()*100),2),
                        'flat_roi_top1_pct': round(float((v['pick_odds']*v['top1']).sum()/len(v)*100 - 100), 2)}
                   for k, v in results.items()},
    'per_year': [{'year': int(y), 'races': int(n), 'v6_top3': int(a), 'fav_top3': int(b), 'v5_top3': int(c)}
                 for y,n,a,b,c in yr_rows],
}
(ROOT/'output'/'verify_v6_results.json').write_text(json.dumps(out, indent=2))
log.info(f"Wrote {ROOT/'output'/'verify_v6_results.json'}")
