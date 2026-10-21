#!/usr/bin/env python3
"""
Fourth pass — RIGOR on the value-betting result, plus artefact verification.

The edge>0 bucket showed +6.31% ROI on 1869 bets. Before that means anything I
must rule out:
  (a) variance — what is the bootstrap CI?
  (b) concentration — do a couple of 51+ shots carry all the profit?
  (c) vig artefact — is "edge>0" just "the market overprices longshots", which
      would be a property of the overround, NOT of the model?
  (d) monotonicity — does the edge RANKING predict returns, or is the effect
      flat across all edge buckets (which would mean the model adds nothing)?

(c) is the critical one. If ROI is flat across edge buckets, then v6 has NO
skill and the apparent edge is pure market vig.

Also verifies the delivered artefacts on disk.
"""
import pickle, sqlite3, json, logging, sys, warnings, hashlib, os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('rigor_v6')

ROOT = Path(__file__).parent.parent
conn = sqlite3.connect(ROOT/'data'/'horse.db')
df = pd.read_sql('''
    SELECT re.race_id, re.horse_name, re.horse_no, re.draw, re.rating,
           re.jockey, re.trainer, re.win_odds, re.finishing_position,
           rd.race_date, rd.venue, r.distance, r.race_class, r.track, r.race_no
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
df = df.sort_values(['horse_name','race_date']).reset_index(drop=True)
df['prev_fp'] = df.groupby('horse_name')['finishing_position'].shift(1)
df['horse_win5']    = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(5,min_periods=1).mean())
df['horse_place5'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(5,min_periods=1).mean())
df['horse_win10']   = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(10,min_periods=1).mean())
df['horse_place10'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(10,min_periods=1).mean())
df['prev_date'] = df.groupby('horse_name')['race_date'].shift(1)
df['days_since'] = (df['race_date']-df['prev_date']).dt.days.fillna(90)
df.drop(['prev_fp','prev_date'], axis=1, inplace=True)
df = df.sort_values(['jockey','race_date']).reset_index(drop=True)
df['jp'] = df.groupby('jockey')['finishing_position'].shift(1)
for n in (10,30):
    df[f'jockey_win_{n}']   = df.groupby('jockey')['jp'].transform(lambda x: x.eq(1).rolling(n,min_periods=1).mean())
    df[f'jockey_place_{n}'] = df.groupby('jockey')['jp'].transform(lambda x: x.isin([1,2,3]).rolling(n,min_periods=1).mean())
df.drop('jp', axis=1, inplace=True)
df = df.sort_values(['jockey','venue','race_date']).reset_index(drop=True)
df['jpv'] = df.groupby(['jockey','venue'])['finishing_position'].shift(1)
df['jockey_win_venue'] = df.groupby(['jockey','venue'])['jpv'].transform(lambda x: x.eq(1).rolling(20,min_periods=1).mean())
df.drop('jpv', axis=1, inplace=True)
df = df.sort_values(['trainer','race_date']).reset_index(drop=True)
df['tp'] = df.groupby('trainer')['finishing_position'].shift(1)
for n in (10,30):
    df[f'trainer_win_{n}']   = df.groupby('trainer')['tp'].transform(lambda x: x.eq(1).rolling(n,min_periods=1).mean())
    df[f'trainer_place_{n}'] = df.groupby('trainer')['tp'].transform(lambda x: x.isin([1,2,3]).rolling(n,min_periods=1).mean())
df.drop('tp', axis=1, inplace=True)
df['field_size'] = df.groupby(['race_date','venue','race_no'])['horse_no'].transform('count')
df['draw_norm']  = df['draw']/df['field_size'].replace(0,1)
df['first_run']  = df.groupby('horse_name').cumcount().eq(0).astype(int)
df['has_odds']    = df['win_odds'].notna() & (df['win_odds']>1)
df['implied_prob'] = np.where(df['has_odds'], 1.0/df['win_odds'], np.nan)
df['log_odds']     = np.where(df['has_odds'], np.log(df['win_odds']), np.nan)
grp = df.groupby(['race_date','venue','race_no'])
df['odds_rank'] = grp['win_odds'].rank(ascending=False, method='min')
df['odds_min']  = grp['win_odds'].transform('min')
df['odds_ratio'] = df['win_odds']/df['odds_min'].replace(0,np.nan)
df['field_odds_sum'] = grp['implied_prob'].transform('sum')
df['prob_norm'] = df['implied_prob']/df['field_odds_sum']
from sklearn.preprocessing import LabelEncoder
class_map = {'Class 5':1,'Class 4':2,'Class 3':3,'Class 2':4,'Class 1':5,'Pattern Race':6,'':0}
df['class_enc'] = df['race_class'].fillna('').map(class_map).fillna(0)
te = LabelEncoder(); te.fit(['Turf','Dirt','AWT']); df['track_enc'] = te.transform(df['track'].fillna('Turf'))
ve = LabelEncoder(); ve.fit(['HV','ST']); df['venue_enc'] = ve.transform(df['venue'].fillna('ST'))
df = df.fillna(0)

with open(ROOT/'models'/'model_v6.pkl','rb') as f:
    pkg = pickle.load(f)
FC = pkg['feature_cols']
CUT_VAL = pd.Timestamp('2025-10-01')
test = df[(df['race_date']>CUT_VAL) & df['has_odds']].copy()
test['pred_v6'] = (pkg['xgb_model'].predict_proba(test[FC].astype(float).values)[:,1]*0.5 +
                   pkg['rf_model'].predict_proba(test[FC].astype(float).values)[:,1]*0.5)

RACE_KEYS = ['race_date','venue','race_no']
rows = []
for key, g in test.groupby(RACE_KEYS):
    w = g[g['finishing_position']==1]
    if len(w)!=1 or len(g)<2: continue
    fav_odds = g['win_odds'].min()
    for _, r in g.iterrows():
        rows.append({'race': key, 'horse': r['horse_name'], 'odds': r['win_odds'],
                     'implied': 1.0/r['win_odds'], 'p_v6': r['pred_v6'],
                     'won': int(r['finishing_position']==1),
                     'is_fav': bool(r['win_odds']==fav_odds)})
H = pd.DataFrame(rows)
H['edge'] = H['p_v6'] - H['implied']
log.info(f"rows={len(H)}  races={H['race'].nunique()}")

print()
print("="*78)
print("(a) VARIANCE — bootstrap CI on ROI for edge>0")
print("="*78)
sel = H[H['edge']>0].reset_index(drop=True)
def roi_of(d):
    return (d['odds']*d['won']).sum()/len(d) - 1
rng = np.random.default_rng(42)
bs = []
for _ in range(5000):
    idx = rng.integers(0, len(sel), len(sel))
    bs.append(roi_of(sel.iloc[idx]))
bs = np.array(bs)
print(f"  n={len(sel)}  ROI={roi_of(sel)*100:+.2f}%  95%CI=[{np.percentile(bs,2.5)*100:+.2f}%, {np.percentile(bs,97.5)*100:+.2f}%]"
      f"  P(ROI<=0)={np.mean(bs<=0)*100:.1f}%")
w = sel[sel['won']==1].sort_values('odds', ascending=False)
contrib = (w['odds']-1)
total_profit = (sel['odds']*sel['won']).sum() - len(sel)
print(f"  gross profit = {total_profit:.2f} units on {len(sel)} bets")
print(f"  top-3 winning bets contribute: {contrib.head(3).sum():.2f} units "
      f"({contrib.head(3).sum()/total_profit*100:.0f}% of profit) "
      f"at odds {list(w['odds'].head(3))}")
print(f"  median winning odds={w['odds'].median():.1f}  mean odds of all bets={sel['odds'].mean():.1f}")

print()
print("="*78)
print("(b) VIG ARTEFACT — is 'edge>0' just the market overpricing longshots?")
print("="*78)
print(f"  {'oddsBucket':>14} {'n':>6} {'mktImplied%':>13} {'model%':>8} {'actual%':>9} {'ROI%':>9}")
buckets = [(1,2.5),(2.5,4),(4,8),(8,15),(15,30),(30,1000)]
for lo,hi in buckets:
    m = (sel['odds']>=lo)&(sel['odds']<hi)
    s = sel[m]
    if len(s)==0: continue
    print(f"  {f'[{lo},{hi})':>14} {len(s):>6} {s['implied'].mean()*100:>12.1f}% {s['p_v6'].mean()*100:>7.1f}%"
          f" {s['won'].mean()*100:>8.1f}% {roi_of(s)*100:>8.1f}%")
print()
print("  --- SAME buckets but for ALL horses (no model selection) — the market's own vig curve ---")
for lo,hi in buckets:
    m = (H['odds']>=lo)&(H['odds']<hi)
    s = H[m]
    if len(s)==0: continue
    print(f"  {f'[{lo},{hi})':>14} {len(s):>6} {s['implied'].mean()*100:>12.1f}% {'-':>8} {s['won'].mean()*100:>8.1f}%"
          f" {((s['odds']*s['won']).sum()/len(s)-1)*100:>8.1f}%")

print()
print("="*78)
print("(c) MONOTONICITY — does the edge RANKING predict returns? (decisive for 'has skill')")
print("="*78)
H2 = H.copy()
H2['edge_q'] = pd.qcut(H2['edge'], 10, labels=False, duplicates='drop')
print(f"  {'decile':>7} {'n':>6} {'meanEdge':>10} {'modelP%':>9} {'actual%':>9} {'ROI%':>9}")
for q in sorted(H2['edge_q'].dropna().unique()):
    s = H2[H2['edge_q']==q]
    print(f"  {int(q):>7} {len(s):>6} {s['edge'].mean()*100:>9.2f}% {s['p_v6'].mean()*100:>8.2f}%"
          f" {s['won'].mean()*100:>8.2f}% {((s['odds']*s['won']).sum()/len(s)-1)*100:>8.1f}%")
print()
print("  --- ROI by edge threshold, WITH bootstrap CI (is the ladder monotone?) ---")
for t in [-0.02,-0.01,0.0,0.01,0.02,0.03,0.05]:
    s = H[H['edge']>t]
    if len(s)<20: 
        print(f"  edge>{t:+.2f}: n={len(s)} (too few)"); continue
    bs=[]
    for _ in range(2000):
        idx = rng.integers(0,len(s),len(s)); bs.append(roi_of(s.iloc[idx]))
    bs=np.array(bs)
    print(f"  edge>{t:+.2f}: n={len(s):>4} ROI={roi_of(s)*100:+7.2f}%  95%CI=[{np.percentile(bs,2.5)*100:+6.2f}%, {np.percentile(bs,97.5)*100:+6.2f}%]"
          f"  P(<=0)={np.mean(bs<=0)*100:4.1f}%")

print()
print("="*78)
print("ARTIFACT VERIFICATION")
print("="*78)
for p in ['models/model_v6.pkl','models/model_v6.json','output/verify_v6_results.json',
          'output/verify_v6_analysis.json','output/verify_v6_value.json',
          'scripts/retrain_v6.py','scripts/verify_v6.py','scripts/analyse_v6.py','scripts/value_v6.py']:
    fp = ROOT/p
    if fp.exists():
        h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
        print(f"  OK  {p:42s} {fp.stat().st_size:>10,} B  sha256:{h}")
    else:
        print(f"  MISSING {p}")
meta = json.loads((ROOT/'models'/'model_v6.json').read_text())
print()
print(f"  meta version={meta['version']} built_at={meta['built_at']}")
print(f"  meta auc={meta['auc']}")
print(f"  meta backtest={meta['backtest_test_period']['from']}..{meta['backtest_test_period']['to']} races={meta['backtest_test_period']['races']}")
print(f"  meta top3 by strategy: " + ", ".join(f"{k}={v['top3']}/{v['races']}" for k,v in meta['backtest_test_period']['results'].items()))
print(f"  'rating' in meta features: {'rating' in meta['features']}  (must be False)")
