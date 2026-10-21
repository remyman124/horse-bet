#!/usr/bin/env python3
"""
Follow-up analysis: WHY does v6 tie the favourite baseline?

The verification run showed v6 top-1 = 143 = favourite top-1 (exact tie), and
v6 top-3 = 295 < favourite top-3 = 301.

Hypothesis: v6 has simply LEARNED TO BE THE MARKET. The odds features dominate
its feature importance (prob_norm 0.378, implied_prob 0.229, log_odds 0.166 =
77% of total importance on three market-derived columns). If v6's top-1 pick is
the favourite most of the time, then "v6 == favourite" is a tautology, not an
achievement — and the honest conclusion is that v6 adds NO edge over the market
while absorbing all the complexity/risk of a trained model.

This script measures that directly, plus:
  - blend vs favourite significance (is the blend's +2 top-3 real?)
  - the 2026-10-19 ST race day specifically (apples-to-apples with the live
    v5 figure of top-1 2/11, top-3 5/11 quoted in the post-mortem)
  - where v6 DISAGREES with the favourite, does it win or lose more?
"""
import pickle, sqlite3, json, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('analyse_v6')

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
df['horse_win5']    = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(5, min_periods=1).mean())
df['horse_place5'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(5, min_periods=1).mean())
df['horse_win10']   = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.eq(1).rolling(10, min_periods=1).mean())
df['horse_place10'] = df.groupby('horse_name')['prev_fp'].transform(lambda x: x.isin([1,2,3]).rolling(10, min_periods=1).mean())
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
test['pred_blend'] = test['prob_norm']*0.6 + test['pred_v6']*0.4

RACE_KEYS = ['race_date','venue','race_no']
races = []
for key, g in test.groupby(RACE_KEYS):
    w = g[g['finishing_position']==1]
    if len(w)!=1 or len(g)<2: continue
    fav = g.loc[g['win_odds'].idxmin()]
    v6p = g.sort_values('pred_v6', ascending=False).head(1)
    blp = g.sort_values('pred_blend', ascending=False).head(1)
    races.append({
        'date': key[0], 'venue': key[1], 'race_no': key[2],
        'winner': w.iloc[0]['horse_name'], 'winner_odds': float(w.iloc[0]['win_odds']),
        'fav_name': fav['horse_name'], 'fav_odds': float(fav['win_odds']),
        'v6_name': v6p.iloc[0]['horse_name'], 'v6_odds': float(v6p.iloc[0]['win_odds']),
        'v6_isfav': bool(v6p.iloc[0]['horse_name'] == fav['horse_name']),
        'v6_hit': bool(v6p.iloc[0]['horse_name'] == w.iloc[0]['horse_name']),
        'fav_hit': bool(fav['horse_name'] == w.iloc[0]['horse_name']),
        'bl_name': blp.iloc[0]['horse_name'], 'bl_hit': bool(blp.iloc[0]['horse_name'] == w.iloc[0]['horse_name']),
        'v6_in_top3': bool(w.iloc[0]['horse_name'] in g.sort_values('pred_v6',ascending=False).head(3)['horse_name'].tolist()),
        'fav_in_top3': bool(w.iloc[0]['horse_name'] in g.sort_values('win_odds').head(3)['horse_name'].tolist()),
        'bl_in_top3': bool(w.iloc[0]['horse_name'] in g.sort_values('pred_blend',ascending=False).head(3)['horse_name'].tolist()),
    })
R = pd.DataFrame(races)
log.info(f"races analysed: {len(R)}")

print()
print("="*78)
print("IS v6 JUST THE MARKET?")
print("="*78)
same = R['v6_name'] == R['fav_name']
print(f"  v6 top-1 pick IS the favourite in {same.sum()}/{len(R)} races = {same.mean()*100:.1f}%")
print(f"  v6 top-1 hit rate overall          : {R['v6_hit'].sum()}/{len(R)} ({R['v6_hit'].mean()*100:.2f}%)")
print(f"  favourite hit rate overall         : {R['fav_hit'].sum()}/{len(R)} ({R['fav_hit'].mean()*100:.2f}%)")
print()
print("  --- split by whether v6 agreed with the favourite ---")
for lbl, m in [('v6 == favourite', same), ('v6 != favourite', ~same)]:
    sub = R[m]
    if len(sub)==0: continue
    print(f"  {lbl:20s} n={len(sub):>4}  v6_hit={sub['v6_hit'].sum():>4} ({sub['v6_hit'].mean()*100:5.1f}%)"
          f"  fav_hit={sub['fav_hit'].sum():>4} ({sub['fav_hit'].mean()*100:5.1f}%)"
          f"  v6_top3={sub['v6_in_top3'].sum():>4} ({sub['v6_in_top3'].mean()*100:5.1f}%)"
          f"  fav_top3={sub['fav_in_top3'].sum():>4} ({sub['fav_in_top3'].mean()*100:5.1f}%)")

print()
print("="*78)
print("BLEND (60% market / 40% ML) vs favourite — is the +2 top-3 real?")
print("="*78)
a = R['bl_in_top3'].values; b = R['fav_in_top3'].values
n01 = int((a&~b).sum()); n10 = int((~a&b).sum())
from math import comb
n = n01+n10
k = min(n01,n10)
p = min(1.0, sum(comb(n,i) for i in range(k+1))/2**n*2) if n else 1.0
rng = np.random.default_rng(7); d = a.astype(float)-b.astype(float)
idx = rng.integers(0,len(d),size=(10000,len(d))); m = d[idx].mean(axis=1)
print(f"  blend top1={R['bl_hit'].sum()}  top3={a.sum()}  | favourite top1={R['fav_hit'].sum()}  top3={b.sum()}")
print(f"  top3 diff={a.sum()-b.sum():+d}  95%CI=[{np.percentile(m,2.5)*100:+.2f}, {np.percentile(m,97.5)*100:+.2f}]pp  McNemar p={p:.3f} (blend-only={n01}, fav-only={n10})")

print()
print("="*78)
print("2026-10-19 ST — same race day as the live post-mortem (v5: top-1 2/11, top-3 5/11)")
print("="*78)
d19 = R[(R['date']==pd.Timestamp('2026-10-19')) & (R['venue']=='ST')]
if len(d19):
    print(f"  races on 2026-10-19 ST in DB with odds: {len(d19)}")
    print(f"  v6 top-1 hits      : {d19['v6_hit'].sum()}/{len(d19)}")
    print(f"  v6 top-3 (winner in) : {d19['v6_in_top3'].sum()}/{len(d19)}")
    print(f"  favourite top-1    : {d19['fav_hit'].sum()}/{len(d19)}")
    print(f"  favourite top-3    : {d19['fav_in_top3'].sum()}/{len(d19)}")
    print(f"  blend top-1        : {d19['bl_hit'].sum()}/{len(d19)}   blend top-3: {d19['bl_in_top3'].sum()}/{len(d19)}")
    print()
    for _, r in d19.sort_values('race_no').iterrows():
        print(f"   R{r['race_no']}: winner={r['winner']} ({r['winner_odds']}) | fav={r['fav_name']} ({r['fav_odds']})"
              f" | v6={r['v6_name']} ({r['v6_odds']}) {'HIT' if r['v6_hit'] else ''}"
              f" | fav_hit={'Y' if r['fav_hit'] else 'n'}")
else:
    print("  no 2026-10-19 ST races with odds in the test window")

print()
print("="*78)
print("IS THE MARKET ITSELF BEATABLE? — v6 vs favourite on the 2026 subset only")
print("="*78)
r26 = R[R['date']>=pd.Timestamp('2026-01-01')]
print(f"  2026 n={len(r26)}: v6 top1={r26['v6_hit'].sum()} ({r26['v6_hit'].mean()*100:.1f}%)"
      f"  fav top1={r26['fav_hit'].sum()} ({r26['fav_hit'].mean()*100:.1f}%)")
print(f"            v6 top3={r26['v6_in_top3'].sum()} ({r26['v6_in_top3'].mean()*100:.1f}%)"
      f"  fav top3={r26['fav_in_top3'].sum()} ({r26['fav_in_top3'].mean()*100:.1f}%)")

out = {
 'races': int(len(R)),
 'v6_pick_is_favourite_pct': round(float(same.mean()*100),2),
 'v6_top1': int(R['v6_hit'].sum()), 'v6_top3': int(R['v6_in_top3'].sum()),
 'fav_top1': int(R['fav_hit'].sum()), 'fav_top3': int(R['fav_in_top3'].sum()),
 'blend_top1': int(R['bl_hit'].sum()), 'blend_top3': int(R['bl_in_top3'].sum()),
 'disagree_races': int((~same).sum()),
 'disagree_v6_hit': int(R[~same]['v6_hit'].sum()), 'disagree_fav_hit': int(R[~same]['fav_hit'].sum()),
 'disagree_v6_top3': int(R[~same]['v6_in_top3'].sum()), 'disagree_fav_top3': int(R[~same]['fav_in_top3'].sum()),
 'blend_vs_fav_top3_mcnemar_p': round(float(p),4),
}
(ROOT/'output'/'verify_v6_analysis.json').write_text(json.dumps(out, indent=2))
log.info(f"Wrote {ROOT/'output'/'verify_v6_analysis.json'}")
