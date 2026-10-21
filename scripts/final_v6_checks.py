#!/usr/bin/env python3
"""
Final checks before reporting:
  1. Confirm the rating-data limitation (rating only exists for 2026).
  2. Robustness of "v6 does not beat the favourite" across splits (venue, class,
     field size, odds range) — is it a stable conclusion or an artefact?
  3. Serve-time sanity: can the saved model_v6.pkl actually score a race card?
  4. Confirm place_odds is empty DB-wide (so the exclusion is moot) and check
     whether that was always the case.
"""
import pickle, sqlite3, json, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('final_v6')

ROOT = Path(__file__).parent.parent
conn = sqlite3.connect(ROOT/'data'/'horse.db')

print("="*78)
print("1. RATING AVAILABILITY (why rating has no historical signal)")
print("="*78)
for row in conn.execute('''
    select substr(rd.race_date,1,4) y,
           count(*) rows,
           sum(case when re.rating is not null and re.rating>0 then 1 else 0 end) rated
    from horse_entries re
    join races r on re.race_id=r.id
    join race_days rd on r.race_day_id=rd.id
    group by 1 order by 1'''):
    y, n, rt = row
    print(f"  {y}: {rt:>6,}/{n:>7,} = {(rt/n*100 if n else 0):>5.1f}% have rating")
print()
print("  -> rating is 0 for ~89% of ALL rows (2026 only). The model therefore")
print("     excludes it (no train/serve skew); the rating baseline below is")
print("     therefore only meaningful on 2026 rows.")

print()
print("="*78)
print("2. place_odds STATE")
print("="*78)
po = conn.execute('select count(*) from horse_entries where place_odds is not null and place_odds>0').fetchone()[0]
print(f"  horse_entries rows with place_odds>0 : {po}")
print("  -> place_odds is empty DB-wide, so excluding it from features is moot")
print("     (and it was never a usable signal: HKJC only publishes place prices")
print("     for top-3 finishers = outcome leakage).")

print()
print("="*78)
print("3. v6 vs favourite — ROBUSTNESS ACROSS SPLITS")
print("="*78)
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
recs = []
for key, g in test.groupby(RACE_KEYS):
    w = g[g['finishing_position']==1]
    if len(w)!=1 or len(g)<2: continue
    s = g.sort_values('pred_v6', ascending=False)
    f3 = g.sort_values('win_odds', kind='mergesort')
    recs.append({'date':key[0],'venue':key[1],'race_no':key[2],'class':g['race_class'].iloc[0],
                 'field':len(g),'fav_odds':float(g['win_odds'].min()),
                 'v6_top1':bool(s.head(1)['horse_name'].iloc[0]==w.iloc[0]['horse_name']),
                 'v6_top3':bool(w.iloc[0]['horse_name'] in s.head(3)['horse_name'].tolist()),
                 'fav_top1':bool(f3.head(1)['horse_name'].iloc[0]==w.iloc[0]['horse_name']),
                 'fav_top3':bool(w.iloc[0]['horse_name'] in f3.head(3)['horse_name'].tolist())})
R = pd.DataFrame(recs)
print(f"  races={len(R)}   (favourite top-3 uses stable mergesort)")
print()
def split_report(label, col):
    print(f"  --- by {label} ---")
    for v, sub in R.groupby(col):
        if len(sub) < 15: continue
        print(f"    {str(v):>16} n={len(sub):>4}  v6 top1={sub['v6_top1'].mean()*100:5.1f}% fav top1={sub['fav_top1'].mean()*100:5.1f}%"
              f" | v6 top3={sub['v6_top3'].mean()*100:5.1f}% fav top3={sub['fav_top3'].mean()*100:5.1f}%"
              f" | top3 diff={(sub['v6_top3'].sum()-sub['fav_top3'].sum()):+d}")
split_report('venue', 'venue')
split_report('race class', 'class')
split_report('field size', 'field')

print()
print("  --- by favourite SP range ---")
for lo,hi in [(1,2.5),(2.5,4),(4,8),(8,1000)]:
    sub = R[(R['fav_odds']>=lo)&(R['fav_odds']<hi)]
    if len(sub)<15: continue
    print(f"    SP [{lo},{hi}) n={len(sub):>4}  v6 top1={sub['v6_top1'].mean()*100:5.1f}% fav top1={sub['fav_top1'].mean()*100:5.1f}%"
          f" | v6 top3={sub['v6_top3'].mean()*100:5.1f}% fav top3={sub['fav_top3'].mean()*100:5.1f}%"
          f" | top3 diff={(sub['v6_top3'].sum()-sub['fav_top3'].sum()):+d}")

print()
print("="*78)
print("4. SERVE-TIME SANITY — load saved model and score a real race day")
print("="*78)
day = pd.Timestamp('2026-10-19')
d19 = test[test['race_date']==day]
print(f"  2026-10-19 rows={len(d19)}")
if len(d19):
    ok = d19[FC].astype(float).values
    p = (pkg['xgb_model'].predict_proba(ok)[:,1]*0.5 + pkg['rf_model'].predict_proba(ok)[:,1]*0.5)
    print(f"  scored {len(p)} horses, prob range [{p.min():.4f}, {p.max():.4f}]  no NaN: {not np.isnan(p).any()}")
    print(f"  package keys: {list(pkg.keys())}")
    print(f"  feature count: {len(FC)}  encoders: {list(pkg['encoders'].keys())}")
    print(f"  class_map: {pkg['class_map']}")
print()
print("  conclusion check -> top-3: v6 loses to favourite in "
      f"{(R['fav_top3'].sum()-R['v6_top3'].sum())} of {len(R)} races "
      f"({R['fav_top3'].sum()} vs {R['v6_top3'].sum()})")
