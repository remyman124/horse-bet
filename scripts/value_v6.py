#!/usr/bin/env python3
"""
Third pass — three questions that decide whether v6 is USABLE:

Q1. TIE-BREAK SENSITIVITY. The favourite's top-3 came out 301 in verify_v6.py
    and 304 in analyse_v6.py. Same 503 races. Why? Likely ties in win_odds at
    the 3rd-pick boundary, resolved differently by quicksort vs stable order.
    Quantify the full range so the headline number is stated honestly.

Q2. VALUE BETTING. Flat staking on the top-1 pick loses money for EVERY
    strategy (-16% to -36%) because HKJC's takeout is ~20%+. The only way a
    pre-race model makes money is by finding horses whose TRUE probability
    exceeds the market's implied probability, and betting only those.
    Test: does v6's probability beat 1/odds? And does betting the disagreements
    make money?

Q3. Calibre of the model's probability: is it even usable as a price?
    Brier score / calibration of v6 vs the market's implied probability.
"""
import pickle, sqlite3, json, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('value_v6')

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
groups = []
for key, g in test.groupby(RACE_KEYS):
    w = g[g['finishing_position']==1]
    if len(w)!=1 or len(g)<2: continue
    groups.append((key, g, w.iloc[0]))
log.info(f"races: {len(groups)}")

# ── Q1: tie-break sensitivity of the favourite's top-3 ─────────────────────
print()
print("="*78)
print("Q1: TIE-BREAK SENSITIVITY — favourite top-3 under different tie-breaks")
print("="*78)
tie_races = 0
for key, g, w in groups:
    s = g.sort_values('win_odds')
    third = s['win_odds'].iloc[2]
    if (s['win_odds']==third).sum() > 1:
        tie_races += 1
print(f"  races where the 3rd-lowest odds is tied: {tie_races}/{len(groups)}")

variants = {}
# (a) stable sort by odds ascending (keeps DB order within ties)
h=0
for key, g, w in groups:
    s = g.sort_values('win_odds', kind='mergesort')
    if w['horse_name'] in s.head(3)['horse_name'].tolist(): h+=1
variants['stable (mergesort)'] = h
# (b) quicksort (what pandas default does — non-deterministic across ties)
h=0
for key, g, w in groups:
    s = g.sort_values('win_odds')
    if w['horse_name'] in s.head(3)['horse_name'].tolist(): h+=1
variants['quicksort (default)'] = h
# (c) tie-break by v6 probability
h=0
for key, g, w in groups:
    s = g.sort_values(['win_odds','pred_v6'], ascending=[True,False], kind='mergesort')
    if w['horse_name'] in s.head(3)['horse_name'].tolist(): h+=1
variants['odds then v6 prob'] = h
# (d) tie-break by horse_no
h=0
for key, g, w in groups:
    s = g.sort_values(['win_odds','horse_no'], ascending=[True,True], kind='mergesort')
    if w['horse_name'] in s.head(3)['horse_name'].tolist(): h+=1
variants['odds then horse_no'] = h
# (e) optimistic: any horse tied at the 3rd boundary counts as in top-3
h=0
for key, g, w in groups:
    s = g.sort_values('win_odds')
    third = s['win_odds'].iloc[2]
    if (s['win_odds']<third).sum() < 3:  # winner strictly inside top3
        if w['win_odds'] <= third: h+=1
    else:
        h+=1
variants['optimistic (ties count)'] = h
for k,v in variants.items():
    print(f"  favourite top-3 [{k:24s}] = {v}/{len(groups)} ({v/len(groups)*100:.2f}%)")

# v6 top-3 for comparison
hv=0
for key, g, w in groups:
    s = g.sort_values('pred_v6', ascending=False)
    if w['horse_name'] in s.head(3)['horse_name'].tolist(): hv+=1
print(f"  v6        top-3 [stable]                = {hv}/{len(groups)} ({hv/len(groups)*100:.2f}%)")

# ── Q2: value betting ──────────────────────────────────────────────────────
print()
print("="*78)
print("Q2: VALUE BETTING — bet only when v6 prob > market implied prob")
print("="*78)
rows = []
for key, g, w in groups:
    for _, r in g.iterrows():
        imp = 1.0/r['win_odds']
        rows.append({'race': key, 'horse': r['horse_name'], 'odds': r['win_odds'],
                     'implied': imp, 'p_v6': r['pred_v6'], 'won': int(r['finishing_position']==1),
                     'is_fav': bool(r['win_odds']==g['win_odds'].min())})
H = pd.DataFrame(rows)
H['edge'] = H['p_v6'] - H['implied']

for thr in [0.0, 0.02, 0.05, 0.10]:
    sel = H[H['edge'] > thr]
    if len(sel)==0:
        print(f"  edge > {thr:+.2f}: no bets"); continue
    ret = (sel['odds']*sel['won']).sum() - len(sel)
    wins = sel['won'].sum()
    print(f"  edge > {thr:+.2f}: n={len(sel):>4} bets  wins={wins:>3} ({wins/len(sel)*100:5.1f}%)"
          f"  ROI={ret/len(sel)*100:+7.2f}%   (model said {sel['p_v6'].mean()*100:.1f}%, market said {sel['implied'].mean()*100:.1f}%)")

# also: value bet on the FAVOURITE only (market's own favourites with an edge)
print()
print("  --- restricted to the race favourite ---")
for thr in [0.0, 0.03, 0.05]:
    sel = H[(H['edge']>thr) & H['is_fav']]
    if len(sel)==0:
        print(f"  fav & edge > {thr:+.2f}: no bets"); continue
    ret = (sel['odds']*sel['won']).sum() - len(sel)
    print(f"  fav & edge > {thr:+.2f}: n={len(sel):>4}  wins={sel['won'].sum():>3}  ROI={ret/len(sel)*100:+7.2f}%")

# ── Q3: calibration ────────────────────────────────────────────────────────
print()
print("="*78)
print("Q3: CALIBRATION — is v6 usable as a PRICE?")
print("="*78)
print(f"  v6    AUC={roc_auc_score(H['won'], H['p_v6']):.4f}  Brier={brier_score_loss(H['won'], H['p_v6']):.4f}"
      f"  mean_pred={H['p_v6'].mean():.4f}  actual={H['won'].mean():.4f}")
print(f"  mkt   AUC={roc_auc_score(H['won'], H['implied']):.4f}  Brier={brier_score_loss(H['won'], H['implied']):.4f}"
      f"  mean_pred={H['implied'].mean():.4f}  actual={H['won'].mean():.4f}")
print()
print("  reliability of v6 (10 bins):")
try:
    from sklearn.calibration import calibration_curve
    pt, pp = calibration_curve(H['won'], H['p_v6'], n_bins=10, strategy='quantile')
    for a,b in zip(pp, pt):
        print(f"    predicted {a*100:5.1f}%  ->  actual {b*100:5.1f}%")
except Exception as e:
    print("   calibration curve failed:", e)

out = {
 'races': len(groups),
 'favourite_top3_variants': variants,
 'v6_top3': hv,
 'value_bet_scan': {str(t): {'n': int(len(H[H['edge']>t])), 'roi_pct': round(float(((H[H['edge']>t]['odds']*H[H['edge']>t]['won']).sum()-len(H[H['edge']>t]))/max(len(H[H['edge']>t]),1)*100),2)}
                    for t in [0.0,0.02,0.05,0.10]},
 'auc_v6': round(float(roc_auc_score(H['won'], H['p_v6'])),4),
 'auc_market': round(float(roc_auc_score(H['won'], H['implied'])),4),
 'brier_v6': round(float(brier_score_loss(H['won'], H['p_v6'])),4),
 'brier_market': round(float(brier_score_loss(H['won'], H['implied'])),4),
}
(ROOT/'output'/'verify_v6_value.json').write_text(json.dumps(out, indent=2))
log.info(f"Wrote {ROOT/'output'/'verify_v6_value.json'}")
