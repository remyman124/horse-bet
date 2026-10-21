"""
Value-filter backtest on the CLEAN model — the evidence that decides whether
tonight's "positive edge" candidates are worth betting.

Why this script exists
-----------------------
The v6 model has no demonstrated top-1 edge over the market:
  top-1  168/597 = 28.14%  vs  favourite 167/597 = 27.97%   (+1 race)
  top-3  363/597 = 60.80%  vs  favourite 363/597 = 60.80%   (tied)
  AUC         0.7781        vs  market implied-prob 0.7889   (worse)
So "which horse wins" is the wrong question. The right question is whether
the model's *probability differences* vs the market are worth more than the
vig. That is what this measures.

Methodology (the part that is easy to get wrong)
-------------------------------------------------
1. Both sides are normalised WITHIN a race. The model's raw probabilities
   sum to ~0.92-1.08 per race, the market's implied probs sum to 1/overround.
   Comparing raw model prob against normalised market prob manufactures
   fake edge: longshots get a free boost, favourites a free haircut.
2. The "race" identity is (race_date, venue, race_no) — never race_no alone.
3. Flat stake 1 unit per bet, ROI = (returned - staked) / staked, using the
   published win odds as the return. No compounding, no best-price.
4. The exact rule that produced tonight's candidates is also measured:
   "bet the single highest-edge runner in each race" — that is what the
   shortlist actually is.

Test set: the same temporal split retrain_v6.py used — race_date > 2025-10-01.
The saved model was trained on <= 2025-06-01 with 2025-06..2025-10 as the
early-stopping set, so the test window is genuinely out of sample.
"""

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import features as F

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'
MODEL = ROOT / 'models' / 'model_v6.pkl'
CUT_VAL = pd.Timestamp('2025-10-01')

print('loading entries ...')
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
    ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
''', conn)
conn.close()

df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce').fillna(0).astype(int)
df['draw'] = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
df['race_date'] = pd.to_datetime(df['race_date'])

df, FEATURE_LIST = F.build_features(df)
pkg = pickle.load(open(MODEL, 'rb'))
COLS = pkg['feature_cols']
print(f'model features: {len(COLS)} | rows: {len(df):,}')

te = df[(df['race_date'] > CUT_VAL) & df['win_odds'].notna()
         & (df['win_odds'] > 0)].copy()
print(f'test rows with odds: {len(te):,}  '
      f'({te.race_date.min().date()} .. {te.race_date.max().date()})')

X = te[COLS].astype(float).fillna(0).values
te['p_model_raw'] = (pkg['xgb_model'].predict_proba(X)[:, 1] * 0.5
                     + pkg['rf_model'].predict_proba(X)[:, 1] * 0.5)

# normalise BOTH sides within the race
RK = ['race_date', 'venue', 'race_no']
te['p_market_raw'] = 1.0 / te['win_odds']
te['p_market'] = te.groupby(RK)['p_market_raw'].transform(lambda s: s / s.sum())
te['p_model'] = te.groupby(RK)['p_model_raw'].transform(lambda s: s / s.sum())
te['edge'] = te['p_model'] - te['p_market']
te['won'] = (te['finishing_position'] == 1).astype(int)

print('\n=== per-race probability mass (model raw vs market) ===')
mass = te.groupby(RK).agg(mdl=('p_model_raw', 'sum'), mkt=('p_market_raw', 'sum'))
print(f'  model raw mass: mean {mass.mdl.mean():.4f}  min {mass.mdl.min():.4f}  max {mass.mdl.max():.4f}')
print(f'  market raw mass (1/overround): mean {mass.mkt.mean():.4f}  '
      f'min {mass.mkt.min():.4f}  max {mass.mkt.max():.4f}')

n_races = te.groupby(RK).ngroups
print(f'  races in test set: {n_races}')


def roi(bets):
    """Flat 1-unit stake, win odds as the return."""
    if len(bets) == 0:
        return dict(n=0, roi_pct=None, hit_pct=None, staked=0, returned=0.0)
    staked = len(bets)
    returned = float((bets['won'] * bets['win_odds']).sum())
    return dict(n=int(staked),
                roi_pct=round((returned - staked) / staked * 100, 2),
                hit_pct=round(bets['won'].mean() * 100, 2),
                staked=int(staked),
                returned=round(returned, 2))


print('\n' + '=' * 74)
print('A) RULE: bet every runner whose edge >= threshold (model > market)')
print('=' * 74)
rows = []
for thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
    b = te[te['edge'] >= thr]
    r = roi(b)
    rows.append({'thr': thr, **r})
    print(f"  edge >= {thr:>5.0%}: n={r['n']:>5}  hit={r['hit_pct']}  ROI={r['roi_pct']}%")

print('\n' + '=' * 74)
print('B) SAME RULE, split by odds bucket (where is the edge coming from?)')
print('=' * 74)
bins = [0, 6, 11, 21, 51, 10000]
labels = ['<6 (fav)', '6-11', '11-21', '21-51', '51+']
te['bucket'] = pd.cut(te['win_odds'], bins=bins, labels=labels, right=False)
for thr in [0.00, 0.02, 0.04]:
    print(f'\n  --- edge >= {thr:.0%} ---')
    for lab in labels:
        b = te[(te['edge'] >= thr) & (te['bucket'] == lab)]
        r = roi(b)
        if r['n']:
            print(f"    {lab:10s} n={r['n']:>5}  hit={r['hit_pct']}%  ROI={r['roi_pct']}%")

print('\n' + '=' * 74)
print('C) RULE THAT PRODUCED TONIGHT\'S SHORTLIST:')
print('   bet the single highest-edge runner in each race (edge >= 2%)')
print('=' * 74)
cand = (te[te['edge'] >= 0.02]
        .sort_values('edge', ascending=False)
        .groupby(RK)
        .head(1))
r = roi(cand)
print(f"  best-edge-per-race, edge>=2%:  n={r['n']}  hit={r['hit_pct']}%  ROI={r['roi_pct']}%")
cand1 = (te.sort_values('edge', ascending=False).groupby(RK).head(1))
r1 = roi(cand1)
print(f"  best-edge-per-race, any edge:   n={r1['n']}  hit={r1['hit_pct']}%  ROI={r1['roi_pct']}%")

print('\n  --- same rule, restricted to longshots (odds >= 14) ---')
for thr in [0.02, 0.03, 0.04]:
    c = (te[(te['edge'] >= thr) & (te['win_odds'] >= 14)]
         .sort_values('edge', ascending=False).groupby(RK).head(1))
    rr = roi(c)
    print(f"    longshot best-edge, edge>={thr:.0%}: n={rr['n']}  hit={rr['hit_pct']}%  ROI={rr['roi_pct']}%")

print('\n' + '=' * 74)
print('D) BASELINES on the same races (flat stake, 1 unit per race)')
print('=' * 74)
fav = te.loc[te.groupby(RK)['win_odds'].idxmin()]
rf = roi(fav)
print(f"  favourite every race:  n={rf['n']}  hit={rf['hit_pct']}%  ROI={rf['roi_pct']}%")
allb = roi(te)
print(f"  every runner:          n={allb['n']}  hit={allb['hit_pct']}%  ROI={allb['roi_pct']}%")

print('\n' + '=' * 74)
print('E) MODEL EDGE BY ODDS BUCKET — is the edge a real skill or a')
print('   systematic longshot bias? (model prob vs market prob, winners only)')
print('=' * 74)
w = te[te['won'] == 1]
for lab in labels:
    g = w[w['bucket'] == lab]
    if not len(g):
        continue
    print(f"  {lab:10s} n={len(g):>5}  model mean p={g['p_model'].mean():.3%}  "
          f"market mean p={g['p_market'].mean():.3%}  "
          f"diff={g['p_model'].mean()-g['p_market'].mean():+.3%}")

json.dump({'threshold_scan': rows,
           'best_edge_per_race': roi(cand),
           'favourite_baseline': rf},
          open(ROOT / 'output' / 'value_scan_clean.json', 'w'), indent=2)
print(f"\nsaved: {ROOT / 'output' / 'value_scan_clean.json'}")
