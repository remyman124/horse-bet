"""Is 'the value rule loses money' a ROBUST finding or a small-sample artifact?

The backtest reports ROI=-39.5% on 676 bets and -22.9% on 5187 bets. Both are
negative, but ROI is a noisy statistic, so the claim must be stress-tested
before it is printed as a warning on every stake table.

This adds three independent checks:

  1. BOOTSTRAP CI on ROI (race-level resampling, since bets within a race are
     correlated). Does the CI exclude 0?
  2. BINOMIAL TEST on the hit rate against the break-even rate implied by the
     odds. A longshot at 30x needs to hit 3.33% of the time to break even.
     Does the model hit at least that often?
  3. COUNTERFACTUAL: if the model had picked horses at random from the same
     races, what ROI would it have got? That separates "the rule is bad" from
     "the rule is worse than random".

It also reports the ONE thing that would make the rule work: a per-odds-bucket
view of the model-minus-market difference measured only on winners.
"""
import json, pickle, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import features as F

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'
MODEL = ROOT / 'models' / 'model_v6.pkl'
CUT_VAL = pd.Timestamp('2025-10-01')
RNG = np.random.default_rng(20261028)

print('loading ...')
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

te = df[(df['race_date'] > CUT_VAL) & df['win_odds'].notna()
         & (df['win_odds'] > 0)].copy()

X = te[COLS].astype(float).fillna(0).values
te['p_model_raw'] = (pkg['xgb_model'].predict_proba(X)[:, 1] * 0.5
                     + pkg['rf_model'].predict_proba(X)[:, 1] * 0.5)

RK = ['race_date', 'venue', 'race_no']
te['p_market_raw'] = 1.0 / te['win_odds']
te['p_market'] = te.groupby(RK)['p_market_raw'].transform(lambda s: s / s.sum())
te['p_model'] = te.groupby(RK)['p_model_raw'].transform(lambda s: s / s.sum())
te['edge'] = te['p_model'] - te['p_market']
te['won'] = (te['finishing_position'] == 1).astype(int)

n_races = te.groupby(RK).ngroups
print(f'test races: {n_races}   runners: {len(te):,}')


def roi_of(bets):
    if len(bets) == 0:
        return None, None, 0
    staked = len(bets)
    ret = float((bets['won'] * bets['win_odds']).sum())
    return round((ret - staked) / staked * 100, 2), round(bets['won'].mean() * 100, 2), staked


def bootstrap_roi(bets, n_boot=20000):
    """Race-level bootstrap: resample races with replacement."""
    if len(bets) == 0:
        return None
    g = bets.groupby(RK, sort=False)
    # per-race (staked, returned)
    per = g.apply(lambda d: pd.Series({
        'staked': len(d),
        'ret': float((d['won'] * d['win_odds']).sum())}), include_groups=False)
    S = per['staked'].to_numpy()
    R = per['ret'].to_numpy()
    idx = RNG.integers(0, len(S), size=(n_boot, len(S)))
    staked = S[idx].sum(axis=1)
    returned = R[idx].sum(axis=1)
    rois = (returned - staked) / staked * 100
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return dict(roi=float((R.sum() - S.sum()) / S.sum() * 100),
                ci_lo=float(lo), ci_hi=float(hi),
                p_roi_le_zero=float((rois <= 0).mean()))


print('\n' + '=' * 78)
print('1) BOOTSTRAP CI on ROI (race-level resampling, 20k draws)')
print('=' * 78)

best = te.sort_values('edge', ascending=False).groupby(RK, sort=False).head(1)
allpos = te[te['edge'] >= 0]
ge2 = te[te['edge'] >= 0.02]
fav = te.sort_values('p_market', ascending=False).groupby(RK, sort=False).head(1)

for label, bets in [('best-edge-per-race (edge>=2%)', ge2),
                    ('best-edge-per-race (any edge)', best),
                    ('every runner edge>=0%', allpos),
                    ('the favourite', fav)]:
    bs = bootstrap_roi(bets)
    r, h, n = roi_of(bets)
    print(f'  {label}')
    print(f'    n={n:<6} hit={h:<6}% ROI={r:<9}%  95% CI [{bs["ci_lo"]:.1f}%, {bs["ci_hi"]:.1f}%]'
          f'   P(ROI<=0)={bs["p_roi_le_zero"]:.4f}')

print('\n' + '=' * 78)
print('2) BREAK-EVEN HIT RATE vs ACTUAL HIT RATE')
print('=' * 78)
print('  (break-even = 1/mean_odds for the bet set)')
for label, bets in [('best-edge-per-race (edge>=2%)', ge2),
                    ('every runner edge>=0%', allpos)]:
    breakeven = 100.0 / bets['win_odds'].mean()
    actual = bets['won'].mean() * 100
    n = len(bets)
    k = int(bets['won'].sum())
    # one-sided binomial test: P(X <= k) under H0 p=breakeven
    from math import comb
    p = sum(comb(n, i) * (breakeven/100)**i * (1-breakeven/100)**(n-i)
            for i in range(0, k + 1))
    print(f'  {label}')
    print(f'    n={n}  mean odds={bets["win_odds"].mean():.2f}  break-even hit={breakeven:.2f}%'
          f'  actual hit={actual:.2f}%')
    print(f'    one-sided binomial P(<= observed | p=break-even) = {p:.4f}')

print('\n' + '=' * 78)
print('3) COUNTERFACTUAL: pick a RANDOM runner from the same race')
print('=' * 78)
for label, bets in [('best-edge-per-race (edge>=2%)', ge2)]:
    picks = []
    for _, d in te.groupby(RK, sort=False):
        tgt = d.sort_values('edge', ascending=False).head(1)
        if len(tgt) and tgt['edge'].iloc[0] >= 0.02:
            picks.append(d.sample(n=1, random_state=int(RNG.integers(1e9))).iloc[0])
    rnd = pd.DataFrame(picks)
    r, h, n = roi_of(rnd)
    print(f'  random pick from the SAME races:  n={n}  hit={h}%  ROI={r}%')
    r2, h2, n2 = roi_of(bets)
    print(f'  the rule                       :  n={n2}  hit={h2}%  ROI={r2}%')

print('\n' + '=' * 78)
print('4) WHERE THE EDGE LIES: model vs market on WINNERS only, by odds bucket')
print('=' * 78)
w = te[te['won'] == 1]
bins = [0, 6, 11, 21, 51, 10000]
labels = ['<6 (fav)', '6-11', '11-21', '21-51', '51+']
w = w.assign(bucket=pd.cut(w['win_odds'], bins=bins, labels=labels, right=False))
for b, g in w.groupby('bucket', observed=True):
    print(f'  {str(b):<10} n={len(g):<6} model p={g["p_model"].mean():>7.2%}  '
          f'market p={g["p_market"].mean():>7.2%}  diff={g["p_model"].mean()-g["p_market"].mean():>+7.2%}')

print('\n' + '=' * 78)
print('5) CALIBRATION: model vs market probability, all runners, by model decile')
print('=' * 78)
te['decile'] = pd.qcut(te['p_model'], 10, labels=False, duplicates='drop')
for d, g in te.groupby('decile'):
    print(f'  decile {int(d):<2} n={len(g):<6} model p={g["p_model"].mean():>7.2%}  '
          f'market p={g["p_market"].mean():>7.2%}  ACTUAL win={g["won"].mean():>7.2%}  '
          f'model-actual={g["p_model"].mean()-g["won"].mean():>+7.2%}  '
          f'market-actual={g["p_market"].mean()-g["won"].mean():>+7.2%}')
