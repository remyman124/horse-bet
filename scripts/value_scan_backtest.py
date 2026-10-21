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
edge_buckets = []
for lab in labels:
    g = w[w['bucket'] == lab]
    if not len(g):
        continue
    print(f"  {lab:10s} n={len(g):>5}  model mean p={g['p_model'].mean():.3%}  "
          f"market mean p={g['p_market'].mean():.3%}  "
          f"diff={g['p_model'].mean()-g['p_market'].mean():+.3%}")
    edge_buckets.append({
        'bucket': lab,
        'n': int(len(g)),
        'model_pct': float(g['p_model'].mean() * 100),
        'market_pct': float(g['p_market'].mean() * 100),
        'diff_pp': float((g['p_model'].mean() - g['p_market'].mean()) * 100),
    })

print('\n' + '=' * 74)
print('F) CALIBRATION — model probability vs realised win rate, by decile')
print('   (if the model were calibrated, model p should equal the win rate)')
print('=' * 74)
te['decile'] = pd.qcut(te['p_model'], 10, labels=False, duplicates='drop')
calib = []
for i, g in te.groupby('decile'):
    row = {'decile': int(i), 'n': int(len(g)),
           'model_p_pct': float(g['p_model'].mean() * 100),
           'market_p_pct': float(g['p_market'].mean() * 100),
           'actual_win_pct': float(g['won'].mean() * 100)}
    row['model_minus_actual_pp'] = row['model_p_pct'] - row['actual_win_pct']
    row['market_minus_actual_pp'] = row['market_p_pct'] - row['actual_win_pct']
    calib.append(row)
    print(f"  decile {i}: model p={row['model_p_pct']:6.3f}%  "
          f"market p={row['market_p_pct']:6.3f}%  actual={row['actual_win_pct']:6.3f}%  "
          f"model-act={row['model_minus_actual_pp']:+6.3f}pp  "
          f"mkt-act={row['market_minus_actual_pp']:+6.3f}pp")

# the headline: where does the model's edge actually come from?
bl = te[te['won'] == 1]
bias = []
for lab in labels:
    g = bl[bl['bucket'] == lab]
    if not len(g):
        continue
    bias.append({'bucket': lab, 'n': int(len(g)),
                 'model_p_pct': float(g['p_model'].mean() * 100),
                 'market_p_pct': float(g['p_market'].mean() * 100),
                 'diff_pp': float((g['p_model'].mean() - g['p_market'].mean()) * 100)})

# ── H. head-to-head on the SAME races (model top-1 vs market favourite) ──
print('\n' + '=' * 74)
print('H) HEAD TO HEAD on the same races')
print('=' * 74)
m_top1 = te.sort_values('p_model', ascending=False).groupby(RK).head(1)
k_top1 = te.sort_values('p_market', ascending=False).groupby(RK).head(1)
m_hit = float(m_top1['won'].mean() * 100)
k_hit = float(k_top1['won'].mean() * 100)
m_top3 = (te.sort_values('p_model', ascending=False).groupby(RK)
           .head(3).groupby(RK)['won'].sum().eq(3).mean() * 100)
k_top3 = (te.sort_values('p_market', ascending=False).groupby(RK)
           .head(3).groupby(RK)['won'].sum().eq(3).mean() * 100)
n_r = int(te.groupby(RK).ngroups)
print(f'  races: {n_r}')
print(f'  model  top1 {m_hit:.2f}%   top3 {m_top3:.2f}%')
print(f'  market top1 {k_hit:.2f}%   top3 {k_top3:.2f}%')
head_to_head = {'races': n_r,
                'model_top1_pct': round(m_hit, 2), 'market_top1_pct': round(k_hit, 2),
                'model_top3_pct': round(float(m_top3), 2),
                'market_top3_pct': round(float(k_top3), 2)}

# ── G: diagnostic on the selected bets (odds distribution + winners) ──────
print('\n' + '=' * 74)
print('G) DIAGNOSTIC — odds distribution of the selected bets')
print('=' * 74)
sel_diag = {}
for thr in (0.00, 0.02, 0.03):
    b = te[te['edge'] >= thr]
    if not len(b):
        continue
    w = b[b['won'] == 1]
    sel_diag[f'edge>={thr:.0%}'] = {
        'n': int(len(b)), 'wins': int(len(w)),
        'staked': int(len(b)), 'returned': round(float((w['win_odds']).sum()), 2),
        'odds_min': float(b['win_odds'].min()), 'odds_max': float(b['win_odds'].max()),
        'odds_median': float(b['win_odds'].median()),
        'winner_odds': [float(x) for x in w['win_odds']],
    }
    print(f"  edge>={thr:.0%}: n={len(b):5d} wins={len(w):3d} "
          f"returned={w['win_odds'].sum():9.1f} odds[{b['win_odds'].min():.1f}..{b['win_odds'].max():.1f}] "
          f"median={b['win_odds'].median():.1f}  winner odds={sorted(float(x) for x in w['win_odds'])}")

# best-edge-per-race: which runner is actually selected?
best = te.sort_values('edge', ascending=False).groupby(RK).head(1)
bw = best[best['won'] == 1]
print(f"\n  best-edge-per-race: n={len(best)} wins={len(bw)} "
      f"returned={bw['win_odds'].sum():.1f} "
      f"odds[{best['win_odds'].min():.1f}..{best['win_odds'].max():.1f}] "
      f"median={best['win_odds'].median():.1f}")
print(f"    selected odds histogram: "
      f"{best['win_odds'].describe()[['min','25%','50%','75%','max']].round(1).to_dict()}")
print(f"    winner odds: {sorted(float(x) for x in bw['win_odds'])}")
sel_diag['best_edge_per_race'] = {
    'n': int(len(best)), 'wins': int(len(bw)),
    'returned': round(float(bw['win_odds'].sum()), 2),
    'odds_min': float(best['win_odds'].min()), 'odds_max': float(best['win_odds'].max()),
    'odds_median': float(best['win_odds'].median()),
    'winner_odds': [float(x) for x in bw['win_odds']],
    'n_by_race': int(len(best)),
}
te.to_csv('/tmp/te_test.csv', index=False)
print('\n  saved test frame -> /tmp/te_test.csv')

json.dump({'threshold_scan': rows,
           'best_edge_per_race': roi(cand),
           'favourite_baseline': rf,
           'edge_by_odds_bucket': edge_buckets,
           'model_calibration': calib,
           'head_to_head': head_to_head,
           'selection_diagnostic': sel_diag,
           'test_races': int(n_races),
           'test_rows_with_odds': int(len(te)),
           'all_runners_baseline': allb},
          open(ROOT / 'output' / 'value_scan_clean.json', 'w'), indent=2)
print(f"\nsaved: {ROOT / 'output' / 'value_scan_clean.json'}")
