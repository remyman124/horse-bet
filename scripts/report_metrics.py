#!/usr/bin/env python3
"""Single source of truth for the report: one test set, one set of numbers.

Everything the report states about model-vs-market is computed here on the
SAME 636 out-of-sample races, so no table can disagree with another.
"""
import json
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = '/home/work/horse-bet'
TE = '/tmp/te_test.csv'
OUT = '/tmp/report_metrics.json'

df = pd.read_csv(TE)
RK = ['race_date', 'venue', 'race_no']

m_auc = roc_auc_score(df['won'], df['p_model'])
k_auc = roc_auc_score(df['won'], df['p_market'])
races = int(df.groupby(RK).ngroups)


def topn(col, n):
    """Share of races whose winner sits inside the model's/market's top n."""
    sel = df.sort_values(col, ascending=False).groupby(RK).head(n)
    return float(sel.groupby(RK)['won'].sum().gt(0).mean() * 100)


m1, k1 = topn('p_model', 1), topn('p_market', 1)
m3, k3 = topn('p_model', 3), topn('p_market', 3)

# AUC of the blend (market + model) — is there anything in the model to add?
df['blend'] = 0.5 * df['p_model'] + 0.5 * df['p_market']
b_auc = roc_auc_score(df['won'], df['blend'])

# calibration: model p vs market p vs realised win rate, by model decile
df['decile'] = pd.qcut(df['p_model'], 10, labels=False, duplicates='drop')
calib = []
for i, g in df.groupby('decile'):
    calib.append({
        'decile': int(i), 'n': int(len(g)),
        'model_pct': float(g['p_model'].mean() * 100),
        'market_pct': float(g['p_market'].mean() * 100),
        'actual_pct': float(g['won'].mean() * 100),
        'model_err_pp': float((g['p_model'].mean() - g['won'].mean()) * 100),
        'market_err_pp': float((g['p_market'].mean() - g['won'].mean()) * 100),
    })

# the longshot bias, measured on winners only (where it actually costs money)
w = df[df['won'] == 1].copy()
bins = [0, 6, 11, 21, 51, 1e9]
labels = ['<6 (fav)', '6-11', '11-21', '21-51', '51+']
w['bucket'] = pd.cut(w['win_odds'], bins=bins, labels=labels, right=False)
bias = []
for lab in labels:
    g = w[w['bucket'] == lab]
    if not len(g):
        continue
    bias.append({'bucket': lab, 'n': int(len(g)),
                 'model_pct': float(g['p_model'].mean() * 100),
                 'market_pct': float(g['p_market'].mean() * 100),
                 'diff_pp': float((g['p_model'].mean() - g['p_market'].mean()) * 100)})

# what the "value" rule actually buys: the odds it selects
best = df.sort_values('edge', ascending=False).groupby(RK).head(1)
sel = {'n': int(len(best)),
       'odds_median': float(best['win_odds'].median()),
       'odds_p25': float(best['win_odds'].quantile(.25)),
       'odds_p75': float(best['win_odds'].quantile(.75)),
       'odds_max': float(best['win_odds'].max()),
       'hit_pct': float(best['won'].mean() * 100),
       'winners_odds': sorted(float(x) for x in best[best['won'] == 1]['win_odds'])}

out = {
    'test_races': races,
    'test_rows': int(len(df)),
    'test_period': [str(df['race_date'].min()), str(df['race_date'].max())],
    'model_auc': round(float(m_auc), 4),
    'market_auc': round(float(k_auc), 4),
    'blend_auc': round(float(b_auc), 4),
    'model_top1_pct': round(m1, 2),
    'market_top1_pct': round(k1, 2),
    'model_top3_pct': round(m3, 2),
    'market_top3_pct': round(k3, 2),
    'calibration': calib,
    'longshot_bias': bias,
    'value_rule_selection': sel,
}
json.dump(out, open(OUT, 'w'), indent=2)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ('calibration', 'longshot_bias')}, indent=2))
print('\ncalibration:')
for c in calib:
    print(f"  d{c['decile']}: model {c['model_pct']:6.3f}%  market {c['market_pct']:6.3f}%  "
          f"actual {c['actual_pct']:6.3f}%   model-act {c['model_err_pp']:+6.3f}pp   "
          f"mkt-act {c['market_err_pp']:+6.3f}pp")
print('\nlongshot bias (winners):')
for b in bias:
    print(f"  {b['bucket']:8s} n={b['n']:4d}  model {b['model_pct']:6.3f}%  "
          f"market {b['market_pct']:6.3f}%  diff {b['diff_pp']:+6.3f}pp")
print('\nsaved', OUT)
