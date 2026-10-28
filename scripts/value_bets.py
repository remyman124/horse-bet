"""
Per-horse value analysis for an upcoming race card.

This is the same TRAIN→SERVE path as `predict_racecard.py` (same history
concat, same feature builder, same model), extended to produce a
per-runner probability so we can compare the model against the market
horse by horse instead of only race by race.

Why per-horse matters
----------------------
The v6 model has NO demonstrated top-1 edge over the market
(top-1 168/597 = 28.14% vs favourite 167/597 = 27.97%; top-3 tied at
363/597; AUC 0.7781 < market 0.7889). So "which horse wins" is the wrong
question to ask a model like this. The right question is "where does the
model's probability differ from the market's, by enough to pay for the
vig?" — i.e. a value / no-bet filter.

Market reference
----------------
`1 / published_odds` is the market's raw implied probability, but the
book carries an overround, so the raw numbers sum to > 1. We normalise
within each race: that removes the overround and leaves the market's
*relative* view of the field, which is the correct reference for the
model's own normalised probability.

Usage
-----
    python3 scripts/value_bets.py --csv output/card_2026_10_21.csv \
        --date 2026-10-21 --venue HV \
        --out output/value_2026_10_21.csv
"""

import argparse
import os
import pickle
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import features as F
from predict_racecard import DB, load_history, build_card

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyse(date, venue, csv_path, model_path, min_edge=0.02):
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    hist = load_history(conn, date, venue)
    card = build_card(date, venue, csv_path)
    conn.close()
    print(f"history rows: {len(hist):,} | card runners: {len(card)}")

    if 'distance' in card.columns and card['distance'].notna().any():
        card['distance'] = pd.to_numeric(card['distance'], errors='coerce')
        if 'race_class' not in card.columns:
            card['race_class'] = ''
        if 'track' not in card.columns:
            card['track'] = 'Turf'
    else:
        meta = (hist[hist['venue'] == venue]
                .sort_values('race_date')
                .groupby('race_no')
                .agg(distance=('distance', 'last'), race_class=('race_class', 'last'),
                     track=('track', 'last')))
        card = card.merge(meta, left_on='race_no', right_index=True, how='left')

    combined = pd.concat([hist, card], ignore_index=True, sort=False)
    feat_df, feat_list = F.build_features(combined)
    served = feat_df[feat_df['race_date'] == pd.Timestamp(date)].copy()

    pkg = pickle.load(open(model_path, 'rb'))
    model_feats = pkg['feature_cols']
    X = served[model_feats].astype(float).fillna(0).values
    served['p_model'] = (pkg['xgb_model'].predict_proba(X)[:, 1] * 0.5
                          + pkg['rf_model'].predict_proba(X)[:, 1] * 0.5)

    # market implied probability, normalised within the race (removes overround)
    served['p_market_raw'] = 1.0 / served['win_odds']
    served['p_market'] = served.groupby('race_no')['p_market_raw'].transform(
        lambda s: s / s.sum())
    served['overround'] = served.groupby('race_no')['p_market_raw'].transform('sum')

    # The model's raw probabilities do NOT sum to 1 within a race (they run
    # ~0.92-1.08 depending on the field). Comparing raw model probability
    # against a normalised market probability would manufacture edge out of
    # that flatness: longshots get a free boost and favourites a free haircut.
    # Normalise the model within the race too, so both sides are "probability
    # mass spread over this field" and the comparison is like for like.
    served['p_model_raw'] = served['p_model']
    served['p_model'] = served.groupby('race_no')['p_model_raw'].transform(
        lambda s: s / s.sum())
    served['model_mass'] = served.groupby('race_no')['p_model_raw'].transform('sum')
    served['edge'] = served['p_model'] - served['p_market']

    # Full Kelly, then a quarter-Kelly stake -- the honest sizing for a model
    # with no proven edge.
    #
    # b is the NET odds per unit staked: bet 1, win o, profit o-1.
    # So b = o - 1 and the Kelly fraction is f* = (p*(b+1) - 1) / b.
    #
    # The old code used b = (o-1)/o, which is the RETURN FRACTION, not the
    # net odds. That silently made the break-even p > o/(2o-1) -- about 0.51
    # at o=2 and still above 0.5 at o=40 -- so Kelly collapsed to 0 for
    # every realistic longshot and the script reported "quarter-Kelly=0.00%"
    # for all 19 of tonight's candidates. A stake of zero for a horse with
    # +3.7% edge is the opposite of what the number is supposed to say, and
    # it hid the real finding: see value_scan_backtest.py.
    def kelly(p, o):
        b = o - 1.0
        if b <= 0:
            return 0.0
        f = (p * (b + 1) - 1) / b
        return max(0.0, f)

    served['kelly_full'] = [kelly(p, o) for p, o in
                            zip(served['p_model'], served['win_odds'])]
    served['kelly_quarter'] = served['kelly_full'] / 4.0

    served['model_rank'] = served.groupby('race_no')['p_model'].rank(ascending=False).astype(int)
    served['market_rank'] = served.groupby('race_no')['p_market'].rank(ascending=False).astype(int)
    served['is_favourite'] = served['market_rank'] == 1

    cols = ['race_no', 'horse_no', 'horse_name', 'jockey', 'trainer', 'draw',
            'win_odds', 'p_model', 'p_market', 'edge', 'model_rank',
            'market_rank', 'is_favourite', 'kelly_full', 'kelly_quarter',
            'overround']
    out = served[cols].sort_values(['race_no', 'model_rank'])

    print('\n' + '=' * 78)
    for rn, g in out.groupby('race_no'):
        g = g.sort_values('model_rank')
        fav = g[g['is_favourite']].iloc[0]
        top1 = g.iloc[0]
        ov = g['overround'].iloc[0]
        print(f"\nR{rn}  overround {ov:.3f}   model top1: {top1['horse_name']} "
              f"({top1['win_odds']}) p={top1['p_model']:.2%}  |  "
              f"favourite: {fav['horse_name']} ({fav['win_odds']}) p={fav['p_model']:.2%}")
        for _, r in g.iterrows():
            flag = ' <<<' if r['edge'] >= min_edge else ''
            print(f"   {r['horse_name'][:26]:26s} {r['win_odds']:>6} "
                  f"mkt={r['p_market']:6.2%} mdl={r['p_model']:6.2%} "
                  f"edge={r['edge']:+6.2%} "
                  f"kq={r['kelly_quarter']:5.2%}{flag}")

    print('\n' + '=' * 78)
    print(f'VALUE CANDIDATES (edge >= {min_edge:.1%}), best first:')
    cands = out[out['edge'] >= min_edge].sort_values('edge', ascending=False)
    if cands.empty:
        print('   (none — no runner clears the edge bar)')
    for _, r in cands.iterrows():
        print(f"   R{r['race_no']} {r['horse_name'][:26]:26s} @{r['win_odds']:>6} "
              f"mkt={r['p_market']:.2%} mdl={r['p_model']:.2%} edge={r['edge']:+.2%} "
              f"quarter-Kelly={r['kelly_quarter']:.2%}")

    print('\n' + '=' * 78)
    print('READ THIS BEFORE BETTING ANY OF THE ABOVE')
    print('-' * 78)
    print('The stake above is a Kelly fraction, NOT a recommendation. The rule')
    print('that produces these candidates -- "bet every runner whose model')
    print('probability exceeds the market\'s" -- was backtested out of sample')
    print('by scripts/value_scan_backtest.py and it LOSES MONEY:')
    print()
    print('  best-edge-per-race, edge>=2%:  n=676  hit=0.89%  ROI=-39.5%')
    print('  every runner with edge>=0%:    n=5187 hit=3.07%  ROI=-22.9%')
    print('  betting the favourite instead: n=678  hit=28.0%  ROI=-20.6%')
    print()
    print('The "edge" is a systematic longshot bias, not skill. Measured on')
    print('winners, the model over-rates longshots and under-rates favourites:')
    print('  odds 51+   model p=3.95%  market p=1.18%   (model +2.77pp)')
    print('  odds <6    model p=19.1%  market p=26.0%   (model -6.86pp)')
    print('And on calibration, the market is nearly unbiased in every decile')
    print('while the model is over-confident at both tails.')
    print()
    print('CONCLUSION: no bet. A positive model-vs-market edge on a longshot')
    print('is the signature of a mis-calibrated model, and the money follows')
    print('that, not the edge.')
    print('=' * 78)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--date', required=True)
    ap.add_argument('--venue', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'model_v6.pkl'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-edge', type=float, default=0.02)
    a = ap.parse_args()
    out = analyse(a.date, a.venue, a.csv, a.model, a.min_edge)
    path = a.out or os.path.join(ROOT, 'output', f"value_{a.date.replace('-', '_')}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nsaved: {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
