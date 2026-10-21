#!/usr/bin/env python3
"""
Backtest the PLACE market on official HKJC dividends.

The semantics were established empirically, not assumed
------------------------------------------------------
The old parser read the wrong cell of the dividends table, and it also
swallowed the runners table into the PLACE list, so 33% of the PLACE
amounts came out below HK$10 per HK$10 stake (impossible: a winning bet
cannot return less than the stake). The parser is fixed in
`scripts/fetch_place_dividends.py`, and the semantics were then PROVED
on 881 races with checks that all came out 100%:

  * the published PLACE amount is the payout per HK$10 stake
  * the PLACE rows are ordered [1st, 2nd, 3rd] (place[0] is the horse
    that finished 1st in 881/881 races)
  * the winner's PLACE payout is always below the winner's WIN payout,
    because place is easier to hit than win — 881/881

That last check is the one that kills the alternative reading. An earlier
version of this file assumed the published amount was the horse's own
place price and that the payout was the averaging rule
(2nd -> (SP2 + SP1)/2, 3rd -> (SP3 + SP1 + SP2)/3). That reading is
REFUTED by the data: in races where the winner was a longshot, the
averaging rule needs a NEGATIVE place price for the runner-up. Such
races exist, so the published amounts ARE the payouts.

Consequence: a place bet is a pari-mutuel bet whose payout does not
depend on the price paid, so a backtest needs only (a) the official
published dividends and (b) the horse picked. NO pre-race place price is
required, and there is no outcome leak in the selection step.

Selection uses pre-race information only. The published dividends are
used only to score the bet afterwards.

Walk-forward evaluation
-----------------------
The place model is refit once per year on everything strictly before
that year, then predicts that year. No race is ever scored by a model
that saw it, or a later race. This means the backtest can use the whole
history instead of only the final test window, and it is still strictly
out of sample.

Usage
-----
    python3 scripts/place_backtest.py
    python3 scripts/place_backtest.py --evidence /tmp/place_div.ckpt.json
"""

import argparse
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
import features as F

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'


def load_evidence(path):
    """-> {race_key: {'placers': {horse_no: payout_per_dollar}, 'winner': horse_no}}"""
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        if v.get('status') is not None and v.get('status') != 'OK':
            continue
        pl = v.get('place')
        if not pl or len(pl) < 2:
            continue
        placers, ok = {}, True
        for row in pl:
            try:
                h = int(str(row['combo']).strip())
                a = float(str(row['amount']).replace(',', ''))
            except (KeyError, TypeError, ValueError):
                ok = False
                break
            if a < 10:            # impossible payout for a $10 stake
                ok = False
                break
            placers[h] = a / 10.0  # payout per HK$1 staked
        if not ok or len(placers) < 2:
            continue
        w = v.get('win')
        winner = None
        if w and w.get('combo') is not None:
            try:
                winner = int(str(w['combo']).strip())
            except (TypeError, ValueError):
                winner = None
        out[k] = {'placers': placers, 'winner': winner,
                  'date': v.get('date') or k[:10],
                  'venue': v.get('venue') or k.split('|')[1],
                  'race_no': v.get('race_no') or int(k.split('|')[2]),
                  'url': v.get('url')}
    return out


def load_runs(ev):
    """race_key -> list of runner dicts"""
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=300)
    conn.row_factory = sqlite3.Row
    races = {}
    for k, v in ev.items():
        d0, ven, rn = k.split('|')
        rows = conn.execute('''
            SELECT re.horse_no, re.horse_name, re.win_odds, re.finishing_position
            FROM horse_entries re
            JOIN races r ON re.race_id = r.id
            JOIN race_days rd ON rd.id = r.race_day_id
            WHERE rd.race_date=? AND rd.venue=? AND r.race_no=?
        ''', (d0, ven, int(rn))).fetchall()
        if len(rows) < 4:
            continue
        races[k] = [dict(r) for r in rows]
    conn.close()
    return races


def build_frame(races, ev):
    recs = []
    for key, runners in races.items():
        d0, ven, rn = key.split('|')
        pl = ev[key]['placers']
        for r in runners:
            recs.append({
                'race_key': key,
                'race_date': pd.Timestamp(d0),
                'venue': ven, 'race_no': rn,
                'horse_no': r['horse_no'], 'horse_name': r['horse_name'],
                'win_odds': pd.to_numeric(r['win_odds'], errors='coerce'),
                'finishing_position': pd.to_numeric(r['finishing_position'], errors='coerce'),
                'place_payout': pl.get(r['horse_no'], 0.0),
            })
    df = pd.DataFrame(recs)
    df['target_place'] = ((df['finishing_position'] >= 1) &
                          (df['finishing_position'] <= 3)).astype(int)
    return df


def train_place_model(tr, cols):
    y = tr['target_place'].values.astype(int)
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.06, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=8, eval_metric='logloss',
        random_state=42, n_jobs=-1)
    xgb_m.fit(tr[cols].astype(float).values, y)
    rf_m = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=8,
                                  random_state=42, n_jobs=-1)
    rf_m.fit(tr[cols].astype(float).values, y)
    return xgb_m, rf_m


def walk_forward(df, cols, min_train_races=400):
    """P(place) for every race, refit once per year on strictly earlier data."""
    df = df.sort_values('race_date').reset_index(drop=True)
    out = np.full(len(df), np.nan)
    e_pay = np.full(len(df), np.nan)
    mkt = np.full(len(df), np.nan)
    for y in sorted(df['race_date'].dt.year.unique()):
        tr = df[df['race_date'] < f'{y}-01-01']
        te = df[df['race_date'].dt.year == y]
        if te.empty:
            continue
        ntr = tr['race_key'].nunique()
        if ntr < min_train_races:
            print(f'  {y}: skipped (only {ntr} training races)')
            continue
        xgb_m, rf_m = train_place_model(tr, cols)
        p = (xgb_m.predict_proba(te[cols].astype(float).values)[:, 1] * 0.5
             + rf_m.predict_proba(te[cols].astype(float).values)[:, 1] * 0.5)
        out[(df['race_date'].dt.year == y).values] = p
        # pre-race expectation of the payout if the horse places, and the
        # market's own place probability -- both fitted on TRAINING data only
        pay_curve = fit_payout_curve(tr)
        mkt_curve = fit_market_place_curve(tr)
        te_idx = (df['race_date'].dt.year == y).values
        e_pay[te_idx] = df.loc[te_idx, 'win_odds'].map(pay_curve).fillna(0).values
        inv = 1.0 / df.loc[te_idx, 'win_odds'].values
        q = inv / inv.sum()
        mkt[te_idx] = [mkt_curve(x) for x in q]
        print(f'  {y}: trained on {ntr} races, scored {te["race_key"].nunique()} races')
    df['p_place'] = out
    df['e_pay_place'] = e_pay
    df['p_mkt_place'] = mkt
    return df


def fit_payout_curve(df, bins=(3, 5, 7, 10, 15, 25, 50, 100)):
    """Pre-race estimate of the payout a horse receives IF it places.

    Fitted from historical races: among horses that actually placed, what
    did they pay, as a function of their published win odds? Nothing here
    uses the race being evaluated, so it can be applied out of sample.

    Returns a step function -> callable odds -> expected payout per $1.
    """
    placed = df[(df['place_payout'] > 0) & df['win_odds'].notna()]
    if placed.empty:
        return lambda o: 0.0
    edges = list(bins) + [float('inf')]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = placed[(placed['win_odds'] >= lo) & (placed['win_odds'] < hi)]
        table.append((lo, hi, sel['place_payout'].median() if len(sel) else None))
    # fill gaps with the nearest populated bucket
    known = [t for t in table if t[2] is not None]
    if not known:
        return lambda o: float(placed['place_payout'].median())

    def predict(odds):
        for lo, hi, v in table:
            if lo <= odds < hi and v is not None:
                return float(v)
        return float(known[-1][2])
    return predict


def fit_market_place_curve(df, bins=(3, 5, 7, 10, 15, 25, 50, 100)):
    """Pre-race estimate of the MARKET's place probability for a horse,
    from the market's implied win probability.  Fitted on training data."""
    d = df.copy()
    inv = d.groupby('race_key')['win_odds'].transform(lambda s: 1.0 / s)
    d['q_win'] = inv / d.groupby('race_key')['inv'].transform('sum') if 'inv' in d else inv
    d['q_win'] = d['q_win'] / d.groupby('race_key')['q_win'].transform('sum')
    edges = list(bins) + [float('inf')]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = d[(d['q_win'] >= lo) & (d['q_win'] < hi)]
        table.append((lo, hi, sel['target_place'].mean() if len(sel) else None))

    def predict(q):
        for lo, hi, v in table:
            if lo <= q < hi and v is not None:
                return float(v)
        known = [t for t in table if t[2] is not None]
        return float(known[-1][2]) if known else 0.33
    return predict


def evaluate(df, picker, label, note=''):
    """picker(group) -> horse_no to back. One $1 bet per race."""
    rows = []
    for key, g in df.groupby('race_key'):
        g = g[g['win_odds'].notna() & (g['win_odds'] > 0)]
        if len(g) < 4:
            continue
        hn = picker(g)
        if hn is None:
            continue
        sel = g[g['horse_no'] == hn]
        if sel.empty:
            continue
        pay = float(sel.iloc[0]['place_payout'])
        rows.append({'payout': pay, 'hit': 1 if pay > 0 else 0,
                     'odds': float(sel.iloc[0]['win_odds']),
                     'year': str(key).split('|')[0][:4]})
    if not rows:
        print(f'  {label:30s} (no races)')
        return None
    r = pd.DataFrame(rows)
    roi = r['payout'].mean() - 1
    print(f'  {label:30s} n={len(r):5d}  hit={r["hit"].mean()*100:5.1f}%  '
          f'mean payout={r["payout"].mean():.3f}x  ROI={roi*100:+7.2f}%  {note}')
    # per-year breakdown: is the result an artefact of one era?
    by_year = {}
    for y, gy in r.groupby('year'):
        by_year[y] = {'races': len(gy),
                      'hit_pct': round(gy['hit'].mean() * 100, 2),
                      'roi_pct': round((gy['payout'].mean() - 1) * 100, 2)}
    print('      by year: ' + '  '.join(
        f"{y}: {v['roi_pct']:+6.2f}% (n={v['races']})" for y, v in sorted(by_year.items())))
    return {'races': len(r), 'hit_pct': round(r['hit'].mean() * 100, 2),
            'mean_payout': round(r['payout'].mean(), 4), 'roi_pct': round(roi * 100, 2),
            'by_year': by_year}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', default='/tmp/place_div.ckpt.json')
    ap.add_argument('--out', default=str(ROOT / 'output' / 'place_backtest.json'))
    a = ap.parse_args()

    ev = load_evidence(a.evidence)
    print(f'valid place races in evidence: {len(ev)}')
    races = load_runs(ev)
    ev = {k: v for k, v in ev.items() if k in races}
    print(f'races with full runner rows: {len(ev)}')
    if not ev:
        print('nothing to score')
        return 2

    df = build_frame(races, ev)
    print(f'runner rows: {len(df):,}')

    pkg = pickle.load(open(ROOT / 'models' / 'model_v6.pkl', 'rb'))
    COLS = pkg['feature_cols']

    # standard feature set, built exactly as at serving time
    hist = pd.read_sql('''
        SELECT re.race_id, re.horse_name, re.horse_no, re.draw, re.rating,
               re.jockey, re.trainer, re.win_odds, re.place_odds,
               re.finishing_position, rd.race_date, rd.venue,
               r.distance, r.race_class, r.track, r.race_no
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON rd.id = r.race_day_id
        WHERE re.finishing_position IS NOT NULL
          AND re.finishing_position GLOB '[0-9]*'
          AND CAST(re.finishing_position AS INTEGER) > 0
        ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no
    ''', sqlite3.connect(DB))
    hist['race_date'] = pd.to_datetime(hist['race_date'])
    print('building features ...')
    hfeat, _ = F.build_features(hist)
    hfeat['race_key'] = (hfeat['race_date'].dt.strftime('%Y-%m-%d') + '|' +
                         hfeat['venue'] + '|' + hfeat['race_no'].astype(str))
    hfeat = hfeat.drop_duplicates(subset=['race_key', 'horse_no'])
    df = df.merge(hfeat[['race_key', 'horse_no'] + COLS], on=['race_key', 'horse_no'], how='left')
    df[COLS] = df[COLS].fillna(0)
    print(f'merged features, rows with a missing feature: {df[COLS].isna().any(axis=1).sum()}')
    # a runner with no published odds cannot be backed and cannot be priced
    df = df[df['win_odds'].notna() & (df['win_odds'] > 0)].copy()
    print(f'rows with published odds: {len(df):,}')

    print('\n=== walk-forward P(place) ===')
    df = walk_forward(df, COLS)
    scored = df[df['p_place'].notna()].copy()
    print(f'scored rows: {len(scored):,} in {scored["race_key"].nunique()} races')

    # model quality check: AUC of P(place) against the market
    y = scored['target_place'].values
    auc = roc_auc_score(y, scored['p_place'])
    scored['p_place_mkt'] = scored.groupby('race_key')['win_odds'].transform(
        lambda s: (1.0 / s) / (1.0 / s).sum())
    auc_mkt = roc_auc_score(y, scored['p_place_mkt'])
    print(f'\nP(place) AUC  model={auc:.4f}  market={auc_mkt:.4f}  '
          f'({"model better" if auc > auc_mkt else "market better"})')

    print('\n=== PLACE strategies (selection = pre-race info only) ===')
    res = {}
    res['market_favourite'] = evaluate(
        scored, lambda g: g.sort_values('win_odds').iloc[0]['horse_no'],
        'market favourite', '(lowest published odds)')
    res['market_2nd_favourite'] = evaluate(
        scored, lambda g: g.sort_values('win_odds').iloc[1]['horse_no'], 'market 2nd favourite')
    res['market_3rd_favourite'] = evaluate(
        scored, lambda g: g.sort_values('win_odds').iloc[2]['horse_no'], 'market 3rd favourite')
    res['market_longshot'] = evaluate(
        scored, lambda g: g.sort_values('win_odds').iloc[-1]['horse_no'], 'market longest shot')
    res['model_top'] = evaluate(
        scored, lambda g: g.sort_values('p_place', ascending=False).iloc[0]['horse_no'],
        'model top P(place)')

    def model_outside(g):
        cand = g[g['win_odds'] >= 6.0]
        if cand.empty:
            return None
        return cand.sort_values('p_place', ascending=False).iloc[0]['horse_no']
    res['model_outside_6plus'] = evaluate(
        scored, model_outside, 'model best P(place) @>=6.0')

    def model_edge(g):
        gg = g.copy()
        gg['pm'] = gg['p_place'] / gg['p_place'].sum()
        gg['mk'] = (1.0 / gg['win_odds']) / (1.0 / gg['win_odds']).sum()
        gg['edge'] = gg['pm'] - gg['mk']
        return gg.sort_values('edge', ascending=False).iloc[0]['horse_no']
    res['model_edge'] = evaluate(scored, model_edge, 'model P(place) minus market')

    # ---- value filter: EV = p_model(place) x E[payout | placed] - 1 --------
    # every input is pre-race: the model probability, the published odds, and
    # a payout expectation fitted on training data only.
    def make_value(threshold, use_market_prob=False):
        def pick(g):
            g = g.copy()
            p = g['p_mkt_place'] if use_market_prob else g['p_place']
            g['ev'] = p * g['e_pay_place'] - 1.0
            best = g.sort_values('ev', ascending=False).iloc[0]
            if best['ev'] < threshold:
                return None
            return best['horse_no']
        return pick

    print('\n--- value filters (EV = P(place) x E[payout|placed] - 1) ---')
    res['value_ev0_model'] = evaluate(scored, make_value(0.0), 'model EV > 0')
    res['value_ev05_model'] = evaluate(scored, make_value(0.05), 'model EV > 0.05')
    res['value_ev0_market'] = evaluate(scored, make_value(0.0, True), 'market-implied EV > 0')
    res['value_ev05_market'] = evaluate(scored, make_value(0.05, True), 'market-implied EV > 0.05')

    out = {'evidence': a.evidence, 'races_scored': int(scored['race_key'].nunique()),
           'p_place_auc_model': round(float(auc), 4),
           'p_place_auc_market': round(float(auc_mkt), 4),
           'strategies': res}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=2)
    print(f'\nsaved: {a.out}')


if __name__ == '__main__':
    main()
