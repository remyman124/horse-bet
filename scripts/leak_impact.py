#!/usr/bin/env python3
"""
How much did the outcome leak actually inflate performance?

The question
------------
`horse_entries.win_odds` used to hold the POST-RACE WIN dividend for 394
rows (393 of them the race winner). The model read those as a pre-race
feature, so it could spot the winner from the odds alone. How much
performance was that leak worth? Answering it needs a controlled
experiment: the same data, the same split, the same model, with ONLY the
odds changed.

Why this is a clean experiment
------------------------------
`repair_provenance` records, for every odds repair, the OLD (contaminated)
value and the NEW (official) value, with the official evidence URL. So the
contaminated arm can be reconstructed EXACTLY, in memory, from the
repaired database — no separate backup needed, no risk to the DB.

Everything else — the runners, the finishing positions, the results — is
identical between the two arms, so the measured difference is attributable
to the odds leak and nothing else. This is strictly better than the
original run (which used two different database snapshots and therefore
also differed in runner coverage).

The trap this script exists to avoid
------------------------------------
The first version of this experiment wrote a column called `clean_odds`
and expected `features.build_features()` to read it. It does not — the
builder reads `win_odds`. So the "decontaminated" arm silently trained
on contaminated data and produced numbers IDENTICAL to the contaminated
arm. Identical numbers are not a result, they are a bug.

GENERAL RULE: a leak experiment that reports "no effect" is evidence of a
broken harness, not evidence of clean data. Always assert that the
contaminated arm reproduces the known contaminated number before reading
any difference.

Usage
-----
    python3 scripts/leak_impact.py
"""

import json
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
CUT_TRAIN = pd.Timestamp('2025-06-01')
CUT_VAL = pd.Timestamp('2025-10-01')


def load_raw():
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
    return df


def contaminated_odds(conn, df):
    """Restore the pre-repair (contaminated) win_odds from repair_provenance."""
    df = df.copy()
    df['rowid'] = np.arange(len(df))
    prov = conn.execute("""
        SELECT row_key, old_value, new_value FROM repair_provenance
        WHERE table_name='horse_entries' AND column_name='win_odds'
          AND old_value IS NOT NULL
    """).fetchall()
    lookup = {}
    for row_key, old_v, new_v in prov:
        try:
            lookup[int(row_key)] = float(old_v)
        except (TypeError, ValueError):
            continue
    conn.row_factory = sqlite3.Row
    ids = conn.execute("""
        SELECT he.id, rd.race_date, rd.venue, r.race_no, he.horse_no
        FROM horse_entries he
        JOIN races r ON r.id = he.race_id
        JOIN race_days rd ON rd.id = r.race_day_id
        WHERE he.finishing_position IS NOT NULL
          AND he.finishing_position GLOB '[0-9]*'
          AND CAST(he.finishing_position AS INTEGER) > 0
        ORDER BY rd.race_date, rd.venue, r.race_no, he.horse_no
    """).fetchall()
    contaminated = df['win_odds'].copy()
    n = 0
    for i, r in enumerate(ids):
        eid = r['id']
        if eid in lookup:
            contaminated.iloc[i] = lookup[eid]
            n += 1
    print(f'  reconstructed {n} contaminated odds values from repair_provenance')
    return contaminated, n


def build(df, odds_series=None):
    d = df.copy()
    if odds_series is not None:
        d['win_odds'] = odds_series.values
    d, feat_list = F.build_features(d)
    d['target_win'] = (d['finishing_position'] == 1).astype(int)
    return d, feat_list


def run_arm(d, cols, tag):
    tr = d[(d['race_date'] <= CUT_TRAIN) & d['has_odds']].copy()
    va = d[(d['race_date'] > CUT_TRAIN) & (d['race_date'] <= CUT_VAL) & d['has_odds']].copy()
    te = d[(d['race_date'] > CUT_VAL) & d['has_odds']].copy()
    y_tr, y_va, y_te = (x['target_win'].values.astype(int) for x in (tr, va, te))
    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric='auc', early_stopping_rounds=25, random_state=42, n_jobs=-1)
    xgb_m.fit(tr[cols].astype(float).values, y_tr,
              eval_set=[(va[cols].astype(float).values, y_va)], verbose=False)
    rf_m = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=8,
                                  random_state=42, n_jobs=-1)
    rf_m.fit(tr[cols].astype(float).values, y_tr)
    p = (xgb_m.predict_proba(te[cols].astype(float).values)[:, 1] * 0.5
         + rf_m.predict_proba(te[cols].astype(float).values)[:, 1] * 0.5)
    te = te.copy()
    te['p'] = p
    auc = roc_auc_score(y_te, p)

    t1 = t3 = n = 0
    for _, g in te.groupby(['race_date', 'venue', 'race_no']):
        if len(g) < 2:
            continue
        w = g[g['finishing_position'] == 1]
        if w.empty:
            continue
        n += 1
        name = w.iloc[0]['horse_name']
        order = g.sort_values('p', ascending=False)
        if order.iloc[0]['horse_name'] == name:
            t1 += 1
        if name in order.head(3)['horse_name'].tolist():
            t3 += 1
    print(f'  [{tag}] AUC={auc:.4f}  top1={t1}/{n} ({t1/n*100:.2f}%)  top3={t3}/{n} ({t3/n*100:.2f}%)')
    return {'auc': round(float(auc), 4), 'races': n,
            'top1': t1, 'top1_pct': round(t1 / n * 100, 2),
            'top3': t3, 'top3_pct': round(t3 / n * 100, 2)}


def fav_baseline(d, tag):
    te = d[(d['race_date'] > CUT_VAL) & d['has_odds']].copy()
    t1 = t3 = n = 0
    for _, g in te.groupby(['race_date', 'venue', 'race_no']):
        if len(g) < 2:
            continue
        w = g[g['finishing_position'] == 1]
        if w.empty:
            continue
        n += 1
        name = w.iloc[0]['horse_name']
        order = g.sort_values('win_odds')
        if order.iloc[0]['horse_name'] == name:
            t1 += 1
        if name in order.head(3)['horse_name'].tolist():
            t3 += 1
    print(f'  [{tag}] favourite baseline top1={t1}/{n} ({t1/n*100:.2f}%)  top3={t3}/{n} ({t3/n*100:.2f}%)')
    return {'races': n, 'top1': t1, 'top1_pct': round(t1 / n * 100, 2),
            'top3': t3, 'top3_pct': round(t3 / n * 100, 2)}


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    df = load_raw()
    print(f'entries: {len(df):,}')

    contaminated, nrecon = contaminated_odds(conn, df)
    conn.close()

    # sanity: the contaminated arm must show the fingerprint, the clean arm must not
    def fingerprint(series):
        s = pd.to_numeric(series, errors='coerce')
        frac = s - s.round()
        return int(((s >= 10) & (frac.abs() > 1e-9)).sum())

    print(f'\n  fingerprint rows — contaminated arm: {fingerprint(contaminated)} | '
          f'clean arm: {fingerprint(df["win_odds"])}')
    if fingerprint(contaminated) == 0 or fingerprint(df['win_odds']) != 0:
        print('  ABORT: the arms are not distinguishable — the experiment would be meaningless.')
        return 2

    pkg = pickle.load(open(ROOT / 'models' / 'model_v6.pkl', 'rb'))
    COLS = pkg['feature_cols']

    print('\nbuilding features (clean arm) ...')
    d_clean, _ = build(df, None)
    print('building features (contaminated arm) ...')
    d_dirty, _ = build(df, contaminated)

    print('\n=== CONTROLLED EXPERIMENT: identical data, identical split, identical model ===')
    res_clean = run_arm(d_clean, COLS, 'CLEAN   ')
    res_dirty = run_arm(d_dirty, COLS, 'CONTAM  ')
    fav_clean = fav_baseline(d_clean, 'CLEAN  ')
    fav_dirty = fav_baseline(d_dirty, 'CONTAM ')

    out = {
        'test_period': f"{d_clean['race_date'].min().date()} .. {d_clean['race_date'].max().date()}",
        'contaminated_rows_reconstructed': nrecon,
        'clean': res_clean, 'contaminated': res_dirty,
        'favourite_clean': fav_clean, 'favourite_contaminated': fav_dirty,
        'leak_cost': {
            'auc_points': round(res_dirty['auc'] - res_clean['auc'], 4),
            'top1_pp': round(res_dirty['top1_pct'] - res_clean['top1_pct'], 2),
            'top3_pp': round(res_dirty['top3_pct'] - res_clean['top3_pct'], 2),
        },
    }
    print('\n=== LEAK COST (contaminated minus clean) ===')
    print(f"  AUC   {out['leak_cost']['auc_points']:+.4f}")
    print(f"  top-1 {out['leak_cost']['top1_pp']:+.2f} pp")
    print(f"  top-3 {out['leak_cost']['top3_pp']:+.2f} pp")
    p = ROOT / 'output' / 'leak_impact.json'
    json.dump(out, open(p, 'w'), indent=2)
    print(f'\nsaved: {p}')


if __name__ == '__main__':
    main()
