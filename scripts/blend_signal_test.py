"""Is there ANY usable signal in the model? (blend test)

The stress test showed the model is badly miscalibrated while the market is
nearly unbiased. But calibration and ranking are different things: a model
can be miscalibrated and still carry ranking information. If a BLEND of the
model and the market probabilities beats the market alone on AUC, then there
is signal worth recalibrating; if not, the project is done and the honest
product is a no-bet filter.

Also fixes the break-even null used in the earlier stress test: the correct
break-even hit rate is 1/E[odds | winner], NOT 1/mean(odds over all bets).
Using the latter understates break-even and makes a losing rule look like a
winner.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/blend_signal_test.py
"""
import pickle, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import features as F

ROOT = Path(__file__).parent.parent
DB = ROOT / 'data' / 'horse.db'
MODEL = ROOT / 'models' / 'model_v6.pkl'
CUT_VAL = pd.Timestamp('2025-10-01')

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
te = df[(df['race_date'] > CUT_VAL) & df['win_odds'].notna() & (df['win_odds'] > 0)].copy()
X = te[COLS].astype(float).fillna(0).values
te['p_model_raw'] = (pkg['xgb_model'].predict_proba(X)[:, 1] * 0.5
                     + pkg['rf_model'].predict_proba(X)[:, 1] * 0.5)
RK = ['race_date', 'venue', 'race_no']
te['p_market_raw'] = 1.0 / te['win_odds']
te['p_market'] = te.groupby(RK)['p_market_raw'].transform(lambda s: s / s.sum())
te['p_model'] = te.groupby(RK)['p_model_raw'].transform(lambda s: s / s.sum())
te['won'] = (te['finishing_position'] == 1).astype(int)

print(f'test races: {te.groupby(RK).ngroups}  runners: {len(te):,}')
y = te['won'].to_numpy()
pm = te['p_market'].to_numpy()
pmod = te['p_model'].to_numpy()

print('\n=== AUC: market vs model vs blends (within-race normalised) ===')
print(f'  market alone      : {roc_auc_score(y, pm):.5f}')
print(f'  model alone       : {roc_auc_score(y, pmod):.5f}')
best = None
for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    p = w * pm + (1 - w) * pmod
    a = roc_auc_score(y, p)
    tag = '  <-- market' if w == 0 else ('  <-- model' if w == 1.0 else '')
    print(f'  {w:>4.0%} market + {1-w:>4.0%} model : AUC {a:.5f}{tag}')
    if best is None or a > best[1]:
        best = (w, a)
print(f'  best blend: {best[0]:.0%} market / {1-best[0]:.0%} model  AUC {best[1]:.5f}')

print('\n=== CORRECTED break-even null: 1/E[odds | winner] ===')
for label, sel in [('every runner edge>=0%', te[te['p_model'] > te['p_market']]),
                   ('favourite every race', te.sort_values('p_market', ascending=False).groupby(RK).head(1))]:
    w = sel[sel['won'] == 1]
    if len(w) == 0:
        continue
    mean_win_odds = w['win_odds'].mean()
    be = 1.0 / mean_win_odds
    actual = sel['won'].mean()
    ret = float((sel['won'] * sel['win_odds']).sum()) / len(sel)
    verdict = 'BEATS' if actual > be else 'LOSES TO'
    print(f'  {label}')
    print(f'    n={len(sel)}  E[odds|winner]={mean_win_odds:.2f}  true break-even hit={be:.2%}')
    print(f'    actual hit={actual:.2%}  -> {verdict} break-even')
    print(f'    actual ROI per unit staked = {ret - 1:+.2%} (this IS the ROI)')

print('\n=== Does an ISOTONIC recalibration of the model fix the edge? ===')
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
iso.fit(te['p_model'], te['won'])
te['p_iso'] = iso.predict(te['p_model'])
print(f'  recalibrated model AUC: {roc_auc_score(y, te["p_iso"]):.5f}  '
      f'(isotonic is monotone, so AUC is unchanged by construction)')
for lo, hi in [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]:
    m = (te['p_iso'] >= lo) & (te['p_iso'] < hi)
    g = te[m]
    if len(g) == 0:
        continue
    w = g[g['won'] == 1]
    if len(w) == 0:
        continue
    be = 1.0 / w['win_odds'].mean()
    ret = float((g['won'] * g['win_odds']).sum()) / len(g)
    print(f'  p_iso in [{lo:.0%},{hi:.0%}): n={len(g):<6} hit={g["won"].mean():>6.2%} '
          f'break-even={be:>6.2%}  ROI={ret-1:>+7.2%}')
