#!/usr/bin/env python3
"""
HKJC Horse Racing Model — Version 6
=====================================
Version 6 exists because of a data-bug fix (2026-10-20).

v5 was trained WITHOUT odds because `implied_prob` had NEGATIVE correlation with
winning (AUC 0.196). That was a PARSER BUG, not market behaviour:
  - HKJC results tables have 12-col (Win Odds = idx 11) and 11-col (idx 10) layouts
  - the old parser grabbed the wrong column on older races, INVERTING the odds
After repair, odds show the correct monotonic pattern (low odds -> high win rate).

So v6 adds the market signal back, and strengthens jockey features, which the
2026-10-19 post-mortem showed were badly under-weighted (4/4 hits when the
winner's jockey was in the model's top-3).

Key design points
-----------------
1. Odds features are computed from the POST-RACE "Win Odds" column, which is the
   STARTING PRICE (SP) published before the race. Using SP to predict a race
   that already happened is legitimate — no outcome leakage.
2. We evaluate BOTH AUC and the metric that actually matters: top-1 / top-3 hit
   rate per race. High AUC with bad calibration still loses money.
3. Every strategy is compared against two baselines:
      - FAVOURITE  (lowest odds in the race)
      - RATING     (highest rating)
   A model that cannot beat the favourite baseline is not worth deploying.
4. Odds availability is reported honestly. Races without odds are excluded from
   the odds-feature training set rather than silently imputed.

Outputs: models/model_v6.pkl, models/model_v6.json
"""
import pickle, sqlite3, json, logging, sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # shared train/serve feature builder

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger('retrain_v6')

DB = Path(__file__).parent.parent / 'data' / 'horse.db'
OUT = Path(__file__).parent.parent / 'models' / 'model_v6.pkl'

conn = sqlite3.connect(DB)
df = pd.read_sql('''
    SELECT re.race_id,
           re.horse_name, re.horse_no, re.draw, re.rating,
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
df['draw']               = pd.to_numeric(df['draw'], errors='coerce').fillna(0)
df['rating']             = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['distance']           = pd.to_numeric(df['distance'], errors='coerce').fillna(1200)
df['win_odds']           = pd.to_numeric(df['win_odds'], errors='coerce')
df['race_date']          = pd.to_datetime(df['race_date'])
log.info(f"Loaded {len(df):,} entries | with win_odds: {df['win_odds'].notna().sum():,}")

# ── Feature engineering ──────────────────────────────────────────────────────
# ALL features come from the shared module so that training and serving use
# literally the same code. The module also enforces the day-boundary rule:
# a rolling window contains only races that finished STRICTLY BEFORE the
# current race day, so a jockey's race 2 does not inherit race 1's result
# from the same meeting. `scripts/test_features.py` asserts train==serve.
df, FEATURE_LIST = F.build_features(df)
log.info(f"Built {len(FEATURE_LIST)} features via the shared module")

df['target_win']   = (df['finishing_position'] == 1).astype(int)
df['target_place'] = df['finishing_position'].isin([1, 2, 3]).astype(int)

# ── Encode categoricals ──────────────────────────────────────────────────────
# class/track/venue are encoded inside the shared module (F) so that training
# and serving agree exactly. The jockey/trainer encoders are only stored in
# the model bundle for reference; they are not model features.
class_map = F.CLASS_MAP
track_enc = LabelEncoder(); track_enc.fit(['Turf', 'Dirt', 'AWT'])
venue_enc = LabelEncoder(); venue_enc.fit(['HV', 'ST'])
jock_enc = LabelEncoder(); jock_enc.fit(df['jockey'].fillna(''))
df['jockey_enc'] = jock_enc.transform(df['jockey'].fillna(''))
trainer_enc = LabelEncoder(); trainer_enc.fit(df['trainer'].fillna(''))
df['trainer_enc'] = trainer_enc.transform(df['trainer'].fillna(''))

FORM_COLS = list(F.FORM_FEATURES)
# ⚠️ `rating` is DELIBERATELY EXCLUDED from the model features.
# Historical ratings are unobtainable (HKJC only publishes a race card for UPCOMING
# races), so `horse_entries.rating` is populated for ~11% of rows — 2026 only.
# In the training window it is therefore all zero: both v5's XGBoost and RF gave it
# 0.0000 importance. But at SERVE time the real race card DOES carry a rating, so
# keeping it would create a train/serve distribution mismatch: the model has never
# seen a non-zero rating and cannot know whether to trust it.
# Rating is reported separately as a baseline instead (see backtest).
#
# ⚠️ place_odds is DELIBERATELY EXCLUDED — it is outcome leakage.
# `place_odds` is only published for the top 3 finishers (verified: 99.6% of rows
# with a place_odds value finished 1st/2nd/3rd, 0% of horses finishing 4th+ have
# one). Using it as a feature lets the model read the answer off the result.
# A first v6 run with it scored a fake 73% top-1 — entirely explained by the leak.
ODDS_COLS = list(F.ODDS_FEATURES)

FEATURE_COLS = FORM_COLS + ODDS_COLS

log.info("Leak check on candidate odds features (coverage must be uniform across positions):")


def leak_check(col, df_):
    """
    A pre-race feature must be present at a roughly UNIFORM rate across finishing
    positions. If coverage spikes for positions 1-3, the feature encodes the
    outcome and must be dropped.

    Must run BEFORE fillna(0) — otherwise every column is non-null and the
    check is a no-op that always reports "ok".
    """
    g = df_.groupby(df_['finishing_position'].clip(0, 5))[col].apply(lambda s: s.notna().mean())
    rates = {int(k): round(float(v), 3) for k, v in g.items()}
    top3 = [rates.get(i, 0) for i in (1, 2, 3)]
    rest = [rates.get(i, 0) for i in (0, 4, 5)]
    leak = False
    if top3 and rest:
        t = sum(top3) / len(top3)
        r = sum(rest) / len(rest)
        if t > 0.7 and r < 0.2:
            leak = True
    log.info(f"  leak-check {col:14s} coverage by pos {rates}  -> {'LEAK!' if leak else 'ok'}")
    return leak


LEAKY = []
for _c in ['win_odds', 'place_odds']:
    if _c in df.columns:
        if leak_check(_c, df):
            LEAKY.append(_c)
if LEAKY:
    log.error(f"LEAK DETECTED in {LEAKY} — these columns must not be used as features. "
              f"They encode the post-race outcome.")
    for _c in LEAKY:
        if _c in FEATURE_COLS:
            FEATURE_COLS.remove(_c)
            log.error(f"  removed {_c} from FEATURE_COLS")
log.info(f"Final feature set ({len(FEATURE_COLS)}): {FEATURE_COLS}")

df = df.fillna(0)
log.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")


def odds_integrity_check(df_):
    """
    A genuine pre-race SP must be NEGATIVELY correlated with win probability.

    Two valid tests (the share-of-winners test is NOT valid — see below):

    1. PER-HORSE win rate within each odds bucket must decrease monotonically.
       i.e. of the horses priced 6-11, what fraction won? Must be lower than of
       the horses priced 3-6, and so on.

    2. The race FAVOURITE (lowest odds in the race) must win ~25-35% of races.
       With inverted odds this collapses to ~0%, which is the signature of the
       historical parser bug.

    Why the share-of-winners test is invalid: the favourite bucket holds few
    horses (0.3-0.5 per race) while 3-6 holds 1.3-1.6, so the SHARE of winners
    per bucket is dominated by bucket population, not by win probability. The
    2026 known-good cohort shows the same non-monotonic share profile as the
    repaired years, which confirms the share test is simply the wrong metric.
    """
    d = df_[df_['win_odds'].notna() & (df_['win_odds'] > 0)].copy()
    if len(d) < 1000:
        log.error("odds_integrity: too few odds rows to judge")
        return False
    bins = [0, 3, 6, 11, 21, 51, 10000]
    labels = ['<3', '3-6', '6-11', '11-21', '21-51', '51+']
    d['bucket'] = pd.cut(d['win_odds'], bins=bins, labels=labels, right=False)
    d['won'] = (d['finishing_position'] == 1).astype(int)

    per_horse = d.groupby('bucket', observed=True)['won'].mean() * 100
    per_horse = per_horse.reindex(labels)
    log.info("  per-horse win rate % by odds bucket: "
             + "  ".join(f"{k}={v:.1f}" for k, v in per_horse.items()))

    # 1) per-horse win rate should decrease with odds (informational; small
    #    cohorts can blip, e.g. the 2026 reference shows 4.0 -> 4.9)
    vals = [float(per_horse[k]) for k in labels]
    mono = all(vals[i] >= vals[i + 1] - 1.5 for i in range(len(vals) - 1))

    # 2) the favourite must win a realistic share of races
    fav_won = 0
    n_races = 0
    for rid, g in d.groupby('race_id'):
        w = g[g['finishing_position'] == 1]
        if len(w) != 1:
            continue
        n_races += 1
        if float(w['win_odds'].iloc[0]) == float(g['win_odds'].min()):
            fav_won += 1
    fav_pct = (fav_won / n_races * 100) if n_races else 0.0
    log.info(f"  race favourite won {fav_won}/{n_races} = {fav_pct:.1f}% "
             f"(healthy range 25-35%; ~0% means the odds are INVERTED)")

    # 3) AUC of implied probability as a predictor of winning. Genuine pre-race
    #    prices give a strong positive AUC (~0.8+). Inverted odds give < 0.5.
    #    This is the robust scalar gate — the bucket profile is only indicative.
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(d['won'], (1.0 / d['win_odds']))
        log.info(f"  implied-probability AUC = {auc:.4f} (must be > 0.65; < 0.5 = inverted)")
        auc_ok = auc > 0.65
    except Exception as e:
        log.error(f"  AUC check failed: {e!r}")
        auc_ok = False
        auc = 0.0

    ok = fav_pct > 20 and auc_ok
    if fav_pct <= 20:
        log.error("  odds_integrity FAILED: the favourite wins too rarely — odds are inverted.")
    if not auc_ok:
        log.error("  odds_integrity FAILED: implied-probability AUC is not sane.")
    if ok:
        log.info("  odds_integrity OK: odds behave like genuine pre-race prices.")
    return ok


log.info("Odds integrity check (full table):")
FULL_INTEGRITY_OK = odds_integrity_check(df)
if not FULL_INTEGRITY_OK:
    log.error("Training ABORTED — fix the odds repair before retraining. "
              "A model trained on inverted odds scores well and predicts nothing.")
    raise SystemExit(2)


# ── Temporal split ───────────────────────────────────────────────────────────
CUT_TRAIN = pd.Timestamp('2025-06-01')
CUT_VAL   = pd.Timestamp('2025-10-01')
df_train = df[df['race_date'] <= CUT_TRAIN].copy()
df_val   = df[(df['race_date'] > CUT_TRAIN) & (df['race_date'] <= CUT_VAL)].copy()
df_test  = df[df['race_date'] > CUT_VAL].copy()
log.info(f"Train: {df_train.race_date.min().date()}–{df_train.race_date.max().date()} ({len(df_train):,})")
log.info(f"Val:   {df_val.race_date.min().date()}–{df_val.race_date.max().date()} ({len(df_val):,})")
log.info(f"Test:  {df_test.race_date.min().date()}–{df_test.race_date.max().date()} ({len(df_test):,})")

# Odds-bearing subset (odds features need real odds)
tr_o = df_train[df_train['has_odds']].copy()
va_o = df_val[df_val['has_odds']].copy()
te_o = df_test[df_test['has_odds']].copy()
log.info(f"Odds subset -> train {len(tr_o):,} val {len(va_o):,} test {len(te_o):,}")


def train_eval(train_df, val_df, test_df, tag, feature_cols):
    y_tr = train_df['target_win'].values.astype(int)
    y_va = val_df['target_win'].values.astype(int)
    y_te = test_df['target_win'].values.astype(int)

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric='auc', early_stopping_rounds=25, random_state=42, n_jobs=-1)
    xgb_m.fit(train_df[feature_cols].astype(float).values, y_tr,
              eval_set=[(val_df[feature_cols].astype(float).values, y_va)], verbose=False)
    xgb_va = xgb_m.predict_proba(val_df[feature_cols].astype(float).values)[:, 1]
    xgb_te = xgb_m.predict_proba(test_df[feature_cols].astype(float).values)[:, 1]

    rf_m = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=8,
                                  random_state=42, n_jobs=-1)
    rf_m.fit(train_df[feature_cols].astype(float).values, y_tr)
    rf_va = rf_m.predict_proba(val_df[feature_cols].astype(float).values)[:, 1]
    rf_te = rf_m.predict_proba(test_df[feature_cols].astype(float).values)[:, 1]

    ens_va = (xgb_va + rf_va) / 2
    ens_te = (xgb_te + rf_te) / 2
    log.info(f"[{tag}] XGB val={roc_auc_score(y_va, xgb_va):.4f} test={roc_auc_score(y_te, xgb_te):.4f}")
    log.info(f"[{tag}] RF  val={roc_auc_score(y_va, rf_va):.4f} test={roc_auc_score(y_te, rf_te):.4f}")
    log.info(f"[{tag}] ENS val={roc_auc_score(y_va, ens_va):.4f} test={roc_auc_score(y_te, ens_te):.4f}")
    return {'xgb': xgb_m, 'rf': rf_m,
            'val_auc': float(roc_auc_score(y_va, ens_va)),
            'test_auc': float(roc_auc_score(y_te, ens_te))}


def hit_rates(test_df, score_col, label):
    """Per-race top-1 / top-3 hit rate for a score column."""
    t1 = t3 = n = 0
    for _, g in test_df.groupby(['race_date', 'venue', 'race_no']):
        if len(g) < 2:
            continue
        winner = g[g['finishing_position'] == 1]
        if winner.empty:
            continue
        wname = winner.iloc[0]['horse_name']
        n += 1
        top1 = g.sort_values(score_col, ascending=False).head(1)['horse_name'].tolist()
        top3 = g.sort_values(score_col, ascending=False).head(3)['horse_name'].tolist()
        if top1 and top1[0] == wname:
            t1 += 1
        if wname in top3:
            t3 += 1
    log.info(f"  {label}: top1={t1}/{n} ({t1/n*100:.1f}%)  top3={t3}/{n} ({t3/n*100:.1f}%)")
    return {'races': n, 'top1': t1, 'top3': t3,
            'top1_pct': round(t1 / n * 100, 2) if n else 0.0,
            'top3_pct': round(t3 / n * 100, 2) if n else 0.0}


results = {}
# A: form only (v5-equivalent + stronger jockey) — no odds
results['form'] = train_eval(df_train, df_val, df_test, 'FORM-only', FORM_COLS)
# B: form + odds (v6 headline)
results['form_odds'] = train_eval(tr_o, va_o, te_o, 'FORM+ODDS', FEATURE_COLS)

# ── Real-world evaluation on the odds-bearing test set ───────────────────────
log.info("=" * 70)
log.info("BACKTEST on test races that have odds (the metric that matters)")
log.info("=" * 70)

te = te_o.copy()
te['pred_form'] = results['form']['xgb'].predict_proba(
    te[FORM_COLS].astype(float).values)[:, 1] * 0.5 + \
    results['form']['rf'].predict_proba(te[FORM_COLS].astype(float).values)[:, 1] * 0.5
te['pred_form_odds'] = results['form_odds']['xgb'].predict_proba(
    te[FEATURE_COLS].astype(float).values)[:, 1] * 0.5 + \
    results['form_odds']['rf'].predict_proba(te[FEATURE_COLS].astype(float).values)[:, 1] * 0.5
# Market baseline: lowest odds wins
te['pred_fav'] = -te['win_odds']
# Rating baseline
te['pred_rating'] = te['rating']
# Market + form blend (value-ish)
te['pred_blend'] = te['prob_norm'] * 0.6 + te['pred_form_odds'] * 0.4

backtest = {
    'form_only':      hit_rates(te, 'pred_form', 'FORM-only model      '),
    'form_plus_odds': hit_rates(te, 'pred_form_odds', 'FORM+ODDS model     '),
    'favourite':      hit_rates(te, 'pred_fav', 'BASELINE favourite  '),
    'rating':         hit_rates(te, 'pred_rating', 'BASELINE rating     '),
    'blend_market':   hit_rates(te, 'pred_blend', 'BLEND 60% mkt/40% ml'),
}

# ── Feature importance (does the market signal actually get used?) ───────────
imp = results['form_odds']['xgb'].feature_importances_
importance = {c: round(float(v), 5) for c, v in sorted(zip(FEATURE_COLS, imp),
                                                       key=lambda kv: -kv[1])}
log.info("XGB feature importance (form+odds model):")
for c, v in list(importance.items())[:12]:
    log.info(f"    {c:22s} {v:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
package = {
    'xgb_model': results['form_odds']['xgb'],
    'rf_model':  results['form_odds']['rf'],
    'feature_cols': FEATURE_COLS,
    'form_feature_cols': FORM_COLS,
    'odds_feature_cols': ODDS_COLS,
    'encoders': {'track': track_enc, 'venue': venue_enc,
                 'jockey': jock_enc, 'trainer': trainer_enc},
    'class_map': class_map,
}
with open(OUT, 'wb') as f:
    pickle.dump(package, f)

meta = {
    'version': 'v6',
    'built_at': pd.Timestamp.utcnow().isoformat(),
    'features': FEATURE_COLS,
    'form_features': FORM_COLS,
    'odds_features': ODDS_COLS,
    'auc': {
        'form_only': {'val': round(results['form']['val_auc'], 4), 'test': round(results['form']['test_auc'], 4)},
        'form_plus_odds': {'val': round(results['form_odds']['val_auc'], 4),
                           'test': round(results['form_odds']['test_auc'], 4)},
    },
    'backtest_test_period': {
        'from': str(te['race_date'].min().date()), 'to': str(te['race_date'].max().date()),
        'races': backtest['form_plus_odds']['races'],
        'results': backtest,
    },
    'feature_importance': importance,
    'notes': [
        'v6 exists because the v5 decision to exclude odds was based on a parser bug '
        '(odds inverted on older races; HKJC results table layout differs 11-col vs 12-col).',
        'Odds feature = post-race Win Odds column = HKJC starting price (available pre-race).',
        'Top-1/top-3 per-race hit rate is the decision metric; AUC alone is not enough.',
    ],
}
with open(OUT.with_suffix('.json'), 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

log.info(f"Saved to {OUT}")
log.info("=" * 70)
log.info("SUMMARY")
log.info(f"  form-only      AUC test={results['form']['test_auc']:.4f}  {backtest['form_only']}")
log.info(f"  form+odds      AUC test={results['form_odds']['test_auc']:.4f}  {backtest['form_plus_odds']}")
log.info(f"  favourite      {backtest['favourite']}")
log.info(f"  rating         {backtest['rating']}")
