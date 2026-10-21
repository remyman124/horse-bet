"""
Shared feature engineering for the HKJC model.

This module is the SINGLE source of truth for feature construction. Both
`retrain_v6.py` (training) and `predict_racecard.py` (serving) import from here.

Why this exists
---------------
A model trained on features built one way and served on features built another
way is silently broken: the same horse looks different at training time and at
race time, and no test catches it because both halves "work" in isolation.
On 2026-10-20 the v6 features lived inside the training script, so there was no
tested path to score a future race card. Extracting the code here makes the
train and serve paths literally the same functions.

THE DAY-BOUNDARY RULE (the important one)
------------------------------------------
Form, jockey and trainer features are rolling windows over PAST races. The
window must contain only races that finished STRICTLY BEFORE the current race
day. A jockey rides several times in one meeting; a naive `shift(1)` within the
jockey group makes race 2 inherit race 1's result from the SAME day, because
the jockey's rows for that meeting are consecutive.

That is wrong twice over:
  * it leaks the outcome — at 09:00 nobody knows how race 1 went;
  * it makes the training features differ from the serving features, because
    at serve time those same-day results simply do not exist yet.
Measured on 2026-10-19 ST: 28/142 rows differed on `jockey_win_10` and 76/142
on `jockey_place_10` between the two paths.

So every rolling window here is anchored on the previous race DAY, not the
previous row. `_prior_day_rolling()` implements that.

LEAKAGE RULES enforced here
---------------------------
* `place_odds` is NEVER a feature. It only exists for the top 3 finishers
  (verified: 99.6% of rows with a value finished 1st/2nd/3rd; 0% of horses
  finishing 4th+ have one). It is the result, not an input.
* `rating` is NOT a feature. Historical ratings cannot be obtained (HKJC only
  publishes a race card for UPCOMING races), so it is populated for ~11% of
  rows — 2026 only. In the training window it is all zero, but at serve time a
  real race card DOES carry a rating. Keeping it would create a train/serve
  distribution mismatch: the model has never seen a non-zero rating.
"""

import re

import numpy as np
import pandas as pd

# A jockey who takes a claiming ride is shown on the race card as
# "C L Chau (-2)" — the number is the weight claimed, not part of the name.
# The DB stores BOTH forms: ~2,150 rows as plain "C L Chau" and a handful as
# "C L Chau (-2)". If the jockey column is used as-is, the claim spelling is
# a DIFFERENT entity with almost no history, so its rolling-window features
# come out NaN and get filled with 0 — the model then scores that rider on
# "no form" when in fact they have hundreds of runs. Measured on the
# 2026-10-21 HV card: 26 of 102 runners had NaN jockey features, and the
# fix changed the model's top-3 in 5 of 9 races.
#
# Normalising HERE — in the shared module, not in the serving script — is
# what keeps train and serve identical. The same rule is applied to the
# historical rows at training time, so the entity is one jockey in both.
_CLAIM_SUFFIX = re.compile(r'\s*\(-?\d+\)\s*$')


def normalize_jockey(name):
    """Strip the claiming-weight suffix from a jockey name."""
    if name is None:
        return name
    s = str(name).strip()
    if '(' not in s:
        return s
    return _CLAIM_SUFFIX.sub('', s).strip()


FORM_FEATURES = [
    'draw', 'draw_norm',
    'horse_win5', 'horse_place5', 'horse_win10', 'horse_place10',
    'days_since',
    'jockey_win_10', 'jockey_place_10', 'jockey_win_30', 'jockey_place_30',
    'jockey_win_venue',
    'trainer_win_10', 'trainer_place_10', 'trainer_win_30', 'trainer_place_30',
    'field_size', 'first_run',
    'class_enc', 'track_enc', 'venue_enc', 'distance',
]
ODDS_FEATURES = [
    'implied_prob', 'log_odds', 'odds_rank', 'odds_ratio', 'prob_norm',
]
ALL_FEATURES = FORM_FEATURES + ODDS_FEATURES

CLASS_MAP = {
    'Class 5': 1, 'Class 4': 2, 'Class 3': 3, 'Class 2': 4, 'Class 1': 5,
    'Pattern Race': 6, 'Group 3': 7, 'Group 2': 8, 'Group 1': 9, '': 0,
}
TRACKS = ['Turf', 'Dirt', 'AWT']
VENUES = ['HV', 'ST']


def _numeric(df, col, default=0.0):
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


def _label_encode(series, categories):
    """Encode exactly like sklearn's LabelEncoder fitted on `categories`.

    LabelEncoder sorts the classes alphabetically and assigns 0..N-1, so
    ['Turf','Dirt','AWT'] becomes AWT=0, Dirt=1, Turf=2. Reproducing that
    exactly matters: an off-by-one here shifts every value of the feature
    while the model still loads and still scores.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(list(categories))
    return pd.Series(le.transform(series.fillna(categories[0])),
                     index=series.index)


def _prior_day_rolling(hit, valid, dates, window):
    """Mean of `hit` over the last `window` VALID rows that belong to a race
    day STRICTLY EARLIER than each row's own day.

    Parameters
    ----------
    hit   : float array, 1/0 indicator (e.g. did the horse win)
    valid : bool array, True where the row has a real finishing position
    dates : array of race dates, the frame must be sorted by date
    window: how many past runs to average over

    Rows are not run yet (finishing position unknown) are excluded from the
    window entirely, so an upcoming race card never contributes zeros to its
    own features.
    """
    n = len(dates)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    vidx = np.flatnonzero(valid)
    if len(vidx) == 0:
        return out
    vhit = np.asarray(hit, dtype=float)[vidx]
    vdates = np.asarray(dates)[vidx]
    uniq, first_idx = np.unique(vdates, return_index=True)
    last_idx = np.append(first_idx[1:] - 1, len(vidx) - 1)
    cs = np.concatenate([[0.0], np.cumsum(vhit)])

    # For each row: how many distinct valid race days are STRICTLY before it.
    cnt_before = np.searchsorted(uniq, dates, side='left')
    anchor = np.where(cnt_before > 0, last_idx[np.maximum(cnt_before - 1, 0)], -1)
    lo = np.maximum(0, anchor - window + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(anchor >= 0,
                       (cs[np.clip(anchor, 0, None) + 1] - cs[lo])
                       / np.maximum(anchor - lo + 1, 1),
                       np.nan)
    out[:] = val
    return out


def _day_rolling(df, entity_cols, value_col, window, extra_mask=None):
    """Apply `_prior_day_rolling` per entity group. `value_col` holds the 1/0
    indicator; `extra_mask` optionally restricts which rows count as valid."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if df.empty:
        return out
    hit_all = df[value_col].to_numpy(dtype=float)
    valid_all = df['finishing_position'].to_numpy() > 0
    if extra_mask is not None:
        valid_all = valid_all & extra_mask
    dates_all = df['race_date'].to_numpy()
    for _, g in df.groupby(entity_cols, sort=False):
        idx = g.index.to_numpy()
        out.loc[idx] = _prior_day_rolling(
            hit_all[idx], valid_all[idx], dates_all[idx], window)
    return out


def build_features(df, class_map=None, tracks=None, venues=None):
    """Build the full feature set.

    Parameters
    ----------
    df : DataFrame with columns
        horse_name, horse_no, draw, jockey, trainer, win_odds,
        finishing_position, race_date, venue, race_no, distance,
        race_class, track
        `finishing_position` may be NaN/0 for a race that has not been run —
        that is exactly the serving case, and every rolling window then uses
        only strictly earlier race days.

    Returns
    -------
    (df_with_features, feature_list)
    """
    class_map = class_map or CLASS_MAP
    tracks = tracks or TRACKS
    venues = venues or VENUES

    df = df.copy()
    df['finishing_position'] = _numeric(df, 'finishing_position')
    df['draw'] = _numeric(df, 'draw')
    df['distance'] = _numeric(df, 'distance', 1200)
    df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
    df['race_date'] = pd.to_datetime(df['race_date'])

    # Canonicalise the jockey entity BEFORE any rolling window is built.
    # The race card writes a claiming ride as "C L Chau (-2)" while the DB
    # stores that rider mostly as plain "C L Chau". Without this, the claim
    # spelling is a separate group with no history and its jockey features
    # come out NaN (filled with 0 at serve time).
    if 'jockey' in df.columns:
        df['jockey'] = df['jockey'].map(normalize_jockey)

    # ── Per-horse form ──────────────────────────────────────────────────────
    df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)
    df['_win'] = (df['finishing_position'] == 1).astype(float)
    df['_place'] = df['finishing_position'].isin([1, 2, 3]).astype(float)
    df['horse_win5'] = _day_rolling(df, ['horse_name'], '_win', 5)
    df['horse_place5'] = _day_rolling(df, ['horse_name'], '_place', 5)
    df['horse_win10'] = _day_rolling(df, ['horse_name'], '_win', 10)
    df['horse_place10'] = _day_rolling(df, ['horse_name'], '_place', 10)

    # days since the horse's previous run (valid runs only)
    prev_date = df.groupby('horse_name')['race_date'].shift(1)
    df['days_since'] = (df['race_date'] - prev_date).dt.days.fillna(90)

    # ── Jockey form, and jockey form AT THIS VENUE ─────────────────────────
    df = df.sort_values(['jockey', 'race_date']).reset_index(drop=True)
    df['jockey_win_10'] = _day_rolling(df, ['jockey'], '_win', 10)
    df['jockey_place_10'] = _day_rolling(df, ['jockey'], '_place', 10)
    df['jockey_win_30'] = _day_rolling(df, ['jockey'], '_win', 30)
    df['jockey_place_30'] = _day_rolling(df, ['jockey'], '_place', 30)

    df = df.sort_values(['jockey', 'venue', 'race_date']).reset_index(drop=True)
    df['jockey_win_venue'] = _day_rolling(df, ['jockey', 'venue'], '_win', 20)

    # ── Trainer form ────────────────────────────────────────────────────────
    df = df.sort_values(['trainer', 'race_date']).reset_index(drop=True)
    df['trainer_win_10'] = _day_rolling(df, ['trainer'], '_win', 10)
    df['trainer_place_10'] = _day_rolling(df, ['trainer'], '_place', 10)
    df['trainer_win_30'] = _day_rolling(df, ['trainer'], '_win', 30)
    df['trainer_place_30'] = _day_rolling(df, ['trainer'], '_place', 30)

    df = df.drop(columns=[c for c in ('_win', '_place') if c in df.columns])

    # ── Race context ────────────────────────────────────────────────────────
    df['field_size'] = df.groupby(
        ['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
    df['draw_norm'] = df['draw'] / df['field_size'].replace(0, 1)
    df['first_run'] = df.groupby('horse_name').cumcount().eq(0).astype(int)

    # ── Odds ────────────────────────────────────────────────────────────────
    # GATE NOTE (2026-10-21): this used to be `win_odds > 1`.
    #
    # HKJC's published 'Win Odds' column legitimately displays "1" for a
    # horse whose starting price was 1.00-1.09 (verified on the official
    # results page: 2022-01-23 ST R7, GOLDEN SIXTY, published cell "1").
    # The published display is a FLOORED value -- for SP < 10 it is
    # floor(SP*10)/10 -- so "1" means "priced at or under 1.09", not
    # "no price".
    #
    # With the old `> 1` gate such a horse was dropped from the odds
    # features entirely, which made it the ONLY horse in its race with a
    # missing odds value. That is an inverted fingerprint: the shortest
    # priced horse in the race is the one the odds features cannot see.
    #
    # So the gate is now `>= 1`: a published "1" is real pre-race market
    # information and is kept. `implied_prob` for that horse is 1/1.0 = 1.0,
    # an upper bound rather than an exact probability, because 1.0 is the
    # floored display of a true price of 1.00-1.09. `prob_norm` then
    # rescales it within the race against the field, which is where the
    # model actually reads it. This affects ~1 row in 900 races, so the
    # numeric impact is negligible; the fingerprint removal is the point.
    df['has_odds'] = df['win_odds'].notna() & (df['win_odds'] >= 1)
    df['implied_prob'] = np.where(df['has_odds'], 1.0 / df['win_odds'], np.nan)
    df['log_odds'] = np.where(df['has_odds'], np.log(df['win_odds']), np.nan)
    grp = df.groupby(['race_date', 'venue', 'race_no'])
    # Rank DESCENDING so the favourite gets the HIGHEST value (1 = longest
    # price). With ascending rank the favourite got 1 while horses with no odds
    # filled to 0 after fillna(), which made "no odds" look BETTER than the
    # favourite.
    df['odds_rank'] = grp['win_odds'].rank(ascending=False, method='min')
    df['odds_min'] = grp['win_odds'].transform('min')
    df['odds_max'] = grp['win_odds'].transform('max')
    df['odds_ratio'] = df['win_odds'] / df['odds_min'].replace(0, np.nan)
    df['field_odds_sum'] = grp['implied_prob'].transform('sum')
    df['prob_norm'] = df['implied_prob'] / df['field_odds_sum']

    # ── Categorical encoding ───────────────────────────────────────────────
    # The encoders must match the training-time code EXACTLY. LabelEncoder
    # sorts its classes alphabetically and assigns 0..N-1, so:
    #   tracks -> ['AWT','Dirt','Turf']  => AWT=0, Dirt=1, Turf=2
    #   venues -> ['HV','ST']            => HV=0,    ST=1
    # Do NOT hand-roll this with enumerate(): that preserves the given order
    # and would silently shift every value.
    df['class_enc'] = df['race_class'].fillna('').map(class_map).fillna(0)
    df['track_enc'] = _label_encode(df['track'].fillna('Turf'), tracks)
    df['venue_enc'] = _label_encode(df['venue'].fillna('ST'), venues)

    return df, list(ALL_FEATURES)
