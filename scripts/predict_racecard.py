"""
Serve the v6 model for an upcoming race card.

This is the TRAIN→SERVE path. It imports the SAME feature builder used for
training (`features.build_features`), so a race card is scored exactly the way
the training rows were built.

Why the history matters
-----------------------
Form, jockey and trainer features are rolling windows over PAST races. To
score an upcoming card we must load the full history up to (and including) the
race day, then compute the features with the upcoming runners appended and
`finishing_position` left empty. The `.shift(1)` inside the feature builder
guarantees the upcoming race never contributes to its own features.

Usage
-----
    python3 scripts/predict_racecard.py --csv output/card_2026_10_21.csv
    python3 scripts/predict_racecard.py --csv ... --model models/model_v6.pkl

The CSV must contain one row per runner with columns:
    race_no, horse_no, horse_name, jockey, trainer, draw
Optional: win_odds, rating, distance, race_class, track
Anything missing is filled from the database (e.g. distance/class per race).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, 'data', 'horse.db')

HISTORY_SQL = """
    SELECT re.horse_name, re.horse_no, re.draw, re.jockey, re.trainer,
           re.win_odds, re.finishing_position, rd.race_date, rd.venue,
           r.race_no, r.distance, r.race_class, r.track
    FROM horse_entries re
    JOIN races r ON re.race_id = r.id
    JOIN race_days rd ON r.race_day_id = rd.id
    WHERE re.finishing_position IS NOT NULL
      AND re.finishing_position GLOB '[0-9]*'
      AND CAST(re.finishing_position AS INTEGER) > 0
"""


def load_history(conn, race_date, venue):
    """All finished races strictly before race_date, plus the race day itself
    is excluded (the card we are scoring is not a finished race)."""
    df = pd.read_sql(HISTORY_SQL, conn)
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df[df['race_date'] < pd.Timestamp(race_date)].copy()
    return df


def build_card(race_date, venue, csv_path):
    card = pd.read_csv(csv_path)
    card['race_date'] = pd.Timestamp(race_date)
    card['venue'] = venue
    card['finishing_position'] = np.nan   # not run yet
    card['win_odds'] = pd.to_numeric(card.get('win_odds'), errors='coerce')
    if 'rating' in card.columns:
        card['rating'] = pd.to_numeric(card['rating'], errors='coerce')
    else:
        card['rating'] = np.nan
    card['draw'] = pd.to_numeric(card['draw'], errors='coerce')
    card['race_no'] = pd.to_numeric(card['race_no'], errors='coerce').astype(int)
    card['horse_no'] = pd.to_numeric(card['horse_no'], errors='coerce').astype(int)

    # Drop withdrawn runners BEFORE any feature is built. A scratched horse
    # still occupies a horse number and a card row, so leaving it in
    # (a) inflates field_size for the whole race, and (b) puts a NaN-odds
    # runner into the model's top-3, because a missing price produces no
    # odds features at all and the model then ranks it on form alone.
    if 'withdrawn' in card.columns:
        w = card['withdrawn'].astype(str).str.lower().isin(
            ['true', '1', 'yes'])
    else:
        w = pd.Series(False, index=card.index)
    w |= card['horse_name'].astype(str).str.contains(
        'Scratched', case=False, na=False)
    w |= card['draw'].isna()
    if w.any():
        for _, r in card[w].iterrows():
            print(f"  withdrawn, excluded: R{r['race_no']} #{r['horse_no']} "
                  f"{r['horse_name']}")
        card = card[~w].copy()
    card = card.reset_index(drop=True)
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='race card CSV')
    ap.add_argument('--date', required=True, help='race date YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV or ST')
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'model_v6.pkl'))
    ap.add_argument('--out', default=None, help='output CSV path')
    args = ap.parse_args()

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True, timeout=60)
    hist = load_history(conn, args.date, args.venue)
    card = build_card(args.date, args.venue, args.csv)
    conn.close()
    print(f"history rows: {len(hist):,} | card runners: {len(card)}")

    # Fill race-level metadata (distance/class/track) from the card itself.
    # The card carries the race header scraped from the official racecard
    # page, which is the ONLY correct source for THIS meeting. Falling back
    # to DB history is tempting but wrong: history is keyed by race_no, and
    # race_no N of a past meeting is a different distance and class from
    # race_no N of tonight (R1 of 10/21 HV is 1650m Class 4; the most recent
    # HV race 1 in the DB is a different race entirely).
    if 'distance' in card.columns and card['distance'].notna().any():
        card['distance'] = pd.to_numeric(card['distance'], errors='coerce')
        if 'race_class' not in card.columns:
            card['race_class'] = ''
        if 'track' not in card.columns:
            card['track'] = 'Turf'
        print("race metadata taken from the official race card")
    else:
        print("WARNING: card has no race metadata; falling back to DB history "
              "(race_no keyed, so class/distance may be WRONG for tonight)")
        meta = (hist[hist['venue'] == args.venue]
                .sort_values('race_date')
                .groupby('race_no')
                .agg(distance=('distance', 'last'), race_class=('race_class', 'last'),
                     track=('track', 'last')))
        card = card.merge(meta, left_on='race_no', right_index=True, how='left')
        print("race metadata filled from DB history:", meta.to_dict('index'))

    # The upcoming runners must be appended to the history so the rolling
    # windows (and field_size) see the full field. The feature builder shifts
    # by 1, so the upcoming race never feeds its own features.
    combined = pd.concat([hist, card], ignore_index=True, sort=False)
    feat_df, feat_list = F.build_features(combined)

    served = feat_df[feat_df['race_date'] == pd.Timestamp(args.date)].copy()
    if len(served) != len(card):
        print(f"WARNING: expected {len(card)} served rows, got {len(served)}")

    pkg = pickle.load(open(args.model, 'rb'))
    model_feats = pkg['feature_cols']
    missing = [c for c in model_feats if c not in feat_list]
    if missing:
        print(f"ERROR: model expects features not produced by the builder: {missing}")
        return 2

    X = served[model_feats].astype(float).fillna(0).values
    served['p_model'] = (pkg['xgb_model'].predict_proba(X)[:, 1] * 0.5
                          + pkg['rf_model'].predict_proba(X)[:, 1] * 0.5)
    # Market reference: the favourite (shortest price) in each race.
    served['p_fav'] = -served['win_odds']

    out_rows = []
    for rn, g in served.groupby('race_no'):
        g = g.sort_values('p_model', ascending=False)
        fav = g.sort_values('p_fav', ascending=False).iloc[0]
        top1 = g.iloc[0]
        top3 = g.head(3)
        out_rows.append({
            'race': int(rn),
            'n_runners': len(g),
            'model_top1': top1['horse_name'],
            'model_top1_odds': top1.get('win_odds'),
            'model_top3': ' | '.join(top3['horse_name'].tolist()),
            'favourite': fav['horse_name'],
            'favourite_odds': fav.get('win_odds'),
            'model_picks_favourite': bool(top1['horse_name'] == fav['horse_name']),
            'model_top1_p': round(float(top1['p_model']), 4),
            'fav_p': round(float(fav['p_model']), 4),
        })
        print(f"\nR{rn} ({len(g)} runners)")
        print(f"  model top3 : " + ", ".join(
            f"{r['horse_name']}({r.get('win_odds')})" for _, r in top3.iterrows()))
        print(f"  favourite  : {fav['horse_name']} ({fav.get('win_odds')})")

    out = pd.DataFrame(out_rows)
    path = args.out or os.path.join(ROOT, 'output', f"predict_{args.date.replace('-', '_')}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nsaved: {path}")
    print(f"model picked the favourite in {int(out['model_picks_favourite'].sum())}/{len(out)} races")
    return 0


if __name__ == '__main__':
    sys.exit(main())
