"""
Temporal backtest for horse-bet ML model.
Train on all meetings BEFORE the test period, evaluate on the last 3 race meetings.
Report ROC AUC and hit rate for predicted 1st/2nd/3rd positions.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DB_PATH = '/home/work/horse-bet/data/horse.db'
MODEL_OUTPUT_DIR = Path('/home/work/horse-bet/models')
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
)

FEATURE_COLS = [
    'draw', 'rating', 'win_odds', 'implied_prob',
    'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
    'jockey_win_rate', 'jockey_place_rate',
    'trainer_win_rate', 'trainer_place_rate',
    'field_size', 'is_different_distance',
    'distance',
]

TARGETS = ['target_win', 'target_place', 'target_top4']


def load_data() -> pd.DataFrame:
    """Load and join horse_entries + races + race_days. Excludes void/cancelled races."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            he.id,
            he.race_id,
            he.horse_no,
            he.draw,
            he.horse_name,
            he.horse_id,
            he.jockey,
            he.trainer,
            he.rating,
            he.win_odds,
            he.place_odds,
            he.finishing_position,
            r.distance,
            r.race_class,
            r.track,
            rd.race_date,
            rd.venue
        FROM horse_entries he
        JOIN races r ON he.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        LEFT JOIN race_results rr ON rr.race_id = r.id
        WHERE rd.race_date IS NOT NULL
          AND he.finishing_position IS NOT NULL
          AND (rr.win_dividend IS NULL OR rr.win_dividend > 1)
        ORDER BY rd.race_date, rd.venue, r.race_no, he.horse_no
    """, conn)
    conn.close()

    df['race_date'] = pd.to_datetime(df['race_date'])
    log.info(f"Loaded {len(df)} entries from {df['race_date'].nunique()} race days")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features IN-PLACE on a date-sorted DataFrame.
    Uses shift(1) rolling so current race data is never used.
    """
    df = df.copy()
    df = df.sort_values(['horse_name', 'race_date']).reset_index(drop=True)

    # Horse rolling stats (shift to exclude current race)
    df['horse_avg_finishing_position'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['horse_win_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(10, min_periods=1).mean()
    )
    df['horse_place_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(10, min_periods=1).mean()
    )

    # Jockey features
    df['jockey_win_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )

    # Trainer features
    df['trainer_win_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )

    # Race context
    df['field_size'] = df.groupby(['race_date', 'venue', 'race_id'])['horse_no'].transform('count')
    df['is_different_distance'] = df.groupby('horse_name')['distance'].transform(
        lambda x: (x != x.shift(1)) & x.shift(1).notna()
    ).astype(int)

    # Odds
    df['implied_prob'] = 1 / df['win_odds'].replace(0, np.nan)

    # Targets
    df['target_win'] = (df['finishing_position'] == 1).astype(int)
    df['target_place'] = df['finishing_position'].isin([1, 2, 3]).astype(int)
    df['target_top4'] = df['finishing_position'].isin([1, 2, 3, 4]).astype(int)

    return df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoders: dict):
    """
    Engineer features on combined train+test (sorted by date) then split.
    Rolling features use shift(1) so only past data is used.
    """
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(['race_date', 'venue', 'race_id', 'horse_no']).reset_index(drop=True)
    combined = build_features(combined)

    # Split back
    test_dates_venues = set(zip(test_df['race_date'], test_df['venue']))
    test_mask = combined.apply(lambda r: (r['race_date'], r['venue']) in test_dates_venues, axis=1)
    fold_train = combined[~test_mask].copy()
    fold_test = combined[test_mask].copy()

    # Encode categoricals
    for col in ['venue', 'race_class', 'track']:
        if col not in label_encoders:
            le = LabelEncoder()
            all_vals = pd.concat([train_df[col], test_df[col]]).fillna('Unknown').astype(str)
            le.fit(all_vals)
            label_encoders[col] = le
        else:
            le = label_encoders[col]

        fold_train[col] = le.transform(fold_train[col].fillna('Unknown').astype(str))
        unknown_mask = ~fold_test[col].fillna('Unknown').astype(str).isin(le.classes_)
        fold_test[col] = fold_test[col].fillna('Unknown').astype(str)
        fold_test.loc[unknown_mask, col] = 'Unknown'
        fold_test[col] = le.transform(fold_test[col])

    fold_train = fold_train.fillna(0)
    fold_test = fold_test.fillna(0)

    available_features = [c for c in FEATURE_COLS if c in fold_train.columns and c in fold_test.columns]

    X_train = fold_train[available_features].astype(float)
    X_test = fold_test[available_features].astype(float)

    return X_train, X_test, fold_train, fold_test, available_features


def hit_rate_at_k(y_true, y_prob, k):
    """
    Compute hit rate at top-k predictions per race.
    For position p (1=1st, 2=2nd, 3=3rd), check if actual p-th finisher
    is among the top-k predicted horses.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['actual_pos'] = df['y_true']  # 1=win, 0=no-win
    return 0.0


def compute_metrics_per_position(X_test, fold_test, model, target_col):
    """
    Compute ROC AUC and hit rate for each predicted position (1st, 2nd, 3rd).

    For position p:
    - ROC AUC: standard AUC using actual finishing position == p as binary label
    - Hit rate at k: for each race, if the top-k predicted horses contain
      the actual p-th finisher, that's a hit

    Returns dict with per-position metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    results_df = fold_test[['race_id', 'horse_no', 'finishing_position', target_col]].copy()
    results_df['prob_win'] = y_prob

    metrics = {}

    for position in [1, 2, 3]:
        # Binary label: did this horse finish in exact position `position`?
        y_true = (results_df['finishing_position'] == position).astype(int)

        if y_true.sum() == 0:
            continue

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = float('nan')

        # Hit rate: for each race, check if the actual `position`-finisher
        # is in the top-`position` predicted horses
        hit_count = 0
        total_count = 0

        for race_id, race_df in results_df.groupby('race_id'):
            # Find the actual horse that finished in `position`
            actual_horse = race_df[race_df['finishing_position'] == position]
            if len(actual_horse) == 0:
                continue

            # Top-k predicted horses for this race
            top_k = race_df.nlargest(position, 'prob_win')['horse_no'].tolist()
            actual_horse_no = actual_horse['horse_no'].values[0]

            total_count += 1
            if actual_horse_no in top_k:
                hit_count += 1

        hit_rate = hit_count / total_count if total_count > 0 else 0.0

        label_name = {1: '1st', 2: '2nd', 3: '3rd'}[position]
        metrics[label_name] = {
            'roc_auc': float(roc_auc),
            'hit_rate': round(float(hit_rate), 4),
            'n_races': int(total_count),
            'n_hits': int(hit_count),
        }

    return metrics


def run_backtest():
    log.info("Loading data...")
    df = load_data()

    df = df.sort_values(['race_date', 'venue', 'race_id', 'horse_no']).reset_index(drop=True)

    # Get unique meetings (date, venue) in temporal order
    meetings = df.groupby(['race_date', 'venue'], sort=True).size().reset_index(name='n_entries')
    meetings = meetings.sort_values('race_date').reset_index(drop=True)
    log.info(f"Total meetings: {len(meetings)}")

    # Last 3 meetings = test set
    test_meetings = meetings.tail(3)
    log.info(f"Test meetings:\n{test_meetings}")

    test_dates_venues = set(zip(test_meetings['race_date'], test_meetings['venue']))

    train_df = df[~df.apply(lambda r: (r['race_date'], r['venue']) in test_dates_venues, axis=1)].copy()
    test_df = df[df.apply(lambda r: (r['race_date'], r['venue']) in test_dates_venues, axis=1)].copy()

    log.info(f"Train entries: {len(train_df)}, Test entries: {len(test_df)}")
    log.info(f"Train date range: {train_df['race_date'].min()} to {train_df['race_date'].max()}")
    log.info(f"Test date range: {test_df['race_date'].min()} to {test_df['race_date'].max()}")

    label_encoders = {}

    # Prepare features
    X_train, X_test, fold_train, fold_test, feat_cols = prepare_data(
        train_df, test_df, label_encoders
    )

    log.info(f"Feature columns: {feat_cols}")
    log.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Train models for each target and compute metrics
    all_metrics = {}

    for target_col, target_name in [('target_win', 'Win'), ('target_place', 'Place'), ('target_top4', 'Top4')]:
        y_train = fold_train[target_col].values
        y_test = fold_test[target_col].values

        if y_train.sum() == 0 or y_test.sum() == 0:
            log.warning(f"Skipping {target_col}: no positive samples")
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        # Standard ROC AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = float('nan')

        # Per-position metrics
        pos_metrics = compute_metrics_per_position(X_test, fold_test, model, target_col)

        all_metrics[target_name] = {
            'overall_roc_auc': float(roc_auc),
            'per_position': pos_metrics,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        }

        log.info(f"\n{target_name}:")
        log.info(f"  Overall ROC AUC: {roc_auc:.4f}")
        for pos in ['1st', '2nd', '3rd']:
            if pos in pos_metrics:
                log.info(f"  {pos} - ROC AUC: {pos_metrics[pos]['roc_auc']:.4f}, "
                        f"Hit Rate@{pos}: {pos_metrics[pos]['hit_rate']:.4f} "
                        f"({pos_metrics[pos]['n_hits']}/{pos_metrics[pos]['n_races']})")

    # Summary output
    output = {
        'generated_at': datetime.now().isoformat(),
        'xgb_params': XGB_PARAMS,
        'feature_cols': feat_cols,
        'test_meetings': [
            {'race_date': str(d), 'venue': v}
            for d, v in test_dates_venues
        ],
        'metrics': all_metrics,
    }

    out_path = MODEL_OUTPUT_DIR / 'temporal_backtest_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved results to {out_path}")

    log.info("\n" + "=" * 70)
    log.info("TEMPORAL BACKTEST SUMMARY")
    sorted_test = sorted(test_dates_venues, key=lambda x: str(x[0]))
    log.info("Train: all meetings before %s" % sorted_test[0])
    log.info("Test: last 3 meetings = %s" % str(sorted_test))
    log.info("=" * 70)

    for target_name, m in all_metrics.items():
        log.info(f"\n{target_name} (overall ROC AUC: {m['overall_roc_auc']:.4f}):")
        for pos in ['1st', '2nd', '3rd']:
            if pos in m['per_position']:
                pm = m['per_position'][pos]
                log.info(f"  {pos:4s}  ROC AUC: {pm['roc_auc']:.4f}  "
                         f"Hit Rate@{pos}: {pm['hit_rate']:.4f}  ({pm['n_hits']}/{pm['n_races']})")

    return output


if __name__ == '__main__':
    run_backtest()