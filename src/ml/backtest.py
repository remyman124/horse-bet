"""
Temporal rolling backtest for horse-bet ML model.
Strict temporal split: train on all races BEFORE each date, test on that date.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
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
    'combo_win_rate',
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


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features IN-PLACE on a pre-sorted (race_date ASC) DataFrame.
    This must be called AFTER sorting by race_date and within each fold
    to avoid future data leakage.

    Same logic as models.py engineer_features().
    """
    df = df.copy()

    # ---- Horse historical features ----
    # Sort by horse + date (should already be sorted, but ensure it)
    df = df.sort_values(['horse_name', 'race_date'])

    # Rolling stats per horse (last 5 races) - shift(1) to exclude current race
    df['horse_avg_finishing_position'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['horse_avg_win_odds'] = df.groupby('horse_name')['win_odds'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Horse win/place rate (shifted to avoid leakage)
    df['horse_win_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(10, min_periods=1).mean()
    )
    df['horse_place_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(10, min_periods=1).mean()
    )

    # ---- Jockey features ----
    df['jockey_win_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )

    # ---- Trainer features ----
    df['trainer_win_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )

    # ---- Race context features ----
    df['field_size'] = df.groupby(['race_date', 'venue', 'race_id'])['horse_no'].transform('count')
    df['is_different_distance'] = df.groupby('horse_name')['distance'].transform(
        lambda x: (x != x.shift(1)) & x.shift(1).notna()
    ).astype(int)

    # ---- Odds as probability ----
    df['implied_prob'] = 1 / df['win_odds'].replace(0, np.nan)

    # ---- Target variables ----
    df['target_win'] = (df['finishing_position'] == 1).astype(int)
    df['target_place'] = df['finishing_position'].isin([1, 2, 3]).astype(int)
    df['target_top4'] = df['finishing_position'].isin([1, 2, 3, 4]).astype(int)

    return df


def _prepare_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoders: dict):
    """
    Engineer features and prepare train/test matrices for one fold.
    Features are engineered on combined train+test sorted by date,
    then split - but rolling features only use past data (shift/rolling).
    """
    # Combine and sort for proper rolling feature computation
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(['race_date', 'venue', 'race_id', 'horse_no']).reset_index(drop=True)

    # Engineer features on full combined set (shift ensures no future leakage)
    combined = _build_features(combined)

    # Split back
    test_dates = set(test_df['race_date'].unique())
    test_mask = combined['race_date'].isin(test_dates)
    fold_train = combined[~test_mask].copy()
    fold_test = combined[test_mask].copy()

    # Encode categoricals
    for col in ['venue', 'race_class', 'track']:
        if col not in label_encoders:
            le = LabelEncoder()
            # Fit on all known values from training
            all_vals = pd.concat([train_df[col], test_df[col]]).fillna('Unknown').astype(str)
            le.fit(all_vals)
            label_encoders[col] = le
        else:
            le = label_encoders[col]

        fold_train[col] = le.transform(fold_train[col].fillna('Unknown').astype(str))
        fold_test[col] = fold_test[col].fillna('Unknown').astype(str).apply(
            lambda x: x if x in le.classes_ else 'Unknown'
        )
        fold_test[col] = le.transform(fold_test[col])

    # Fill missing numerics
    fold_train = fold_train.fillna(0)
    fold_test = fold_test.fillna(0)

    # Filter to available features
    available_features = [c for c in FEATURE_COLS if c in fold_train.columns and c in fold_test.columns]

    X_train = fold_train[available_features].astype(float)
    X_test = fold_test[available_features].astype(float)

    return X_train, X_test, fold_train, fold_test, available_features


def _evaluate_model(model, X_test, y_test):
    """Compute metrics for a single model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    return {
        'brier_score': float(brier_score_loss(y_test, y_prob)),
        'log_loss': float(log_loss(y_test, y_prob)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
    }


def run_backtest():
    """Main temporal backtest loop."""
    log.info("Loading data...")
    df = load_data()

    # Ensure sorted by date
    df = df.sort_values(['race_date', 'venue', 'race_id', 'horse_no']).reset_index(drop=True)

    # Get unique (race_date, venue) combinations in order
    race_days = df.groupby(['race_date', 'venue'], sort=True).size().reset_index(name='n_entries')
    race_days = race_days.sort_values('race_date').reset_index(drop=True)
    log.info(f"Total race days to iterate: {len(race_days)}")

    all_results = []
    label_encoders = {}

    for i, (_, row) in enumerate(race_days.itertuples()):
        test_date = row.race_date
        test_venue = row.venue

        # Train = all entries from strictly before test_date
        train_mask = df['race_date'] < test_date
        test_mask = (df['race_date'] == test_date) & (df['venue'] == test_venue)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(test_df) == 0:
            continue

        # Need at least some training samples
        if len(train_df) < 10:
            log.debug(f"Skip fold {i}: only {len(train_df)} training samples")
            continue

        # Prepare features
        try:
            X_train, X_test, fold_train, fold_test, feat_cols = _prepare_fold(
                train_df, test_df, label_encoders
            )
        except Exception as e:
            log.warning(f"Fold {i} ({test_date}, {test_venue}) failed during feature prep: {e}")
            continue

        if len(X_test) == 0 or X_train.shape[0] == 0:
            continue

        fold_metrics = {'target': {}, 'n_train': len(X_train), 'n_test': len(X_test)}

        for target in TARGETS:
            y_train = fold_train[target].values
            y_test = fold_test[target].values

            if y_train.sum() == 0 or y_test.sum() == 0:
                # Not enough positive samples - skip this target for this fold
                log.debug(f"Skip target {target} fold {i}: no positive samples")
                continue

            try:
                model = xgb.XGBClassifier(**XGB_PARAMS)
                model.fit(X_train, y_train)
                metrics = _evaluate_model(model, X_test, y_test)
                fold_metrics['target'][target] = metrics
            except Exception as e:
                log.warning(f"Training failed for {target} at fold {i}: {e}")
                continue

        if fold_metrics.get('target'):
            fold_metrics['date'] = str(test_date.date())
            fold_metrics['venue'] = test_venue
            all_results.append(fold_metrics)

        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(race_days)} race days")

    log.info(f"Completed backtest: {len(all_results)} folds with valid results")

    # ---- Aggregate metrics ----
    aggregate = _aggregate_metrics(all_results)
    aggregate['n_folds'] = len(all_results)
    aggregate['n_folds_per_target'] = {
        t: sum(1 for r in all_results if t in r.get('target', {})) for t in TARGETS
    }

    # Per-month breakdown
    per_month = _per_month_breakdown(all_results)

    output = {
        'generated_at': datetime.now().isoformat(),
        'xgb_params': XGB_PARAMS,
        'feature_cols': FEATURE_COLS,
        'aggregate': aggregate,
        'per_month': per_month,
        'folds': all_results,
    }

    out_path = MODEL_OUTPUT_DIR / 'backtest_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Saved backtest metrics to {out_path}")

    # Print summary
    log.info("=" * 60)
    log.info("BACKTEST SUMMARY")
    log.info("=" * 60)
    for target in TARGETS:
        agg = aggregate.get(target, {})
        log.info(f"\n{target}:")
        log.info(f"  Brier Score : {agg.get('brier_score', 0):.4f} (weighted avg)")
        log.info(f"  Log Loss    : {agg.get('log_loss', 0):.4f} (weighted avg)")
        log.info(f"  ROC AUC     : {agg.get('roc_auc', 0):.4f} (weighted avg)")
        log.info(f"  Accuracy    : {agg.get('accuracy', 0):.4f} (weighted avg)")

    return output


def _aggregate_metrics(results: list) -> dict:
    """Compute weighted average metrics across all folds."""
    agg = {}
    for target in TARGETS:
        weights = []
        weighted_brier = []
        weighted_logloss = []
        weighted_roc = []
        weighted_acc = []

        for r in results:
            m = r.get('target', {}).get(target)
            if m is None:
                continue
            n = r['n_test']
            weights.append(n)
            weighted_brier.append(m['brier_score'] * n)
            weighted_logloss.append(m['log_loss'] * n)
            weighted_roc.append(m['roc_auc'] * n)
            weighted_acc.append(m['accuracy'] * n)

        total = sum(weights)
        if total > 0:
            agg[target] = {
                'brier_score': sum(weighted_brier) / total,
                'log_loss': sum(weighted_logloss) / total,
                'roc_auc': sum(weighted_roc) / total,
                'accuracy': sum(weighted_acc) / total,
                'total_test_samples': total,
            }
    return agg


def _per_month_breakdown(results: list) -> dict:
    """Compute per-month aggregated metrics."""
    monthly = {}
    for r in results:
        date_str = r.get('date')
        if not date_str:
            continue
        try:
            month_key = date_str[:7]  # YYYY-MM
        except Exception:
            continue

        if month_key not in monthly:
            monthly[month_key] = {'weights': [], 'brier': [], 'logloss': [], 'roc': [], 'acc': []}

        for target in TARGETS:
            m = r.get('target', {}).get(target)
            if m is None:
                continue
            n = r['n_test']
            monthly[month_key]['weights'].append(n)
            monthly[month_key]['brier'].append(m['brier_score'] * n)
            monthly[month_key]['logloss'].append(m['log_loss'] * n)
            monthly[month_key]['roc'].append(m['roc_auc'] * n)
            monthly[month_key]['acc'].append(m['accuracy'] * n)

    breakdown = {}
    for month, vals in sorted(monthly.items()):
        total = sum(vals['weights'])
        if total > 0:
            breakdown[month] = {
                'n_test_samples': total,
                'brier_score': sum(vals['brier']) / total,
                'log_loss': sum(vals['logloss']) / total,
                'roc_auc': sum(vals['roc']) / total,
                'accuracy': sum(vals['acc']) / total,
            }
    return breakdown


if __name__ == '__main__':
    run_backtest()
