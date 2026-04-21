"""
Horse racing ML prediction models.
Multi-output: win probability, place probability, finishing position.
"""
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, log_loss, classification_report,
    roc_auc_score, accuracy_score
)
import xgboost as xgb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for each horse-race entry.

    Features:
    - Horse: recent form (last N races), win rate, avg finishing position
    - Jockey: win rate, place rate, recent winners
    - Trainer: win rate, strike rate
    - Race: distance, class, track type, field size
    - Odds: historical win probability implied by odds
    """
    df = df.copy()

    # ---- Horse historical features ----
    # Sort by horse + date
    df = df.sort_values(['horse_name', 'race_date'])

    # Rolling stats per horse (last 5 races)
    for col, agg in [('finishing_position', 'mean'), ('win_odds', 'last')]:
        df[f'horse_avg_{col}'] = df.groupby('horse_name')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    df['horse_races_last30'] = df.groupby('horse_name')['race_date'].transform(
        lambda x: (x.shift(1) > (x.max() - pd.Timedelta(days=30)))
    ).sum()

    # Win/place rate per horse
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
    df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
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


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class HorseBetModel:
    """Train and serve horse racing prediction models."""

    FEATURE_COLS = [
        'draw', 'rating', 'win_odds', 'implied_prob',
        'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
        'jockey_win_rate', 'jockey_place_rate',
        'trainer_win_rate', 'trainer_place_rate',
        'field_size', 'is_different_distance',
        'distance',
    ]

    TARGET_WIN = 'target_win'
    TARGET_PLACE = 'target_place'
    TARGET_TOP4 = 'target_top4'

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.encoders = {}
        self.scalers = {}
        self.models = {}
        self.feature_cols = self.FEATURE_COLS

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets from raw DataFrame."""
        df = engineer_features(df)

        # Drop rows with no finishing position (upcoming races)
        df_train = df.dropna(subset=['finishing_position']).copy()

        # Encode categoricals
        for col in ['venue', 'race_class', 'track']:
            if col in df_train.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_train[col] = self.encoders[col].fit_transform(df_train[col].fillna('Unknown'))
                else:
                    df_train[col] = self.encoders[col].transform(df_train[col].fillna('Unknown').astype(str))

        # Fill missing numerics
        df_train = df_train.fillna(0)

        # Keep only rows with at least basic features
        available_features = [c for c in self.feature_cols if c in df_train.columns]
        X = df_train[available_features].astype(float)
        self.feature_cols = available_features

        y_win = df_train[self.TARGET_WIN]
        y_place = df_train[self.TARGET_PLACE]
        y_top4 = df_train[self.TARGET_TOP4]

        return X, y_win, y_place, y_top4, df_train

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train all models and evaluate."""
        X, y_win, y_place, y_top4, df_train = self.prepare_data(df)

        X_train, X_test, yw_train, yw_test = train_test_split(
            X, y_win, test_size=test_size, random_state=42
        )
        _, _, yp_train, yp_test = train_test_split(
            X, y_place, test_size=test_size, random_state=42
        )
        _, _, yt_train, yt_test = train_test_split(
            X, y_top4, test_size=test_size, random_state=42
        )

        results = {}

        # ---- Win model (XGBoost) ----
        log.info("Training win model...")
        self.models['win'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
        )
        self.models['win'].fit(X_train, yw_train)
        results['win'] = self._evaluate(self.models['win'], X_test, yw_test, y_place=y_place)

        # ---- Place model (XGBoost) ----
        log.info("Training place model...")
        self.models['place'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
        )
        self.models['place'].fit(X_train, yp_train)
        results['place'] = self._evaluate(self.models['place'], X_test, yp_test)

        # ---- Top-4 model ----
        log.info("Training top-4 model...")
        self.models['top4'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
        )
        self.models['top4'].fit(X_train, yt_train)
        results['top4'] = self._evaluate(self.models['top4'], X_test, yt_test)

        return results

    def _evaluate(self, model, X_test, y_test, y_place=None) -> dict:
        """Evaluate a single model."""
        y_prob = model.predict_proba(X_test)[:, 1]

        return {
            'brier_score': brier_score_loss(y_test, y_prob),
            'log_loss': log_loss(y_test, y_prob),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'accuracy': accuracy_score(y_test, (y_prob > 0.5).astype(int)),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict win/place probabilities for each horse entry."""
        df = engineer_features(df.copy())

        # Encode categoricals
        for col in ['venue', 'race_class', 'track']:
            if col in df.columns and col in self.encoders:
                vals = df[col].fillna('Unknown').astype(str)
                # Handle unseen labels
                known = set(self.encoders[col].classes_)
                vals = vals.apply(lambda x: x if x in known else 'Unknown')
                df[col] = self.encoders[col].transform(vals)

        df = df.fillna(0)
        X = df[self.feature_cols].astype(float)

        preds = df[['race_date', 'venue', 'race_no', 'horse_no', 'horse_name', 'win_odds', 'draw', 'rating']].copy()
        preds['prob_win'] = self.models['win'].predict_proba(X)[:, 1]
        preds['prob_place'] = self.models['place'].predict_proba(X)[:, 1]
        preds['prob_top4'] = self.models['top4'].predict_proba(X)[:, 1]

        # Normalize probabilities within each race (softmax-style)
        for col in ['prob_win', 'prob_place', 'prob_top4']:
            total = preds.groupby(['race_date', 'venue', 'race_no'])[col].transform('sum')
            preds[f'{col}_norm'] = preds[col] / total.replace(0, 1)

        # Sort by win probability
        preds = preds.sort_values('prob_win', ascending=False)

        return preds

    def save(self, path: str = "models/model.json"):
        """Save model artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifacts = {
            'models': {k: v.save_model(str(path.parent / f'{k}_xgb.json')) for k, v in self.models.items()},
            'encoders': {k: v.classes_.tolist() for k, v in self.encoders.items()},
            'feature_cols': self.feature_cols,
        }

        with open(path, 'w') as f:
            json.dump(artifacts, f, default=str)

        log.info(f"Model saved to {path}")

    def load(self, path: str = "models/model.json"):
        """Load model artifacts."""
        with open(path) as f:
            artifacts = json.load(f)

        self.feature_cols = artifacts['feature_cols']
        self.encoders = {
            k: LabelEncoder().fit(v) for k, v in artifacts['encoders'].items()
        }

        for name in ['win', 'place', 'top4']:
            model = xgb.XGBClassifier()
            model.load_model(str(Path(path).parent / f'{name}_xgb.json'))
            self.models[name] = model
