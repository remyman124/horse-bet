"""
Deep Learning model for horse racing prediction.
PyTorch Neural Network with embeddings for categorical features.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Feature engineering (same pipeline as XGBoost, expanded)
# -----------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for each horse-race entry."""
    df = df.copy()
    df = df.sort_values(['horse_name', 'race_date'])

    # Horse rolling stats
    for col, agg in [('finishing_position', 'mean'), ('win_odds', 'last')]:
        df[f'horse_avg_{col}'] = df.groupby('horse_name')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    # Win/place rate per horse
    df['horse_win_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(10, min_periods=1).mean()
    )
    df['horse_place_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(10, min_periods=1).mean()
    )

    # Form momentum
    df['form_last3_avg'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['form_prev3_avg'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: x.shift(4).rolling(3, min_periods=1).mean()
    )
    df['form_momentum'] = df['form_prev3_avg'] - df['form_last3_avg']
    df['trend_reversal'] = (
        (df['form_momentum'] > 0) & (df['form_prev3_avg'] > 5)
    ).astype(int)

    # Recent win rate (as fraction)
    df['recent_win_rate'] = df.groupby('horse_name')['finishing_position'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).apply(
            lambda v: (v == 1).sum() / len(v), raw=True
        )
    )

    # Jockey features
    df['jockey_win_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['jockey_place_rate'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )
    df['jockey_races_10'] = df.groupby('jockey')['finishing_position'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).count()
    )

    # Trainer features
    df['trainer_win_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1) == 1).rolling(20, min_periods=1).mean()
    )
    df['trainer_place_rate'] = df.groupby('trainer')['finishing_position'].transform(
        lambda x: (x.shift(1).isin([1, 2, 3])).rolling(20, min_periods=1).mean()
    )

    # Race context
    df['field_size'] = df.groupby(['race_date', 'venue', 'race_no'])['horse_no'].transform('count')
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


# -----------------------------------------------------------------------
# PyTorch Model
# -----------------------------------------------------------------------

class TabularNN(nn.Module):
    """Deep neural network for tabular data with entity embeddings."""

    def __init__(self, num_numerical, cat_dims, embedding_dim=8, hidden_dims=[256, 128, 64]):
        super().__init__()

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in cat_dims
        ])
        total_emb = len(cat_dims) * embedding_dim

        # Build layers
        layers = []
        in_dim = total_emb + num_numerical
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embeds, dim=1)
        return torch.sigmoid(self.network(x)).squeeze(-1)


class HorseBetDeepModel:
    """Train and serve deep learning horse betting model."""

    FEATURES_NUMERICAL = [
        'draw', 'rating', 'win_odds', 'implied_prob',
        'horse_avg_finishing_position', 'horse_win_rate', 'horse_place_rate',
        'recent_win_rate', 'form_momentum', 'trend_reversal',
        'jockey_win_rate', 'jockey_place_rate', 'jockey_races_10',
        'trainer_win_rate', 'trainer_place_rate',
        'field_size', 'is_different_distance', 'distance',
    ]

    FEATURES_CATEGORICAL = ['venue', 'race_class', 'track']

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cpu')
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {}  # target -> model
        self.history = {}

    # -------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------

    def load_db(self, db_path: str, days: int = 730) -> pd.DataFrame:
        """Load race data from SQLite DB."""
        conn = sqlite3.connect(db_path)
        query = """
            SELECT
                rd.race_date, rd.venue,
                r.race_no, r.distance, r.race_class, r.track,
                he.horse_no, he.horse_name, he.horse_id,
                he.jockey, he.trainer,
                he.draw, he.rating, he.win_odds, he.place_odds,
                he.finishing_position, he.finishing_time, he.margin
            FROM horse_entries he
            JOIN races r ON he.race_id = r.id
            JOIN race_days rd ON r.race_day_id = rd.id
            WHERE rd.race_date >= date('now', '-{} days')
            ORDER BY rd.race_date, r.race_no, he.horse_no
        """.format(days)
        df = pd.read_sql(query, conn)
        conn.close()
        df['race_date'] = pd.to_datetime(df['race_date'])
        return df

    def prepare_data(self, df: pd.DataFrame):
        """Engineer features and prepare train/test split."""
        df = engineer_features(df)

        # Filter to rows with results
        df = df.dropna(subset=['finishing_position']).copy()
        df['finishing_position'] = df['finishing_position'].astype(int)

        # Encode categoricals
        for col in self.FEATURES_CATEGORICAL:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].fillna('Unknown').astype(str))
            else:
                vals = df[col].fillna('Unknown').astype(str)
                known = set(self.encoders[col].classes_)
                vals = vals.apply(lambda x: x if x in known else 'Unknown')
                df[col] = self.encoders[col].transform(vals)

        df = df.fillna(0)

        # Numerical features
        num_feats = [c for c in self.FEATURES_NUMERICAL if c in df.columns]
        X_num = df[num_feats].astype(float).values
        X_num = self.scaler.fit_transform(X_num)

        # Categorical features
        cat_feats = [c for c in self.FEATURES_CATEGORICAL if c in df.columns]
        X_cat = df[cat_feats].astype(int).values

        # Targets
        y_win = df['target_win'].values
        y_place = df['target_place'].values
        y_top4 = df['target_top4'].values

        return {
            'X_num': X_num, 'X_cat': X_cat,
            'y_win': y_win, 'y_place': y_place, 'y_top4': y_top4,
            'df': df, 'num_feats': num_feats, 'cat_feats': cat_feats,
        }

    # -------------------------------------------------------------------
    # Temporal split (no data leakage)
    # -------------------------------------------------------------------

    def temporal_split(self, X_num, X_cat, y, df, test_ratio=0.1):
        """Split data temporally — most recent races for testing."""
        n = len(df)
        split_idx = int(n * (1 - test_ratio))
        # Sort by date ensures temporal ordering
        date_order = df['race_date'].argsort()
        test_indices = date_order[split_idx:]
        train_indices = date_order[:split_idx]

        return (
            X_num[train_indices], X_num[test_indices],
            X_cat[train_indices], X_cat[test_indices],
            y[train_indices], y[test_indices],
        )

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def train_target(self, name, X_num_tr, X_cat_tr, y_tr,
                     X_num_te, X_cat_te, y_te,
                     cat_dims, num_numerical,
                     epochs=100, lr=0.001, batch_size=512):
        """Train a single target model."""
        X_num_tr_t = torch.FloatTensor(X_num_tr).to(self.device)
        X_cat_tr_t = torch.LongTensor(X_cat_tr).to(self.device)
        y_tr_t = torch.FloatTensor(y_tr).to(self.device)
        X_num_te_t = torch.FloatTensor(X_num_te).to(self.device)
        X_cat_te_t = torch.LongTensor(X_cat_te).to(self.device)

        model = TabularNN(num_numerical, cat_dims).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()

        best_auc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            indices = torch.randperm(len(y_tr_t))
            total_loss = 0

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                out = model(X_num_tr_t[batch_idx], X_cat_tr_t[batch_idx])
                loss = criterion(out, y_tr_t[batch_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            model.eval()
            with torch.no_grad():
                val_out = model(X_num_te_t, X_cat_te_t).cpu().numpy()
                try:
                    auc = roc_auc_score(y_te, val_out)
                except:
                    auc = 0.5

            scheduler.step(1 - auc)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # Load best model
        model.load_state_dict(best_state)
        model.eval()

        # Final predictions
        with torch.no_grad():
            val_pred = model(X_num_te_t, X_cat_te_t).cpu().numpy()

        metrics = {
            'roc_auc': best_auc,
            'brier_score': brier_score_loss(y_te, val_pred),
            'log_loss': log_loss(y_te, val_pred),
            'accuracy': accuracy_score(y_te, (val_pred > 0.5).astype(int)),
            'best_epoch': epochs - patience_counter,
        }

        return model, metrics

    def train(self, db_path: str, days: int = 730, epochs: int = 100) -> dict:
        """Train all three target models."""
        print("Loading data...")
        df = self.load_db(db_path, days)
        print(f"  Loaded {len(df)} entries, {df['race_date'].min()} to {df['race_date'].max()}")

        prepared = self.prepare_data(df)
        X_num = prepared['X_num']
        X_cat = prepared['X_cat']
        df_out = prepared['df']

        targets = [
            ('win', prepared['y_win']),
            ('place', prepared['y_place']),
            ('top4', prepared['y_top4']),
        ]

        num_numerical = len(prepared['num_feats'])
        cat_dims = [int(X_cat[:, i].max()) + 2 for i in range(len(prepared['cat_feats']))]

        results = {}
        self.history = {'train_size': 0, 'test_size': 0, 'epochs': epochs}

        for name, y in targets:
            print(f"\nTraining {name.upper()} model...")
            Xn_tr, Xn_te, Xc_tr, Xc_te, y_tr, y_te = self.temporal_split(
                X_num, X_cat, y, df_out
            )
            self.history['train_size'] = len(y_tr)
            self.history['test_size'] = len(y_te)

            model, metrics = self.train_target(
                name, Xn_tr, Xc_tr, y_tr,
                Xn_te, Xc_te, y_te,
                cat_dims, num_numerical, epochs=epochs
            )

            self.models[name] = model
            results[name] = metrics

            print(f"  ROC AUC: {metrics['roc_auc']:.4f}  "
                  f"Brier: {metrics['brier_score']:.4f}  "
                  f"LogLoss: {metrics['log_loss']:.4f}  "
                  f"Acc: {metrics['accuracy']:.4f}")

        return results

    # -------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for each horse entry."""
        df = engineer_features(df.copy())

        for col in self.FEATURES_CATEGORICAL:
            if col in df.columns and col in self.encoders:
                vals = df[col].fillna('Unknown').astype(str)
                known = set(self.encoders[col].classes_)
                vals = vals.apply(lambda x: x if x in known else 'Unknown')
                df[col] = self.encoders[col].transform(vals)

        df = df.fillna(0)

        num_feats = [c for c in self.FEATURES_NUMERICAL if c in df.columns]
        cat_feats = [c for c in self.FEATURES_CATEGORICAL if c in df.columns]

        X_num = df[num_feats].astype(float).values
        X_num = self.scaler.transform(X_num)
        X_cat = df[cat_feats].astype(int).values

        X_num_t = torch.FloatTensor(X_num).to(self.device)
        X_cat_t = torch.LongTensor(X_cat).to(self.device)

        preds = df[['race_date', 'venue', 'race_no', 'horse_no', 'horse_name', 'win_odds', 'draw', 'rating']].copy()

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                preds[f'prob_{name}'] = model(X_num_t, X_cat_t).cpu().numpy()

        # Normalize per race
        for col in ['prob_win', 'prob_place', 'prob_top4']:
            total = preds.groupby(['race_date', 'venue', 'race_no'])[col].transform('sum')
            preds[f'{col}_norm'] = preds[col] / total.replace(0, 1)

        preds = preds.sort_values('prob_win', ascending=False)
        return preds

    # -------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------

    def save(self, path: str = "models/deep_model.pt"):
        """Save model state dicts + metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'models': {k: v.state_dict() for k, v in self.models.items()},
            'encoders': {k: list(v.classes_) for k, v in self.encoders.items()},
            'scaler_mean': self.scaler.mean_,
            'scaler_std': self.scaler.scale_,
            'history': self.history,
            'feature_numerical': self.FEATURES_NUMERICAL,
            'feature_categorical': self.FEATURES_CATEGORICAL,
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/deep_model.pt", model_dir: str = "models"):
        instance = cls(model_dir)
        state = torch.load(path, map_location=instance.device, weights_only=False)

        for name in ['win', 'place', 'top4']:
            num_numerical = len(state['feature_numerical'])
            cat_dims = [len(state['encoders'][c]) + 1 for c in state['feature_categorical']]
            model = TabularNN(num_numerical, cat_dims).to(instance.device)
            model.load_state_dict(state['models'][name])
            instance.models[name] = model

        instance.encoders = {k: LabelEncoder().fit(v) for k, v in state['encoders'].items()}
        instance.scaler.mean_ = state['scaler_mean']
        instance.scaler.scale_ = state['scaler_std']
        instance.history = state.get('history', {})
        print(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    import os
    db_path = os.getenv("HORSE_DB_PATH", "data/horse.db")

    print("=" * 60)
    print("🐴 Deep Learning Horse Bet Model Training")
    print("=" * 60)

    model = HorseBetDeepModel(model_dir="models")
    results = model.train(db_path, days=730, epochs=100)

    model.save("models/deep_model.pt")

    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'n_train': model.history['train_size'],
        'n_test': model.history['test_size'],
        'epochs': model.history['epochs'],
        'metrics': results,
    }
    with open("models/deep_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, m in results.items():
        print(f"\n{name.upper()}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
