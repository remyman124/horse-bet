#!/usr/bin/env python3
"""
Batch scrape + predict pipeline for upcoming HKJC race day.
Usage: python scripts/batch_predict.py [--date 2026/04/26] [--venue ST] [--races 1-11]
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import sqlite3
import pickle

# Register date adapter to silence Python 3.12 deprecation warning
sqlite3.register_adapter(date, lambda d: d.isoformat())

from src.scraper import HKJCScraper
from src.ml.models import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_predict")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def load_historical_entries(conn: sqlite3.Connection,
                            exclude_date: str = None) -> pd.DataFrame:
    """Load all historical entries joined with race info for feature engineering."""
    query = """
        SELECT
            re.horse_name,
            re.horse_no,
            re.draw,
            re.rating,
            re.win_odds,
            re.finishing_position,
            re.jockey,
            re.trainer,
            re.jockey_weight,
            r.distance,
            r.race_class,
            r.track,
            r.race_no,
            rd.race_date,
            rd.venue
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
    """
    params = []
    if exclude_date:
        date_parts = exclude_date.split('/')
        py_date = date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
        query += " WHERE rd.race_date < ?"
        params.append(py_date)

    query += " ORDER BY rd.race_date, rd.venue, r.race_no, re.horse_no"

    df = pd.read_sql(query, conn, params=params if params else None)
    df['race_date'] = pd.to_datetime(df['race_date'])
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    log.info(f"Loaded {len(df):,} historical entries" +
             (f" (excluding {exclude_date})" if exclude_date else ""))
    return df


def save_race_card_to_db(conn: sqlite3.Connection, scraper: HKJCScraper,
                          race_date: str, venue: str, race_no: int):
    """Scrape race card and save entries + race to DB. Upserts by date/venue/race_no."""
    log.info(f"  Scraping R{race_no}...")
    card = scraper.get_race_card(race_date, venue, race_no)

    race_info = card.get('race_info', {})
    race_class = race_info.get('race_class')
    distance = race_info.get('distance')
    rating_range = race_info.get('rating')

    # Determine track from URL or default
    track = 'Turf' if venue == 'ST' else 'AWT'
    if race_class and 'AWT' in race_class:
        track = 'AWT'
    elif race_class and 'Dirt' in race_class:
        track = 'Dirt'

    cur = conn.cursor()

    # Upsert race_day
    date_parts = race_date.split('/')
    py_date = date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
    cur.execute("""
        INSERT INTO race_days (race_date, venue, total_races)
        VALUES (?, ?, 0)
        ON CONFLICT(race_date, venue) DO UPDATE SET total_races=excluded.total_races
    """, (py_date, venue))

    race_day_id = cur.execute(
        "SELECT id FROM race_days WHERE race_date=? AND venue=?",
        (py_date, venue)
    ).fetchone()[0]

    # Upsert race
    cur.execute("""
        INSERT INTO races (race_day_id, race_no, distance, race_class, track)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(race_day_id, race_no) DO UPDATE SET
            distance=excluded.distance,
            race_class=excluded.race_class,
            track=excluded.track
    """, (race_day_id, race_no, distance, race_class, track))

    race_id = cur.execute(
        "SELECT id FROM races WHERE race_day_id=? AND race_no=?",
        (race_day_id, race_no)
    ).fetchone()[0]

    # Upsert horse entries
    horses = card.get('horses', [])
    for h in horses:
        horse_no = h.get('horse_no')
        if not horse_no:
            continue
        cur.execute("""
            INSERT INTO horse_entries
            (race_id, horse_no, draw, horse_name, jockey, trainer,
             rating, win_odds, finishing_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(race_id, horse_no) DO UPDATE SET
                draw=excluded.draw,
                horse_name=excluded.horse_name,
                jockey=excluded.jockey,
                trainer=excluded.trainer,
                rating=excluded.rating,
                win_odds=excluded.win_odds
        """, (
            race_id,
            horse_no,
            h.get('draw'),
            h.get('horse_name'),
            h.get('jockey'),
            h.get('trainer'),
            h.get('rating'),
            h.get('win_odds'),
        ))
    conn.commit()
    log.info(f"  R{race_no}: {len(horses)} horses saved (class={race_class}, dist={distance})")
    return card


def load_upcoming_entries(conn: sqlite3.Connection, race_date: str, venue: str) -> pd.DataFrame:
    """Load entries for upcoming races (no finishing_position yet)."""
    date_parts = race_date.split('/')
    py_date = date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
    query = """
        SELECT
            re.horse_name,
            re.horse_no,
            re.draw,
            re.rating,
            re.win_odds,
            CAST(NULL AS INTEGER) AS finishing_position,
            re.jockey,
            re.trainer,
            re.jockey_weight,
            r.distance,
            r.race_class,
            r.track,
            r.race_no,
            rd.race_date,
            rd.venue
        FROM horse_entries re
        JOIN races r ON re.race_id = r.id
        JOIN race_days rd ON r.race_day_id = rd.id
        WHERE rd.race_date = ? AND rd.venue = ?
        ORDER BY r.race_no, re.horse_no
    """
    df = pd.read_sql(query, conn, params=(py_date, venue))
    df['race_date'] = pd.to_datetime(df['race_date'])
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    log.info(f"Loaded {len(df)} upcoming entries for {race_date} {venue}")
    log.info(f"  race_class sample: {df['race_class'].unique()[:3].tolist()}")
    log.info(f"  jockey sample: {df['jockey'].unique()[:3].tolist()}")
    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_race_day(model, historical_df: pd.DataFrame,
                      upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features using full history (including upcoming rows to get
    rolling/historical features computed correctly), then run XGBoost
    predictions only for the upcoming race day entries.
    """
    upcoming_df = upcoming_df.copy()
    upcoming_df['finishing_position'] = pd.NA  # nullable Int64

    # Combine: upcoming rows have NA finishing_position
    combined = pd.concat([historical_df, upcoming_df], ignore_index=True)

    # Engineer features in-place (shift-based features only use past data)
    combined = engineer_features(combined)

    # Extract upcoming rows BEFORE prepare_data (which fills NAs and drops rows)
    upcoming_date = upcoming_df['race_date'].iloc[0]
    upcoming_mask = combined['race_date'] == upcoming_date
    upcoming_features = combined[upcoming_mask].copy()

    if len(upcoming_features) == 0:
        log.error("No upcoming features extracted — check date filter")
        return pd.DataFrame()

    # Apply the race_class mapping before encoding
    race_class_map = {
        'Group-One': '', 'Group-Two': '', 'Group-Three': '',
        'Listed': '', 'Pattern': '',
    }
    if 'race_class' in upcoming_features.columns:
        upcoming_features['race_class'] = (
            upcoming_features['race_class'].fillna('').replace(race_class_map)
        )

    # Debug: check columns before encoding
    log.info(f"  upcoming_features cols: {list(upcoming_features.columns)}")
    log.info(f"  model.encoders keys: {list(model.encoders.keys())}")
    log.info(f"  race_class before encode: {upcoming_features['race_class'].unique().tolist()}")

    # Encode categoricals manually using label→index mapping
    # (avoids sklearn LabelEncoder KeyError on unknown/empty strings)
    # Keep raw string columns for model.predict() output; add _enc columns for features
    for col, enc_col in [('venue', 'venue_enc'), ('race_class', 'race_class_enc'),
                          ('track', 'track_enc'), ('jockey', 'jockey_enc'),
                          ('trainer', 'trainer_enc')]:
        if col in upcoming_features.columns:
            enc = model.encoders[col]
            fb  = '' if col != 'track' else 'Turf'
            vals = upcoming_features[col].astype(str).replace({'nan': fb, '': fb, 'None': fb}).fillna(fb)
            # Map each value to its integer index using the encoder's classes_
            classes = enc.classes_
            mapping = {v: i for i, v in enumerate(classes)}
            upcoming_features[enc_col] = vals.map(lambda x: mapping.get(x, mapping.get(fb, 0)))
            log.info(f"  {enc_col} dtype={upcoming_features[enc_col].dtype}, unique={upcoming_features[enc_col].nunique()}")
        else:
            log.info(f"  WARNING: {col} NOT in upcoming_features columns!")

    missing = [c for c in model.FEATURE_COLS if c not in upcoming_features.columns]
    if missing:
        log.error(f"  Missing columns: {missing}")
    else:
        log.info(f"  All FEATURE_COLS present ✓")
    preds = model.predict(upcoming_features)

    # Rename the normalized columns to prob_win/place/top4
    preds = preds.rename(columns={
        'prob_win_norm': 'prob_win',
        'prob_place_norm': 'prob_place',
        'prob_top4_norm': 'prob_top4',
    })
    return preds


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_output(preds: pd.DataFrame, race_date: str, venue: str) -> str:
    """Format predictions as readable text output."""
    lines = []
    header = f"🏇 {race_date.replace('/','-')} {venue} — ML Predictions (XGBoost 0.72 AUC)"
    lines.append(header)
    lines.append("=" * 60)

    for (rd, v, rn), group in preds.groupby(['race_date', 'venue', 'race_no']):
        # Race header
        first = group.iloc[0]
        race_label = f"Race {rn}"
        race_class = first.get('race_class', '?') or '?'
        distance_val = first.get('distance', '?')
        if distance_val not in (None, '?', ''):
            try:
                distance_str = f"{int(float(distance_val))}"
            except (ValueError, TypeError):
                distance_str = str(distance_val)
        else:
            distance_str = '?'
        lines.append(f"  {race_label}  |  {race_class}  |  {distance_str}M")

        # Column headers
        lines.append(f"  {'#':<3} {'Horse':<20} {'Draw':<5} {'Rt':<4} {'Odds':<6} {'Win%':<6} {'Place%':<7}")
        lines.append(f"  {'-'*3} {'-'*20} {'-'*5} {'-'*4} {'-'*6} {'-'*6} {'-'*7}")

        for _, row in group.iterrows():
            prob_win = row.get('prob_win')
            prob_place = row.get('prob_place')
            win_pct = f"{float(prob_win)*100:.1f}%" if prob_win is not None and (isinstance(prob_win, (int, float)) or (hasattr(prob_win, '__float__'))) else "N/A"
            place_pct = f"{float(prob_place)*100:.1f}%" if prob_place is not None and (isinstance(prob_place, (int, float)) or (hasattr(prob_place, '__float__'))) else "N/A"
            odds = f"{float(row.get('win_odds', 0)):.1f}" if row.get('win_odds') not in (None, 0) else "N/A"
            rating = str(int(row['rating'])) if pd.notna(row.get('rating')) and row.get('rating', 0) > 0 else "-"
            draw = str(int(row['draw'])) if pd.notna(row.get('draw')) and row.get('draw', 0) > 0 else "-"

            name = (row.get('horse_name') or 'Unknown')[:20]
            lines.append(
                f"  {int(row['horse_no']):<3} {name:<20} {draw:<5} {rating:<4} {odds:<6} {win_pct:<6} {place_pct:<7}"
            )

    return "\n".join(lines)


def save_json(preds: pd.DataFrame, path: Path):
    """Save predictions to JSON for programmatic use."""
    records = preds.to_dict(orient='records')
    # Convert non-serializable types
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif isinstance(v, (pd.Timestamp, date)):
                r[k] = str(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'model': 'XGBoost (Win AUC 0.72)',
            'predictions': records,
        }, f, ensure_ascii=False, indent=2)
    log.info(f"JSON saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch scrape + predict HKJC races")
    parser.add_argument("--date", default="2026/04/26")
    parser.add_argument("--venue", default="ST")
    parser.add_argument("--races", default="1-11", help="Race range, e.g. 1-11 or 1,3,5")
    parser.add_argument("--db", default="data/horse.db")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--output", default="output/predictions.json")
    parser.add_argument("--no-scrape", action='store_true', help="Skip scraping, use existing DB data")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    db_path = project_root / args.db
    model_path = project_root / args.model
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)

    # Parse race range
    race_nos = []
    if '-' in args.races:
        start, end = args.races.split('-')
        race_nos = list(range(int(start), int(end) + 1))
    else:
        race_nos = [int(x) for x in args.races.split(',')]

    log.info(f"Target: {args.date} {args.venue} races {race_nos}")

    # Load model
    log.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    log.info("Model loaded ✓")

    # DB connection
    conn = sqlite3.connect(str(db_path))

    # Scrape missing races
    if not args.no_scrape:
        scraper = HKJCScraper(delay=0.5)
        for rn in race_nos:
            try:
                save_race_card_to_db(conn, scraper, args.date, args.venue, rn)
            except Exception as e:
                log.error(f"Failed to scrape R{rn}: {e}")
            time.sleep(0.5)

    # Load historical + upcoming data
    hist_df = load_historical_entries(conn, exclude_date=args.date)
    upcoming_df = load_upcoming_entries(conn, args.date, args.venue)

    if upcoming_df.empty:
        log.error("No upcoming entries found in DB. Run without --no-scrape to fetch data.")
        sys.exit(1)

    # Predict
    log.info("Running predictions...")
    preds = predict_race_day(model, hist_df, upcoming_df)

    # Output
    text_out = format_output(preds, args.date, args.venue)
    save_json(preds, output_path)

    print("\n" + text_out)
    print(f"\n✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
