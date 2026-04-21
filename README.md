# 🐴 horse-bet

HKJC Horse Racing ML Prediction System — scrape historical data, train models, and query predictions via Telegram.

## Features

- **HKJC Scraper** — scrapes race results, odds, horse info from racing.hkjc.com
- **SQLAlchemy DB** — stores races, horses, entries, and results persistently
- **ML Models** — XGBoost classifiers for win/place/top-4 prediction
- **Telegram Bot** — query predictions and race results from anywhere
- **CLI** — scrape, train, predict from the command line

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Scrape a race day
python -m src scrape --race-day 2026/04/19 --venue ST

# Train model (needs historical data)
python -m src train --days 730

# Predict
python -m src predict 2026/04/19 ST

# Run Telegram bot
python -m src.bot.telegram_bot
```

## Environment Variables

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
HORSE_DB_PATH=data/horse.db
MODEL_PATH=models/model.json
```

## Project Structure

```
src/
├── scraper/hkjc_scraper.py   # HKJC web scraper
├── models/race_models.py      # SQLAlchemy models
├── data/pipeline.py           # Scrape → store pipeline
├── ml/models.py              # XGBoost prediction models
├── bot/telegram_bot.py        # Telegram bot
└── main.py                   # CLI entry point
```

## Commands

| Command | Description |
|---------|-------------|
| `scrape --race-day YYYY/MM/DD --venue ST\|HV` | Scrape one race day |
| `scrape --start YYYY/MM/DD --end YYYY/MM/DD` | Scrape date range |
| `train --days N` | Train ML model on N days of history |
| `predict YYYY/MM/DD ST` | Show predictions for a race day |
| `export` | Export all data to CSV |

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/race YYYY/MM/DD ST` | Race card |
| `/predict YYYY/MM/DD ST` | ML predictions |
| `/results YYYY/MM/DD ST` | Official results + dividends |
| `/schedule` | Upcoming race days |
| `/reload` | Reload ML model |

## ML Model

- **Target**: Win probability, place probability, top-4 probability
- **Features**: Horse form (win rate, avg position), jockey/trainer stats, draw, rating, odds, field size, distance change
- **Algorithm**: XGBoost classifier (n_estimators=200, max_depth=6)
- **Training**: Requires 500+ historical races minimum

## Data Schema

- `race_days` — race meeting dates (unique per venue)
- `races` — individual races with distance, class, track
- `horse_entries` — each horse in a race (odds, weight, draw, result)
- `race_results` — official dividends

## Notes

- HKJC may return 503 under load — scraper has automatic retry with backoff
- Rate limit: 1 request/second recommended
- Happy Valley (HV) races may not always be available on the results page
