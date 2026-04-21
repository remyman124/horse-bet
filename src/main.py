#!/usr/bin/env python3
"""
Horse Bet CLI - Main entry point for horse-bet.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .scraper.hkjc_scraper import HKJCScraper
from .data.pipeline import DataPipeline
from .ml.models import HorseBetModel

console = Console()
log = logging.getLogger(__name__)


def cmd_scrape(args):
    """Scrape and store race data."""
    db = os.getenv("HORSE_DB_PATH", "data/horse.db")
    Path(db).parent.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(db)

    if args.race_day:
        race_date, venue = args.race_day, args.venue.upper()
        console.print(f"[cyan]Scraping {race_date} {venue}...[/cyan]")
        result = pipeline.scrape_race_day(race_date, venue)
        console.print(f"  ✓ {result['saved']} races saved, {result['skipped']} skipped")
        console.print(f"[green]✓ Done[/green]")
        return

    # Scrape date range
    if args.start and args.end:
        from datetime import datetime, timedelta
        start = datetime.strptime(args.start, "%Y/%m/%d")
        end = datetime.strptime(args.end, "%Y/%m/%d")
        current = start
        while current <= end:
            date_str = current.strftime("%Y/%m/%d")
            for venue in ["ST", "HV"]:
                try:
                    pipeline.scrape_race_day(date_str, venue)
                    console.print(f"  ✓ {date_str} {venue}")
                except Exception as e:
                    console.print(f"  ✗ {date_str} {venue}: {e}")
            current += timedelta(days=1)


def cmd_train(args):
    """Train ML model."""
    db = os.getenv("HORSE_DB_PATH", "data/horse.db")
    model_path = os.getenv("MODEL_PATH", "models/model.json")

    if not os.path.exists(db):
        console.print("[red]No database found. Scrape data first.[/red]")
        return 1

    pipeline = DataPipeline(db)
    console.print("[cyan]Loading historical data...[/cyan]")
    df = pipeline.get_recent_races(days=args.days)

    if df.empty:
        console.print("[red]No data found in database.[/red]")
        return 1

    console.print(f"  Loaded {len(df)} entries from {df['race_date'].min()} to {df['race_date'].max()}")

    # Only train on races with results
    df_finished = df.dropna(subset=['finishing_position'])
    console.print(f"  {len(df_finished)} entries with results")

    model = HorseBetModel(model_dir=os.path.dirname(model_path))
    console.print("[cyan]Training models...[/cyan]")
    results = model.train(df_finished)

    console.print("\n[bold]Results:[/bold]")
    for target, metrics in results.items():
        console.print(f"\n  {target.upper()} model:")
        console.print(f"    Brier Score: {metrics['brier_score']:.4f}")
        console.print(f"    Log Loss:    {metrics['log_loss']:.4f}")
        console.print(f"    ROC AUC:     {metrics['roc_auc']:.4f}")
        console.print(f"    Accuracy:    {metrics['accuracy']:.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    console.print(f"\n[green]✓ Model saved to {model_path}[/green]")


def cmd_predict(args):
    """Run predictions for a race day."""
    db = os.getenv("HORSE_DB_PATH", "data/horse.db")
    model_path = os.getenv("MODEL_PATH", "models/model.json")

    if not os.path.exists(model_path):
        console.print("[red]Model not found. Train first: horsebet train[/red]")
        return 1

    pipeline = DataPipeline(db)
    model = HorseBetModel()
    model.load(model_path)

    df = pipeline.get_recent_races(days=90)
    df_race = df[
        (df['race_date'].astype(str) == args.date.replace('/', '-')) &
        (df['venue'] == args.venue.upper())
    ]

    if df_race.empty:
        console.print(f"[yellow]No data for {args.date} {args.venue}.[/yellow]")
        return 1

    preds = model.predict(df_race)

    console.print(f"\n[bold]🎯 Predictions for {args.date} {args.venue.upper()}[/bold]\n")

    from rich.table import Table
    table = Table()
    table.add_column("Pos", style="cyan")
    table.add_column("Horse", style="white")
    table.add_column("Draw", style="yellow")
    table.add_column("Odds", style="magenta")
    table.add_column("P(Win)", style="green")
    table.add_column("P(Place)", style="green")

    for _, row in preds.head(8).iterrows():
        odds = f"{row['win_odds']:.1f}" if row['win_odds'] else "N/A"
        pwin = f"{row['prob_win_norm']*100:.1f}%"
        pplace = f"{row['prob_place_norm']*100:.1f}%"
        table.add_row(
            str(_ + 1),
            row['horse_name'] or "Unknown",
            str(row['draw']) if row['draw'] else "-",
            odds,
            pwin,
            pplace,
        )

    console.print(table)


def cmd_export(args):
    """Export data to CSV."""
    db = os.getenv("HORSE_DB_PATH", "data/horse.db")
    pipeline = DataPipeline(db)
    df = pipeline.export_to_csv(args.output or "data/processed")
    console.print(f"[green]Exported {len(df)} rows[/green]")


def main():
    parser = argparse.ArgumentParser(prog="horsebet", description="🐴 Horse Bet — HKJC ML predictions")
    sub = parser.add_subparsers()

    p_scrape = sub.add_parser("scrape", help="Scrape HKJC data")
    p_scrape.add_argument("--race-day", help="Date YYYY/MM/DD")
    p_scrape.add_argument("--venue", default="ST", help="Venue (ST/HV)")
    p_scrape.add_argument("--start", help="Start date YYYY/MM/DD")
    p_scrape.add_argument("--end", help="End date YYYY/MM/DD")
    p_scrape.set_defaults(func=cmd_scrape)

    p_train = sub.add_parser("train", help="Train ML model")
    p_train.add_argument("--days", type=int, default=730, help="Days of history to use")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Predict a race day")
    p_predict.add_argument("date", help="Date YYYY/MM/DD")
    p_predict.add_argument("venue", help="Venue (ST/HV)")
    p_predict.set_defaults(func=cmd_predict)

    p_export = sub.add_parser("export", help="Export to CSV")
    p_export.add_argument("--output", help="Output directory")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        sys.exit(args.func(args) or 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
