#!/usr/bin/env python3
"""
Scrape all HKJC race results from 2025-01-01 to 2026-04-22.
Resumable - skips already-scraped race days.
Run in background with: python scripts/scrape_historical.py
"""
import sys
import os
import logging
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.hkjc_scraper import HKJCScraper
from src.data.pipeline import DataPipeline
from src.models.race_models import init_db, RaceDay
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/horse-scrape.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scrape_historical")

DB_PATH = os.getenv("HORSE_DB_PATH", "data/horse.db")
engine, SessionFactory = init_db(DB_PATH)


def get_scraped_dates() -> set:
    """Return set of (date_str, venue) already in DB."""
    session: Session = SessionFactory()
    try:
        rows = session.query(RaceDay).all()
        return {(rd.race_date.strftime("%Y/%m/%d"), rd.venue) for rd in rows}
    finally:
        session.close()


def main():
    scraper = HKJCScraper(delay=1.5)
    pipeline = DataPipeline(DB_PATH)

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 4, 22)
    venues = ["ST", "HV"]

    total_days = (end_date - start_date).days + 1
    log.info(f"Date range: {start_date.date()} → {end_date.date()} ({total_days} days)")

    scraped = get_scraped_dates()
    log.info(f"Already scraped: {len(scraped)} race days in DB")

    current = start_date
    saved_total = 0
    skipped_total = 0
    errors = []

    while current <= end_date:
        date_str = current.strftime("%Y/%m/%d")
        day_num = (current - start_date).days + 1

        for venue in venues:
            key = (date_str, venue)

            if key in scraped:
                log.debug(f"  SKIP {date_str} {venue} (already in DB)")
                skipped_total += 1
                continue

            try:
                result = pipeline.scrape_race_day(date_str, venue)
                saved = result.get("saved", 0)
                saved_total += saved
                if saved > 0:
                    log.info(
                        f"[{day_num}/{total_days}] ✓ {date_str} {venue}: "
                        f"{saved} races saved"
                    )
                else:
                    log.info(f"[{day_num}/{total_days}] - {date_str} {venue}: no races (no meeting)")
                scraped.add(key)
            except Exception as e:
                err_msg = f"ERROR {date_str} {venue}: {e}"
                log.error(err_msg)
                errors.append(err_msg)
                # Add to scraped so we don't retry forever
                scraped.add(key)

        current += timedelta(days=1)

        # Progress heartbeat every 10 days
        if day_num % 10 == 0:
            log.info(f"--- Progress: day {day_num}/{total_days} ---")

    log.info("=" * 50)
    log.info(f"SCRAPE COMPLETE")
    log.info(f"  Saved:   {saved_total} race days")
    log.info(f"  Skipped: {skipped_total} (already in DB or no meeting)")
    log.info(f"  Errors:  {len(errors)}")
    if errors:
        for e in errors:
            log.error(f"    {e}")

    # Final count
    session: Session = SessionFactory()
    try:
        from src.models.race_models import Race
        from src.models.race_models import HorseEntry
        race_count = session.query(Race).count()
        entry_count = session.query(HorseEntry).count()
        log.info(f"  Total in DB: {race_count} races, {entry_count} horse entries")
    finally:
        session.close()


if __name__ == "__main__":
    main()
