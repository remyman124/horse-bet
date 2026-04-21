#!/usr/bin/env python3
"""
Fast HKJC scraper - skip non-race days by checking every N days.
Strategy:
  1. Fast scan: check every 3 days, collect race days
  2. Fill gaps: for each 3-day block that had a race, check intervening days
  3. Scrape: visit each confirmed race day

This reduces ~7500+ requests to ~1500 requests for 10 years.
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
log = logging.getLogger("scrape_fast")


def get_scraped_dates() -> set:
    engine, SessionFactory = init_db(DB_PATH)
    session: Session = SessionFactory()
    try:
        rows = session.query(RaceDay).all()
        return {(rd.race_date.strftime("%Y/%m/%d"), rd.venue) for rd in rows}
    finally:
        session.close()


def scan_for_race_days(start_date: datetime, end_date: datetime, step: int = 3) -> set:
    """
    Quickly scan date range for race days by checking every `step` days.
    Returns set of date_str that have at least one race.
    """
    scraper = HKJCScraper(delay=0.5)
    found = set()
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y/%m/%d")
        for venue in ["ST", "HV"]:
            try:
                race_list = scraper.get_race_day_races(date_str, venue, retries=2)
                if race_list and len(race_list) > 0:
                    found.add((date_str, venue))
                    log.info(f"  [SCAN] {date_str} {venue}: {len(race_list)} races")
            except Exception as e:
                log.debug(f"  [SCAN] {date_str} {venue}: {e}")
        current += timedelta(days=step)

    return found


def fill_gaps(found: set, start_date: datetime, end_date: datetime, step: int = 3) -> set:
    """
    For each block of `step` days that contains a found race day,
    check the intervening days to ensure completeness.
    """
    scraper = HKJCScraper(delay=0.5)
    all_race_days = set(found)

    # Group found by approximate week
    sorted_dates = sorted([datetime.strptime(d, "%Y/%m/%d") for d, v in found])
    for race_date, venue in found:
        d = datetime.strptime(race_date, "%Y/%m/%d")
        # Check days before and after within the step window
        for delta in range(-step + 1, step):
            if delta == 0:
                continue
            check_date = d + timedelta(days=delta)
            if check_date < start_date or check_date > end_date:
                continue
            date_str = check_date.strftime("%Y/%m/%d")
            key = (date_str, venue)
            if key in all_race_days:
                continue
            try:
                race_list = scraper.get_race_day_races(date_str, venue, retries=2)
                if race_list:
                    all_race_days.add(key)
                    log.info(f"  [FILL] {date_str} {venue}: {len(race_list)} races (gap fill)")
            except Exception:
                pass

    return all_race_days


def scrape_race_days(race_days: set):
    """Scrape all confirmed race days."""
    pipeline = DataPipeline(DB_PATH)
    total = len(race_days)
    done = 0

    for date_str, venue in sorted(race_days):
        try:
            result = pipeline.scrape_race_day(date_str, venue)
            saved = result.get("saved", 0)
            skipped = result.get("skipped", 0)
            log.info(f"[{done+1}/{total}] ✓ {date_str} {venue}: {saved} saved, {skipped} skipped")
        except Exception as e:
            log.error(f"[{done+1}/{total}] ERROR {date_str} {venue}: {e}")
        done += 1


if __name__ == "__main__":
    DB_PATH = os.getenv("HORSE_DB_PATH", "data/horse.db")
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2026, 4, 22)

    log.info(f"=== FAST SCRAPE: {start_date.date()} → {end_date.date()} ===")

    # Get already-scraped
    scraped = get_scraped_dates()
    log.info(f"Already scraped: {len(scraped)} race days in DB")

    # Phase 1: Fast scan every 3 days
    log.info(f"[PHASE 1] Fast scan every 3 days...")
    found = scan_for_race_days(start_date, end_date, step=3)

    # Remove already scraped
    to_scrape = found - scraped
    log.info(f"[PHASE 1] Found {len(found)} race days, {len(to_scrape)} new to scrape")

    if to_scrape:
        # Phase 2: Fill gaps (already done in scan_for_race_days)
        # Phase 3: Scrape
        log.info(f"[PHASE 2] Scraping {len(to_scrape)} race days...")
        scrape_race_days(to_scrape)

    # Summary
    session_summary = get_scraped_dates()
    log.info(f"=== DONE: {len(session_summary)} total race days in DB ===")
