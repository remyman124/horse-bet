"""
Data pipeline - orchestrates scraping, storing, and processing.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ..scraper.hkjc_scraper import HKJCScraper
from ..models.race_models import init_db, RaceDay, Race, HorseEntry, RaceResult, Base
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates the full data lifecycle: scrape → store → process."""

    def __init__(self, db_path: str, data_dir: str = "data"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.engine, self.Session = init_db(db_path)
        self.scraper = HKJCScraper(delay=1.5)

    def scrape_race_day(self, race_date: str, venue: str = "ST", skip_existing: bool = True) -> dict:
        """
        Scrape a full race day and store in DB.
        Iterates each race individually with retry logic.
        """
        session: Session = self.Session()
        try:
            log.info(f"Fetching race list for {race_date} {venue}...")
            race_list = self.scraper.get_race_day_races(race_date, venue)
            log.info(f"Found race numbers: {[r['race_no'] for r in race_list]}")

            # Parse date
            date_obj = datetime.strptime(race_date, "%Y/%m/%d").date()

            # Upsert race day
            rd = session.query(RaceDay).filter_by(race_date=date_obj, venue=venue).first()
            if not rd:
                rd = RaceDay(race_date=date_obj, venue=venue)
                session.add(rd)
                session.flush()
            rd.total_races = len(race_list)

            saved = 0
            skipped = 0

            for race_info in race_list:
                race_no = race_info['race_no']

                # Check if already scraped (optional skip)
                if skip_existing:
                    existing = session.query(Race).filter_by(
                        race_day_id=rd.id, race_no=race_no
                    ).first()
                    if existing and existing.race_time:
                        log.debug(f"Race {race_no} already has results, skipping")
                        skipped += 1
                        continue

                try:
                    race_data = self.scraper.get_race_result(race_date, venue, race_no)
                    self._upsert_race(session, rd.id, race_no, race_data)
                    saved += 1
                    log.info(f"  ✓ Race {race_no}: {len(race_data.get('horses', []))} horses, "
                             f"dividends={bool(race_data.get('dividends'))}")
                except Exception as e:
                    log.warning(f"  ✗ Race {race_no}: {e}")
                    skipped += 1

            session.commit()
            log.info(f"Saved {race_date} {venue}: {saved} races saved, {skipped} skipped")
            return {'saved': saved, 'skipped': skipped, 'races': race_list}

        except Exception as e:
            session.rollback()
            log.error(f"Failed to scrape race day: {e}")
            raise
        finally:
            session.close()

    def scrape_range(self, start_date: str, end_date: str, venues=None):
        """Scrape a range of dates."""
        venues = venues or ["ST", "HV"]
        start = datetime.strptime(start_date, "%Y/%m/%d")
        end = datetime.strptime(end_date, "%Y/%m/%d")
        current = start
        while current <= end:
            date_str = current.strftime("%Y/%m/%d")
            for venue in venues:
                try:
                    self.scrape_race_day(date_str, venue)
                except Exception as e:
                    log.warning(f"Failed {date_str} {venue}: {e}")
            current += timedelta(days=1)

    def _upsert_race(self, session: Session, race_day_id: int, race_no: int, data: dict):
        """Upsert a single race and its horse entries."""
        race = session.query(Race).filter_by(
            race_day_id=race_day_id, race_no=race_no
        ).first()
        if not race:
            race = Race(race_day_id=race_day_id, race_no=race_no)
            session.add(race)

        # Update race info
        info = data.get('race_info', {})
        race.distance = self._parse_distance(info.get('distance', ''))
        race.race_class = info.get('race_class', '')
        race.track = info.get('track', '')
        race.prize_money = info.get('prize', '')
        # race_time not in result, it's start time

        session.flush()

        # Upsert horse entries
        for entry_data in data.get('horses', []):
            horse_no = entry_data.get('horse_no', '')
            if not horse_no:
                continue

            entry = session.query(HorseEntry).filter_by(
                race_id=race.id, horse_no=int(horse_no)
            ).first()
            if not entry:
                entry = HorseEntry(race_id=race.id, horse_no=int(horse_no))
                session.add(entry)

            entry.horse_name = entry_data.get('horse_name', '')
            entry.draw = entry_data.get('draw')
            entry.jockey = entry_data.get('jockey', '')
            entry.trainer = entry_data.get('trainer', '')
            entry.jockey_weight = entry_data.get('actual_weight', '')
            entry.rating = self._parse_int(entry_data.get('rating', ''))
            entry.win_odds = self._parse_float(entry_data.get('win_odds', ''))
            entry.finishing_position = entry_data.get('position')
            entry.finishing_time = entry_data.get('finish_time', '')
            entry.margin = entry_data.get('lbw', '')

            if 'finish_type' in entry_data:
                entry.finish_type = entry_data['finish_type']

        # Upsert dividends
        if data.get('dividends'):
            result = session.query(RaceResult).filter_by(race_id=race.id).first()
            if not result:
                result = RaceResult(race_id=race.id)
                session.add(result)
            divs = data['dividends']
            result.win_dividend = divs.get('win', '')
            result.place_dividend_1 = divs.get('place', '')

        session.flush()
        return race

    def _parse_distance(self, val: str) -> Optional[int]:
        if not val:
            return None
        import re
        m = re.search(r'(\d+)', str(val))
        return int(m.group(1)) if m else None

    def _parse_int(self, val) -> Optional[int]:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _parse_float(self, val) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def get_recent_races(self, days: int = 30) -> pd.DataFrame:
        """Load recent race data as DataFrame for ML."""
        session: Session = self.Session()
        try:
            cutoff = datetime.now() - timedelta(days=days)
            query = session.query(Race, RaceDay, HorseEntry).join(
                RaceDay, Race.race_day_id == RaceDay.id
            ).join(
                HorseEntry, HorseEntry.race_id == Race.id
            ).filter(
                RaceDay.race_date >= cutoff.date()
            ).filter(
                HorseEntry.finishing_position.isnot(None)
            )
            rows = []
            for race, day, entry in query:
                rows.append({
                    'race_date': day.race_date,
                    'venue': day.venue,
                    'race_no': race.race_no,
                    'distance': race.distance,
                    'race_class': race.race_class,
                    'track': race.track,
                    'horse_no': entry.horse_no,
                    'horse_name': entry.horse_name,
                    'draw': entry.draw,
                    'jockey': entry.jockey,
                    'trainer': entry.trainer,
                    'jockey_weight': entry.jockey_weight,
                    'rating': entry.rating,
                    'win_odds': entry.win_odds,
                    'finishing_position': entry.finishing_position,
                    'finishing_time': entry.finishing_time,
                    'margin': entry.margin,
                    'finish_type': entry.finish_type,
                })
            return pd.DataFrame(rows)
        finally:
            session.close()

    def get_race_day(self, race_date: str, venue: str) -> pd.DataFrame:
        """Get all entries for a specific race day (includes upcoming races without results)."""
        session: Session = self.Session()
        try:
            date_obj = datetime.strptime(race_date, "%Y/%m/%d").date()
            query = session.query(Race, RaceDay, HorseEntry).join(
                RaceDay, Race.race_day_id == RaceDay.id
            ).join(
                HorseEntry, HorseEntry.race_id == Race.id
            ).filter(
                RaceDay.race_date == date_obj,
                RaceDay.venue == venue.upper()
            )
            rows = []
            for race, day, entry in query:
                rows.append({
                    'race_date': day.race_date,
                    'venue': day.venue,
                    'race_no': race.race_no,
                    'distance': race.distance,
                    'race_class': race.race_class,
                    'track': race.track,
                    'horse_no': entry.horse_no,
                    'horse_name': entry.horse_name,
                    'draw': entry.draw,
                    'jockey': entry.jockey,
                    'trainer': entry.trainer,
                    'rating': entry.rating,
                    'win_odds': entry.win_odds,
                    'finishing_position': entry.finishing_position,
                })
            return pd.DataFrame(rows)
        finally:
            session.close()

    def save_raw_json(self, race_date: str, venue: str, data: dict):
        """Save raw scraped data as JSON for debugging."""
        out = self.data_dir / 'raw' / f"{race_date.replace('/', '')}_{venue}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"Saved raw data to {out}")

    def export_to_csv(self, output_dir: str = "data/processed"):
        """Export all data to CSV for analysis."""
        df = self.get_recent_races(days=3650)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / 'races.csv', index=False)
        log.info(f"Exported {len(df)} rows to {out_dir}/races.csv")
        return df
