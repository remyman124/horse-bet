"""
HKJC Horse Racing Scraper v2
Scrapes race results from racing.hkjc.com
"""
import re
import time
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


class HKJCScraper:
    """Scraper for HKJC horse racing data."""

    BASE_URL = "https://racing.hkjc.com"
    EN_URL = f"{BASE_URL}/en-us/local/information"

    def __init__(self, delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en,zh-HK;q=0.9',
        })
        self.delay = delay
        self._last_request = 0

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_race_day_races(self, race_date: str, venue: str = "ST", retries: int = 3) -> list:
        """
        Get list of race numbers for a race day.
        Returns list of dicts with race_no, href.
        """
        url = f"{self.EN_URL}/localresults?racedate={race_date}&Racecourse={venue}"
        last_err = None
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 503:
                    wait = (attempt + 1) * 2
                    log.warning(f"503 on race day listing (attempt {attempt+1}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except Exception as e:
                last_err = e
                wait = (attempt + 1) * 2
                log.warning(f"Attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                time.sleep(wait)

        soup = BeautifulSoup(resp.text, 'lxml')
        races = []

        for a in soup.find_all('a', href=lambda h: h and 'RaceNo=' in h and 'localresults' in h):
            href = a.get('href', '')
            m = re.search(r'RaceNo=(\d+)', href)
            if m:
                race_no = int(m.group(1))
                races.append({'race_no': race_no, 'href': href})

        # Deduplicate by race number, keep first (some have S2 etc)
        seen = set()
        unique = []
        for r in races:
            if r['race_no'] not in seen:
                seen.add(r['race_no'])
                unique.append(r)

        unique = sorted(unique, key=lambda x: x['race_no'])

        # HKJC always has Race 1 - if not in list, add it (it may be the current race)
        if not any(r['race_no'] == 1 for r in unique):
            unique.insert(0, {'race_no': 1, 'href': f'?racedate={race_date}&Racecourse={venue}&RaceNo=1'})

        return unique

    def get_race_result(self, race_date: str, venue: str, race_no: int, retries: int = 5) -> dict:
        """
        Scrape detailed result for a single race.
        Returns dict with race_info, horses, dividends.
        """
        # Normalize date format: 2026/04/19 or 2026-04-19 -> 2026/04/19
        race_date = race_date.replace('-', '/')
        url = f"{self.EN_URL}/localresults?racedate={race_date}&Racecourse={venue}&RaceNo={race_no}"

        last_err = None
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 503:
                    wait = (attempt + 1) * 3
                    log.warning(f"503 for race {race_no} (attempt {attempt+1}/{retries}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except Exception as e:
                last_err = e
                wait = (attempt + 1) * 3
                log.warning(f"Attempt {attempt+1} failed for race {race_no}: {e}")
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    log.error(f"All retries exhausted for race {race_no}: {e}")
                    raise

        return self._parse_race_result(resp.text, race_date, venue, race_no)

    def get_race_card(self, race_date: str, venue: str, race_no: int) -> dict:
        """Get race card (pre-race info) for a race."""
        race_date = race_date.replace('-', '/')
        url = f"{self.EN_URL}/racecard?racedate={race_date}&Racecourse={venue}&RaceNo={race_no}"
        self._rate_limit()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return self._parse_race_card(resp.text, race_date, venue, race_no)

    def get_upcoming_meetings(self) -> list:
        """Get upcoming race meetings."""
        url = f"{self.EN_URL}/meetinglist"
        self._rate_limit()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return self._parse_meeting_list(resp.text)

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def _parse_race_result(self, html: str, race_date: str, venue: str, race_no: int) -> dict:
        """Parse race result page."""
        soup = BeautifulSoup(html, 'lxml')
        result = {
            'race_date': race_date,
            'venue': venue,
            'race_no': race_no,
            'race_info': {},
            'horses': [],
            'finishing_order': [],
            'dividends': {},
        }

        text = soup.get_text()

        # ---- Race info from page text ----
        # e.g. "Class 5 - 1800M - (40-0)" and "Race Meeting: 19/04/2026  Sha Tin"
        info = {}

        # Extract class, distance, rating from first major text block
        class_match = re.search(r'Class\s+(\d+)\s*-\s*(\d+)[Mm]\s*-\s*\(([\d-]+)\)', text)
        if class_match:
            info['race_class'] = f"Class {class_match.group(1)}"
            info['distance'] = class_match.group(2)
            info['rating'] = class_match.group(3)

        # Extract prize race name
        prize_match = re.search(r'(Race Meeting:[^\n]+)', text)
        if prize_match:
            info['meeting'] = prize_match.group(1).strip()

        # Track type - usually in class name or separate
        if 'AWT' in text:
            info['track'] = 'AWT'
        elif 'Dirt' in text:
            info['track'] = 'Dirt'
        else:
            info['track'] = 'Turf'

        result['race_info'] = info

        # ---- Horse table (Table 2 in our tests) ----
        tables = soup.find_all('table')
        horse_table = None
        dividend_table = None

        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
            header_text = ' '.join(t.get_text(strip=True) for t in rows[0].find_all(['td', 'th']))
            # Horse results table has: Pla. Horse No. Horse Jockey Trainer (no 'Incident')
            # Incident table has: Pla. Horse No. Horse Incident
            if 'Horse No.' in header_text and 'Pla.' in header_text and 'Jockey' in header_text:
                horse_table = table
            elif 'Dividend' in header_text or (len(rows) > 1 and 'WIN' in rows[1].get_text()):
                dividend_table = table

        if horse_table:
            result['horses'] = self._parse_horse_table(horse_table)

        # ---- Dividends ----
        if dividend_table:
            result['dividends'] = self._parse_dividend_table(dividend_table)

        # ---- Finishing order ----
        result['finishing_order'] = [
            h for h in result['horses'] if h.get('position')
        ]

        return result

    def _parse_horse_table(self, table) -> list:
        """Parse the horse results table."""
        horses = []
        rows = table.find_all('tr')

        if len(rows) < 2:
            return horses

        # Parse header to find column indices
        headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
        # Headers: Pla. Horse No. Horse Jockey Trainer Act. Wt. Declar. Horse Wt. Dr. LBW RunningPosition Finish Time W

        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 5:
                continue

            cell_text = [c.get_text(strip=True) for c in cells]
            raw_html = [str(c) for c in cells]

            horse = self._extract_horse_from_cells(headers, cell_text, raw_html)
            if horse:
                horses.append(horse)

        return horses

    def _extract_horse_from_cells(self, headers, cell_text, raw_html) -> Optional[dict]:
        """Extract horse dict from table row cells."""
        try:
            # Build header -> index map
            hmap = {h: i for i, h in enumerate(headers)}

            horse = {}

            # Position (Pla.)
            if 'Pla.' in hmap:
                v = cell_text[hmap['Pla.']].strip()
                if v.isdigit():
                    horse['position'] = int(v)
                elif v in ['DH', 'WV', 'DSQ', 'TEMP', 'PU', 'UR', 'FE', 'AB']:
                    horse['finish_type'] = v  # dead heat, withdrawn, disqualified etc

            # Horse No.
            if 'Horse No.' in hmap:
                v = cell_text[hmap['Horse No.']].strip()
                if v.isdigit():
                    horse['horse_no'] = int(v)

            # Horse name - may be in <a> tag
            if 'Horse' in hmap:
                idx = hmap['Horse']
                # Look for link in cell
                cell_html = raw_html[idx]
                a_match = re.search(r'>([^<]+)</a>', cell_html)
                if a_match:
                    horse['horse_name'] = a_match.group(1).strip()
                else:
                    horse['horse_name'] = cell_text[idx].strip()

            # Jockey
            if 'Jockey' in hmap:
                horse['jockey'] = cell_text[hmap['Jockey']].strip()

            # Trainer
            if 'Trainer' in hmap:
                horse['trainer'] = cell_text[hmap['Trainer']].strip()

            # Actual Weight
            if 'Act. Wt.' in hmap:
                horse['actual_weight'] = cell_text[hmap['Act. Wt.']].strip()

            # Declared Horse Weight
            if 'Declar. Horse Wt.' in hmap:
                horse['declared_weight'] = cell_text[hmap['Declar. Horse Wt.']].strip()

            # Draw (Barrier)
            if 'Dr.' in hmap:
                v = cell_text[hmap['Dr.']].strip()
                if v.isdigit():
                    horse['draw'] = int(v)

            # LBW (Last Behind Winner)
            if 'LBW' in hmap:
                horse['lbw'] = cell_text[hmap['LBW']].strip()

            # Running position (comma-separated positions at each 400m call)
            if 'RunningPosition' in hmap:
                horse['running_position'] = cell_text[hmap['RunningPosition']].strip()

            # Finish time
            if 'Finish Time' in hmap:
                horse['finish_time'] = cell_text[hmap['Finish Time']].strip()

            # Win odds - column with 'Win Odds' header
            for hdr, i in hmap.items():
                if 'win odds' in hdr.lower():
                    v = cell_text[i].strip()
                    if re.match(r'^\d+\.\d+$', v):
                        horse['win_odds'] = float(v)
                    break

            if horse.get('horse_no'):
                return horse

        except Exception as e:
            log.debug(f"Failed to parse horse row: {e}")

        return None

    def _parse_dividend_table(self, table) -> dict:
        """Parse dividend (payout) table."""
        divs = {}
        rows = table.find_all('tr')
        for row in rows[1:]:  # skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            key = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True)
            key_lower = key.lower()
            if 'win' in key_lower:
                divs['win'] = val
            elif 'place' in key_lower:
                divs['place'] = val
            elif 'quinella' in key_lower:
                divs['quinella'] = val
            elif 'quintuple' in key_lower:
                divs['quintuple'] = val
            elif 'quartet' in key_lower or '4th' in key_lower:
                divs['quartet'] = val
            else:
                divs[key] = val
        return divs

    def _parse_race_card(self, html: str, race_date: str, venue: str, race_no: int) -> dict:
        """Parse race card (pre-race) page."""
        soup = BeautifulSoup(html, 'lxml')
        result = {
            'race_date': race_date,
            'venue': venue,
            'race_no': race_no,
            'race_info': {},
            'horses': [],
        }

        text = soup.get_text()

        # Extract race info
        class_match = re.search(r'Class\s+(\d+)\s*-\s*(\d+)[Mm]\s*-\s*\(([\d-]+)\)', text)
        if class_match:
            result['race_info'] = {
                'race_class': f"Class {class_match.group(1)}",
                'distance': class_match.group(2),
                'rating': class_match.group(3),
            }

        # Find horse table
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
            header_text = ' '.join(t.get_text(strip=True) for t in rows[0].find_all(['td', 'th']))
            if 'Horse No.' in header_text and 'Draw' in header_text:
                result['horses'] = self._parse_race_card_table(table)
                break

        return result

    def _parse_race_card_table(self, table) -> list:
        """Parse race card horse table (pre-race)."""
        horses = []
        rows = table.find_all('tr')

        if len(rows) < 2:
            return horses

        headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 5:
                continue

            cell_text = [c.get_text(strip=True) for c in cells]
            raw_html = [str(c) for c in cells]
            hmap = {h: i for i, h in enumerate(headers)}

            horse = {}

            if 'No.' in hmap:
                v = cell_text[hmap['No.']].strip()
                if v.isdigit():
                    horse['horse_no'] = int(v)

            if 'Horse' in hmap:
                idx = hmap['Horse']
                cell_html = raw_html[idx]
                a_match = re.search(r'>([^<]+)</a>', cell_html)
                if a_match:
                    horse['horse_name'] = a_match.group(1).strip()
                else:
                    horse['horse_name'] = cell_text[idx].strip()

            if 'Draw' in hmap:
                v = cell_text[hmap['Draw']].strip()
                if v.isdigit():
                    horse['draw'] = int(v)

            if 'Jockey' in hmap:
                horse['jockey'] = cell_text[hmap['Jockey']].strip()

            if 'Trainer' in hmap:
                horse['trainer'] = cell_text[hmap['Trainer']].strip()

            if 'Wt.' in hmap:
                horse['weight'] = cell_text[hmap['Wt.']].strip()

            if 'Rating' in hmap:
                v = cell_text[hmap['Rating']].strip()
                m = re.search(r'(\d+)', v)
                if m:
                    horse['rating'] = int(m.group(1))

            if 'Odds' in hmap:
                v = cell_text[hmap['Odds']].strip()
                if re.match(r'^\d+\.\d+$', v):
                    horse['win_odds'] = float(v)

            if horse.get('horse_no'):
                horses.append(horse)

        return horses

    def _parse_meeting_list(self, html: str) -> list:
        """Parse the meeting list page."""
        soup = BeautifulSoup(html, 'lxml')
        meetings = []

        for a in soup.find_all('a', href=lambda h: h and 'racedate=' in h):
            href = a.get('href', '')
            date_match = re.search(r'racedate=(\d{4})%2f(\d{2})%2f(\d{2})', href)
            venue_match = re.search(r'Racecourse=([A-Z]+)', href)
            if date_match:
                year, month, day = date_match.groups()
                date_str = f"{year}/{month}/{day}"
                meetings.append({
                    'date': date_str,
                    'venue': venue_match.group(1) if venue_match else 'ST',
                    'text': a.get_text(strip=True),
                    'href': href,
                })

        return meetings
