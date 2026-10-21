#!/usr/bin/env python3
"""
Scrape PUBLISHED odds for an UPCOMING HKJC meeting from bet.hkjc.com.

Why this exists
---------------
The v6 model's feature importance is dominated by odds terms (prob_norm,
implied_prob, log_odds together ~79%). The official race card from
racing.hkjc.com carries the field, jockey, trainer, draw and rating but NOT
the published odds. Serving the model without odds turns every odds feature
into 0 after fillna — an input distribution the model has never seen, and the
predictions are worthless.

So the serving path is: racecard (racing.hkjc.com) + published odds
(bet.hkjc.com), joined on horse number.

Verified 2026-10-21: odds for the next meeting ARE published ~overnight
before, so a pre-race prediction can use them.

DOM facts (verified against the live site, not assumed):
  * URL date format is DASHED `YYYY-MM-DD` — the slash form renders an empty
    shell with no odds cells.
  * Win odds cell id is `odds_WIN_<raceNo>_<horseNo>`; place odds is
    `odds_PLA_<raceNo>_<horseNo>`. The horse number is the LAST segment.
  * Each odds cell is a DIV inside a TD, inside a TR with class
    `rc-odds-row`. The TR text is:
        <no>\t<name>\t<draw>\t<rating>\t<jockey>\t<trainer>\t<win>\t<place>
  * The favourite carries class `favourite` on its odds DIV.

Usage:
    python3 scripts/scrape_upcoming_odds.py --date 2026-10-21 --venue HV
    python3 scripts/scrape_upcoming_odds.py --date 2026-10-21 --venue HV --races 1 2 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = ("https://bet.hkjc.com/ch/racing/wp/{date}/{venue}/{raceNo}"
        "?b_cid=SPLDSPA_hkjc-home_MegaMenu")

# Runs inside the page: pull every runner row out of the rendered DOM.
#
# The page renders TWO tables per race:
#   (a) the full race card row  -> cells: [no, colour, name, draw, rating,
#                                      jockey, trainer, win, place]
#   (b) an odds summary row     -> cells: [no, "<name>騎<jockey>練<trainer>",
#                                      win, place, '', '']
# Both contain an `odds_WIN_<race>_<horseNo>` cell, so a naive query returns
# every horse twice. Table (a) is the one with the metadata, so keep only
# rows that have >= 7 cells, and take the metadata from the correct indices
# (colour occupies index 1).
EXTRACT_JS = """
(raceNo) => {
  const out = [];
  const cells = document.querySelectorAll(`[id^="odds_WIN_${raceNo}_"]`);
  for (const cell of cells) {
    const id = cell.id;                       // odds_WIN_<race>_<horseNo>
    const horseNo = id.split('_').pop();
    const row = cell.closest('tr');
    if (!row) continue;
    const tds = [...row.querySelectorAll('td')].map(t => t.innerText.trim());
    // Skip the odds-summary duplicate rows (5 cells); keep the card rows.
    if (tds.length < 7) continue;
    const winText = cell.innerText.trim();
    const placeEl = row.querySelector(`#odds_PLA_${raceNo}_${horseNo}`);
    const placeText = placeEl ? placeEl.innerText.trim() : '';
    out.push({
      race_no: Number(raceNo),
      horse_no: Number(horseNo),
      horse_name: (tds[2] || '').split('\\n')[0].trim(),
      draw: tds[3] || '',
      rating: tds[4] || '',
      jockey: (tds[5] || '').split('\\n')[0].trim(),
      trainer: (tds[6] || '').split('\\n')[0].trim(),
      win_odds: winText === '' ? null : parseFloat(winText),
      place_odds: placeText === '' ? null : parseFloat(placeText),
      is_favourite: cell.className.includes('favourite'),
      withdrawn: (tds[4] || '').includes('退出') || (tds[3] || '').includes('退出'),
    });
  }
  return out;
}
"""


def scrape(page, date, venue, race_no, wait_ms=6000):
    # DASH format is required. The slash format silently renders an empty SPA
    # shell — HTTP 200, 38KB of chrome, zero odds cells.
    page.goto(BASE.format(date=date, venue=venue, raceNo=race_no),
              timeout=45000)
    try:
        page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:
        pass
    time.sleep(wait_ms / 1000.0)
    return page.evaluate(EXTRACT_JS, str(race_no))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True,
                    help='YYYY-MM-DD (DASH format required by bet.hkjc.com)')
    ap.add_argument('--venue', required=True, help='HV or ST')
    ap.add_argument('--races', type=int, nargs='*', default=None,
                    help='race numbers; default = 1..12 until empty')
    ap.add_argument('--out', default=None)
    ap.add_argument('--wait-ms', type=int, default=6000)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright missing; run with /home/work/projects/gymbro/.venv/bin/python",
              file=sys.stderr)
        return 2

    if '-' not in args.date:
        print("date must be YYYY-MM-DD (dashes)", file=sys.stderr)
        return 2

    result = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        races = args.races if args.races else list(range(1, 13))
        for rn in races:
            try:
                horses = scrape(page, args.date, args.venue, rn, args.wait_ms)
                if horses:
                    result[rn] = horses
                    w = [h['win_odds'] for h in horses if h.get('win_odds')]
                    print(f"  R{rn}: {len(horses)} runners, win odds "
                          f"{min(w)}~{max(w)}")
                else:
                    print(f"  R{rn}: no odds published yet")
            except Exception as e:
                print(f"  R{rn}: ERROR {type(e).__name__}: {e}")
            time.sleep(1.5)
        browser.close()

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"wrote {args.out} ({sum(len(v) for v in result.values())} runners)")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())
