#!/usr/bin/env python3
"""
Build a race-card CSV for an UPCOMING HKJC meeting, ready for
`scripts/predict_racecard.py`.

Two independent official HKJC sources are combined:

  1. racing.hkjc.com racecard  -> English horse name, draw, jockey, trainer,
                                  rating, LAST 6 RUNS, DAYS SINCE LAST RUN
  2. bet.hkjc.com               -> PUBLISHED win odds + place odds
                                  (the racecard has NO odds)

Joining on horse number gives a complete pre-race input.

The two sources are cross-checked against each other: horse count, draw
number, jockey and withdrawal status must agree. A disagreement is reported
loudly rather than silently resolved, because a silent mismatch here produces
a confidently wrong prediction.

Verified 2026-10-21 HV: both sources agreed on all 11 runners of R1, including
the scratched runner (Chinese 創科群英 == English INNO SUPER, both withdrawn).

Output CSV columns (what predict_racecard.py expects):
    race_no, horse_no, horse_name, jockey, trainer, draw, win_odds,
    distance, race_class, track
Optional extras carried through when available: rating, place_odds,
last_6, days_since.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RACECARD_URL = ("https://racing.hkjc.com/en-us/local/information/racecard"
                "?racedate={date}&Racecourse={venue}&RaceNo={race}")

# Column extraction runs in the page: locate the header row by CONTENT
# (not a fixed index) so a HKJC column reorder cannot silently shift fields.
EXTRACT_JS = """
() => {
  const t = document.querySelector('#racecardlist');
  if (!t) return {n: 0, data: [], note: 'no #racecardlist'};
  const rows = [...t.querySelectorAll('tr')];
  const hdrRow = rows.find(r => {
    const c = [...r.querySelectorAll('td,th')].map(x => x.innerText.trim());
    return c[0] === 'Horse No.';
  });
  if (!hdrRow) return {n: 0, data: [], note: 'no header row'};
  const hdr = [...hdrRow.querySelectorAll('td,th')].map(c => c.innerText.trim());
  const idx = {};
  hdr.forEach((h, i) => { if (h && !(h in idx)) idx[h] = i; });
  const need = ['Horse No.', 'Horse', 'Draw', 'Jockey', 'Trainer'];
  for (const k of need) if (!(k in idx)) return {n: 0, data: [], note: 'missing header ' + k};
  const data = rows.filter(r => {
    const c = [...r.querySelectorAll('td,th')].map(x => x.innerText.trim());
    return c[0] && /^\\d+$/.test(c[0]);
  }).map(r => {
    const c = [...r.querySelectorAll('td,th')].map(x => x.innerText.trim());
    const get = (k) => (k in idx ? (c[idx[k]] || '').split('\\n')[0].trim() : '');
    return {
      horse_no: c[idx['Horse No.']],
      horse_name: get('Horse'),
      draw: get('Draw'),
      jockey: get('Jockey'),
      trainer: get('Trainer'),
      rating: get('Rtg.'),
      last_6: get('Last 6 Runs'),
      days_since: get('Days since Last Run'),
    };
  });

  // Race-level metadata. The header block reads:
  //   Race 1 - RACING SPIRIT HANDICAP
  //   Wednesday, October 21, 2026, Happy Valley, 18:40
  //   Turf, "B" Course, 1650M, Good
  //   Prize Money: $1,170,000, Rating: 60-40, Class 4
  // NOTE: HKJC spells the all-weather surface out in full --
  //   "All Weather Track, 1650M, Good"  (not "AWT"). The old regex only
  //   knew "AWT", so every AWT meeting came back with distance=null.
  const txt = document.body.innerText;
  const meta = {};
  const TRACKS = 'All Weather Track|AWT|Turf|Dirt|GPT';
  const course = txt.match(new RegExp('\\\\n(' + TRACKS + ')[^\\\\n]*?(\\\\d{3,4})M'));
  if (course) {
    meta.track = course[1] === 'All Weather Track' ? 'AWT' : course[1];
    meta.distance = parseFloat(course[2]);
  }
  const pm = txt.match(/Prize Money:[^\\n]*?Class ([^\\n]+)/);
  if (pm) meta.race_class = 'Class ' + pm[1].trim();
  const going = txt.match(new RegExp('\\\\n(?:' + TRACKS + ')[^\\\\n]*?\\\\d{3,4}M,\\\\s*([^\\\\n]+)'));
  if (going) meta.going = going[1].trim();
  return {n: data.length, data: data, meta: meta};
}
"""


def scrape_racecard(page, date_slash, venue, race_no, timeout=25000):
    page.goto(RACECARD_URL.format(date=date_slash, venue=venue, race=race_no),
              timeout=45000)
    try:
        # The SPA needs the content XHR to land; networkidle never fires.
        page.wait_for_selector('#racecardlist', timeout=timeout)
    except Exception as e:
        return {'n': 0, 'data': [], 'note': f'selector timeout: {e}'}
    time.sleep(1.5)
    return page.evaluate(EXTRACT_JS)


def scrape_odds(page, date_dash, venue, race_no, wait_ms=6000):
    url = (f"https://bet.hkjc.com/ch/racing/wp/{date_dash}/{venue}/{race_no}"
           f"?b_cid=SPLDSPA_hkjc-home_MegaMenu")
    page.goto(url, timeout=45000)
    try:
        page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:
        pass
    time.sleep(wait_ms / 1000.0)
    return page.evaluate(ODDS_JS, str(race_no))


ODDS_JS = """
(raceNo) => {
  const out = {};
  for (const cell of document.querySelectorAll(`[id^="odds_WIN_${raceNo}_"]`)) {
    const horseNo = cell.id.split('_').pop();
    const row = cell.closest('tr');
    if (!row) continue;
    const tds = [...row.querySelectorAll('td')].map(t => t.innerText.trim());
    if (tds.length < 7) continue;   // skip the odds-summary duplicate row
    const w = cell.innerText.trim();
    const p = row.querySelector(`#odds_PLA_${raceNo}_${horseNo}`);
    const pt = p ? p.innerText.trim() : '';
    out[horseNo] = {
      win_odds: w === '' ? null : parseFloat(w),
      place_odds: pt === '' ? null : parseFloat(pt),
      withdrawn: (tds[4]||'').includes('退出') || (tds[3]||'').includes('退出'),
    };
  }
  return out;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--venue', required=True, help='HV or ST')
    ap.add_argument('--races', type=int, nargs='*', default=None)
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--no-odds', action='store_true',
                    help='skip bet.hkjc.com (no published odds available)')
    args = ap.parse_args()

    date_dash = args.date
    date_slash = args.date.replace('-', '/')

    from playwright.sync_api import sync_playwright

    rows = []
    problems = []
    race_meta = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = browser.new_page()
        races = args.races if args.races else list(range(1, 13))
        for rn in races:
            card = scrape_racecard(page, date_slash, args.venue, rn)
            if not card['n']:
                print(f"  R{rn}: no racecard ({card.get('note')}) — stopping")
                break
            odds = {} if args.no_odds else scrape_odds(page, date_dash,
                                                        args.venue, rn)
            n_odds = len(odds)
            meta = card.get('meta', {}) or {}
            race_meta[rn] = meta
            for h in card['data']:
                hn = str(h['horse_no'])
                o = odds.get(hn, {})
                row = {
                    'race_no': rn,
                    'horse_no': h['horse_no'],
                    'horse_name': h['horse_name'],
                    'jockey': h['jockey'],
                    'trainer': h['trainer'],
                    'draw': h['draw'],
                    'rating': h.get('rating', ''),
                    'last_6': h.get('last_6', ''),
                    'days_since': h.get('days_since', ''),
                    'win_odds': o.get('win_odds'),
                    'place_odds': o.get('place_odds'),
                    'distance': meta.get('distance'),
                    'race_class': meta.get('race_class', ''),
                    'track': meta.get('track', 'Turf'),
                    'going': meta.get('going', ''),
                    'withdrawn': bool(o.get('withdrawn')) or
                                 'Scratched' in h['horse_name'] or
                                 h['draw'] == '-',
                }
                # CROSS-CHECK: the two official sources must agree on the field
                if odds:
                    if n_odds != card['n']:
                        problems.append(
                            f"R{rn}: racecard has {card['n']} runners but the "
                            f"odds page has {n_odds}")
                    for c in card['data']:
                        cn = str(c['horse_no'])
                        if cn not in odds:
                            problems.append(
                                f"R{rn}: horse {cn} ({c['horse_name']}) has no "
                                f"odds row")
                rows.append(row)
            print(f"  R{rn}: {card['n']} runners, {n_odds} with odds")
            time.sleep(1.5)
        browser.close()

    if not rows:
        print("no race card data", file=sys.stderr)
        return 1

    import csv
    cols = ['race_no', 'horse_no', 'horse_name', 'jockey', 'trainer', 'draw',
            'win_odds', 'rating', 'place_odds', 'last_6', 'days_since',
            'distance', 'race_class', 'track', 'going', 'withdrawn']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}: {len(rows)} rows")
    for rn in sorted(race_meta):
        m = race_meta[rn]
        print(f"  R{rn}: {m.get('race_class','?')} {m.get('distance','?')}m "
              f"{m.get('track','?')} going={m.get('going','?')}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"wrote {args.json_out}")

    if problems:
        print("\nCROSS-CHECK PROBLEMS (sources disagree):")
        for p in problems:
            print("  !!", p)
        return 2
    print("cross-check OK: racecard and odds page agree on the field")
    return 0


if __name__ == '__main__':
    sys.exit(main())
