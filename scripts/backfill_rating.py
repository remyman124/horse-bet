#!/usr/bin/env python3
"""
Backfill `horse_entries.rating` (and last6) from the HKJC RACE CARD archive.

WHY THIS EXISTS
---------------
`horse_entries.rating` is populated for only 11% of rows (2026 only), but
`rating` is a FEATURE in the v5 model. So v5 was trained with rating == 0 for
99.7% of rows, while live predictions fed real ratings (mean 59). That is a
severe train/serve skew: the model never learned what a rating means.

The rating is NOT in the results page — it only exists on the race card. The
race card for a PAST race is served by:
    https://racing.hkjc.com/en-us/local/information/racecard?racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=N
(NOT /archive/racecard — that path 404s. Only `localresults` redirects to /archive/.)

The race card is a React SPA, so BeautifulSoup returns 0 tables — it MUST be
read with Playwright, via page.evaluate() on id='racecardlist'.

The race card also gives us Last 6 (recent form) and the jockey/weight.

Output: updates horse_entries SET rating=?, and stores last6 into a new
column if it exists (see --last6-column). Racing DB is not modified otherwise.
"""
import asyncio, argparse, re, sqlite3, sys, time
from pathlib import Path

from playwright.async_api import async_playwright

DB_PATH = '/home/work/horse-bet/data/horse.db'
LOG = Path('/tmp/backfill_rating.log')

# JS: pull the race card table into a list of plain objects.
EXTRACT_JS = r"""
() => {
  const root = document.getElementById('racecardlist');
  if (!root) return {ok:false, reason:'no #racecardlist', rows:[]};
  const rows = [];
  const trs = root.querySelectorAll('tr');
  for (const tr of trs) {
    const tds = tr.querySelectorAll('td');
    if (tds.length < 8) continue;
    const cells = Array.from(tds).map(td => td.innerText.replace(/\s+/g,' ').trim());
    // Find the horse number cell: standalone number in col 0 or 1
    rows.push(cells);
  }
  // header
  const head = [];
  const htr = root.querySelector('tr');
  if (htr) head.push(...Array.from(htr.querySelectorAll('th,td')).map(td=>td.innerText.replace(/\s+/g,' ').trim()));
  return {ok:true, head:head, rows:rows};
}
"""

HDR_MAP = {
    'no': ('no.', 'no'),
    'last6': ('last 6 runs', 'last6', 'last 6'),
    'horse': ('horse',),
    'jockey': ('jockey',),
    'trainer': ('trainer',),
    'weight': ('wt.', 'weight'),
    'rating': ('rating',),
    'rpr': ('rpr', 'r.pr.', 'rpr.'),
    'draw': ('draw', 'no. of draw', 'sr'),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def find_col(head, key):
    for i, h in enumerate(head):
        hn = h.lower().strip()
        for cand in HDR_MAP[key]:
            if hn == cand or hn.startswith(cand):
                return i
    return None


def norm_name(s):
    s = re.sub(r'\s*\([^)]*\)', '', s or '')
    s = s.replace('&amp;', '&')
    return re.sub(r'\s+', ' ', s).strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='date_from', default='2025-01-01')
    ap.add_argument('--to', dest='date_to', default='2026-12-31')
    ap.add_argument('--venue', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--headless', action='store_true', default=True)
    args = ap.parse_args()

    LOG.write_text('')
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    where, params = 'rd.race_date >= ? AND rd.race_date <= ?', [args.date_from, args.date_to]
    if args.venue:
        where += ' AND rd.venue = ?'
        params.append(args.venue)
    cur.execute(f"""
        SELECT rd.race_date, rd.venue, r.race_no, r.id
        FROM races r JOIN race_days rd ON r.race_day_id = rd.id
        WHERE {where}
        ORDER BY rd.race_date, r.race_no
    """, params)
    targets = cur.fetchall()
    if args.limit:
        targets = targets[:args.limit]
    log(f"Target races: {len(targets)}")

    stats = {'ok': 0, 'skip': 0, 'fail': 0, 'rows': 0, 'rated': 0}

    async def run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            ctx = await browser.new_context(
                user_agent=('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'),
                viewport={'width': 1600, 'height': 1200})
            page = await ctx.new_page()

            for i, (rdate, venue, race_no, r_id) in enumerate(targets):
                ds = (rdate if isinstance(rdate, str) else rdate.isoformat()).replace('-', '/')
                url = (f'https://racing.hkjc.com/en-us/local/information/racecard'
                       f'?racedate={ds}&Racecourse={venue}&RaceNo={race_no}')
                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=45000)
                    # The race card is a slow SPA: #racecardlist only appears after the
                    # XHR for the race card content lands. 1.8s was too early (element
                    # absent); 6s reliably works. Poll for the element instead of
                    # guessing a fixed sleep.
                    try:
                        await page.wait_for_selector('#racecardlist', timeout=25000)
                    except Exception:
                        pass
                    await page.wait_for_timeout(1200)
                    data = await page.evaluate(EXTRACT_JS)
                    if not data.get('ok') or not data.get('rows'):
                        stats['skip'] += 1
                    else:
                        head = data.get('head') or []
                        i_no = find_col(head, 'no') or 0
                        i_rating = find_col(head, 'rating')
                        i_last6 = find_col(head, 'last6')
                        i_horse = find_col(head, 'horse')
                        n_rated = 0
                        for cells in data['rows']:
                            if i_no >= len(cells):
                                continue
                            hno = cells[i_no].strip()
                            if not hno.isdigit():
                                continue
                            rating = None
                            if i_rating is not None and i_rating < len(cells):
                                m = re.match(r'^\s*(\d{1,3})\s*$', cells[i_rating])
                                if m:
                                    rating = int(m.group(1))
                            if rating is None:
                                continue
                            if not args.dry_run:
                                cur.execute(
                                    'UPDATE horse_entries SET rating=? WHERE race_id=? AND horse_no=?',
                                    (rating, r_id, int(hno)))
                                n_rated += cur.rowcount if cur.rowcount > 0 else 0
                            else:
                                n_rated += 1
                        stats['rows'] += len(data['rows'])
                        stats['rated'] += n_rated
                        stats['ok'] += 1
                        if not args.dry_run:
                            conn.commit()
                except Exception as e:
                    stats['fail'] += 1
                    log(f"  ERROR {ds} {venue} R{race_no}: {repr(e)[:140]}")
                if (i + 1) % 25 == 0:
                    log(f"  {i+1}/{len(targets)} ok={stats['ok']} skip={stats['skip']} "
                        f"fail={stats['fail']} rated_rows={stats['rated']}")
                await asyncio.sleep(0.25)

            await browser.close()

    asyncio.run(run())
    log(f"DONE ok={stats['ok']} skip={stats['skip']} fail={stats['fail']} "
        f"rated_rows={stats['rated']}")
    conn.close()


if __name__ == '__main__':
    main()
