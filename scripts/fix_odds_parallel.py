#!/usr/bin/env python3
"""
Parallel driver for fix_odds_and_dividends.py.

The original script awaits each request inside a for-loop -> effective concurrency 1
(~0.5 races/s => 7+ hours for 13,216 races). This driver runs N workers over a shared
work queue so we finish in minutes, with a resume-skip so already-repaired races are
not re-fetched.

Safety: modest concurrency (default 6) + per-worker delay. HKJC rate-limits aggressive
parallelism, so if we see failures we back off rather than hammer.
"""
import asyncio, argparse, time, sqlite3, sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from fix_odds_and_dividends import parse_results_page, HEADERS, UNIT, LOG_FILE

DB_PATH = '/home/work/horse-bet/data/horse.db'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def already_repaired(conn, race_id):
    """
    A race counts as repaired only when BOTH hold:

    1. The winner's win_odds has at most 1 decimal place. Values with 2+ decimals
       were derived from the post-race WIN dividend, which only exists for the
       winner — that leaves a winner-only fingerprint the model can exploit.
       Every horse must come from the same published "Win Odds" column.
    2. The stored win_dividend is consistent with that odds. The tolerance is
       wide because in tote-era races the WIN payout legitimately exceeds
       SP x 10 (e.g. published SP 14.0 -> dividend 148.50, i.e. 14.85 effective).
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT w.win_odds, rr.win_dividend
        FROM horse_entries w JOIN race_results rr ON rr.race_id = w.race_id
        WHERE w.race_id = ? AND w.finishing_position = 1
    """, (race_id,))
    row = cur.fetchone()
    if not row:
        return False
    odds, div = row
    if odds is None or div is None:
        return False
    try:
        odds = float(odds)
        div = float(div)
    except (TypeError, ValueError):
        return False
    # More than 1 decimal place => dividend-derived fingerprint => must re-scrape.
    if abs(odds * 10 - round(odds * 10)) > 1e-6:
        return False
    return abs(div - odds * 10) < 10.5


async def worker(queue, client, conn, stats, delay):
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        rdate, venue, race_no, r_id = item
        ds = (rdate if isinstance(rdate, str) else rdate.isoformat()).replace('-', '/')
        url = (f'https://racing.hkjc.com/en-us/local/information/localresults'
               f'?racedate={ds}&Racecourse={venue}&RaceNo={race_no}')
        try:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code != 200 or 'No information' in resp.text:
                stats['skipped'] += 1
            else:
                parsed = parse_results_page(resp.text)
                if not parsed:
                    stats['failed'] += 1
                else:
                    cur = conn.cursor()
                    n_odds = 0
                    for hno, h in parsed['horses'].items():
                        if h['odds'] is None:
                            continue
                        cur.execute('UPDATE horse_entries SET win_odds=? WHERE race_id=? AND horse_no=?',
                                    (h['odds'], r_id, hno))
                        n_odds += cur.rowcount if cur.rowcount > 0 else 0
                    win_div = parsed['win_dividend']
                    if win_div is None:
                        winners = [h for h in parsed['horses'].values() if h['pos'] == 1]
                        if winners and winners[0]['odds']:
                            win_div = winners[0]['odds'] * UNIT
                    # ⚠️ DO NOT overwrite the winner's win_odds with dividend/10.
                    # The WIN dividend is a POST-RACE quantity, and the winner is the
                    # only horse for which we have it. Storing a 2-decimal value for
                    # the winner and a 1-decimal published value for every other
                    # horse creates a provenance fingerprint that lets a model
                    # identify the winner without any real signal. Every horse must
                    # keep the SAME source (the published "Win Odds" column).
                    # The precise dividend lives in race_results.win_dividend.
                    if win_div:
                        pd = parsed['place_dividends']
                        cur.execute("""
                            UPDATE race_results SET
                              win_dividend=?,
                              place_dividend_1=COALESCE(?, place_dividend_1),
                              place_dividend_2=COALESCE(?, place_dividend_2),
                              place_dividend_3=COALESCE(?, place_dividend_3)
                            WHERE race_id=?
                        """, (win_div, pd[0] if len(pd) > 0 else None,
                              pd[1] if len(pd) > 1 else None,
                              pd[2] if len(pd) > 2 else None, r_id))
                        stats['div'] += 1
                    stats['odds'] += n_odds
                    stats['ok'] += 1
        except Exception as e:
            stats['failed'] += 1
            stats['last_err'] = repr(e)[:160]
        finally:
            # COMMIT PER RACE. With isolation_level=None (autocommit) the UPDATEs above
            # are already durable, but an explicit commit keeps that guarantee obvious
            # and makes the durability boundary explicit per race.
            try:
                conn.commit()
            except Exception as e:
                stats['failed'] += 1
                stats['last_err'] = repr(e)[:160]
            queue.task_done()
            await asyncio.sleep(delay)
            stats['done'] += 1
            if stats['done'] % 200 == 0:
                log(f"  {stats['done']}/{stats['total']} ok={stats['ok']} skip={stats['skipped']} "
                    f"fail={stats['failed']} odds_rows={stats['odds']} div={stats['div']} "
                    f"({stats['done']/(time.time()-stats['t0']):.1f}/s)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='date_from', default='2006-01-01')
    ap.add_argument('--to', dest='date_to', default='2026-12-31')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--delay', type=float, default=0.5)
    ap.add_argument('--resume', action='store_true', help='skip already-repaired races')
    args = ap.parse_args()

    LOG_FILE.write_text('')
    conn = sqlite3.connect(DB_PATH, timeout=60.0, isolation_level=None)
    # WAL + a generous busy timeout. Two concurrent repair jobs on a rollback-journal
    # DB produced "database is locked" and SILENTLY LOST most UPDATEs (the job
    # reported ok=N while the rows never changed). WAL lets readers run during a
    # write and makes lock contention far less likely.
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=60000')
    conn.execute('PRAGMA synchronous=NORMAL')
    cur = conn.cursor()
    cur.execute("""
        SELECT rd.race_date, rd.venue, r.race_no, r.id
        FROM races r JOIN race_days rd ON r.race_day_id = rd.id
        WHERE EXISTS (SELECT 1 FROM horse_entries he WHERE he.race_id = r.id)
          AND rd.race_date >= ? AND rd.race_date <= ?
        ORDER BY rd.race_date, r.race_no
    """, (args.date_from, args.date_to))
    targets = cur.fetchall()
    log(f"Target races: {len(targets)}  workers={args.workers} delay={args.delay}")

    if args.resume:
        before = len(targets)
        targets = [t for t in targets if not already_repaired(conn, t[3])]
        log(f"Resume: skipped {before - len(targets)} already-repaired -> {len(targets)} to do")

    queue = asyncio.Queue()
    for t in targets:
        queue.put_nowait(t)

    stats = {'ok': 0, 'skipped': 0, 'failed': 0, 'odds': 0, 'div': 0, 'done': 0,
             'total': len(targets), 't0': time.time(), 'last_err': ''}

    async with httpx.AsyncClient(headers=HEADERS, limits=httpx.Limits(
            max_connections=args.workers, max_keepalive_connections=args.workers),
            follow_redirects=True) as client:
        tasks = [asyncio.create_task(worker(queue, client, conn, stats, args.delay))
                 for _ in range(args.workers)]
        await asyncio.gather(*tasks)

    conn.commit()
    el = time.time() - stats['t0']
    log(f"DONE: ok={stats['ok']} skipped={stats['skipped']} failed={stats['failed']} "
        f"odds_rows={stats['odds']} div_races={stats['div']} elapsed={el:.0f}s "
        f"rate={stats['done']/el:.1f}/s")
    if stats['last_err']:
        log(f"last error: {stats['last_err']}")

    # ── VERIFY: did the rows actually change? ─────────────────────────────────
    # The 2026-10-20 run reported "ok=1600" while the DB was completely unchanged
    # (lock contention silently dropped every UPDATE). A job that counts HTTP
    # successes is not a job that counts database changes. Re-open a FRESH
    # connection and count how many races are now consistent.
    vconn = sqlite3.connect(DB_PATH, timeout=60.0)
    vcur = vconn.cursor()
    vcur.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN w.win_odds IS NOT NULL AND rr.win_dividend IS NOT NULL
                         AND ABS(CAST(rr.win_dividend AS REAL) - w.win_odds*10) < 5.1
                        THEN 1 ELSE 0 END)
        FROM races r
        JOIN race_days rd ON r.race_day_id = rd.id
        JOIN horse_entries w ON w.race_id = r.id AND w.finishing_position = 1
        LEFT JOIN race_results rr ON rr.race_id = r.id
        WHERE rd.race_date >= ? AND rd.race_date <= ?
    """, (args.date_from, args.date_to))
    total, consistent = vcur.fetchone()
    log(f"VERIFY: {consistent}/{total} races in range now have win_dividend ~= winner_odds*10 "
        f"(tolerance 5.1 covers the published-odds integer truncation)")
    if total and (consistent or 0) / total < 0.5:
        log("WARNING: fewer than half the races verified — writes are NOT landing. "
            "Check for another process holding the DB lock.")
    vconn.close()
    conn.close()


if __name__ == '__main__':
    asyncio.run(main())
