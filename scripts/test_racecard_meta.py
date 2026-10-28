"""
Regression test for the race-card header parser.

THE BUG THIS TEST EXISTS FOR
----------------------------
`build_racecard.py` extracts race-level metadata (track / distance / class /
going) from the header block of the official HKJC racecard page. The header
reads:

    Race 1 - WA MEI SHAN HANDICAP
    All Weather Track, 1650M, Good
    Prize Money: $875,000, Rating: 40-0, Class 5

The original regex only knew the abbreviation "AWT", so on an all-weather
meeting (every Sha Tin meeting in the winter, and Happy Valley all year) it
matched NOTHING. The result was a race card where:

    distance  -> empty for every runner
    track     -> silently defaulted to 'Turf'
    going     -> empty

`predict_racecard.py` then detected the missing distance and fell back to
filling it from the DATABASE keyed by race_no -- which is exactly the wrong
source, because race_no 1 of a past meeting is a different distance and class
from race_no 1 of tonight. The serving path therefore scored tonight's races
with a past meeting's distance (1200m/1400m/1600m/2000m) instead of the real
one (1650M/1200M/1800M), and with 'Turf' instead of 'AWT'.

The failure was SILENT: the scraper reported success, the card had the right
number of runners, and the odds cross-check passed. Only the prediction log
gave it away, in a line that says "may be WRONG for tonight".

The lesson: the metadata parser is a first-class piece of logic, so it gets
a test with real HKJC header strings in it.

Run:
    cd /home/work/horse-bet && .venv/bin/python scripts/test_racecard_meta.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The exact regexes from build_racecard.py's EXTRACT_JS, lifted verbatim into
# Python so they can be tested without a browser. If they drift, this file
# must be updated in the same commit -- that is the point.
TRACKS = 'All Weather Track|AWT|Turf|Dirt|GPT'

COURSE_RE = re.compile(r'\n(' + TRACKS + r')[^\n]*?(\d{3,4})M')
CLASS_RE = re.compile(r'Prize Money:[^\n]*?Class ([^\n]+)')
GOING_RE = re.compile(r'\n(?:' + TRACKS + r')[^\n]*?\d{3,4}M,\s*([^\n]+)')


def parse_meta(txt):
    """Mirror of the metadata extraction in build_racecard.py EXTRACT_JS."""
    meta = {}
    course = COURSE_RE.search(txt)
    if course:
        meta['track'] = 'AWT' if course.group(1) == 'All Weather Track' else course.group(1)
        meta['distance'] = float(course.group(2))
    pm = CLASS_RE.search(txt)
    if pm:
        meta['race_class'] = 'Class ' + pm.group(1).strip()
    going = GOING_RE.search(txt)
    if going:
        meta['going'] = going.group(1).strip()
    return meta


# Real header blocks captured from racing.hkjc.com.
# Meeting: 2026-10-28 Sha Tin (ST), 9 races -- all-weather surface.
AWT_HEADERS = {
    1: 'Race 1 - WA MEI SHAN HANDICAP\n'
       'All Weather Track, 1650M, Good\n'
       'Prize Money: $875,000, Rating: 40-0, Class 5\n',
    2: 'Race 2 - WA MEI SHAN HANDICAP\n'
       'All Weather Track, 1650M, Good\n'
       'Prize Money: $875,000, Rating: 40-0, Class 5\n',
    3: 'Race 3 - TSIM MEI FUNG HANDICAP\n'
       'All Weather Track, 1200M, Good\n'
       'Prize Money: $875,000, Rating: 40-0, Class 5\n',
    4: 'Race 4 - TIU YUE YUNG HANDICAP\n'
       'All Weather Track, 1800M, Good\n'
       'Prize Money: $1,170,000, Rating: 60-40, Class 4\n',
    5: 'Race 5 - SHEK UK SHAN HANDICAP\n'
       'All Weather Track, 1200M, Good\n'
       'Prize Money: $1,170,000, Rating: 60-40, Class 4\n',
    6: 'Race 6 - TIN HA SHAN HANDICAP\n'
       'All Weather Track, 1650M, Good\n'
       'Prize Money: $1,170,000, Rating: 60-40, Class 4\n',
    7: 'Race 7 - SHEK UK SHAN HANDICAP\n'
       'All Weather Track, 1200M, Good\n'
       'Prize Money: $1,170,000, Rating: 60-40, Class 4\n',
    8: 'Race 8 - HEBE HILL HANDICAP\n'
       'All Weather Track, 1200M, Good\n'
       'Prize Money: $1,860,000, Rating: 85-60, Class 3\n',
    9: 'Race 9 - PYRAMID HILL HANDICAP\n'
       'All Weather Track, 1650M, Good\n'
       'Prize Money: $2,050,000, Rating: 85-60, Class 3\n',
}

# A turf meeting (Happy Valley) -- the format the old regex DID handle.
TURF_HEADER = ('Race 1 - RACING SPIRIT HANDICAP\n'
               'Turf, "B" Course, 1650M, Good\n'
               'Prize Money: $1,170,000, Rating: 60-40, Class 4\n')

# AWT spelled as the abbreviation, in case HKJC reverts.
AWT_ABBR_HEADER = ('Race 1 - SOME HANDICAP\n'
                   'AWT, 1650M, Good\n'
                   'Prize Money: $875,000, Rating: 40-0, Class 5\n')

DIRT_HEADER = ('Race 1 - SOME HANDICAP\n'
               'Dirt, 1200M, Good\n'
               'Prize Money: $875,000, Rating: 40-0, Class 5\n')


def check(label, got, want):
    if got != want:
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
        return 1
    print(f"  ok   {label}: {got!r}")
    return 0


def main():
    fails = 0
    total = 0

    print("AWT meeting (2026-10-28 ST) -- the case that used to break:")
    for rn, header in AWT_HEADERS.items():
        m = parse_meta(header)
        total += 3
        fails += check(f"R{rn} track", m.get('track'), 'AWT')
        fails += check(f"R{rn} distance", m.get('distance'), float(re.search(r'(\d{3,4})M', header).group(1)))
        fails += check(f"R{rn} going", m.get('going'), 'Good')

    print("\nClass extraction:")
    total += 1
    fails += check("R1 class", parse_meta(AWT_HEADERS[1]).get('race_class'), 'Class 5')
    total += 1
    fails += check("R8 class", parse_meta(AWT_HEADERS[8]).get('race_class'), 'Class 3')
    total += 1
    fails += check("R4 class", parse_meta(AWT_HEADERS[4]).get('race_class'), 'Class 4')

    print("\nOther surfaces:")
    m = parse_meta(TURF_HEADER)
    total += 3
    fails += check("turf track", m.get('track'), 'Turf')
    fails += check("turf distance", m.get('distance'), 1650.0)
    fails += check("turf class", m.get('race_class'), 'Class 4')

    m = parse_meta(AWT_ABBR_HEADER)
    total += 2
    fails += check("AWT abbr track", m.get('track'), 'AWT')
    fails += check("AWT abbr distance", m.get('distance'), 1650.0)

    m = parse_meta(DIRT_HEADER)
    total += 2
    fails += check("dirt track", m.get('track'), 'Dirt')
    fails += check("dirt distance", m.get('distance'), 1200.0)

    print("\nGuards:")
    # Every race in a meeting must yield a distance, or the card is unusable.
    missing = [rn for rn, h in AWT_HEADERS.items() if not parse_meta(h).get('distance')]
    total += 1
    if missing:
        print(f"  FAIL races missing distance: {missing}")
        fails += 1
    else:
        print("  ok   every race in the meeting yields a distance")

    # The whole point: no race in the meeting may fall back to 'Turf'.
    wrong = [rn for rn, h in AWT_HEADERS.items() if parse_meta(h).get('track') != 'AWT']
    total += 1
    if wrong:
        print(f"  FAIL AWT races mis-parsed as non-AWT: {wrong}")
        fails += 1
    else:
        print("  ok   every AWT race is labelled AWT (no silent 'Turf' default)")

    print(f"\n{total - fails}/{total} checks passed")
    if fails:
        print("FAILED")
        return 1
    print("PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
