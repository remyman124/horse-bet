#!/usr/bin/env node
/**
 * Regression test for the race-card METADATA parser in
 * scripts/build_racecard.py (EXTRACT_JS).
 *
 * Bug: the surface regex only knew "AWT", but HKJC spells it out in full --
 *   "All Weather Track, 1650M, Good"
 * so every all-weather meeting came back with distance=null and track
 * defaulted to "Turf". The model then got the wrong distance/class/track for
 * every runner of that meeting.
 *
 * EXTRACT_JS lives in a Python triple-quoted string, so the JS source has to
 * come from Python (which applies the string escapes). The test shells out to
 * python3 to dump it, then runs it under a stub DOM.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PY = join(HERE, '..', 'scripts', 'build_racecard.py');

const DUMP = `
import importlib.util, sys
spec = importlib.util.spec_from_file_location("br", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.stdout.write(m.EXTRACT_JS)
`;

const EXTRACT_JS = execFileSync('python3', ['-c', DUMP, PY], {
  encoding: 'utf8',
});

// A minimal but REAL race-card table: the header row must carry every column
// the extractor needs, or it bails out before it ever reaches the metadata.
const HEADER_CELLS = [
  'Horse No.', 'Last 6 Runs', 'Colour', 'Horse', 'Brand No.', 'Wt.',
  'Jockey', 'Over Wt.', 'Draw', 'Trainer', "Int'l Rtg.", 'Rtg.',
  'Rtg.+/-', 'Horse Wt. (Declaration)', 'Days since Last Run', 'Gear',
];
const DATA_CELLS = [
  '1', '14/14/11/10/12', '', 'LUCKY SECRET', 'L073', '135', 'M Chadwick',
  '', '10', 'W K Mo', '-', '40', '-6', '1218', '199', 'TT',
];

const row = (cells) => ({
  querySelectorAll: () => cells.map((t) => ({ innerText: t })),
});

/** Run EXTRACT_JS against a synthetic race-card page body. */
function parseMeta(bodyText) {
  const table = {
    querySelectorAll: (q) => (q === 'tr' ? [row(HEADER_CELLS), row(DATA_CELLS)] : []),
  };
  const document = {
    body: { innerText: bodyText },
    querySelector: (sel) => (sel === '#racecardlist' ? table : null),
  };
  // eslint-disable-next-line no-new-func
  const fn = new Function('document', `return (${EXTRACT_JS})(document);`);
  return fn(document);
}

const PAGE = (surface, distance, going) => [
  'Home 中文',
  '',
  'Race 1 - SOME HANDICAP',
  'Wednesday, October 28, 2026, Sha Tin, 18:45',
  `${surface}, ${distance}M, ${going}`,
  'Prize Money: $875,000, Rating: 40-0, Class 5',
].join('\n');

test('the all-weather surface spelled out in full is parsed', () => {
  const r = parseMeta(PAGE('All Weather Track', 1650, 'Good'));
  assert.equal(r.meta.distance, 1650);
  assert.equal(r.meta.track, 'AWT');
  assert.equal(r.meta.going, 'Good');
  assert.equal(r.meta.race_class, 'Class 5');
});

test('the AWT abbreviation still parses', () => {
  const r = parseMeta(PAGE('AWT', 1650, 'Good'));
  assert.equal(r.meta.distance, 1650);
  assert.equal(r.meta.track, 'AWT');
});

test('turf meetings are unaffected', () => {
  const r = parseMeta(PAGE('Turf', 1200, 'Good To Firm'));
  assert.equal(r.meta.distance, 1200);
  assert.equal(r.meta.track, 'Turf');
  assert.equal(r.meta.going, 'Good To Firm');
});

test('dirt meetings are unaffected', () => {
  const r = parseMeta(PAGE('Dirt', 1800, 'Standard'));
  assert.equal(r.meta.distance, 1800);
  assert.equal(r.meta.track, 'Dirt');
});

test('the runner row is still extracted', () => {
  const r = parseMeta(PAGE('All Weather Track', 1650, 'Good'));
  assert.equal(r.n, 1);
  assert.equal(r.data[0].horse_name, 'LUCKY SECRET');
  assert.equal(r.data[0].draw, '10');
  assert.equal(r.data[0].rating, '40');
});
