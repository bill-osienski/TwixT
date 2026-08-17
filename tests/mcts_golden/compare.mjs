#!/usr/bin/env node
/**
 * The exact eager↔lazy trace comparator (design §4.3, §4.4).
 *
 *   node tests/mcts_golden/compare.mjs <lazy_corpus_dir>
 *
 * Built BEFORE any lazy capture exists. Writing a comparator after seeing the
 * traces it judges would let the comparison be shaped by its own result, which
 * is the whole reason §4 is preregistered.
 *
 * ## What it compares, exactly
 *
 *   visit_counts    values AND order
 *   root_value
 *   selected_move
 *   progress        restricted to done, total, valueEstimate
 *
 * ## What it deliberately does NOT compare
 *
 *   progress_elapsed_ms   wall-clock metadata. `search()` derives it from
 *                         Date.now() (server/mcts.js:73), so no two runs
 *                         reproduce it and requiring equality would fail a
 *                         CORRECT implementation. §4.3 excludes it by design,
 *                         and a test asserts an elapsed-only difference PASSES.
 *
 * There are no tolerances, no optional checks, no caller-supplied standards and
 * no skipped cases. All 92 are compared or the run has no result.
 *
 * ## The standard is frozen; only the subject is supplied
 *
 * The eager corpus is read from a frozen path and pinned by a frozen
 * fingerprint. The caller supplies only the lazy directory — the thing being
 * judged. A caller who could point the comparator at a different "eager" corpus
 * would be supplying the standard it is judged by.
 *
 * ## PRECOMMITTED INTERPRETATION — fixed before any result exists
 *
 * A PASS means **exact agreement on those fields, across these 92 cases only.**
 *
 * It does NOT establish global implementation equivalence, heap safety,
 * performance, strength, or correctness outside the frozen corpus. Any
 * compared-field mismatch is preserved and STOPS the programme before any heap
 * or timing work.
 *
 * Exit codes — three outcomes kept distinct on purpose:
 *   0  MATCH             exact agreement on all 92 cases
 *   1  MISMATCH          a genuine behavioural difference
 *   2  usage
 *   3  CORPUS_INVALID    either corpus failed validation or identity
 *   4  error             any other harness fault
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import {
  CaptureError,
  EXPECTED_CASE_COUNT,
  REPO_ROOT,
  artifactName,
  codeOfThrown,
  describeThrown,
  enumerateCases,
  sha256,
  stageConfig,
  validateCorpus,
} from './cases.mjs';
import { deriveExpectedFixtures } from './worker.mjs';

export const EXIT_MATCH = 0;
export const EXIT_MISMATCH = 1;
export const EXIT_USAGE = 2;
export const EXIT_CORPUS_INVALID = 3;
export const EXIT_ERROR = 4;

/** The frozen standard: which eager corpus, and proof it is that one. */
export const EAGER_CORPUS = Object.freeze({
  stage: 'eager',
  relDir: 'tests/mcts_golden/golden/841df60/artifacts',
  captureCommit: '841df6040a740a4b9f1753253e0e8bfc63e15366',
  fingerprint: '9e3a9037409c6eb4e72206a8b01697c1c138b0a90291462d713116106f2239f6',
});

/** The subject: which stage a lazy corpus must declare. Its surface follows. */
export const LAZY_STAGE = 'lazy';

/** §4.3. Frozen so the comparison cannot be widened or narrowed at call time. */
export const COMPARED_FIELDS = Object.freeze([
  'visit_counts',
  'root_value',
  'selected_move',
  'progress',
]);
export const PROGRESS_COMPARED_KEYS = Object.freeze(['done', 'total', 'valueEstimate']);
export const EXCLUDED_FIELDS = Object.freeze(['progress_elapsed_ms']);

const CORPUS_INVALID_CODES = new Set([
  'EAGER_FINGERPRINT_MISMATCH',
  'EAGER_CORPUS_INVALID',
  'LAZY_CORPUS_INVALID',
  'LAZY_COMMIT_NOT_UNIFORM',
  'CORPUS_UNREADABLE',
]);

/** Map any thrown value to an exit code, keeping the three outcomes distinct. */
export function exitCodeForError(err) {
  const code = codeOfThrown(err);
  return code !== null && CORPUS_INVALID_CODES.has(code) ? EXIT_CORPUS_INVALID : EXIT_ERROR;
}

/**
 * The fingerprint of a corpus directory: sha256 over the sorted
 * `filename:sha256` manifest. Identical construction to the one recorded in
 * GOLDEN.md, so the committed value can be checked against a fresh read.
 */
export function corpusFingerprint(dir, readdir = readdirSync, read = readFileSync) {
  // Each line is TERMINATED by a newline, including the last. That is not
  // cosmetic: the frozen fingerprint was produced by
  //   for f in $(ls *.json | sort); do printf '%s:%s\n' ...; done | shasum -a 256
  // and `printf '%s\n'` terminates every line, so a `join('\n')` — which
  // separates instead of terminating — hashes different bytes and reproduces a
  // different digest. GOLDEN.md documents that command; this must agree with it.
  const lines = [...readdir(dir)]
    .sort()
    .map((f) => `${f}:${sha256(read(join(dir, f)))}\n`);
  return sha256(lines.join(''));
}

/** The compared projection of one artifact, and nothing else. */
export function comparedProjection(artifact) {
  return {
    visit_counts: artifact.trace.visit_counts,
    root_value: artifact.trace.root_value,
    selected_move: artifact.trace.selected_move,
    progress: artifact.trace.progress.map((entry) => {
      const projected = {};
      for (const key of PROGRESS_COMPARED_KEYS) projected[key] = entry[key];
      return projected;
    }),
  };
}

/**
 * Compare one pair of artifacts on the compared fields only.
 *
 * Returns a list of differences; empty means exact agreement. Equality is
 * exact — no tolerance anywhere, including on `root_value`, which is a float
 * produced by identical arithmetic on identical inputs and must reproduce
 * bit-for-bit. (`!==` is used rather than `Object.is` so a `-0`/`0` pair is not
 * reported as a difference; JSON cannot represent `-0` anyway, so both sides
 * always carry `0`.)
 */
export function compareArtifacts(caseId, eager, lazy) {
  const diffs = [];
  const differ = (field, detail) => diffs.push({ caseId, field, ...detail });

  const e = comparedProjection(eager);
  const l = comparedProjection(lazy);

  // visit_counts: values AND order.
  if (!Array.isArray(e.visit_counts) || !Array.isArray(l.visit_counts)) {
    differ('visit_counts', { eager: typeof e.visit_counts, lazy: typeof l.visit_counts });
  } else if (e.visit_counts.length !== l.visit_counts.length) {
    differ('visit_counts.length', {
      eager: e.visit_counts.length,
      lazy: l.visit_counts.length,
    });
  } else {
    for (let i = 0; i < e.visit_counts.length; i++) {
      const [ek, ev] = e.visit_counts[i];
      const [lk, lv] = l.visit_counts[i];
      if (ek !== lk) {
        // Order is part of the comparison: the same keys in a different
        // sequence is a difference, not a reordering to be normalised away.
        differ('visit_counts.order', { index: i, eager: ek, lazy: lk });
        break;
      }
      if (ev !== lv) {
        differ('visit_counts.value', { index: i, move: ek, eager: ev, lazy: lv });
      }
    }
  }

  if (e.root_value !== l.root_value) {
    differ('root_value', { eager: e.root_value, lazy: l.root_value });
  }
  if (e.selected_move !== l.selected_move) {
    differ('selected_move', { eager: e.selected_move, lazy: l.selected_move });
  }

  if (e.progress.length !== l.progress.length) {
    differ('progress.length', { eager: e.progress.length, lazy: l.progress.length });
  } else {
    for (let i = 0; i < e.progress.length; i++) {
      for (const key of PROGRESS_COMPARED_KEYS) {
        if (e.progress[i][key] !== l.progress[i][key]) {
          differ(`progress.${key}`, {
            index: i,
            eager: e.progress[i][key],
            lazy: l.progress[i][key],
          });
        }
      }
    }
  }

  return diffs;
}

/**
 * Compare a lazy corpus against the frozen eager one.
 *
 * The eager side is always read from the real filesystem at the frozen path;
 * only the lazy reader is injectable, because only the lazy corpus is the
 * subject. Every one of the 92 cases is compared — a case that cannot be read
 * is a corpus failure, never a skip.
 */
export function compareCorpora({
  lazyDir,
  readdirSync: lazyReaddir = readdirSync,
  readFileSync: lazyRead = readFileSync,
}) {
  if (typeof lazyDir !== 'string' || lazyDir.length === 0) {
    throw new CaptureError('USAGE', 'a lazy corpus directory is required');
  }

  const eagerDir = join(REPO_ROOT, EAGER_CORPUS.relDir);
  const expectedFixtures = deriveExpectedFixtures();

  // --- the standard: is this the corpus we froze? --------------------------
  let eagerFingerprint;
  try {
    eagerFingerprint = corpusFingerprint(eagerDir);
  } catch (err) {
    throw new CaptureError('CORPUS_UNREADABLE', `eager corpus: ${describeThrown(err)}`);
  }
  if (eagerFingerprint !== EAGER_CORPUS.fingerprint) {
    throw new CaptureError(
      'EAGER_FINGERPRINT_MISMATCH',
      `eager corpus fingerprint is ${eagerFingerprint}, frozen as ${EAGER_CORPUS.fingerprint}`
    );
  }

  const eagerFailures = validateCorpus(eagerDir, {
    mode: 'capture',
    stage: EAGER_CORPUS.stage,
    expectedCaptureCommit: EAGER_CORPUS.captureCommit,
    expectedFixtures,
    readdirSync,
    readFileSync,
  });
  if (eagerFailures.length) {
    throw new CaptureError(
      'EAGER_CORPUS_INVALID',
      `${eagerFailures.length} failure(s): ${eagerFailures.map((f) => f.code).join(', ')}`
    );
  }

  // --- the subject: a valid lazy corpus of exactly 92 artifacts ------------
  stageConfig(LAZY_STAGE);
  let lazyNames;
  try {
    lazyNames = [...lazyReaddir(lazyDir)];
  } catch (err) {
    throw new CaptureError('CORPUS_UNREADABLE', `lazy corpus: ${describeThrown(err)}`);
  }

  // The lazy capture commit is a recorded property OF THE SUBJECT, not a
  // standard, so it is read from the artifacts — and requiring every artifact
  // to carry the same one is what validateCorpus then enforces.
  const first = enumerateCases()[0];
  let lazyCommit;
  try {
    lazyCommit = JSON.parse(lazyRead(join(lazyDir, artifactName(first.caseId)), 'utf8'))
      .capture_commit;
  } catch (err) {
    throw new CaptureError(
      'CORPUS_UNREADABLE',
      `lazy corpus: cannot read ${artifactName(first.caseId)}: ${describeThrown(err)}`
    );
  }
  if (typeof lazyCommit !== 'string' || !/^[0-9a-f]{40}$/.test(lazyCommit)) {
    throw new CaptureError(
      'LAZY_COMMIT_NOT_UNIFORM',
      `lazy corpus carries no usable capture_commit: ${String(lazyCommit)}`
    );
  }

  const lazyFailures = validateCorpus(lazyDir, {
    mode: 'capture',
    stage: LAZY_STAGE,
    expectedCaptureCommit: lazyCommit,
    expectedFixtures,
    readdirSync: lazyReaddir,
    readFileSync: lazyRead,
  });
  if (lazyFailures.length) {
    throw new CaptureError(
      'LAZY_CORPUS_INVALID',
      `${lazyFailures.length} failure(s): ${lazyFailures.map((f) => f.code).join(', ')}`
    );
  }

  // --- the comparison ------------------------------------------------------
  const cases = enumerateCases();
  const mismatches = [];
  let compared = 0;

  for (const testCase of cases) {
    const name = artifactName(testCase.caseId);
    const eager = JSON.parse(readFileSync(join(eagerDir, name), 'utf8'));
    const lazy = JSON.parse(lazyRead(join(lazyDir, name), 'utf8'));
    mismatches.push(...compareArtifacts(testCase.caseId, eager, lazy));
    compared += 1;
  }

  if (compared !== EXPECTED_CASE_COUNT) {
    // Unreachable while enumerateCases() is frozen at 92, but a comparison that
    // silently covered fewer cases would be the worst possible failure mode.
    throw new CaptureError(
      'LAZY_CORPUS_INVALID',
      `compared ${compared} cases, expected ${EXPECTED_CASE_COUNT}`
    );
  }

  return {
    outcome: mismatches.length === 0 ? 'match' : 'mismatch',
    comparedCases: compared,
    comparedFields: [...COMPARED_FIELDS],
    excludedFields: [...EXCLUDED_FIELDS],
    eagerFingerprint,
    eagerCaptureCommit: EAGER_CORPUS.captureCommit,
    lazyCaptureCommit: lazyCommit,
    lazyArtifactCount: lazyNames.length,
    mismatches,
  };
}

// --- CLI ---------------------------------------------------------------------

const USAGE = 'usage: compare.mjs <lazy_corpus_dir>';

export function parseArgs(argv) {
  if (argv.length !== 1 || !argv[0] || argv[0].startsWith('--')) {
    throw new CaptureError('USAGE', USAGE);
  }
  return { lazyDir: argv[0] };
}

export function mainWithCode(argv, { compareFn = compareCorpora } = {}) {
  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (err) {
    console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
    return EXIT_USAGE;
  }

  let result;
  try {
    result = compareFn({ lazyDir: parsed.lazyDir });
  } catch (err) {
    console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
    return exitCodeForError(err);
  }

  console.log(JSON.stringify({ ...result, mismatches: result.mismatches.slice(0, 20) }, null, 2));
  console.log('');
  console.log(`compared ${result.comparedCases} cases on ${result.comparedFields.join(', ')}`);
  console.log(`excluded ${result.excludedFields.join(', ')}`);

  if (result.outcome === 'match') {
    console.log('EXACT MATCH');
    console.log(
      'Interpretation, fixed before this ran: exact agreement on those fields across ' +
        'these 92 cases ONLY. It does not establish global implementation equivalence, ' +
        'heap safety, performance, strength, or correctness outside the frozen corpus.'
    );
    return EXIT_MATCH;
  }

  console.log(`MISMATCH: ${result.mismatches.length} compared-field difference(s)`);
  console.log('Preserve this result. It stops the programme before any heap or timing work.');
  return EXIT_MISMATCH;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  try {
    process.exitCode = mainWithCode(process.argv.slice(2));
  } catch (err) {
    console.error(`ERROR: ${describeThrown(err)}`);
    process.exitCode = EXIT_ERROR;
  }
}
