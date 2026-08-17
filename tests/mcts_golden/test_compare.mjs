#!/usr/bin/env node
/**
 * Tests for the exact eager↔lazy comparator.
 *
 * Built and tested BEFORE any lazy capture exists, so the comparison cannot be
 * shaped by the traces it will judge.
 *
 * The synthetic "lazy" corpus is derived from the REAL committed eager corpus by
 * rewriting only the stage-identifying fields. That makes the baseline case a
 * genuine exact match, so every negative test perturbs exactly one thing away
 * from a known-passing state.
 *
 * Nothing here loads a model.
 */
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { EXPECTED_CASE_COUNT, REPO_ROOT, STAGES, enumerateCases } from './cases.mjs';
import {
  COMPARED_FIELDS,
  EAGER_CORPUS,
  EXCLUDED_FIELDS,
  EXIT_CORPUS_INVALID,
  EXIT_ERROR,
  EXIT_MATCH,
  EXIT_MISMATCH,
  EXIT_USAGE,
  PROGRESS_COMPARED_KEYS,
  compareArtifacts,
  compareCorpora,
  comparedProjection,
  corpusFingerprint,
  exitCodeForError,
  mainWithCode,
  parseArgs,
} from './compare.mjs';

const EAGER_DIR = join(REPO_ROOT, EAGER_CORPUS.relDir);
const LAZY_DIR = '/synthetic/lazy';

/**
 * A synthetic lazy corpus that is behaviourally IDENTICAL to the eager one.
 *
 * Only the stage-identifying fields are rewritten — schema, stage, surface,
 * pinned commit, capture commit — which is exactly what a real lazy capture
 * would differ by if the implementations agree.
 */
function syntheticLazy({ mutate } = {}) {
  const LAZY_COMMIT = 'c'.repeat(40);
  const files = new Map();
  for (const name of readdirSync(EAGER_DIR)) {
    const a = JSON.parse(readFileSync(join(EAGER_DIR, name), 'utf8'));
    files.set(name, {
      ...a,
      schema: STAGES.lazy.artifactSchema,
      stage: 'lazy',
      pinned_surface_commit: STAGES.lazy.surfaceCommit,
      execution_surface_sha256: STAGES.lazy.surfaceSha256,
      capture_commit: LAZY_COMMIT,
    });
  }
  mutate?.(files);
  return {
    lazyDir: LAZY_DIR,
    readdirSync: () => [...files.keys()],
    readFileSync: (p) => JSON.stringify(files.get(String(p).split('/').pop())),
  };
}

const run = (fs) => compareCorpora(fs);

/** One artifact from the real corpus, for pairwise tests. */
const sampleEager = () =>
  JSON.parse(readFileSync(join(EAGER_DIR, 'G_P01_baseline_s8.json'), 'utf8'));
const clone = (o) => JSON.parse(JSON.stringify(o));

// --- frozen configuration ----------------------------------------------------

test('the compared and excluded field sets are frozen and disjoint', () => {
  assert.deepEqual([...COMPARED_FIELDS], [
    'visit_counts',
    'root_value',
    'selected_move',
    'progress',
  ]);
  assert.deepEqual([...PROGRESS_COMPARED_KEYS], ['done', 'total', 'valueEstimate']);
  assert.deepEqual([...EXCLUDED_FIELDS], ['progress_elapsed_ms']);
  for (const f of EXCLUDED_FIELDS) assert.equal(COMPARED_FIELDS.includes(f), false);
  assert.throws(() => {
    COMPARED_FIELDS.push('nope');
  }, TypeError);
  assert.throws(() => {
    EAGER_CORPUS.fingerprint = 'x';
  }, TypeError);
});

test('the eager standard is frozen to the committed corpus', () => {
  assert.equal(EAGER_CORPUS.stage, 'eager');
  assert.equal(
    EAGER_CORPUS.fingerprint,
    '9e3a9037409c6eb4e72206a8b01697c1c138b0a90291462d713116106f2239f6'
  );
  assert.equal(EAGER_CORPUS.captureCommit, '841df6040a740a4b9f1753253e0e8bfc63e15366');
  // ...and it really is that corpus, re-derived from disk.
  assert.equal(corpusFingerprint(EAGER_DIR), EAGER_CORPUS.fingerprint);
  assert.equal(readdirSync(EAGER_DIR).length, EXPECTED_CASE_COUNT);
});

test('the projection carries the compared fields and NOTHING else', () => {
  const projected = comparedProjection(sampleEager());
  assert.deepEqual(Object.keys(projected).sort(), [...COMPARED_FIELDS].sort());
  for (const entry of projected.progress) {
    assert.deepEqual(Object.keys(entry).sort(), [...PROGRESS_COMPARED_KEYS].sort());
    assert.equal('elapsed' in entry, false);
  }
});

// --- the baseline: identical corpora match -----------------------------------

test('POSITIVE CONTROL: a behaviourally identical lazy corpus MATCHES', () => {
  const result = run(syntheticLazy());
  assert.equal(result.outcome, 'match');
  assert.deepEqual(result.mismatches, []);
  assert.equal(result.comparedCases, EXPECTED_CASE_COUNT);
  assert.equal(result.eagerFingerprint, EAGER_CORPUS.fingerprint);
});

test('EXCLUDED: an elapsed-only difference still MATCHES', () => {
  // The field §4.3 excludes. Requiring it would fail a correct implementation,
  // since search() derives it from Date.now().
  const fs = syntheticLazy({
    mutate: (files) => {
      for (const a of files.values()) {
        a.trace.progress_elapsed_ms = a.trace.progress_elapsed_ms.map((v) => v + 12345);
      }
    },
  });
  const result = run(fs);
  assert.equal(result.outcome, 'match', 'an excluded field was compared');
  assert.deepEqual(result.mismatches, []);
});

// --- negative: every compared field, at the PAIRWISE level -------------------
// Field-level negatives are exercised through compareArtifacts directly. Going
// through compareCorpora cannot reach them: validateCorpus rejects an
// internally inconsistent artifact as an INVALID CORPUS before any comparison
// happens — a visit count that no longer sums to the simulation count, or a
// selected_move that no longer follows from the counts, is a broken artifact,
// not a behavioural difference. Keeping those outcomes distinct is the point.

const pairDiffFields = (mutate) => {
  const a = sampleEager();
  const b = clone(a);
  mutate(b);
  return compareArtifacts('X', a, b).map((d) => d.field);
};

test('NEGATIVE: a visit-count VALUE difference is a mismatch', () => {
  const fields = pairDiffFields((b) => {
    b.trace.visit_counts[0][1] += 1;
  });
  assert.ok(fields.includes('visit_counts.value'), fields.join(','));
});

test('NEGATIVE: a visit-count ORDER difference is a mismatch', () => {
  // Same keys, same counts, different sequence. Order is part of the comparison.
  const fields = pairDiffFields((b) => {
    const vc = b.trace.visit_counts;
    [vc[0], vc[1]] = [vc[1], vc[0]];
  });
  assert.ok(fields.includes('visit_counts.order'), fields.join(','));
});

test('NEGATIVE: a visit-count LENGTH difference is a mismatch', () => {
  const fields = pairDiffFields((b) => {
    b.trace.visit_counts.pop();
  });
  assert.ok(fields.includes('visit_counts.length'), fields.join(','));
});

test('NEGATIVE: a root_value difference is a mismatch, with NO tolerance', () => {
  const fields = pairDiffFields((b) => {
    b.trace.root_value += 1e-12; // far below any plausible tolerance
  });
  assert.ok(fields.includes('root_value'), fields.join(','));
});

test('NEGATIVE: a selected_move difference is a mismatch', () => {
  const fields = pairDiffFields((b) => {
    b.trace.selected_move = '0,0';
  });
  assert.ok(fields.includes('selected_move'), fields.join(','));
});

test('NEGATIVE: each compared progress key is a mismatch', () => {
  for (const key of PROGRESS_COMPARED_KEYS) {
    const fields = pairDiffFields((b) => {
      b.trace.progress[0][key] =
        key === 'valueEstimate' ? b.trace.progress[0][key] + 1e-12 : 999;
    });
    assert.ok(fields.includes(`progress.${key}`), `${key}: ${fields.join(',')}`);
  }
});

test('NEGATIVE: a progress LENGTH difference is a mismatch', () => {
  const fields = pairDiffFields((b) => {
    b.trace.progress.pop();
  });
  assert.ok(fields.includes('progress.length'), fields.join(','));
});

// --- negative: end-to-end mismatch, with VALID artifacts ---------------------
// A real behavioural difference produces artifacts that are each internally
// valid and merely disagree. These mutations preserve every invariant
// validateCorpus enforces, so they reach the comparator.

test('NEGATIVE: an internally VALID root_value difference reaches the comparator', () => {
  const result = run(
    syntheticLazy({
      mutate: (files) => {
        const t = files.get('G_P01_baseline_s8.json').trace;
        t.root_value = t.root_value === 0 ? 0.5 : t.root_value / 2; // still within [-1, 1]
      },
    })
  );
  assert.equal(result.outcome, 'mismatch');
  assert.deepEqual(result.mismatches.map((m) => m.field), ['root_value']);
});

test('NEGATIVE: an internally VALID valueEstimate difference reaches the comparator', () => {
  const result = run(
    syntheticLazy({
      mutate: (files) => {
        const t = files.get('G_P01_baseline_s8.json').trace;
        t.progress[0].valueEstimate = t.progress[0].valueEstimate / 2;
      },
    })
  );
  assert.equal(result.outcome, 'mismatch');
  assert.deepEqual(result.mismatches.map((m) => m.field), ['progress.valueEstimate']);
});

test('NEGATIVE: a REDISTRIBUTED visit count reaches the comparator', () => {
  // Move one visit between moves and recompute the readout: the sum, the key
  // order and the selected_move rule all still hold, so the artifact is valid
  // and differs only behaviourally — exactly the shape of a real divergence.
  const result = run(
    syntheticLazy({
      mutate: (files) => {
        const t = files.get('G_P01_baseline_s8.json').trace;
        const from = t.visit_counts.findIndex((e) => e[1] > 0);
        const to = t.visit_counts.findIndex((e, i) => i !== from && e[1] === 0);
        t.visit_counts[from][1] -= 1;
        t.visit_counts[to][1] += 1;
        let best = null;
        let max = -1;
        for (const [k, c] of t.visit_counts) {
          if (c > max || (c === max && k < best)) {
            max = c;
            best = k;
          }
        }
        t.selected_move = best;
      },
    })
  );
  assert.equal(result.outcome, 'mismatch');
  assert.ok(result.mismatches.some((m) => m.field === 'visit_counts.value'));
});

test('NEGATIVE: a mismatch in ANY single case fails the whole comparison', () => {
  for (const caseId of ['G_P01_baseline_s1', 'G_P16_baseline_s800', 'A2']) {
    const result = run(
      syntheticLazy({
        mutate: (files) => {
          const t = files.get(`${caseId}.json`).trace;
          t.root_value = t.root_value === 0 ? 0.25 : t.root_value / 2;
        },
      })
    );
    assert.equal(result.outcome, 'mismatch', caseId);
    assert.ok(result.mismatches.some((m) => m.caseId === caseId), caseId);
  }
});

test('an internally INCONSISTENT lazy artifact is a CORPUS failure, not a mismatch', () => {
  // The outcomes must stay distinct: a broken artifact is not evidence that the
  // implementations disagree.
  assert.throws(
    () =>
      run(
        syntheticLazy({
          mutate: (files) => {
            files.get('G_P01_baseline_s8.json').trace.visit_counts[0][1] += 1;
          },
        })
      ),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

// --- negative: corpus identity and validity ----------------------------------

test('NEGATIVE: a MISSING case is a corpus failure, never a skip', () => {
  assert.throws(
    () => run(syntheticLazy({ mutate: (files) => files.delete('A2.json') })),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

test('NEGATIVE: an EXTRA artifact is a corpus failure', () => {
  assert.throws(
    () =>
      run(
        syntheticLazy({
          mutate: (files) => files.set('G_P99_baseline_s1.json', { schema: 'x' }),
        })
      ),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

test('NEGATIVE: the WRONG STAGE in the lazy corpus is a corpus failure', () => {
  assert.throws(
    () =>
      run(
        syntheticLazy({
          mutate: (files) => {
            for (const a of files.values()) a.stage = 'eager';
          },
        })
      ),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

test('NEGATIVE: the WRONG SURFACE in the lazy corpus is a corpus failure', () => {
  assert.throws(
    () =>
      run(
        syntheticLazy({
          mutate: (files) => {
            for (const a of files.values()) {
              a.execution_surface_sha256 = STAGES.eager.surfaceSha256;
            }
          },
        })
      ),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

test('NEGATIVE: a non-uniform lazy capture commit is a corpus failure', () => {
  assert.throws(
    () =>
      run(
        syntheticLazy({
          mutate: (files) => {
            files.get('A1.json').capture_commit = 'd'.repeat(40);
          },
        })
      ),
    (err) => err.code === 'LAZY_CORPUS_INVALID'
  );
});

test('NEGATIVE: an unreadable lazy corpus is a corpus failure, not a mismatch', () => {
  assert.throws(
    () =>
      compareCorpora({
        lazyDir: LAZY_DIR,
        readdirSync: () => {
          throw new Error('ENOENT');
        },
        readFileSync: () => '{}',
      }),
    (err) => err.code === 'CORPUS_UNREADABLE'
  );
});

// --- outcomes stay distinct --------------------------------------------------

test('the three outcomes map to distinct exit codes', () => {
  assert.equal(EXIT_MATCH, 0);
  assert.equal(EXIT_MISMATCH, 1);
  assert.equal(EXIT_USAGE, 2);
  assert.equal(EXIT_CORPUS_INVALID, 3);
  assert.equal(EXIT_ERROR, 4);
  assert.equal(new Set([EXIT_MATCH, EXIT_MISMATCH, EXIT_CORPUS_INVALID, EXIT_ERROR]).size, 4);

  // A corpus problem must never be reported as a behavioural mismatch.
  for (const code of ['EAGER_FINGERPRINT_MISMATCH', 'LAZY_CORPUS_INVALID', 'CORPUS_UNREADABLE']) {
    assert.equal(exitCodeForError({ code }), EXIT_CORPUS_INVALID, code);
  }
  // Any other fault is an error, never a mismatch and never a corpus verdict.
  for (const thrown of [null, undefined, 0, '', new Error('x'), { code: 'WEIRD' }]) {
    const c = exitCodeForError(thrown);
    assert.equal(c, EXIT_ERROR, String(thrown));
    assert.notEqual(c, EXIT_MISMATCH);
  }
});

test('the CLI returns 0 on match, 1 on mismatch, 3 on corpus failure, 4 on fault', () => {
  assert.equal(
    mainWithCode(['/x'], { compareFn: () => ({ outcome: 'match', comparedCases: 92, comparedFields: [], excludedFields: [], mismatches: [] }) }),
    EXIT_MATCH
  );
  assert.equal(
    mainWithCode(['/x'], {
      compareFn: () => ({
        outcome: 'mismatch',
        comparedCases: 92,
        comparedFields: [],
        excludedFields: [],
        mismatches: [{ caseId: 'A1', field: 'root_value' }],
      }),
    }),
    EXIT_MISMATCH
  );
  assert.equal(
    mainWithCode(['/x'], {
      compareFn: () => {
        throw Object.assign(new Error('bad'), { code: 'LAZY_CORPUS_INVALID' });
      },
    }),
    EXIT_CORPUS_INVALID
  );
  for (const thrown of [null, undefined, Object.freeze(new Error('frozen'))]) {
    assert.equal(
      mainWithCode(['/x'], {
        compareFn: () => {
          throw thrown;
        },
      }),
      EXIT_ERROR,
      String(thrown)
    );
  }
});

test('the CLI takes exactly one argument: the LAZY directory', () => {
  assert.deepEqual(parseArgs(['/lazy']), { lazyDir: '/lazy' });
  for (const argv of [[], ['/a', '/b'], ['--eager', '/a'], ['']]) {
    assert.throws(() => parseArgs(argv), (err) => err.code === 'USAGE', JSON.stringify(argv));
  }
  assert.equal(mainWithCode([]), EXIT_USAGE);
  // The eager corpus is NOT a parameter: there is no way to supply a different
  // standard on the command line.
  assert.equal(mainWithCode(['/a', '/b']), EXIT_USAGE);
});

// --- pairwise comparator, directly -------------------------------------------

test('compareArtifacts reports no difference for identical inputs', () => {
  const a = sampleEager();
  assert.deepEqual(compareArtifacts('X', a, clone(a)), []);
});

test('compareArtifacts ignores every non-compared field', () => {
  const a = sampleEager();
  const b = clone(a);
  b.pid = 999999;
  b.capture_commit = 'f'.repeat(40);
  b.stage = 'lazy';
  b.schema = 'anything';
  b.fixture = { ...b.fixture, n_legal: 1 };
  b.trace.progress_elapsed_ms = b.trace.progress_elapsed_ms.map((v) => v + 1);
  assert.deepEqual(compareArtifacts('X', a, b), [], 'a non-compared field was compared');
});

test('compareArtifacts stops after the first ordering difference in a case', () => {
  // Reporting every subsequent index would bury the actual divergence point.
  const a = sampleEager();
  const b = clone(a);
  b.trace.visit_counts.reverse();
  const diffs = compareArtifacts('X', a, b);
  assert.equal(diffs.filter((d) => d.field === 'visit_counts.order').length, 1);
  assert.equal(diffs[0].index, 0);
});
