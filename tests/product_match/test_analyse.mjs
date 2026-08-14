/**
 * Tests for the independent match analyser.
 *
 *   node --test tests/product_match/test_analyse.mjs
 *
 * A valid run is played once at 2 simulations on throwaway openings, then each
 * test corrupts exactly one property of a COPY and asserts the analyser refuses
 * it. The point of the analyser is that it rejects, so most of these are
 * negative.
 *
 * These games are machinery fixtures and constitute no evidence about either
 * model.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import {
  mkdtemp,
  rm,
  mkdir,
  readdir,
  readFile,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  FROZEN_SPEC,
  analyse,
  bootstrapInterval,
  classify,
  decide,
  mulberry32,
  tInterval,
} from './analyse.mjs';
import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  FINGERPRINT_FIELDS,
  runMatch,
} from './harness.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');

const P = 3;
const SIMS = 2;
const OPENINGS = [
  [
    [5, 5],
    [10, 10],
    [7, 6],
    [12, 11],
  ],
  [
    [6, 6],
    [11, 11],
    [8, 7],
    [13, 12],
  ],
  [
    [4, 4],
    [9, 9],
    [6, 5],
    [11, 10],
  ],
];

/** Frozen everywhere except the two knobs the fixture genuinely differs on. */
const TEST_SPEC = {
  ...FROZEN_SPEC,
  nSimulations: SIMS,
  tCritical: { 3: 4.30265273 }, // t(0.975, df=2)
};

let root;
let validDir;
let validSidecars;

/**
 * Write a run directory from sidecars, optionally mutated.
 *
 * Files are named by index, not by pair/slot, so a malformed SET (duplicate
 * slots, wrong pair numbering) can be written at all — naming by pair/slot
 * would silently collapse collisions and hide the very cases under test.
 */
async function runFrom(name, mutate = () => {}, meta = {}) {
  const d = join(root, name);
  await mkdir(join(d, 'match'), { recursive: true });
  const cars = structuredClone(validSidecars);
  const out = mutate(cars) ?? cars;
  for (let i = 0; i < out.length; i++) {
    await writeFile(
      join(d, 'match', `game_${String(i).padStart(4, '0')}.json`),
      JSON.stringify(out[i], null, 2)
    );
  }
  const fp = Object.fromEntries(
    FINGERPRINT_FIELDS.map((f) => [f, validSidecars[0][f]])
  );
  if (meta.runJson !== null) {
    await writeFile(
      join(d, 'run.json'),
      JSON.stringify(
        { fingerprint: fp, P: meta.P ?? P, ...meta.extra },
        null,
        2
      )
    );
  }
  return d;
}

const expectReject = async (name, code, mutate, meta) => {
  const d = await runFrom(name, mutate, meta);
  const r = await analyse(d, OPENINGS, TEST_SPEC);
  assert.strictEqual(r.verdict, 'REJECTED', `expected rejection for ${name}`);
  const codes = r.failures.map((f) => f.code);
  assert.ok(
    codes.includes(code),
    `expected ${code}, got: ${[...new Set(codes)].join(', ')}`
  );
};

describe('statistics are exactly determined', () => {
  it('the bootstrap is reproducible and uses the frozen order statistics', () => {
    const scores = [0, 0.25, 0.5, 0.75, 1, 0.5, 0.5, 0.25];
    const a = bootstrapInterval(scores, FROZEN_SPEC);
    const b = bootstrapInterval(scores, FROZEN_SPEC);
    assert.deepStrictEqual(a, b, 'same seed must give identical bounds');
    assert.ok(a.lower <= a.upper);

    // Re-derive the endpoints independently, to prove the convention is
    // r[250] / r[9749] with no interpolation.
    const rand = mulberry32(FROZEN_SPEC.bootstrapSeed);
    const means = [];
    for (let k = 0; k < FROZEN_SPEC.bootstrapReplicates; k++) {
      let sum = 0;
      for (let i = 0; i < scores.length; i++) {
        sum +=
          scores[
            Math.min(scores.length - 1, Math.floor(rand() * scores.length))
          ];
      }
      means.push(sum / scores.length);
    }
    means.sort((x, y) => x - y);
    assert.strictEqual(a.lower, means[250]);
    assert.strictEqual(a.upper, means[9749]);
  });

  it('a degenerate sample yields a degenerate interval', () => {
    const r = bootstrapInterval([1, 1, 1, 1], FROZEN_SPEC);
    assert.strictEqual(r.lower, 1);
    assert.strictEqual(r.upper, 1);
  });

  it('the t interval uses the frozen critical value', () => {
    const scores = Array.from({ length: 100 }, (_, i) => (i < 60 ? 1 : 0));
    const r = tInterval(scores, FROZEN_SPEC);
    const sd = Math.sqrt((60 * 0.4 ** 2 + 40 * 0.6 ** 2) / 99);
    assert.ok(Math.abs(r.mean - 0.6) < 1e-12);
    assert.ok(Math.abs(r.sd - sd) < 1e-12);
    assert.ok(Math.abs(r.lower - (0.6 - 1.9842169515 * (sd / 10))) < 1e-12);
  });

  it('refuses a P the specification does not allow', () => {
    // Only 100 and 200 are reachable; anything else means the run was not the
    // one that was preregistered.
    assert.throws(
      () => tInterval([0.5, 0.5, 0.5], FROZEN_SPEC),
      /no frozen t critical value/
    );
  });

  it('requires both methods to agree, in either direction', () => {
    const above = { lower: 0.55, upper: 0.7 };
    const below = { lower: 0.3, upper: 0.45 };
    const straddle = { lower: 0.45, upper: 0.6 };
    assert.strictEqual(decide(above, above), 'CANDIDATE_STRONGER');
    assert.strictEqual(decide(below, below), 'CANDIDATE_WEAKER');
    assert.strictEqual(
      decide(above, straddle),
      'UNRESOLVED',
      'disagreement is unresolved'
    );
    assert.strictEqual(decide(straddle, above), 'UNRESOLVED');
    assert.strictEqual(
      decide(below, straddle),
      'UNRESOLVED',
      'symmetric for harm'
    );
    assert.strictEqual(decide(straddle, straddle), 'UNRESOLVED');
  });

  it('classifies three-valued, so weaker and inconclusive do not "agree"', () => {
    const below = { lower: 0.3, upper: 0.45 };
    const straddle = { lower: 0.45, upper: 0.6 };
    assert.strictEqual(classify(below), 'weaker');
    assert.strictEqual(classify(straddle), 'inconclusive');
    // Both have a lower bound under 0.5, so a lower-bound comparison would
    // have called this agreement.
    assert.notStrictEqual(classify(below), classify(straddle));
  });
});

describe('analysis of a real run', () => {
  before(async () => {
    root = await mkdtemp(join(tmpdir(), 'twixt-analyse-'));
    validDir = join(root, 'valid');
    await runMatch({
      runDir: validDir,
      P,
      openings: OPENINGS,
      baselineDir: join(REPO_ROOT, 'models', BASELINE_MODEL_ID),
      candidateDir: join(REPO_ROOT, 'models', CANDIDATE_MODEL_ID),
      nSimulations: SIMS,
      requireCleanWorktree: false,
    });
    validSidecars = [];
    for (const n of (await readdir(join(validDir, 'match'))).sort()) {
      validSidecars.push(
        JSON.parse(await readFile(join(validDir, 'match', n), 'utf8'))
      );
    }
  });
  after(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it('accepts a well-formed run and reaches a decision', async () => {
    const r = await analyse(validDir, OPENINGS, TEST_SPEC);
    assert.strictEqual(
      r.verdict,
      'ACCEPTED',
      JSON.stringify(r.failures?.slice(0, 3))
    );
    assert.strictEqual(r.P, P);
    assert.strictEqual(r.pair_scores.length, P);
    for (const s of r.pair_scores)
      assert.ok([0, 0.25, 0.5, 0.75, 1].includes(s));
    assert.ok(
      ['CANDIDATE_STRONGER', 'CANDIDATE_WEAKER', 'UNRESOLVED'].includes(
        r.decision
      )
    );
    assert.strictEqual(
      r.pair_tally.win + r.pair_tally.draw + r.pair_tally.loss,
      P
    );
  });

  it('the fixture really is complete, or every negative test is vacuous', () => {
    assert.strictEqual(validSidecars.length, 2 * P);
    for (const s of validSidecars) {
      assert.strictEqual(s.kind, 'match');
      assert.strictEqual(s.opening_id, s.pair_index);
      assert.ok(s.moves.length > 4);
    }
  });

  // --- structural rejections ------------------------------------------------

  it('rejects a timing sidecar in the match namespace', () =>
    expectReject('kind', 'NOT_A_MATCH_SIDECAR', (c) => {
      c[0].kind = 'timing';
    }));

  it('rejects a run missing a game, measured against the committed P', () =>
    // Bound to run.json's P, so a partial run cannot pass as a smaller complete
    // one — an interim peek must not be able to wear a final verdict.
    expectReject('halfpair', 'SIDECAR_COUNT', (c) => c.slice(0, 2 * P - 1)));

  it('rejects an incomplete pair even when the total count is right', () =>
    expectReject('pairshape', 'PAIR_NOT_COMPLETE', (c) => {
      // Three sidecars land on pair 0 and one on pair 2: 2P files, broken shape.
      c[c.length - 1].pair_index = 0;
      c[c.length - 1].opening_id = 0;
    }));

  it('rejects analysing a P=200 run as though it were P=100', () =>
    expectReject('shortP', 'SIDECAR_COUNT', undefined, { P: P + 1 }));

  it('rejects a run with no committed metadata', () =>
    expectReject('nometa', 'NO_RUN_METADATA', undefined, { runJson: null }));

  it('rejects a fingerprint that is not the committed run', () =>
    expectReject('notthisrun', 'FINGERPRINT_NOT_THE_COMMITTED_RUN', (c) => {
      for (const s of c) s.execution_commit = 'c'.repeat(40);
    }));

  it('rejects a sidecar missing required fields', () =>
    // Equality against an ABSENT field succeeds vacuously, so structure is
    // checked before any semantic comparison.
    expectReject('missingfields', 'MISSING_OR_MALFORMED_FIELD', (c) => {
      for (const s of c) {
        delete s.execution_commit;
        delete s.ort_version;
        delete s.elapsed_ms;
      }
    }));

  it('rejects a sidecar carrying unexpected fields', () =>
    expectReject('extrafields', 'UNEXPECTED_FIELDS', (c) => {
      c[0].injected_note = 'not part of the schema';
    }));

  it('rejects an empty match directory instead of throwing', async () => {
    const d = await runFrom('empty', () => []);
    const r = await analyse(d, OPENINGS, TEST_SPEC);
    assert.strictEqual(r.verdict, 'REJECTED');
    assert.ok(r.failures.map((f) => f.code).includes('EMPTY_MATCH_DIRECTORY'));
  });

  it('rejects both games of a pair sharing a slot', () =>
    expectReject('slots', 'BAD_GAME_IN_PAIR', (c) => {
      c[1].game_in_pair = 0;
    }));

  it('rejects opening_id that is not pair_index', () =>
    expectReject('openingid', 'OPENING_ID_NOT_PAIR_INDEX', (c) => {
      c[0].opening_id = 2;
    }));

  it('rejects an opening set that is not the frozen prefix', () =>
    // The cherry-pick this rule exists to stop: right count, wrong set.
    expectReject('prefix', 'OPENING_SET_NOT_PREFIX', (c) =>
      c.map((s) =>
        s.pair_index === 0 ? { ...s, pair_index: 9, opening_id: 9 } : s
      )
    ));

  it('rejects fingerprint drift across sidecars', () =>
    expectReject('fingerprint', 'FINGERPRINT_DRIFT', (c) => {
      c[3].execution_commit = 'b'.repeat(40);
    }));

  it('rejects a run whose simulation count is not the specified one', async () => {
    const d = await runFrom('sims');
    const r = await analyse(d, OPENINGS, FROZEN_SPEC); // demands 800
    assert.strictEqual(r.verdict, 'REJECTED');
    assert.ok(r.failures.map((f) => f.code).includes('WRONG_SIMULATIONS'));
  });

  // --- colour rejections ----------------------------------------------------

  it('rejects a colour set that is not the two roles', () =>
    expectReject('colourset', 'COLOUR_SET_MISMATCH', (c) => {
      c[0].red_model_id = c[0].black_model_id;
    }));

  it('rejects a colour assignment that contradicts game_in_pair', () =>
    expectReject('colourassign', 'COLOUR_ASSIGNMENT', (c) => {
      const s = c.find((x) => x.game_in_pair === 0);
      [s.red_model_id, s.black_model_id] = [s.black_model_id, s.red_model_id];
    }));

  // --- re-derivation rejections ---------------------------------------------

  it('rejects a tampered result even when internally consistent', () =>
    // The sidecar is rewritten so result and candidate_score agree with each
    // other; only replaying the moves exposes it.
    expectReject('result', 'RESULT_MISMATCH', (c) => {
      const s = c[0];
      s.result = s.result === 'red' ? 'black' : 'red';
      const candidateColour =
        s.red_model_id === s.candidate_model_id ? 'red' : 'black';
      s.candidate_score = s.result === candidateColour ? 1 : 0;
    }));

  it('rejects a tampered candidate_score', () =>
    expectReject('score', 'CANDIDATE_SCORE_MISMATCH', (c) => {
      c[0].candidate_score = c[0].candidate_score === 1 ? 0 : 1;
    }));

  it('rejects a tampered ply count', () =>
    expectReject('ply', 'PLY_COUNT_MISMATCH', (c) => {
      c[0].ply_count += 1;
    }));

  it('rejects a tampered termination reason', () =>
    expectReject('termination', 'TERMINATION_MISMATCH', (c) => {
      c[0].termination = 'max_plies';
    }));

  it('rejects an illegal move in the sequence', () =>
    expectReject('illegal', 'REPLAY_ILLEGAL', (c) => {
      c[0].moves[6] = c[0].moves[5];
    }));

  it('rejects a truncated game that never reached a terminal state', () =>
    expectReject('truncated', 'REPLAY_NOT_TERMINAL', (c) => {
      const s = c[0];
      s.moves = s.moves.slice(0, 8);
      s.ply_count = 8;
    }));

  it('rejects moves that do not start with the declared opening', () =>
    // Swapping the first two moves keeps the count, the hash and legality
    // intact, so only the prefix check can catch it.
    expectReject('prefixmoves', 'OPENING_PREFIX_MISMATCH', (c) => {
      const m = c[0].moves;
      c[0].moves = [m[1], m[0], ...m.slice(2)];
    }));

  it('rejects an opening hash that does not match the pool', () =>
    expectReject('hash', 'OPENING_HASH_MISMATCH', (c) => {
      c[0].opening_sha256 = 'f'.repeat(64);
    }));
});
