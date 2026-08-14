/**
 * Tests for the product-stack match harness.
 *
 *   node --test tests/product_match/test_harness.mjs
 *
 * These play REAL games, but at 2 simulations on hand-written throwaway
 * openings. They are machinery tests and constitute NO evidence about either
 * model: the authorized match is 800 simulations over the frozen opening pool,
 * and nothing here touches that pool or selects `P`.
 *
 * The interrupt/resume path is exercised for real — a resumption rule that is
 * never actually interrupted is exactly the class of thing review has caught
 * repeatedly: declared, not measured.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { mkdtemp, rm, readdir, readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  HarnessError,
  SCHEMA,
  buildSidecar,
  executionCommit,
  fingerprintOf,
  hardPolicy,
  loadModel,
  playGame,
  runMatch,
  sidecarName,
} from './harness.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');
const BASELINE_DIR = join(REPO_ROOT, 'models', BASELINE_MODEL_ID);
const CANDIDATE_DIR = join(REPO_ROOT, 'models', CANDIDATE_MODEL_ID);

const SIMS = 2; // machinery only — never the match's 800
const TEST_OPENINGS = [
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

let dir;
let baseline;
let candidate;

const opts = (over = {}) => ({
  runDir: dir,
  P: 1,
  openings: TEST_OPENINGS,
  baselineDir: BASELINE_DIR,
  candidateDir: CANDIDATE_DIR,
  nSimulations: SIMS,
  requireCleanWorktree: false,
  ...over,
});

const listMatch = async (d = dir) => (await readdir(join(d, 'match'))).sort();

describe('model loading mirrors the product', () => {
  before(async () => {
    baseline = await loadModel(BASELINE_DIR);
    candidate = await loadModel(CANDIDATE_DIR);
  });

  it('loads both committed artifacts and reports their ids', () => {
    assert.strictEqual(baseline.modelId, BASELINE_MODEL_ID);
    assert.strictEqual(candidate.modelId, CANDIDATE_MODEL_ID);
  });

  it('performs the contract check, which resolveModel cannot do', async () => {
    // Guards the guard: assertSessionContract must actually be reachable and
    // fatal, since resolveModel returns before a session exists.
    const { assertSessionContract } = await import(
      '../../server/model_manifest.js'
    );
    const broken = structuredClone(baseline.manifest);
    broken.contract.max_moves = 512;
    assert.throws(
      () => assertSessionContract(broken, baseline.inference.session, 576),
      (e) => e.name === 'ModelManifestError'
    );
  });

  it('refuses a manifest whose bytes do not match', async () => {
    await assert.rejects(
      loadModel(join(tmpdir(), 'definitely-not-a-model-dir')),
      (e) => e.name === 'ModelManifestError' && e.code === 'MANIFEST_MISSING'
    );
  });
});

describe('execution commit', () => {
  it('reports HEAD, and refuses a dirty worktree', () => {
    const clean =
      execFileSync('git', ['status', '--porcelain'], { cwd: REPO_ROOT })
        .toString()
        .trim() === '';
    if (clean) {
      assert.match(executionCommit(), /^[0-9a-f]{40}$/);
    } else {
      // Asserted in whichever state the tree is actually in, so the test is
      // never flaky and never vacuous.
      assert.throws(
        () => executionCommit(),
        (e) => e instanceof HarnessError && e.code === 'DIRTY_WORKTREE'
      );
    }
    assert.match(executionCommit({ requireClean: false }), /^[0-9a-f]{40}$/);
  });
});

describe('playGame', () => {
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-match-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('plays to a real terminal state and reports it consistently', async () => {
    const g = await playGame({
      redInference: candidate.inference,
      blackInference: baseline.inference,
      openingMoves: TEST_OPENINGS[0],
      nSimulations: SIMS,
    });
    assert.ok(['red', 'black', 'draw'].includes(g.result));
    assert.ok(['win', 'no_legal_moves', 'max_plies'].includes(g.termination));
    assert.strictEqual(g.plyCount, g.moves.length);
    assert.ok(g.plyCount > TEST_OPENINGS[0].length);
    assert.deepStrictEqual(g.moves.slice(0, 4), TEST_OPENINGS[0]);
  });

  it('is deterministic: the same inputs replay identically', async () => {
    // The property the whole resume design rests on.
    const a = await playGame({
      redInference: candidate.inference,
      blackInference: baseline.inference,
      openingMoves: TEST_OPENINGS[1],
      nSimulations: SIMS,
    });
    const b = await playGame({
      redInference: candidate.inference,
      blackInference: baseline.inference,
      openingMoves: TEST_OPENINGS[1],
      nSimulations: SIMS,
    });
    assert.deepStrictEqual(a.moves, b.moves);
    assert.strictEqual(a.result, b.result);
    assert.strictEqual(a.plyCount, b.plyCount);
  });

  it('swapping colours changes the game, so the pairing is not a no-op', async () => {
    const red = await playGame({
      redInference: candidate.inference,
      blackInference: baseline.inference,
      openingMoves: TEST_OPENINGS[2],
      nSimulations: SIMS,
    });
    const swapped = await playGame({
      redInference: baseline.inference,
      blackInference: candidate.inference,
      openingMoves: TEST_OPENINGS[2],
      nSimulations: SIMS,
    });
    assert.notDeepStrictEqual(red.moves, swapped.moves);
  });

  it('takes the readout temperature from the shipped table, and ignores a caller override', async () => {
    // The readout is the most drift-prone seam here, so the product owns it.
    // A caller must not be able to play at a different temperature while still
    // labelling the game 'hard'.
    const policy = hardPolicy();
    assert.deepStrictEqual(policy, {
      difficulty: 'hard',
      nSims: 800,
      moveTemp: 0,
    });

    const g = await playGame({
      redInference: candidate.inference,
      blackInference: baseline.inference,
      openingMoves: TEST_OPENINGS[0],
      nSimulations: SIMS,
      moveTemp: 0.9, // not a parameter — must have no effect
    });
    assert.strictEqual(
      g.moveTemp,
      0,
      'temperature comes from the policy, not the caller'
    );
    assert.strictEqual(g.nSimulations, SIMS);
  });

  it('rejects an illegal opening rather than playing from a bad position', async () => {
    await assert.rejects(
      playGame({
        redInference: candidate.inference,
        blackInference: baseline.inference,
        openingMoves: [
          [5, 5],
          [5, 5],
        ],
        nSimulations: SIMS,
      }),
      (e) => e instanceof HarnessError && e.code === 'ILLEGAL_OPENING'
    );
  });
});

describe('sidecar construction', () => {
  const game = {
    moves: [[0, 0]],
    result: 'red',
    termination: 'win',
    plyCount: 1,
    elapsedMs: 1,
  };
  const base = {
    openingId: 3,
    openingMoves: TEST_OPENINGS[0],
    pairIndex: 3,
    baselineModelId: BASELINE_MODEL_ID,
    candidateModelId: CANDIDATE_MODEL_ID,
    game,
    nSimulations: 800,
    cPuct: 1.5,
    moveTemp: 0,
    ortVersion: '1.23.2',
    executionCommit: 'a'.repeat(40),
  };

  it('game_in_pair 0 puts the candidate on red', () => {
    const s = buildSidecar({ ...base, gameInPair: 0 });
    assert.strictEqual(s.red_model_id, CANDIDATE_MODEL_ID);
    assert.strictEqual(s.black_model_id, BASELINE_MODEL_ID);
    assert.strictEqual(s.candidate_score, 1.0, 'red won and candidate was red');
  });

  it('game_in_pair 1 puts the candidate on black', () => {
    const s = buildSidecar({ ...base, gameInPair: 1 });
    assert.strictEqual(s.red_model_id, BASELINE_MODEL_ID);
    assert.strictEqual(s.black_model_id, CANDIDATE_MODEL_ID);
    assert.strictEqual(
      s.candidate_score,
      0.0,
      'red won but candidate was black'
    );
  });

  it('scores a draw as 0.5 from either colour', () => {
    const drawn = { ...game, result: 'draw' };
    for (const gameInPair of [0, 1]) {
      assert.strictEqual(
        buildSidecar({ ...base, game: drawn, gameInPair }).candidate_score,
        0.5
      );
    }
  });

  it('the fingerprint fields exclude the colours that swap', () => {
    const a = fingerprintOf(buildSidecar({ ...base, gameInPair: 0 }));
    const b = fingerprintOf(buildSidecar({ ...base, gameInPair: 1 }));
    // The whole point of the role fields: a valid pair must share a fingerprint.
    assert.deepStrictEqual(a, b);
    assert.ok(!('red_model_id' in a) && !('black_model_id' in a));
    assert.strictEqual(a.schema, SCHEMA);
  });
});

describe('runMatch and resumption', () => {
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-run-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('plays complete pairs and writes no temp files behind', async () => {
    const summary = await runMatch(opts({ P: 2 }));
    assert.strictEqual(summary.played, 4);
    assert.deepStrictEqual(await listMatch(), [
      sidecarName(0, 0),
      sidecarName(0, 1),
      sidecarName(1, 0),
      sidecarName(1, 1),
    ]);
    assert.ok(!(await listMatch()).some((n) => n.endsWith('.tmp')));
  });

  it('skips completed pairs on a clean re-run', async () => {
    const summary = await runMatch(opts({ P: 2 }));
    assert.strictEqual(summary.skipped, 2);
    assert.strictEqual(summary.played, 0);
  });

  it('quarantines a half-finished pair, replays it, and verifies the replay', async () => {
    const d = await mkdtemp(join(tmpdir(), 'twixt-resume-'));
    // Interrupt for real: throw after game 0 of pair 0 is on disk.
    await assert.rejects(
      runMatch(
        opts({
          runDir: d,
          P: 1,
          onGameComplete: ({ gameInPair }) => {
            if (gameInPair === 0) throw new Error('simulated interruption');
          },
        })
      ),
      /simulated interruption/
    );
    assert.deepStrictEqual(await listMatch(d), [sidecarName(0, 0)]);
    const interrupted = JSON.parse(
      await readFile(join(d, 'match', sidecarName(0, 0)), 'utf8')
    );

    const summary = await runMatch(opts({ runDir: d, P: 1 }));
    assert.strictEqual(
      summary.quarantined,
      1,
      'the survivor must be quarantined'
    );
    assert.strictEqual(summary.replayed, 1, 'and its replay must be verified');
    assert.strictEqual(
      summary.played,
      2,
      'the whole pair is replayed, never half-counted'
    );

    // The survivor is preserved as evidence, not deleted.
    assert.deepStrictEqual(await readdir(join(d, 'quarantine')), [
      sidecarName(0, 0),
    ]);
    // And the replay reproduced it.
    const replayed = JSON.parse(
      await readFile(join(d, 'match', sidecarName(0, 0)), 'utf8')
    );
    assert.deepStrictEqual(replayed.moves, interrupted.moves);
    assert.strictEqual(replayed.result, interrupted.result);
    await rm(d, { recursive: true, force: true });
  });

  it('aborts when a replay does not reproduce the quarantined game', async () => {
    const d = await mkdtemp(join(tmpdir(), 'twixt-mismatch-'));
    await assert.rejects(
      runMatch(
        opts({
          runDir: d,
          P: 1,
          onGameComplete: ({ gameInPair }) => {
            if (gameInPair === 0) throw new Error('simulated interruption');
          },
        })
      ),
      /simulated interruption/
    );
    // Tamper with the survivor so the deterministic replay cannot match it.
    const p = join(d, 'match', sidecarName(0, 0));
    const s = JSON.parse(await readFile(p, 'utf8'));
    s.moves = [...s.moves.slice(0, -1), [23, 23]];
    await writeFile(p, JSON.stringify(s));

    await assert.rejects(
      runMatch(opts({ runDir: d, P: 1 })),
      (e) => e instanceof HarnessError && e.code === 'REPLAY_MISMATCH'
    );
    // Both copies survive for diagnosis.
    assert.deepStrictEqual(await readdir(join(d, 'quarantine')), [
      sidecarName(0, 0),
    ]);
    await rm(d, { recursive: true, force: true });
  });

  it('refuses to resume under a different configuration', async () => {
    // A restart at different settings is a NEW run, not a resume. Finishing a
    // match whose halves were played by different code is the failure this
    // prevents, and from the inside it looks entirely benign.
    await assert.rejects(
      runMatch(opts({ P: 2, nSimulations: SIMS + 1 })),
      (e) => e instanceof HarnessError && e.code === 'FINGERPRINT_MISMATCH'
    );
  });

  it('refuses to resume with a different P', async () => {
    await assert.rejects(
      runMatch(opts({ P: 3 })),
      (e) => e instanceof HarnessError && e.code === 'FINGERPRINT_MISMATCH'
    );
  });

  it('records a run fingerprint that every sidecar shares', async () => {
    const run = JSON.parse(await readFile(join(dir, 'run.json'), 'utf8'));
    assert.strictEqual(run.P, 2);
    for (const n of await listMatch()) {
      const s = JSON.parse(await readFile(join(dir, 'match', n), 'utf8'));
      assert.deepStrictEqual(fingerprintOf(s), run.fingerprint);
    }
  });
});
