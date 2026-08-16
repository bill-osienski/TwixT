/**
 * Tests for the timing smoke, the P-decision artifact, and the binding that
 * makes a match refuse to start without one.
 *
 *   node --test tests/product_match/test_timing.mjs
 *
 * NO timing game is played here. `runTimingSmoke` takes injected clock and game
 * seams, so its schedule, sidecars, wall-clock span and decision derivation are
 * exercised without invoking `playGame`; the production CLIs are never called
 * and the reserved openings `200…209` never reach a real game. Fixtures live in
 * temporary directories and are deleted, so no timing evidence is retained.
 */
import { describe, it, after, before } from 'node:test';
import assert from 'node:assert';
import {
  copyFile,
  mkdtemp,
  mkdir,
  rm,
  readdir,
  readFile,
  writeFile,
} from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  EXECUTION_SURFACE_FILES,
  POOL_RELPATH,
  P_DECISION_RELPATH,
  PDecisionError,
  P_IF_AT_OR_ABOVE,
  P_IF_BELOW,
  SCHEMA,
  THRESHOLD_GAMES_PER_HOUR,
  TIMING_GAMES,
  TIMING_OPENING_MAPPING,
  buildDecision,
  computeThroughput,
  decisionFailures,
  deriveP,
  executionSurfaceDigest,
  expectedTimingFilenames,
  loadCommittedDecision,
  readCommittedBlob,
  sha256,
} from './p_decision.mjs';
import { runTimingSmoke, timingSchedule } from './timing.mjs';
import { analyse, analyseEvidence, FROZEN_SPEC } from './analyse.mjs';
import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  runMatchFromCommittedDecision,
} from './harness.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');

/** A wall time that yields exactly `gamesPerHour` for ten games. */
const wallMsFor = (gamesPerHour) => (TIMING_GAMES * 3_600_000) / gamesPerHour;

const validDecision = (gamesPerHour = 13.2, over = {}) =>
  buildDecision({
    totalSequentialWallMs: wallMsFor(gamesPerHour),
    // Schedule-derived filenames with DISTINCT digests: ten records of one game
    // would otherwise satisfy a filename-only check.
    timingEvidence: expectedTimingFilenames().map((file, i) => ({
      file,
      sha256: String(i).padStart(64, 'a'),
    })),
    openingPoolSha256: 'b'.repeat(64),
    executionCommit: 'c'.repeat(40),
    executionSurfaceSha256: 'd'.repeat(64),
    baselineModelId: BASELINE_MODEL_ID,
    candidateModelId: CANDIDATE_MODEL_ID,
    ortVersion: '1.23.2',
    ortConfig: 'no options supplied',
    nSimulations: 800,
    cPuct: 1.5,
    moveTemp: 0,
    ...over,
  });

describe('P is derived mechanically, never chosen', () => {
  it('applies the frozen rule at, above and below the threshold', () => {
    assert.strictEqual(THRESHOLD_GAMES_PER_HOUR, 8.8);
    assert.strictEqual(
      deriveP(8.8),
      P_IF_AT_OR_ABOVE,
      'the boundary is inclusive'
    );
    assert.strictEqual(deriveP(8.800001), P_IF_AT_OR_ABOVE);
    assert.strictEqual(deriveP(8.799999), P_IF_BELOW);
  });

  it('permits no third outcome', () => {
    for (const g of [0.01, 5, 8.79, 8.8, 100, 1e6]) {
      assert.ok([P_IF_AT_OR_ABOVE, P_IF_BELOW].includes(deriveP(g)));
    }
  });

  it('refuses a throughput that is not a number', () => {
    for (const bad of [NaN, Infinity]) {
      assert.throws(
        () => deriveP(bad),
        (e) => e.code === 'BAD_THROUGHPUT'
      );
    }
  });
});

describe('throughput is one wall-clock span', () => {
  it('uses the exact formula from the specification', () => {
    assert.strictEqual(computeThroughput(3_600_000, 10), 10);
    assert.ok(Math.abs(computeThroughput(wallMsFor(8.8)) - 8.8) < 1e-9);
  });

  it('refuses a non-positive span rather than dividing by it', () => {
    for (const bad of [0, -1, NaN]) {
      assert.throws(
        () => computeThroughput(bad),
        (e) => e.code === 'BAD_WALL_TIME'
      );
    }
  });
});

describe('the timing schedule is frozen and outcome-blind', () => {
  it('is ten games, five per self-play arm, in the specified order', () => {
    const s = timingSchedule();
    assert.strictEqual(s.length, TIMING_GAMES);
    assert.deepStrictEqual(
      s.filter((g) => g.arm === 'baseline_self_play').map((g) => g.openingId),
      [200, 201, 202, 203, 204]
    );
    assert.deepStrictEqual(
      s.filter((g) => g.arm === 'candidate_self_play').map((g) => g.openingId),
      [205, 206, 207, 208, 209]
    );
  });

  it('touches only the reserved openings, never a match opening', () => {
    for (const g of timingSchedule()) {
      assert.ok(g.openingId >= 200 && g.openingId <= 209);
    }
  });

  it('the frozen mapping cannot be mutated in place', () => {
    assert.throws(() => {
      TIMING_OPENING_MAPPING.baseline_self_play.push(1);
    });
  });
});

describe('runTimingSmoke, with the clock and games injected', () => {
  let dir;
  let result;
  const SPAN_MS = wallMsFor(13.2);

  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-timing-'));
    // A synthetic 210-entry pool, so no real opening is used and no game runs.
    const openings = Array.from({ length: 210 }, (_, i) => [
      [0, i % 24],
      [1, i % 24],
      [2, i % 24],
      [3, i % 24],
    ]);
    const clock = [1000, 1000 + SPAN_MS];
    let calls = 0;
    result = await runTimingSmoke({
      runDir: dir,
      openings,
      baseline: { modelId: BASELINE_MODEL_ID, inference: 'baseline-stub' },
      candidate: { modelId: CANDIDATE_MODEL_ID, inference: 'candidate-stub' },
      nSimulations: 800,
      moveTemp: 0,
      ortVersion: '1.23.2',
      executionCommit: 'd'.repeat(40),
      poolSha256: 'b'.repeat(64),
      executionSurfaceSha256: 'e'.repeat(64),
      now: () => clock[Math.min(calls++, clock.length - 1)],
      playGameFn: async ({ redInference, blackInference, onFirstSearch }) => {
        // Self-play: both sides must be the SAME model instance.
        assert.strictEqual(redInference, blackInference);
        // The clock starts at playGame's pre-search seam, so a stub must fire
        // it exactly as the real one does.
        if (onFirstSearch) onFirstSearch();
        return {
          moves: [
            [0, 0],
            [1, 1],
          ],
          result: 'draw',
          termination: 'no_legal_moves',
          plyCount: 2,
          elapsedMs: 1,
          nSimulations: 800,
          moveTemp: 0,
        };
      },
    });
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('writes exactly ten timing sidecars, in its own namespace', async () => {
    const files = (await readdir(join(dir, 'timing'))).sort();
    assert.deepStrictEqual(files, [...expectedTimingFilenames()].sort());
    await assert.rejects(
      readdir(join(dir, 'match')),
      'never in the analysed namespace'
    );
  });

  it('marks every sidecar as timing, self-play, on a reserved opening', async () => {
    for (const f of await readdir(join(dir, 'timing'))) {
      const s = JSON.parse(await readFile(join(dir, 'timing', f), 'utf8'));
      assert.strictEqual(s.kind, 'timing');
      assert.strictEqual(s.red_model_id, s.black_model_id, 'must be self-play');
      assert.ok(s.opening_id >= 200 && s.opening_id <= 209);
    }
  });

  it('opens the span at the pre-search seam and closes it at the last rename', () => {
    // The clock is read exactly twice: when game 1 reaches its first search,
    // and after game 10's atomic rename. Anything wider would time setup or
    // bookkeeping the match does not repeat per game.
    assert.strictEqual(result.totalSequentialWallMs, SPAN_MS);
    // Each fake game reported 1 ms, so a sum would be 10 — proof the span is
    // not assembled from per-game elapsed times.
    assert.notStrictEqual(result.totalSequentialWallMs, TIMING_GAMES);
  });

  it('derives the decision from that span alone', () => {
    const d = result.decision;
    assert.strictEqual(d.schema, SCHEMA);
    assert.strictEqual(d.measured.total_sequential_wall_ms, SPAN_MS);
    assert.ok(Math.abs(d.measured.games_per_hour - 13.2) < 1e-9);
    assert.strictEqual(d.selected_p, 200);
    assert.deepStrictEqual(d.opening_mapping, TIMING_OPENING_MAPPING);
  });

  it('records the pool and surface digests it was given, not ones read later', () => {
    // Both are frozen before the first game, so a repository edited during a
    // multi-hour run cannot change what the decision claims.
    assert.strictEqual(result.decision.opening_pool_sha256, 'b'.repeat(64));
    assert.strictEqual(
      result.decision.execution_surface_sha256,
      'e'.repeat(64)
    );
  });

  it('hashes the evidence as written to disk', async () => {
    for (const e of result.decision.timing_evidence) {
      assert.strictEqual(
        sha256(await readFile(join(dir, 'timing', e.file))),
        e.sha256
      );
    }
  });

  it('produces a decision that validates', () => {
    assert.deepStrictEqual(decisionFailures(result.decision), []);
  });
});

describe('the decision validator re-derives rather than trusting', () => {
  const expectFail = (code, mutate) => {
    const d = validDecision();
    mutate(d);
    const codes = decisionFailures(d).map((f) => f.code);
    assert.ok(
      codes.includes(code),
      `expected ${code}, got: ${[...new Set(codes)].join(', ')}`
    );
  };

  it('accepts a well-formed decision (control)', () => {
    assert.deepStrictEqual(decisionFailures(validDecision()), []);
  });

  it('rejects a P that its own throughput does not imply', () =>
    expectFail('P_NOT_DERIVED_FROM_THROUGHPUT', (d) => {
      d.selected_p = d.selected_p === 200 ? 100 : 200;
    }));

  it('rejects a throughput that its own wall time does not imply', () =>
    expectFail('THROUGHPUT_NOT_DERIVED_FROM_WALL_TIME', (d) => {
      d.measured.games_per_hour += 1;
    }));

  it('rejects a restated threshold', () =>
    expectFail('WRONG_THRESHOLD', (d) => {
      d.threshold_games_per_hour = 1;
    }));

  it('rejects a P outside the two permitted values', () =>
    expectFail('P_NOT_A_PERMITTED_VALUE', (d) => {
      d.selected_p = 150;
    }));

  it('rejects an altered opening mapping', () =>
    expectFail('OPENING_MAPPING_NOT_FROZEN', (d) => {
      d.opening_mapping = {
        baseline_self_play: [0, 1, 2, 3, 4],
        candidate_self_play: [5, 6, 7, 8, 9],
      };
    }));

  it('rejects the wrong number of timing games', () =>
    expectFail('WRONG_TIMING_GAME_COUNT', (d) => {
      d.measured.timing_games = 5;
    }));

  it('requires exactly the schedule-derived filenames, not ten arbitrary ones', () => {
    expectFail('TIMING_EVIDENCE_FILENAMES', (d) => {
      d.timing_evidence[0].file = 'timing_99_opening_999.json';
    });
    expectFail('TIMING_EVIDENCE_FILENAMES', (d) => {
      d.timing_evidence = d.timing_evidence.slice(0, 3);
    });
  });

  it('rejects two different files carrying the SAME digest', () =>
    // Ten records of one game: distinct filenames pass a name-only check.
    expectFail('DUPLICATE_TIMING_DIGEST', (d) => {
      d.timing_evidence[1].sha256 = d.timing_evidence[0].sha256;
    }));

  it('rejects malformed digests', () =>
    expectFail('MALFORMED_TIMING_EVIDENCE', (d) => {
      d.timing_evidence[0].sha256 = 'not-a-digest';
    }));

  it('rejects missing required fields before comparing anything', () => {
    expectFail('MISSING_OR_MALFORMED_FIELD', (d) => {
      delete d.execution_commit;
    });
    expectFail('MISSING_OR_MALFORMED_FIELD', (d) => {
      delete d.execution_surface_sha256;
    });
  });

  it('rejects a decision bound to a different pool, surface or settings', () => {
    const d = validDecision();
    const codes = (opts) => decisionFailures(d, opts).map((f) => f.code);
    assert.ok(
      codes({ poolSha256: 'f'.repeat(64) }).includes('POOL_HASH_MISMATCH')
    );
    assert.ok(
      codes({ surfaceSha256: 'f'.repeat(64) }).includes(
        'EXECUTION_SURFACE_MISMATCH'
      )
    );
    assert.ok(
      codes({ expected: { n_simulations: 400 } }).includes('BINDING_MISMATCH')
    );
  });

  it('rejects a non-object', () => {
    for (const bad of [null, 'x', [1]]) {
      assert.strictEqual(
        decisionFailures(bad)[0].code,
        'DECISION_NOT_AN_OBJECT'
      );
    }
  });
});

describe('committed means committed, not merely tracked', () => {
  // These build a REAL temporary git repository, because what git says is the
  // whole point. `git ls-files` was not enough: it succeeds for a newly staged
  // file and for a tracked file whose working-tree bytes were edited
  // afterwards, and a working-tree read would then read mutable content.
  let root;
  const git = (dir, ...args) =>
    execFileSync('git', args, { cwd: dir, stdio: 'ignore' });

  async function makeRepo({
    commitDecision = true,
    stageOnly = false,
    mutate = null,
  } = {}) {
    const dir = await mkdtemp(join(root, 'repo-'));
    git(dir, 'init', '-q');
    git(dir, 'config', 'user.email', 't@example.com');
    git(dir, 'config', 'user.name', 'test');

    for (const rel of [...EXECUTION_SURFACE_FILES, POOL_RELPATH]) {
      await mkdir(join(dir, dirname(rel)), { recursive: true });
      await copyFile(join(REPO_ROOT, rel), join(dir, rel));
    }
    git(dir, 'add', '-A');
    git(dir, 'commit', '-qm', 'surface');

    const decision = validDecision(13.2, {
      openingPoolSha256: sha256(readCommittedBlob(POOL_RELPATH, 'HEAD', dir)),
      executionSurfaceSha256: executionSurfaceDigest('HEAD', dir),
    });
    if (mutate) mutate(decision);

    const path = join(dir, P_DECISION_RELPATH);
    await writeFile(path, `${JSON.stringify(decision, null, 2)}\n`);
    if (stageOnly) {
      git(dir, 'add', P_DECISION_RELPATH);
    } else if (commitDecision) {
      git(dir, 'add', P_DECISION_RELPATH);
      git(dir, 'commit', '-qm', 'decision');
    }
    return { dir, path };
  }

  before(async () => {
    root = await mkdtemp(join(tmpdir(), 'twixt-repo-'));
  });
  after(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it('accepts a committed, fully bound decision', async () => {
    const { dir } = await makeRepo();
    assert.strictEqual(
      (await loadCommittedDecision({ repoRoot: dir })).selected_p,
      200
    );
  });

  it('REFUSES a decision that is staged but never committed', async () => {
    const { dir } = await makeRepo({ commitDecision: false, stageOnly: true });
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e instanceof PDecisionError && e.code === 'NOT_COMMITTED'
    );
  });

  it('REFUSES a decision that exists only in the working tree', async () => {
    const { dir } = await makeRepo({ commitDecision: false });
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'NOT_COMMITTED'
    );
  });

  it('reads the COMMITTED bytes when the working tree was edited afterwards', async () => {
    // The subtle case: the file is tracked, so a tracking check passes, and a
    // working-tree read would return the edited P.
    const { dir, path } = await makeRepo();
    const tampered = JSON.parse(await readFile(path, 'utf8'));
    tampered.selected_p = 100;
    await writeFile(path, JSON.stringify(tampered, null, 2));

    assert.strictEqual(
      (await loadCommittedDecision({ repoRoot: dir })).selected_p,
      200,
      'the committed decision must win'
    );
    assert.strictEqual(
      JSON.parse(await readFile(path, 'utf8')).selected_p,
      100,
      'and the tamper was real, or this proves nothing'
    );
  });

  it('refuses committed content that is unparseable', async () => {
    const { dir, path } = await makeRepo({ commitDecision: false });
    await writeFile(path, '{ not json');
    git(dir, 'add', P_DECISION_RELPATH);
    git(dir, 'commit', '-qm', 'bad');
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'DECISION_UNPARSEABLE'
    );
  });

  it('refuses a committed decision whose P its own throughput does not imply', async () => {
    const { dir } = await makeRepo({ mutate: (d) => (d.selected_p = 100) });
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });

  it('refuses a decision bound to a different execution surface', async () => {
    const { dir } = await makeRepo({
      mutate: (d) => (d.execution_surface_sha256 = 'e'.repeat(64)),
    });
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });

  it('notices when an execution-surface file changes AFTER the decision', async () => {
    // Ancestry would accept this: the new commit is a descendant. Rewriting
    // search after timing means the measurement no longer describes the code,
    // and the surface digest catches it where ancestry cannot.
    const { dir } = await makeRepo();
    const target = join(dir, 'server/mcts.js');
    await writeFile(target, `${await readFile(target, 'utf8')}\n// drift\n`);
    git(dir, 'add', '-A');
    git(dir, 'commit', '-qm', 'rewrote search after timing');
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });

  it('refuses a decision bound to a different pool', async () => {
    const { dir } = await makeRepo({
      mutate: (d) => (d.opening_pool_sha256 = 'b'.repeat(64)),
    });
    await assert.rejects(
      loadCommittedDecision({ repoRoot: dir }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });

  it('refuses a decision bound to different search settings', async () => {
    const { dir } = await makeRepo();
    await assert.rejects(
      loadCommittedDecision({
        repoRoot: dir,
        expected: { n_simulations: 400 },
      }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });
});

describe('nothing may run a match without the committed decision', () => {
  let dir;
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-bind-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('no decision is committed in THIS repository, and that is the intended state', () => {
    assert.throws(
      () => readCommittedBlob(P_DECISION_RELPATH, 'HEAD', REPO_ROOT),
      (e) => e.code === 'NOT_COMMITTED'
    );
  });

  it('the production match entry point refuses, before loading a model', async () => {
    await assert.rejects(
      runMatchFromCommittedDecision({
        runDir: join(dir, 'run'),
        openings: [],
        requireCleanWorktree: false,
      }),
      (e) => e instanceof PDecisionError && e.code === 'NOT_COMMITTED'
    );
    await assert.rejects(
      readdir(join(dir, 'run')),
      'no run directory may be created'
    );
  });

  it('the production analyser refuses when no decision is committed', async () => {
    const runDir = join(dir, 'analysed');
    await mkdir(join(runDir, 'match'), { recursive: true });
    await writeFile(
      join(runDir, 'run.json'),
      JSON.stringify({ P: 100, fingerprint: { execution_commit: 'HEAD' } })
    );
    const r = await analyse(runDir, [], FROZEN_SPEC);
    assert.strictEqual(r.verdict, 'REJECTED');
    assert.strictEqual(r.failures[0].code, 'P_DECISION_UNAVAILABLE');
  });

  it('the production analyser refuses a run with no commit in its fingerprint', async () => {
    const runDir = join(dir, 'nocommit');
    await mkdir(join(runDir, 'match'), { recursive: true });
    await writeFile(
      join(runDir, 'run.json'),
      JSON.stringify({ P: 100, fingerprint: {} })
    );
    const r = await analyse(runDir, [], FROZEN_SPEC);
    assert.strictEqual(r.failures[0].code, 'NO_RUN_COMMIT');
  });

  it('the evidence pipeline rejects a P that disagrees with the decision', async () => {
    const runDir = join(dir, 'mismatch');
    await mkdir(join(runDir, 'match'), { recursive: true });
    // A structurally VALID run.json, so the rejection can only come from the
    // P comparison rather than from metadata validation firing first.
    await writeFile(
      join(runDir, 'run.json'),
      JSON.stringify({
        P: 100,
        fingerprint: {
          execution_commit: 'e'.repeat(40),
          schema: 'twixt-product-match/1',
          ort_version: '1.23.2',
          ort_config: 'no options supplied',
          n_simulations: 800,
          c_puct: 1.5,
          move_temp: 0,
          baseline_model_id: BASELINE_MODEL_ID,
          candidate_model_id: CANDIDATE_MODEL_ID,
        },
      })
    );
    const r = await analyseEvidence(runDir, [], FROZEN_SPEC, 200);
    assert.strictEqual(r.verdict, 'REJECTED');
    assert.ok(
      r.failures
        .map((f) => f.code)
        .includes('P_DOES_NOT_MATCH_COMMITTED_DECISION')
    );
  });

  it('the production analyser has no opt-out and no run.json fallback', async () => {
    const src = await readFile(join(HERE, 'analyse.mjs'), 'utf8');
    const production = src.slice(src.indexOf('export async function analyse('));
    assert.ok(
      production.includes('loadCommittedDecision'),
      'must load the decision'
    );
    assert.ok(!production.includes('requirePDecision'), 'no opt-out may exist');
    assert.ok(
      !production.includes('requireTracked'),
      'commitment may not be waived'
    );
  });

  it('the unchecked core is named so its nature is visible at every call site', async () => {
    const src = await readFile(join(HERE, 'harness.mjs'), 'utf8');
    assert.ok(src.includes('export async function runMatchWithExplicitP('));
    assert.ok(
      !/export async function runMatch\(/.test(src),
      'no innocuously named bypass'
    );
    assert.ok(
      src.includes('runMatchFromCommittedDecision({ runDir })'),
      'the operational entry point must be the gated one'
    );
  });
});

describe('scope: this test file runs no timing game', () => {
  it('imports no way to run a real timing game', async () => {
    // Checked against the IMPORT LIST, not a literal: an assertion that greps
    // for a string can match its own source and pass vacuously.
    const src = await readFile(join(HERE, 'test_timing.mjs'), 'utf8');
    const named = [...src.matchAll(/^import\s*\{([^}]*)\}/gms)]
      .flatMap((m) => m[1].split(','))
      .map((x) => x.trim().split(/\s+as\s+/)[0]);
    assert.ok(
      !named.includes('playGame'),
      'playGame is only ever injected as a stub'
    );
    assert.ok(
      !named.includes('loadModel'),
      'no model is loaded by these tests'
    );
    assert.ok(!named.includes('runMatchWithExplicitP'), 'no match is run here');
  });

  it('no decision artifact was created at the committed path', async () => {
    await assert.rejects(
      readFile(join(REPO_ROOT, P_DECISION_RELPATH)),
      (e) => e.code === 'ENOENT'
    );
  });
});
